from ml_dtypes import bfloat16
import numpy as np
import argparse
import sys

# Upload all necessary packages
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import * 
from aie.dialects.aiex import *
import aie.utils.trace as trace_utils
from aie.utils.trace import PortEvent
from aie.helpers.dialects.ext.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern



def main():
    # Parse arguments
    argparser = argparse.ArgumentParser(
        prog="AIE Linear Layer Implementation of a GRU Cell",
        description="Computes output of a linear layer with input x, weight W, and bias b",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    
    # Default dimensions of the matrices, weights, and biases: change later
    argparser.add_argument("-T", type=int, default=256)
    argparser.add_argument("-D", type=int, default=256)
    argparser.add_argument("-H", type=int, default=256)
    argparser.add_argument("-t", type=int, default=64)
    argparser.add_argument("-d", type=int, default=64)
    argparser.add_argument("-hval", type=int, default=32)

    # For this code, assume both --dtype_in and --dtype_out are bf16
    argparser.add_argument("--trace_size", type=int, default=0)
    args = argparser.parse_args()

    # Call to the linear function
    linear(
        args.dev,
        args.T,
        args.D,
        args.H,
        args.t, 
        args.d,
        args.hval,
        args.trace_size,
    )

def ceildiv(a, b):
    return (a + b - 1) // b

def linear(
    dev, T, D, H, t, d, h, trace_size, dtype_in_str="bf16", dtype_out_str="bf16" 
):
    assert T % t == 0
    assert D % d == 0
    assert H % h == 0

    # r, p, q are the dimensions required by the microkernel MAC instructions.
    r, p, q = 4, 8, 8 # In this case, use r, p, q values of npu2 bf16, emulate_bf16_mmul_with_bfp16 = False 
    
    assert t % r == 0
    assert d % p == 0
    assert h % q == 0
    
    vectorized = True
    enable_tracing = True if trace_size > 0 else False

    dtype_in = bfloat16
    dtype_out = bfloat16

    X_sz = T * D
    W_sz = D * H
    b_sz = H
    Y_sz = T * H

    T_div_t = T // t 
    D_div_d = D // d 
    H_div_h = H // h 
    tiles = T_div_t * H_div_h

    with mlir_mod_ctx() as ctx:


        if dev == "npu":
            dev_ty = AIEDevice.npu1_1col
        else:
            dev_ty = AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            # x_ty is declared as (T, D) instead of (B, T, D) since the value of B is always 1
            x_ty = np.ndarray[(t, d), np.dtype[dtype_in]]
            w_ty = np.ndarray[(d, h), np.dtype[dtype_in]]
            b_ty = np.ndarray[(h,), np.dtype[dtype_in]]
            y_ty = np.ndarray[(t, h), np.dtype[dtype_out]]

            # AIE Core Function declarations - only matmul is used in this case
            func_type = "" if vectorized else "scalar_"
            zero = external_func(f"zero_{func_type}bf16", inputs = [y_ty])
            matmul_func_name = f"matmul_{func_type}bf16_bf16"
            matmul = external_func(
                matmul_func_name,
                inputs=[x_ty, w_ty, y_ty],
            )
            row_wise_add_func = Kernel(
                f"row_wise_bias_add_bf16_bf16", "kernel.o", [y_ty, b_ty, y_ty]
            )
            row_wise_add_func.resolve()  # Add this line

            # Tile declarations - can change later
            shim_tile1 = tile(0, 0)
            shim_tile2 = tile(1, 0)
            mem_tile = tile(0, 1)
            compute_tile1_col, compute_tile1_row = 0, 2
            compute_tile2_col, compute_tile2_row = 0, 3
            compute_tile1 = tile(compute_tile1_col, compute_tile1_row)
            compute_tile2 = tile(compute_tile2_col, compute_tile2_row)

            # AIE-array data movement with object fifos
            # Input X
            inX = object_fifo("inX", shim_tile1, mem_tile, 2, x_ty)
            memX = object_fifo(
                "memX", 
                mem_tile, 
                compute_tile1, 
                2, 
                x_ty,
                (
                    [
                        (t // r, r * d),
                        (d // p, p),
                        (r, d),
                        (p, 1),
                    ]
                    if vectorized
                    else []
                ),
            )
            object_fifo_link(inX, memX)

            # Input W
            inW = object_fifo("inW", shim_tile1, mem_tile, 2, w_ty)
            memW = object_fifo(
                "memW", 
                mem_tile, 
                compute_tile1, 
                2, 
                w_ty,
                (
                    [
                        # This transformation assumes the matrix is row major
                        (d // p, p * h),
                        (h // q, q),
                        (p, h), 
                        (q, 1),
                    ]
                ),
            )
            object_fifo_link(inW, memW)

            # Input b
            inB = object_fifo("inB", shim_tile2, mem_tile, 2, b_ty)
            memB = object_fifo("memB", mem_tile, compute_tile2, 2, b_ty)
            object_fifo_link(inB, memB)

            # Output Y
            memY = object_fifo("memY", compute_tile2, mem_tile, 2, y_ty)
            outY = object_fifo(
                "outY",
                mem_tile,
                shim_tile2,
                2,
                y_ty,
                (
                    [
                        (t // r, r * h),
                        (r, q),
                        (h // q, r * q),
                        (q, 1),
                    ]
                    if vectorized
                    else []
                ),
            )
            object_fifo_link(memY, outY)

            # Intermediate result FIFO (connects compute_tile1 and compute_tile2)
            mm_result = object_fifo("mm_result", compute_tile1, mem_tile, 2, y_ty)
            mm_result_mem = object_fifo("mm_result_mem", mem_tile, compute_tile2, 2, y_ty)
            object_fifo_link(mm_result, mm_result_mem)

            # Set up a packet-switched flow from core to shim for tracing information
            tiles_to_trace = [compute_tile1, compute_tile2]
            if trace_size > 0:
                trace_utils.configure_packet_tracing_flow(tiles_to_trace, shim_tile2)
                            # trace_utils.CoreEvent.INSTR_VECTOR,

            # Setup for compute_tile1 - Matrix multiplication only
            @core(compute_tile1, f"linear_mm_{t}x{d}x{h}.o", stack_size=0xD00)
            def core_mm_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(tiles) if tiles > 1 else range(1):
                        elem_out_mm = mm_result.acquire(ObjectFifoPort.Produce, 1)
                        zero(elem_out_mm)

                        for _ in (
                            range_(D_div_d) if D_div_d > 1 else range(1)
                        ):
                            # Get input matrices
                            elem_inX = memX.acquire(ObjectFifoPort.Consume, 1)
                            elem_inW = memW.acquire(ObjectFifoPort.Consume, 1)

                            matmul(elem_inX, elem_inW, elem_out_mm)
                            memX.release(ObjectFifoPort.Consume, 1)
                            memW.release(ObjectFifoPort.Consume, 1)
                    
                    mm_result.release(ObjectFifoPort.Produce, 1)
            @core(compute_tile2, f"linear_bias_{t}x{d}x{h}.o", stack_size=0xD00)
            def core_bias_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(tiles) if tiles > 1 else range(1):
                        elem_mm_result = mm_result_mem.acquire(ObjectFifoPort.Consume, 1)
                        elem_inb = memB.acquire(ObjectFifoPort.Consume, 1)
                        elem_out = memY.acquire(ObjectFifoPort.Produce, 1)

                        row_wise_add_func(elem_mm_result, elem_inb, elem_out)

                        mm_result_mem.release(ObjectFifoPort.Consume, 1)
                        memB.release(ObjectFifoPort.Consume, 1)
                        memY.release(ObjectFifoPort.Produce, 1)

            @runtime_sequence(
                np.ndarray[(X_sz,), np.dtype[dtype_in]],
                np.ndarray[(W_sz,), np.dtype[dtype_in]],
                np.ndarray[(b_sz,), np.dtype[dtype_in]],
                np.ndarray[(Y_sz,), np.dtype[dtype_out]],
            )
            def sequence(X, W, B, Y):
                if enable_tracing:
                    trace_utils.configure_packet_tracing_aie2(
                        tiles_to_trace = [compute_tile1, compute_tile2],
                        shim=shim_tile2,
                        trace_size=trace_size,
                        coretile_events=[
                            # captures input X (PORT_RUNNING_0, at port number 1, master for inputs)
                            trace_utils.PortEvent(
                                trace_utils.CoreEvent.PORT_RUNNING_0,
                                port_number=1,
                                master=True,
                            ),
                            # captures input W (PORT_RUNNING_1, at port number 2, master for inputs)
                            trace_utils.PortEvent(
                                trace_utils.CoreEvent.PORT_RUNNING_1,
                                port_number=2,
                                master=True,
                            ),
                            # captures input b (PORT_RUNNING_2, at port number 3, master for inputs)
                            trace_utils.PortEvent(
                                trace_utils.CoreEvent.PORT_RUNNING_2,
                                port_number=3,
                                master=True,
                            ),
                            # captures output Y (PORT_RUNNING_3, at port number 1, slave for outputs)
                            trace_utils.PortEvent(
                                trace_utils.CoreEvent.PORT_RUNNING_3,
                                port_number=1,
                                master=False,
                            ),
                            trace_utils.CoreEvent.INSTR_EVENT_0,
                            trace_utils.CoreEvent.INSTR_EVENT_1,
                            trace_utils.CoreEvent.MEMORY_STALL,
                            trace_utils.CoreEvent.LOCK_STALL,
                            # trace_utils.CoreEvent.INSTR_VECTOR,
                        ],
                    )

                rows_per_block = 4
                for tile_row_block in range(ceildiv(T_div_t, rows_per_block)):
                    for pingpong in [0, 1]:
                        Y_row_offset = (
                            tile_row_block * rows_per_block * t * T
                            + pingpong * rows_per_block // 2 * t * T
                        )
                        row_base = (
                            tile_row_block * rows_per_block + pingpong + rows_per_block // 2
                        )
                        bd_id_base = 8 * pingpong
                        num_tile_rows = min([rows_per_block // 2, T_div_t - row_base])
                        if num_tile_rows <= 0: 
                            break
                        
                        npu_dma_memcpy_nd(
                            metadata=outY, 
                            bd_id=bd_id_base, 
                            mem=Y, 
                            offsets=[0, 0, 0, Y_row_offset],
                            sizes=[num_tile_rows, H // h, t, h], 
                            strides=[t * H, h, H, 1],
                        )

                        npu_dma_memcpy_nd(
                            metadata=inB, 
                            bd_id=bd_id_base + 6,
                            mem=B,
                            sizes=[1, 1, 1, H],  # 4D tensor format
                            strides=[H*2, H*2, H*2, 2])

                        for tile_row in range(num_tile_rows):
                            X_row_offset = (row_base + tile_row) * t * D
                            npu_dma_memcpy_nd(
                                metadata=inX, 
                                bd_id=bd_id_base + 2 * tile_row + 1, 
                                mem=X, 
                                offsets=[0, 0, 0, X_row_offset],
                                sizes=[H // h, D // d, t, d], 
                                strides=[0, d, D, 1],
                            )

                            npu_dma_memcpy_nd(
                                metadata=inW, 
                                bd_id=bd_id_base + 2 * tile_row + 2, 
                                mem=W, 
                                sizes = [H // h, D // d, d, h],
                                strides=[h, d * H, H, 1])
                        if tile_row_block > 0 or (tile_row_block == 0 and pingpong > 0):
                            dma_wait(outY)
                dma_wait(outY)

                # Finalize tracing if enabled
                if enable_tracing:
                    trace_utils.gen_trace_done_aie2(shim_tile2)

    print(ctx.module)

if __name__ == "__main__":
    main()
else:
    print("Not meant to be imported")
    sys.exit(1)
