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
    argparser.add_argument("-B", type=int, default=1)
    argparser.add_argument("-T", type=int, default=56)
    argparser.add_argument("-D", type=int, default=41)
    argparser.add_argument("-H", type=int, default=4)

    # For this code, assume both --dtype_in and --dtype_out are bf16
    argparser.add_argument("--trace_size", type=int, default=0)
    args = argparser.parse_args()

    # Call to the linear function
    linear(
        args.dev,
        args.B,
        args.T,
        args.D,
        args.H,
        args.trace_size,
    )

def linear(
    dev, B, T, D, H, trace_size      
):
    # No tiling needed for the implementation 
    vectorized = True
    enable_tracing = True if trace_size > 0 else False

    dtype_in = np.bfloat16
    dtype_out = np.bfloat16

    X_sz = B * T * D
    W_sz = H * D
    b_sz = H
    i_sz = T * H
    Y_sz = B * T * H

    with mlir_mod_ctx() as ctx:
        
        Y_sz_in_bytes = Y_sz * np.dtype(dtype_out).itemsize

        if dev == "npu":
            dev_ty = AIEDevice.npu1_1col
        else:
            dev_ty = AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            # x_ty is declared as (T, D) instead of (B, T, D) since the value of B is always 1
            x_ty = np.ndarray[(T, D), np.dtype[dtype_in]]
            w_ty = np.ndarray[(H, D), np.dtype[dtype_in]]
            b_ty = np.ndarray[(H), np.dtype[dtype_in]]
            y_ty = np.ndarray[(B, T, H), np.dtype(dtype_out)]
            # Type of intermediate variable, represents the product of matrix multiplication of X and W
            i_ty = np.ndarray[(T, H), np.dtype(dtype_out)]

            # Compute the transform pattern of weight W
            transpose_W = TensorAccessPattern(
                (H, D), offset=0, sizes=[1, 1, D, H], strides=[1, 1, 1, D]
            )

            # AIE Core Function declarations - only matmul is used in this case
            # func_type = "" if vectorized else "scalar_"
            # zero = external_func(f"zero_{func_type}{dtype_out_str}", inputs=[y_ty])
            
            # eltwise_mul_bf16_scalar = external_func(
            #     "eltwise_mul_bf16_scalar", inputs = [tile_ty, tile_ty, tile_ty]
            # )
            # eltwise_mul_bf16_vector = external_func(
            #     "eltwise_mul_bf16_vector", inputs = [tile_ty, tile_ty, tile_ty]
            # )
            # reduce_add_vector = Kernel(
            #     "reduce_add_vector", "reduce_add.cc.o", [in_ty, out_ty, np.bfloat16]
            # )
            # function_type = "" if vectorized else "scalar_"
            # zero = external_func(f"zero_{func_type}{dtype_out_str}", inputs=[y_ty])
            matmul = external_func(
                matmul_func_name,
                inputs=[x_ty, w_ty, i_ty]
            )

            # Tile declarations - can change later
            shim_tile = tile(0, 0)
            mem_tile = tile(0, 1)
            compute_tile2_col, compute_tile2_row = 0, 2
            compute_tile2 = tile(compute_tile2_col, compute_tile2_row)

            # AIE-array data movement with object fifos
            # Input X
            inX = object_fifo("inX", shim_tile, mem_tile, 1, x_ty)
            memX = object_fifo("memX", mem_tile, compute_tile2, 1, x_ty)
            obejct_fifo_link(inX, memX)

            # Input W
            inW = object_fifo("inW", shim_tile, mem_tile, 1, w_ty)
            memW = object_fifo("memW", mem_tile, compute_tile2, 1, w_ty)
            object_fifo_link(inW, memW)

            # Input b
            inB = object_fifo("inB", shim_tile, mem_tile, 1, b_ty)
            memB = object_fifo("memB", mem_tile, compute_tile2, 1, b_ty)
            object_fifo_link(inB, memB)

            # Output Y
            memY = object_fifo("memY", compute_tile2, mem_tile, 2, y_ty)
            outY = object_fifo("outY", mem_tile, shim_tile, 1, y_ty)
            object_fifo_link(memY, outY)

            # Buffer declarations to hold intermediate values
            intermediate = buffer(
                compute_tile2,
                i_ty,
                "intermediate", 
                use_write_buffer = False,
            )

            # Set up a packet-switched flow from core to shim for tracing information
            tiles_to_trace = [compute_tile2]
            if trace_size > 0:
                trace_utils.configure_packet_tracing_flow(tiles_to_trace, shim_tile)

            tiles = 1
            # Set up compute tiles - compute tile 2
            @core(compute_tile2, f"linear_{B}x{T}x{D}x{H}.o")
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(tiles) if tiles > 1 else range(1):
                        elem_out = memY.acquire(ObjectFifoPort.Produce, 1)
                        elem_inX = memX.acquire(ObjectFifoPort.Consume, 1)
                        elem_inW = memW.acquire(ObjectFifoPort.Consume, 1)
                        elem_inB = memB.acquire(ObjectFifoPort.Consume, 1)

                        matmul(elem_inX, elem_inW, intermediate)
                        memX.release(ObjectFifoPort.Consume, 1)
                        memW.release(ObjectFifoPort.Consume, 1)

                        for i in range(T):
                            for j in range(H):
                                elem_out = intermediate[i, j] + elem_inB[j]

                        memB.release(ObjectFifoPort.Consume, 1)
                        memY.release(ObjectFIfoPort.Produce, 1)

            @runtime_sequence(
                np.ndarray[(X_sz,), np.dtype[dtype_in]],
                np.ndarray[(W_sz,), np.dtype[dtype_in]],
                np.ndarray[(b_sz,), np.dtype(dtype_in)],
                np.ndarray[(Y_sz,), np.dtype[dtype_out]],
            )
            def sequence(X, W, B, Y):
                if enable_tracing:
                    trace_utils.configure_packet_tracing_aie2(
                        tiles_to_trace = tiles_to_trace,
                        shim=shim_tile,
                        trace_size=trace_size,
                        coretile_event=[
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
                            trace_utils.CoreEvent.INSTR_EVENT_2,
                            trace_utils.CoreEvent.MEMORY_STALL,
                            trace_utils.CoreEvent.LOCK_STALL,
                            trace_utils.CoreEvent.INSTR_VECTOR,
                        ],
                    )

                # Transfer input data into AIE. Take the transpose of matrix W in the process
                npu_dma_memcpy_nd(metadata=memX, bd_id=0, mem=X)
                npu_dma_memcpy_nd(metadata=memW, bd_id=1, mem=W, tap=transpose_W)
                npu_dma_memcpy_nd(metadata=memB, bd_id=2, mem=B)
                
                # Set up output DMA transfer
                npu_dma_memcpy_nd(metadata=outY, bd_id=3, mem=Y)

                # Wait for all transfers to complete
                dma_wait(inX, inW, inB)
                
                # Wait for computation to finish and output to be ready
                dma_wait(outY)

                # Finalize tracing if enabled
                if enable_tracing:
                    trace_utils.gen_trace_done_aie2(shim_tile)

    print(ctx.module)

if __name__ == "__main__":
    main()
else:
    print("Not meant to be imported")
    sys.exit(1)
