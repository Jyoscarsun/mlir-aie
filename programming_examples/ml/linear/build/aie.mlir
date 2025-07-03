module {
  aie.device(npu2) {
    func.func private @zero_bf16(memref<64x32xbf16>)
    func.func private @matmul_bf16_bf16(memref<64x64xbf16>, memref<64x32xbf16>, memref<64x32xbf16>)
    func.func private @row_wise_bias_add_bf16_bf16(memref<64x32xbf16>, memref<32xbf16>, memref<64x32xbf16>)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    aie.objectfifo @inX(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo @memX(%mem_tile_0_1 dimensionsToStream [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>> 
    aie.objectfifo.link [@inX] -> [@memX]([] [])
    aie.objectfifo @inW(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x32xbf16>> 
    aie.objectfifo @memW(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x32xbf16>> 
    aie.objectfifo.link [@inW] -> [@memW]([] [])
    aie.objectfifo @inB(%shim_noc_tile_1_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32xbf16>> 
    aie.objectfifo @memB(%mem_tile_0_1, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<32xbf16>> 
    aie.objectfifo.link [@inB] -> [@memB]([] [])
    aie.objectfifo @memY(%tile_0_3, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x32xbf16>> 
    aie.objectfifo @outY(%mem_tile_0_1 dimensionsToStream [<size = 16, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<64x32xbf16>> 
    aie.objectfifo.link [@memY] -> [@outY]([] [])
    aie.objectfifo @mm_result(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x32xbf16>> 
    aie.objectfifo @mm_result_mem(%mem_tile_0_1, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<64x32xbf16>> 
    aie.objectfifo.link [@mm_result] -> [@mm_result_mem]([] [])
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_2, Trace : 0>
      aie.packet_dest<%shim_noc_tile_1_0, DMA : 1>
    } {keep_pkt_header = true}
    aie.packet_flow(2) {
      aie.packet_source<%tile_0_3, Trace : 0>
      aie.packet_dest<%shim_noc_tile_1_0, DMA : 1>
    } {keep_pkt_header = true}
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c32 step %c1_1 {
          %0 = aie.objectfifo.acquire @mm_result(Produce, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
          func.call @zero_bf16(%1) : (memref<64x32xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c4 step %c1_3 {
            %2 = aie.objectfifo.acquire @memX(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %4 = aie.objectfifo.acquire @memW(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
            func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<64x64xbf16>, memref<64x32xbf16>, memref<64x32xbf16>) -> ()
            aie.objectfifo.release @memX(Consume, 1)
            aie.objectfifo.release @memW(Consume, 1)
          }
        }
        aie.objectfifo.release @mm_result(Produce, 1)
      }
      aie.end
    } {link_with = "linear_mm_64x64x32.o", stack_size = 3328 : i32}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c32 step %c1_1 {
          %0 = aie.objectfifo.acquire @mm_result_mem(Consume, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
          %2 = aie.objectfifo.acquire @memB(Consume, 1) : !aie.objectfifosubview<memref<32xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32xbf16>> -> memref<32xbf16>
          %4 = aie.objectfifo.acquire @memY(Produce, 1) : !aie.objectfifosubview<memref<64x32xbf16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x32xbf16>> -> memref<64x32xbf16>
          func.call @row_wise_bias_add_bf16_bf16(%1, %3, %5) : (memref<64x32xbf16>, memref<32xbf16>, memref<64x32xbf16>) -> ()
          aie.objectfifo.release @mm_result_mem(Consume, 1)
          aie.objectfifo.release @memB(Consume, 1)
          aie.objectfifo.release @memY(Produce, 1)
        }
      }
      aie.end
    } {link_with = "linear_bias_64x64x32.o", stack_size = 3328 : i32}
    aiex.runtime_sequence @sequence(%arg0: memref<65536xbf16>, %arg1: memref<65536xbf16>, %arg2: memref<256xbf16>, %arg3: memref<65536xbf16>) {
      aiex.npu.write32 {address = 213200 : ui32, column = 0 : i32, row = 2 : i32, value = 2038038528 : ui32}
      aiex.npu.write32 {address = 213204 : ui32, column = 0 : i32, row = 2 : i32, value = 1 : ui32}
      aiex.npu.write32 {address = 213216 : ui32, column = 0 : i32, row = 2 : i32, value = 1465077579 : ui32}
      aiex.npu.write32 {address = 213220 : ui32, column = 0 : i32, row = 2 : i32, value = 437723681 : ui32}
      aiex.npu.write32 {address = 261888 : ui32, column = 0 : i32, row = 2 : i32, value = 19079713 : ui32}
      aiex.npu.write32 {address = 261892 : ui32, column = 0 : i32, row = 2 : i32, value = 0 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 2 : i32, value = 31232 : ui32}
      aiex.npu.writebd {bd_id = 15 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 1 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 1 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 33673700 : ui32, arg_idx = 4 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119308 : ui32, column = 1 : i32, row = 0 : i32, value = 15 : ui32}
      aiex.npu.write32 {address = 213200 : ui32, column = 0 : i32, row = 3 : i32, value = 2038038528 : ui32}
      aiex.npu.write32 {address = 213204 : ui32, column = 0 : i32, row = 3 : i32, value = 2 : ui32}
      aiex.npu.write32 {address = 213216 : ui32, column = 0 : i32, row = 3 : i32, value = 1465077579 : ui32}
      aiex.npu.write32 {address = 213220 : ui32, column = 0 : i32, row = 3 : i32, value = 437723681 : ui32}
      aiex.npu.write32 {address = 261888 : ui32, column = 0 : i32, row = 3 : i32, value = 19079713 : ui32}
      aiex.npu.write32 {address = 261892 : ui32, column = 0 : i32, row = 3 : i32, value = 0 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 3 : i32, value = 31232 : ui32}
      aiex.npu.writebd {bd_id = 14 : i32, buffer_length = 8192 : i32, buffer_offset = 0 : i32, burst_length = 64 : i32, column = 1 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 2 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
      aiex.npu.address_patch {addr = 33673668 : ui32, arg_idx = 4 : i32, arg_plus = 0 : i32}
      aiex.npu.write32 {address = 119308 : ui32, column = 1 : i32, row = 0 : i32, value = 14 : ui32}
      aiex.npu.write32 {address = 212992 : ui32, column = 1 : i32, row = 0 : i32, value = 32512 : ui32}
      aiex.npu.write32 {address = 213068 : ui32, column = 1 : i32, row = 0 : i32, value = 127 : ui32}
      aiex.npu.write32 {address = 213000 : ui32, column = 1 : i32, row = 0 : i32, value = 127 : ui32}
      aiex.npu.dma_memcpy_nd(%arg3[0, 0, 0, 0][2, 8, 64, 32][16384, 32, 256, 1]) {id = 0 : i64, metadata = @outY} : memref<65536xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 1, 1, 256][512, 512, 512, 2]) {id = 6 : i64, metadata = @inB} : memref<256xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 32768][8, 4, 64, 64][0, 64, 256, 1]) {id = 1 : i64, metadata = @inX} : memref<65536xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 4, 64, 32][32, 16384, 256, 1]) {id = 2 : i64, metadata = @inW} : memref<65536xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 49152][8, 4, 64, 64][0, 64, 256, 1]) {id = 3 : i64, metadata = @inX} : memref<65536xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 4, 64, 32][32, 16384, 256, 1]) {id = 4 : i64, metadata = @inW} : memref<65536xbf16>
      aiex.npu.dma_memcpy_nd(%arg3[0, 0, 0, 32768][1, 8, 64, 32][16384, 32, 256, 1]) {id = 8 : i64, metadata = @outY} : memref<65536xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 1, 1, 256][512, 512, 512, 2]) {id = 14 : i64, metadata = @inB} : memref<256xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 49152][8, 4, 64, 64][0, 64, 256, 1]) {id = 9 : i64, metadata = @inX} : memref<65536xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][8, 4, 64, 32][32, 16384, 256, 1]) {id = 10 : i64, metadata = @inW} : memref<65536xbf16>
      aiex.npu.dma_wait {symbol = @outY}
      aiex.npu.dma_wait {symbol = @outY}
      aiex.npu.write32 {address = 213064 : ui32, column = 1 : i32, row = 0 : i32, value = 126 : ui32}
      aiex.npu.write32 {address = 213000 : ui32, column = 1 : i32, row = 0 : i32, value = 126 : ui32}
    }
  }
}

