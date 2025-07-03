module {
  aie.device(npu2) {
    memref.global "public" @mm_result_mem_cons : memref<64x32xbf16>
    memref.global "public" @mm_result_mem : memref<64x32xbf16>
    memref.global "public" @mm_result_cons : memref<64x32xbf16>
    memref.global "public" @mm_result : memref<64x32xbf16>
    memref.global "public" @outY_cons : memref<64x32xbf16>
    memref.global "public" @outY : memref<64x32xbf16>
    memref.global "public" @memY_cons : memref<64x32xbf16>
    memref.global "public" @memY : memref<64x32xbf16>
    memref.global "public" @memB_cons : memref<32xbf16>
    memref.global "public" @memB : memref<32xbf16>
    memref.global "public" @inB_cons : memref<32xbf16>
    memref.global "public" @inB : memref<32xbf16>
    memref.global "public" @memW_cons : memref<64x32xbf16>
    memref.global "public" @memW : memref<64x32xbf16>
    memref.global "public" @inW_cons : memref<64x32xbf16>
    memref.global "public" @inW : memref<64x32xbf16>
    memref.global "public" @memX_cons : memref<64x64xbf16>
    memref.global "public" @memX : memref<64x64xbf16>
    memref.global "public" @inX_cons : memref<64x64xbf16>
    memref.global "public" @inX : memref<64x64xbf16>
    func.func private @zero_bf16(memref<64x32xbf16>)
    func.func private @matmul_bf16_bf16(memref<64x64xbf16>, memref<64x32xbf16>, memref<64x32xbf16>)
    func.func private @row_wise_bias_add_bf16_bf16(memref<64x32xbf16>, memref<32xbf16>, memref<64x32xbf16>)
    %shim_noc_tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_noc_tile_1_0 = aie.tile(1, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %mem_tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %tile_0_3 = aie.tile(0, 3) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 29>}
    %mm_result_mem_cons_buff_0 = aie.buffer(%tile_0_3) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "mm_result_mem_cons_buff_0"} : memref<64x32xbf16> 
    %mm_result_mem_cons_buff_1 = aie.buffer(%tile_0_3) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "mm_result_mem_cons_buff_1"} : memref<64x32xbf16> 
    %mm_result_mem_cons_prod_lock_0 = aie.lock(%tile_0_3, 4) {init = 2 : i32, sym_name = "mm_result_mem_cons_prod_lock_0"}
    %mm_result_mem_cons_cons_lock_0 = aie.lock(%tile_0_3, 5) {init = 0 : i32, sym_name = "mm_result_mem_cons_cons_lock_0"}
    %mm_result_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 131072 : i32, mem_bank = 2 : i32, sym_name = "mm_result_cons_buff_0"} : memref<64x32xbf16> 
    %mm_result_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 196608 : i32, mem_bank = 3 : i32, sym_name = "mm_result_cons_buff_1"} : memref<64x32xbf16> 
    %mm_result_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 8) {init = 2 : i32, sym_name = "mm_result_cons_prod_lock_0"}
    %mm_result_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 9) {init = 0 : i32, sym_name = "mm_result_cons_cons_lock_0"}
    %mm_result_buff_0 = aie.buffer(%tile_0_2) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "mm_result_buff_0"} : memref<64x32xbf16> 
    %mm_result_buff_1 = aie.buffer(%tile_0_2) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "mm_result_buff_1"} : memref<64x32xbf16> 
    %mm_result_prod_lock_0 = aie.lock(%tile_0_2, 4) {init = 2 : i32, sym_name = "mm_result_prod_lock_0"}
    %mm_result_cons_lock_0 = aie.lock(%tile_0_2, 5) {init = 0 : i32, sym_name = "mm_result_cons_lock_0"}
    %outY_cons_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 2) {init = 1 : i32, sym_name = "outY_cons_prod_lock_0"}
    %outY_cons_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 3) {init = 0 : i32, sym_name = "outY_cons_cons_lock_0"}
    %memY_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 262144 : i32, mem_bank = 4 : i32, sym_name = "memY_cons_buff_0"} : memref<64x32xbf16> 
    %memY_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 327680 : i32, mem_bank = 5 : i32, sym_name = "memY_cons_buff_1"} : memref<64x32xbf16> 
    %memY_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 6) {init = 2 : i32, sym_name = "memY_cons_prod_lock_0"}
    %memY_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 7) {init = 0 : i32, sym_name = "memY_cons_cons_lock_0"}
    %memY_buff_0 = aie.buffer(%tile_0_3) {address = 32768 : i32, mem_bank = 2 : i32, sym_name = "memY_buff_0"} : memref<64x32xbf16> 
    %memY_buff_1 = aie.buffer(%tile_0_3) {address = 49152 : i32, mem_bank = 3 : i32, sym_name = "memY_buff_1"} : memref<64x32xbf16> 
    %memY_prod_lock_0 = aie.lock(%tile_0_3, 2) {init = 2 : i32, sym_name = "memY_prod_lock_0"}
    %memY_cons_lock_0 = aie.lock(%tile_0_3, 3) {init = 0 : i32, sym_name = "memY_cons_lock_0"}
    %memB_cons_buff_0 = aie.buffer(%tile_0_3) {address = 7424 : i32, mem_bank = 0 : i32, sym_name = "memB_cons_buff_0"} : memref<32xbf16> 
    %memB_cons_buff_1 = aie.buffer(%tile_0_3) {address = 20480 : i32, mem_bank = 1 : i32, sym_name = "memB_cons_buff_1"} : memref<32xbf16> 
    %memB_cons_prod_lock_0 = aie.lock(%tile_0_3, 0) {init = 2 : i32, sym_name = "memB_cons_prod_lock_0"}
    %memB_cons_cons_lock_0 = aie.lock(%tile_0_3, 1) {init = 0 : i32, sym_name = "memB_cons_cons_lock_0"}
    %inB_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 8192 : i32, mem_bank = 0 : i32, sym_name = "inB_cons_buff_0"} : memref<32xbf16> 
    %inB_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 73728 : i32, mem_bank = 1 : i32, sym_name = "inB_cons_buff_1"} : memref<32xbf16> 
    %inB_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 4) {init = 2 : i32, sym_name = "inB_cons_prod_lock_0"}
    %inB_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 5) {init = 0 : i32, sym_name = "inB_cons_cons_lock_0"}
    %inB_prod_lock_0 = aie.lock(%shim_noc_tile_1_0, 0) {init = 1 : i32, sym_name = "inB_prod_lock_0"}
    %inB_cons_lock_0 = aie.lock(%shim_noc_tile_1_0, 1) {init = 0 : i32, sym_name = "inB_cons_lock_0"}
    %memW_cons_buff_0 = aie.buffer(%tile_0_2) {address = 11520 : i32, mem_bank = 0 : i32, sym_name = "memW_cons_buff_0"} : memref<64x32xbf16> 
    %memW_cons_buff_1 = aie.buffer(%tile_0_2) {address = 24576 : i32, mem_bank = 1 : i32, sym_name = "memW_cons_buff_1"} : memref<64x32xbf16> 
    %memW_cons_prod_lock_0 = aie.lock(%tile_0_2, 2) {init = 2 : i32, sym_name = "memW_cons_prod_lock_0"}
    %memW_cons_cons_lock_0 = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "memW_cons_cons_lock_0"}
    %inW_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 393216 : i32, mem_bank = 6 : i32, sym_name = "inW_cons_buff_0"} : memref<64x32xbf16> 
    %inW_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 458752 : i32, mem_bank = 7 : i32, sym_name = "inW_cons_buff_1"} : memref<64x32xbf16> 
    %inW_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 2) {init = 2 : i32, sym_name = "inW_cons_prod_lock_0"}
    %inW_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 3) {init = 0 : i32, sym_name = "inW_cons_cons_lock_0"}
    %inW_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 2) {init = 1 : i32, sym_name = "inW_prod_lock_0"}
    %inW_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 3) {init = 0 : i32, sym_name = "inW_cons_lock_0"}
    %memX_cons_buff_0 = aie.buffer(%tile_0_2) {address = 3328 : i32, mem_bank = 0 : i32, sym_name = "memX_cons_buff_0"} : memref<64x64xbf16> 
    %memX_cons_buff_1 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "memX_cons_buff_1"} : memref<64x64xbf16> 
    %memX_cons_prod_lock_0 = aie.lock(%tile_0_2, 0) {init = 2 : i32, sym_name = "memX_cons_prod_lock_0"}
    %memX_cons_cons_lock_0 = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "memX_cons_cons_lock_0"}
    %inX_cons_buff_0 = aie.buffer(%mem_tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "inX_cons_buff_0"} : memref<64x64xbf16> 
    %inX_cons_buff_1 = aie.buffer(%mem_tile_0_1) {address = 65536 : i32, mem_bank = 1 : i32, sym_name = "inX_cons_buff_1"} : memref<64x64xbf16> 
    %inX_cons_prod_lock_0 = aie.lock(%mem_tile_0_1, 0) {init = 2 : i32, sym_name = "inX_cons_prod_lock_0"}
    %inX_cons_cons_lock_0 = aie.lock(%mem_tile_0_1, 1) {init = 0 : i32, sym_name = "inX_cons_cons_lock_0"}
    %inX_prod_lock_0 = aie.lock(%shim_noc_tile_0_0, 0) {init = 1 : i32, sym_name = "inX_prod_lock_0"}
    %inX_cons_lock_0 = aie.lock(%shim_noc_tile_0_0, 1) {init = 0 : i32, sym_name = "inX_cons_lock_0"}
    %switchbox_0_0 = aie.switchbox(%shim_noc_tile_0_0) {
      aie.connect<South : 3, North : 3>
      aie.connect<South : 7, North : 5>
      aie.connect<East : 3, North : 4>
      aie.connect<North : 3, East : 1>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<5> (3)
      %2 = aie.masterset(South : 0, %1) {keep_pkt_header = true}
      %3 = aie.masterset(East : 2, %0)
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %1)
      }
      aie.packet_rules(North : 2) {
        aie.rule(31, 1, %0)
      }
    }
    %shim_mux_0_0 = aie.shim_mux(%shim_noc_tile_0_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
    }
    %switchbox_0_1 = aie.switchbox(%mem_tile_0_1) {
      aie.connect<South : 3, DMA : 0>
      aie.connect<DMA : 0, North : 1>
      aie.connect<South : 5, DMA : 1>
      aie.connect<DMA : 1, North : 5>
      aie.connect<South : 4, DMA : 2>
      aie.connect<DMA : 2, North : 0>
      aie.connect<North : 3, DMA : 3>
      aie.connect<DMA : 3, South : 3>
      aie.connect<North : 1, DMA : 4>
      aie.connect<DMA : 4, North : 2>
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(South : 2, %0)
      aie.packet_rules(North : 2) {
        aie.rule(31, 1, %0)
      }
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 1, DMA : 0>
      aie.connect<South : 5, DMA : 1>
      aie.connect<South : 0, North : 5>
      aie.connect<North : 0, South : 3>
      aie.connect<DMA : 0, South : 1>
      aie.connect<South : 2, North : 0>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.masterset(South : 2, %1)
      %3 = aie.masterset(East : 3, %0)
      aie.packet_rules(North : 1) {
        aie.rule(31, 2, %0)
      }
      aie.packet_rules(Trace : 0) {
        aie.rule(31, 1, %1)
      }
    }
    %switchbox_1_0 = aie.switchbox(%shim_noc_tile_1_0) {
      aie.connect<South : 3, West : 3>
      aie.connect<West : 1, South : 2>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<5> (3)
      %2 = aie.masterset(South : 3, %0) {keep_pkt_header = true}
      %3 = aie.masterset(South : 0, %1) {keep_pkt_header = true}
      aie.packet_rules(TileControl : 0) {
        aie.rule(31, 15, %1)
      }
      aie.packet_rules(North : 2) {
        aie.rule(31, 2, %0)
      }
      aie.packet_rules(West : 2) {
        aie.rule(31, 1, %0)
      }
    }
    %shim_mux_1_0 = aie.shim_mux(%shim_noc_tile_1_0) {
      aie.connect<North : 3, DMA : 1>
      aie.connect<DMA : 0, North : 3>
      aie.connect<North : 2, DMA : 0>
    }
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      aie.connect<South : 5, DMA : 0>
      aie.connect<DMA : 0, South : 0>
      aie.connect<South : 0, DMA : 1>
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(South : 1, %0)
      aie.packet_rules(Trace : 0) {
        aie.rule(31, 2, %0)
      }
    }
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
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb11
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb12
    ^bb2:  // pred: ^bb1
      %c0_0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c1_1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb3(%c0_0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb10
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb11
    ^bb4:  // pred: ^bb3
      aie.use_lock(%mm_result_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @zero_bf16(%mm_result_buff_0) : (memref<64x32xbf16>) -> ()
      %c0_2 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1_3 = arith.constant 1 : index
      %c2_4 = arith.constant 2 : index
      cf.br ^bb5(%c0_2 : index)
    ^bb5(%4: index):  // 2 preds: ^bb4, ^bb6
      %5 = arith.cmpi slt, %4, %c4 : index
      cf.cond_br %5, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      aie.use_lock(%memX_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%memW_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%memX_cons_buff_0, %memW_cons_buff_0, %mm_result_buff_0) : (memref<64x64xbf16>, memref<64x32xbf16>, memref<64x32xbf16>) -> ()
      aie.use_lock(%memX_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memW_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memX_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%memW_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%memX_cons_buff_1, %memW_cons_buff_1, %mm_result_buff_0) : (memref<64x64xbf16>, memref<64x32xbf16>, memref<64x32xbf16>) -> ()
      aie.use_lock(%memX_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memW_cons_prod_lock_0, Release, 1)
      %6 = arith.addi %4, %c2_4 : index
      cf.br ^bb5(%6 : index)
    ^bb7:  // pred: ^bb5
      func.call @zero_bf16(%mm_result_buff_0) : (memref<64x32xbf16>) -> ()
      %c0_5 = arith.constant 0 : index
      %c4_6 = arith.constant 4 : index
      %c1_7 = arith.constant 1 : index
      %c2_8 = arith.constant 2 : index
      cf.br ^bb8(%c0_5 : index)
    ^bb8(%7: index):  // 2 preds: ^bb7, ^bb9
      %8 = arith.cmpi slt, %7, %c4_6 : index
      cf.cond_br %8, ^bb9, ^bb10
    ^bb9:  // pred: ^bb8
      aie.use_lock(%memX_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%memW_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%memX_cons_buff_0, %memW_cons_buff_0, %mm_result_buff_0) : (memref<64x64xbf16>, memref<64x32xbf16>, memref<64x32xbf16>) -> ()
      aie.use_lock(%memX_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memW_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memX_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%memW_cons_cons_lock_0, AcquireGreaterEqual, 1)
      func.call @matmul_bf16_bf16(%memX_cons_buff_1, %memW_cons_buff_1, %mm_result_buff_0) : (memref<64x64xbf16>, memref<64x32xbf16>, memref<64x32xbf16>) -> ()
      aie.use_lock(%memX_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memW_cons_prod_lock_0, Release, 1)
      %9 = arith.addi %7, %c2_8 : index
      cf.br ^bb8(%9 : index)
    ^bb10:  // pred: ^bb8
      %10 = arith.addi %2, %c2 : index
      cf.br ^bb3(%10 : index)
    ^bb11:  // pred: ^bb3
      aie.use_lock(%mm_result_cons_lock_0, Release, 1)
      %11 = arith.addi %0, %c1 : index
      cf.br ^bb1(%11 : index)
    ^bb12:  // pred: ^bb1
      aie.end
    } {link_with = "linear_mm_64x64x32.o", stack_size = 3328 : i32}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
      %1 = arith.cmpi slt, %0, %c4294967295 : index
      cf.cond_br %1, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      %c0_0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c1_1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb3(%c0_0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c32 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%mm_result_mem_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%memB_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%memY_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @row_wise_bias_add_bf16_bf16(%mm_result_mem_cons_buff_0, %memB_cons_buff_0, %memY_buff_0) : (memref<64x32xbf16>, memref<32xbf16>, memref<64x32xbf16>) -> ()
      aie.use_lock(%mm_result_mem_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memB_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memY_cons_lock_0, Release, 1)
      aie.use_lock(%mm_result_mem_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%memB_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.use_lock(%memY_prod_lock_0, AcquireGreaterEqual, 1)
      func.call @row_wise_bias_add_bf16_bf16(%mm_result_mem_cons_buff_1, %memB_cons_buff_1, %memY_buff_1) : (memref<64x32xbf16>, memref<32xbf16>, memref<64x32xbf16>) -> ()
      aie.use_lock(%mm_result_mem_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memB_cons_prod_lock_0, Release, 1)
      aie.use_lock(%memY_cons_lock_0, Release, 1)
      %4 = arith.addi %2, %c2 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      %5 = arith.addi %0, %c1 : index
      cf.br ^bb1(%5 : index)
    ^bb6:  // pred: ^bb1
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
    aie.shim_dma_allocation @inX(MM2S, 0, 0)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%inX_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inX_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%inX_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%inX_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inX_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%inX_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%inX_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inX_cons_buff_0 : memref<64x64xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>]) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%inX_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%inX_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inX_cons_buff_1 : memref<64x64xbf16>, 0, 4096, [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>]) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%inX_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%inW_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inW_cons_buff_0 : memref<64x32xbf16>, 0, 2048) {bd_id = 24 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%inW_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%inW_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inW_cons_buff_1 : memref<64x32xbf16>, 0, 2048) {bd_id = 25 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%inW_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%inW_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inW_cons_buff_0 : memref<64x32xbf16>, 0, 2048, [<size = 8, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 26 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%inW_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%inW_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inW_cons_buff_1 : memref<64x32xbf16>, 0, 2048, [<size = 8, stride = 256>, <size = 4, stride = 8>, <size = 8, stride = 32>, <size = 8, stride = 1>]) {bd_id = 27 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%inW_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%inB_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inB_cons_buff_0 : memref<32xbf16>, 0, 32) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%inB_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%inB_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inB_cons_buff_1 : memref<32xbf16>, 0, 32) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%inB_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(MM2S, 2, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%inB_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inB_cons_buff_0 : memref<32xbf16>, 0, 32) {bd_id = 6 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%inB_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%inB_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%inB_cons_buff_1 : memref<32xbf16>, 0, 32) {bd_id = 7 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%inB_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      %6 = aie.dma_start(S2MM, 3, ^bb19, ^bb21)
    ^bb19:  // 2 preds: ^bb18, ^bb20
      aie.use_lock(%memY_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memY_cons_buff_0 : memref<64x32xbf16>, 0, 2048) {bd_id = 28 : i32, next_bd_id = 29 : i32}
      aie.use_lock(%memY_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb20
    ^bb20:  // pred: ^bb19
      aie.use_lock(%memY_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memY_cons_buff_1 : memref<64x32xbf16>, 0, 2048) {bd_id = 29 : i32, next_bd_id = 28 : i32}
      aie.use_lock(%memY_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb19
    ^bb21:  // pred: ^bb18
      %7 = aie.dma_start(MM2S, 3, ^bb22, ^bb24)
    ^bb22:  // 2 preds: ^bb21, ^bb23
      aie.use_lock(%memY_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memY_cons_buff_0 : memref<64x32xbf16>, 0, 2048, [<size = 16, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>]) {bd_id = 30 : i32, next_bd_id = 31 : i32}
      aie.use_lock(%memY_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb23
    ^bb23:  // pred: ^bb22
      aie.use_lock(%memY_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memY_cons_buff_1 : memref<64x32xbf16>, 0, 2048, [<size = 16, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>]) {bd_id = 31 : i32, next_bd_id = 30 : i32}
      aie.use_lock(%memY_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb22
    ^bb24:  // pred: ^bb21
      %8 = aie.dma_start(S2MM, 4, ^bb25, ^bb27)
    ^bb25:  // 2 preds: ^bb24, ^bb26
      aie.use_lock(%mm_result_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%mm_result_cons_buff_0 : memref<64x32xbf16>, 0, 2048) {bd_id = 8 : i32, next_bd_id = 9 : i32}
      aie.use_lock(%mm_result_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb26
    ^bb26:  // pred: ^bb25
      aie.use_lock(%mm_result_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%mm_result_cons_buff_1 : memref<64x32xbf16>, 0, 2048) {bd_id = 9 : i32, next_bd_id = 8 : i32}
      aie.use_lock(%mm_result_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb25
    ^bb27:  // pred: ^bb24
      %9 = aie.dma_start(MM2S, 4, ^bb28, ^bb30)
    ^bb28:  // 2 preds: ^bb27, ^bb29
      aie.use_lock(%mm_result_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%mm_result_cons_buff_0 : memref<64x32xbf16>, 0, 2048) {bd_id = 10 : i32, next_bd_id = 11 : i32}
      aie.use_lock(%mm_result_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb29
    ^bb29:  // pred: ^bb28
      aie.use_lock(%mm_result_cons_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%mm_result_cons_buff_1 : memref<64x32xbf16>, 0, 2048) {bd_id = 11 : i32, next_bd_id = 10 : i32}
      aie.use_lock(%mm_result_cons_prod_lock_0, Release, 1)
      aie.next_bd ^bb28
    ^bb30:  // pred: ^bb27
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memX_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memX_cons_buff_0 : memref<64x64xbf16>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%memX_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memX_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memX_cons_buff_1 : memref<64x64xbf16>, 0, 4096) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%memX_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%memW_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memW_cons_buff_0 : memref<64x32xbf16>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%memW_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%memW_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memW_cons_buff_1 : memref<64x32xbf16>, 0, 2048) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%memW_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%mm_result_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%mm_result_buff_0 : memref<64x32xbf16>, 0, 2048) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%mm_result_prod_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%mm_result_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%mm_result_buff_1 : memref<64x32xbf16>, 0, 2048) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%mm_result_prod_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @inW(MM2S, 1, 0)
    aie.shim_dma_allocation @inB(MM2S, 0, 1)
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%memB_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memB_cons_buff_0 : memref<32xbf16>, 0, 32) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%memB_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%memB_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memB_cons_buff_1 : memref<32xbf16>, 0, 32) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%memB_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%memY_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memY_buff_0 : memref<64x32xbf16>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%memY_prod_lock_0, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%memY_cons_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%memY_buff_1 : memref<64x32xbf16>, 0, 2048) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%memY_prod_lock_0, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%mm_result_mem_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%mm_result_mem_cons_buff_0 : memref<64x32xbf16>, 0, 2048) {bd_id = 4 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%mm_result_mem_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%mm_result_mem_cons_prod_lock_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%mm_result_mem_cons_buff_1 : memref<64x32xbf16>, 0, 2048) {bd_id = 5 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%mm_result_mem_cons_cons_lock_0, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @outY(S2MM, 0, 1)
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_0_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%shim_noc_tile_1_0, TileControl : 0>
      aie.packet_dest<%shim_noc_tile_1_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    %mem_tile_1_1 = aie.tile(1, 1)
    %switchbox_1_1 = aie.switchbox(%mem_tile_1_1) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(South : 2, %0)
      aie.packet_rules(North : 2) {
        aie.rule(31, 2, %0)
      }
    }
    %tile_1_2 = aie.tile(1, 2)
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.masterset(South : 2, %0)
      aie.packet_rules(West : 3) {
        aie.rule(31, 2, %0)
      }
    }
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%shim_noc_tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%mem_tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%mem_tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
    aie.wire(%tile_0_3 : Core, %switchbox_0_3 : Core)
    aie.wire(%tile_0_3 : DMA, %switchbox_0_3 : DMA)
    aie.wire(%switchbox_0_2 : North, %switchbox_0_3 : South)
    aie.wire(%switchbox_0_0 : East, %switchbox_1_0 : West)
    aie.wire(%shim_mux_1_0 : North, %switchbox_1_0 : South)
    aie.wire(%shim_noc_tile_1_0 : DMA, %shim_mux_1_0 : DMA)
    aie.wire(%switchbox_0_1 : East, %switchbox_1_1 : West)
    aie.wire(%mem_tile_1_1 : Core, %switchbox_1_1 : Core)
    aie.wire(%mem_tile_1_1 : DMA, %switchbox_1_1 : DMA)
    aie.wire(%switchbox_1_0 : North, %switchbox_1_1 : South)
    aie.wire(%switchbox_0_2 : East, %switchbox_1_2 : West)
    aie.wire(%tile_1_2 : Core, %switchbox_1_2 : Core)
    aie.wire(%tile_1_2 : DMA, %switchbox_1_2 : DMA)
    aie.wire(%switchbox_1_1 : North, %switchbox_1_2 : South)
  }
}

