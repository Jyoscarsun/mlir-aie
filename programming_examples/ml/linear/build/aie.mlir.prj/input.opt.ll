; ModuleID = '/home/npu-sm/mlir-aie/programming_examples/ml/linear/build/aie.mlir.prj/input.llpeanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@memX_cons_buff_1 = external global [64 x [64 x bfloat]]
@memX_cons_buff_0 = external global [64 x [64 x bfloat]]
@memW_cons_buff_1 = external global [64 x [32 x bfloat]]
@memW_cons_buff_0 = external global [64 x [32 x bfloat]]
@memB_cons_buff_1 = external global [32 x bfloat]
@memB_cons_buff_0 = external global [32 x bfloat]
@memY_buff_1 = external global [64 x [32 x bfloat]]
@memY_buff_0 = external global [64 x [32 x bfloat]]
@mm_result_buff_0 = external global [64 x [32 x bfloat]]
@mm_result_mem_cons_buff_1 = external global [64 x [32 x bfloat]]
@mm_result_mem_cons_buff_0 = external global [64 x [32 x bfloat]]

; Function Attrs: nounwind
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: nounwind
declare void @llvm.aie2p.release(i32, i32) #0

declare void @zero_bf16(ptr) local_unnamed_addr

declare void @matmul_bf16_bf16(ptr, ptr, ptr) local_unnamed_addr

declare void @row_wise_bias_add_bf16_bf16(ptr, ptr, ptr) local_unnamed_addr

define void @core_0_3() local_unnamed_addr {
  br label %.preheader

.preheader:                                       ; preds = %0, %.preheader
  %1 = phi i64 [ 0, %0 ], [ %2, %.preheader ]
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_0, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_0, ptr nonnull @memB_cons_buff_0, ptr nonnull @memY_buff_0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_1, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_1, ptr nonnull @memB_cons_buff_1, ptr nonnull @memY_buff_1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_0, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_0, ptr nonnull @memB_cons_buff_0, ptr nonnull @memY_buff_0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_1, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_1, ptr nonnull @memB_cons_buff_1, ptr nonnull @memY_buff_1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_0, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_0, ptr nonnull @memB_cons_buff_0, ptr nonnull @memY_buff_0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_1, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_1, ptr nonnull @memB_cons_buff_1, ptr nonnull @memY_buff_1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_0, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_0, ptr nonnull @memB_cons_buff_0, ptr nonnull @memY_buff_0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_1, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_1, ptr nonnull @memB_cons_buff_1, ptr nonnull @memY_buff_1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_0, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_0, ptr nonnull @memB_cons_buff_0, ptr nonnull @memY_buff_0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_1, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_1, ptr nonnull @memB_cons_buff_1, ptr nonnull @memY_buff_1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_0, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_0, ptr nonnull @memB_cons_buff_0, ptr nonnull @memY_buff_0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_1, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_1, ptr nonnull @memB_cons_buff_1, ptr nonnull @memY_buff_1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_0, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_0, ptr nonnull @memB_cons_buff_0, ptr nonnull @memY_buff_0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_1, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_1, ptr nonnull @memB_cons_buff_1, ptr nonnull @memY_buff_1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_0, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_0, ptr nonnull @memB_cons_buff_0, ptr nonnull @memY_buff_0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_1, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_1, ptr nonnull @memB_cons_buff_1, ptr nonnull @memY_buff_1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_0, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_0, ptr nonnull @memB_cons_buff_0, ptr nonnull @memY_buff_0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_1, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_1, ptr nonnull @memB_cons_buff_1, ptr nonnull @memY_buff_1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_0, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_0, ptr nonnull @memB_cons_buff_0, ptr nonnull @memY_buff_0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_1, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_1, ptr nonnull @memB_cons_buff_1, ptr nonnull @memY_buff_1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_0, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_0, ptr nonnull @memB_cons_buff_0, ptr nonnull @memY_buff_0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_1, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_1, ptr nonnull @memB_cons_buff_1, ptr nonnull @memY_buff_1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_0, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_0, ptr nonnull @memB_cons_buff_0, ptr nonnull @memY_buff_0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_1, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_1, ptr nonnull @memB_cons_buff_1, ptr nonnull @memY_buff_1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_0, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_0, ptr nonnull @memB_cons_buff_0, ptr nonnull @memY_buff_0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_1, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_1, ptr nonnull @memB_cons_buff_1, ptr nonnull @memY_buff_1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_0, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_0, ptr nonnull @memB_cons_buff_0, ptr nonnull @memY_buff_0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_1, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_1, ptr nonnull @memB_cons_buff_1, ptr nonnull @memY_buff_1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_0, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_0, ptr nonnull @memB_cons_buff_0, ptr nonnull @memY_buff_0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_1, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_1, ptr nonnull @memB_cons_buff_1, ptr nonnull @memY_buff_1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_0, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_0, ptr nonnull @memB_cons_buff_0, ptr nonnull @memY_buff_0)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 53, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_1, i64 32) ]
  tail call void @row_wise_bias_add_bf16_bf16(ptr nonnull @mm_result_mem_cons_buff_1, ptr nonnull @memB_cons_buff_1, ptr nonnull @memY_buff_1)
  tail call void @llvm.aie2p.release(i32 52, i32 1)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %2 = add nuw nsw i64 %1, 1
  %3 = icmp ult i64 %1, 4294967294
  br i1 %3, label %.preheader, label %4

4:                                                ; preds = %.preheader
  ret void
}

define void @core_0_2() local_unnamed_addr {
  br label %.preheader

.preheader:                                       ; preds = %0, %7
  %1 = phi i64 [ 0, %0 ], [ %8, %7 ]
  br label %2

2:                                                ; preds = %2, %.preheader
  %3 = phi i64 [ 0, %.preheader ], [ %5, %2 ]
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_buff_0, i64 32) ]
  tail call void @zero_bf16(ptr nonnull @mm_result_buff_0)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memW_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memX_cons_buff_0, i64 32) ]
  tail call void @matmul_bf16_bf16(ptr nonnull @memX_cons_buff_0, ptr nonnull @memW_cons_buff_0, ptr nonnull @mm_result_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memW_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memX_cons_buff_1, i64 32) ]
  tail call void @matmul_bf16_bf16(ptr nonnull @memX_cons_buff_1, ptr nonnull @memW_cons_buff_1, ptr nonnull @mm_result_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memW_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memX_cons_buff_0, i64 32) ]
  tail call void @matmul_bf16_bf16(ptr nonnull @memX_cons_buff_0, ptr nonnull @memW_cons_buff_0, ptr nonnull @mm_result_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memW_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memX_cons_buff_1, i64 32) ]
  tail call void @matmul_bf16_bf16(ptr nonnull @memX_cons_buff_1, ptr nonnull @memW_cons_buff_1, ptr nonnull @mm_result_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_buff_0, i64 32) ]
  tail call void @zero_bf16(ptr nonnull @mm_result_buff_0)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memW_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memX_cons_buff_0, i64 32) ]
  tail call void @matmul_bf16_bf16(ptr nonnull @memX_cons_buff_0, ptr nonnull @memW_cons_buff_0, ptr nonnull @mm_result_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memW_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memX_cons_buff_1, i64 32) ]
  tail call void @matmul_bf16_bf16(ptr nonnull @memX_cons_buff_1, ptr nonnull @memW_cons_buff_1, ptr nonnull @mm_result_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memW_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memX_cons_buff_0, i64 32) ]
  tail call void @matmul_bf16_bf16(ptr nonnull @memX_cons_buff_0, ptr nonnull @memW_cons_buff_0, ptr nonnull @mm_result_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memW_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memX_cons_buff_1, i64 32) ]
  tail call void @matmul_bf16_bf16(ptr nonnull @memX_cons_buff_1, ptr nonnull @memW_cons_buff_1, ptr nonnull @mm_result_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %4 = or disjoint i64 %3, 2
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_buff_0, i64 32) ]
  tail call void @zero_bf16(ptr nonnull @mm_result_buff_0)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memW_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memX_cons_buff_0, i64 32) ]
  tail call void @matmul_bf16_bf16(ptr nonnull @memX_cons_buff_0, ptr nonnull @memW_cons_buff_0, ptr nonnull @mm_result_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memW_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memX_cons_buff_1, i64 32) ]
  tail call void @matmul_bf16_bf16(ptr nonnull @memX_cons_buff_1, ptr nonnull @memW_cons_buff_1, ptr nonnull @mm_result_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memW_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memX_cons_buff_0, i64 32) ]
  tail call void @matmul_bf16_bf16(ptr nonnull @memX_cons_buff_0, ptr nonnull @memW_cons_buff_0, ptr nonnull @mm_result_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memW_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memX_cons_buff_1, i64 32) ]
  tail call void @matmul_bf16_bf16(ptr nonnull @memX_cons_buff_1, ptr nonnull @memW_cons_buff_1, ptr nonnull @mm_result_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_buff_0, i64 32) ]
  tail call void @zero_bf16(ptr nonnull @mm_result_buff_0)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memW_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memX_cons_buff_0, i64 32) ]
  tail call void @matmul_bf16_bf16(ptr nonnull @memX_cons_buff_0, ptr nonnull @memW_cons_buff_0, ptr nonnull @mm_result_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memW_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memX_cons_buff_1, i64 32) ]
  tail call void @matmul_bf16_bf16(ptr nonnull @memX_cons_buff_1, ptr nonnull @memW_cons_buff_1, ptr nonnull @mm_result_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memW_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memX_cons_buff_0, i64 32) ]
  tail call void @matmul_bf16_bf16(ptr nonnull @memX_cons_buff_0, ptr nonnull @memW_cons_buff_0, ptr nonnull @mm_result_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memW_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memX_cons_buff_1, i64 32) ]
  tail call void @matmul_bf16_bf16(ptr nonnull @memX_cons_buff_1, ptr nonnull @memW_cons_buff_1, ptr nonnull @mm_result_buff_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 50, i32 1)
  %5 = add nuw nsw i64 %3, 4
  %6 = icmp ult i64 %4, 30
  br i1 %6, label %2, label %7

7:                                                ; preds = %2
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  %8 = add nuw nsw i64 %1, 1
  %9 = icmp ult i64 %1, 4294967294
  br i1 %9, label %.preheader, label %10

10:                                               ; preds = %7
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #1

attributes #0 = { nounwind }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
