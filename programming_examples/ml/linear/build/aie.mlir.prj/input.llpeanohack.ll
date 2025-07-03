; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2p"

@inX_cons_buff_1 = external global [64 x [64 x bfloat]]
@inX_cons_buff_0 = external global [64 x [64 x bfloat]]
@memX_cons_buff_1 = external global [64 x [64 x bfloat]]
@memX_cons_buff_0 = external global [64 x [64 x bfloat]]
@inW_cons_buff_1 = external global [64 x [32 x bfloat]]
@inW_cons_buff_0 = external global [64 x [32 x bfloat]]
@memW_cons_buff_1 = external global [64 x [32 x bfloat]]
@memW_cons_buff_0 = external global [64 x [32 x bfloat]]
@inB_cons_buff_1 = external global [32 x bfloat]
@inB_cons_buff_0 = external global [32 x bfloat]
@memB_cons_buff_1 = external global [32 x bfloat]
@memB_cons_buff_0 = external global [32 x bfloat]
@memY_buff_1 = external global [64 x [32 x bfloat]]
@memY_buff_0 = external global [64 x [32 x bfloat]]
@memY_cons_buff_1 = external global [64 x [32 x bfloat]]
@memY_cons_buff_0 = external global [64 x [32 x bfloat]]
@mm_result_buff_1 = external global [64 x [32 x bfloat]]
@mm_result_buff_0 = external global [64 x [32 x bfloat]]
@mm_result_cons_buff_1 = external global [64 x [32 x bfloat]]
@mm_result_cons_buff_0 = external global [64 x [32 x bfloat]]
@mm_result_mem_cons_buff_1 = external global [64 x [32 x bfloat]]
@mm_result_mem_cons_buff_0 = external global [64 x [32 x bfloat]]
@mm_result_mem_cons = external global [64 x [32 x bfloat]]
@mm_result_mem = external global [64 x [32 x bfloat]]
@mm_result_cons = external global [64 x [32 x bfloat]]
@mm_result = external global [64 x [32 x bfloat]]
@outY_cons = external global [64 x [32 x bfloat]]
@outY = external global [64 x [32 x bfloat]]
@memY_cons = external global [64 x [32 x bfloat]]
@memY = external global [64 x [32 x bfloat]]
@memB_cons = external global [32 x bfloat]
@memB = external global [32 x bfloat]
@inB_cons = external global [32 x bfloat]
@inB = external global [32 x bfloat]
@memW_cons = external global [64 x [32 x bfloat]]
@memW = external global [64 x [32 x bfloat]]
@inW_cons = external global [64 x [32 x bfloat]]
@inW = external global [64 x [32 x bfloat]]
@memX_cons = external global [64 x [64 x bfloat]]
@memX = external global [64 x [64 x bfloat]]
@inX_cons = external global [64 x [64 x bfloat]]
@inX = external global [64 x [64 x bfloat]]

declare void @debug_i32(i32)

declare void @llvm.aie2p.put.ms(i32, i32)

declare { i32, i32 } @llvm.aie2p.get.ss()

declare void @llvm.aie2p.mcd.write.vec(<16 x i32>, i32)

declare <16 x i32> @llvm.aie2p.scd.read.vec(i32)

declare void @llvm.aie2p.acquire(i32, i32)

declare void @llvm.aie2p.release(i32, i32)

declare void @zero_bf16(ptr)

declare void @matmul_bf16_bf16(ptr, ptr, ptr)

declare void @row_wise_bias_add_bf16_bf16(ptr, ptr, ptr)

define void @core_0_3() {
  br label %1

1:                                                ; preds = %9, %0
  %2 = phi i64 [ %10, %9 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %11

4:                                                ; preds = %7, %1
  %5 = phi i64 [ %8, %7 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 32
  br i1 %6, label %7, label %9

7:                                                ; preds = %4
  call void @llvm.aie2p.acquire(i32 53, i32 -1)
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_0, i64 32) ]
  call void @row_wise_bias_add_bf16_bf16(ptr @mm_result_mem_cons_buff_0, ptr @memB_cons_buff_0, ptr @memY_buff_0)
  call void @llvm.aie2p.release(i32 52, i32 1)
  call void @llvm.aie2p.release(i32 48, i32 1)
  call void @llvm.aie2p.release(i32 51, i32 1)
  call void @llvm.aie2p.acquire(i32 53, i32 -1)
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @llvm.aie2p.acquire(i32 50, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_mem_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memY_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memB_cons_buff_1, i64 32) ]
  call void @row_wise_bias_add_bf16_bf16(ptr @mm_result_mem_cons_buff_1, ptr @memB_cons_buff_1, ptr @memY_buff_1)
  call void @llvm.aie2p.release(i32 52, i32 1)
  call void @llvm.aie2p.release(i32 48, i32 1)
  call void @llvm.aie2p.release(i32 51, i32 1)
  %8 = add i64 %5, 2
  br label %4

9:                                                ; preds = %4
  %10 = add i64 %2, 1
  br label %1

11:                                               ; preds = %1
  ret void
}

define void @core_0_2() {
  br label %1

1:                                                ; preds = %21, %0
  %2 = phi i64 [ %22, %21 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967295
  br i1 %3, label %4, label %23

4:                                                ; preds = %19, %1
  %5 = phi i64 [ %20, %19 ], [ 0, %1 ]
  %6 = icmp slt i64 %5, 32
  br i1 %6, label %7, label %21

7:                                                ; preds = %4
  call void @llvm.aie2p.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_buff_0, i64 32) ]
  call void @zero_bf16(ptr @mm_result_buff_0)
  br label %8

8:                                                ; preds = %11, %7
  %9 = phi i64 [ %12, %11 ], [ 0, %7 ]
  %10 = icmp slt i64 %9, 4
  br i1 %10, label %11, label %13

11:                                               ; preds = %8
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memW_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memX_cons_buff_0, i64 32) ]
  call void @matmul_bf16_bf16(ptr @memX_cons_buff_0, ptr @memW_cons_buff_0, ptr @mm_result_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memW_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memX_cons_buff_1, i64 32) ]
  call void @matmul_bf16_bf16(ptr @memX_cons_buff_1, ptr @memW_cons_buff_1, ptr @mm_result_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  %12 = add i64 %9, 2
  br label %8

13:                                               ; preds = %8
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_buff_0, i64 32) ]
  call void @zero_bf16(ptr @mm_result_buff_0)
  br label %14

14:                                               ; preds = %17, %13
  %15 = phi i64 [ %18, %17 ], [ 0, %13 ]
  %16 = icmp slt i64 %15, 4
  br i1 %16, label %17, label %19

17:                                               ; preds = %14
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memW_cons_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memX_cons_buff_0, i64 32) ]
  call void @matmul_bf16_bf16(ptr @memX_cons_buff_0, ptr @memW_cons_buff_0, ptr @mm_result_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  call void @llvm.aie2p.acquire(i32 49, i32 -1)
  call void @llvm.aie2p.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 true) [ "align"(ptr @mm_result_buff_0, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memW_cons_buff_1, i64 32) ]
  call void @llvm.assume(i1 true) [ "align"(ptr @memX_cons_buff_1, i64 32) ]
  call void @matmul_bf16_bf16(ptr @memX_cons_buff_1, ptr @memW_cons_buff_1, ptr @mm_result_buff_0)
  call void @llvm.aie2p.release(i32 48, i32 1)
  call void @llvm.aie2p.release(i32 50, i32 1)
  %18 = add i64 %15, 2
  br label %14

19:                                               ; preds = %14
  %20 = add i64 %5, 2
  br label %4

21:                                               ; preds = %4
  call void @llvm.aie2p.release(i32 53, i32 1)
  %22 = add i64 %2, 1
  br label %1

23:                                               ; preds = %1
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
