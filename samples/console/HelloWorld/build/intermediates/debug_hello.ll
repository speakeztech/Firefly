; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@dbg1 = private constant [10 x i8] c"Reading: \00"
@dbg2 = private constant [14 x i8] c"Got N bytes: \00"
@newline = private constant [2 x i8] c"\0A\00"

define i32 @main(i32 %0) {
  %2 = alloca i8, i64 64, align 1
  %3 = call i64 asm sideeffect "syscall", "=r,{rax},{rdi},{rsi},{rdx}"(i64 1, i64 1, ptr @dbg1, i64 9)
  %4 = call i64 asm sideeffect "syscall", "=r,{rax},{rdi},{rsi},{rdx}"(i64 0, i64 0, ptr %2, i64 64)
  %5 = call i64 asm sideeffect "syscall", "=r,{rax},{rdi},{rsi},{rdx}"(i64 1, i64 1, ptr @dbg2, i64 13)
  %6 = call i64 asm sideeffect "syscall", "=r,{rax},{rdi},{rsi},{rdx}"(i64 1, i64 1, ptr %2, i64 %4)
  %7 = call i64 asm sideeffect "syscall", "=r,{rax},{rdi},{rsi},{rdx}"(i64 1, i64 1, ptr @newline, i64 1)
  ret i32 0
}

define void @_start() {
  %1 = call i32 @main(i32 0)
  %2 = sext i32 %1 to i64
  %3 = call i64 asm sideeffect "syscall", "=r,{rax},{rdi}"(i64 60, i64 %2)
  ret void
}

define void @_mlir_ciface__start() {
  call void @_start()
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
