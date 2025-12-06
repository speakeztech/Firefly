; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@str0 = private constant [20 x i8] c"What is your name? \00"
@str1 = private constant [8 x i8] c"Hello, \00"
@str2 = private constant [3 x i8] c"!\0A\00"

define i32 @main(i32 %0) {
  %2 = alloca i8, i64 64, align 1
  %3 = call i64 asm sideeffect "syscall", "=r,{rax},{rdi},{rsi},{rdx}"(i64 1, i64 1, ptr @str0, i64 19)
  %4 = call i64 asm sideeffect "syscall", "=r,{rax},{rdi},{rsi},{rdx}"(i64 0, i64 0, ptr %2, i64 64)
  %5 = trunc i64 %4 to i32
  %6 = call i64 asm sideeffect "syscall", "=r,{rax},{rdi},{rsi},{rdx}"(i64 1, i64 1, ptr @str1, i64 7)
  %7 = sub i32 %5, 1
  %8 = sext i32 %7 to i64
  %9 = call i64 asm sideeffect "syscall", "=r,{rax},{rdi},{rsi},{rdx}"(i64 1, i64 1, ptr %2, i64 %8)
  %10 = call i64 asm sideeffect "syscall", "=r,{rax},{rdi},{rsi},{rdx}"(i64 1, i64 1, ptr @str2, i64 2)
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
