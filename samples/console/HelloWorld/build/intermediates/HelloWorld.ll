; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@str2 = private constant [2 x i8] c"!\0A"
@str1 = private constant [7 x i8] c"Hello, "
@str0 = private constant [19 x i8] c"What is your name? "

define i32 @main(i32 %0) {
  %2 = call i64 asm sideeffect "syscall", "=r,{rax},{rdi},{rsi},{rdx}"(i64 1, i64 1, ptr @str0, i64 19)
  %3 = call i64 asm sideeffect "syscall", "=r,{rax},{rdi},{rsi},{rdx}"(i64 1, i64 1, ptr @str1, i64 7)
  %4 = call i64 asm sideeffect "syscall", "=r,{rax},{rdi},{rsi},{rdx}"(i64 1, i64 1, ptr @str2, i64 2)
  %5 = call i64 asm sideeffect "syscall", "=r,{rax},{rdi}"(i64 60, i64 0)
  ret i32 0
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
