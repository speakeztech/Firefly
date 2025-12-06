; Test with proper clobbers
@dbg1 = private constant [10 x i8] c"Reading: \00"
@dbg2 = private constant [14 x i8] c"Got N bytes: \00"
@newline = private constant [2 x i8] c"\0A\00"

define i32 @main(i32 %0) {
  %2 = alloca i8, i64 64, align 1
  ; Write "Reading: " with clobbers
  %3 = call i64 asm sideeffect "syscall", "=r,{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}"(i64 1, i64 1, ptr @dbg1, i64 9)
  ; Read from stdin with clobbers
  %4 = call i64 asm sideeffect "syscall", "=r,{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}"(i64 0, i64 0, ptr %2, i64 64)
  ; Write "Got N bytes: " with clobbers
  %5 = call i64 asm sideeffect "syscall", "=r,{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}"(i64 1, i64 1, ptr @dbg2, i64 13)
  ; Write buffer content with clobbers
  %6 = call i64 asm sideeffect "syscall", "=r,{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}"(i64 1, i64 1, ptr %2, i64 %4)
  ; Write newline with clobbers
  %7 = call i64 asm sideeffect "syscall", "=r,{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}"(i64 1, i64 1, ptr @newline, i64 1)
  ret i32 0
}

define void @_start() {
  %1 = call i32 @main(i32 0)
  %2 = sext i32 %1 to i64
  %3 = call i64 asm sideeffect "syscall", "=r,{rax},{rdi},~{rcx},~{r11}"(i64 60, i64 %2)
  ret void
}
