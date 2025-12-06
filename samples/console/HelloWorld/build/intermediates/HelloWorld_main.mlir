// Minimal HelloWorld MLIR - just main and _start
module {

  func.func @main(%arg0: i32) -> i32 {
    // Stack allocation for buffer
    %v0 = llvm.mlir.constant(64 : i64) : i64
    %v1 = llvm.alloca %v0 x i8 : (i64) -> !llvm.ptr

    // Prompt: write "What is your name? "
    %v2 = llvm.mlir.addressof @str0 : !llvm.ptr
    %v3 = arith.constant 19 : i64
    %v4 = arith.constant 1 : i64
    %v5 = arith.constant 1 : i64
    %v6 = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %v5, %v4, %v2, %v3 : (i64, i64, !llvm.ptr, i64) -> i64

    // Read from stdin into buffer
    %v7 = arith.constant 64 : i64
    %v8 = arith.constant 0 : i64
    %v9 = arith.constant 0 : i64
    %v10 = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %v9, %v8, %v1, %v7 : (i64, i64, !llvm.ptr, i64) -> i64
    %v11 = arith.trunci %v10 : i64 to i32

    // Write "Hello, "
    %v12 = llvm.mlir.addressof @str1 : !llvm.ptr
    %v13 = arith.constant 7 : i64
    %v14 = arith.constant 1 : i64
    %v15 = arith.constant 1 : i64
    %v16 = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %v15, %v14, %v12, %v13 : (i64, i64, !llvm.ptr, i64) -> i64

    // Write buffer contents (user's input)
    // Need to subtract 1 from bytesRead to remove newline
    %one = arith.constant 1 : i32
    %len = arith.subi %v11, %one : i32
    %len64 = arith.extsi %len : i32 to i64
    %fd = arith.constant 1 : i64
    %syswrite = arith.constant 1 : i64
    %v17 = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %syswrite, %fd, %v1, %len64 : (i64, i64, !llvm.ptr, i64) -> i64

    // Write "!\n"
    %v18 = llvm.mlir.addressof @str2 : !llvm.ptr
    %v19 = arith.constant 2 : i64
    %v20 = arith.constant 1 : i64
    %v21 = arith.constant 1 : i64
    %v22 = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %v21, %v20, %v18, %v19 : (i64, i64, !llvm.ptr, i64) -> i64

    %v23 = arith.constant 0 : i32
    return %v23 : i32
  }

  // Entry point wrapper for freestanding binary
  func.func @_start() attributes {llvm.emit_c_interface} {
    %argc = arith.constant 0 : i32
    %retval = func.call @main(%argc) : (i32) -> i32
    %retval64 = arith.extsi %retval : i32 to i64
    %sys_exit = arith.constant 60 : i64
    %unused = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi}" %sys_exit, %retval64 : (i64, i64) -> i64
    return
  }

  // String constants
  llvm.mlir.global private constant @str0("What is your name? \00") : !llvm.array<20 x i8>
  llvm.mlir.global private constant @str1("Hello, \00") : !llvm.array<8 x i8>
  llvm.mlir.global private constant @str2("!\0A\00") : !llvm.array<3 x i8>
}
