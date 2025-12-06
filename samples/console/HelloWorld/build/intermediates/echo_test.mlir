// Simple echo test
module {
  func.func @main(%arg0: i32) -> i32 {
    // Allocate buffer
    %bufsize = llvm.mlir.constant(64 : i64) : i64
    %buf = llvm.alloca %bufsize x i8 : (i64) -> !llvm.ptr

    // Read from stdin
    %size = arith.constant 64 : i64
    %fd_stdin = arith.constant 0 : i64
    %sys_read = arith.constant 0 : i64
    %bytesRead = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %sys_read, %fd_stdin, %buf, %size : (i64, i64, !llvm.ptr, i64) -> i64

    // Write back to stdout (same buffer, same length)
    %fd_stdout = arith.constant 1 : i64
    %sys_write = arith.constant 1 : i64
    %written = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %sys_write, %fd_stdout, %buf, %bytesRead : (i64, i64, !llvm.ptr, i64) -> i64

    %ret = arith.constant 0 : i32
    return %ret : i32
  }

  func.func @_start() attributes {llvm.emit_c_interface} {
    %argc = arith.constant 0 : i32
    %retval = func.call @main(%argc) : (i32) -> i32
    %retval64 = arith.extsi %retval : i32 to i64
    %sys_exit = arith.constant 60 : i64
    %unused = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi}" %sys_exit, %retval64 : (i64, i64) -> i64
    return
  }
}
