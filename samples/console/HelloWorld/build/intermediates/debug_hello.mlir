// Debug HelloWorld
module {
  func.func @main(%arg0: i32) -> i32 {
    // Stack allocation for buffer
    %v0 = llvm.mlir.constant(64 : i64) : i64
    %v1 = llvm.alloca %v0 x i8 : (i64) -> !llvm.ptr

    // Write "Reading: "
    %d1 = llvm.mlir.addressof @dbg1 : !llvm.ptr
    %d1len = arith.constant 9 : i64
    %d1fd = arith.constant 1 : i64
    %d1sys = arith.constant 1 : i64
    %d1r = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %d1sys, %d1fd, %d1, %d1len : (i64, i64, !llvm.ptr, i64) -> i64

    // Read from stdin into buffer
    %v7 = arith.constant 64 : i64
    %v8 = arith.constant 0 : i64
    %v9 = arith.constant 0 : i64
    %v10 = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %v9, %v8, %v1, %v7 : (i64, i64, !llvm.ptr, i64) -> i64

    // Write "Got N bytes: "
    %d2 = llvm.mlir.addressof @dbg2 : !llvm.ptr
    %d2len = arith.constant 13 : i64
    %d2fd = arith.constant 1 : i64
    %d2sys = arith.constant 1 : i64
    %d2r = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %d2sys, %d2fd, %d2, %d2len : (i64, i64, !llvm.ptr, i64) -> i64

    // Write buffer contents directly (with full bytes read)
    %wfd = arith.constant 1 : i64
    %wsys = arith.constant 1 : i64
    %wr = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %wsys, %wfd, %v1, %v10 : (i64, i64, !llvm.ptr, i64) -> i64

    // Write newline
    %nl = llvm.mlir.addressof @newline : !llvm.ptr
    %nllen = arith.constant 1 : i64
    %nlfd = arith.constant 1 : i64
    %nlsys = arith.constant 1 : i64
    %nlr = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %nlsys, %nlfd, %nl, %nllen : (i64, i64, !llvm.ptr, i64) -> i64

    %v23 = arith.constant 0 : i32
    return %v23 : i32
  }

  func.func @_start() attributes {llvm.emit_c_interface} {
    %argc = arith.constant 0 : i32
    %retval = func.call @main(%argc) : (i32) -> i32
    %retval64 = arith.extsi %retval : i32 to i64
    %sys_exit = arith.constant 60 : i64
    %unused = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi}" %sys_exit, %retval64 : (i64, i64) -> i64
    return
  }

  llvm.mlir.global private constant @dbg1("Reading: \00") : !llvm.array<10 x i8>
  llvm.mlir.global private constant @dbg2("Got N bytes: \00") : !llvm.array<14 x i8>
  llvm.mlir.global private constant @newline("\0A\00") : !llvm.array<2 x i8>
}
