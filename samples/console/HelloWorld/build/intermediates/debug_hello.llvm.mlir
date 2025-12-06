module {
  llvm.func @main(%arg0: i32) -> i32 {
    %0 = llvm.mlir.constant(64 : i64) : i64
    %1 = llvm.alloca %0 x i8 : (i64) -> !llvm.ptr
    %2 = llvm.mlir.addressof @dbg1 : !llvm.ptr
    %3 = llvm.mlir.constant(9 : i64) : i64
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.mlir.constant(1 : i64) : i64
    %6 = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %5, %4, %2, %3 : (i64, i64, !llvm.ptr, i64) -> i64
    %7 = llvm.mlir.constant(64 : i64) : i64
    %8 = llvm.mlir.constant(0 : i64) : i64
    %9 = llvm.mlir.constant(0 : i64) : i64
    %10 = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %9, %8, %1, %7 : (i64, i64, !llvm.ptr, i64) -> i64
    %11 = llvm.mlir.addressof @dbg2 : !llvm.ptr
    %12 = llvm.mlir.constant(13 : i64) : i64
    %13 = llvm.mlir.constant(1 : i64) : i64
    %14 = llvm.mlir.constant(1 : i64) : i64
    %15 = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %14, %13, %11, %12 : (i64, i64, !llvm.ptr, i64) -> i64
    %16 = llvm.mlir.constant(1 : i64) : i64
    %17 = llvm.mlir.constant(1 : i64) : i64
    %18 = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %17, %16, %1, %10 : (i64, i64, !llvm.ptr, i64) -> i64
    %19 = llvm.mlir.addressof @newline : !llvm.ptr
    %20 = llvm.mlir.constant(1 : i64) : i64
    %21 = llvm.mlir.constant(1 : i64) : i64
    %22 = llvm.mlir.constant(1 : i64) : i64
    %23 = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %22, %21, %19, %20 : (i64, i64, !llvm.ptr, i64) -> i64
    %24 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %24 : i32
  }
  llvm.func @_start() attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.call @main(%0) : (i32) -> i32
    %2 = llvm.sext %1 : i32 to i64
    %3 = llvm.mlir.constant(60 : i64) : i64
    %4 = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi}" %3, %2 : (i64, i64) -> i64
    llvm.return
  }
  llvm.func @_mlir_ciface__start() attributes {llvm.emit_c_interface} {
    llvm.call @_start() : () -> ()
    llvm.return
  }
  llvm.mlir.global private constant @dbg1("Reading: \00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @dbg2("Got N bytes: \00") {addr_space = 0 : i32}
  llvm.mlir.global private constant @newline("\0A\00") {addr_space = 0 : i32}
}

