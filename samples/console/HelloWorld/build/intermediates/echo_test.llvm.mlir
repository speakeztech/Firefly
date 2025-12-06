module {
  llvm.func @main(%arg0: i32) -> i32 {
    %0 = llvm.mlir.constant(64 : i64) : i64
    %1 = llvm.alloca %0 x i8 : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(64 : i64) : i64
    %3 = llvm.mlir.constant(0 : i64) : i64
    %4 = llvm.mlir.constant(0 : i64) : i64
    %5 = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %4, %3, %1, %2 : (i64, i64, !llvm.ptr, i64) -> i64
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.mlir.constant(1 : i64) : i64
    %8 = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %7, %6, %1, %5 : (i64, i64, !llvm.ptr, i64) -> i64
    %9 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %9 : i32
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
}

