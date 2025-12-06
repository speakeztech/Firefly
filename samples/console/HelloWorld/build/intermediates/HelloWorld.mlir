// Firefly-generated MLIR for HelloWorld (via Alex)
// Target: x86_64-unknown-linux-gnu
// PSG: 1723 nodes, 1933 edges, 1 entry points

module {
  
  func.func @main(%arg0: i32) -> i32 {
    %v0 = llvm.mlir.addressof @str0 : !llvm.ptr
    %v1 = arith.constant 19 : i64
    %v2 = arith.constant 1 : i64
    %v3 = arith.constant 1 : i64
    %v4 = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %v3, %v2, %v0, %v1 : (i64, i64, !llvm.ptr, i64) -> i64
    %v5 = llvm.mlir.addressof @str1 : !llvm.ptr
    %v6 = arith.constant 7 : i64
    %v7 = arith.constant 1 : i64
    %v8 = arith.constant 1 : i64
    %v9 = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %v8, %v7, %v5, %v6 : (i64, i64, !llvm.ptr, i64) -> i64
    %v10 = llvm.mlir.addressof @str2 : !llvm.ptr
    %v11 = arith.constant 2 : i64
    %v12 = arith.constant 1 : i64
    %v13 = arith.constant 1 : i64
    %v14 = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %v13, %v12, %v10, %v11 : (i64, i64, !llvm.ptr, i64) -> i64
    %v15 = arith.constant 0 : i32
    %v16 = arith.extsi %v15 : i32 to i64
    %v17 = arith.constant 60 : i64
    %v18 = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi}" %v17, %v16 : (i64, i64) -> i64
    func.return %v15 : i32
  }
  llvm.mlir.global private constant @str2("!\0A") : !llvm.array<2 x i8>
  llvm.mlir.global private constant @str1("Hello, ") : !llvm.array<7 x i8>
  llvm.mlir.global private constant @str0("What is your name? ") : !llvm.array<19 x i8>
}
