; Firefly-generated MLIR for HelloWorld
; Target: x86_64-unknown-linux-gnu
; PSG: 865 nodes, 951 edges, 1 entry points

module {
  
  func.func @main(%arg0: i32) -> i32 {
    ; Unhandled node: Const:Int32 0
    %v0 = llvm.mlir.addressof @str0 : !llvm.ptr
    %v1 = arith.constant 2 : i64
    %v2 = arith.constant 1 : i64
    %v3 = arith.constant 1 : i64
    %v4 = llvm.inline_asm "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %v3, %v2, %v0, %v1 : (i64, i64, !llvm.ptr, i64) -> i64
    %v5 = llvm.mlir.addressof @str1 : !llvm.ptr
    %v6 = arith.constant 7 : i64
    %v7 = arith.constant 1 : i64
    %v8 = arith.constant 1 : i64
    %v9 = llvm.inline_asm "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %v8, %v7, %v5, %v6 : (i64, i64, !llvm.ptr, i64) -> i64
    ; Unhandled node: Pattern:Named:bytesRead
    ; Unhandled node: Pattern:Named:buffer
    %v10 = llvm.mlir.addressof @str2 : !llvm.ptr
    %v11 = arith.constant 19 : i64
    %v12 = arith.constant 1 : i64
    %v13 = arith.constant 1 : i64
    %v14 = llvm.inline_asm "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %v13, %v12, %v10, %v11 : (i64, i64, !llvm.ptr, i64) -> i64
    %v15 = arith.constant 0 : i32
    return %v15 : i32
  }
  
  func.func @write(%arg0: !llvm.ptr) -> !opaque<Microsoft_FSharp_Core_Unit> {
    %v16 = arith.constant 0 : i32
    return %v16 : i32
  }
  
  func.func @writeLine(%arg0: !llvm.ptr) -> !opaque<Microsoft_FSharp_Core_Unit> {
    %v17 = arith.constant 0 : i32
    return %v17 : i32
  }
  
  func.func @readLine(%arg0: i32, %arg1: i32) -> i32 {
    %v18 = arith.constant 0 : i32
    return %v18 : i32
  }
  
  ; String constants
  llvm.mlir.global private constant @str0("!\0A\00") : !llvm.array<3 x i8>
  llvm.mlir.global private constant @str1("Hello, \00") : !llvm.array<8 x i8>
  llvm.mlir.global private constant @str2("What is your name?\0A\00") : !llvm.array<20 x i8>
}
