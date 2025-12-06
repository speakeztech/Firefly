// Firefly-generated MLIR for HelloWorld
// Target: x86_64-unknown-linux-gnu
// Output: freestanding
// PSG: 1540 nodes, 1712 edges, 1 entry points

module {
  
  func.func @main(%arg0: i32) -> i32 {
    %v0 = llvm.mlir.constant(64 : i64) : i64
    %v1 = llvm.alloca %v0 x i8 : (i64) -> !llvm.ptr
    // Registering local: buffer = %v1
    %v2 = llvm.mlir.addressof @str0 : !llvm.ptr
    %v3 = arith.constant 19 : i64
    %v4 = arith.constant 1 : i64
    %v5 = arith.constant 1 : i64
    %v6 = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %v5, %v4, %v2, %v3 : (i64, i64, !llvm.ptr, i64) -> i64
    %v7 = arith.constant 64 : i64
    %v8 = arith.constant 0 : i64
    %v9 = arith.constant 0 : i64
    %v10 = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %v9, %v8, %v1, %v7 : (i64, i64, !llvm.ptr, i64) -> i64
    %v11 = arith.trunci %v10 : i64 to i32
    // Unhandled node: Pattern:UnionCase:Ok
    // Unhandled node: Const:String ("User", Regular, (23,23--23,29))
    // Unhandled node: Pattern:UnionCase:Error
    // Unhandled node: Const:String ("Unknown", Regular, (24,21--24,30))
    // Warning: no SSA result for local: name
    %v12 = llvm.mlir.addressof @str1 : !llvm.ptr
    %v13 = arith.constant 7 : i64
    %v14 = arith.constant 1 : i64
    %v15 = arith.constant 1 : i64
    %v16 = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %v15, %v14, %v12, %v13 : (i64, i64, !llvm.ptr, i64) -> i64
    // Console.Write with non-literal arg
    %v17 = llvm.mlir.addressof @str2 : !llvm.ptr
    %v18 = arith.constant 2 : i64
    %v19 = arith.constant 1 : i64
    %v20 = arith.constant 1 : i64
    %v21 = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %v20, %v19, %v17, %v18 : (i64, i64, !llvm.ptr, i64) -> i64
    // Unhandled node: Const:Int32 0
    %v22 = arith.constant 0 : i32
    return %v22 : i32
  }
  
  func.func @readInto(%arg0: i32) -> i32 {
    %v23 = arith.constant 0 : i32
    return %v23 : i32
  }
  
  func.func @formatInt(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
    %v24 = arith.constant 0 : i32
    return %v24 : i32
  }
  
  func.func @writeBytes(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
    %v25 = arith.constant 0 : i32
    return %v25 : i32
  }
  
  func.func @Prompt(%arg0: !llvm.ptr) -> i32 {
    // Console.Write with non-literal arg
    %v26 = arith.constant 0 : i32
    return %v26 : i32
  }
  
  func.func @writeNewLine(%arg0: i32) -> i32 {
    %v27 = arith.constant 0 : i32
    return %v27 : i32
  }
  
  func.func @Write(%arg0: !llvm.ptr) -> i32 {
    %v28 = llvm.mlir.constant(64 : i64) : i64
    %v29 = llvm.alloca %v28 x i8 : (i64) -> !llvm.ptr
    // Registering local: buffer = %v29
    %v30 = func.call @stringToBytes() : () -> i32
    // Outer App children (2): [Ident:stringToBytes, Ident:message]
    // firstChildResult = Some "%v30"
    %v31 = func.call @stringToBytes() : () -> i32
    // Outer App children (2): [App:FunctionCall, Ident:buffer]
    // firstChildResult = Some "%v31"
    %v32 = func.call @stringToBytes() : () -> i32
    // Outer App children (2): [App:FunctionCall, Const:Int32 4096]
    // firstChildResult = Some "%v32"
    %v33 = func.call @stringToBytes() : () -> i32
    // Registering local: len = %v33
    // writeBytes: Firefly primitive syscall
    %v34 = arith.constant 0 : i32
    return %v34 : i32
  }
  
  func.func @WriteLine(%arg0: !llvm.ptr) -> i32 {
    // Console.Write with non-literal arg
    // Unknown Console operation: writeNewLine
    %v35 = arith.constant 0 : i32
    return %v35 : i32
  }
  
  func.func @stackBuffer(%arg0: i32) -> i32 {
    %v36 = arith.constant 0 : i32
    return %v36 : i32
  }
  
  func.func @replace(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> !llvm.ptr {
    %v37 = arith.constant 0 : i32
    return %v37 : i32
  }
  
  func.func @stringToBytes(%arg0: !llvm.ptr, %arg1: i32, %arg2: i32) -> i32 {
    %v38 = arith.constant 0 : i32
    return %v38 : i32
  }
  
  func.func @bytesToString(%arg0: i32, %arg1: i32) -> !llvm.ptr {
    %v39 = arith.constant 0 : i32
    return %v39 : i32
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
