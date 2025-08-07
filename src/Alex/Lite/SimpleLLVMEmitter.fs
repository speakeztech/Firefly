module Alex.Lite.SimpleLLVMEmitter

open System
open System.Text
open Core.PSG.Types
open Alex.Lite.TimeEmitter

/// Simple type mapping from F# to LLVM types
let mapTypeToLLVM (typeName: string option) =
    match typeName with
    | Some "System.Int32" | Some "int" -> "i32"
    | Some "System.Int64" | Some "int64" -> "i64"
    | Some "System.Boolean" | Some "bool" -> "i1"
    | Some "System.Byte" | Some "byte" -> "i8"
    | Some "System.String" | Some "string" -> "!llvm.ptr<i8>"
    | Some "System.Void" | Some "unit" -> "!llvm.void"
    | Some t when t.Contains("[]") -> "!llvm.ptr<i8>"  // Arrays as pointers
    | Some t when t.Contains("Span") -> "!llvm.ptr<i8>"  // Spans as pointers
    | _ -> "i32"  // Default fallback

/// Generate LLVM function signature from PSG node
let generateFunctionSignature (node: PSGNode) =
    match node.Symbol with
    | Some symbol when symbol.IsFunction ->
        let funcName = symbol.Name
        let returnType = mapTypeToLLVM node.TypeName
        
        // For now, assume simple signatures based on what we see in HelloWorld
        match funcName with
        | "main" -> 
            sprintf "llvm.func @main(%%argc: i32, %%argv: !llvm.ptr<!llvm.ptr<i8>>) -> i32"
        | "hello" ->
            sprintf "llvm.func @hello() -> !llvm.void"
        | _ ->
            sprintf "llvm.func @%s() -> %s" funcName returnType
    | _ -> ""

/// Generate LLVM operations for expression nodes
let rec generateExpression (psg: ProgramSemanticGraph) (node: PSGNode) (indent: int) =
    let ind = String.replicate indent "  "
    let sb = StringBuilder()
    
    match node.SyntaxKind with
    | "Const" when node.ConstantValue.IsSome ->
        let constVal = node.ConstantValue.Value
        let llvmType = mapTypeToLLVM node.TypeName
        sprintf "%s%%const_%d = llvm.mlir.constant(%s : %s) : %s" ind node.Id constVal llvmType llvmType
        
    | "MethodCall" | "FunctionCall" ->
        // Extract the function being called
        let calledFunc = 
            match node.Symbol with
            | Some sym -> sym.Name
            | None -> "unknown_func"
        
        // Check if this is a Time library call
        if node.Symbol.IsSome && node.Symbol.Value.FullName.Contains("Time.") then
            // Use specialized Time emitter
            let platform = detectPlatform psg
            mapTimeCall node platform
        else
            // Map common BCL/Alloy functions to runtime calls
            let mappedFunc = 
                match calledFunc with
                | "WriteLine" -> "console_writeline"
                | "Write" -> "console_write"
                | "ReadLine" -> "console_readline"
                | "sprintf" -> "sprintf_impl"
                | "stackBuffer" -> "stack_alloc"
                | _ -> calledFunc
            
            // Generate the call
            sprintf "%s%%result_%d = llvm.call @%s() : () -> !llvm.void" ind node.Id mappedFunc
        
    | "IfThenElse" ->
        // Simple conditional structure
        sb.AppendLine(sprintf "%sllvm.cond_br %%cond_%d, ^then_%d, ^else_%d" ind node.Id node.Id node.Id)
        sb.AppendLine(sprintf "%s^then_%d:" ind node.Id)
        sb.AppendLine(sprintf "%s  // Then branch code here" ind)
        sb.AppendLine(sprintf "%s  llvm.br ^merge_%d" ind node.Id)
        sb.AppendLine(sprintf "%s^else_%d:" ind node.Id)
        sb.AppendLine(sprintf "%s  // Else branch code here" ind)
        sb.AppendLine(sprintf "%s  llvm.br ^merge_%d" ind node.Id)
        sb.AppendLine(sprintf "%s^merge_%d:" ind node.Id)
        sb.ToString()
        
    | "Sequential" ->
        // Process sequential statements
        match node.Children with
        | Parent children ->
            children
            |> List.map (fun childId -> 
                match Map.tryFind childId.Value psg.Nodes with
                | Some childNode -> generateExpression psg childNode indent
                | None -> "")
            |> String.concat "\n"
        | _ -> ""
        
    | "LetOrUse" when node.SyntaxKind.Contains("Use") ->
        // Stack allocation for 'use' bindings
        sprintf "%s%%stack_%d = llvm.alloca %%c256 x i8 : (i32) -> !llvm.ptr<i8>" ind node.Id
        
    | "Binding" ->
        // Variable binding - in LLVM we need alloca + store
        sprintf "%s%%var_%d = llvm.alloca i32 : (i32) -> !llvm.ptr<i32>" ind node.Id
        
    | _ -> 
        sprintf "%s// Unhandled node kind: %s" ind node.SyntaxKind

/// Generate function body from PSG nodes
let generateFunctionBody (psg: ProgramSemanticGraph) (funcNode: PSGNode) =
    let sb = StringBuilder()
    
    // Find the function's body nodes
    match funcNode.Children with
    | Parent children ->
        for childId in children do
            match Map.tryFind childId.Value psg.Nodes with
            | Some childNode when childNode.IsReachable ->
                sb.AppendLine(generateExpression psg childNode 1)
            | _ -> ()
    | _ -> ()
    
    // Add return statement
    match funcNode.Symbol with
    | Some sym when sym.Name = "main" ->
        sb.AppendLine("    %ret = llvm.mlir.constant(0 : i32) : i32")
        sb.AppendLine("    llvm.return %ret : i32")
    | _ ->
        sb.AppendLine("    llvm.return")
    
    sb.ToString()

/// Generate runtime function declarations
let generateRuntimeDeclarations() = """
  // Runtime function declarations
  llvm.func @console_writeline(!llvm.ptr<i8>) -> !llvm.void
  llvm.func @console_write(!llvm.ptr<i8>) -> !llvm.void  
  llvm.func @console_readline() -> !llvm.ptr<i8>
  llvm.func @sprintf_impl(!llvm.ptr<i8>, ...) -> !llvm.ptr<i8>
  llvm.func @stack_alloc(i32) -> !llvm.ptr<i8>
  llvm.func @memcpy(!llvm.ptr<i8>, !llvm.ptr<i8>, i32) -> !llvm.void
  
  // Time library runtime functions (platform-specific)
  llvm.func @QueryPerformanceCounter(!llvm.ptr<i64>) -> i32      // Windows
  llvm.func @QueryPerformanceFrequency(!llvm.ptr<i64>) -> i32    // Windows
  llvm.func @GetSystemTimeAsFileTime(!llvm.ptr<i64>) -> !llvm.void  // Windows
  llvm.func @Sleep(i32) -> !llvm.void                             // Windows
  
  // Or for Unix/Linux:
  llvm.func @clock_gettime(i32, !llvm.ptr<i8>) -> i32            // POSIX
  llvm.func @nanosleep(!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32      // POSIX
"""

/// Main entry point - generate LLVM dialect MLIR from PSG
let emitLLVMDialect (psg: ProgramSemanticGraph) (outputPath: string) =
    let sb = StringBuilder()
    
    // MLIR module header
    sb.AppendLine("module {")
    
    // Add runtime declarations
    sb.AppendLine(generateRuntimeDeclarations())
    
    // Find and generate all reachable functions
    let functions = 
        psg.Nodes
        |> Map.filter (fun _ node -> 
            node.IsReachable && 
            node.SyntaxKind = "Binding" &&
            node.Symbol.IsSome &&
            node.Symbol.Value.IsFunction)
    
    // Generate entry point first
    match psg.EntryPoints |> Set.toList with
    | mainSymbol :: _ ->
        // Find the main function node
        let mainNode = 
            functions 
            |> Map.tryPick (fun _ node -> 
                if node.Symbol.IsSome && node.Symbol.Value.Name = "main" then Some node
                else None)
        
        match mainNode with
        | Some node ->
            sb.AppendLine("  // Entry point")
            sb.AppendLine("  llvm.func @main(%argc: i32, %argv: !llvm.ptr<!llvm.ptr<i8>>) -> i32 {")
            sb.AppendLine(generateFunctionBody psg node)
            sb.AppendLine("  }")
            sb.AppendLine()
        | None ->
            // Generate default main
            sb.AppendLine("  // Default entry point")
            sb.AppendLine("  llvm.func @main(%argc: i32, %argv: !llvm.ptr<!llvm.ptr<i8>>) -> i32 {")
            sb.AppendLine("    %ret = llvm.mlir.constant(0 : i32) : i32")
            sb.AppendLine("    llvm.return %ret : i32")
            sb.AppendLine("  }")
            sb.AppendLine()
    | [] -> ()
    
    // Generate other functions
    for KeyValue(_, node) in functions do
        if node.Symbol.IsSome && node.Symbol.Value.Name <> "main" then
            let funcName = node.Symbol.Value.Name
            let returnType = mapTypeToLLVM node.TypeName
            
            sb.AppendLine(sprintf "  // Function: %s" funcName)
            sb.AppendLine(sprintf "  llvm.func @%s() -> %s {" funcName returnType)
            sb.AppendLine(generateFunctionBody psg node)
            sb.AppendLine("  }")
            sb.AppendLine()
    
    // Close module
    sb.AppendLine("}")
    
    // Write to file
    System.IO.File.WriteAllText(outputPath, sb.ToString())
    
    printfn "[SimpleLLVMEmitter] Generated LLVM dialect MLIR:"
    printfn "  Output: %s" outputPath
    printfn "  Functions: %d" (Map.count functions)
    printfn "  Total size: %d bytes" (sb.Length)

/// Helper to generate a minimal working example for testing
let generateMinimalExample (outputPath: string) =
    let example = """
module {
  // Minimal HelloWorld in LLVM dialect
  
  // External functions (would link to libc or custom runtime)
  llvm.func @puts(!llvm.ptr<i8>) -> i32
  
  // String constant
  llvm.mlir.global internal constant @.str("Hello, World!\00") : !llvm.array<14 x i8>
  
  // Main function
  llvm.func @main(%argc: i32, %argv: !llvm.ptr<!llvm.ptr<i8>>) -> i32 {
    // Get pointer to string
    %0 = llvm.mlir.addressof @.str : !llvm.ptr<!llvm.array<14 x i8>>
    %1 = llvm.getelementptr %0[%c0, %c0] : (!llvm.ptr<!llvm.array<14 x i8>>, i32, i32) -> !llvm.ptr<i8>
    
    // Call puts
    %2 = llvm.call @puts(%1) : (!llvm.ptr<i8>) -> i32
    
    // Return 0
    %c0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %c0 : i32
  }
}
"""
    System.IO.File.WriteAllText(outputPath, example)
    printfn "[SimpleLLVMEmitter] Generated minimal example at: %s" outputPath