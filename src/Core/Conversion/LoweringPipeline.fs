module Core.Conversion.LoweringPipeline

open System
open System.Text.RegularExpressions
open Core.MLIRGeneration.Dialect

/// Represents a lowering pass in the MLIR pipeline
type LoweringPass = {
    Name: string
    SourceDialect: MLIRDialect
    TargetDialect: MLIRDialect
    PassOptions: Map<string, string>
}

/// Creates the standard lowering pipeline from high-level to LLVM dialect
let createStandardLoweringPipeline() : LoweringPass list = [
    { Name = "convert-std-to-llvm"; SourceDialect = Standard; TargetDialect = LLVM; PassOptions = Map.empty }
    { Name = "convert-arith-to-llvm"; SourceDialect = Arith; TargetDialect = LLVM; PassOptions = Map.empty }
    { Name = "convert-func-to-llvm"; SourceDialect = Func; TargetDialect = LLVM; PassOptions = Map.empty }
    { Name = "reconcile-unrealized-casts"; SourceDialect = Standard; TargetDialect = Standard; PassOptions = Map.empty }
]

/// Converts function dialect operations to LLVM dialect
let private convertFuncToLLVM (mlirText: string) : string =
    mlirText
        .Replace("func.func", "llvm.func")
        .Replace("func.return", "llvm.return")
        .Replace("func.call", "llvm.call")

/// Converts arithmetic dialect operations to LLVM dialect  
let private convertArithToLLVM (mlirText: string) : string =
    let patterns = [
        (@"arith\.addi\s+(%\w+),\s*(%\w+)\s*:\s*(\w+)", "llvm.add $1, $2 : $3")
        (@"arith\.subi\s+(%\w+),\s*(%\w+)\s*:\s*(\w+)", "llvm.sub $1, $2 : $3")
        (@"arith\.muli\s+(%\w+),\s*(%\w+)\s*:\s*(\w+)", "llvm.mul $1, $2 : $3")
        (@"arith\.divsi\s+(%\w+),\s*(%\w+)\s*:\s*(\w+)", "llvm.sdiv $1, $2 : $3")
        (@"arith\.constant\s+(\d+)\s*:\s*(\w+)", "llvm.mlir.constant($1) : $2")
    ]
    
    patterns
    |> List.fold (fun text (pattern, replacement) ->
        Regex.Replace(text, pattern, replacement)
    ) mlirText

/// Converts standard dialect operations to LLVM dialect
let private convertStdToLLVM (mlirText: string) : string =
    mlirText
        .Replace("std.alloc", "llvm.alloca")
        .Replace("std.load", "llvm.load")
        .Replace("std.store", "llvm.store")

/// Reconciles unrealized cast operations
let private reconcileUnrealizedCasts (mlirText: string) : string =
    let castPattern = @"builtin\.unrealized_conversion_cast\s+(%\w+)\s*:\s*([^}]+)\s+to\s+([^}]+)"
    Regex.Replace(mlirText, castPattern, "$1")

/// Applies a single lowering pass to MLIR text
let private applyLoweringPass (pass: LoweringPass) (mlirText: string) : string =
    match pass.Name with
    | "convert-func-to-llvm" -> convertFuncToLLVM mlirText
    | "convert-arith-to-llvm" -> convertArithToLLVM mlirText
    | "convert-std-to-llvm" -> convertStdToLLVM mlirText
    | "reconcile-unrealized-casts" -> reconcileUnrealizedCasts mlirText
    | _ -> mlirText

/// Applies the complete lowering pipeline to MLIR text
let applyLoweringPipeline (mlirModule: string) : string =
    let passes = createStandardLoweringPipeline()
    
    passes
    |> List.fold (fun currentText pass ->
        applyLoweringPass pass currentText
    ) mlirModule

/// Validates that the lowered MLIR is in pure LLVM dialect
let validateLLVMDialect (llvmDialectModule: string) : bool =
    let invalidPatterns = [
        @"func\." 
        @"arith\."
        @"std\."
        @"builtin\.unrealized_conversion_cast"
    ]
    
    invalidPatterns
    |> List.forall (fun pattern ->
        not (Regex.IsMatch(llvmDialectModule, pattern))
    )