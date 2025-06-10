module Core.Conversion.LoweringPipeline

open System
open Core.XParsec.Foundation
open Core.MLIRGeneration.Dialect

/// Lowering transformation state
type LoweringState = {
    CurrentDialect: MLIRDialect
    TargetDialect: MLIRDialect
    TransformedOperations: string list
    SymbolTable: Map<string, string>
    PassName: string
    TransformationHistory: (string * string) list
}

/// Lowering pass definition with transformation logic
type LoweringPass = {
    Name: string
    SourceDialect: MLIRDialect
    TargetDialect: MLIRDialect
    Transform: string -> CompilerResult<string>
    Validate: string -> bool
}

/// MLIR operation parsing and transformation
module MLIROperations =
    
    /// Extracts operation name from an MLIR line
    let extractOperationName (line: string) : string option =
        let trimmed = line.Trim()
        if trimmed.Contains(" = ") then
            let parts = trimmed.Split([|'='|], 2)
            if parts.Length = 2 then
                let opPart = parts.[1].Trim()
                let spaceIndex = opPart.IndexOf(' ')
                if spaceIndex > 0 then
                    Some (opPart.Substring(0, spaceIndex))
                else
                    Some opPart
            else None
        elif trimmed.Contains('.') then
            let spaceIndex = trimmed.IndexOf(' ')
            if spaceIndex > 0 then
                Some (trimmed.Substring(0, spaceIndex))
            else
                Some trimmed
        else None
    
    /// Extracts SSA values from an MLIR operation
    let extractSSAValues (line: string) : string list =
        let parts = line.Split([|' '; ','|], StringSplitOptions.RemoveEmptyEntries)
        parts 
        |> Array.filter (fun part -> part.StartsWith("%"))
        |> Array.toList
    
    /// Extracts function name from func.func operation
    let extractFunctionName (line: string) : string option =
        if line.Contains("func.func") then
            let parts = line.Split([|' '|], StringSplitOptions.RemoveEmptyEntries)
            parts 
            |> Array.tryFind (fun part -> part.StartsWith("@"))
        else None
    
    /// Extracts type information from operation
    let extractTypeInfo (line: string) : string option =
        if line.Contains(":") then
            let colonIndex = line.LastIndexOf(':')
            if colonIndex >= 0 && colonIndex < line.Length - 1 then
                Some (line.Substring(colonIndex + 1).Trim())
            else None
        else None

/// Dialect-specific transformation functions
module DialectTransformers =
    
    /// Transforms func dialect operations to LLVM dialect
    let transformFuncToLLVM (line: string) : CompilerResult<string> =
        let trimmed = line.Trim()
        
        try
            if trimmed.StartsWith("func.func") then
                // Transform func.func to llvm.func
                let transformed = trimmed.Replace("func.func", "llvm.func")
                Success transformed
            
            elif trimmed.Contains("func.return") then
                // Transform func.return to llvm.return
                if trimmed.Contains("func.return ") then
                    let parts = trimmed.Split([|' '|], 2)
                    if parts.Length = 2 then
                        let value = parts.[1]
                        Success (sprintf "llvm.return %s" value)
                    else
                        Success "llvm.return"
                else
                    Success "llvm.return"
            
            elif trimmed.Contains("func.call") then
                // Transform func.call to llvm.call
                let transformed = trimmed.Replace("func.call", "llvm.call")
                Success transformed
            
            else
                Success trimmed
        
        with ex ->
            CompilerFailure [ConversionError("func-to-llvm", trimmed, "llvm operation", ex.Message)]
    
    /// Transforms arith dialect operations to LLVM dialect
    let transformArithToLLVM (line: string) : CompilerResult<string> =
        let trimmed = line.Trim()
        
        try
            if trimmed.Contains("arith.constant") then
                // Transform arith.constant to llvm.mlir.constant
                let transformed = trimmed.Replace("arith.constant", "llvm.mlir.constant")
                Success transformed
            
            elif trimmed.Contains("arith.addi") then
                // Transform arith.addi to llvm.add
                let transformed = trimmed.Replace("arith.addi", "llvm.add")
                Success transformed
            
            elif trimmed.Contains("arith.subi") then
                // Transform arith.subi to llvm.sub
                let transformed = trimmed.Replace("arith.subi", "llvm.sub")
                Success transformed
            
            elif trimmed.Contains("arith.muli") then
                // Transform arith.muli to llvm.mul
                let transformed = trimmed.Replace("arith.muli", "llvm.mul")
                Success transformed
            
            elif trimmed.Contains("arith.divsi") then
                // Transform arith.divsi to llvm.sdiv
                let transformed = trimmed.Replace("arith.divsi", "llvm.sdiv")
                Success transformed
            
            elif trimmed.Contains("arith.cmpi") then
                // Transform arith.cmpi to llvm.icmp
                let transformed = trimmed.Replace("arith.cmpi", "llvm.icmp")
                Success transformed
            
            else
                Success trimmed
        
        with ex ->
            CompilerFailure [ConversionError("arith-to-llvm", trimmed, "llvm operation", ex.Message)]
    
    /// Transforms memref dialect operations to LLVM dialect
    let transformMemrefToLLVM (line: string) : CompilerResult<string> =
        let trimmed = line.Trim()
        
        try
            if trimmed.Contains("memref.alloc") then
                // Transform memref.alloc to llvm.alloca
                let transformed = trimmed.Replace("memref.alloc", "llvm.alloca")
                Success transformed
            
            elif trimmed.Contains("memref.alloca") then
                // Transform memref.alloca to llvm.alloca
                let transformed = trimmed.Replace("memref.alloca", "llvm.alloca")
                Success transformed
            
            elif trimmed.Contains("memref.load") then
                // Transform memref.load to llvm.load
                let transformed = trimmed.Replace("memref.load", "llvm.load")
                Success transformed
            
            elif trimmed.Contains("memref.store") then
                // Transform memref.store to llvm.store
                let transformed = trimmed.Replace("memref.store", "llvm.store")
                Success transformed
            
            elif trimmed.Contains("memref.dealloc") then
                // Transform memref.dealloc (remove for stack-based allocation)
                Success ("  ; removed " + trimmed + " (stack-based allocation)")
            
            else
                Success trimmed
        
        with ex ->
            CompilerFailure [ConversionError("memref-to-llvm", trimmed, "llvm operation", ex.Message)]
    
    /// Transforms standard dialect operations to LLVM dialect
    let transformStdToLLVM (line: string) : CompilerResult<string> =
        let trimmed = line.Trim()
        
        try
            if trimmed.Contains("std.constant") then
                // Transform std.constant to llvm.mlir.constant
                let transformed = trimmed.Replace("std.constant", "llvm.mlir.constant")
                Success transformed
            
            elif trimmed.Contains("std.br") then
                // Transform std.br to llvm.br
                let transformed = trimmed.Replace("std.br", "llvm.br")
                Success transformed
            
            elif trimmed.Contains("std.cond_br") then
                // Transform std.cond_br to llvm.cond_br
                let transformed = trimmed.Replace("std.cond_br", "llvm.cond_br")
                Success transformed
            
            else
                Success trimmed
        
        with ex ->
            CompilerFailure [ConversionError("std-to-llvm", trimmed, "llvm operation", ex.Message)]

/// Pass management and execution
module PassManagement =
    
    /// Creates a lowering pass with validation
    let createLoweringPass (name: string) (source: MLIRDialect) (target: MLIRDialect) 
                          (transformer: string -> CompilerResult<string>) : LoweringPass =
        {
            Name = name
            SourceDialect = source
            TargetDialect = target
            Transform = transformer
            Validate = fun _ -> true  // Simple validation
        }
    
    /// Applies a single lowering pass to MLIR text
    let applyPass (pass: LoweringPass) (mlirText: string) : CompilerResult<string> =
        let lines = mlirText.Split('\n') |> Array.toList
        
        let transformLine (line: string) : CompilerResult<string> =
            let trimmed = line.Trim()
            if String.IsNullOrEmpty(trimmed) || trimmed.StartsWith("//") || 
               trimmed.StartsWith("module") || trimmed.StartsWith("}") || trimmed = "{" then
                Success line
            else
                pass.Transform line
        
        let transformAllLines (lines: string list) : CompilerResult<string list> =
            List.fold (fun acc line ->
                match acc with
                | Success accLines ->
                    match transformLine line with
                    | Success transformedLine -> Success (transformedLine :: accLines)
                    | CompilerFailure errors -> CompilerFailure errors
                | CompilerFailure errors -> CompilerFailure errors
            ) (Success []) lines
            |> function
               | Success transformedLines -> Success (List.rev transformedLines)
               | CompilerFailure errors -> CompilerFailure errors
        
        match transformAllLines lines with
        | Success transformedLines -> Success (String.concat "\n" transformedLines)
        | CompilerFailure errors -> CompilerFailure errors
    
    /// Validates that transformed MLIR is in target dialect
    let validateDialectPurity (targetDialect: MLIRDialect) (mlirText: string) : CompilerResult<unit> =
        let lines = mlirText.Split('\n')
        let dialectPrefix = dialectToString targetDialect + "."
        
        let invalidLines = 
            lines 
            |> Array.mapi (fun i line -> (i + 1, line.Trim()))
            |> Array.filter (fun (_, line) -> 
                not (String.IsNullOrEmpty(line)) && 
                not (line.StartsWith("//")) &&
                not (line.StartsWith("module")) &&
                not (line.StartsWith("func")) &&
                not (line.StartsWith("}")) &&
                not (line = "{") &&
                line.Contains(".") &&
                not (line.Contains(dialectPrefix)) &&
                not (line.Contains("llvm.")))
        
        if invalidLines.Length > 0 then
            let errorDetails = 
                invalidLines 
                |> Array.take (min 5 invalidLines.Length)  // Limit error details
                |> Array.map (fun (lineNum, line) -> sprintf "Line %d: %s" lineNum line)
                |> String.concat "\n"
            CompilerFailure [ConversionError(
                "dialect validation", 
                "mixed dialects", 
                dialectToString targetDialect, 
                sprintf "Found non-%s operations:\n%s" (dialectToString targetDialect) errorDetails)]
        else
            Success ()

/// Standard lowering pipeline creation
let createStandardLoweringPipeline() : LoweringPass list = [
    PassManagement.createLoweringPass "convert-func-to-llvm" Func LLVM DialectTransformers.transformFuncToLLVM
    PassManagement.createLoweringPass "convert-arith-to-llvm" Arith LLVM DialectTransformers.transformArithToLLVM  
    PassManagement.createLoweringPass "convert-memref-to-llvm" MemRef LLVM DialectTransformers.transformMemrefToLLVM
    PassManagement.createLoweringPass "convert-std-to-llvm" Standard LLVM DialectTransformers.transformStdToLLVM
]

/// Applies the complete lowering pipeline
let applyLoweringPipeline (mlirModule: string) : CompilerResult<string> =
    if String.IsNullOrWhiteSpace(mlirModule) then
        CompilerFailure [ConversionError("lowering pipeline", "empty input", "LLVM dialect", "Input MLIR module is empty")]
    else
        let passes = createStandardLoweringPipeline()
        
        let applyPassSequence (currentText: string) (remainingPasses: LoweringPass list) : CompilerResult<string> =
            List.fold (fun acc pass ->
                match acc with
                | Success text ->
                    PassManagement.applyPass pass text
                | CompilerFailure errors -> CompilerFailure errors
            ) (Success currentText) remainingPasses
        
        match applyPassSequence mlirModule passes with
        | Success loweredMLIR ->
            match PassManagement.validateDialectPurity LLVM loweredMLIR with
            | Success () -> Success loweredMLIR
            | CompilerFailure errors -> CompilerFailure errors
        | CompilerFailure errors -> CompilerFailure errors

/// Validates that lowered MLIR contains only LLVM dialect operations
let validateLLVMDialectOnly (llvmDialectModule: string) : CompilerResult<unit> =
    PassManagement.validateDialectPurity LLVM llvmDialectModule

/// Creates a custom lowering pipeline for specific dialect combinations
let createCustomLoweringPipeline (sourceDialects: MLIRDialect list) (targetDialect: MLIRDialect) : CompilerResult<LoweringPass list> =
    let availableTransformers = [
        (Func, LLVM, DialectTransformers.transformFuncToLLVM)
        (Arith, LLVM, DialectTransformers.transformArithToLLVM)
        (MemRef, LLVM, DialectTransformers.transformMemrefToLLVM)
        (Standard, LLVM, DialectTransformers.transformStdToLLVM)
    ]
    
    let createPassForDialect (source: MLIRDialect) : CompilerResult<LoweringPass> =
        match availableTransformers |> List.tryFind (fun (s, t, _) -> s = source && t = targetDialect) with
        | Some (s, t, transformer) ->
            let passName = sprintf "convert-%s-to-%s" (dialectToString s) (dialectToString t)
            Success (PassManagement.createLoweringPass passName s t transformer)
        | None ->
            CompilerFailure [ConversionError(
                "pipeline creation", 
                dialectToString source, 
                dialectToString targetDialect, 
                sprintf "No transformer available for %s -> %s" (dialectToString source) (dialectToString targetDialect))]
    
    sourceDialects
    |> List.map createPassForDialect
    |> List.fold (fun acc result ->
        match acc, result with
        | Success passes, Success pass -> Success (pass :: passes)
        | CompilerFailure errors, Success _ -> CompilerFailure errors
        | Success _, CompilerFailure errors -> CompilerFailure errors
        | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
    ) (Success [])
    |> function
       | Success passes -> Success (List.rev passes)
       | CompilerFailure errors -> CompilerFailure errors

/// Analyzes MLIR for dialect usage statistics
let analyzeDialectUsage (mlirText: string) : CompilerResult<Map<string, int>> =
    try
        let lines = mlirText.Split('\n')
        let dialectCounts = 
            lines
            |> Array.map (fun line -> line.Trim())
            |> Array.filter (fun line -> 
                not (String.IsNullOrEmpty(line)) && 
                not (line.StartsWith("//")) &&
                line.Contains("."))
            |> Array.choose MLIROperations.extractOperationName
            |> Array.map (fun opName ->
                if opName.Contains(".") then
                    opName.Substring(0, opName.IndexOf("."))
                else "unknown")
            |> Array.groupBy id
            |> Array.map (fun (dialect, ops) -> (dialect, ops.Length))
            |> Map.ofArray
        
        Success dialectCounts
    
    with ex ->
        CompilerFailure [ConversionError("dialect analysis", "MLIR text", "dialect statistics", ex.Message)]

/// Validates that an MLIR module is ready for lowering
let validateMLIRForLowering (mlirText: string) : CompilerResult<unit> =
    try
        let lines = mlirText.Split('\n')
        let hasValidStructure = 
            lines |> Array.exists (fun line -> line.Trim().StartsWith("module"))
        
        if not hasValidStructure then
            CompilerFailure [ConversionError("mlir validation", "input text", "valid MLIR module", "Missing module declaration")]
        else
            let hasOperations = 
                lines |> Array.exists (fun line -> 
                    let trimmed = line.Trim()
                    trimmed.Contains(".") && not (trimmed.StartsWith("//")))
            
            if not hasOperations then
                CompilerFailure [ConversionError("mlir validation", "input text", "valid MLIR module", "No operations found")]
            else
                Success ()
    
    with ex ->
        CompilerFailure [ConversionError("mlir validation", "input text", "valid MLIR module", ex.Message)]