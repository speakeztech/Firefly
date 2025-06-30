module Dabbit.Pipeline.LoweringPipeline

open System
open System.IO
open System.Diagnostics
open Core.XParsec.Foundation

/// Represents an MLIR optimization pass with target dialect
type MLIRPass = {
    Name: string
    PassOptions: string
}

/// Standard passes for MLIR optimization
let standardPasses = [
    { Name = "func-to-llvm"; PassOptions = "--convert-func-to-llvm" }    
    { Name = "arith-to-llvm"; PassOptions = "--convert-arith-to-llvm" }  
    { Name = "cf-to-llvm"; PassOptions = "--convert-cf-to-llvm" }          
    { Name = "scf-to-cf"; PassOptions = "--convert-scf-to-cf" }            
    { Name = "finalizing"; PassOptions = "--reconcile-unrealized-casts" }  
]

/// Runs an external command and returns the result
let private runExternalCommand (program: string) (arguments: string) : CompilerResult<string> =
    try
        let psi = ProcessStartInfo(
            FileName = program,
            Arguments = arguments,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true)
            
        use processObj = Process.Start(psi)
        let output = processObj.StandardOutput.ReadToEnd()
        let error = processObj.StandardError.ReadToEnd()
        processObj.WaitForExit()
        
        if processObj.ExitCode = 0 then
            Success output
        else
            CompilerFailure [
                ConversionError(
                    "external-tool", 
                    program,
                    "process output", 
                    sprintf "Process exited with code %d: %s" processObj.ExitCode error)
            ]
    with ex ->
        CompilerFailure [
            ConversionError(
                "external-tool", 
                program,
                "process execution", 
                sprintf "Failed to execute process: %s" ex.Message)
        ]

/// Applies MLIR optimization using external mlir-opt tool
let applyMLIROptimization (mlirText: string) (intermediateDir: string option) : CompilerResult<string> =
    if String.IsNullOrWhiteSpace(mlirText) then
        CompilerFailure [ConversionError("mlir-optimization", "empty input", "optimized MLIR", "Input MLIR module is empty")]
    else
        // Create temporary files
        let tempDir = 
            match intermediateDir with 
            | Some dir -> dir 
            | None -> Path.GetTempPath()
            
        let inputPath = Path.Combine(tempDir, "input.mlir")
        let outputPath = Path.Combine(tempDir, "optimized.mlir")
        
        try
            // Write input MLIR to file
            File.WriteAllText(inputPath, mlirText)
            
            // Build optimization pass arguments
            let passArgs = standardPasses |> List.map (fun p -> p.PassOptions) |> String.concat " "
            let mlirOptArgs = sprintf "%s %s -o %s" passArgs inputPath outputPath
            
            // Run mlir-opt
            match runExternalCommand "mlir-opt" mlirOptArgs with
            | Success _ ->
                // Read optimized MLIR
                if File.Exists(outputPath) then
                    let optimizedMLIR = File.ReadAllText(outputPath)
                    Success optimizedMLIR
                else
                    CompilerFailure [ConversionError("mlir-optimization", "output file", "optimized MLIR", "Output file not found")]
            | CompilerFailure errors -> CompilerFailure errors
        finally
            // Clean up temporary files if not keeping intermediates
            if intermediateDir.IsNone then
                if File.Exists(inputPath) then File.Delete(inputPath)
                if File.Exists(outputPath) then File.Delete(outputPath)

/// Translates MLIR to LLVM IR using mlir-translate
let translateToLLVMIR (mlirText: string) (intermediateDir: string option) : CompilerResult<string> =
    if String.IsNullOrWhiteSpace(mlirText) then
        CompilerFailure [ConversionError("mlir-translation", "empty input", "LLVM IR", "Input MLIR module is empty")]
    else
        // Create temporary files
        let tempDir = 
            match intermediateDir with 
            | Some dir -> dir 
            | None -> Path.GetTempPath()
            
        let inputPath = Path.Combine(tempDir, "input.mlir")
        let outputPath = Path.Combine(tempDir, "output.ll")
        
        try
            // Write input MLIR to file
            File.WriteAllText(inputPath, mlirText)
            
            // Run mlir-translate
            let translateArgs = sprintf "--mlir-to-llvmir %s -o %s" inputPath outputPath
            match runExternalCommand "mlir-translate" translateArgs with
            | Success _ ->
                // Read LLVM IR
                if File.Exists(outputPath) then
                    let llvmIR = File.ReadAllText(outputPath)
                    Success llvmIR
                else
                    CompilerFailure [ConversionError("mlir-translation", "output file", "LLVM IR", "Output file not found")]
            | CompilerFailure errors -> CompilerFailure errors
        finally
            // Clean up temporary files if not keeping intermediates
            if intermediateDir.IsNone then
                if File.Exists(inputPath) then File.Delete(inputPath)
                if File.Exists(outputPath) then File.Delete(outputPath)

/// Validates that MLIR module is well-formed (minimal validation)
let validateMLIR (mlirText: string) : CompilerResult<unit> =
    if String.IsNullOrWhiteSpace(mlirText) then
        CompilerFailure [ConversionError("mlir-validation", "empty input", "valid MLIR", "Input MLIR module is empty")]
    else if not (mlirText.Contains("module")) then
        CompilerFailure [ConversionError("mlir-validation", "malformed input", "valid MLIR", "Input does not contain a module declaration")]
    else
        Success ()

/// This is the function needed by the translator - simplified implementation
let applyLoweringPipeline (mlirModule: string) : CompilerResult<string> =
    if String.IsNullOrWhiteSpace(mlirModule) then
        CompilerFailure [ConversionError("lowering pipeline", "empty input", "LLVM dialect", "Input MLIR module is empty")]
    else
        // We'll use mlir-opt directly instead of custom conversions
        let tempDir = Path.GetTempPath()
        let inputPath = Path.Combine(tempDir, "input.mlir")
        let outputPath = Path.Combine(tempDir, "lowered.mlir")
        
        try
            // Write input MLIR to file
            File.WriteAllText(inputPath, mlirModule)
            
            // Build dialect conversion arguments
            let passArgs = standardPasses |> List.map (fun p -> p.PassOptions) |> String.concat " "
            let mlirOptArgs = sprintf "%s %s -o %s" passArgs inputPath outputPath
            
            // Run mlir-opt
            match runExternalCommand "mlir-opt" mlirOptArgs with
            | Success _ ->
                // Read optimized MLIR
                if File.Exists(outputPath) then
                    let loweredMLIR = File.ReadAllText(outputPath)
                    Success loweredMLIR
                else
                    CompilerFailure [ConversionError("lowering pipeline", "output file", "lowered MLIR", "Output file not found")]
            | CompilerFailure errors -> CompilerFailure errors
        finally
            // Clean up temporary files
            if File.Exists(inputPath) then File.Delete(inputPath)
            if File.Exists(outputPath) then File.Delete(outputPath)