module Dabbit.Parsing.Translator

open System
open System.IO
open Core.XParsec.Foundation
open Dabbit.Parsing.OakAst
open Dabbit.Parsing.AstConverter
open Dabbit.Closures.ClosureTransformer
open Dabbit.UnionLayouts.FixedLayoutCompiler
open Core.MLIRGeneration.XParsecMLIRGenerator

/// Pipeline phase identifier and metadata
type PipelinePhase = {
    Name: string
    Description: string
    IsOptional: bool
}

/// Complete pipeline output with diagnostic information
type TranslationPipelineOutput = {
    FinalMLIR: string
    PhaseOutputs: Map<string, string>
    SymbolMappings: Map<string, string>
    Diagnostics: (string * string) list
    SuccessfulPhases: string list
}

/// Defines the standard Firefly compilation pipeline phases
let standardPipeline = [
    { Name = "parse"; Description = "Parse F# source code to AST"; IsOptional = false }
    { Name = "oak-convert"; Description = "Convert F# AST to Oak AST"; IsOptional = false }
    { Name = "closure-elimination"; Description = "Eliminate closures with explicit parameters"; IsOptional = false }
    { Name = "union-layout"; Description = "Compile discriminated unions to fixed layouts"; IsOptional = false }
    { Name = "mlir-generation"; Description = "Generate MLIR from Oak AST"; IsOptional = false }
    { Name = "mlir-lowering"; Description = "Lower MLIR to LLVM dialect"; IsOptional = false }
]

/// Pipeline execution with proper error tracking
module PipelineExecution =
    /// Runs a single phase with diagnostics
    let runPhase<'Input, 'Output> 
                (phaseName: string) 
                (transform: 'Input -> CompilerResult<'Output>) 
                (input: 'Input)
                (diagnostics: (string * string) list) : CompilerResult<'Output * (string * string) list> =
        
        let startDiagnostic = (phaseName, sprintf "Starting %s phase" phaseName)
        let updatedDiagnostics = startDiagnostic :: diagnostics
        
        match transform input with
        | Success output ->
            let successDiagnostic = (phaseName, sprintf "Successfully completed %s phase" phaseName)
            Success (output, successDiagnostic :: updatedDiagnostics)
            
        | CompilerFailure errors ->
            let errorMessages = errors |> List.map (fun e -> e.ToString())
            let errorDiagnostic = (phaseName, sprintf "Failed in %s phase: %s" phaseName (String.concat "; " errorMessages))
            CompilerFailure errors
    
    /// Saves intermediate output for debugging
    let savePhaseOutput 
                    (phaseName: string) 
                    (output: 'Output) 
                    (toString: 'Output -> string)
                    (phaseOutputs: Map<string, string>) : Map<string, string> =
        
        let outputStr = toString output
        Map.add phaseName outputStr phaseOutputs

/// Core transformations using unified pipeline
module Transformations =
    /// Converts F# source to Oak AST using the unified function
    let sourceToOak (sourceFile: string) (sourceCode: string) : CompilerResult<OakProgram * string> =
        try
            printfn "Phase: F# source to Oak AST conversion"
            printfn "Using unified AstConverter.parseAndConvertToOakAst function"
            
            // Use the unified function that combines working configuration with diagnostics
            let result = AstConverter.parseAndConvertToOakAst sourceFile sourceCode
            
            // Check for parse failures
            if result.OakProgram.Modules.IsEmpty && result.Diagnostics.Length > 0 then
                CompilerFailure [SyntaxError(
                    { Line = 1; Column = 1; File = sourceFile; Offset = 0 },
                    sprintf "Failed to parse source: %s" (String.concat "; " result.Diagnostics),
                    ["parsing"])]
            else
                // Convert to string representation for intermediate output
                let oakAstText = sprintf "%A" result.OakProgram
                printfn "Oak AST generation completed successfully"
                
                Success (result.OakProgram, oakAstText)
        with ex ->
            CompilerFailure [SyntaxError(
                { Line = 1; Column = 1; File = sourceFile; Offset = 0 },
                sprintf "Exception during parsing: %s" ex.Message,
                ["parsing"; ex.StackTrace])]
    
    /// Applies closure elimination transformation
    let applyClosure (program: OakProgram) : CompilerResult<OakProgram> =
        printfn "Phase: Closure elimination"
        eliminateClosures program
    
    /// Applies union layout transformation
    let applyUnionLayout (program: OakProgram) : CompilerResult<OakProgram> =
        printfn "Phase: Union layout compilation"
        compileFixedLayouts program
    
    /// Generates MLIR from Oak AST
    let generateMLIR (program: OakProgram) : CompilerResult<string> =
        printfn "Phase: MLIR generation"
        generateMLIRModuleText program
    
    /// Lowers MLIR to LLVM dialect
    let lowerMLIR (mlirText: string) : CompilerResult<string> =
        printfn "Phase: MLIR lowering to LLVM dialect"
        Core.Conversion.LoweringPipeline.applyLoweringPipeline mlirText

/// Main translation pipeline - single route only
let executeTranslationPipeline (sourceFile: string) (sourceCode: string) : CompilerResult<TranslationPipelineOutput> =
    printfn "=== Starting Firefly Translation Pipeline ==="
    printfn "Source file: %s" sourceFile
    printfn "Source length: %d characters" sourceCode.Length
    
    let initialDiagnostics = [("pipeline", "Starting unified translation pipeline")]
    let initialPhaseOutputs = Map.empty<string, string>
    
    try
        // Step 1: Parse F# source to Oak AST (unified function)
        match PipelineExecution.runPhase "parsing" 
            (Transformations.sourceToOak sourceFile) 
            sourceCode 
            initialDiagnostics with
        | Success (parseResult, diagnostics1) ->
            
            let (oakProgram, oakAstText) = parseResult
            let phaseOutputs1 = PipelineExecution.savePhaseOutput "oak-ast" oakAstText id initialPhaseOutputs
            printfn "✓ Parsing phase completed"
            
            // Step 2: Apply closure elimination
            match PipelineExecution.runPhase "closure-elimination" 
                Transformations.applyClosure 
                oakProgram 
                diagnostics1 with
            | Success (transformedProgram1, diagnostics2) ->
                
                let closureTransformedText = sprintf "%A" transformedProgram1
                let phaseOutputs2 = PipelineExecution.savePhaseOutput "closure-transformed" closureTransformedText id phaseOutputs1
                printfn "✓ Closure elimination phase completed"
                
                // Step 3: Apply union layout transformation
                match PipelineExecution.runPhase "union-layout" 
                    Transformations.applyUnionLayout 
                    transformedProgram1 
                    diagnostics2 with
                | Success (transformedProgram2, diagnostics3) ->
                    
                    let layoutTransformedText = sprintf "%A" transformedProgram2
                    let phaseOutputs3 = PipelineExecution.savePhaseOutput "layout-transformed" layoutTransformedText id phaseOutputs2
                    printfn "✓ Union layout phase completed"
                    
                    // Step 4: Generate MLIR
                    match PipelineExecution.runPhase "mlir-generation" 
                        Transformations.generateMLIR 
                        transformedProgram2 
                        diagnostics3 with
                    | Success (mlirText, diagnostics4) ->
                        
                        printfn "MLIR text generated: %d characters" mlirText.Length
                        printfn "First 100 chars: %s" (mlirText.Substring(0, min 100 mlirText.Length))
                        
                        let phaseOutputs4 = PipelineExecution.savePhaseOutput "mlir" mlirText id phaseOutputs3
                        printfn "Phase outputs after MLIR save: %A" (phaseOutputs4.Keys |> Seq.toList)
                        
                        // Step 5: Lower MLIR to LLVM dialect
                        match PipelineExecution.runPhase "mlir-lowering" 
                            Transformations.lowerMLIR 
                            mlirText 
                            diagnostics4 with
                        | Success (loweredMlir, diagnostics5) ->
                            
                            printfn "Lowered MLIR text: %d characters" loweredMlir.Length
                            
                            let phaseOutputs5 = PipelineExecution.savePhaseOutput "lowered-mlir" loweredMlir id phaseOutputs4
                            printfn "Phase outputs after lowered MLIR save: %A" (phaseOutputs5.Keys |> Seq.toList)
                            printfn "✓ MLIR lowering phase completed"
                            
                            let successfulPhases = 
                                ["parsing"; "closure-elimination"; "union-layout"; "mlir-generation"; "mlir-lowering"]
                            
                            printfn "=== Pipeline completed successfully with %d phases ===" successfulPhases.Length
                            
                            Success {
                                FinalMLIR = loweredMlir
                                PhaseOutputs = phaseOutputs5
                                SymbolMappings = Map.empty
                                Diagnostics = List.rev diagnostics5
                                SuccessfulPhases = successfulPhases
                            }
                        | CompilerFailure errors -> 
                            printfn "✗ MLIR lowering phase failed"
                            CompilerFailure errors
                    | CompilerFailure errors -> 
                        printfn "✗ MLIR generation phase failed"
                        CompilerFailure errors
                | CompilerFailure errors -> 
                    printfn "✗ Union layout phase failed"
                    CompilerFailure errors
            | CompilerFailure errors -> 
                printfn "✗ Closure elimination phase failed"
                CompilerFailure errors
        | CompilerFailure errors -> 
            printfn "✗ Parsing phase failed"
            CompilerFailure errors
    with ex ->
        printfn "✗ Pipeline execution exception: %s" ex.Message
        CompilerFailure [ConversionError(
            "pipeline-execution", 
            "translation pipeline", 
            "MLIR", 
            sprintf "Unexpected exception: %s\n%s" ex.Message ex.StackTrace)]

/// Simple entry point for translation (maintains backward compatibility)
let translateFsToMLIR (sourceFile: string) (sourceCode: string) : CompilerResult<string> =
    match executeTranslationPipeline sourceFile sourceCode with
    | Success pipelineOutput -> Success pipelineOutput.FinalMLIR
    | CompilerFailure errors -> CompilerFailure errors

/// Entry point with full diagnostic information (primary interface)
let translateFsToMLIRWithDiagnostics (sourceFile: string) (sourceCode: string) : CompilerResult<TranslationPipelineOutput> =
    executeTranslationPipeline sourceFile sourceCode

/// Controls whether to save intermediate outputs for debugging
let mutable saveIntermediates = false

/// Sets whether to save intermediate files
let setSaveIntermediates (value: bool) : unit =
    saveIntermediates <- value

/// Helper function to create intermediate files from pipeline output
let saveIntermediateFiles (basePath: string) (baseName: string) (pipelineOutput: TranslationPipelineOutput) : unit =
    if not saveIntermediates then
        printfn "Intermediate file generation disabled"
        ()
    else
        try
            // Debug output to show what phase outputs are available
            printfn "=== Intermediate File Saving Debug ==="
            printfn "Phase outputs available for saving: %A" (pipelineOutput.PhaseOutputs.Keys |> Seq.toList)
            pipelineOutput.PhaseOutputs |> Map.iter (fun key value -> 
                printfn "  %s: %d characters" key value.Length)
            printfn "Base path: %s" basePath
            printfn "Base name: %s" baseName
            
            let intermediatesDir = Path.Combine(basePath, "intermediates")
            if not (Directory.Exists(intermediatesDir)) then
                Directory.CreateDirectory(intermediatesDir) |> ignore
            
            // Save all phase outputs with proper extensions
            pipelineOutput.PhaseOutputs
            |> Map.iter (fun phaseName output ->
                let extension = 
                    match phaseName with
                    | "mlir" -> ".mlir"
                    | "lowered-mlir" -> ".lowered.mlir"
                    | name when name.Contains("mlir") -> ".mlir"
                    | name when name.Contains("oak") -> ".oak"
                    | _ -> ".txt"
                let fileName = sprintf "%s.%s%s" baseName phaseName extension
                let filePath = Path.Combine(intermediatesDir, fileName)
                File.WriteAllText(filePath, output)
                printfn "Saved intermediate file: %s (%d chars)" filePath output.Length)
            
            // Also save MLIR directly to main directory for easy access
            match pipelineOutput.PhaseOutputs.TryFind "mlir" with
            | Some mlirContent ->
                let mlirPath = Path.ChangeExtension(Path.Combine(basePath, baseName), ".mlir")
                File.WriteAllText(mlirPath, mlirContent)
                printfn "Saved MLIR file: %s" mlirPath
            | None ->
                printfn "Warning: No MLIR content found in pipeline output"
            
            let diagnosticsPath = Path.Combine(intermediatesDir, baseName + ".diagnostics.txt")
            let diagnosticsContent = 
                pipelineOutput.Diagnostics
                |> List.map (fun (phase, message) -> sprintf "[%s] %s" phase message)
                |> String.concat "\n"
            File.WriteAllText(diagnosticsPath, diagnosticsContent)
            
            printfn "Saved intermediate files to: %s" intermediatesDir
            printfn "=== End Intermediate File Saving Debug ==="
        with ex ->
            printfn "Warning: Failed to save intermediate files: %s" ex.Message