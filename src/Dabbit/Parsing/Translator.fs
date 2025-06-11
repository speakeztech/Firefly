module Dabbit.Parsing.Translator

open System
open System.IO
open Core.XParsec.Foundation
open Dabbit.Parsing.OakAst
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
        
        // Add start diagnostic
        let startDiagnostic = (phaseName, sprintf "Starting %s phase" phaseName)
        let updatedDiagnostics = startDiagnostic :: diagnostics
        
        // Run transformation
        match transform input with
        | Success output ->
            // Add success diagnostic
            let successDiagnostic = (phaseName, sprintf "Successfully completed %s phase" phaseName)
            Success (output, successDiagnostic :: updatedDiagnostics)
            
        | CompilerFailure errors ->
            // Add failure diagnostic and propagate error
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

/// Core transformations without excessive XParsec complexity
module Transformations =
    /// Converts F# source to Oak AST
    let sourceToOak (sourceFile: string) (sourceCode: string) : CompilerResult<OakProgram * string> =
        try
            // Parse using Fantomas/FCS via AstConverter
            // Fix: Pass both sourceFile and sourceCode to match the updated function signature
            let result = AstConverter.parseAndConvertWithDiagnostics sourceFile sourceCode
            
            // Check for parse failures
            if result.OakProgram.Modules.IsEmpty && result.Diagnostics.Length > 0 then
                CompilerFailure [SyntaxError(
                    { Line = 1; Column = 1; File = sourceFile; Offset = 0 },
                    sprintf "Failed to parse source: %s" (String.concat "; " result.Diagnostics),
                    ["parsing"])]
            else
                // Convert to string representation for intermediate output
                let oakAstText = sprintf "%A" result.OakProgram
                
                Success (result.OakProgram, oakAstText)
        with ex ->
            CompilerFailure [SyntaxError(
                { Line = 1; Column = 1; File = sourceFile; Offset = 0 },
                sprintf "Exception during parsing: %s" ex.Message,
                ["parsing"; ex.StackTrace])]
    
    /// Applies closure elimination transformation
    let applyClosure (program: OakProgram) : CompilerResult<OakProgram> =
        eliminateClosures program
    
    /// Applies union layout transformation
    let applyUnionLayout (program: OakProgram) : CompilerResult<OakProgram> =
        compileFixedLayouts program
    
    /// Generates MLIR from Oak AST
    let generateMLIR (program: OakProgram) : CompilerResult<string> =
        generateMLIRModuleText program
    
    /// Lowers MLIR to LLVM dialect
    let lowerMLIR (mlirText: string) : CompilerResult<string> =
        Core.Conversion.LoweringPipeline.applyLoweringPipeline mlirText

/// Helper function to combine multiple results safely
let combineResults (results: CompilerResult<'T> list) : CompilerResult<'T list> =
    let folder acc result =
        match acc, result with
        | Success accValues, Success value -> Success (value :: accValues)
        | CompilerFailure errors, Success _ -> CompilerFailure errors
        | Success _, CompilerFailure errors -> CompilerFailure errors
        | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
    
    results
    |> List.fold folder (Success [])
    |> function
       | Success values -> Success (List.rev values)
       | CompilerFailure errors -> CompilerFailure errors

/// Main translation pipeline without excessive parser combinators
let executeTranslationPipeline (sourceFile: string) (sourceCode: string) : CompilerResult<TranslationPipelineOutput> =
    // Initialize pipeline state
    let initialDiagnostics = [("pipeline", "Starting translation pipeline")]
    let initialPhaseOutputs = Map.empty<string, string>
    
    try
        // Step 1: Parse F# source to Oak AST
        match PipelineExecution.runPhase "parsing" 
            (Transformations.sourceToOak sourceFile) 
            sourceCode 
            initialDiagnostics with
        | Success (parseResult, diagnostics1) ->
            
            let (oakProgram, oakAstText) = parseResult
            let phaseOutputs1 = PipelineExecution.savePhaseOutput "oak-ast" oakAstText id initialPhaseOutputs
            
            // Step 2: Apply closure elimination
            match PipelineExecution.runPhase "closure-elimination" 
                Transformations.applyClosure 
                oakProgram 
                diagnostics1 with
            | Success (transformedProgram1, diagnostics2) ->
                
                let closureTransformedText = sprintf "%A" transformedProgram1
                let phaseOutputs2 = PipelineExecution.savePhaseOutput "closure-transformed" closureTransformedText id phaseOutputs1
                
                // Step 3: Apply union layout transformation
                match PipelineExecution.runPhase "union-layout" 
                    Transformations.applyUnionLayout 
                    transformedProgram1 
                    diagnostics2 with
                | Success (transformedProgram2, diagnostics3) ->
                    
                    let layoutTransformedText = sprintf "%A" transformedProgram2
                    let phaseOutputs3 = PipelineExecution.savePhaseOutput "layout-transformed" layoutTransformedText id phaseOutputs2
                    
                    // Step 4: Generate MLIR
                    match PipelineExecution.runPhase "mlir-generation" 
                        Transformations.generateMLIR 
                        transformedProgram2 
                        diagnostics3 with
                    | Success (mlirText, diagnostics4) ->
                        
                        let phaseOutputs4 = PipelineExecution.savePhaseOutput "mlir" mlirText id phaseOutputs3
                        
                        // Step 5: Lower MLIR to LLVM dialect
                        match PipelineExecution.runPhase "mlir-lowering" 
                            Transformations.lowerMLIR 
                            mlirText 
                            diagnostics4 with
                        | Success (loweredMlir, diagnostics5) ->
                            
                            let phaseOutputs5 = PipelineExecution.savePhaseOutput "lowered-mlir" loweredMlir id phaseOutputs4
                            
                            // Assemble successful phases
                            let successfulPhases = 
                                ["parsing"; "closure-elimination"; "union-layout"; "mlir-generation"; "mlir-lowering"]
                            
                            // Build final output
                            Success {
                                FinalMLIR = loweredMlir
                                PhaseOutputs = phaseOutputs5
                                SymbolMappings = Map.empty  // Symbol mappings would be tracked throughout the pipeline
                                Diagnostics = List.rev diagnostics5
                                SuccessfulPhases = successfulPhases
                            }
                        | CompilerFailure errors -> CompilerFailure errors
                    | CompilerFailure errors -> CompilerFailure errors
                | CompilerFailure errors -> CompilerFailure errors
            | CompilerFailure errors -> CompilerFailure errors
        | CompilerFailure errors -> CompilerFailure errors
    with ex ->
        // Handle unexpected exceptions
        CompilerFailure [ConversionError(
            "pipeline-execution", 
            "translation pipeline", 
            "MLIR", 
            sprintf "Unexpected exception: %s\n%s" ex.Message ex.StackTrace)]

/// Simple entry point for translation
let translateFsToMLIR (sourceFile: string) (sourceCode: string) : CompilerResult<string> =
    match executeTranslationPipeline sourceFile sourceCode with
    | Success pipelineOutput -> Success pipelineOutput.FinalMLIR
    | CompilerFailure errors -> CompilerFailure errors

/// Entry point with full diagnostic information
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
        printfn "Skipping intermediate file generation (disabled)"
        ()
    else
        try
            // Create intermediates directory
            let intermediatesDir = Path.Combine(basePath, "intermediates")
            if not (Directory.Exists(intermediatesDir)) then
                Directory.CreateDirectory(intermediatesDir) |> ignore
            
            // Save all phase outputs
            pipelineOutput.PhaseOutputs
            |> Map.iter (fun phaseName output ->
                let extension = 
                    match phaseName with
                    | name when name.Contains("mlir") -> ".mlir"
                    | name when name.Contains("oak") -> ".oak"
                    | _ -> ".txt"
                let fileName = sprintf "%s.%s%s" baseName phaseName extension
                let filePath = Path.Combine(intermediatesDir, fileName)
                File.WriteAllText(filePath, output))
            
            // Save diagnostics
            let diagnosticsPath = Path.Combine(intermediatesDir, baseName + ".diagnostics.txt")
            let diagnosticsContent = 
                pipelineOutput.Diagnostics
                |> List.map (fun (phase, message) -> sprintf "[%s] %s" phase message)
                |> String.concat "\n"
            File.WriteAllText(diagnosticsPath, diagnosticsContent)
            
            printfn "Saved intermediate files to: %s" intermediatesDir
        with ex ->
            printfn "Warning: Failed to save intermediate files: %s" ex.Message