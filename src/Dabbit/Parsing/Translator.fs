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
    let sourceToOak (sourceFile: string) (sourceCode: string) : CompilerResult<OakProgram * string * string> =
        try
            let result = AstConverter.parseAndConvertToOakAst sourceFile sourceCode
            
            if result.OakProgram.Modules.IsEmpty && result.Diagnostics.Length > 0 then
                CompilerFailure [SyntaxError(
                    { Line = 1; Column = 1; File = sourceFile; Offset = 0 },
                    sprintf "Failed to parse source: %s" (String.concat "; " result.Diagnostics),
                    ["parsing"])]
            else
                let oakAstText = sprintf "%A" result.OakProgram
                Success (result.OakProgram, oakAstText, result.FSharpASTText)
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

/// Main translation pipeline - clean, focused version
let executeTranslationPipeline (sourceFile: string) (sourceCode: string) : CompilerResult<TranslationPipelineOutput> =
    printfn "=== Firefly Translation Pipeline ==="
    printfn "Source: %s (%d chars)" (Path.GetFileName(sourceFile)) sourceCode.Length
    
    let initialDiagnostics = [("pipeline", "Starting unified translation pipeline")]
    let initialPhaseOutputs = Map.empty<string, string>
    
    try
        // Step 1: Parse F# source to Oak AST
        match PipelineExecution.runPhase "parsing" 
            (Transformations.sourceToOak sourceFile) 
            sourceCode 
            initialDiagnostics with
        | Success (parseResult, diagnostics1) ->
            
            let (oakProgram, oakAstText, fsharpAstText) = parseResult
            let phaseOutputs1 = 
                initialPhaseOutputs
                |> Map.add "fsharp-ast" fsharpAstText
                |> Map.add "oak-ast" oakAstText
            printfn "✓ Parsing completed"
            
            // Step 2: Apply closure elimination
            match PipelineExecution.runPhase "closure-elimination" 
                Transformations.applyClosure 
                oakProgram 
                diagnostics1 with
            | Success (transformedProgram1, diagnostics2) ->
                
                let closureTransformedText = sprintf "%A" transformedProgram1
                let phaseOutputs2 = PipelineExecution.savePhaseOutput "closure-transformed" closureTransformedText id phaseOutputs1
                printfn "✓ Closure elimination completed"
                
                // Step 3: Apply union layout transformation
                match PipelineExecution.runPhase "union-layout" 
                    Transformations.applyUnionLayout 
                    transformedProgram1 
                    diagnostics2 with
                | Success (transformedProgram2, diagnostics3) ->
                    
                    let layoutTransformedText = sprintf "%A" transformedProgram2
                    let phaseOutputs3 = PipelineExecution.savePhaseOutput "layout-transformed" layoutTransformedText id phaseOutputs2
                    printfn "✓ Union layout completed"
                    
                    // Step 4: Generate MLIR
                    match PipelineExecution.runPhase "mlir-generation" 
                        Transformations.generateMLIR 
                        transformedProgram2 
                        diagnostics3 with
                    | Success (mlirText, diagnostics4) ->
                        
                        let phaseOutputs4 = PipelineExecution.savePhaseOutput "mlir" mlirText id phaseOutputs3
                        printfn "✓ MLIR generated (%d chars)" mlirText.Length
                        
                        // Step 5: Lower MLIR to LLVM dialect
                        match PipelineExecution.runPhase "mlir-lowering" 
                            Transformations.lowerMLIR 
                            mlirText 
                            diagnostics4 with
                        | Success (loweredMlir, diagnostics5) ->
                            
                            let phaseOutputs5 = PipelineExecution.savePhaseOutput "lowered-mlir" loweredMlir id phaseOutputs4
                            printfn "✓ MLIR lowering completed (%d chars)" loweredMlir.Length
                            
                            let successfulPhases = 
                                ["parsing"; "closure-elimination"; "union-layout"; "mlir-generation"; "mlir-lowering"]
                            
                            printfn "=== Pipeline completed successfully ==="
                            
                            Success {
                                FinalMLIR = loweredMlir
                                PhaseOutputs = phaseOutputs5
                                SymbolMappings = Map.empty
                                Diagnostics = List.rev diagnostics5
                                SuccessfulPhases = successfulPhases
                            }
                        | CompilerFailure errors -> 
                            printfn "✗ MLIR lowering failed"
                            CompilerFailure errors
                    | CompilerFailure errors -> 
                        printfn "✗ MLIR generation failed"
                        CompilerFailure errors
                | CompilerFailure errors -> 
                    printfn "✗ Union layout failed"
                    CompilerFailure errors
            | CompilerFailure errors -> 
                printfn "✗ Closure elimination failed"
                CompilerFailure errors
        | CompilerFailure errors -> 
            printfn "✗ Parsing failed"
            CompilerFailure errors
    with ex ->
        printfn "✗ Pipeline exception: %s" ex.Message
        CompilerFailure [ConversionError(
            "pipeline-execution", 
            "translation pipeline", 
            "MLIR", 
            sprintf "Exception: %s" ex.Message)]

/// Simple entry point for translation (maintains backward compatibility)
let translateFsToMLIR (sourceFile: string) (sourceCode: string) : CompilerResult<string> =
    match executeTranslationPipeline sourceFile sourceCode with
    | Success pipelineOutput -> Success pipelineOutput.FinalMLIR
    | CompilerFailure errors -> CompilerFailure errors

/// Entry point with full diagnostic information (primary interface)
let translateFsToMLIRWithDiagnostics (sourceFile: string) (sourceCode: string) : CompilerResult<TranslationPipelineOutput> =
    executeTranslationPipeline sourceFile sourceCode