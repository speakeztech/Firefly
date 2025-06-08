module Dabbit.Parsing.Translator

open System
open System.IO
open XParsec
open Core.XParsec.Foundation
open Core.XParsec.Foundation.Combinators
open Core.XParsec.Foundation.ErrorHandling
open Dabbit.Parsing.OakAst
open Core.MLIRGeneration.XParsecMLIRGenerator

/// Translation pipeline state for tracking the complete transformation
type TranslationPipelineState = {
    SourceFile: string
    CurrentPhase: string
    PhaseHistory: string list
    Diagnostics: (string * string) list
    IntermediateResults: Map<string, string>
    SymbolMappings: Map<string, string>
    ErrorContext: string list
}

/// Complete pipeline output with full traceability including AST representations
type TranslationPipelineOutput = {
    FinalMLIR: string
    PhaseOutputs: Map<string, string>
    SymbolMappings: Map<string, string>
    Diagnostics: (string * string) list
    SuccessfulPhases: string list
}

/// Phase execution tracking
module PhaseExecution =
    
    let enterPhase (phaseName: string) : Parser<unit, TranslationPipelineState> =
        fun state ->
            printfn "[PHASE-DEBUG] Entering phase: %s" phaseName
            let newState = { 
                state with 
                    CurrentPhase = phaseName
                    PhaseHistory = phaseName :: state.PhaseHistory
            }
            Reply(Ok (), newState)
    
    let recordDiagnostic (message: string) : Parser<unit, TranslationPipelineState> =
        fun state ->
            printfn "[DIAG-DEBUG] %s: %s" state.CurrentPhase message
            let diagnostic = (state.CurrentPhase, message)
            let newState = { 
                state with 
                    Diagnostics = diagnostic :: state.Diagnostics
            }
            Reply(Ok (), newState)
    
    let saveIntermediateResult (key: string) (result: string) : Parser<unit, TranslationPipelineState> =
        fun state ->
            printfn "[INTERMEDIATE-DEBUG] Saving result for key '%s' (%d characters)" key result.Length
            let newState = { 
                state with 
                    IntermediateResults = Map.add key result state.IntermediateResults
            }
            printfn "[INTERMEDIATE-DEBUG] Pipeline state now contains keys: %A" (Map.keys newState.IntermediateResults |> Seq.toList)
            Reply(Ok (), newState)

/// F# source to Oak AST transformation
module ASTPhase =
    
    let createASTPhase() : (string -> Parser<OakProgram, TranslationPipelineState>) =
        fun sourceCode ->
            enterPhase "ast-generation" >>= fun _ ->
            recordDiagnostic "Starting F# to Oak AST conversion" >>= fun _ ->
            
            fun state ->
                try
                    printfn "[AST-DEBUG] Starting AST generation for %d characters of source code" sourceCode.Length
                    printfn "[AST-DEBUG] Current pipeline state has %d intermediate results" state.IntermediateResults.Count
                    
                    let parsingResult = Dabbit.Parsing.AstConverter.parseAndConvertToOakAstWithIntermediate sourceCode
                    
                    printfn "[AST-DEBUG] AST conversion completed successfully"
                    printfn "[AST-DEBUG] FCS AST text length: %d characters" parsingResult.FCSAstText.Length
                    printfn "[AST-DEBUG] Oak AST text length: %d characters" parsingResult.OakAstText.Length
                    
                    if String.IsNullOrWhiteSpace(parsingResult.FCSAstText) then
                        printfn "[AST-DEBUG] WARNING: FCS AST text is null or whitespace!"
                    else
                        printfn "[AST-DEBUG] FCS AST preview: %s" (parsingResult.FCSAstText.Substring(0, min 100 parsingResult.FCSAstText.Length))
                    
                    if String.IsNullOrWhiteSpace(parsingResult.OakAstText) then
                        printfn "[AST-DEBUG] WARNING: Oak AST text is null or whitespace!"
                    else
                        printfn "[AST-DEBUG] Oak AST preview: %s" (parsingResult.OakAstText.Substring(0, min 100 parsingResult.OakAstText.Length))
                    
                    // Save FCS AST to pipeline state
                    printfn "[AST-DEBUG] Storing FCS AST with key 'f#-compiler-services-ast'"
                    let stateWithFCS = { state with IntermediateResults = Map.add "f#-compiler-services-ast" parsingResult.FCSAstText state.IntermediateResults }
                    printfn "[AST-DEBUG] Pipeline state after FCS AST: %A" (Map.keys stateWithFCS.IntermediateResults |> Seq.toList)
                    
                    // Save Oak AST to pipeline state
                    printfn "[AST-DEBUG] Storing Oak AST with key 'oak-ast'"
                    let finalState = { stateWithFCS with IntermediateResults = Map.add "oak-ast" parsingResult.OakAstText stateWithFCS.IntermediateResults }
                    printfn "[AST-DEBUG] Pipeline state after Oak AST: %A" (Map.keys finalState.IntermediateResults |> Seq.toList)
                    
                    printfn "[AST-DEBUG] Final pipeline state contains %d intermediate results" finalState.IntermediateResults.Count
                    printfn "[AST-DEBUG] AST phase completed successfully"
                    
                    Reply(Ok parsingResult.OakProgram, finalState)
                
                with
                | ex ->
                    printfn "[AST-DEBUG] EXCEPTION in AST generation: %s" ex.Message
                    printfn "[AST-DEBUG] Exception type: %s" (ex.GetType().Name)
                    printfn "[AST-DEBUG] Stack trace: %s" ex.StackTrace
                    Reply(Error, sprintf "AST generation failed: %s" ex.Message)

/// Oak transformation phases
module OakTransformationPhases =
    
    let createClosureEliminationPhase() : (OakProgram -> Parser<OakProgram, TranslationPipelineState>) =
        fun program ->
            enterPhase "closure-elimination" >>= fun _ ->
            recordDiagnostic "Starting closure elimination transformation" >>= fun _ ->
            
            match Dabbit.Closures.ClosureTransformer.eliminateClosures program with
            | Success transformedProgram ->
                recordDiagnostic "Successfully eliminated closures" >>= fun _ ->
                succeed transformedProgram
            | CompilerFailure errors ->
                let errorMsg = errors |> List.map (fun e -> e.ToString()) |> String.concat "; "
                compilerFail (TransformError("closure-elimination", "Oak AST with closures", "Oak AST without closures", errorMsg))
    
    let createUnionLayoutPhase() : (OakProgram -> Parser<OakProgram, TranslationPipelineState>) =
        fun program ->
            enterPhase "union-layout-compilation" >>= fun _ ->
            recordDiagnostic "Starting union layout compilation" >>= fun _ ->
            
            match Dabbit.UnionLayouts.FixedLayoutCompiler.compileFixedLayouts program with
            | Success transformedProgram ->
                recordDiagnostic "Successfully compiled union layouts" >>= fun _ ->
                succeed transformedProgram
            | CompilerFailure errors ->
                let errorMsg = errors |> List.map (fun e -> e.ToString()) |> String.concat "; "
                compilerFail (TransformError("union-layout-compilation", "Oak AST with unions", "Oak AST with fixed layouts", errorMsg))

/// MLIR generation phase
module MLIRGenerationPhase =
    
    let createMLIRGenerationPhase() : (OakProgram -> Parser<string, TranslationPipelineState>) =
        fun program ->
            enterPhase "mlir-generation" >>= fun _ ->
            recordDiagnostic "Starting MLIR generation" >>= fun _ ->
            
            match generateMLIRModuleText program with
            | Success mlirText ->
                recordDiagnostic (sprintf "Generated MLIR module (%d lines)" (mlirText.Split('\n').Length)) >>= fun _ ->
                saveIntermediateResult "mlir" mlirText >>= fun _ ->
                succeed mlirText
            | CompilerFailure errors ->
                let errorMsg = errors |> List.map (fun e -> e.ToString()) |> String.concat "; "
                compilerFail (TransformError("mlir-generation", "Oak AST", "MLIR", errorMsg))

/// MLIR lowering phase
module MLIRLoweringPhase =
    
    let createMLIRLoweringPhase() : (string -> Parser<string, TranslationPipelineState>) =
        fun mlirText ->
            enterPhase "mlir-lowering" >>= fun _ ->
            recordDiagnostic "Starting MLIR dialect lowering" >>= fun _ ->
            
            match Core.Conversion.LoweringPipeline.applyLoweringPipeline mlirText with
            | Success loweredMLIR ->
                recordDiagnostic "Successfully lowered MLIR to LLVM dialect" >>= fun _ ->
                succeed loweredMLIR
            | CompilerFailure errors ->
                let errorMsg = errors |> List.map (fun e -> e.ToString()) |> String.concat "; "
                compilerFail (TransformError("mlir-lowering", "high-level MLIR", "LLVM dialect MLIR", errorMsg))

/// Pipeline execution
module PipelineEngine =
    
    let executePhase<'Input, 'Output> (phase: 'Input -> Parser<'Output, TranslationPipelineState>) (input: 'Input) : Parser<'Output, TranslationPipelineState> =
        phase input

/// Main F# to MLIR translation pipeline
let executeTranslationPipeline (sourceFile: string) (sourceCode: string) (keepIntermediates: bool) : CompilerResult<TranslationPipelineOutput> =
    if String.IsNullOrWhiteSpace(sourceCode) then
        CompilerFailure [ParseError(
            { Line = 1; Column = 1; File = sourceFile; Offset = 0 },
            "Source code is empty",
            ["pipeline validation"])]
    else
        let basePath = Path.GetDirectoryName(sourceFile)
        let baseName = Path.GetFileNameWithoutExtension(sourceFile)
        
        printfn "[PIPELINE-DEBUG] Starting translation pipeline for file: %s" sourceFile
        printfn "[PIPELINE-DEBUG] Keep intermediates: %b" keepIntermediates
        
        let initialState = {
            SourceFile = sourceFile
            CurrentPhase = "initialization"
            PhaseHistory = []
            Diagnostics = []
            IntermediateResults = Map.empty
            SymbolMappings = Map.empty
            ErrorContext = []
        }
        
        // Create all phases
        let astPhase = ASTPhase.createASTPhase()
        let closureEliminationPhase = OakTransformationPhases.createClosureEliminationPhase()
        let unionLayoutPhase = OakTransformationPhases.createUnionLayoutPhase()
        let mlirGenerationPhase = MLIRGenerationPhase.createMLIRGenerationPhase()
        let mlirLoweringPhase = MLIRLoweringPhase.createMLIRLoweringPhase()
        
        // Execute complete pipeline
        let pipelineExecution = 
            PhaseExecution.enterPhase "pipeline-start" >>= fun _ ->
            PhaseExecution.recordDiagnostic "Starting F# to MLIR translation pipeline" >>= fun _ ->
            
            PipelineEngine.executePhase astPhase sourceCode >>= fun oakProgram ->
            printfn "Transforming closures..."
            PipelineEngine.executePhase closureEliminationPhase oakProgram >>= fun transformedOak1 ->
            printfn "Computing fixed layouts..."
            PipelineEngine.executePhase unionLayoutPhase transformedOak1 >>= fun transformedOak2 ->
            printfn "Generating MLIR using XParsec..."
            PipelineEngine.executePhase mlirGenerationPhase transformedOak2 >>= fun mlirText ->
            printfn "Applying MLIR lowering passes..."
            PipelineEngine.executePhase mlirLoweringPhase mlirText >>= fun loweredMLIR ->
            
            PhaseExecution.recordDiagnostic "Successfully completed all pipeline phases" >>= fun _ ->
            succeed loweredMLIR
        
        printfn "Parsing F# source code..."
        match pipelineExecution initialState with
        | Reply(Ok finalMLIR, finalState) ->
            printfn "[PIPELINE-DEBUG] Pipeline completed successfully"
            printfn "[PIPELINE-DEBUG] Final state contains %d intermediate results: %A" 
                    finalState.IntermediateResults.Count 
                    (Map.keys finalState.IntermediateResults |> Seq.toList)
            
            Success {
                FinalMLIR = finalMLIR
                PhaseOutputs = finalState.IntermediateResults
                SymbolMappings = finalState.SymbolMappings
                Diagnostics = List.rev finalState.Diagnostics
                SuccessfulPhases = List.rev finalState.PhaseHistory
            }
        
        | Reply(Error, errorMsg) ->
            printfn "[PIPELINE-DEBUG] Pipeline failed with error: %s" errorMsg
            CompilerFailure [TransformError("pipeline execution", "F# source", "MLIR", errorMsg)]

/// Entry point functions
let translateFSharpToMLIR (sourceFile: string) (sourceCode: string) : CompilerResult<string> =
    executeTranslationPipeline sourceFile sourceCode false >>= fun pipelineOutput ->
    Success pipelineOutput.FinalMLIR

let translateFSharpToMLIRWithDiagnostics (sourceFile: string) (sourceCode: string) : CompilerResult<TranslationPipelineOutput> =
    executeTranslationPipeline sourceFile sourceCode true

let translateFSharpToMLIRWithIntermediates (sourceFile: string) (sourceCode: string) (keepIntermediates: bool) : CompilerResult<TranslationPipelineOutput> =
    executeTranslationPipeline sourceFile sourceCode keepIntermediates

/// Validation functions
let validatePipelineOutput (output: TranslationPipelineOutput) : CompilerResult<unit> =
    if String.IsNullOrWhiteSpace(output.FinalMLIR) then
        CompilerFailure [TransformError("pipeline validation", "pipeline output", "valid MLIR", "Final MLIR output is empty")]
    elif output.SuccessfulPhases.Length < 3 then
        CompilerFailure [TransformError("pipeline validation", "pipeline phases", "complete pipeline", sprintf "Pipeline only completed %d phases, expected at least 3" output.SuccessfulPhases.Length)]
    else
        Success ()