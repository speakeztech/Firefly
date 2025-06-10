module Dabbit.Parsing.Translator

open System
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
    Diagnostics: (string * string) list  // Phase, Message
    IntermediateResults: Map<string, string>
    SymbolMappings: Map<string, string>
    ErrorContext: string list
}

/// Complete pipeline output with full traceability
type TranslationPipelineOutput = {
    FinalMLIR: string
    PhaseOutputs: Map<string, string>
    SymbolMappings: Map<string, string>
    Diagnostics: (string * string) list
    SuccessfulPhases: string list
}

/// Pipeline phase definition using XParsec
type PipelinePhase<'Input, 'Output> = {
    Name: string
    Description: string
    Transform: 'Input -> Parser<'Output, TranslationPipelineState>
    Validate: 'Output -> CompilerResult<unit>
    SaveIntermediate: bool
}

/// Phase execution tracking
module PhaseExecution =
    
    /// Records entering a new phase
    let enterPhase (phaseName: string) : Parser<unit, TranslationPipelineState> =
        fun state ->
            let newState = { 
                state with 
                    CurrentPhase = phaseName
                    PhaseHistory = phaseName :: state.PhaseHistory
            }
            Reply(Ok (), newState)
    
    /// Records a diagnostic message for current phase
    let recordDiagnostic (message: string) : Parser<unit, TranslationPipelineState> =
        fun state ->
            let diagnostic = (state.CurrentPhase, message)
            let newState = { 
                state with 
                    Diagnostics = diagnostic :: state.Diagnostics
            }
            Reply(Ok (), newState)
    
    /// Saves intermediate result for debugging/inspection
    let saveIntermediateResult (key: string) (result: string) : Parser<unit, TranslationPipelineState> =
        fun state ->
            let newState = { 
                state with 
                    IntermediateResults = Map.add key result state.IntermediateResults
            }
            Reply(Ok (), newState)
    
    /// Records a symbol mapping transformation
    let recordSymbolMapping (originalSymbol: string) (transformedSymbol: string) : Parser<unit, TranslationPipelineState> =
        fun state ->
            let newState = { 
                state with 
                    SymbolMappings = Map.add originalSymbol transformedSymbol state.SymbolMappings
            }
            Reply(Ok (), newState)

/// F# source to Oak AST transformation phase
module SourceToOakPhase =
    
    /// Creates F# source parsing phase
    let createSourceParsingPhase() : PipelinePhase<string, OakProgram> = {
        Name = "f#-source-parsing"
        Description = "Parse F# source code into Oak AST using XParsec combinators"
        Transform = fun sourceCode ->
            enterPhase "f#-source-parsing" >>= fun _ ->
            recordDiagnostic "Starting F# source code parsing" >>= fun _ ->
            
            // Use the correct AST converter function
            try
                let program = Dabbit.Parsing.AstConverter.parseAndConvertToOakAst sourceCode
                recordDiagnostic (sprintf "Successfully parsed %d modules" program.Modules.Length) >>= fun _ ->
                succeed program
            with
            | ex ->
                compilerFail (TransformError("f#-source-parsing", "source code", "Oak AST", ex.Message))
        
        Validate = fun program ->
            if program.Modules.IsEmpty then
                CompilerFailure [TransformError("source validation", "parsed program", "valid Oak AST", "Program must contain at least one module")]
            elif program.Modules |> List.exists (fun m -> m.Declarations.IsEmpty) then
                CompilerFailure [TransformError("source validation", "parsed modules", "valid Oak AST", "All modules must contain at least one declaration")]
            else
                Success ()
        
        SaveIntermediate = true
    }

/// Oak AST transformation phases
module OakTransformationPhases =
    
    /// Creates closure elimination phase
    let createClosureEliminationPhase() : PipelinePhase<OakProgram, OakProgram> = {
        Name = "closure-elimination"
        Description = "Transform closures to use explicit parameters using XParsec-based transformations"
        Transform = fun program ->
            enterPhase "closure-elimination" >>= fun _ ->
            recordDiagnostic "Starting closure elimination transformation" >>= fun _ ->
            
            match Dabbit.Closures.ClosureTransformer.eliminateClosures program with
            | Success transformedProgram ->
                recordDiagnostic "Successfully eliminated closures" >>= fun _ ->
                succeed transformedProgram
            | CompilerFailure errors ->
                let errorMsg = errors |> List.map (fun e -> e.ToString()) |> String.concat "; "
                compilerFail (TransformError("closure-elimination", "Oak AST with closures", "Oak AST without closures", errorMsg))
        
        Validate = fun program ->
            // Would validate that no closure expressions remain
            Success ()
        
        SaveIntermediate = true
    }
    
    /// Creates union layout compilation phase
    let createUnionLayoutPhase() : PipelinePhase<OakProgram, OakProgram> = {
        Name = "union-layout-compilation"
        Description = "Compile discriminated unions to fixed memory layouts using XParsec transformations"
        Transform = fun program ->
            enterPhase "union-layout-compilation" >>= fun _ ->
            recordDiagnostic "Starting union layout compilation" >>= fun _ ->
            
            match Dabbit.UnionLayouts.FixedLayoutCompiler.compileFixedLayouts program with
            | Success transformedProgram ->
                recordDiagnostic "Successfully compiled union layouts" >>= fun _ ->
                succeed transformedProgram
            | CompilerFailure errors ->
                let errorMsg = errors |> List.map (fun e -> e.ToString()) |> String.concat "; "
                compilerFail (TransformError("union-layout-compilation", "Oak AST with unions", "Oak AST with fixed layouts", errorMsg))
        
        Validate = fun program ->
            match Dabbit.UnionLayouts.FixedLayoutCompiler.validateZeroAllocationLayouts program with
            | Success true -> Success ()
            | Success false -> CompilerFailure [TransformError("union validation", "compiled unions", "zero-allocation layouts", "Some union layouts may cause heap allocations")]
            | CompilerFailure errors -> CompilerFailure errors
        
        SaveIntermediate = true
    }

/// MLIR generation phase
module MLIRGenerationPhase =
    
    /// Creates MLIR generation phase
    let createMLIRGenerationPhase() : PipelinePhase<OakProgram, string> = {
        Name = "mlir-generation"
        Description = "Generate MLIR from Oak AST using XParsec-based MLIR builders"
        Transform = fun program ->
            enterPhase "mlir-generation" >>= fun _ ->
            recordDiagnostic "Starting MLIR generation" >>= fun _ ->
            
            match generateMLIRModuleText program with
            | Success mlirText ->
                recordDiagnostic (sprintf "Generated MLIR module (%d lines)" (mlirText.Split('\n').Length)) >>= fun _ ->
                succeed mlirText
            | CompilerFailure errors ->
                let errorMsg = errors |> List.map (fun e -> e.ToString()) |> String.concat "; "
                compilerFail (TransformError("mlir-generation", "Oak AST", "MLIR", errorMsg))
        
        Validate = fun mlirText ->
            if String.IsNullOrWhiteSpace(mlirText) then
                CompilerFailure [TransformError("MLIR validation", "generated MLIR", "valid MLIR", "Generated MLIR is empty")]
            elif not (mlirText.Contains("module")) then
                CompilerFailure [TransformError("MLIR validation", "generated MLIR", "valid MLIR", "Generated MLIR must contain a module")]
            else
                Success ()
        
        SaveIntermediate = true
    }

/// MLIR lowering phase
module MLIRLoweringPhase =
    
    /// Creates MLIR lowering phase
    let createMLIRLoweringPhase() : PipelinePhase<string, string> = {
        Name = "mlir-lowering"
        Description = "Lower MLIR dialects to LLVM dialect using XParsec transformations"
        Transform = fun mlirText ->
            enterPhase "mlir-lowering" >>= fun _ ->
            recordDiagnostic "Starting MLIR dialect lowering" >>= fun _ ->
            
            match Core.Conversion.LoweringPipeline.applyLoweringPipeline mlirText with
            | Success loweredMLIR ->
                recordDiagnostic "Successfully lowered MLIR to LLVM dialect" >>= fun _ ->
                succeed loweredMLIR
            | CompilerFailure errors ->
                let errorMsg = errors |> List.map (fun e -> e.ToString()) |> String.concat "; "
                compilerFail (TransformError("mlir-lowering", "high-level MLIR", "LLVM dialect MLIR", errorMsg))
        
        Validate = fun loweredMLIR ->
            match Core.Conversion.LoweringPipeline.validateLLVMDialectOnly loweredMLIR with
            | Success () -> Success ()
            | CompilerFailure errors -> CompilerFailure errors
        
        SaveIntermediate = true
    }

/// LLVM IR generation phase
module LLVMGenerationPhase =
    
    /// Creates LLVM IR generation phase
    let createLLVMGenerationPhase() : PipelinePhase<string, Core.Conversion.LLVMTranslator.LLVMOutput> = {
        Name = "llvm-ir-generation"
        Description = "Generate LLVM IR from MLIR using XParsec-based translation"
        Transform = fun mlirText ->
            enterPhase "llvm-ir-generation" >>= fun _ ->
            recordDiagnostic "Starting LLVM IR generation" >>= fun _ ->
            
            match Core.Conversion.LLVMTranslator.translateToLLVM mlirText with
            | Success llvmOutput ->
                recordDiagnostic (sprintf "Generated LLVM IR for module '%s'" llvmOutput.ModuleName) >>= fun _ ->
                succeed llvmOutput
            | CompilerFailure errors ->
                let errorMsg = errors |> List.map (fun e -> e.ToString()) |> String.concat "; "
                compilerFail (TransformError("llvm-ir-generation", "MLIR", "LLVM IR", errorMsg))
        
        Validate = fun llvmOutput ->
            if String.IsNullOrWhiteSpace(llvmOutput.LLVMIRText) then
                CompilerFailure [TransformError("LLVM IR validation", "generated LLVM IR", "valid LLVM IR", "Generated LLVM IR is empty")]
            elif not (llvmOutput.LLVMIRText.Contains("define")) then
                CompilerFailure [TransformError("LLVM IR validation", "generated LLVM IR", "valid LLVM IR", "Generated LLVM IR must contain at least one function definition")]
            else
                Success ()
        
        SaveIntermediate = true
    }

/// Pipeline execution engine using XParsec combinators
module PipelineEngine =
    
    /// Executes a single phase with full error handling and validation
    let executePhase<'Input, 'Output> 
        (phase: PipelinePhase<'Input, 'Output>) 
        (input: 'Input) : Parser<'Output, TranslationPipelineState> =
        
        recordDiagnostic (sprintf "Executing phase: %s" phase.Description) >>= fun _ ->
        
        phase.Transform input >>= fun output ->
        
        // Validate output
        (match phase.Validate output with
         | Success () -> succeed ()
         | CompilerFailure errors ->
             let errorMsg = errors |> List.map (fun e -> e.ToString()) |> String.concat "; "
             compilerFail (TransformError(phase.Name + "-validation", "phase output", "validated output", errorMsg))) >>= fun _ ->
        
        // Save intermediate result if requested
        (if phase.SaveIntermediate then
            let outputStr = output.ToString()
            saveIntermediateResult phase.Name outputStr
         else
            succeed ()) >>= fun _ ->
        
        recordDiagnostic (sprintf "Successfully completed phase: %s" phase.Name) >>= fun _ ->
        succeed output
    
    /// Composes multiple phases into a pipeline
    let composePipeline<'A, 'B, 'C> 
        (phase1: PipelinePhase<'A, 'B>) 
        (phase2: PipelinePhase<'B, 'C>) : Parser<'C, TranslationPipelineState> =
        fun input ->
            executePhase phase1 input >>= fun intermediate ->
            executePhase phase2 intermediate

/// Complete F# to MLIR translation pipeline - NO FALLBACKS
let executeCompleteTranslationPipeline (sourceFile: string) (sourceCode: string) : CompilerResult<TranslationPipelineOutput> =
    if String.IsNullOrWhiteSpace(sourceCode) then
        CompilerFailure [ParseError(
            { Line = 1; Column = 1; File = sourceFile; Offset = 0 },
            "Source code is empty",
            ["pipeline validation"])]
    else
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
        let sourceParsingPhase = SourceToOakPhase.createSourceParsingPhase()
        let closureEliminationPhase = OakTransformationPhases.createClosureEliminationPhase()
        let unionLayoutPhase = OakTransformationPhases.createUnionLayoutPhase()
        let mlirGenerationPhase = MLIRGenerationPhase.createMLIRGenerationPhase()
        let mlirLoweringPhase = MLIRLoweringPhase.createMLIRLoweringPhase()
        
        // Execute complete pipeline
        let pipelineExecution = 
            PhaseExecution.enterPhase "pipeline-start" >>= fun _ ->
            PhaseExecution.recordDiagnostic "Starting complete F# to MLIR translation pipeline" >>= fun _ ->
            
            PipelineEngine.executePhase sourceParsingPhase sourceCode >>= fun oakProgram ->
            PipelineEngine.executePhase closureEliminationPhase oakProgram >>= fun transformedOak1 ->
            PipelineEngine.executePhase unionLayoutPhase transformedOak1 >>= fun transformedOak2 ->
            PipelineEngine.executePhase mlirGenerationPhase transformedOak2 >>= fun mlirText ->
            PipelineEngine.executePhase mlirLoweringPhase mlirText >>= fun loweredMLIR ->
            
            PhaseExecution.recordDiagnostic "Successfully completed all pipeline phases" >>= fun _ ->
            succeed loweredMLIR
        
        match pipelineExecution initialState with
        | Reply(Ok finalMLIR, finalState) ->
            Success {
                FinalMLIR = finalMLIR
                PhaseOutputs = finalState.IntermediateResults
                SymbolMappings = finalState.SymbolMappings
                Diagnostics = List.rev finalState.Diagnostics
                SuccessfulPhases = List.rev finalState.PhaseHistory
            }
        
        | Reply(Error, errorMsg) ->
            CompilerFailure [TransformError("pipeline execution", "F# source", "MLIR", errorMsg)]

/// Entry point for translation with comprehensive error reporting
let translateFsToMLIR (sourceFile: string) (sourceCode: string) : CompilerResult<string> =
    executeCompleteTranslationPipeline sourceFile sourceCode >>= fun pipelineOutput ->
    Success pipelineOutput.FinalMLIR

/// Entry point for translation with full pipeline output for debugging
let translateFsToMLIRWithDiagnostics (sourceFile: string) (sourceCode: string) : CompilerResult<TranslationPipelineOutput> =
    executeCompleteTranslationPipeline sourceFile sourceCode

/// Validates a complete translation pipeline result
let validatePipelineOutput (output: TranslationPipelineOutput) : CompilerResult<unit> =
    if String.IsNullOrWhiteSpace(output.FinalMLIR) then
        CompilerFailure [TransformError("pipeline validation", "pipeline output", "valid MLIR", "Final MLIR output is empty")]
    elif output.SuccessfulPhases.Length < 5 then  // Should have at least 5 major phases
        CompilerFailure [TransformError("pipeline validation", "pipeline phases", "complete pipeline", sprintf "Pipeline only completed %d phases, expected at least 5" output.SuccessfulPhases.Length)]
    else
        Success ()

/// Creates a custom translation pipeline with specific phases enabled/disabled
let createCustomPipeline (enableClosureElimination: bool) (enableUnionLayoutOptimization: bool) (enableMLIROptimizations: bool) 
                        : (string -> Parser<string, TranslationPipelineState>) =
    fun sourceCode ->
        let sourceParsingPhase = SourceToOakPhase.createSourceParsingPhase()
        let mlirGenerationPhase = MLIRGenerationPhase.createMLIRGenerationPhase()
        let mlirLoweringPhase = MLIRLoweringPhase.createMLIRLoweringPhase()
        
        PipelineEngine.executePhase sourceParsingPhase sourceCode >>= fun oakProgram ->
        
        let processedOak1 = 
            if enableClosureElimination then
                let closurePhase = OakTransformationPhases.createClosureEliminationPhase()
                PipelineEngine.executePhase closurePhase oakProgram
            else
                succeed oakProgram
        
        processedOak1 >>= fun oak1 ->
        
        let processedOak2 = 
            if enableUnionLayoutOptimization then
                let unionPhase = OakTransformationPhases.createUnionLayoutPhase()
                PipelineEngine.executePhase unionPhase oak1
            else
                succeed oak1
        
        processedOak2 >>= fun oak2 ->
        
        PipelineEngine.executePhase mlirGenerationPhase oak2 >>= fun mlirText ->
        PipelineEngine.executePhase mlirLoweringPhase mlirText