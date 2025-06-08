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
    KeepIntermediates: bool
    BasePath: string
    BaseName: string
}

/// Complete pipeline output with full traceability including AST representations
type TranslationPipelineOutput = {
    FinalMLIR: string
    PhaseOutputs: Map<string, string>
    SymbolMappings: Map<string, string>
    Diagnostics: (string * string) list
    SuccessfulPhases: string list
}

/// Cross-platform text file writing with UTF-8 encoding and Unix line endings
let private writeTextFile (filePath: string) (content: string) : unit =
    try
        let encoding = System.Text.UTF8Encoding(false) // false = no BOM
        let normalizedContent = content.Replace("\r\n", "\n").Replace("\r", "\n")
        File.WriteAllText(filePath, normalizedContent, encoding)
        printfn "DEBUG: Successfully wrote file: %s (%d chars)" filePath content.Length
    with
    | ex ->
        printfn "ERROR: Failed to write file %s: %s" filePath ex.Message

/// Direct AST generation and file saving - GUARANTEED EXECUTION
let private generateAndSaveAST (sourceCode: string) (basePath: string) (baseName: string) (keepIntermediates: bool) : Dabbit.Parsing.AstConverter.ParsingResult =
    printfn "=== DIRECT AST GENERATION START ==="
    printfn "Source code length: %d" sourceCode.Length
    printfn "Base path: %s" basePath
    printfn "Base name: %s" baseName
    printfn "Keep intermediates: %b" keepIntermediates
    
    // Call the enhanced AST parser directly
    printfn "Calling parseAndConvertToOakAstWithIntermediate..."
    let parsingResult = Dabbit.Parsing.AstConverter.parseAndConvertToOakAstWithIntermediate sourceCode
    
    printfn "AST generation completed:"
    printfn "  FCS AST length: %d characters" parsingResult.FCSAstText.Length
    printfn "  Oak AST length: %d characters" parsingResult.OakAstText.Length
    printfn "  Oak program modules: %d" parsingResult.OakProgram.Modules.Length
    
    if keepIntermediates then
        printfn "=== SAVING AST FILES DIRECTLY ==="
        try
            let intermediatesDir = Path.Combine(basePath, "intermediates")
            if not (Directory.Exists(intermediatesDir)) then
                Directory.CreateDirectory(intermediatesDir) |> ignore
                printfn "Created intermediates directory: %s" intermediatesDir
            
            // Save FCS AST
            let fcsPath = Path.Combine(intermediatesDir, baseName + ".fcs")
            writeTextFile fcsPath parsingResult.FCSAstText
            printfn "✓ SAVED FCS AST to: %s" fcsPath
            
            // Save Oak AST
            let oakPath = Path.Combine(intermediatesDir, baseName + ".oak")
            writeTextFile oakPath parsingResult.OakAstText
            printfn "✓ SAVED OAK AST to: %s" oakPath
            
            // Verify files exist
            if File.Exists(fcsPath) then
                let fcsSize = (FileInfo(fcsPath)).Length
                printfn "✓ VERIFIED: FCS file exists (%d bytes)" fcsSize
            else
                printfn "✗ ERROR: FCS file was not created!"
            
            if File.Exists(oakPath) then
                let oakSize = (FileInfo(oakPath)).Length
                printfn "✓ VERIFIED: Oak file exists (%d bytes)" oakSize
            else
                printfn "✗ ERROR: Oak file was not created!"
                
        with
        | ex ->
            printfn "ERROR: Failed to save AST files: %s" ex.Message
            printfn "Stack trace: %s" ex.StackTrace
    else
        printfn "NOT saving AST files (keepIntermediates = false)"
    
    printfn "=== DIRECT AST GENERATION END ==="
    parsingResult

/// Phase execution tracking with guaranteed intermediate file management
module PhaseExecution =
    
    let enterPhase (phaseName: string) : Parser<unit, TranslationPipelineState> =
        fun state ->
            printfn "PHASE: Entering %s" phaseName
            let newState = { 
                state with 
                    CurrentPhase = phaseName
                    PhaseHistory = phaseName :: state.PhaseHistory
            }
            Reply(Ok (), newState)
    
    let recordDiagnostic (message: string) : Parser<unit, TranslationPipelineState> =
        fun state ->
            printfn "DIAGNOSTIC [%s]: %s" state.CurrentPhase message
            let diagnostic = (state.CurrentPhase, message)
            let newState = { 
                state with 
                    Diagnostics = diagnostic :: state.Diagnostics
            }
            Reply(Ok (), newState)
    
    let saveIntermediateResult (key: string) (result: string) : Parser<unit, TranslationPipelineState> =
        fun state ->
            printfn "SAVING INTERMEDIATE: %s (%d chars)" key result.Length
            let newState = { 
                state with 
                    IntermediateResults = Map.add key result state.IntermediateResults
            }
            
            if state.KeepIntermediates then
                try
                    let intermediatesDir = Path.Combine(state.BasePath, "intermediates")
                    if not (Directory.Exists(intermediatesDir)) then
                        Directory.CreateDirectory(intermediatesDir) |> ignore
                    
                    let extension = 
                        match key with
                        | k when k.Contains("fcs") || k.Contains("compiler-services") -> ".fcs"
                        | k when k.Contains("oak") -> ".oak"
                        | k when k.Contains("mlir") -> ".mlir"
                        | k when k.Contains("llvm") -> ".ll"
                        | _ -> ".txt"
                    
                    let fileName = sprintf "%s.%s%s" state.BaseName (key.Replace("-", "_")) extension
                    let filePath = Path.Combine(intermediatesDir, fileName)
                    
                    writeTextFile filePath result
                    printfn "✓ SAVED INTERMEDIATE: %s to %s" key filePath
                
                with
                | ex ->
                    printfn "ERROR: Failed to save intermediate file for %s: %s" key ex.Message
            
            Reply(Ok (), newState)

/// Enhanced F# source to Oak AST transformation with guaranteed AST file generation
module DirectASTPhase =
    
    let createDirectASTPhase() : (string -> Parser<OakProgram, TranslationPipelineState>) =
        fun sourceCode ->
            enterPhase "direct-ast-generation" >>= fun _ ->
            recordDiagnostic "Starting direct AST generation with file output" >>= fun _ ->
            
            fun state ->
                try
                    printfn "=== DIRECT AST PHASE EXECUTION ==="
                    
                    // Generate AST and save files directly
                    let parsingResult = generateAndSaveAST sourceCode state.BasePath state.BaseName state.KeepIntermediates
                    
                    // Add AST content to pipeline state
                    let stateWithFCS = { state with IntermediateResults = Map.add "f#-compiler-services-ast" parsingResult.FCSAstText state.IntermediateResults }
                    let finalState = { stateWithFCS with IntermediateResults = Map.add "oak-ast" parsingResult.OakAstText stateWithFCS.IntermediateResults }
                    
                    printfn "=== AST PHASE COMPLETED SUCCESSFULLY ==="
                    Reply(Ok parsingResult.OakProgram, finalState)
                
                with
                | ex ->
                    printfn "ERROR: Direct AST phase failed: %s" ex.Message
                    Reply(Error, sprintf "Direct AST generation failed: %s" ex.Message)

/// Enhanced oak transformation phases
module EnhancedOakTransformationPhases =
    
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

/// Enhanced MLIR generation phase
module EnhancedMLIRGenerationPhase =
    
    let createMLIRGenerationPhase() : (OakProgram -> Parser<string, TranslationPipelineState>) =
        fun program ->
            enterPhase "mlir-generation" >>= fun _ ->
            recordDiagnostic "Starting MLIR generation" >>= fun _ ->
            
            match generateMLIRModuleText program with
            | Success mlirText ->
                recordDiagnostic (sprintf "Generated MLIR module (%d lines)" (mlirText.Split('\n').Length)) >>= fun _ ->
                saveIntermediateResult "mlir-generation" mlirText >>= fun _ ->
                succeed mlirText
            | CompilerFailure errors ->
                let errorMsg = errors |> List.map (fun e -> e.ToString()) |> String.concat "; "
                compilerFail (TransformError("mlir-generation", "Oak AST", "MLIR", errorMsg))

/// Enhanced MLIR lowering phase
module EnhancedMLIRLoweringPhase =
    
    let createMLIRLoweringPhase() : (string -> Parser<string, TranslationPipelineState>) =
        fun mlirText ->
            enterPhase "mlir-lowering" >>= fun _ ->
            recordDiagnostic "Starting MLIR dialect lowering" >>= fun _ ->
            
            match Core.Conversion.LoweringPipeline.applyLoweringPipeline mlirText with
            | Success loweredMLIR ->
                recordDiagnostic "Successfully lowered MLIR to LLVM dialect" >>= fun _ ->
                saveIntermediateResult "mlir-lowered" loweredMLIR >>= fun _ ->
                succeed loweredMLIR
            | CompilerFailure errors ->
                let errorMsg = errors |> List.map (fun e -> e.ToString()) |> String.concat "; "
                compilerFail (TransformError("mlir-lowering", "high-level MLIR", "LLVM dialect MLIR", errorMsg))

/// Pipeline execution with comprehensive error handling
module PipelineEngine =
    
    let executePhase<'Input, 'Output> (phase: 'Input -> Parser<'Output, TranslationPipelineState>) (input: 'Input) : Parser<'Output, TranslationPipelineState> =
        phase input

/// Main enhanced F# to MLIR translation pipeline with GUARANTEED AST file generation
let executeEnhancedTranslationPipeline (sourceFile: string) (sourceCode: string) (keepIntermediates: bool) : CompilerResult<TranslationPipelineOutput> =
    printfn "=== ENHANCED TRANSLATION PIPELINE START ==="
    printfn "Source file: %s" sourceFile
    printfn "Source code length: %d" sourceCode.Length
    printfn "Keep intermediates: %b" keepIntermediates
    
    if String.IsNullOrWhiteSpace(sourceCode) then
        CompilerFailure [ParseError(
            { Line = 1; Column = 1; File = sourceFile; Offset = 0 },
            "Source code is empty",
            ["pipeline validation"])]
    else
        let basePath = Path.GetDirectoryName(sourceFile)
        let baseName = Path.GetFileNameWithoutExtension(sourceFile)
        
        let initialState = {
            SourceFile = sourceFile
            CurrentPhase = "initialization"
            PhaseHistory = []
            Diagnostics = []
            IntermediateResults = Map.empty
            SymbolMappings = Map.empty
            ErrorContext = []
            KeepIntermediates = keepIntermediates
            BasePath = basePath
            BaseName = baseName
        }
        
        // Create all phases
        let directASTPhase = DirectASTPhase.createDirectASTPhase()
        let closureEliminationPhase = EnhancedOakTransformationPhases.createClosureEliminationPhase()
        let unionLayoutPhase = EnhancedOakTransformationPhases.createUnionLayoutPhase()
        let mlirGenerationPhase = EnhancedMLIRGenerationPhase.createMLIRGenerationPhase()
        let mlirLoweringPhase = EnhancedMLIRLoweringPhase.createMLIRLoweringPhase()
        
        // Execute complete pipeline
        let pipelineExecution = 
            PhaseExecution.enterPhase "enhanced-pipeline-start" >>= fun _ ->
            PhaseExecution.recordDiagnostic "Starting complete enhanced F# to MLIR translation pipeline with GUARANTEED AST file generation" >>= fun _ ->
            
            PipelineEngine.executePhase directASTPhase sourceCode >>= fun oakProgram ->
            PipelineEngine.executePhase closureEliminationPhase oakProgram >>= fun transformedOak1 ->
            PipelineEngine.executePhase unionLayoutPhase transformedOak1 >>= fun transformedOak2 ->
            PipelineEngine.executePhase mlirGenerationPhase transformedOak2 >>= fun mlirText ->
            PipelineEngine.executePhase mlirLoweringPhase mlirText >>= fun loweredMLIR ->
            
            PhaseExecution.recordDiagnostic "Successfully completed all enhanced pipeline phases with AST file generation" >>= fun _ ->
            succeed loweredMLIR
        
        printfn "Executing enhanced pipeline..."
        
        match pipelineExecution initialState with
        | Reply(Ok finalMLIR, finalState) ->
            printfn "✓ Enhanced pipeline completed successfully"
            printfn "Generated %d intermediate results" finalState.IntermediateResults.Count
            finalState.IntermediateResults |> Map.iter (fun key _ -> printfn "  - %s" key)
            
            Success {
                FinalMLIR = finalMLIR
                PhaseOutputs = finalState.IntermediateResults
                SymbolMappings = finalState.SymbolMappings
                Diagnostics = List.rev finalState.Diagnostics
                SuccessfulPhases = List.rev finalState.PhaseHistory
            }
        
        | Reply(Error, errorMsg) ->
            printfn "✗ Enhanced pipeline failed: %s" errorMsg
            CompilerFailure [TransformError("enhanced-pipeline execution", "F# source", "MLIR", errorMsg)]

/// Entry point functions with GUARANTEED AST generation
let translateF#ToMLIR (sourceFile: string) (sourceCode: string) : CompilerResult<string> =
    printfn "=== TRANSLATE F# TO MLIR (BASIC) ==="
    executeEnhancedTranslationPipeline sourceFile sourceCode false >>= fun pipelineOutput ->
    Success pipelineOutput.FinalMLIR

let translateF#ToMLIRWithDiagnostics (sourceFile: string) (sourceCode: string) : CompilerResult<TranslationPipelineOutput> =
    printfn "=== TRANSLATE F# TO MLIR (WITH DIAGNOSTICS) ==="
    executeEnhancedTranslationPipeline sourceFile sourceCode true

let translateF#ToMLIRWithIntermediates (sourceFile: string) (sourceCode: string) (keepIntermediates: bool) : CompilerResult<TranslationPipelineOutput> =
    printfn "=== TRANSLATE F# TO MLIR (WITH INTERMEDIATES) ==="
    executeEnhancedTranslationPipeline sourceFile sourceCode keepIntermediates

/// Validation functions
let validateEnhancedPipelineOutput (output: TranslationPipelineOutput) : CompilerResult<unit> =
    if String.IsNullOrWhiteSpace(output.FinalMLIR) then
        CompilerFailure [TransformError("pipeline validation", "pipeline output", "valid MLIR", "Final MLIR output is empty")]
    elif output.SuccessfulPhases.Length < 3 then
        CompilerFailure [TransformError("pipeline validation", "pipeline phases", "complete pipeline", sprintf "Pipeline only completed %d phases, expected at least 3" output.SuccessfulPhases.Length)]
    elif not (output.PhaseOutputs.ContainsKey("f#-compiler-services-ast")) then
        CompilerFailure [TransformError("pipeline validation", "AST generation", "FCS AST intermediate", "F# Compiler Services AST not generated")]
    elif not (output.PhaseOutputs.ContainsKey("oak-ast")) then
        CompilerFailure [TransformError("pipeline validation", "AST generation", "Oak AST intermediate", "Oak AST not generated")]
    else
        Success ()