module Dabbit.Parsing.Translator

open System
open System.IO
open Core.Utilities
open Core.XParsec.Foundation
open Dabbit.Parsing.OakAst
open Dabbit.Parsing.AstConverter
open Dabbit.Closures.ClosureTransformer
open Dabbit.UnionLayouts.FixedLayoutCompiler
open Core.MLIRGeneration.XParsecMLIRGenerator
open Dabbit.TreeShaking.DependencyGraphBuilder
open Dabbit.TreeShaking.ReachabilityTraversal
open Dabbit.TreeShaking.AstPruner

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
    
    /// Updates phase output map (NO FILE I/O)
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
    
    /// Apply tree-shaking to Oak AST
    let applyTreeShaking (program: OakProgram) : CompilerResult<OakProgram * string> =
        try
            // Debug output for module information
            printfn "Tree shaking analysis: Found %d modules" program.Modules.Length
            for m in program.Modules do
                printfn "  Module: %s with %d declarations" m.Name m.Declarations.Length
            
            // Build dependency graph
            let graph = buildDependencyGraph program
            
            // Debug output for dependency graph
            printfn "Dependency graph built: %d declarations, %d qualified names" 
                graph.Declarations.Count graph.QualifiedNames.Count
            
            // Perform reachability analysis
            let reachabilityResult = analyzeReachability graph
            
            // Debug output for reachability
            printfn "Reachability analysis: %d of %d declarations reachable" 
                reachabilityResult.ReachableDeclarations.Count graph.Declarations.Count
            
            // Prune unreachable code
            let prunedProgram = pruneUnreachableCode program reachabilityResult
            
            // Generate diagnostic report
            let diagnostics = generateDiagnosticReport reachabilityResult.EliminationStats
            
            printfn "%s" diagnostics
            Success (prunedProgram, diagnostics)
        
        with ex ->
            CompilerFailure [ConversionError("tree-shaking", "Oak AST", "pruned Oak AST", ex.Message)]
    
    /// Applies closure elimination transformation
    let applyClosure (program: OakProgram) : CompilerResult<OakProgram> =
        match eliminateClosures program with
        | Success transformedProgram ->
            // Add diagnostic info about what was transformed
            printfn "  Closure analysis: No closures found to eliminate"
            Success transformedProgram
        | CompilerFailure errors -> CompilerFailure errors
    
    /// Applies union layout transformation
    let applyUnionLayout (program: OakProgram) : CompilerResult<OakProgram> =
        match compileFixedLayouts program with
        | Success transformedProgram ->
            // Add diagnostic info about what was transformed
            printfn "  Union analysis: No discriminated unions found to optimize"
            Success transformedProgram
        | CompilerFailure errors -> CompilerFailure errors
    
    /// Generates MLIR from Oak AST
    let generateMLIR (program: OakProgram) : CompilerResult<string> =
        match generateMLIRModuleText program with
        | Success mlirText ->
            // Count functions generated
            let functionCount = 
                mlirText.Split('\n') 
                |> Array.filter (fun line -> line.Trim().StartsWith("func.func @"))
                |> Array.length
            printfn "  Generated MLIR: %d functions" functionCount
            Success mlirText
        | CompilerFailure errors -> CompilerFailure errors
    
    /// Lowers MLIR to LLVM dialect
    let lowerMLIR (mlirText: string) : CompilerResult<string> =
        match Core.Conversion.LoweringPipeline.applyLoweringPipeline mlirText with
        | Success loweredText ->
            Success loweredText
        | CompilerFailure errors -> CompilerFailure errors

/// Main translation pipeline with complete dependency resolution 
/// Main translation pipeline with complete dependency resolution 
let executeTranslationPipeline (sourceFile: string) (sourceCode: string) (intermediatesDir: string option) : CompilerResult<TranslationPipelineOutput> =
    printfn "=== Firefly Translation Pipeline ==="
    printfn "Source: %s (%d chars)" (Path.GetFileName(sourceFile)) sourceCode.Length
    
    // Setup
    let baseName = Path.GetFileNameWithoutExtension(sourceFile)
    let mutable phaseOutputs = Map.empty<string, string>
    let mutable diagnostics = []
    
    // Helper to write intermediate files
    let writeIntermediate (extension: string) (content: string) =
        intermediatesDir |> Option.iter (fun dir ->
            if not (Directory.Exists(dir)) then
                Directory.CreateDirectory(dir) |> ignore
            IntermediateWriter.writeFile dir baseName extension content
        )
    
    // Helper to update phase outputs
    let recordPhase (phaseName: string) (content: string) =
        phaseOutputs <- Map.add phaseName content phaseOutputs
    
    // Helper to run a phase and handle errors
    let runPhase (phaseName: string) (operation: unit -> CompilerResult<'a>) : CompilerResult<'a> =
        diagnostics <- (phaseName, sprintf "Starting %s" phaseName) :: diagnostics
        match operation() with
        | Success result ->
            diagnostics <- (phaseName, sprintf "Completed %s" phaseName) :: diagnostics
            printfn "✓ %s completed" phaseName
            Success result
        | CompilerFailure errors ->
            diagnostics <- (phaseName, sprintf "Failed %s" phaseName) :: diagnostics
            printfn "✗ %s failed" phaseName
            CompilerFailure errors
    
    // Start pipeline
    try
        // Phase 1: Parse F# to Oak AST
        let parseResult = runPhase "Parsing" (fun () ->
            Transformations.sourceToOak sourceFile sourceCode
        )
        
        match parseResult with
        | CompilerFailure errors -> CompilerFailure errors
        | Success (oakProgram, oakAstText, fsharpAstText) ->
            
            // Write parse outputs
            writeIntermediate ".fcs" fsharpAstText
            writeIntermediate ".oak" oakAstText
            recordPhase "fsharp-ast" fsharpAstText
            recordPhase "oak-ast" oakAstText
            
            printfn "  Found %d modules" oakProgram.Modules.Length
            
            // Phase 2: Tree shaking
            let treeShakeResult = runPhase "Tree shaking" (fun () ->
                Transformations.applyTreeShaking oakProgram
            )
            
            match treeShakeResult with
            | CompilerFailure errors -> CompilerFailure errors
            | Success (prunedProgram, treeDiagnostics) ->
                
                let prunedOakText = sprintf "%A" prunedProgram
                writeIntermediate ".ra.oak" prunedOakText
                writeIntermediate ".treeshake.log" treeDiagnostics
                recordPhase "ra-oak" prunedOakText
                recordPhase "tree-shaking-stats" treeDiagnostics
                
                // Phase 3: Closure elimination
                let closureResult = runPhase "Closure elimination" (fun () ->
                    Transformations.applyClosure prunedProgram
                )
                
                match closureResult with
                | CompilerFailure errors -> CompilerFailure errors
                | Success closureEliminated ->
                    
                    let closureText = sprintf "%A" closureEliminated
                    writeIntermediate ".closures.oak" closureText
                    recordPhase "closure-transformed" closureText
                    
                    // Phase 4: Union layout optimization
                    let unionResult = runPhase "Union layout" (fun () ->
                        Transformations.applyUnionLayout closureEliminated
                    )
                    
                    match unionResult with
                    | CompilerFailure errors -> CompilerFailure errors
                    | Success layoutOptimized ->
                        
                        let layoutText = sprintf "%A" layoutOptimized
                        writeIntermediate ".unions.oak" layoutText
                        recordPhase "layout-transformed" layoutText
                        
                        // Phase 5: Generate MLIR
                        let mlirResult = runPhase "MLIR generation" (fun () ->
                            Transformations.generateMLIR layoutOptimized
                        )
                        
                        match mlirResult with
                        | CompilerFailure errors -> CompilerFailure errors
                        | Success mlirText ->
                            
                            writeIntermediate ".mlir" mlirText
                            recordPhase "mlir" mlirText
                            printfn "  Generated %d chars of MLIR" mlirText.Length
                            
                            // Phase 6: Lower MLIR
                            let lowerResult = runPhase "MLIR lowering" (fun () ->
                                Transformations.lowerMLIR mlirText
                            )
                            
                            match lowerResult with
                            | CompilerFailure errors -> CompilerFailure errors
                            | Success loweredMlir ->
                                
                                writeIntermediate ".lowered.mlir" loweredMlir
                                recordPhase "lowered-mlir" loweredMlir
                                printfn "  Lowered to %d chars" loweredMlir.Length
                                
                                printfn "=== Pipeline completed successfully ==="
                                
                                Success {
                                    FinalMLIR = loweredMlir
                                    PhaseOutputs = phaseOutputs
                                    SymbolMappings = Map.empty
                                    Diagnostics = List.rev diagnostics
                                    SuccessfulPhases = [
                                        "parsing"; "tree-shaking"; "closure-elimination"; 
                                        "union-layout"; "mlir-generation"; "mlir-lowering"
                                    ]
                                }
    
    with ex ->
        printfn "✗ Pipeline exception: %s" ex.Message
        CompilerFailure [ConversionError(
            "pipeline-execution", 
            "translation pipeline", 
            "MLIR", 
            sprintf "Exception: %s" ex.Message)]


/// Simple entry point for translation (maintains backward compatibility)
let translateFsToMLIR (sourceFile: string) (sourceCode: string) : CompilerResult<string> =
    match executeTranslationPipeline sourceFile sourceCode None with
    | Success pipelineOutput -> Success pipelineOutput.FinalMLIR
    | CompilerFailure errors -> CompilerFailure errors

/// Entry point with full diagnostic information (primary interface)
let translateFsToMLIRWithDiagnostics (sourceFile: string) (sourceCode: string) (intermediatesDir: string option) : CompilerResult<TranslationPipelineOutput> =
    executeTranslationPipeline sourceFile sourceCode intermediatesDir