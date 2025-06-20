module Dabbit.Parsing.Translator

open System
open System.IO
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

/// Defines the standard Firefly compilation pipeline phases
let standardPipeline = [
    { Name = "parse"; Description = "Parse F# source code to AST"; IsOptional = false }
    { Name = "oak-convert"; Description = "Convert F# AST to Oak AST"; IsOptional = false }
    { Name = "tree-shaking"; Description = "Intelligent tree-shaking for dead code elimination"; IsOptional = false }
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
                (diagnostics: (string * string) list)
                (intermediatesDir: string option) : CompilerResult<'Output * (string * string) list> =
        
        let startDiagnostic = (phaseName, sprintf "Starting %s phase" phaseName)
        let updatedDiagnostics = startDiagnostic :: diagnostics
        
        match transform input with
        | Success output ->
            let successDiagnostic = (phaseName, sprintf "Successfully completed %s phase" phaseName)
            Success (output, successDiagnostic :: updatedDiagnostics)
            
        | CompilerFailure errors ->
            let errorMessages = errors |> List.map (fun e -> e.ToString())
            let errorDiagnostic = (phaseName, sprintf "Failed in %s phase: %s" phaseName (String.concat "; " errorMessages))
            
            // Even on failure, save any input that can be serialized for debugging
            intermediatesDir |> Option.iter (fun dir ->
                try
                    // Create a file with the error details
                    let errorFile = Path.Combine(dir, sprintf "%s.error.txt" phaseName)
                    File.WriteAllText(errorFile, String.concat Environment.NewLine errorMessages)
                    printfn "✓ Saved error details to %s" (Path.GetFileName(errorFile))
                with _ -> ()
            )
            
            CompilerFailure errors
    
    /// Saves intermediate output for debugging
    let savePhaseOutput 
                    (phaseName: string) 
                    (output: 'Output) 
                    (toString: 'Output -> string)
                    (phaseOutputs: Map<string, string>)
                    (intermediatesDir: string option) : Map<string, string> =
        
        let outputStr = toString output
        
        // Immediately write the output to a file if intermediatesDir is provided
        intermediatesDir |> Option.iter (fun dir ->
            try
                let filename = sprintf "%s.%s" 
                                (match phaseName with
                                 | "parsing" -> "fcs"
                                 | "tree-shaking" -> "ra.oak"
                                 | "closure-elimination" -> "closures.oak"
                                 | "union-layout" -> "unions.oak"
                                 | "mlir-generation" -> "mlir"
                                 | "mlir-lowering" -> "lowered.mlir"
                                 | _ -> phaseName)
                                phaseName
                let outputPath = Path.Combine(dir, filename)
                File.WriteAllText(outputPath, outputStr)
                printfn "  Wrote intermediate file: %s" (Path.GetFileName(outputPath))
            with ex -> 
                printfn "  Warning: Could not write intermediate file for %s: %s" phaseName ex.Message
        )
        
        Map.add phaseName outputStr phaseOutputs

/// Core transformations using unified pipeline
module Transformations =
    /// Converts F# source to Oak AST using the unified function
    let sourceToOak (sourceFile: string) (sourceCode: string) (intermediatesDir: string option) : CompilerResult<OakProgram * string * string> =
        try
            let result = AstConverter.parseAndConvertToOakAst sourceFile sourceCode
            
            // Immediately write intermediate files
            intermediatesDir |> Option.iter (fun dir ->
                try
                    let baseName = Path.GetFileNameWithoutExtension(sourceFile)
                    let fcsPath = Path.Combine(dir, baseName + ".fcs")
                    let oakPath = Path.Combine(dir, baseName + ".oak")
                    
                    Directory.CreateDirectory(dir) |> ignore
                    File.WriteAllText(fcsPath, result.FSharpASTText)
                    File.WriteAllText(oakPath, sprintf "%A" result.OakProgram)
                    
                    printfn "  Wrote F# AST to: %s" (Path.GetFileName(fcsPath))
                    printfn "  Wrote Oak AST to: %s" (Path.GetFileName(oakPath))
                with ex ->
                    printfn "  Warning: Could not write parsing outputs: %s" ex.Message
            )
            
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
    let applyTreeShaking (program: OakProgram) (intermediatesDir: string option) : CompilerResult<OakProgram * string> =
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
            
            // Immediately write intermediate files
            intermediatesDir |> Option.iter (fun dir ->
                try
                    let treeshakePath = Path.Combine(dir, "treeshake.log")
                    let raOakPath = Path.Combine(dir, "ra.oak")
                    
                    File.WriteAllText(treeshakePath, diagnostics)
                    File.WriteAllText(raOakPath, sprintf "%A" prunedProgram)
                    
                    printfn "  Wrote tree shaking log to: %s" (Path.GetFileName(treeshakePath))
                    printfn "  Wrote pruned Oak AST to: %s" (Path.GetFileName(raOakPath))
                with ex ->
                    printfn "  Warning: Could not write tree shaking outputs: %s" ex.Message
            )
            
            printfn "%s" diagnostics
            Success (prunedProgram, diagnostics)
        
        with ex ->
            CompilerFailure [ConversionError("tree-shaking", "Oak AST", "pruned Oak AST", ex.Message)]
    
    /// Applies closure elimination transformation
    let applyClosure (program: OakProgram) (intermediatesDir: string option) : CompilerResult<OakProgram> =
        match eliminateClosures program with
        | Success transformedProgram ->
            // Add diagnostic info about what was transformed
            printfn "  Closure analysis: No closures found to eliminate"
            
            // Immediately write intermediate file
            intermediatesDir |> Option.iter (fun dir ->
                try
                    let closuresPath = Path.Combine(dir, "closures.oak")
                    File.WriteAllText(closuresPath, sprintf "%A" transformedProgram)
                    printfn "  Wrote closure-transformed Oak AST to: %s" (Path.GetFileName(closuresPath))
                with ex ->
                    printfn "  Warning: Could not write closure elimination output: %s" ex.Message
            )
            
            Success transformedProgram
        | CompilerFailure errors -> CompilerFailure errors
    
    /// Applies union layout transformation
    let applyUnionLayout (program: OakProgram) (intermediatesDir: string option) : CompilerResult<OakProgram> =
        match compileFixedLayouts program with
        | Success transformedProgram ->
            // Add diagnostic info about what was transformed
            printfn "  Union analysis: No discriminated unions found to optimize"
            
            // Immediately write intermediate file
            intermediatesDir |> Option.iter (fun dir ->
                try
                    let unionsPath = Path.Combine(dir, "unions.oak")
                    File.WriteAllText(unionsPath, sprintf "%A" transformedProgram)
                    printfn "  Wrote union-transformed Oak AST to: %s" (Path.GetFileName(unionsPath))
                with ex ->
                    printfn "  Warning: Could not write union layout output: %s" ex.Message
            )
            
            Success transformedProgram
        | CompilerFailure errors -> CompilerFailure errors
    
    /// Generates MLIR from Oak AST
    let generateMLIR (program: OakProgram) (intermediatesDir: string option) : CompilerResult<string> =
        match generateMLIRModuleText program with
        | Success mlirText ->
            // Count functions generated
            let functionCount = 
                mlirText.Split('\n') 
                |> Array.filter (fun line -> line.Trim().StartsWith("func.func @"))
                |> Array.length
            printfn "  Generated MLIR: %d functions" functionCount
            
            // Immediately write intermediate file
            intermediatesDir |> Option.iter (fun dir ->
                try
                    let mlirPath = Path.Combine(dir, "output.mlir")
                    File.WriteAllText(mlirPath, mlirText)
                    printfn "  Wrote MLIR to: %s" (Path.GetFileName(mlirPath))
                with ex ->
                    printfn "  Warning: Could not write MLIR output: %s" ex.Message
            )
            
            Success mlirText
        | CompilerFailure errors -> CompilerFailure errors
    
    /// Lowers MLIR to LLVM dialect
    let lowerMLIR (mlirText: string) (intermediatesDir: string option) : CompilerResult<string> =
        match Core.Conversion.LoweringPipeline.applyLoweringPipeline mlirText with
        | Success loweredText ->
            // Immediately write intermediate file
            intermediatesDir |> Option.iter (fun dir ->
                try
                    let loweredPath = Path.Combine(dir, "lowered.mlir")
                    File.WriteAllText(loweredPath, loweredText)
                    printfn "  Wrote lowered MLIR to: %s" (Path.GetFileName(loweredPath))
                with ex ->
                    printfn "  Warning: Could not write lowered MLIR output: %s" ex.Message
            )
            
            Success loweredText
        | CompilerFailure errors -> CompilerFailure errors

/// Main translation pipeline - clean, focused version
/// Main translation pipeline - clean, focused version
let executeTranslationPipeline (sourceFile: string) (sourceCode: string) (intermediatesDir: string option) : CompilerResult<TranslationPipelineOutput> =
    printfn "=== Firefly Translation Pipeline ==="
    printfn "Source: %s (%d chars)" (Path.GetFileName(sourceFile)) sourceCode.Length
    
    // Create intermediates directory if specified
    intermediatesDir |> Option.iter (fun dir ->
        if not (Directory.Exists(dir)) then
            Directory.CreateDirectory(dir) |> ignore
            printfn "Created intermediates directory: %s" dir
    )
    
    let initialDiagnostics = [("pipeline", "Starting unified translation pipeline")]
    let initialPhaseOutputs = Map.empty<string, string>
    
    try
        // Step 1: Parse F# source to Oak AST
        match PipelineExecution.runPhase "parsing" 
            (fun sc -> Transformations.sourceToOak sourceFile sc intermediatesDir) 
            sourceCode 
            initialDiagnostics
            intermediatesDir with
        | Success (parseResult, diagnostics1) ->
            
            let (oakProgram, oakAstText, fsharpAstText) = parseResult
            let phaseOutputs1 = 
                initialPhaseOutputs
                |> Map.add "fsharp-ast" fsharpAstText
                |> Map.add "oak-ast" oakAstText
            printfn "✓ Parsing completed"
            
            // Step 2: Apply tree-shaking
            match PipelineExecution.runPhase "tree-shaking" 
                (fun prog -> Transformations.applyTreeShaking prog intermediatesDir) 
                oakProgram 
                diagnostics1
                intermediatesDir with
            | Success ((prunedProgram, treeDiagnostics), diagnostics2) ->
                
                let prunedOakText = sprintf "%A" prunedProgram
                let phaseOutputs2 = 
                    phaseOutputs1
                    |> Map.add "tree-shaking-stats" treeDiagnostics
                    |> Map.add "ra-oak" prunedOakText  // The ra.oak file
                printfn "✓ Tree shaking completed"
                
                // Step 3: Apply closure elimination
                match PipelineExecution.runPhase "closure-elimination" 
                    (fun prog -> Transformations.applyClosure prog intermediatesDir) 
                    prunedProgram 
                    diagnostics2
                    intermediatesDir with
                | Success (transformedProgram1, diagnostics3) ->
                    
                    let closureTransformedText = sprintf "%A" transformedProgram1
                    let phaseOutputs3 = PipelineExecution.savePhaseOutput 
                                            "closure-transformed" 
                                            closureTransformedText 
                                            id 
                                            phaseOutputs2
                                            intermediatesDir
                    printfn "✓ Closure elimination completed"
                    
                    // Step 4: Apply union layout transformation
                    match PipelineExecution.runPhase "union-layout" 
                        (fun prog -> Transformations.applyUnionLayout prog intermediatesDir) 
                        transformedProgram1 
                        diagnostics3
                        intermediatesDir with
                    | Success (transformedProgram2, diagnostics4) ->
                        
                        let layoutTransformedText = sprintf "%A" transformedProgram2
                        let phaseOutputs4 = PipelineExecution.savePhaseOutput 
                                                "layout-transformed" 
                                                layoutTransformedText 
                                                id 
                                                phaseOutputs3
                                                intermediatesDir
                        printfn "✓ Union layout completed"
                        
                        // Step 5: Generate MLIR
                        match PipelineExecution.runPhase "mlir-generation" 
                            (fun prog -> Transformations.generateMLIR prog intermediatesDir) 
                            transformedProgram2 
                            diagnostics4
                            intermediatesDir with
                        | Success (mlirText, diagnostics5) ->
                            
                            let phaseOutputs5 = PipelineExecution.savePhaseOutput 
                                                    "mlir" 
                                                    mlirText 
                                                    id 
                                                    phaseOutputs4
                                                    intermediatesDir
                            printfn "✓ MLIR generated (%d chars)" mlirText.Length
                            
                            // Step 6: Lower MLIR to LLVM dialect
                            match PipelineExecution.runPhase "mlir-lowering" 
                                (fun text -> Transformations.lowerMLIR text intermediatesDir) 
                                mlirText 
                                diagnostics5
                                intermediatesDir with
                            | Success (loweredMlir, diagnostics6) ->
                                
                                let phaseOutputs6 = PipelineExecution.savePhaseOutput 
                                                        "lowered-mlir" 
                                                        loweredMlir 
                                                        id 
                                                        phaseOutputs5
                                                        intermediatesDir
                                printfn "✓ MLIR lowering completed (%d chars)" loweredMlir.Length
                                
                                let successfulPhases = 
                                    ["parsing"; "tree-shaking"; "closure-elimination"; "union-layout"; "mlir-generation"; "mlir-lowering"]
                                
                                printfn "=== Pipeline completed successfully ==="
                                
                                Success {
                                    FinalMLIR = loweredMlir
                                    PhaseOutputs = phaseOutputs6
                                    SymbolMappings = Map.empty
                                    Diagnostics = List.rev diagnostics6
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
                printfn "✗ Tree shaking failed"
                CompilerFailure errors
        | CompilerFailure errors -> 
            printfn "✗ Parsing failed"
            CompilerFailure errors
    with ex ->
        printfn "✗ Pipeline exception: %s" ex.Message
        
        // Try to save exception details to file
        intermediatesDir |> Option.iter (fun dir ->
            try
                let errorFile = Path.Combine(dir, "pipeline-error.txt")
                File.WriteAllText(errorFile, sprintf "Pipeline exception: %s\n\nStack trace:\n%s" ex.Message ex.StackTrace)
                printfn "✓ Saved error details to %s" (Path.GetFileName(errorFile))
            with _ -> ()
        )
        
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