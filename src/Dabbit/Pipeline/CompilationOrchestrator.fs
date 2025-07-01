module Dabbit.Pipeline.CompilationOrchestrator

open System
open System.IO
open FSharp.Compiler.Syntax
open FSharp.Compiler.CodeAnalysis
open Core.XParsec.Foundation
open Core.Utilities.IntermediateWriter
open Dabbit.Pipeline.CompilationTypes
open Dabbit.CodeGeneration.TypeMapping
open Dabbit.Bindings.SymbolRegistry.SymbolResolution
open Dabbit.Bindings.SymbolRegistry.Registry
open Dabbit.Bindings.PatternLibrary
open Dabbit.Integration.AlloyBindings
open Dabbit.Analysis.CompilationUnit
open Dabbit.Analysis.ReachabilityAnalyzer
open Dabbit.Analysis.DependencyGraphBuilder
open Dabbit.Pipeline.FCSPipeline
open Dabbit.Pipeline.LoweringPipeline
open Dabbit.Pipeline.OptimizationPipeline
open Dabbit.CodeGeneration.MLIRModuleGenerator

/// Initialize compilation state with all necessary components
let private initializeCompilationState (projectOptions: FSharpProjectOptions) (config: PipelineConfiguration) (intermediatesDir: string option) =
    let typeCtx = TypeContextBuilder.create()
    let symbolRegistry = 
        let registry = createRegistry alloyPatterns
        registerAlloySymbols registry |> ignore
        registry
    
    {
        Checker = FSharpChecker.Create(keepAssemblyContents = true)
        ProjectOptions = projectOptions
        TypeContext = typeCtx
        SymbolRegistry = symbolRegistry
        Configuration = config
        IntermediatesDirectory = intermediatesDir
    }

/// Extract symbols from all parsed ASTs and populate the symbol registry
let private populateSymbolRegistry (compilationUnits: (string * ParsedInput) list) (state: CompilationState) =
    let mutable updatedRegistry = state.SymbolRegistry
    
    for (filePath, ast) in compilationUnits do
        // Extract symbols using the existing function from SymbolRegistry
        let symbols = extractSymbolsFromParsedInput ast
        
        // Register each symbol in the registry
        for symbol in symbols do
            let updatedState = registerSymbol symbol updatedRegistry.State
            updatedRegistry <- { updatedRegistry with State = updatedState }
    
    { state with SymbolRegistry = updatedRegistry }

/// Build Dabbit compilation unit from FCS results
let private buildDabbitCompilationUnit (fcsUnits: (string * ParsedInput * FSharpCheckFileResults) list) =
    async {
        match fcsUnits with
        | [] -> return CompilerFailure [InternalError("buildDabbitCompilationUnit", "No compilation units provided", None)]
        | (mainFile, mainAst, _) :: _ ->
            // Create a file parser that returns already-parsed ASTs
            let fileParser (path: string) = async {
                match List.tryFind (fun (p, _, _) -> p = path) fcsUnits with
                | Some (_, ast, _) -> return Success ast
                | None -> return CompilerFailure [InternalError("buildDabbitCompilationUnit", sprintf "File not found: %s" path, Some "Referenced file was not in compilation units")]
            }
            
            // Build the compilation unit using existing infrastructure
            return! buildCompilationUnit fileParser mainFile mainAst
    }

/// Process all compilation units through analysis and transformation
let private processCompilationUnits (fcsUnits: (string * ParsedInput * FSharpCheckFileResults) list) (state: CompilationState) (progress: ProgressCallback) =
    async {
        progress SymbolExtraction "Extracting symbols from all compilation units"
        
        // Populate symbol registry with all symbols
        let compilationInputs = fcsUnits |> List.map (fun (path, ast, _) -> (path, ast))
        let stateWithSymbols = populateSymbolRegistry compilationInputs state
        
        // Build Dabbit compilation unit
        progress ReachabilityAnalysis "Building compilation unit for analysis"
        let! compilationUnitResult = buildDabbitCompilationUnit fcsUnits
        
        match compilationUnitResult with
        | CompilerFailure errors -> return CompilerFailure errors
        | Success compilationUnit ->
            // Perform reachability analysis
            progress ReachabilityAnalysis "Performing global reachability analysis"
            let analysis = analyzeCompilationUnit compilationUnit
            
            // Report statistics
            progress ReachabilityAnalysis (sprintf "Analysis complete: %d/%d symbols reachable (%.1f%% eliminated)" 
                analysis.GlobalReachability.Statistics.ReachableSymbols
                analysis.GlobalReachability.Statistics.TotalSymbols
                ((float analysis.GlobalReachability.Statistics.EliminatedSymbols / float analysis.GlobalReachability.Statistics.TotalSymbols) * 100.0))
            
            // Write intermediate files if configured
            if state.Configuration.PreserveIntermediateASTs && state.IntermediatesDirectory.IsSome then
                writeIntermediateASTs state.IntermediatesDirectory.Value analysis.PrunedAsts
            
            // Build processing result
            let processedUnits = 
                analysis.PrunedAsts
                |> Map.map (fun filePath prunedAst ->
                    let originalAst = fcsUnits |> List.find (fun (p, _, _) -> p = filePath) |> (fun (_, ast, _) -> ast)
                    let symbols = analysis.Unit.SourceFiles.[filePath].DefinedSymbols
                    let reachableSymbols = Map.tryFind filePath analysis.PerFileReachable |> Option.defaultValue Set.empty
                    
                    {
                        FilePath = filePath
                        OriginalAST = originalAst
                        TransformedAST = prunedAst
                        Symbols = symbols
                        ReachableSymbols = reachableSymbols
                    })
            
            return Success {
                ProcessedUnits = processedUnits
                CompilationAnalysis = analysis
                TypeContext = stateWithSymbols.TypeContext
                SymbolRegistry = stateWithSymbols.SymbolRegistry
                GlobalReachability = analysis.GlobalReachability
            }
    }

/// Generate MLIR from processed results
let private generateMLIR (result: MultiFileProcessingResult) (state: CompilationState) (progress: ProgressCallback) =
    progress MLIRGeneration "Generating MLIR from transformed ASTs"
    
    // For now, generate MLIR from the main file's transformed AST
    let mainFile = result.CompilationAnalysis.Unit.MainFile
    match Map.tryFind mainFile result.ProcessedUnits with
    | None -> CompilerFailure [InternalError("generateMLIR", "Main file not found in processed units", None)]
    | Some processedUnit ->
        try
            let mlirText = generateModuleFromAST processedUnit.TransformedAST result.TypeContext result.SymbolRegistry
            
            // Write MLIR if configured
            if state.IntermediatesDirectory.IsSome then
                let mlirPath = Path.Combine(state.IntermediatesDirectory.Value, Path.GetFileNameWithoutExtension(mainFile) + ".mlir")
                File.WriteAllText(mlirPath, mlirText)
                progress MLIRGeneration (sprintf "Wrote MLIR to %s" mlirPath)
            
            Success mlirText
        with ex ->
            CompilerFailure [MLIRGenerationError("generateMLIR", ex.Message, Some mainFile)]

/// Main compilation pipeline orchestration
let compileProject (inputPath: string) 
                   (outputPath: string) 
                   (projectOptions: FSharpProjectOptions) 
                   (config: PipelineConfiguration) 
                   (intermediatesDir: string option)
                   (progress: ProgressCallback) =
    async {
        let startTime = DateTime.UtcNow
        progress Initialization "Initializing compilation pipeline"
        
        // Initialize compilation state
        let state = initializeCompilationState projectOptions config intermediatesDir
        
        // Parse and check all files
        progress Parsing "Parsing source files"
        let parsingOptions, _ = state.Checker.GetParsingOptionsFromProjectOptions(projectOptions)
        let! parseResults = 
            projectOptions.SourceFiles
            |> Array.map (fun file -> async {
                let! sourceText = File.ReadAllTextAsync(file) |> Async.AwaitTask
                let! parseResult = state.Checker.ParseFile(file, FSharp.Compiler.Text.SourceText.ofString sourceText, parsingOptions)
                return (file, parseResult)
            })
            |> Async.Parallel
        
        // Check for parse errors
        let parseErrors = 
            parseResults 
            |> Array.collect (fun (file, result) -> 
                if result.ParseHadErrors then 
                    result.Diagnostics |> Array.map (fun d -> 
                        SyntaxError({ Line = d.StartLine; Column = d.StartColumn; File = file; Offset = 0 }, d.Message, ["parsing"]))
                else [||])
        
        if parseErrors.Length > 0 then
            return {
                Success = false
                MLIROutput = None
                LLVMOutput = None
                Diagnostics = Array.toList parseErrors
                Statistics = {
                    TotalFiles = projectOptions.SourceFiles.Length
                    TotalSymbols = 0
                    ReachableSymbols = 0
                    EliminatedSymbols = 0
                    CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                }
            }
        else
            // Type check all files
            progress TypeChecking "Type checking all files"
            let! checkResults =
                parseResults
                |> Array.map (fun (file, parseResult) -> async {
                    let! sourceText = File.ReadAllTextAsync(file) |> Async.AwaitTask
                    let sourceTextObj = FSharp.Compiler.Text.SourceText.ofString sourceText
                    let fileVersion = 0 // Version number for incremental compilation
                    let! checkAnswer = state.Checker.CheckFileInProject(parseResult, file, fileVersion, sourceTextObj, projectOptions)
                    match checkAnswer with
                    | FSharpCheckFileAnswer.Succeeded checkResult ->
                        return (file, parseResult.ParseTree, checkResult)
                    | FSharpCheckFileAnswer.Aborted ->
                        return failwith (sprintf "Type checking aborted for file: %s" file)
                })
                |> Async.Sequential
            
            let fcsUnits = Array.toList checkResults
            
            // Process compilation units
            let! processingResult = processCompilationUnits fcsUnits state progress
            
            match processingResult with
            | CompilerFailure errors ->
                return {
                    Success = false
                    MLIROutput = None
                    LLVMOutput = None
                    Diagnostics = errors
                    Statistics = {
                        TotalFiles = projectOptions.SourceFiles.Length
                        TotalSymbols = 0
                        ReachableSymbols = 0
                        EliminatedSymbols = 0
                        CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                    }
                }
            | Success multiFileResult ->
                // Generate MLIR
                match generateMLIR multiFileResult state progress with
                | CompilerFailure errors ->
                    return {
                        Success = false
                        MLIROutput = None
                        LLVMOutput = None
                        Diagnostics = errors
                        Statistics = {
                            TotalFiles = projectOptions.SourceFiles.Length
                            TotalSymbols = multiFileResult.GlobalReachability.Statistics.TotalSymbols
                            ReachableSymbols = multiFileResult.GlobalReachability.Statistics.ReachableSymbols
                            EliminatedSymbols = multiFileResult.GlobalReachability.Statistics.EliminatedSymbols
                            CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                        }
                    }
                | Success mlirText ->
                    // Lower to LLVM
                    progress LLVMGeneration "Lowering MLIR to LLVM IR"
                    let loweringResult = 
                        let llvmOutput = {
                            ModuleName = Path.GetFileNameWithoutExtension(inputPath)
                            LLVMIRText = mlirText  // This contains MLIR text that will be lowered
                            Metadata = Map.empty
                            OptimizationLevel = Zero
                        }
                        let passes = createOptimizationPipeline Zero
                        match optimizeLLVMIR llvmOutput passes with
                        | CompilerFailure errors -> CompilerFailure errors
                        | Success optimized -> translateToLLVMIR optimized.LLVMIRText intermediatesDir
                    
                    match loweringResult with
                    | CompilerFailure errors ->
                        return {
                            Success = false
                            MLIROutput = Some mlirText
                            LLVMOutput = None
                            Diagnostics = errors
                            Statistics = {
                                TotalFiles = projectOptions.SourceFiles.Length
                                TotalSymbols = multiFileResult.GlobalReachability.Statistics.TotalSymbols
                                ReachableSymbols = multiFileResult.GlobalReachability.Statistics.ReachableSymbols
                                EliminatedSymbols = multiFileResult.GlobalReachability.Statistics.EliminatedSymbols
                                CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                            }
                        }
                    | Success llvmIR ->
                        progress Finalization "Compilation complete"
                        return {
                            Success = true
                            MLIROutput = Some mlirText
                            LLVMOutput = Some llvmIR
                            Diagnostics = []
                            Statistics = {
                                TotalFiles = projectOptions.SourceFiles.Length
                                TotalSymbols = multiFileResult.GlobalReachability.Statistics.TotalSymbols
                                ReachableSymbols = multiFileResult.GlobalReachability.Statistics.ReachableSymbols
                                EliminatedSymbols = multiFileResult.GlobalReachability.Statistics.EliminatedSymbols
                                CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                            }
                        }
    }