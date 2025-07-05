module Dabbit.Pipeline.CompilationOrchestrator

open System
open System.IO
open Core.XParsec.Foundation
open Core.Utilities.IntermediateWriter
open Core.FCSIngestion.ProjectOptionsLoader
open Core.FCSIngestion.ProjectProcessor
open Core.FCSIngestion.SymbolCollector
open Dabbit.Analysis.ReachabilityAnalyzer
open Dabbit.Pipeline.CompilationTypes

/// Progress callback for reporting compilation stages
type ProgressCallback = CompilationPhase -> string -> unit

and CompilationPhase =
    | ProjectLoading
    | FCSProcessing
    | SymbolCollection
    | ReachabilityAnalysis
    | IntermediateGeneration
    // Future phases (placeholders):
    | ASTTransformation     // TODO: AST transformations and optimizations
    | MLIRGeneration       // TODO: MLIR code generation
    | LLVMGeneration       // TODO: LLVM IR generation
    | NativeCompilation    // TODO: Native binary compilation

/// Result of the compilation process
type CompilationResult = {
    Success: bool
    IntermediatesGenerated: bool
    ReachabilityReport: string option
    Diagnostics: FireflyError list
    Statistics: CompilationStatistics
}

and CompilationStatistics = {
    TotalFiles: int
    TotalSymbols: int
    ReachableSymbols: int
    EliminatedSymbols: int
    CompilationTimeMs: float
}

/// Write intermediate files for debugging and verification
let private writeIntermediateFiles 
    (intermediatesDir: string)
    (processedProject: ProcessedProject)
    (symbolCollection: SymbolCollectionResult) 
    (reachability: ReachabilityResult) : unit =
    
    // Create directory structure
    let analysisDir = Path.Combine(intermediatesDir, "analysis")
    Directory.CreateDirectory(analysisDir) |> ignore
    
    // NOTE: Initial AST already written immediately after FCS processing
    
    // Write symbol collection results
    let symbolsPath = Path.Combine(analysisDir, "symbols.analysis")
    let symbolsContent = sprintf "%A" symbolCollection
    writeIntermediateFile symbolsPath symbolsContent
    
    // Write reachability analysis
    let reachabilityPath = Path.Combine(analysisDir, "reachability.analysis")
    let reachabilityContent = sprintf "%A" reachability
    writeIntermediateFile reachabilityPath reachabilityContent
    
    // Write reachability report (human readable)
    let reportPath = Path.Combine(analysisDir, "reachability.report")
    let reportContent = generateReport reachability
    writeIntermediateFile reportPath reportContent
    
    // Write pruned AST (placeholder - will contain only reachable symbols)
    let fcsDir = Path.Combine(intermediatesDir, "fcs")
    let prunedAstPath = Path.Combine(fcsDir, "project.pruned.ast")
    let prunedAstContent = "// TODO: Generate pruned AST containing only reachable symbols"
    writeIntermediateFile prunedAstPath prunedAstContent
    
    printfn "✅ Analysis intermediate files written"

/// Validate zero-allocation requirements and report violations
let validateZeroAllocationRequirements (reachability: ReachabilityResult) : CompilerResult<unit> =
    validateZeroAllocation reachability

/// Main compilation orchestrator - currently handles FCS ingestion through reachability analysis
let compile 
    (inputPath: string) 
    (intermediatesDir: string option)
    (progress: ProgressCallback) : Async<CompilationResult> =
    
    async {
        let startTime = DateTime.UtcNow
        
        try
            // Phase 1: Load project options
            progress ProjectLoading "Loading project configuration"
            let checker = FSharp.Compiler.CodeAnalysis.FSharpChecker.Create(keepAssemblyContents=true)
            let! projectResult = loadProject inputPath checker
            
            match projectResult with
            | Error errorMsg ->
                return {
                    Success = false
                    IntermediatesGenerated = false
                    ReachabilityReport = None
                    Diagnostics = [InternalError("ProjectLoading", errorMsg, None)]
                    Statistics = {
                        TotalFiles = 0
                        TotalSymbols = 0
                        ReachableSymbols = 0
                        EliminatedSymbols = 0
                        CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                    }
                }
            | Ok projectOptions ->
                
                // Phase 2: Process project through FCS
                progress FCSProcessing "Processing project with F# Compiler Services"
                let! processResult = processProject projectOptions checker
                
                match processResult with
                | Error processingError ->
                    let diagnostics = 
                        processingError.CriticalErrors
                        |> Array.map (fun diag ->
                            TypeCheckError("FCS", diag.Message, 
                                { Line = diag.StartLine; Column = diag.StartColumn; 
                                  File = diag.FileName; Offset = 0 }))
                        |> Array.toList
                    
                    return {
                        Success = false
                        IntermediatesGenerated = false
                        ReachabilityReport = None
                        Diagnostics = diagnostics
                        Statistics = {
                            TotalFiles = projectOptions.SourceFiles.Length
                            TotalSymbols = 0
                            ReachableSymbols = 0
                            EliminatedSymbols = 0
                            CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                        }
                    }
                | Ok processedProject ->
                    
                    // Write initial AST immediately after FCS processing
                    match intermediatesDir with
                    | Some dir ->
                        let projectName = Path.GetFileNameWithoutExtension inputPath
                        let initialAstPath = Path.Combine(dir, sprintf "%s.initial.ast" projectName)

                        let initialAstContent = 
                            let assemblyContents = processedProject.CheckResults.AssemblyContents
                            let sb = System.Text.StringBuilder()
                            
                            sb.AppendLine("=== TYPED AST CONTENT ===") |> ignore
                            sb.AppendLine(sprintf "Files: %d" assemblyContents.ImplementationFiles.Length) |> ignore
                            sb.AppendLine() |> ignore
                            
                            for implFile in assemblyContents.ImplementationFiles do
                                sb.AppendLine(sprintf "FILE: %s" implFile.FileName) |> ignore
                                sb.AppendLine(sprintf "Qualified Name: %s" implFile.QualifiedName) |> ignore
                                sb.AppendLine(sprintf "Declarations: %d" implFile.Declarations.Length) |> ignore
                                
                                for i, decl in implFile.Declarations |> List.indexed do
                                    sb.AppendLine(sprintf "  [%d] Declaration: %A" i decl) |> ignore
                                
                                sb.AppendLine() |> ignore
                            
                            sb.ToString()
                        writeIntermediateFile initialAstPath initialAstContent
                        
                        printfn "✅ Initial typed AST written: %s" initialAstPath
                    | None -> ()
                    
                    // Phase 3: Collect and analyze symbols
                    progress SymbolCollection "Collecting and categorizing symbols"
                    let symbolCollection = collectSymbols processedProject
                    
                    // Phase 4: Perform reachability analysis
                    progress ReachabilityAnalysis "Performing reachability analysis"
                    let reachabilityResult = analyze symbolCollection
                    
                    match reachabilityResult with
                    | CompilerFailure errors ->
                        return {
                            Success = false
                            IntermediatesGenerated = false
                            ReachabilityReport = None
                            Diagnostics = errors
                            Statistics = {
                                TotalFiles = projectOptions.SourceFiles.Length
                                TotalSymbols = symbolCollection.Statistics.TotalSymbols
                                ReachableSymbols = 0
                                EliminatedSymbols = 0
                                CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                            }
                        }
                    | Success reachability ->
                        
                        // Phase 5: Generate intermediate files
                        progress IntermediateGeneration "Writing intermediate files"
                        let intermediatesGenerated = 
                            match intermediatesDir with
                            | Some dir ->
                                writeIntermediateFiles dir processedProject symbolCollection reachability
                                true
                            | None -> false
                        
                        // Generate reachability report
                        let report = generateReport reachability
                        
                        // TODO: Phase 6 - AST Transformation
                        // Apply optimizations and transformations to the reachable AST
                        
                        // TODO: Phase 7 - MLIR Generation  
                        // Generate MLIR code from transformed AST
                        
                        // TODO: Phase 8 - LLVM Generation
                        // Lower MLIR to LLVM IR
                        
                        // TODO: Phase 9 - Native Compilation
                        // Compile LLVM IR to native binary
                        
                        let stats = {
                            TotalFiles = projectOptions.SourceFiles.Length
                            TotalSymbols = reachability.Statistics.TotalSymbols
                            ReachableSymbols = reachability.Statistics.ReachableSymbols
                            EliminatedSymbols = reachability.Statistics.EliminatedSymbols
                            CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                        }
                        
                        return {
                            Success = true
                            IntermediatesGenerated = intermediatesGenerated
                            ReachabilityReport = Some report
                            Diagnostics = []
                            Statistics = stats
                        }
        
        with ex ->
            return {
                Success = false
                IntermediatesGenerated = false
                ReachabilityReport = None
                Diagnostics = [InternalError("CompilationOrchestrator", ex.Message, Some ex.StackTrace)]
                Statistics = {
                    TotalFiles = 0
                    TotalSymbols = 0
                    ReachableSymbols = 0
                    EliminatedSymbols = 0
                    CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                }
            }
    }

