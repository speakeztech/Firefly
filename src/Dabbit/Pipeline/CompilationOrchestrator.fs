module Dabbit.Pipeline.CompilationOrchestrator

open System
open System.IO
open FSharp.Compiler.CodeAnalysis
open Core.XParsec.Foundation
open Core.Utilities.IntermediateWriter
open Core.FCSIngestion.ProjectOptionsLoader
open Core.FCSIngestion.ProjectProcessor
open Core.FCSIngestion.SymbolCollector
open Dabbit.Pipeline.CompilationTypes
open Dabbit.Pipeline.ReachabilityIntegration

/// Result of the compilation process
type CompilationResult = {
    Success: bool
    IntermediatesGenerated: bool
    ReachabilityReport: string option
    Diagnostics: FireflyError list
    Statistics: CompilationStatistics
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
    let reportContent = ReachabilityHelpers.generateReport reachability
    writeIntermediateFile reportPath reportContent

/// Main compilation function with enhanced pipeline
let compileProject 
    (projectFile: string) 
    (intermediatesDir: string option) 
    (progress: ProgressCallback)
    : Async<CompilationResult> = async {
    
    let startTime = DateTime.UtcNow
    
    try
        // Phase 1: Load and analyze project
        progress ProjectLoading "Loading project options"
        let checker = FSharpChecker.Create(keepAssemblyContents=true)
        let! projectResult = loadProject projectFile checker
        
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
            
            // Phase 2: Process source files with FCS
            progress FCSProcessing "Processing F# source files"
            let! processedResult = processProject projectOptions checker
            
            match processedResult with
            | Error processingError ->
                let diagnostics = 
                    processingError.CriticalErrors 
                    |> Array.map (fun diag -> 
                        InternalError("FCSProcessing", diag.Message, None))
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
                
                // Phase 3: Collect symbols and build dependency graph
                progress SymbolCollection "Collecting symbols and dependencies"
                let symbolCollection = collectSymbols processedProject
                
                // Phase 4: Perform reachability analysis
                progress ReachabilityAnalysis "Performing reachability analysis"
                let reachabilityResult = ReachabilityHelpers.analyze symbolCollection
                
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
                    let report = ReachabilityHelpers.generateReport reachability
                    
                    // Validate zero allocation constraint
                    match ReachabilityHelpers.validateZeroAllocation reachability with
                    | CompilerFailure errors ->
                        return {
                            Success = false
                            IntermediatesGenerated = intermediatesGenerated
                            ReachabilityReport = Some report
                            Diagnostics = errors
                            Statistics = {
                                TotalFiles = projectOptions.SourceFiles.Length
                                TotalSymbols = reachability.Statistics.TotalSymbols
                                ReachableSymbols = reachability.Statistics.ReachableSymbols
                                EliminatedSymbols = reachability.Statistics.EliminatedSymbols
                                CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                            }
                        }
                    | Success () ->
                        
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

/// Compile with minimal configuration for testing
let compileMinimal (projectFile: string) (verbose: bool) : Async<CompilationResult> =
    let intermediatesDir = 
        if verbose then 
            Some (Path.Combine(Path.GetDirectoryName(projectFile), "build", "intermediates"))
        else None
    
    let progress = fun phase message ->
        if verbose then
            printfn "[%A] %s" phase message
    
    compileProject projectFile intermediatesDir progress

/// Quick AST-only analysis without full compilation
let analyzeASTOnly (projectFile: string) : Async<CompilationResult> =
    let progress = fun phase message ->
        printfn "[%A] %s" phase message
    
    // Run only up to reachability analysis
    async {
        let! result = compileProject projectFile None progress
        return { result with Success = true } // Mark as success even if incomplete
    }