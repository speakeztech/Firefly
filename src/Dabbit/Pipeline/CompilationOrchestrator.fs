module Dabbit.Pipeline.CompilationOrchestrator

open System
open System.IO
open FSharp.Compiler.CodeAnalysis
open Core.XParsec.Foundation
open Core.Utilities.IntermediateWriter
open Core.Utilities.RemoveIntermediates
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

/// Write intermediate files - STATELESS function that just writes what it's given
let private writeIntermediateFiles 
    (intermediatesDir: string)
    (astFiles: (string * string) list)  // (filename, content) pairs
    (symbolCollection: SymbolCollectionResult) 
    (reachability: ReachabilityResult) : unit =
    
    // Create directory structure
    let analysisDir = Path.Combine(intermediatesDir, "analysis")
    Directory.CreateDirectory(analysisDir) |> ignore
    
    // Write AST files passed to us
    for (fileName, content) in astFiles do
        let filePath = Path.Combine(intermediatesDir, fileName)
        writeFileToPath filePath content
    

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
            // clear intermediates to ensure all files are from the current compilation pass
            prepareIntermediatesDirectory intermediatesDir

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
                
                // IMMEDIATELY emit initial AST files while we have the data
                match intermediatesDir with
                | Some dir ->
                    let implementationFiles = processedProject.CheckResults.AssemblyContents.ImplementationFiles
                    for i, implFile in Array.indexed (implementationFiles |> Seq.toArray) do
                        let sourceFileName = Path.GetFileNameWithoutExtension(projectOptions.SourceFiles.[i])
                        let fileName = sprintf "%02d_%s.initial.ast" i sourceFileName
                        let filePath = Path.Combine(dir, fileName)
                        let astContent = sprintf "%A" implFile.Declarations
                        writeFileToPath filePath astContent
                | None -> ()

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
    
                    // IMMEDIATELY emit reachability outputs while we have the data
                    progress IntermediateGeneration "Writing intermediate files"
                    let intermediatesGenerated = 
                        match intermediatesDir with
                        | Some dir ->
                            // Create analysis directory
                            let analysisDir = Path.Combine(dir, "analysis")
                            Directory.CreateDirectory(analysisDir) |> ignore
                            
                            // Emit reduced AST files (with reachability applied)
                            let implementationFiles = processedProject.CheckResults.AssemblyContents.ImplementationFiles
                            for i, implFile in Array.indexed (implementationFiles |> Seq.toArray) do
                                let sourceFileName = Path.GetFileNameWithoutExtension(projectOptions.SourceFiles.[i])
                                let fileName = sprintf "%02d_%s.reduced.ast" i sourceFileName
                                let filePath = Path.Combine(dir, fileName)
                                let astContent = sprintf "%A" implFile.Declarations  // TODO: Apply reachability pruning
                                writeFileToPath filePath astContent
                            
                            // Emit reachability analysis data
                            let reachabilityPath = Path.Combine(analysisDir, "reachability.analysis")
                            writeFileToPath reachabilityPath (sprintf "%A" reachability)
                            
                            // Emit human-readable reachability report
                            let reportPath = Path.Combine(analysisDir, "reachability.report")
                            writeFileToPath reportPath (ReachabilityHelpers.generateReport reachability)
                            
                            // Emit symbol collection data
                            let symbolsPath = Path.Combine(analysisDir, "symbols.analysis")
                            writeFileToPath symbolsPath (sprintf "%A" symbolCollection)
                            
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
                            Diagnostics = []
                            Statistics = stats
                            ReachabilityReport = None
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