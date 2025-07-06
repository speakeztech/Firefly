module Dabbit.Pipeline.CompilationOrchestrator

open System
open System.IO
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Text
open Core.Utilities.IntermediateWriter
open Core.Utilities.RemoveIntermediates
open Core.FCSIngestion.ProjectOptionsLoader
open Core.FCSIngestion.ProjectProcessor
open Core.AST.Extraction
open Core.AST.Dependencies
open Core.AST.Reachability
open Core.AST.Validation
open Dabbit.Pipeline.CompilationTypes

type CompilationResult = {
    Success: bool
    IntermediatesGenerated: bool
    ReachabilityReport: string option
    Diagnostics: FireflyError list
    Statistics: CompilationStatistics
}

let compileProject 
    (projectFile: string) 
    (intermediatesDir: string option) 
    (progress: ProgressCallback)
    : Async<CompilationResult> = async {
    
    let startTime = DateTime.UtcNow
    
    try
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
                    TotalFiles = 0; TotalSymbols = 0; ReachableSymbols = 0
                    EliminatedSymbols = 0; CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                }
            }
        | Ok projectOptions ->
            prepareIntermediatesDirectory intermediatesDir

            progress FCSProcessing "Processing F# source files"
            let! processedResult = processProject projectOptions checker
            
            match processedResult with
            | Error processingError ->
                let diagnostics = 
                    processingError.CriticalErrors 
                    |> Array.map (fun diag -> InternalError("FCSProcessing", diag.Message, None))
                    |> Array.toList
                
                return {
                    Success = false; IntermediatesGenerated = false; ReachabilityReport = None
                    Diagnostics = diagnostics
                    Statistics = {
                        TotalFiles = projectOptions.SourceFiles.Length; TotalSymbols = 0
                        ReachableSymbols = 0; EliminatedSymbols = 0
                        CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                    }
                }
            | Ok processedProject ->
                
                progress SymbolCollection "Extracting functions from AST"
                let typedFunctions = extractFunctions processedProject.CheckResults

                printfn "[AST] Found %d functions" typedFunctions.Length
                typedFunctions |> Array.iter (fun f ->
                    printfn "  FUNC: %s at %s:%d-%d" f.FullName 
                        (Path.GetFileName f.Range.FileName) f.Range.Start.Line f.Range.End.Line)

                progress ReachabilityAnalysis "Building dependency graph"  
                let dependencies = buildDependencies typedFunctions
                printfn "[AST] Built %d dependencies" dependencies.Length

                let entryPoints = typedFunctions |> Array.filter (fun f -> f.IsEntryPoint) |> Array.map (fun f -> f.FullName)
                
                if Array.isEmpty entryPoints then
                    return {
                        Success = false; IntermediatesGenerated = false; ReachabilityReport = None
                        Diagnostics = [InternalError("Reachability", "No entry points found", None)]
                        Statistics = {
                            TotalFiles = projectOptions.SourceFiles.Length; TotalSymbols = typedFunctions.Length
                            ReachableSymbols = 0; EliminatedSymbols = typedFunctions.Length
                            CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                        }
                    }
                else
                    let reachableNames = computeReachable dependencies entryPoints
                    let reachableFunctions = 
                        reachableNames
                        |> Set.toArray
                        |> Array.choose (fun name -> 
                            typedFunctions |> Array.tryFind (fun f -> f.FullName = name))

                    printfn "[AST] %d functions reachable from entry points" reachableFunctions.Length
                    
                    match verifyZeroAllocation reachableFunctions with
                    | Ok () -> 
                        printfn "[AST] ✓ Zero allocations verified"
                        
                        return {
                            Success = true; IntermediatesGenerated = false; ReachabilityReport = None
                            Diagnostics = []
                            Statistics = {
                                TotalFiles = projectOptions.SourceFiles.Length; TotalSymbols = typedFunctions.Length
                                ReachableSymbols = reachableFunctions.Length
                                EliminatedSymbols = typedFunctions.Length - reachableFunctions.Length
                                CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                            }
                        }
                    
                    | Error allocations -> 
                        printfn "[AST] ❌ %d allocations found" allocations.Length
                        allocations |> Array.iter (fun alloc ->
                            printfn "  ALLOCATION: %s at %s:%d" alloc.TypeName 
                                (Path.GetFileName alloc.Location.FileName) alloc.Location.Start.Line)
                        
                        return {
                            Success = false; IntermediatesGenerated = false; ReachabilityReport = None
                            Diagnostics = allocations |> Array.map (fun alloc -> 
                                AllocationDetected(alloc.TypeName, alloc.Location)) |> Array.toList
                            Statistics = {
                                TotalFiles = projectOptions.SourceFiles.Length; TotalSymbols = typedFunctions.Length
                                ReachableSymbols = reachableFunctions.Length; EliminatedSymbols = 0
                                CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                            }
                        }

    with ex ->
        return {
            Success = false; IntermediatesGenerated = false; ReachabilityReport = None
            Diagnostics = [InternalError("CompilationOrchestrator", ex.Message, Some ex.StackTrace)]
            Statistics = {
                TotalFiles = 0; TotalSymbols = 0; ReachableSymbols = 0; EliminatedSymbols = 0
                CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
            }
        }
    }

let compileMinimal (projectFile: string) (verbose: bool) : Async<CompilationResult> =
    let intermediatesDir = 
        if verbose then Some (Path.Combine(Path.GetDirectoryName(projectFile), "build", "intermediates"))
        else None
    
    let progress = fun phase message ->
        if verbose then printfn "[%A] %s" phase message
    
    compileProject projectFile intermediatesDir progress