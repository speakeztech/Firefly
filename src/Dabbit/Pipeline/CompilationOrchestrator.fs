module Dabbit.Pipeline.CompilationOrchestrator

open System
open System.IO
open System.Diagnostics
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Text
open Core.XParsec.Foundation    // Import shared types
open Core.AST.Extraction        // Import TypedFunction from here
open Core.AST.Dependencies
open Core.AST.Reachability
open Core.AST.Validation
open Core.Utilities.IntermediateWriter
open Dabbit.Pipeline.CompilationTypes

// ===================================================================
// Main Compilation Pipeline Orchestrator
// ===================================================================

/// Main compilation pipeline orchestrator
let compile (projectPath: string) (intermediatesDir: string option) (progress: ProgressCallback) : Async<CompilationResult> = async {
    let stopwatch = Stopwatch.StartNew()
    let mutable stats = emptyStatistics
    let mutable intermediates = emptyIntermediates
    
    try
        // Phase 1: Project Loading
        progress ProjectLoading "Loading project and creating FCS options"
        
        let checker = FSharpChecker.Create(keepAssemblyContents = true)
        let projectOptions = 
            if projectPath.EndsWith(".fsproj") then
                {
                    ProjectFileName = projectPath
                    ProjectId = None
                    SourceFiles = [|projectPath.Replace(".fsproj", ".fs")|]
                    OtherOptions = [|"--warn:3"; "--noframework"|]
                    ReferencedProjects = [||]
                    IsIncompleteTypeCheckEnvironment = false
                    UseScriptResolutionRules = false
                    LoadTime = DateTime.Now
                    UnresolvedReferences = None
                    OriginalLoadReferences = []
                    Stamp = None
                }
            else
                {
                    ProjectFileName = "single-file.fsproj"
                    ProjectId = None
                    SourceFiles = [|projectPath|]
                    OtherOptions = [|"--warn:3"; "--noframework"|]
                    ReferencedProjects = [||]
                    IsIncompleteTypeCheckEnvironment = false
                    UseScriptResolutionRules = false
                    LoadTime = DateTime.Now
                    UnresolvedReferences = None
                    OriginalLoadReferences = []
                    Stamp = None
                }
        
        stats <- { stats with TotalFiles = projectOptions.SourceFiles.Length }
        
        // Phase 2: FCS Processing
        progress FCSProcessing "Type-checking and building AST"
        
        let! checkResults = checker.ParseAndCheckProject(projectOptions)
        
        if checkResults.HasCriticalErrors then
            let errors = 
                checkResults.Diagnostics
                |> Array.filter (fun d -> d.Severity = FSharp.Compiler.Diagnostics.FSharpDiagnosticSeverity.Error)
                |> Array.map (fun d -> 
                    TypeCheckError(
                        "FCS", 
                        d.Message, 
                        { Line = d.StartLine; Column = d.StartColumn; File = d.FileName; Offset = 0 }))
                |> List.ofArray
            
            return failureResult errors
        else
            // Phase 3: Symbol Collection
            progress SymbolCollection "Extracting typedFunctions and building symbol table"
            // In whichever file initializes your FSharpChecker
            
            let typedFunctions = extractFunctions checkResults
            stats <- { stats with TotalSymbols = typedFunctions.Length }
            
            // Write F# AST intermediate if requested
            match intermediatesDir with
            | Some dir ->
                let astContent = 
                    typedFunctions 
                    |> Array.map (fun f -> $"{f.FullName}: {f.Module}")
                    |> String.concat "\n"
                let astPath = Path.Combine(dir, Path.GetFileNameWithoutExtension(projectPath) + ".fcs")
                writeFileToPath astPath astContent
                intermediates <- { intermediates with FSharpAST = Some astContent }
            | None -> ()
            
            // Phase 4: Dependency Analysis
            progress SymbolCollection "Building dependency graph"
            
            let dependencies = buildDependencies typedFunctions
            
            // Phase 5: Reachability Analysis
            progress ReachabilityAnalysis "Computing reachable typedFunctions"
            
            let reachabilityResult = analyzeReachability typedFunctions dependencies
            let reachableFunctions = 
                typedFunctions 
                |> Array.filter (fun f -> Set.contains f.FullName reachabilityResult.ReachableFunctions)
            
            stats <- { 
                stats with 
                    ReachableSymbols = reachabilityResult.ReachableFunctions.Count
                    EliminatedSymbols = reachabilityResult.UnreachableFunctions.Count
            }
            
            // Write reduced AST intermediate if requested
            match intermediatesDir with
            | Some dir ->
                let reducedAstContent = generateReachabilityReport reachabilityResult
                let reducedAstPath = Path.Combine(dir, Path.GetFileNameWithoutExtension(projectPath) + ".ra.fcs")
                writeFileToPath reducedAstPath reducedAstContent
                intermediates <- { intermediates with ReducedAST = Some reducedAstContent }
            | None -> ()
            
            // Phase 6: Validation (Zero-allocation check)
            progress IntermediateGeneration "Validating zero-allocation constraints"
            
            match verifyZeroAllocation reachableFunctions with
            | Ok () ->
                progress IntermediateGeneration "✓ Zero-allocation verification passed"
            | Error allocationSites ->
                let allocationReport = getAllocationReport allocationSites
                progress IntermediateGeneration $"⚠ Allocation violations found: {allocationSites.Length}"
                
                // Write allocation report if intermediate files are enabled
                match intermediatesDir with
                | Some dir ->
                    let reportPath = Path.Combine(dir, Path.GetFileNameWithoutExtension(projectPath) + ".allocations")
                    writeFileToPath reportPath allocationReport
                | None -> ()
            
            stopwatch.Stop()
            stats <- { stats with CompilationTimeMs = stopwatch.Elapsed.TotalMilliseconds }
            
            return successResult stats intermediates None
    
    with
    | ex ->
        let error = InternalError("compile", ex.Message, Some ex.StackTrace)
        return failureResult [error]
}

