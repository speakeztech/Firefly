module Dabbit.Pipeline.CompilationOrchestrator

open System
open System.IO
open System.Diagnostics
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Text
open Core.XParsec.Foundation
open Core.AST.Extraction  
open Core.AST.Dependencies
open Core.AST.Reachability
open Core.AST.Validation
open Core.FCSIngestion.ProjectOptionsLoader
open Core.FCSIngestion.ProjectProcessor
open Core.Utilities.IntermediateWriter
open Dabbit.Pipeline.CompilationTypes

// ===================================================================
// Raw AST Output Function
// ===================================================================

/// Write raw AST files for each source file in the project
let writeRawAsts (checkResults: FSharpCheckProjectResults) (intermediatesDir: string) =

    Directory.CreateDirectory intermediatesDir |> ignore
    
    printfn "Writing raw AST files for %d source files" 
        checkResults.AssemblyContents.ImplementationFiles.Length
    
    // Process and write each implementation file's AST
    checkResults.AssemblyContents.ImplementationFiles
    |> List.iteri (fun index implFile ->
        let fileName = Path.GetFileName implFile.FileName 
        let baseName = sprintf "%02d_%s" index (Path.GetFileNameWithoutExtension fileName )
        let astFilePath = Path.Combine(intermediatesDir, baseName + ".initial.ast")
        
        // Create a more detailed header with file information
        let header = sprintf "// Raw AST for %s\n// Module: %s\n// File: %s\n\n" 
                        fileName implFile.QualifiedName implFile.FileName
        
        // Format the declarations in a readable way
        let astText = 
            header +
            (implFile.Declarations
            |> List.map (fun decl -> sprintf "%A" decl)
            |> String.concat "\n\n")
        
        // Write the AST to file
        File.WriteAllText(astFilePath, astText)
        
        printfn "  Wrote raw AST: %s (%d bytes)" baseName (astText.Length)
    )

// ===================================================================
// Main Compilation Pipeline Orchestrator
// ===================================================================

/// Main compilation pipeline orchestrator
let compile (projectPath: string) (intermediatesDir: string option) (progress: ProgressCallback) = async {
    let stopwatch = Stopwatch.StartNew()
    let mutable stats = emptyStatistics
    let mutable intermediates = emptyIntermediates
    
    try
        // Phase 1: Project Loading and Processing
        progress ProjectLoading "Loading and processing project with FCS"
        
        // Create checker instance
        let checker = FSharpChecker.Create(keepAssemblyContents = true)
        
        // Parse project options
        let! projectOptionsResult = parseProjectOptions projectPath checker
        
        // Process based on project options result
        match projectOptionsResult with
        | Error errorMsg ->
            // Early return for project loading errors
            return failureResult [InternalError("ProjectLoading", errorMsg, None)]
            
        | Ok projectOptions ->
            // Process the project with valid options
            let! processResult = processProject projectPath projectOptions checker
            
            // Handle processing result
            match processResult with
            | Error processingError -> 
                // Convert ProcessingError to our error format
                let errors = 
                    processingError.CriticalErrors
                    |> Array.map (fun err -> 
                        TypeCheckError(
                            "FCS", 
                            err.Message, 
                            { Line = err.StartLine; Column = err.StartColumn; File = err.FileName; Offset = 0 }))
                    |> List.ofArray
                
                // Return the failure result at the proper level
                return failureResult errors
                
            | Ok processedProject ->
                // We now have a fully processed project with CheckResults and SymbolUses
                
                // Log basic project information
                let sourceFiles = projectOptions.SourceFiles
                sourceFiles |> Array.iteri (fun i file ->
                    printfn "  [%d] %s" i file)
                
                stats <- { stats with TotalFiles = sourceFiles.Length }
                
                // Write raw AST files if requested
                match intermediatesDir with
                | Some dir ->
                    writeRawAsts processedProject.CheckResults dir
                | None -> ()
                
                // Phase 2: Symbol Collection
                progress SymbolCollection "Extracting typedFunctions and building symbol table"
                
                let typedFunctions = extractFunctions processedProject.CheckResults
                debugEntryPoints typedFunctions
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
                
                // Phase 3: Dependency Analysis
                progress SymbolCollection "Building dependency graph"
                
                let dependencies = buildDependencies typedFunctions
                
                // Phase 4: Reachability Analysis
                progress ReachabilityAnalysis "Computing reachable typedFunctions"
                
                let reachabilityResult = analyzeReachability typedFunctions dependencies true
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
                
                // Phase 5: Validation (Zero-allocation check)
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
    
    with ex ->
        let error = InternalError("compile", ex.Message, Some ex.StackTrace)
        return failureResult [error]
}