module Dabbit.Pipeline.CompilationOrchestrator

open System
open System.IO
open FSharp.Compiler.CodeAnalysis
open Core.XParsec.Foundation
open Core.Utilities.IntermediateWriter
open Core.IngestionPipeline
open Core.FCS.ProjectContext
open Core.Analysis.Reachability
open Core.Analysis.MemoryLayout
open Dabbit.Pipeline.CompilationTypes

// Import necessary modules for MLIR generation (these would need to be implemented)
// open Dabbit.MLIRGeneration

/// Progress callback for reporting compilation phases
type ProgressCallback = CompilationPhase -> string -> unit

/// Convert pipeline config to ingestion pipeline config
let private toPipelineConfig (config: PipelineConfiguration) (templateName: string option) = 
    {
        CacheStrategy = Balanced
        TemplateName = templateName
        CustomTemplateDir = None
        EnableCouplingAnalysis = config.EnableReachabilityAnalysis
        EnableMemoryOptimization = config.EnableStackAllocation
        OutputIntermediates = config.PreserveIntermediateASTs
        IntermediatesDir = None
    }

/// Compile a single F# file (simplified entry point)
let compile (inputPath: string) (intermediatesDir: string option) (progress: ProgressCallback) : Async<CompilationResult> = 
    async {
        let startTime = DateTime.UtcNow
        
        progress ProjectLoading $"Loading {Path.GetFileName(inputPath)}"
        
        // Create a simple project options for single file
        let projectDir = Path.GetDirectoryName(inputPath)
        let sourceFiles = [| inputPath |]
        let projectOptions = buildProjectOptions inputPath sourceFiles
        
        // Create pipeline config
        let config = {
            CacheStrategy = Balanced
            TemplateName = None
            CustomTemplateDir = None
            EnableCouplingAnalysis = true
            EnableMemoryOptimization = false
            OutputIntermediates = intermediatesDir.IsSome
            IntermediatesDir = intermediatesDir
        }
        
        // Run the ingestion pipeline
        progress FCSProcessing "Running FCS analysis..."
        let! pipelineResult = runPipeline inputPath config
        
        if not pipelineResult.Success then
            let errors = 
                pipelineResult.Diagnostics 
                |> List.filter (fun d -> d.Severity = Error)
                |> List.map (fun d -> SyntaxError({ Line = 0; Column = 0; File = inputPath; Offset = 0 }, d.Message, []))
            
            return {
                Success = false
                Statistics = {
                    TotalFiles = 1
                    TotalSymbols = 0
                    ReachableSymbols = 0
                    EliminatedSymbols = 0
                    CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                }
                MLIROutput = None
                LLVMOutput = None
                Diagnostics = errors
            }
        else
            // Extract statistics from pipeline result
            let stats = 
                match pipelineResult.ReachabilityAnalysis with
                | Some ra ->
                    let basic = ra.BasicResult
                    let totalReachable = basic.ReachableSymbols.Count
                    let totalUnreachable = basic.UnreachableSymbols.Count
                    {
                        TotalFiles = 
                            match pipelineResult.ProjectResults with
                            | Some pr -> pr.CompilationOrder.Length
                            | None -> 1
                        TotalSymbols = totalReachable + totalUnreachable
                        ReachableSymbols = totalReachable
                        EliminatedSymbols = totalUnreachable
                        CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                    }
                | None ->
                    {
                        TotalFiles = 1
                        TotalSymbols = 0
                        ReachableSymbols = 0
                        EliminatedSymbols = 0
                        CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                    }
            
            // Write reachability report if intermediates enabled
            match intermediatesDir, pipelineResult.ReachabilityAnalysis with
            | Some dir, Some ra ->
                progress IntermediateGeneration "Writing reachability report..."
                let report = generateReport ra []
                let reportJson = System.Text.Json.JsonSerializer.Serialize(report)
                File.WriteAllText(Path.Combine(dir, "reachability.json"), reportJson)
            | _ -> ()
            
            // TODO: Generate MLIR from the analysis results
            progress MLIRGeneration "Generating MLIR..."
            let mlirOutput = None // This would call into Dabbit MLIR generation
            
            // TODO: Generate LLVM if MLIR succeeded
            progress LLVMGeneration "Generating LLVM IR..."
            let llvmOutput = None // This would call MLIR->LLVM lowering
            
            return {
                Success = true
                Statistics = stats
                MLIROutput = mlirOutput
                LLVMOutput = llvmOutput
                Diagnostics = []
            }
    }

/// Compile an F# project
let compileProject 
    (projectPath: string) 
    (outputPath: string)
    (projectOptions: FSharpProjectOptions) 
    (config: PipelineConfiguration)
    (intermediatesDir: string option)
    (progress: ProgressCallback) : Async<CompilationResult> =
    
    async {
        let startTime = DateTime.UtcNow
        
        progress ProjectLoading $"Loading project {Path.GetFileName(projectPath)}"
        
        // Create ingestion pipeline config
        let pipelineConfig = toPipelineConfig config None
        
        // Run the ingestion pipeline
        progress FCSProcessing "Running FCS analysis..."
        let! pipelineResult = runPipeline projectPath pipelineConfig
        
        if not pipelineResult.Success then
            let errors = 
                pipelineResult.Diagnostics 
                |> List.filter (fun d -> d.Severity = Error)
                |> List.map (fun d -> 
                    SyntaxError(
                        { Line = 0; Column = 0; File = d.Location |> Option.defaultValue projectPath; Offset = 0 }, 
                        d.Message, 
                        []
                    )
                )
            
            return {
                Success = false
                Statistics = {
                    TotalFiles = 0
                    TotalSymbols = 0
                    ReachableSymbols = 0
                    EliminatedSymbols = 0
                    CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                }
                MLIROutput = None
                LLVMOutput = None
                Diagnostics = errors
            }
        else
            // Extract statistics
            let stats = 
                match pipelineResult.ReachabilityAnalysis with
                | Some ra ->
                    let basic = ra.BasicResult
                    let totalReachable = basic.ReachableSymbols.Count
                    let totalUnreachable = basic.UnreachableSymbols.Count
                    {
                        TotalFiles = 
                            match pipelineResult.ProjectResults with
                            | Some pr -> pr.CompilationOrder.Length
                            | None -> 0
                        TotalSymbols = totalReachable + totalUnreachable
                        ReachableSymbols = totalReachable
                        EliminatedSymbols = totalUnreachable
                        CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                    }
                | None ->
                    {
                        TotalFiles = 0
                        TotalSymbols = 0
                        ReachableSymbols = 0
                        EliminatedSymbols = 0
                        CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                    }
            
            // Write analysis results if intermediates enabled
            if config.PreserveIntermediateASTs && intermediatesDir.IsSome then
                progress IntermediateGeneration "Writing analysis results..."
                let dir = intermediatesDir.Value
                
                // Write coupling analysis
                match pipelineResult.CouplingAnalysis with
                | Some ca ->
                    let couplingJson = System.Text.Json.JsonSerializer.Serialize(ca.Report)
                    File.WriteAllText(Path.Combine(dir, "coupling.json"), couplingJson)
                | None -> ()
                
                // Write reachability report
                match pipelineResult.ReachabilityAnalysis with
                | Some ra ->
                    let report = generateReport ra []
                    let reportJson = System.Text.Json.JsonSerializer.Serialize(report)
                    File.WriteAllText(Path.Combine(dir, "reachability.json"), reportJson)
                | None -> ()
                
                // Write memory layout
                match pipelineResult.MemoryLayout with
                | Some layout ->
                    let layoutReport = generateLayoutReport layout []
                    let layoutJson = System.Text.Json.JsonSerializer.Serialize(layoutReport)
                    File.WriteAllText(Path.Combine(dir, "memory_layout.json"), layoutJson)
                | None -> ()
            
            // TODO: Generate MLIR from the analysis results
            progress MLIRGeneration "Generating MLIR..."
            let mlirOutput = None // This would integrate with existing Dabbit MLIR generation
            
            // TODO: Generate LLVM if MLIR succeeded
            progress LLVMGeneration "Generating LLVM IR..."
            let llvmOutput = None // This would call MLIR->LLVM lowering
            
            return {
                Success = true
                Statistics = stats
                MLIROutput = mlirOutput
                LLVMOutput = llvmOutput
                Diagnostics = []
            }
    }

/// Run allocation verification on generated code
let verifyZeroAllocation (mlirCode: string) : Result<unit, string> =
    // This is a placeholder - would need real implementation
    if mlirCode.Contains("alloc") && not (mlirCode.Contains("alloca")) then
        Error "Heap allocation detected"
    else
        Ok ()

/// Generate allocation report
let getAllocationReport (projectResults: PipelineResult) : string =
    match projectResults.MemoryLayout with
    | Some layout ->
        let report = generateLayoutReport layout []
        System.Text.Json.JsonSerializer.Serialize(report, System.Text.Json.JsonSerializerOptions(WriteIndented = true))
    | None -> 
        "No memory layout analysis available"