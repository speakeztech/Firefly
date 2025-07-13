module Alex.Pipeline.CompilationOrchestrator

open System
open System.IO
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Text
open Core.IngestionPipeline
open Core.FCS.ProjectContext
open Core.Utilities.IntermediateWriter
open Alex.Pipeline.CompilationTypes

// ===================================================================
// Pipeline Integration Types
// ===================================================================

/// Extended compilation result with outputs
type CompilationResult = {
    Success: bool
    Diagnostics: CompilerError list
    Statistics: CompilationStatistics
    Intermediates: IntermediateOutputs
}

/// Intermediate file outputs
and IntermediateOutputs = {
    ProjectAnalysis: string option
    PSGRepresentation: string option
    ReachabilityAnalysis: string option
    PrunedSymbols: string option
}

/// Empty intermediates for initialization
let emptyIntermediates = {
    ProjectAnalysis = None
    PSGRepresentation = None
    ReachabilityAnalysis = None
    PrunedSymbols = None
}

// ===================================================================
// Pipeline Configuration Bridge
// ===================================================================

/// Convert compilation config to ingestion pipeline config
let createPipelineConfig (config: CompilationConfig) (intermediatesDir: string option) : PipelineConfig = {
    CacheStrategy = Balanced
    TemplateName = None
    CustomTemplateDir = None
    EnableCouplingAnalysis = config.EnableReachabilityAnalysis
    EnableMemoryOptimization = config.EnableStackAllocation
    OutputIntermediates = config.PreserveIntermediateASTs
    IntermediatesDir = intermediatesDir
}

// ===================================================================
// Type Conversions
// ===================================================================

/// Convert IngestionPipeline.DiagnosticSeverity to CompilationTypes.ErrorSeverity
let convertSeverity (severity: DiagnosticSeverity) : ErrorSeverity =
    match severity with
    | DiagnosticSeverity.Error -> ErrorSeverity.Error
    | DiagnosticSeverity.Warning -> ErrorSeverity.Warning
    | DiagnosticSeverity.Info -> ErrorSeverity.Info

/// Convert IngestionPipeline.Diagnostic to CompilerError
let convertDiagnostic (diag: Diagnostic) : CompilerError = {
    Phase = "Pipeline"
    Message = diag.Message
    Location = diag.Location
    Severity = convertSeverity diag.Severity
}

// ===================================================================
// Progress Reporting
// ===================================================================

/// Report compilation phase progress
let reportPhase (progress: ProgressCallback) (phase: CompilationPhase) (message: string) =
    progress phase message

// ===================================================================
// Statistics Collection
// ===================================================================

/// Generate meaningful compilation statistics from pipeline results
let generateStatistics (pipelineResult: PipelineResult) (startTime: DateTime) : CompilationStatistics =
    let endTime = DateTime.UtcNow
    let duration = endTime - startTime
    
    let totalFiles = 
        match pipelineResult.ProjectResults with
        | Some projectResults -> projectResults.CompilationOrder.Length
        | None -> 0
    
    let totalSymbols =
        match pipelineResult.ReachabilityAnalysis with
        | Some analysis -> analysis.PruningStatistics.TotalSymbols
        | None -> 0
    
    let reachableSymbols =
        match pipelineResult.ReachabilityAnalysis with
        | Some analysis -> analysis.PruningStatistics.ReachableSymbols
        | None -> 0
    
    let eliminatedSymbols =
        match pipelineResult.ReachabilityAnalysis with
        | Some analysis -> analysis.PruningStatistics.EliminatedSymbols
        | None -> 0
    
    {
        TotalFiles = totalFiles
        TotalSymbols = totalSymbols
        ReachableSymbols = reachableSymbols
        EliminatedSymbols = eliminatedSymbols
        CompilationTimeMs = float duration.TotalMilliseconds
    }

// ===================================================================
// Intermediate File Management
// ===================================================================

/// Collect intermediate file paths from pipeline execution
let collectIntermediates (intermediatesDir: string option) : IntermediateOutputs =
    match intermediatesDir with
    | None -> emptyIntermediates
    | Some dir ->
        let tryFindFile fileName =
            let path = Path.Combine(dir, fileName)
            if File.Exists(path) then Some path else None
        
        {
            ProjectAnalysis = tryFindFile "project.analysis.json"
            PSGRepresentation = tryFindFile "psg.summary.json"
            ReachabilityAnalysis = tryFindFile "reachability.analysis.json"
            PrunedSymbols = tryFindFile "psg.pruned.symbols.json"
        }

// ===================================================================
// Main Compilation Entry Points
// ===================================================================

/// Compile a project file using the ingestion pipeline
let compileProject 
    (projectPath: string) 
    (outputPath: string)
    (projectOptions: FSharpProjectOptions)
    (compilationConfig: CompilationConfig)
    (intermediatesDir: string option)
    (progress: ProgressCallback) = async {
    
    let startTime = DateTime.UtcNow
    
    // Convert CompilationConfig to PipelineConfig
    let pipelineConfig = createPipelineConfig compilationConfig intermediatesDir
    
    try
        // Execute the complete ingestion and analysis pipeline
        printfn "[Compilation] Starting compilation pipeline..."
        let! pipelineResult = runPipeline projectPath pipelineConfig
        
        // Convert diagnostics and generate statistics
        let diagnostics = pipelineResult.Diagnostics |> List.map convertDiagnostic
        let statistics = generateStatistics pipelineResult startTime
        let intermediates = collectIntermediates intermediatesDir
        
        // Report final results
        if pipelineResult.Success then
            printfn "[Compilation] Compilation completed successfully"
            
            match pipelineResult.ReachabilityAnalysis with
            | Some analysis ->
                printfn "[Compilation] Final statistics: %d/%d symbols reachable (%.1f%% eliminated)" 
                    analysis.PruningStatistics.ReachableSymbols
                    analysis.PruningStatistics.TotalSymbols
                    ((float analysis.PruningStatistics.EliminatedSymbols / float analysis.PruningStatistics.TotalSymbols) * 100.0)
            | None -> ()
        else
            printfn "[Compilation] Compilation failed"
        
        return {
            Success = pipelineResult.Success
            Diagnostics = diagnostics
            Statistics = statistics
            Intermediates = intermediates
        }
        
    with ex ->
        printfn "[Compilation] Compilation failed: %s" ex.Message
        return {
            Success = false
            Diagnostics = [{
                Phase = "Compilation"
                Message = ex.Message
                Location = None
                Severity = ErrorSeverity.Error
            }]
            Statistics = CompilationStatistics.empty
            Intermediates = emptyIntermediates
        }
}

/// Simplified entry point using file path
let compile 
    (projectPath: string) 
    (intermediatesDir: string option) 
    (progress: ProgressCallback) = async {
    
    // Create default compilation configuration
    let compilationConfig : CompilationConfig = {
        EnableClosureElimination = true
        EnableStackAllocation = true
        EnableReachabilityAnalysis = true
        PreserveIntermediateASTs = intermediatesDir.IsSome
        VerboseOutput = false
    }
    
    // Create F# checker and load project
    let checker = FSharpChecker.Create()
    
    try
        // Read the project file content and create ISourceText
        let content = File.ReadAllText(projectPath)
        let sourceText = SourceText.ofString content
        
        // Get project options from script
        let! (projectOptions, diagnostics) = checker.GetProjectOptionsFromScript(projectPath, sourceText)
        
        // Check for critical errors in diagnostics
        if diagnostics.Length > 0 then
            printfn "[Compilation] Project loading diagnostics:"
            for diag in diagnostics do
                printfn "  %s" diag.Message
        
        // Use a default output path
        let outputPath = Path.ChangeExtension(projectPath, ".exe")
        
        return! compileProject projectPath outputPath projectOptions compilationConfig intermediatesDir progress
    
    with ex ->
        return {
            Success = false
            Diagnostics = [{
                Phase = "ProjectLoading"
                Message = ex.Message
                Location = Some projectPath
                Severity = ErrorSeverity.Error
            }]
            Statistics = CompilationStatistics.empty
            Intermediates = emptyIntermediates
        }
}