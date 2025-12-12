module Alex.Pipeline.CompilationOrchestrator

open System
open System.IO
open System.Text
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Text
open FSharp.Compiler.Symbols
open FSharp.Compiler.Symbols.FSharpExprPatterns
open Core.IngestionPipeline
open Core.FCS.ProjectContext
open Core.Utilities.IntermediateWriter
open Core.PSG.Types
open Alex.Pipeline.CompilationTypes
open Alex.Bindings.BindingTypes
open Alex.CodeGeneration.MLIRBuilder
open Alex.Traversal.PSGZipper
open Alex.Generation.MLIRGeneration
open Alex.Patterns.PSGPatterns

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
// MLIR Generation via Alex
// ===================================================================

/// Result of MLIR generation including any errors
type MLIRGenerationResult = {
    Content: string
    Errors: CompilerError list
    HasErrors: bool
}

/// Collect emission errors during MLIR generation
module EmissionErrors =
    let mutable private errors : CompilerError list = []

    let reset () = errors <- []
    let add (err: CompilerError) = errors <- err :: errors
    let toCompilerErrors () = errors |> List.rev
    let hasErrors () = not (List.isEmpty errors)

/// Generate MLIR from PSG using the Alex traversal architecture
///
/// ARCHITECTURE (correct flow):
///   PSG → EmitContext → genNode (local pattern match) → MLIR accumulation
///
/// The traversal follows PSG structure. At each node, local pattern matching
/// determines the emission. Extern primitives dispatch to platform bindings.
/// MLIR accumulates in the EmitContext builder (correct centralization point).
///
/// Variable resolution uses def-use edges via findDefiningNode, NOT name-based
/// parameter bindings. This is the correct architectural approach.
let generateMLIRViaAlex (psg: ProgramSemanticGraph) (memberBodies: Map<string, Baker.Types.MemberBodyMapping>) (projectName: string) (targetTriple: string) : MLIRGenerationResult =
    // Reset error collector for this compilation
    EmissionErrors.reset()

    // Register all platform bindings
    Alex.Bindings.Time.TimeBindings.registerBindings ()
    Alex.Bindings.Console.ConsoleBindings.registerBindings ()
    Alex.Bindings.Process.ProcessBindings.registerBindings ()

    // Set target platform from triple
    match TargetPlatform.parseTriple targetTriple with
    | Some platform -> ExternDispatch.setTargetPlatform platform
    | None -> ()  // Use default (auto-detect)

    // Use MLIRGeneration module (proper EmitContext + def-use edge architecture)
    // This uses PSGXParsec.EmitContext with NodeSSA tracking and def-use edge resolution
    let result = Alex.Generation.MLIRGeneration.generateMLIR psg targetTriple

    // Convert errors to CompilerError format
    let errors =
        result.Errors
        |> List.map (fun msg -> { Phase = "MLIR Generation"; Message = msg; Location = None; Severity = Error })

    {
        Content = result.Content
        Errors = errors
        HasErrors = result.HasErrors
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
        let! pipelineResult = runPipeline projectPath pipelineConfig

        // Convert diagnostics and generate statistics
        let diagnostics = pipelineResult.Diagnostics |> List.map convertDiagnostic
        let statistics = generateStatistics pipelineResult startTime
        let intermediates = collectIntermediates intermediatesDir

        return {
            Success = pipelineResult.Success
            Diagnostics = diagnostics
            Statistics = statistics
            Intermediates = intermediates
        }

    with ex ->
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
        let! (projectOptions, _diagnostics) = checker.GetProjectOptionsFromScript(projectPath, sourceText)

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
