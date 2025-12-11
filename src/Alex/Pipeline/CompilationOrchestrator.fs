module Alex.Pipeline.CompilationOrchestrator

open System
open System.IO
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Text
open Core.IngestionPipeline
open Core.FCS.ProjectContext
open Core.Utilities.IntermediateWriter
open Core.PSG.Types
open Alex.Pipeline.CompilationTypes
open Alex.Bindings.BindingTypes
open Alex.CodeGeneration.MLIRBuilder

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

/// Generate MLIR from PSG using proper layered architecture
///
/// ARCHITECTURE (correct flow):
///   PSG → Zipper (traversal) → XParsec (pattern match) → Bindings (MLIR generation)
///
/// The Zipper provides bidirectional traversal with context preservation.
/// XParsec provides composable pattern matching on PSG node children.
/// Bindings dispatch extern primitives to platform-specific MLIR generators.
///
/// PSGScribe was removed because it violated layer separation by:
/// - Doing ad-hoc traversal instead of using the Zipper
/// - Matching on operation names (Alloy awareness in wrong layer)
/// - Generating MLIR strings directly instead of using Bindings
///
/// TODO: Implement proper Zipper-based traversal that:
/// 1. Walks the PSG from entry points using PSGZipper
/// 2. Uses XParsec combinators to match node patterns
/// 3. Dispatches extern primitives via ExternDispatch.dispatch
/// 4. Uses MLIR monad (MLIRBuilder) for code generation
///
let generateMLIRViaAlex (psg: ProgramSemanticGraph) (projectName: string) (targetTriple: string) : MLIRGenerationResult =
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

    // TODO: Replace with proper Zipper + XParsec + Bindings traversal
    // For now, emit a placeholder that documents the architectural requirement
    let mlirContent =
        String.concat "\n" [
            "// MLIR generation pending - PSGScribe removed for architectural cleanup"
            "//"
            "// The correct implementation requires:"
            "// 1. PSGZipper traversal from entry points"
            "// 2. XParsec pattern matching at each node"
            "// 3. ExternDispatch for extern primitive binding"
            "// 4. MLIR monad for type-safe code generation"
            "//"
            "// See: docs/Native_Library_Binding_Architecture.md"
            "// See: Alex/Traversal/PSGZipper.fs"
            "// See: Alex/Traversal/PSGXParsec.fs"
            "// See: Alex/Bindings/BindingTypes.fs"
            ""
            "module {"
            sprintf "  // Placeholder - proper MLIR generation not yet implemented"
            sprintf "  // PSG has %d nodes, %d edges, %d entry points" psg.Nodes.Count psg.Edges.Length psg.EntryPoints.Length
            "  llvm.func @main() -> i32 {"
            "    %zero = arith.constant 0 : i32"
            "    llvm.return %zero : i32"
            "  }"
            "}"
        ]

    // Wrap with module header comments
    let header =
        sprintf "// Firefly-generated MLIR for %s\n// Target: %s\n// Status: Architectural rebuild in progress\n\n"
            projectName targetTriple

    // Collect any emission errors
    let emissionErrors = EmissionErrors.toCompilerErrors()

    {
        Content = header + mlirContent
        Errors = emissionErrors
        HasErrors = EmissionErrors.hasErrors()
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