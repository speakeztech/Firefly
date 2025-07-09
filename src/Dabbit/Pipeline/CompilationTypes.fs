module Dabbit.Pipeline.CompilationTypes

open System

// ===================================================================
// Compilation Pipeline Types
// ===================================================================

/// Phases of the compilation pipeline
type CompilationPhase =
    | Initialization
    | Parsing
    | TypeChecking
    | SymbolExtraction
    | ReachabilityAnalysis
    | ASTTransformation
    | MLIRGeneration
    | LLVMGeneration
    | IntermediateGeneration
    | Finalization

/// Error severity levels
type ErrorSeverity =
    | Error
    | Warning
    | Info

/// Compiler error with context
type CompilerError = {
    Phase: string
    Message: string
    Location: string option
    Severity: ErrorSeverity
}

/// Compilation statistics
type CompilationStatistics = {
    TotalFiles: int
    TotalSymbols: int
    ReachableSymbols: int
    EliminatedSymbols: int
    CompilationTimeMs: float
}

/// Empty statistics for initialization
module CompilationStatistics =
    let empty = {
        TotalFiles = 0
        TotalSymbols = 0
        ReachableSymbols = 0
        EliminatedSymbols = 0
        CompilationTimeMs = 0.0
    }

/// Configuration for the compilation pipeline
type CompilationConfig = {
    EnableClosureElimination: bool
    EnableStackAllocation: bool
    EnableReachabilityAnalysis: bool
    PreserveIntermediateASTs: bool
    VerboseOutput: bool
}

/// Progress callback for reporting compilation status
type ProgressCallback = CompilationPhase -> string -> unit

// ===================================================================
// Project Loading Types (for FCS integration)
// ===================================================================

/// Project loading phase (subset of compilation phases)
type ProjectLoadingPhase =
    | ProjectLoading
    | FCSProcessing
    | SymbolCollection
    | IntermediateGeneration

// Note: Removed projectPhaseToCompilationPhase as it was causing type issues
// The phases are handled differently in the orchestrator