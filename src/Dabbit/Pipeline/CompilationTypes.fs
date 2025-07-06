module Dabbit.Pipeline.CompilationTypes

open System
open Core.XParsec.Foundation

// ===================================================================
// TypedAST Processing Pipeline Types
// ===================================================================

/// Processing phases for TypedAST analysis
type CompilationPhase =
    | ProjectLoading
    | FCSProcessing
    | SymbolCollection
    | ReachabilityAnalysis
    | IntermediateGeneration
    | ASTTransformation
    | MLIRGeneration
    | LLVMGeneration
    | NativeCompilation

/// Progress callback for pipeline orchestration
type ProgressCallback = CompilationPhase -> string -> unit

// ===================================================================
// Analysis Statistics
// ===================================================================

/// Statistics collected during TypedAST processing
type CompilationStatistics = {
    TotalFiles: int
    TotalSymbols: int
    ReachableSymbols: int
    EliminatedSymbols: int
    CompilationTimeMs: float
}

/// Empty statistics for initialization
let emptyStatistics = {
    TotalFiles = 0
    TotalSymbols = 0
    ReachableSymbols = 0
    EliminatedSymbols = 0
    CompilationTimeMs = 0.0
}

// ===================================================================
// Intermediate Outputs
// ===================================================================

/// Intermediate analysis outputs
type IntermediateOutputs = {
    /// F# AST serialized to text
    FSharpAST: string option
    /// Reduced AST after reachability analysis
    ReducedAST: string option
}

/// Empty intermediate outputs
let emptyIntermediates = {
    FSharpAST = None
    ReducedAST = None
}

// ===================================================================
// Processing Results
// ===================================================================

/// Result of TypedAST processing
type CompilationResult = {
    Success: bool
    Statistics: CompilationStatistics
    Diagnostics: FireflyError list
    Intermediates: IntermediateOutputs
}

/// Create a successful processing result
let successResult stats intermediates _ = {
    Success = true
    Statistics = stats
    Diagnostics = []
    Intermediates = intermediates
}

/// Create a failed processing result
let failureResult errors = {
    Success = false
    Statistics = emptyStatistics
    Diagnostics = errors
    Intermediates = emptyIntermediates
}