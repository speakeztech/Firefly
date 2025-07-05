module Dabbit.Pipeline.CompilationTypes

open Core.XParsec.Foundation

/// Basic compilation configuration for minimal testing
type CompilerConfiguration = {
    /// Enable verbose output
    Verbose: bool
    /// Keep intermediate files
    KeepIntermediates: bool
    /// Output directory for intermediates
    IntermediatesDirectory: string option
}

/// Pipeline phase identifier - only active phases
type CompilationPhase =
    | ProjectLoading
    | FCSProcessing
    | SymbolCollection
    | ReachabilityAnalysis
    | IntermediateGeneration
    // Future phases (placeholders):
    | ASTTransformation
    | MLIRGeneration
    | LLVMGeneration
    | NativeCompilation

/// Progress callback for pipeline operations
type ProgressCallback = CompilationPhase -> string -> unit

/// Result of the compilation process
type CompilationResult = {
    Success: bool
    IntermediatesGenerated: bool
    ReachabilityReport: string option
    Diagnostics: FireflyError list
    Statistics: CompilationStatistics
}

and CompilationStatistics = {
    TotalFiles: int
    TotalSymbols: int
    ReachableSymbols: int
    EliminatedSymbols: int
    CompilationTimeMs: float
}

/// Create default minimal configuration
let createMinimalConfig (verbose: bool) (intermediatesDir: string option) : CompilerConfiguration =
    {
        Verbose = verbose
        KeepIntermediates = intermediatesDir.IsSome
        IntermediatesDirectory = intermediatesDir
    }