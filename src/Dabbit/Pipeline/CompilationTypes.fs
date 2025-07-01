module Dabbit.Pipeline.CompilationTypes

open FSharp.Compiler.Syntax
open FSharp.Compiler.CodeAnalysis
open Core.XParsec.Foundation
open Dabbit.CodeGeneration.TypeMapping
open Dabbit.Bindings.SymbolRegistry
open Dabbit.Analysis.CompilationUnit
open Dabbit.Analysis.ReachabilityAnalyzer

/// Result of processing a single compilation unit through the pipeline
type ProcessedUnit = {
    FilePath: string
    OriginalAST: ParsedInput
    TransformedAST: ParsedInput
    Symbols: Set<string>
    ReachableSymbols: Set<string>
}

/// Complete result of multi-file processing
type MultiFileProcessingResult = {
    ProcessedUnits: Map<string, ProcessedUnit>
    CompilationAnalysis: CompilationUnitAnalysis
    TypeContext: TypeContext
    SymbolRegistry: SymbolRegistry
    GlobalReachability: ReachabilityResult
}

/// Configuration for the compilation pipeline
type PipelineConfiguration = {
    EnableClosureElimination: bool
    EnableStackAllocation: bool
    EnableReachabilityAnalysis: bool
    PreserveIntermediateASTs: bool
    VerboseOutput: bool
}

/// State maintained throughout compilation
type CompilationState = {
    Checker: FSharpChecker
    ProjectOptions: FSharpProjectOptions
    TypeContext: TypeContext
    SymbolRegistry: SymbolRegistry
    Configuration: PipelineConfiguration
    IntermediatesDirectory: string option
}

/// Result of a complete compilation pipeline run
type CompilationResult = {
    Success: bool
    MLIROutput: string option
    LLVMOutput: string option
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

/// Pipeline phase identifier for progress reporting
type PipelinePhase =
    | Initialization
    | Parsing
    | TypeChecking
    | SymbolExtraction
    | ReachabilityAnalysis
    | ASTTransformation
    | MLIRGeneration
    | LLVMGeneration
    | Finalization

/// Progress callback for pipeline operations
type ProgressCallback = PipelinePhase -> string -> unit