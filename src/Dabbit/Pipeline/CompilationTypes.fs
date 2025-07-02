module Dabbit.Pipeline.CompilationTypes

open FSharp.Compiler.Syntax
open FSharp.Compiler.Text
open FSharp.Compiler.CodeAnalysis
open Core.Types.TypeSystem
open Core.XParsec.Foundation
open Core.FCSIngestion.SymbolExtraction
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

/// Pipeline phase identifier
type PipelinePhase =
    | Initialization
    | Parsing
    | TypeChecking
    // New 5-pass phases
    | SymbolDiscovery    // Pass 1
    | ImportResolution   // Pass 2
    | TypeResolution     // Pass 3
    | ReferenceResolution // Pass 4
    | ReachabilityAnalysis // Pass 5
    // Transformation phases
    | ASTTransformation
    | MLIRGeneration
    | LLVMGeneration
    | Finalization

/// Progress callback for pipeline operations
type ProgressCallback = PipelinePhase -> string -> unit

// Add at the top of CompilationTypes.fs, before the pipeline types

/// Comparable source location for use as Map keys
[<CustomComparison; CustomEquality>]
type SourceLocation = {
    FilePath: string
    StartLine: int
    StartColumn: int
    EndLine: int
    EndColumn: int
} with
    static member FromRange(filePath: string, range: Range) = {
        FilePath = filePath
        StartLine = range.StartLine
        StartColumn = range.StartColumn
        EndLine = range.EndLine
        EndColumn = range.EndColumn
    }
    
    interface System.IComparable with
        member x.CompareTo(obj) =
            match obj with
            | :? SourceLocation as y ->
                match compare x.FilePath y.FilePath with
                | 0 -> 
                    match compare x.StartLine y.StartLine with
                    | 0 -> compare x.StartColumn y.StartColumn
                    | n -> n
                | n -> n
            | _ -> -1
    
    override x.Equals(obj) =
        match obj with
        | :? SourceLocation as y -> 
            x.FilePath = y.FilePath && 
            x.StartLine = y.StartLine && 
            x.StartColumn = y.StartColumn
        | _ -> false
    
    override x.GetHashCode() =
        hash (x.FilePath, x.StartLine, x.StartColumn)

/// ===== 5-Pass Pipeline Intermediate Results =====

/// Result of Pass 1: Symbol Discovery
type SymbolDiscoveryResult = {
    /// All parsed files
    ParsedFiles: Map<string, ParsedInput>
    /// Symbols discovered per file
    FileSymbols: Map<string, ExtractedSymbol list>  
    /// Module paths per file
    ModulePaths: Map<string, string list>
    /// Entry points found (functions with [<EntryPoint>])
    EntryPoints: Set<string>
}

/// Result of Pass 2: Import Resolution  
type ImportResolutionResult = {
    /// Input from previous pass
    Discovery: SymbolDiscoveryResult
    /// Opened modules per file
    FileOpenedModules: Map<string, string list list>
    /// Module resolution state per file
    FileScopes: Map<string, ModuleResolutionState>
    /// Global symbol table (module -> symbols)
    GlobalSymbolTable: Map<string, Set<string>>
}

/// Result of Pass 3: Type Resolution
type TypeResolutionResult = {
    /// Input from previous pass
    ImportContext: ImportResolutionResult
    /// Type of every expression by location
    ExpressionTypes: Map<SourceLocation, MLIRType>
    /// Type of every binding by (file, symbol name)
    BindingTypes: Map<string * string, MLIRType>
    /// Members available on each type
    TypeMembers: Map<string, Set<string>>
    /// Interface implementations
    InterfaceImplementations: Map<string, Set<string>>  // type -> interfaces
}

/// Result of Pass 4: Reference Resolution
type ReferenceResolutionResult = {
    /// Input from previous pass
    TypeContext: TypeResolutionResult
    /// Every reference resolved to qualified name
    ResolvedReferences: Map<SourceLocation, string>
    /// Complete dependency graph
    DependencyGraph: Map<string, Set<string>>
    /// Unresolved references (for error reporting)
    UnresolvedReferences: (SourceLocation * string) list  // location, name
}

/// Result of Pass 5: Reachability Analysis
type ReachabilityAnalysisResult = {
    /// Input from previous pass
    ReferenceContext: ReferenceResolutionResult
    /// Final reachable symbols
    ReachableSymbols: Set<string>
    /// Reachable symbols per file
    FileReachableSymbols: Map<string, Set<string>>
    /// Statistics
    Statistics: ReachabilityStatistics
}

/// Result of the complete 5-pass pipeline
type FivePassPipelineResult = {
    SymbolDiscovery: SymbolDiscoveryResult
    ImportResolution: ImportResolutionResult
    TypeResolution: TypeResolutionResult
    ReferenceResolution: ReferenceResolutionResult
    ReachabilityAnalysis: ReachabilityAnalysisResult
}

// MARK FOR DELETION after migration:
// - MultiFileProcessingResult (replaced by FivePassPipelineResult)
// - ProcessedUnit (data distributed across pass results)
// - Parts of CompilationUnitAnalysis (functionality split across passes)