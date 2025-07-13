module Core.PSG.Types

open System
open FSharp.Compiler.Symbols
open FSharp.Compiler.Text

/// Unique identifier for nodes in the PSG
type NodeId = 
    | SymbolNode of symbolHash: int * symbolName: string
    | RangeNode of fileName: string * startLine: int * startCol: int * endLine: int * endCol: int
    
    member this.Value =
        match this with
        | SymbolNode(hash, name) -> sprintf "sym_%s_%08x" name hash
        | RangeNode(file, sl, sc, el, ec) -> 
            sprintf "rng_%s_%d_%d_%d_%d" 
                (System.IO.Path.GetFileNameWithoutExtension file) sl sc el ec
    
    static member FromSymbol(symbol: FSharpSymbol) =
        SymbolNode(symbol.GetHashCode(), symbol.DisplayName.Replace(".", "_"))
    
    static member FromRange(fileName: string, range: range) =
        RangeNode(fileName, range.StartLine, range.StartColumn, range.EndLine, range.EndColumn)

    static member FromRangeWithKind(fileName: string, range: range, syntaxKind: string) =
        RangeNode(fileName, range.StartLine, range.StartColumn, range.EndLine, range.EndColumn)

/// Types of control flow
type ControlFlowKind =
    | Sequential
    | Conditional
    | Loop
    | Match
    | Exception

/// Comprehensive edge types for complete graph representation
type EdgeKind =
    | SymbolDef       // Symbol definition site
    | SymbolUse       // Symbol usage
    | FunctionCall    // Direct function invocation (renamed from CallsFunction)
    | TypeInstantiation of typeArgs: FSharpType list
    | ControlFlow of kind: ControlFlowKind
    | DataDependency
    | ModuleContainment
    | TypeMembership
    | ChildOf         // Parent-child relationship (kept for compatibility)
    | SymRef          // Symbol reference (kept for compatibility)
    | TypeOf          // Type relationship (kept for compatibility)
    | Instantiates    // Generic instantiation (kept for compatibility)

/// PSG node
type PSGNode = {
    Id: NodeId
    SyntaxKind: string
    Symbol: FSharpSymbol option
    Range: range
    SourceFile: string
    ParentId: NodeId option
    Children: NodeId list
}

/// Edge between PSG nodes
type PSGEdge = {
    Source: NodeId
    Target: NodeId
    Kind: EdgeKind
}

/// Symbol relationship types for analysis
type SymbolRelation =
    | DefinesType of FSharpEntity
    | UsesType of FSharpEntity
    | CallsSymbol of FSharpMemberOrFunctionOrValue      // Renamed from CallsFunction
    | ImplementsInterface of FSharpEntity
    | InheritsFrom of FSharpEntity
    | ReferencesSymbol of FSharpSymbol                  // Renamed from References

/// Complete Program Semantic Graph - keeping original structure
type ProgramSemanticGraph = {
    Nodes: Map<string, PSGNode>
    Edges: PSGEdge list
    SymbolTable: Map<string, FSharpSymbol>
    EntryPoints: NodeId list
    SourceFiles: Map<string, string>
    CompilationOrder: string list
}

/// Result type for PSG operations
type PSGResult<'T> = 
    | Success of 'T
    | Failure of PSGError list

and PSGError = {
    Message: string
    Location: range option
    ErrorKind: PSGErrorKind
}

and PSGErrorKind =
    | CorrelationFailure
    | MissingSymbol
    | InvalidNode
    | BuilderError
    | TypeResolutionError
    | MemoryAnalysisError