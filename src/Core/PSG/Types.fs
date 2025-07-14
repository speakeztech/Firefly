module Core.PSG.Types

open FSharp.Compiler.Text
open FSharp.Compiler.Symbols

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
        RangeNode(fileName, range.Start.Line, range.Start.Column, range.End.Line, range.End.Column)

    static member FromRangeWithKind(fileName: string, range: range, syntaxKind: string) =
        RangeNode(fileName, range.Start.Line, range.Start.Column, range.End.Line, range.End.Column)

/// Explicit state representation for children relationships eliminating ambiguity
type ChildrenState =
    | NotProcessed                    // Children relationships not yet established during construction
    | Leaf                           // Affirmatively verified as having no children (terminal nodes)
    | Parent of NodeId list          // Verified parent with specific children

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

/// PSG node with explicit children state
type PSGNode = {
    Id: NodeId
    SyntaxKind: string
    Symbol: FSharpSymbol option
    Range: range
    SourceFile: string
    ParentId: NodeId option
    Children: ChildrenState          // Enhanced from NodeId list to explicit state
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

/// Helper functions for working with ChildrenState
module ChildrenStateHelpers =
    
    /// Create a new node with NotProcessed children state
    let createWithNotProcessed id syntaxKind symbol range sourceFile parentId = {
        Id = id
        SyntaxKind = syntaxKind
        Symbol = symbol
        Range = range
        SourceFile = sourceFile
        ParentId = parentId
        Children = NotProcessed
    }
    
    /// Add a child to a node's children state
    let addChild childId node =
        match node.Children with
        | NotProcessed -> { node with Children = Parent [childId] }
        | Leaf -> { node with Children = Parent [childId] }  // Convert leaf to parent
        | Parent existingChildren -> { node with Children = Parent (childId :: existingChildren) }
    
    /// Finalize a node's children state (convert NotProcessed to Leaf if no children added)
    let finalizeChildren node =
        match node.Children with
        | NotProcessed -> { node with Children = Leaf }
        | other -> node
    
    /// Get children as list for compatibility with existing code
    let getChildrenList node =
        match node.Children with
        | NotProcessed -> []
        | Leaf -> []
        | Parent children -> children