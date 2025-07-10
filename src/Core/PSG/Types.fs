module Core.PSG.Types

open FSharp.Compiler.Symbols
open FSharp.Compiler.Text

/// Node identifier using symbol hash or range fallback
type NodeId = 
    | SymbolNode of symbolHash: int * symbolName: string
    | RangeNode of file: string * startLine: int * startCol: int * endLine: int * endCol: int
    
    member this.Value =
        match this with
        | SymbolNode(hash, name) -> sprintf "sym_%s_%08x" name hash
        | RangeNode(file, sl, sc, el, ec) -> 
            sprintf "rng_%s_%d_%d_%d_%d" 
                (System.IO.Path.GetFileNameWithoutExtension file) sl sc el ec
    
    static member FromSymbol(symbol: FSharpSymbol) =
        SymbolNode(symbol.GetHashCode(), symbol.DisplayName)
    
    static member FromRange(file: string, range: range) =
        RangeNode(file, range.Start.Line, range.Start.Column, range.End.Line, range.End.Column)

/// Unified PSG node containing both syntax and semantic information
type PSGNode = {
    Id: NodeId
    SyntaxKind: string  // "Binding", "Expression", "Type", etc.
    Symbol: FSharpSymbol option
    Range: range
    SourceFile: string
    ParentId: NodeId option
    Children: NodeId list
}

/// Edge types in the PSG
type EdgeKind =
    | ChildOf          // Syntactic parent-child
    | SymRef       // Symbol reference
    | TypeOf          // Type relationship
    | CallsFunction   // Function call
    | Instantiates    // Generic instantiation

/// Directed edge in the PSG
type PSGEdge = {
    Source: NodeId
    Target: NodeId
    Kind: EdgeKind
}

/// Relationship types for reachability analysis
type RelationType =
    | Calls           // Function call
    | References      // Type or value reference
    | Inherits        // Type inheritance
    | Implements      // Interface implementation
    | Contains        // Module/namespace containment

/// Symbol relationship for reachability analysis
type SymbolRelation = {
    From: FSharpSymbol
    To: FSharpSymbol
    RelationType: RelationType
    Location: range
}

/// The complete Program Semantic Graph
type ProgramSemanticGraph = {
    /// All nodes indexed by their ID
    Nodes: Map<string, PSGNode>
    
    /// All edges in the graph
    Edges: PSGEdge list
    
    /// Symbol table for quick lookup
    SymbolTable: Map<string, FSharpSymbol>
    
    /// Entry point nodes
    EntryPoints: NodeId list
    
    /// Source file content for reference
    SourceFiles: Map<string, string>
    
    /// Compilation order
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

/// JSON-friendly representation of a PSG node
type NodeJson = {
    Id: string
    Kind: string
    Symbol: string option
    Range: {| StartLine: int; StartColumn: int; EndLine: int; EndColumn: int |}
    SourceFile: string
    ParentId: string option
    Children: string[]
}

/// JSON-friendly representation of an edge
type EdgeJson = {
    Source: string
    Target: string
    Kind: string
}

/// JSON-friendly representation of correlation entry
type CorrelationJson = {
    Range: {| File: string; StartLine: int; StartColumn: int; EndLine: int; EndColumn: int |}
    SymbolName: string
    SymbolKind: string
    SymbolHash: int
}

/// Semantic unit representing a cohesive component
type SemanticUnit = 
    | Module of FSharpEntity
    | Namespace of string
    | FunctionGroup of FSharpMemberOrFunctionOrValue list
    | TypeCluster of FSharpEntity list

/// Coupling measurement between semantic units
type Coupling = {
    From: SemanticUnit
    To: SemanticUnit
    Strength: float  // 0.0 to 1.0
    Dependencies: SymbolRelation list
}

/// Cohesion measurement within a semantic unit
type Cohesion = {
    Unit: SemanticUnit
    Score: float  // 0.0 to 1.0
    InternalRelations: int
    ExternalRelations: int
}

/// Component identified through coupling/cohesion analysis
type CodeComponent = {
    Id: string
    Units: SemanticUnit list
    Cohesion: float
    AverageCoupling: float
    Boundaries: ComponentBoundary list
}

and ComponentBoundary = {
    Interface: FSharpSymbol list
    Direction: BoundaryDirection
}

and BoundaryDirection = Inbound | Outbound | Bidirectional

