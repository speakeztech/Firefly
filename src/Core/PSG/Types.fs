module Core.PSG.Types

open System
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
        RangeNode(file, range.StartLine, range.StartColumn, range.EndLine, range.EndColumn)

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
    | References       // Symbol reference
    | TypeOf          // Type relationship
    | CallsFunction   // Function call
    | Instantiates    // Generic instantiation

/// Directed edge in the PSG
type PSGEdge = {
    Source: NodeId
    Target: NodeId
    Kind: EdgeKind
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