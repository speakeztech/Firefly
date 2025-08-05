module Core.PSG.Types

open System
open FSharp.Compiler.Text
open FSharp.Compiler.Symbols

/// Unique identifier for nodes in the PSG
type NodeId = {
    Value: string
}
with
    static member Create(value: string) = { Value = value }
    static member FromRange(fileName: string, range: range) =
        let cleanFileName = System.IO.Path.GetFileNameWithoutExtension(fileName)
        let rangeStr = sprintf "%d_%d_%d_%d" range.Start.Line range.Start.Column range.End.Line range.End.Column
        { Value = sprintf "rng_%s_%s" cleanFileName rangeStr }
    static member FromSymbol(symbol: FSharpSymbol) =
        let hashCode = symbol.GetHashCode().ToString("x8")
        { Value = sprintf "sym_%s_%s" symbol.DisplayName hashCode }

/// Child processing state with compile-time guarantees
type ChildrenState =
    | NotProcessed
    | Parent of NodeId list
    | NoChildren

/// Types of edges in the PSG
type EdgeKind =
    | ChildOf
    | FunctionCall
    | SymRef
    | TypeOf
    | Instantiates
    | SymbolDef
    | SymbolUse
    | TypeInstantiation of typeArgs: FSharpType list
    | ControlFlow of kind: string
    | DataDependency
    | ModuleContainment

/// Edge in the program semantic graph
type PSGEdge = {
    Source: NodeId
    Target: NodeId
    Kind: EdgeKind
}

/// Symbol relationship types for analysis
type SymbolRelation =
    | DefinesType of FSharpEntity
    | UsesType of FSharpEntity
    | CallsSymbol of FSharpMemberOrFunctionOrValue
    | ImplementsInterface of FSharpEntity
    | InheritsFrom of FSharpEntity
    | ReferencesSymbol of FSharpSymbol

/// Context requirements for continuation compilation decisions
type ContextRequirement =
    | Pure              // No external dependencies
    | AsyncBoundary     // Suspension point
    | ResourceAccess    // File/network access

/// Computation patterns for optimization decisions
type ComputationPattern =
    | DataDriven        // Push-based, eager evaluation
    | DemandDriven      // Pull-based, lazy evaluation

/// PSG node with soft-delete support added to existing structure
type PSGNode = {
    // EXISTING FIELDS - DO NOT CHANGE
    Id: NodeId
    SyntaxKind: string
    Symbol: FSharpSymbol option
    Type: FSharpType option          
    Constraints: FSharpGenericParameterConstraint list option  
    Range: range
    SourceFile: string
    ParentId: NodeId option
    Children: ChildrenState
    
    // NEW FIELDS - Soft-delete support
    IsReachable: bool
    EliminationPass: int option
    EliminationReason: string option
    ReachabilityDistance: int option
    
    // NEW FIELDS - Context tracking for continuation compilation
    ContextRequirement: ContextRequirement option
    ComputationPattern: ComputationPattern option
}

/// Complete Program Semantic Graph
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
        Type = None                  
        Constraints = None           
        Range = range
        SourceFile = sourceFile
        ParentId = parentId
        Children = NotProcessed
        
        // Initialize new soft-delete fields with defaults
        IsReachable = false  
        EliminationPass = None
        EliminationReason = None
        ReachabilityDistance = None
        
        // Initialize context tracking fields
        ContextRequirement = None
        ComputationPattern = None
    }
    
    /// Add a child to a node's children state
    let addChild childId node =
        match node.Children with
        | NotProcessed -> { node with Children = Parent [childId] }
        | Parent existingChildren -> { node with Children = Parent (childId :: existingChildren) }
        | NoChildren -> { node with Children = Parent [childId] }
    
    /// Finalize a node's children state
    let finalizeChildren node =
        match node.Children with
        | NotProcessed -> { node with Children = NoChildren }
        | other -> node
    
    /// Get children as list for compatibility with DebugOutput.fs
    let getChildrenList node =
        match node.Children with
        | NotProcessed -> []
        | NoChildren -> []
        | Parent children -> children

/// Helper functions for soft-delete reachability
module ReachabilityHelpers =
    
    /// Mark a node as reachable
    let markReachable (distance: int) (node: PSGNode) =
        { node with 
            IsReachable = true
            ReachabilityDistance = Some distance
            EliminationPass = None
            EliminationReason = None }
    
    /// Mark a node as unreachable
    let markUnreachable (pass: int) (reason: string) (node: PSGNode) =
        { node with 
            IsReachable = false
            EliminationPass = Some pass
            EliminationReason = Some reason
            ReachabilityDistance = None }
    
    /// Check if node is reachable
    let isReachable (node: PSGNode) = node.IsReachable
    
    /// Analyze context requirement from syntax kind and symbol
    let analyzeContextRequirement (node: PSGNode) : ContextRequirement option =
        match node.SyntaxKind, node.Symbol with
        | sk, _ when sk.Contains("Async") || sk.Contains("async") -> Some AsyncBoundary
        | sk, _ when sk.Contains("File") || sk.Contains("IO") || sk.Contains("Network") -> Some ResourceAccess
        | _, Some symbol ->
            match symbol with
            | :? FSharpMemberOrFunctionOrValue as mfv ->
                if mfv.FullName.Contains("Async") || mfv.FullName.Contains("async") then
                    Some AsyncBoundary
                elif mfv.FullName.Contains("File") || mfv.FullName.Contains("IO") then
                    Some ResourceAccess
                else
                    Some Pure
            | _ -> Some Pure
        | _ -> None
    
    /// Analyze computation pattern from node structure
    let analyzeComputationPattern (node: PSGNode) : ComputationPattern option =
        match node.SyntaxKind with
        | sk when sk.Contains("Seq") || sk.Contains("Lazy") || sk.Contains("seq") -> Some DemandDriven
        | sk when sk.Contains("List") || sk.Contains("Array") -> Some DataDriven
        | _ ->
            match node.Symbol with
            | Some symbol ->
                if symbol.FullName.Contains("Seq") || symbol.FullName.Contains("Lazy") then
                    Some DemandDriven
                else
                    Some DataDriven
            | None -> None
    
    /// Update node with analyzed context
    let updateNodeContext (node: PSGNode) =
        { node with
            ContextRequirement = node.ContextRequirement |> Option.orElse (analyzeContextRequirement node)
            ComputationPattern = node.ComputationPattern |> Option.orElse (analyzeComputationPattern node) }