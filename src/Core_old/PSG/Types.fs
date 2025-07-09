module Core.PSG.Types

open System
open FSharp.Compiler.Syntax
open FSharp.Compiler.Symbols
open FSharp.Compiler.Text

/// Unique identifier for PSG nodes
[<Struct>]
type NodeId = 
    private | NodeId of string
    
    /// Creates a new node ID with an optional prefix for categorization
    static member Create(prefix: string, uniqueValue: string) : NodeId =
        NodeId (sprintf "%s_%s" prefix (uniqueValue.Replace(" ", "_")))
        
    /// Creates a deterministic node ID from a symbol
    static member FromSymbol(symbol: FSharpSymbol) : NodeId =
        let symbolId = 
            match symbol with
            | :? FSharpMemberOrFunctionOrValue as mfv -> 
                sprintf "%s_%s_%d_%d" 
                    (if mfv.IsModuleValueOrMember then "val" else "func")
                    mfv.LogicalName 
                    mfv.DeclarationLocation.Start.Line
                    mfv.DeclarationLocation.Start.Column
            | :? FSharpEntity as ent -> 
                sprintf "type_%s_%d_%d" 
                    ent.LogicalName
                    ent.DeclarationLocation.Start.Line
                    ent.DeclarationLocation.Start.Column
            | _ -> 
                sprintf "sym_%s_%d" 
                    (symbol.GetType().Name)
                    (symbol.GetHashCode())
        NodeId symbolId
        
    /// Creates a node ID from an AST node with location information
    static member FromSyntaxNode(nodeType: string, range: range) : NodeId =
        let fileName = System.IO.Path.GetFileNameWithoutExtension(range.FileName)
        let idString = sprintf "%s_%s_%d_%d" 
                        nodeType 
                        fileName
                        range.Start.Line 
                        range.Start.Column
        NodeId idString
                
    /// Get the string representation
    member this.Value = 
        let (NodeId value) = this
        value
        
    override this.ToString() = this.Value

/// Comparable wrapper for FSharp.Compiler.Text.range
[<Struct; CustomEquality; CustomComparison>]
type RangeKey =
    { FileName: string
      StartLine: int
      StartColumn: int
      EndLine: int
      EndColumn: int }
    
    static member FromRange(range: range) =
        { FileName = range.FileName
          StartLine = range.Start.Line
          StartColumn = range.Start.Column
          EndLine = range.End.Line
          EndColumn = range.End.Column }
          
    override this.Equals(obj) =
        match obj with
        | :? RangeKey as other -> 
            this.FileName = other.FileName &&
            this.StartLine = other.StartLine &&
            this.StartColumn = other.StartColumn &&
            this.EndLine = other.EndLine &&
            this.EndColumn = other.EndColumn
        | _ -> false
    
    override this.GetHashCode() =
        hash (this.FileName, this.StartLine, this.StartColumn, this.EndLine, this.EndColumn)
        
    interface System.IComparable with
        member this.CompareTo(obj) =
            match obj with
            | :? RangeKey as other ->
                let fileCompare = this.FileName.CompareTo(other.FileName)
                if fileCompare <> 0 then fileCompare
                else
                    let startLineCompare = this.StartLine.CompareTo(other.StartLine)
                    if startLineCompare <> 0 then startLineCompare
                    else
                        let startColCompare = this.StartColumn.CompareTo(other.StartColumn)
                        if startColCompare <> 0 then startColCompare
                        else
                            let endLineCompare = this.EndLine.CompareTo(other.EndLine)
                            if endLineCompare <> 0 then endLineCompare
                            else
                                this.EndColumn.CompareTo(other.EndColumn)
            | _ -> raise (InvalidOperationException("Cannot compare RangeKey with other types"))

/// MLIR dialect name with validation
[<Struct>]
type DialectName = 
    private | DialectName of string
    
    static member Create(name: string) : Result<DialectName, string> =
        match name.ToLowerInvariant() with
        | "std" | "llvm" | "func" | "arith" | "scf" | "memref" 
        | "index" | "affine" | "builtin" -> Ok (DialectName name)
        | _ -> Error (sprintf "Unknown MLIR dialect: '%s'" name)
        
    member this.Value = 
        let (DialectName value) = this
        value
        
    override this.ToString() = this.Value

/// MLIR operation name with validation
[<Struct>]
type OperationName = 
    private | OperationName of string
    
    static member Create(name: string) : Result<OperationName, string> =
        if String.IsNullOrWhiteSpace(name) then
            Error "Operation name cannot be empty"
        elif name.Contains(" ") then
            Error "Operation name cannot contain spaces"
        else
            Ok (OperationName name)
            
    member this.Value = 
        let (OperationName value) = this
        value
        
    override this.ToString() = this.Value

/// Parameter name for MLIR operation
[<Struct>]
type ParameterName = ParameterName of string with
    member this.Value = let (ParameterName v) = this in v
    override this.ToString() = this.Value

/// Parameter value for MLIR operation  
[<Struct>]
type ParameterValue = ParameterValue of string with
    member this.Value = let (ParameterValue v) = this in v
    override this.ToString() = this.Value

/// Attribute name for MLIR operation
[<Struct>]
type AttributeName = AttributeName of string with
    member this.Value = let (AttributeName v) = this in v
    override this.ToString() = this.Value

/// Attribute value for MLIR operation
[<Struct>]
type AttributeValue = AttributeValue of string with
    member this.Value = let (AttributeValue v) = this in v
    override this.ToString() = this.Value

/// Metadata extracted from XML documentation
type MLIRMetadata = {
    Dialect: DialectName option
    Operation: OperationName option
    Parameters: Map<ParameterName, ParameterValue>
    Attributes: Map<AttributeName, AttributeValue>
}

/// Location information with source mapping
type SourceLocation = {
    Range: range
    OriginalSourceText: string option
}

/// Result of parsing MLIR metadata from XML documentation
type MLIRMetadataParseResult =
    | Valid of MLIRMetadata
    | Missing
    | Invalid of string list // Error messages

/// Enriched node with both syntactic and semantic information
type EnrichedNode<'TNode> = {
    Syntax: 'TNode
    Symbol: FSharpSymbol option
    Metadata: MLIRMetadata option
    SourceLocation: SourceLocation
    Id: NodeId
    ParentId: NodeId option
    Children: NodeId list
}

/// Complete program representation
type ProgramSemanticGraph = {
    // Original parsed input (for reference)
    SourceASTs: Map<string, ParsedInput>
    
    // Enriched top-level nodes
    ModuleNodes: Map<NodeId, EnrichedNode<SynModuleOrNamespace>>
    TypeNodes: Map<NodeId, EnrichedNode<SynTypeDefn>>
    ValueNodes: Map<NodeId, EnrichedNode<SynBinding>>
    ExpressionNodes: Map<NodeId, EnrichedNode<SynExpr>>
    PatternNodes: Map<NodeId, EnrichedNode<SynPat>>
    
    // Cross-reference maps
    SymbolTable: Map<string, FSharpSymbol>
    RangeToSymbol: Map<RangeKey, FSharpSymbol>
    SymbolToNodes: Map<string, Set<NodeId>>
    
    // Type information
    TypeDefinitions: Map<string, FSharpEntity>
    
    // Dependencies
    DependencyGraph: Map<NodeId, Set<NodeId>>
    
    // Compilation context
    EntryPoints: NodeId list
    AlloyReferences: Set<string>
    
    // Source file information
    SourceFiles: Map<string, string>
}

/// Diagnostic severity level
type DiagnosticSeverity =
    | Error
    | Warning
    | Info

/// Diagnostic message with source location
type DiagnosticMessage = {
    Severity: DiagnosticSeverity
    Code: string
    Message: string
    Location: range option
    RelatedLocations: (range * string) list
}

/// Result type specific to F# Compiler Services operations
type FcsResult<'T> =
    | Success of 'T
    | Failure of DiagnosticMessage list
