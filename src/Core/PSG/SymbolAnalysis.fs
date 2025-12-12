module Core.PSG.SymbolAnalysis

open FSharp.Compiler.Symbols
open FSharp.Compiler.CodeAnalysis
open Core.FCS.Helpers
open Core.PSG.Types

/// Extract relationships from the existing PSG instead of rebuilding from scratch
let extractRelationships (psg: ProgramSemanticGraph) : SymbolRelation[] =
    let relationships = 
        psg.Edges
        |> List.choose (fun edge ->
            // Get source and target nodes from PSG using NodeId.Value as key
            match Map.tryFind edge.Source.Value psg.Nodes, Map.tryFind edge.Target.Value psg.Nodes with
            | Some sourceNode, Some targetNode ->
                // Convert to SymbolRelation if both nodes have symbols
                match sourceNode.Symbol, targetNode.Symbol with
                | Some fromSymbol, Some toSymbol ->
                    // Map EdgeKind to SymbolRelation
                    match edge.Kind with
                    | FunctionCall -> 
                        match toSymbol with
                        | :? FSharpMemberOrFunctionOrValue as mfv -> 
                            Some (CallsSymbol mfv)
                        | _ -> None
                    | SymRef | SymbolUse -> Some (ReferencesSymbol toSymbol)
                    | ModuleContainment | ChildOf -> None  // Skip structural relationships
                    | TypeOf -> 
                        match toSymbol with
                        | :? FSharpEntity as entity -> Some (InheritsFrom entity)
                        | _ -> None
                    | Instantiates | TypeInstantiation _ -> 
                        match toSymbol with
                        | :? FSharpEntity as entity -> Some (ImplementsInterface entity)
                        | _ -> None
                    | _ -> None
                | _ -> None
            | _ -> None
        )
        |> List.toArray

    relationships
    
/// Get all symbols defined in a specific file
let getFileSymbols (fileName: string) (symbolUses: FSharpSymbolUse[]) =
    symbolUses
    |> Array.filter (fun symbolUse -> 
        symbolUse.IsFromDefinition && 
        symbolUse.Range.FileName.EndsWith(fileName)
    )
    |> Array.map (fun symbolUse -> symbolUse.Symbol)
    |> Array.distinct

/// Find all references to a symbol
let findReferences (symbol: FSharpSymbol) (symbolUses: FSharpSymbolUse[]) =
    symbolUses
    |> Array.filter (fun symbolUse -> 
        symbolUse.Symbol = symbol && symbolUse.IsFromUse
    )

/// Find symbol definition
let findDefinition (symbol: FSharpSymbol) (symbolUses: FSharpSymbolUse[]) =
    symbolUses
    |> Array.tryFind (fun symbolUse -> 
        symbolUse.Symbol = symbol && symbolUse.IsFromDefinition
    )

/// Group symbols by containing module/namespace
let groupByContainer (symbolUses: FSharpSymbolUse[]) =
    symbolUses
    |> Array.filter (fun symbolUse -> symbolUse.IsFromDefinition)
    |> Array.groupBy (fun symbolUse ->
        let symbol = symbolUse.Symbol
        match getDeclaringEntity symbol with  
        | Some entity -> entity.FullName
        | None -> "<global>"
    )
    |> Map.ofArray

/// Calculate symbol complexity metrics
type SymbolMetrics = {
    InDegree: int      // How many symbols reference this one
    OutDegree: int     // How many symbols this one references
    Depth: int         // Nesting depth in module hierarchy
}

let calculateMetrics (symbol: FSharpSymbol) (psg: ProgramSemanticGraph) =
    // Calculate in-degree: count edges pointing to this symbol's nodes
    let symbolNodes = 
        psg.Nodes 
        |> Map.toSeq
        |> Seq.filter (fun (_, node) -> 
            node.Symbol |> Option.map (fun s -> s = symbol) |> Option.defaultValue false
        )
        |> Seq.map fst
        |> Set.ofSeq
    
    let inDegree = 
        psg.Edges
        |> List.filter (fun edge -> 
            Set.contains edge.Target.Value symbolNodes
        )
        |> List.length
    
    // Calculate out-degree: count edges originating from this symbol's nodes
    let outDegree = 
        psg.Edges
        |> List.filter (fun edge -> 
            Set.contains edge.Source.Value symbolNodes
        )
        |> List.length
    
    // Calculate depth in module hierarchy
    let depth =
        let rec countDepth (s: FSharpSymbol) acc =
            match getDeclaringEntity s with
            | Some entity -> countDepth (entity :> FSharpSymbol) (acc + 1)
            | None -> acc
        countDepth symbol 0
    
    {
        InDegree = inDegree
        OutDegree = outDegree
        Depth = depth
    }

// ===== PSG Correlation Extensions =====

/// Symbol correlation data for PSG building
type SymbolCorrelation = {
    Symbol: FSharpSymbol
    Definition: FSharpSymbolUse option
    Uses: FSharpSymbolUse[]
    Hash: int
}

/// Extract comprehensive symbol data for PSG correlation
let extractSymbolCorrelations (checkResults: FSharpCheckProjectResults) =
    // Get all symbol uses using FCS bulk extraction
    let allSymbolUses = checkResults.GetAllUsesOfAllSymbols() |> Array.ofSeq
    
    // Group by symbol for efficient correlation
    let symbolGroups = 
        allSymbolUses
        |> Array.groupBy (fun symbolUse -> symbolUse.Symbol)
        |> Array.map (fun (symbol, uses) ->
            let definition = uses |> Array.tryFind (fun u -> u.IsFromDefinition)
            let references = uses |> Array.filter (fun u -> u.IsFromUse)
            {
                Symbol = symbol
                Definition = definition
                Uses = references
                Hash = symbol.GetHashCode()
            }
        )
    
    symbolGroups

/// Build position-to-symbol mapping for range-based lookup
let buildPositionIndex (symbolUses: FSharpSymbolUse[]) =
    symbolUses
    |> Array.map (fun symbolUse ->
        let range = symbolUse.Range
        let key = (range.FileName, range.StartLine, range.StartColumn, range.EndLine, range.EndColumn)
        (key, symbolUse)
    )
    |> Map.ofArray

/// Get all entry point symbols from the project
let getEntryPointSymbols (symbolUses: FSharpSymbolUse[]) =
    symbolUses
    |> Array.filter (fun symbolUse ->
        symbolUse.IsFromDefinition &&
        match symbolUse.Symbol with
        | :? FSharpMemberOrFunctionOrValue as mfv ->
            // Check for EntryPoint attribute
            let hasEntryPointAttr =
                mfv.Attributes |> Seq.exists (fun attr ->
                    let displayName = attr.AttributeType.DisplayName
                    let fullName = attr.AttributeType.FullName
                    displayName = "EntryPointAttribute" ||
                    displayName = "EntryPoint" ||
                    fullName.EndsWith("EntryPointAttribute") ||
                    fullName.EndsWith("EntryPoint") ||
                    fullName = "System.EntryPointAttribute"
                )

            // Check for main function pattern
            let isMainFunc =
                (mfv.DisplayName = "main" || mfv.LogicalName = "main") &&
                mfv.IsModuleValueOrMember

            hasEntryPointAttr || isMainFunc
        | _ -> false
    )
    |> Array.map (fun symbolUse -> symbolUse.Symbol)
