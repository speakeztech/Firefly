module Core.PSG.SymbolAnalysis

open FSharp.Compiler.Symbols
open FSharp.Compiler.CodeAnalysis
open Core.FCS.Helpers
open Core.PSG.Types

/// Extract relationships from the existing PSG instead of rebuilding from scratch
let extractRelationships (psg: ProgramSemanticGraph) : SymbolRelation[] =
    printfn "[RELATIONSHIP] === Using Existing PSG ==="
    printfn "[RELATIONSHIP] PSG has %d nodes and %d edges" psg.Nodes.Count psg.Edges.Length
    
    let relationships = 
        psg.Edges
        |> List.choose (fun edge ->
            // Get source and target nodes from PSG using NodeId.Value as key
            match Map.tryFind edge.Source.Value psg.Nodes, Map.tryFind edge.Target.Value psg.Nodes with
            | Some sourceNode, Some targetNode ->
                // Convert to SymbolRelation if both nodes have symbols
                match sourceNode.Symbol, targetNode.Symbol with
                | Some fromSymbol, Some toSymbol ->
                    let relationType = 
                        match edge.Kind with
                        | CallsFunction -> Calls
                        | SymRef -> References
                        | ChildOf -> Contains
                        | TypeOf -> Inherits
                        | Instantiates -> Implements
                    
                    // Debug main/hello relationships
                    if fromSymbol.DisplayName.Contains("main") || 
                       toSymbol.DisplayName.Contains("hello") then
                        printfn "[RELATIONSHIP] ✅ %s -> %s (%A)" 
                            fromSymbol.FullName toSymbol.FullName relationType
                    
                    Some {
                        From = fromSymbol
                        To = toSymbol
                        RelationType = relationType
                        Location = sourceNode.Range
                    }
                | _ -> None
            | _ -> None
        )
        |> List.toArray
    
    printfn "[RELATIONSHIP] Converted %d PSG edges to %d symbol relationships" 
        psg.Edges.Length relationships.Length
    
    // Check for entry point relationships
    let entryPointRels = 
        relationships 
        |> Array.filter (fun r -> 
            r.From.DisplayName.Contains("main") || 
            r.From.FullName.Contains("HelloWorldDirect"))
    
    if entryPointRels.Length > 0 then
        printfn "[RELATIONSHIP] ✅ Entry point relationships from PSG:"
        entryPointRels |> Array.iter (fun r ->
            printfn "  %s -> %s" r.From.FullName r.To.FullName)
    else
        printfn "[RELATIONSHIP] ⚠️ No entry point relationships in PSG"
    
    relationships
    
/// Get all symbols defined in a specific file
let getFileSymbols (fileName: string) (symbolUses: FSharpSymbolUse[]) =
    symbolUses
    |> Array.filter (fun (symbolUse: FSharpSymbolUse) -> 
        symbolUse.IsFromDefinition && 
        symbolUse.Range.FileName.EndsWith(fileName)
    )
    |> Array.map (fun (symbolUse: FSharpSymbolUse) -> symbolUse.Symbol)
    |> Array.distinct

/// Find all references to a symbol
let findReferences (symbol: FSharpSymbol) (symbolUses: FSharpSymbolUse[]) =
    symbolUses
    |> Array.filter (fun (symbolUse: FSharpSymbolUse) -> 
        symbolUse.Symbol = symbol && symbolUse.IsFromUse
    )

/// Find symbol definition
let findDefinition (symbol: FSharpSymbol) (symbolUses: FSharpSymbolUse[]) =
    symbolUses
    |> Array.tryFind (fun (symbolUse: FSharpSymbolUse) -> 
        symbolUse.Symbol = symbol && symbolUse.IsFromDefinition
    )

/// Group symbols by containing module/namespace
let groupByContainer (symbolUses: FSharpSymbolUse[]) =
    symbolUses
    |> Array.filter (fun (symbolUse: FSharpSymbolUse) -> symbolUse.IsFromDefinition)
    |> Array.groupBy (fun (symbolUse: FSharpSymbolUse) ->
        let symbol = symbolUse.Symbol
        match symbol with
        | :? FSharpEntity as entity when entity.IsFSharpModule || entity.IsNamespace ->
            entity.FullName
        | _ ->
            // Use helper function to get declaring entity
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

let calculateMetrics (symbol: FSharpSymbol) (relationships: SymbolRelation[]) =
    let inDegree = 
        relationships 
        |> Array.filter (fun r -> r.To = symbol)
        |> Array.length
    
    let outDegree = 
        relationships 
        |> Array.filter (fun r -> r.From = symbol)
        |> Array.length
    
    let depth =
        let rec countDepth (s: FSharpSymbol) acc =
            match getDeclaringEntity s with  // Use helper function
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
    let allSymbolUses = checkResults.GetAllUsesOfAllSymbols()
    
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
        let key = (range.FileName, range.Start.Line, range.Start.Column, range.End.Line, range.End.Column)
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
            mfv.Attributes 
            |> Seq.exists (fun attr -> 
                attr.AttributeType.FullName = "Microsoft.FSharp.Core.EntryPointAttribute" ||
                attr.AttributeType.FullName = "System.STAThreadAttribute"
            ) ||
            // Also check for main function pattern
            (mfv.DisplayName = "main" && mfv.IsModuleValueOrMember)
        | _ -> false
    )
    |> Array.map (fun symbolUse -> symbolUse.Symbol)

/// Extract symbols in definition order for a file
let getFileSymbolsInOrder (fileName: string) (symbolUses: FSharpSymbolUse[]) =
    symbolUses
    |> Array.filter (fun symbolUse -> 
        symbolUse.IsFromDefinition && 
        symbolUse.Range.FileName = fileName
    )
    |> Array.sortBy (fun symbolUse -> 
        symbolUse.Range.Start.Line, symbolUse.Range.Start.Column
    )
    |> Array.map (fun symbolUse -> symbolUse.Symbol, symbolUse.Range)

/// Build symbol dependency graph for reachability analysis
type SymbolDependency = {
    Symbol: FSharpSymbol
    DirectDependencies: FSharpSymbol[]
    TransitiveClosure: FSharpSymbol[] option  // Computed lazily
}

let buildDependencyGraph (symbolCorrelations: SymbolCorrelation[]) =
    let symbolMap = 
        symbolCorrelations 
        |> Array.map (fun sc -> sc.Hash, sc)
        |> Map.ofArray
    
    symbolCorrelations
    |> Array.map (fun correlation ->
        // Find all symbols referenced in the definition scope
        let dependencies = 
            match correlation.Definition with
            | Some def ->
                // Get all uses in the same range as the definition
                correlation.Uses
                |> Array.filter (fun useSym ->
                    useSym.Range.FileName = def.Range.FileName &&
                    useSym.Range.Start.Line >= def.Range.Start.Line &&
                    useSym.Range.End.Line <= def.Range.End.Line
                )
                |> Array.map (fun useSym -> useSym.Symbol)
                |> Array.distinct
            | None -> [||]
        
        {
            Symbol = correlation.Symbol
            DirectDependencies = dependencies
            TransitiveClosure = None
        }
    )