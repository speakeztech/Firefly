module Core.FCS.SymbolAnalysis

open FSharp.Compiler.Symbols
open FSharp.Compiler.Text

/// Symbol relationship extracted from FCS
type SymbolRelation = {
    From: FSharpSymbol
    To: FSharpSymbol
    RelationType: RelationType
    Location: range
}

and RelationType =
    | Calls           // Function call
    | References      // Type or value reference
    | Inherits        // Type inheritance
    | Implements      // Interface implementation
    | Contains        // Module/namespace containment

/// Extract all symbol relationships from project
let extractRelationships (symbolUses: FSharpSymbolUse[]) =
    // Group by file and position for efficient processing
    let usesByPosition = 
        symbolUses
        |> Array.groupBy (fun symbolUse -> symbolUse.Range.FileName, symbolUse.Range.StartLine)
        |> Map.ofArray
    
    // Find containing symbol for each use
    let findContainingSymbol (symbolUse: FSharpSymbolUse) =
        let key = (symbolUse.Range.FileName, symbolUse.Range.StartLine)
        match Map.tryFind key usesByPosition with
        | Some uses ->
            uses 
            |> Array.tryFind (fun u -> 
                u.IsFromDefinition && 
                u.Range.StartLine <= symbolUse.Range.StartLine &&
                u.Range.EndLine >= symbolUse.Range.EndLine &&
                u <> symbolUse
            )
            |> Option.map (fun u -> u.Symbol)
        | None -> None
    
    // Build relationships
    symbolUses
    |> Array.choose (fun symbolUse ->
        if symbolUse.IsFromUse then
            match findContainingSymbol symbolUse with
            | Some containerSymbol ->
                let relationType = 
                    match symbolUse.Symbol with
                    | :? FSharpMemberOrFunctionOrValue -> RelationType.Calls
                    | :? FSharpEntity -> RelationType.References
                    | _ -> RelationType.References
                
                Some {
                    From = containerSymbol
                    To = symbolUse.Symbol
                    RelationType = relationType
                    Location = symbolUse.Range
                }
            | None -> None
        else None
    )

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
        match symbolUse.Symbol with
        | :? FSharpEntity as entity when entity.IsFSharpModule || entity.IsNamespace ->
            entity.FullName
        | symbol ->
            match symbol.DeclaringEntity with
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
            match s.DeclaringEntity with
            | Some entity -> countDepth entity (acc + 1)
            | None -> acc
        countDepth symbol 0
    
    {
        InDegree = inDegree
        OutDegree = outDegree
        Depth = depth
    }