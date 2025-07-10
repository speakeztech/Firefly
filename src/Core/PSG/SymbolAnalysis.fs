module Core.PSG.SymbolAnalysis

open FSharp.Compiler.Symbols
open FSharp.Compiler.Text
open FSharp.Compiler.CodeAnalysis  // Add this for FSharpSymbolUse
open Core.FCS.Helpers  // Import our helper functions

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
        |> Array.groupBy (fun (symbolUse: FSharpSymbolUse) -> 
            symbolUse.Range.FileName, symbolUse.Range.Start.Line)  // Use Start.Line not StartLine
        |> Map.ofArray
    
    // Find containing symbol for each use
    let findContainingSymbol (symbolUse: FSharpSymbolUse) =
        let key = (symbolUse.Range.FileName, symbolUse.Range.Start.Line)
        match Map.tryFind key usesByPosition with
        | Some uses ->
            uses 
            |> Array.tryFind (fun (u: FSharpSymbolUse) -> 
                u.IsFromDefinition && 
                u.Range.Start.Line <= symbolUse.Range.Start.Line &&
                u.Range.End.Line >= symbolUse.Range.End.Line &&
                u <> symbolUse
            )
            |> Option.map (fun u -> u.Symbol)
        | None -> None
    
    // Build relationships
    symbolUses
    |> Array.choose (fun (symbolUse: FSharpSymbolUse) ->
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
                    useSym.Range.StartLine >= def.Range.StartLine &&
                    useSym.Range.EndLine <= def.Range.EndLine
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