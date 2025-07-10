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

/// Fixed relationship extraction with proper range containment
let extractRelationships (symbolUses: FSharpSymbolUse[]) =
    // Group definitions by file for efficient lookup
    let definitionsByFile = 
        symbolUses
        |> Array.filter (fun su -> su.IsFromDefinition)
        |> Array.groupBy (fun su -> su.Range.FileName)
        |> Map.ofArray
    
    // Find containing symbol using proper range containment
    let findContainingSymbol (symbolUse: FSharpSymbolUse) =
        match Map.tryFind symbolUse.Range.FileName definitionsByFile with
        | Some definitions ->
            definitions
            |> Array.filter (fun def ->
                let useRange = symbolUse.Range
                let defRange = def.Range
                
                // Proper range containment: definition must completely contain the use
                (defRange.Start.Line < useRange.Start.Line || 
                 (defRange.Start.Line = useRange.Start.Line && defRange.Start.Column <= useRange.Start.Column)) &&
                (defRange.End.Line > useRange.End.Line || 
                 (defRange.End.Line = useRange.End.Line && defRange.End.Column >= useRange.End.Column)) &&
                def.Symbol <> symbolUse.Symbol  // Not the same symbol
            )
            |> Array.sortBy (fun def -> 
                // Find the most specific (smallest) containing definition
                let defRange = def.Range
                (defRange.End.Line - defRange.Start.Line), 
                (defRange.End.Column - defRange.Start.Column)
            )
            |> Array.tryHead
            |> Option.map (fun def -> def.Symbol)
        | None -> None
    
    // Build relationships with detailed debugging
    let relationships = 
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
                    
                    // Debug critical relationships
                    if containerSymbol.DisplayName = "main" || symbolUse.Symbol.DisplayName = "hello" then
                        printfn "[RELATIONSHIP] %s -> %s (at %s:%d)" 
                            containerSymbol.FullName
                            symbolUse.Symbol.FullName
                            (System.IO.Path.GetFileName(symbolUse.Range.FileName))
                            symbolUse.Range.Start.Line
                    
                    Some {
                        From = containerSymbol
                        To = symbolUse.Symbol
                        RelationType = relationType
                        Location = symbolUse.Range
                    }
                | None -> 
                    // Debug missing containers for entry point calls
                    if symbolUse.Symbol.DisplayName = "hello" then
                        printfn "[RELATIONSHIP] No container found for hello() call at %s:%d" 
                            (System.IO.Path.GetFileName(symbolUse.Range.FileName))
                            symbolUse.Range.Start.Line
                        
                        // Show nearby definitions for debugging
                        let fileName = symbolUse.Range.FileName
                        match Map.tryFind fileName definitionsByFile with
                        | Some defs ->
                            printfn "[RELATIONSHIP] Nearby definitions in %s:" (System.IO.Path.GetFileName(fileName))
                            defs 
                            |> Array.filter (fun def -> 
                                abs(def.Range.Start.Line - symbolUse.Range.Start.Line) <= 5)
                            |> Array.iter (fun def ->
                                printfn "  %s at line %d-%d" 
                                    def.Symbol.FullName 
                                    def.Range.Start.Line 
                                    def.Range.End.Line)
                        | None -> ()
                    None
            else None
        )
    
    // Additional debug: count relationships by type
    let callCount = relationships |> Array.filter (fun r -> r.RelationType = RelationType.Calls) |> Array.length
    let refCount = relationships |> Array.filter (fun r -> r.RelationType = RelationType.References) |> Array.length
    
    printfn "[RELATIONSHIP] Extracted %d total relationships (%d calls, %d references)" 
        relationships.Length callCount refCount
    
    // Show critical entry point relationships
    let entryPointRels = 
        relationships 
        |> Array.filter (fun r -> r.From.DisplayName = "main")
    
    if entryPointRels.Length > 0 then
        printfn "[RELATIONSHIP] Entry point relationships found:"
        entryPointRels |> Array.iter (fun r ->
            printfn "  main -> %s" r.To.FullName)
    else
        printfn "[RELATIONSHIP] ⚠️  NO entry point relationships found! This will break reachability."
    
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