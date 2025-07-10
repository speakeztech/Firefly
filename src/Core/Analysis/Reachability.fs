module Core.Analysis.Reachability

open System
open FSharp.Compiler.Symbols
open FSharp.Compiler.CodeAnalysis  // For FSharpSymbolUse
open Core.FCS.SymbolAnalysis
open Core.FCS.Helpers              // For getDeclaringEntity
open Core.Analysis.CouplingCohesion // For CodeComponent type

/// Reachability analysis result
type ReachabilityResult = {
    EntryPoints: FSharpSymbol list
    ReachableSymbols: Set<string>        // Changed to Set<string>
    UnreachableSymbols: Set<string>      // Changed to Set<string>
    CallGraph: Map<string, string list>  // Changed to use strings
}

/// Generate dead code elimination opportunities
type EliminationOpportunity = {
    Symbol: FSharpSymbol
    Reason: EliminationReason
    EstimatedSaving: int // bytes
}

and EliminationReason =
    | Unreachable
    | UnusedComponent of componentId: string
    | DeadBranch
    | UnusedType

type LibraryCategory =
    | UserCode
    | AlloyLibrary  
    | FSharpCore
    | Other of libraryName: string

/// Enhanced reachability result with library boundary information
type LibraryAwareReachability = {
    BasicResult: ReachabilityResult
    LibraryCategories: Map<string, LibraryCategory>
    PruningStatistics: PruningStatistics
}

and PruningStatistics = {
    TotalSymbols: int
    ReachableSymbols: int
    EliminatedSymbols: int
    ComputationTimeMs: int64
}

/// Helper to get symbol identifier for comparison
let private getSymbolId (symbol: FSharpSymbol) = symbol.FullName

/// Helper function to estimate symbol size (simplified)
let private estimateSymbolSize (symbol: FSharpSymbol) =
    match symbol with
    | :? FSharpMemberOrFunctionOrValue as mfv ->
        if mfv.IsFunction then 100 else 50  // Rough estimate
    | :? FSharpEntity as entity ->
        if entity.IsFSharpModule then 200
        elif entity.IsFSharpRecord then 150
        else 100
    | _ -> 50

/// Find entry points in the project
let findEntryPoints (symbolUses: FSharpSymbolUse[]) =
    symbolUses
    |> Array.filter (fun (symbolUse: FSharpSymbolUse) -> symbolUse.IsFromDefinition)
    |> Array.choose (fun (symbolUse: FSharpSymbolUse) ->
        match symbolUse.Symbol with
        | :? FSharpMemberOrFunctionOrValue as mfv ->
            // Check for [<EntryPoint>] attribute
            let hasEntryPoint = 
                mfv.Attributes 
                |> Seq.exists (fun attr -> 
                    attr.AttributeType.DisplayName = "EntryPoint"
                )
            
            // Check for main function
            let isMain = 
                mfv.DisplayName = "main" && 
                mfv.IsModuleValueOrMember &&
                mfv.GenericParameters.Count = 0
            
            if hasEntryPoint || isMain then Some (mfv :> FSharpSymbol)
            else None
        | _ -> None
    )
    |> Array.distinct
    |> List.ofArray

/// Build call graph from symbol relationships
let buildCallGraph (relationships: SymbolRelation[]) =
    relationships
    |> Array.filter (fun r -> r.RelationType = RelationType.Calls)
    |> Array.groupBy (fun r -> getSymbolId r.From)
    |> Array.map (fun (callerId, calls) ->
        callerId, calls |> Array.map (fun r -> getSymbolId r.To) |> Array.distinct |> List.ofArray
    )
    |> Map.ofArray

/// Compute reachable symbols from entry points
let computeReachable (entryPoints: FSharpSymbol list) (callGraph: Map<string, string list>) =
    let rec traverse (visited: Set<string>) (currentId: string) =
        if Set.contains currentId visited then visited
        else
            let newVisited = Set.add currentId visited
            match Map.tryFind currentId callGraph with
            | Some calleeIds ->
                calleeIds |> List.fold traverse newVisited
            | None -> newVisited
    
    let entryPointIds = entryPoints |> List.map getSymbolId
    entryPointIds |> List.fold traverse Set.empty

/// Perform complete reachability analysis
let analyzeReachability (symbolUses: FSharpSymbolUse[]) (relationships: SymbolRelation[]) =
    // Find all defined symbols
    let allSymbolIds = 
        symbolUses
        |> Array.filter (fun (useSymbol: FSharpSymbolUse) -> useSymbol.IsFromDefinition)
        |> Array.map (fun (useSymbol: FSharpSymbolUse) -> getSymbolId useSymbol.Symbol)
        |> Set.ofArray
    
    // Find entry points
    let entryPoints = findEntryPoints symbolUses
    
    // Build call graph
    let callGraph = buildCallGraph relationships
    
    // Compute reachable symbols
    let reachableIds = computeReachable entryPoints callGraph
    
    // Find unreachable symbols
    let unreachableIds = Set.difference allSymbolIds reachableIds
    
    {
        EntryPoints = entryPoints
        ReachableSymbols = reachableIds
        UnreachableSymbols = unreachableIds
        CallGraph = callGraph
    }

/// Enhanced reachability with coupling/cohesion awareness
type EnhancedReachability = {
    BasicResult: ReachabilityResult
    ComponentReachability: Map<string, ComponentReachability>
}

and ComponentReachability = {
    ComponentId: string
    ReachableUnits: int
    TotalUnits: int
    PartiallyReachable: bool
}

/// Analyze reachability with component awareness
let analyzeComponentReachability (basicResult: ReachabilityResult) (codeComponents: CodeComponent list) =
    let componentReachability = 
        codeComponents
        |> List.map (fun codeComp ->
            let reachableInComponent = 
                codeComp.Units
                |> List.filter (fun compUnit ->
                    match compUnit with
                    | Module compEntity ->
                        Set.contains (getSymbolId (compEntity :> FSharpSymbol)) basicResult.ReachableSymbols
                    | FunctionGroup functions ->
                        functions |> List.exists (fun f -> 
                            Set.contains (getSymbolId (f :> FSharpSymbol)) basicResult.ReachableSymbols
                        )
                    | _ -> false
                )
                |> List.length
            
            let total = codeComp.Units.Length
            
            codeComp.Id, {
                ComponentId = codeComp.Id
                ReachableUnits = reachableInComponent
                TotalUnits = total
                PartiallyReachable = reachableInComponent > 0 && reachableInComponent < total
            }
        )
        |> Map.ofList
    
    {
        BasicResult = basicResult
        ComponentReachability = componentReachability
    }

/// Identify elimination opportunities
let identifyEliminationOpportunities (result: ReachabilityResult) (codeComponents: CodeComponent list) (symbolUses: FSharpSymbolUse[]) =
    // Create a map from symbol ID to symbol for lookup
    let symbolMap = 
        symbolUses
        |> Array.filter (fun su -> su.IsFromDefinition)
        |> Array.map (fun su -> getSymbolId su.Symbol, su.Symbol)
        |> Map.ofArray
    
    // Direct unreachable symbols
    let unreachableOpportunities = 
        result.UnreachableSymbols
        |> Set.toList
        |> List.choose (fun symbolId ->
            Map.tryFind symbolId symbolMap
            |> Option.map (fun symbol ->
                {
                    Symbol = symbol
                    Reason = Unreachable
                    EstimatedSaving = estimateSymbolSize symbol
                }
            )
        )
    
    // Unused components
    let unusedComponentOpportunities =
        codeComponents
        |> List.filter (fun codeComp ->
            codeComp.Units |> List.forall (fun unit ->
                match unit with
                | Module entity -> 
                    Set.contains (getSymbolId (entity :> FSharpSymbol)) result.UnreachableSymbols
                | FunctionGroup functions ->
                    functions |> List.forall (fun f ->
                        Set.contains (getSymbolId (f :> FSharpSymbol)) result.UnreachableSymbols
                    )
                | _ -> false
            )
        )
        |> List.collect (fun codeComp ->
            codeComp.Units |> List.collect (fun unit ->
                match unit with
                | Module entity -> 
                    [{
                        Symbol = entity :> FSharpSymbol
                        Reason = UnusedComponent codeComp.Id
                        EstimatedSaving = estimateSymbolSize (entity :> FSharpSymbol)
                    }]
                | FunctionGroup functions ->
                    functions |> List.map (fun f ->
                        {
                            Symbol = f :> FSharpSymbol
                            Reason = UnusedComponent codeComp.Id
                            EstimatedSaving = estimateSymbolSize (f :> FSharpSymbol)
                        }
                    )
                | _ -> []
            )
        )
    
    unreachableOpportunities @ unusedComponentOpportunities

/// Generate reachability report
let generateReport (result: EnhancedReachability) (opportunities: EliminationOpportunity list) =
    let basic = result.BasicResult
    let totalSymbols = Set.count basic.ReachableSymbols + Set.count basic.UnreachableSymbols
    let eliminationRate = 
        if totalSymbols > 0 then
            float (Set.count basic.UnreachableSymbols) / float totalSymbols * 100.0
        else 0.0
    let totalSavings = opportunities |> List.sumBy (fun o -> o.EstimatedSaving)
    
    {|
        Summary = {|
            TotalSymbols = totalSymbols
            ReachableSymbols = Set.count basic.ReachableSymbols
            UnreachableSymbols = Set.count basic.UnreachableSymbols
            EliminationRate = eliminationRate
            EstimatedSavings = $"{totalSavings / 1024} KB"
        |}
        
        EntryPoints = 
            basic.EntryPoints 
            |> List.map (fun ep -> ep.FullName)
        
        ComponentAnalysis = 
            result.ComponentReachability
            |> Map.toList
            |> List.map (fun (id, cr) ->
                {|
                    ComponentId = id
                    ReachableUnits = cr.ReachableUnits
                    TotalUnits = cr.TotalUnits
                    Status = 
                        if cr.ReachableUnits = 0 then "Unused"
                        elif cr.PartiallyReachable then "Partially Used"
                        else "Fully Used"
                |}
            )
        
        TopOpportunities = 
            opportunities
            |> List.sortByDescending (fun o -> o.EstimatedSaving)
            |> List.truncate 10
            |> List.map (fun o ->
                {|
                    Symbol = o.Symbol.FullName
                    Reason = string o.Reason
                    EstimatedSaving = $"{o.EstimatedSaving} bytes"
                |}
            )
    |}

/// Classify symbol by library boundary
let classifySymbol (symbol: FSharpSymbol) : LibraryCategory =
    let fullName = getSymbolId symbol
    match fullName with
    | name when name.StartsWith("Alloy.") -> AlloyLibrary
    | name when name.StartsWith("FSharp.Core.") || name.StartsWith("Microsoft.FSharp.") -> FSharpCore
    | _ -> UserCode

/// Check if symbol should be included based on library boundaries
let shouldIncludeSymbol (symbol: FSharpSymbol) : bool =
    match classifySymbol symbol with
    | UserCode | AlloyLibrary -> true
    | FSharpCore -> 
        // Only include primitives and SRTP targets
        match symbol with
        | :? FSharpMemberOrFunctionOrValue as mfv ->
            let primNames = [ "op_Addition"; "op_Subtraction"; "op_Multiply"; "op_Division"; "printf"; "printfn" ]
            List.contains mfv.LogicalName primNames || mfv.IsCompilerGenerated
        | _ -> false
    | Other _ -> false

/// Enhanced reachability analysis with library boundary awareness
let analyzeReachabilityWithBoundaries (symbolUses: FSharpSymbolUse[]) : LibraryAwareReachability =
    let startTime = DateTime.UtcNow
    
    // Extract relationships from ALL symbol uses
    let relationships = extractRelationships symbolUses
    
    // Run existing analysis on ALL symbols to get complete reachability
    let basicResult = analyzeReachability symbolUses relationships
    
    // DEBUG: Print some diagnostics to understand what's happening
    printfn "[DEBUG] Total symbol uses: %d" symbolUses.Length
    printfn "[DEBUG] Total relationships: %d" relationships.Length
    printfn "[DEBUG] Call relationships: %d" (relationships |> Array.filter (fun r -> r.RelationType = RelationType.Calls) |> Array.length)
    printfn "[DEBUG] Entry points found: %d (%s)" 
        basicResult.EntryPoints.Length 
        (basicResult.EntryPoints |> List.map (fun ep -> ep.DisplayName) |> String.concat ", ")
    printfn "[DEBUG] Call graph entries: %d" (Map.count basicResult.CallGraph)
    printfn "[DEBUG] Reachable symbols: %d" (Set.count basicResult.ReachableSymbols)
    
    // For now, don't filter at all - just categorize
    let libraryCategories = 
        symbolUses
        |> Array.map (fun symbolUse -> getSymbolId symbolUse.Symbol, classifySymbol symbolUse.Symbol)
        |> Map.ofArray
    
    let endTime = DateTime.UtcNow
    let computationTime = (endTime - startTime).TotalMilliseconds |> int64
    
    let pruningStats = {
        TotalSymbols = symbolUses.Length
        ReachableSymbols = Set.count basicResult.ReachableSymbols
        EliminatedSymbols = symbolUses.Length - Set.count basicResult.ReachableSymbols
        ComputationTimeMs = computationTime
    }
    
    {
        BasicResult = basicResult
        LibraryCategories = libraryCategories
        PruningStatistics = pruningStats
    }

/// Generate pruned symbol list for debug output
let generatePrunedSymbolData (result: LibraryAwareReachability) (symbolUses: FSharpSymbolUse[]) =
    symbolUses
    |> Array.map (fun symbolUse ->
        let symbolId = getSymbolId symbolUse.Symbol
        {|
            SymbolName = symbolUse.Symbol.DisplayName
            SymbolKind = symbolUse.Symbol.GetType().Name
            SymbolHash = symbolUse.Symbol.GetHashCode()
            Range = {|
                File = symbolUse.Range.FileName
                StartLine = symbolUse.Range.Start.Line
                StartColumn = symbolUse.Range.Start.Column
                EndLine = symbolUse.Range.End.Line
                EndColumn = symbolUse.Range.End.Column
            |}
            IsReachable = Set.contains symbolId result.BasicResult.ReachableSymbols
            LibraryCategory = Map.tryFind symbolId result.LibraryCategories |> Option.map (sprintf "%A")
        |})
    |> Array.filter (fun symbolData -> symbolData.IsReachable)

/// Generate reachability comparison report
let generateComparisonData (beforeSymbols: FSharpSymbolUse[]) (result: LibraryAwareReachability) =
    {|
        Summary = {|
            OriginalSymbolCount = beforeSymbols.Length
            ReachableSymbolCount = result.PruningStatistics.ReachableSymbols
            EliminatedCount = result.PruningStatistics.EliminatedSymbols
            EliminationRate = 
                if beforeSymbols.Length > 0 then
                    (float result.PruningStatistics.EliminatedSymbols / float beforeSymbols.Length) * 100.0
                else 0.0
            ComputationTimeMs = result.PruningStatistics.ComputationTimeMs
        |}
        EntryPoints = result.BasicResult.EntryPoints |> List.map (fun ep -> ep.DisplayName)
        CallGraph = result.BasicResult.CallGraph
    |}

/// Generate call graph data structure for visualization
let generateCallGraphData (result: LibraryAwareReachability) =
    // Generate nodes for ALL reachable symbols, not just entry points
    let nodes = 
        result.BasicResult.ReachableSymbols 
        |> Set.toArray 
        |> Array.map (fun symbolId -> {|
            Id = symbolId
            Name = symbolId
            IsReachable = true
        |})
    
    // Generate edges only between reachable symbols
    let edges = 
        result.BasicResult.CallGraph 
        |> Map.toSeq
        |> Seq.collect (fun (from, targets) ->
            targets |> List.map (fun target -> {|
                Source = from
                Target = target
                Kind = "References"
            |}))
        |> Seq.toArray
    
    {|
        Nodes = nodes
        Edges = edges
        NodeCount = nodes.Length
        EdgeCount = edges.Length
    |}

/// Generate library boundary analysis data
let generateLibraryBoundaryData (result: LibraryAwareReachability) =
    result.LibraryCategories
    |> Map.toSeq
    |> Seq.groupBy (fun (_, category) -> category)
    |> Seq.map (fun (category, symbols) -> {|
        LibraryCategory = sprintf "%A" category
        SymbolCount = Seq.length symbols
        IncludedInAnalysis = 
            match category with
            | UserCode | AlloyLibrary -> true
            | FSharpCore -> false
            | Other _ -> false
    |})
    |> Seq.toArray

/// Generate all debug assets for pruned PSG
let generatePrunedPSGAssets (beforeSymbols: FSharpSymbolUse[]) (result: LibraryAwareReachability) =
    {|
        CorrelationData = generatePrunedSymbolData result beforeSymbols
        ComparisonData = generateComparisonData beforeSymbols result
        CallGraphData = generateCallGraphData result
        LibraryBoundaryData = generateLibraryBoundaryData result
    |}