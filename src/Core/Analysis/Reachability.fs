module Core.Analysis.Reachability

open FSharp.Compiler.Symbols
open Core.FCS.SymbolAnalysis
open Core.Analysis.CouplingCohesion  // This imports CodeComponent type

/// Reachability analysis result
type ReachabilityResult = {
    EntryPoints: FSharpSymbol list
    ReachableSymbols: Set<FSharpSymbol>
    UnreachableSymbols: Set<FSharpSymbol>
    CallGraph: Map<FSharpSymbol, FSharpSymbol list>
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


/// Find entry points in the project
let findEntryPoints (symbolUses: FSharpSymbolUse[]) =
    symbolUses
    |> Array.filter (fun symbolUse -> symbolUse.IsFromDefinition)
    |> Array.choose (fun symbolUse ->
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
    |> Array.groupBy (fun r -> r.From)
    |> Array.map (fun (caller, calls) ->
        caller, calls |> Array.map (fun r -> r.To) |> Array.distinct |> List.ofArray
    )
    |> Map.ofArray

/// Compute reachable symbols from entry points
let computeReachable (entryPoints: FSharpSymbol list) (callGraph: Map<FSharpSymbol, FSharpSymbol list>) =
    let rec traverse (visited: Set<FSharpSymbol>) (current: FSharpSymbol) =
        if Set.contains current visited then visited
        else
            let newVisited = Set.add current visited
            match Map.tryFind current callGraph with
            | Some callees ->
                callees |> List.fold traverse newVisited
            | None -> newVisited
    
    entryPoints |> List.fold traverse Set.empty

/// Perform complete reachability analysis
let analyzeReachability (symbolUses: FSharpSymbolUse[]) (relationships: SymbolRelation[]) =
    // Find all defined symbols
    let allSymbols = 
        symbolUses
        |> Array.filter (fun useSymbol -> useSymbol.IsFromDefinition)
        |> Array.map (fun useSymbol -> useSymbol.Symbol)
        |> Set.ofArray
    
    // Find entry points
    let entryPoints = findEntryPoints symbolUses
    
    // Build call graph
    let callGraph = buildCallGraph relationships
    
    // Compute reachable symbols
    let reachable = computeReachable entryPoints callGraph
    
    // Find unreachable symbols
    let unreachable = Set.difference allSymbols reachable
    
    {
        EntryPoints = entryPoints
        ReachableSymbols = reachable
        UnreachableSymbols = unreachable
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
let analyzeComponentReachability (simpleRResult: ReachabilityResult) (codeComponents: CodeComponent list) =
    let componentReachability = 
        codeComponents
        |> List.map (fun codeComp ->
            let reachableInComponent = 
                codeComp.Units
                |> List.filter (fun compUnit ->
                    match compUnit with
                    | Module compEntity ->
                        Set.contains (compEntity :> FSharpSymbol) simpleRResult.ReachableSymbols
                    | FunctionGroup functions ->
                        functions |> List.exists (fun f -> 
                            Set.contains (f :> FSharpSymbol) simpleRResult.ReachableSymbols
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
        BasicResult = simpleRResult
        ComponentReachability = componentReachability
    }



let identifyEliminationOpportunities (result: ReachabilityResult) (codeComponents: CodeComponent list) =
    // Direct unreachable symbols
    let unreachableOpportunities = 
        result.UnreachableSymbols
        |> Set.toList
        |> List.map (fun symbol ->
            {
                Symbol = symbol
                Reason = Unreachable
                EstimatedSaving = estimateSymbolSize symbol
            }
        )
    
    // Unused components
    let unusedComponentOpportunities =
        codeComponents
        |> List.filter (fun codeComp ->
            codeComp.Units |> List.forall (fun unit ->
                match unit with
                | Module entity -> 
                    Set.contains (entity :> FSharpSymbol) result.UnreachableSymbols
                | FunctionGroup functions ->
                    functions |> List.forall (fun f ->
                        Set.contains (f :> FSharpSymbol) result.UnreachableSymbols
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

// Helper function to estimate symbol size (simplified)
and private estimateSymbolSize (symbol: FSharpSymbol) =
    match symbol with
    | :? FSharpMemberOrFunctionOrValue as mfv ->
        if mfv.IsFunction then 100 else 50  // Rough estimate
    | :? FSharpEntity as entity ->
        if entity.IsFSharpModule then 200
        elif entity.IsFSharpRecord then 150
        else 100
    | _ -> 50

// Helper function to estimate symbol size (simplified)
and estimateSymbolSize (symbol: FSharpSymbol) =
    match symbol with
    | :? FSharpMemberOrFunctionOrValue as mfv ->
        if mfv.IsFunction then 100 else 50  // Rough estimate
    | :? FSharpEntity as entity ->
        if entity.IsFSharpModule then 200
        elif entity.IsFSharpRecord then 150
        else 100
    | _ -> 50

/// Generate reachability report
let generateReport (result: EnhancedReachability) (opportunities: EliminationOpportunity list) =
    let basic = result.BasicResult
    let totalSymbols = Set.count basic.ReachableSymbols + Set.count basic.UnreachableSymbols
    let eliminationRate = float (Set.count basic.UnreachableSymbols) / float totalSymbols * 100.0
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