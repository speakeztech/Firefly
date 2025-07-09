module Core.Analysis.CouplingCohesion

open FSharp.Compiler.Symbols
open Core.FCS.SymbolAnalysis  // This gives us access to SymbolRelation, RelationType, etc.

/// Semantic unit representing a cohesive component
type SemanticUnit = 
    | Module of FSharpEntity
    | Namespace of string
    | FunctionGroup of FSharpMemberOrFunctionOrValue list
    | TypeCluster of FSharpEntity list

/// Coupling measurement between semantic units
type Coupling = {
    From: SemanticUnit
    To: SemanticUnit
    Strength: float  // 0.0 to 1.0
    Dependencies: SymbolRelation list
}

/// Cohesion measurement within a semantic unit
type Cohesion = {
    Unit: SemanticUnit
    Score: float  // 0.0 to 1.0
    InternalRelations: int
    ExternalRelations: int
}

/// Component identified through coupling/cohesion analysis
type CodeComponent = {
    Id: string
    Units: SemanticUnit list
    Cohesion: float
    AverageCoupling: float
    Boundaries: ComponentBoundary list
}

and ComponentBoundary = {
    Interface: FSharpSymbol list
    Direction: BoundaryDirection
}

and BoundaryDirection = Inbound | Outbound | Bidirectional

/// Calculate cohesion for a semantic unit
let calculateCohesion (unit: SemanticUnit) (relationships: SymbolRelation[]) =
    let unitSymbols = 
        match unit with
        | Module entity -> 
            relationships 
            |> Array.filter (fun r -> 
                match r.From.DeclaringEntity with
                | Some e -> e = entity
                | None -> false
            )
            |> Array.map (fun r -> r.From)
            |> Set.ofArray
        
        | FunctionGroup functions ->
            functions |> List.map (fun f -> f :> FSharpSymbol) |> Set.ofList
        
        | TypeCluster types ->
            types |> List.map (fun t -> t :> FSharpSymbol) |> Set.ofList
        
        | Namespace ns ->
            relationships
            |> Array.filter (fun r -> r.From.FullName.StartsWith(ns))
            |> Array.map (fun r -> r.From)
            |> Set.ofArray
    
    let internalRelations = 
        relationships 
        |> Array.filter (fun r -> 
            Set.contains r.From unitSymbols && Set.contains r.To unitSymbols
        )
        |> Array.length
    
    let externalRelations = 
        relationships 
        |> Array.filter (fun r -> 
            Set.contains r.From unitSymbols && not (Set.contains r.To unitSymbols)
        )
        |> Array.length
    
    let total = internalRelations + externalRelations
    let score = if total > 0 then float internalRelations / float total else 0.0
    
    {
        Unit = unit
        Score = score
        InternalRelations = internalRelations
        ExternalRelations = externalRelations
    }

/// Calculate coupling between two semantic units
let calculateCoupling (from: SemanticUnit) (to': SemanticUnit) (relationships: SymbolRelation[]) =
    let fromSymbols = 
        match from with
        | Module e -> Set.singleton (e :> FSharpSymbol)
        | FunctionGroup fs -> fs |> List.map (fun f -> f :> FSharpSymbol) |> Set.ofList
        | TypeCluster ts -> ts |> List.map (fun t -> t :> FSharpSymbol) |> Set.ofList
        | Namespace _ -> Set.empty  // Simplified
    
    let toSymbols = 
        match to' with
        | Module e -> Set.singleton (e :> FSharpSymbol)
        | FunctionGroup fs -> fs |> List.map (fun f -> f :> FSharpSymbol) |> Set.ofList
        | TypeCluster ts -> ts |> List.map (fun t -> t :> FSharpSymbol) |> Set.ofList
        | Namespace _ -> Set.empty  // Simplified
    
    let dependencies = 
        relationships
        |> Array.filter (fun r -> 
            Set.contains r.From fromSymbols && Set.contains r.To toSymbols
        )
        |> Array.toList
    
    let strength = 
        let fromSize = Set.count fromSymbols
        let depCount = dependencies.Length
        if fromSize > 0 then float depCount / float fromSize else 0.0
    
    {
        From = from
        To = to'
        Strength = min 1.0 strength
        Dependencies = dependencies
    }

/// Identify semantic units through clustering
let identifySemanticUnits (symbolUses: FSharpSymbolUse[]) (relationships: SymbolRelation[]) =
    // Group symbols by module
    let moduleGroups = 
        symbolUses
        |> Array.filter (fun useSymbol -> useSymbol.IsFromDefinition)
        |> Array.groupBy (fun useSymbol ->
            match useSymbol.Symbol.DeclaringEntity with
            | Some entity when entity.IsFSharpModule -> Some entity
            | _ -> None
        )
        |> Array.choose (fun (entityOpt, uses) ->
            entityOpt |> Option.map (fun entity -> Module entity)
        )
    
    // Find function clusters (functions that call each other frequently)
    let functionClusters = 
        let functions = 
            symbolUses
            |> Array.choose (fun useSymbol ->
                match useSymbol.Symbol with
                | :? FSharpMemberOrFunctionOrValue as mfv when mfv.IsFunction ->
                    Some mfv
                | _ -> None
            )
            |> Array.distinct
        
        // Simple clustering: functions in same module that call each other
        functions
        |> Array.groupBy (fun f -> f.DeclaringEntity)
        |> Array.map (fun (_, funcs) -> FunctionGroup (List.ofArray funcs))
    
    Array.append moduleGroups functionClusters |> List.ofArray

/// Detect components through cohesion threshold
let detectComponents (units: SemanticUnit list) (relationships: SymbolRelation[]) (threshold: float) =
    // Calculate cohesion for all units
    let cohesions = 
        units 
        |> List.map (fun unit -> unit, calculateCohesion unit relationships)
        |> List.filter (fun (_, cohesion) -> cohesion.Score >= threshold)
    
    // Group highly cohesive units that have coupling
    let codeComponents = ResizeArray<CodeComponent>()
    let processed = System.Collections.Generic.HashSet<SemanticUnit>()
    
    for (unit, cohesion) in cohesions do
        if not (processed.Contains unit) then
            // Find all units coupled to this one
            let cluster = ResizeArray<SemanticUnit>()
            cluster.Add unit
            processed.Add unit |> ignore
            
            // Add coupled units above threshold
            for (otherUnit, otherCohesion) in cohesions do
                if not (processed.Contains otherUnit) then
                    let coupling = calculateCoupling unit otherUnit relationships
                    if coupling.Strength > 0.3 then  // Coupling threshold
                        cluster.Add otherUnit
                        processed.Add otherUnit |> ignore
            
            if cluster.Count > 0 then
                let avgCohesion = 
                    cluster 
                    |> Seq.map (fun u -> 
                        let (_, cohesion) = 
                            cohesions 
                            |> List.find (fun (unit', _) -> unit' = u)
                        cohesion.Score
                    )
                    |> Seq.average
                
                codeComponents.Add {
                    Id = $"Component_{codeComponents.Count + 1}"
                    Units = List.ofSeq cluster
                    Cohesion = avgCohesion
                    AverageCoupling = 0.0  // Calculate if needed
                    Boundaries = []  // Calculate if needed
                }
    
    List.ofSeq codeComponents

/// Analyze coupling patterns for memory layout
type MemoryLayoutHint = 
    | Contiguous of units: SemanticUnit list
    | Isolated of unit: SemanticUnit
    | SharedRegion of units: SemanticUnit list * accessPattern: AccessPattern
    | Tiered of hot: SemanticUnit list * cold: SemanticUnit list

and AccessPattern = 
    | Sequential
    | Random
    | Streaming
    | Concurrent

/// Generate memory layout hints from coupling/cohesion analysis
let generateMemoryLayoutHints (codeComponents: CodeComponent list) (couplings: Coupling list) =
    codeComponents
    |> List.map (fun codeComp ->
        if codeComp.Cohesion > 0.8 then
            // Very high cohesion: keep everything contiguous
            Contiguous codeComp.Units
        elif codeComp.Cohesion < 0.3 then
            // Low cohesion: isolate units
            codeComp.Units |> List.map Isolated
        else
            // Medium cohesion: analyze access patterns
            let relevantCouplings = 
                couplings 
                |> List.filter (fun c -> 
                    List.exists (fun u -> u = c.From) codeComp.Units
                )
            
            if relevantCouplings.Length = 0 then
                Contiguous codeComp.Units
            else
                // Determine access pattern from coupling types
                let pattern = 
                    if relevantCouplings |> List.forall (fun c -> c.Strength < 0.5) then
                        Random
                    else
                        Sequential
                
                SharedRegion(codeComp.Units, pattern)
    )
    |> List.concat

/// Generate report of coupling/cohesion analysis
let generateAnalysisReport (codeComponents: CodeComponent list) (allUnits: SemanticUnit list) =
    {|
        TotalUnits = allUnits.Length
        ComponentCount = codeComponents.Length
        AverageCohesion = 
            if codeComponents.IsEmpty then 0.0 
            else codeComponents |> List.averageBy (fun c -> c.Cohesion)
        HighCohesionComponents = 
            codeComponents |> List.filter (fun c -> c.Cohesion > 0.7) |> List.length
        IsolatedUnits = 
            allUnits.Length - (codeComponents |> List.sumBy (fun c -> c.Units.Length))
        LargestComponent = 
            if codeComponents.IsEmpty then 0
            else codeComponents |> List.map (fun c -> c.Units.Length) |> List.max
    |}