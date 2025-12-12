module Core.PSG.DebugOutput

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization
open FSharp.Compiler.Text
open FSharp.Compiler.Symbols
open Core.PSG.Types
open Core.PSG.Correlation
open Core.PSG.Reachability

/// Configure JSON serialization with F# support
let private jsonOptions = 
    let options = JsonSerializerOptions(WriteIndented = true)
    options.Converters.Add(JsonFSharpConverter())
    options

/// Helper to convert range to simple object for JSON
let private rangeToJson (range: range) = {|
    File = Path.GetFileName(range.FileName)
    StartLine = range.Start.Line
    StartColumn = range.Start.Column
    EndLine = range.End.Line
    EndColumn = range.End.Column
|}

/// Helper to convert NodeId to string for JSON
let private nodeIdToString (nodeId: NodeId) = nodeId.Value

/// Enhanced PSG Node serialization with detailed symbol correlation info
let private preparePSGNodeForJson (node: PSGNode) = {|
    Id = nodeIdToString node.Id
    SyntaxKind = node.SyntaxKind
    SymbolFullName = node.Symbol |> Option.map (fun s -> s.FullName)
    SymbolDisplayName = node.Symbol |> Option.map (fun s -> s.DisplayName)
    SymbolTypeName = node.Symbol |> Option.map (fun s -> s.GetType().Name)
    SymbolHash = node.Symbol |> Option.map (fun s -> s.GetHashCode())
    TypeName = node.Type |> Option.map (fun t -> 
        try t.Format(FSharpDisplayContext.Empty)
        with _ -> "unknown_type")
    Range = rangeToJson node.Range
    SourceFile = Path.GetFileName(node.SourceFile)
    ParentId = node.ParentId |> Option.map nodeIdToString
    Children = 
        match node.Children with
        | NotProcessed -> [||]
        | NoChildren -> [||]
        | Parent children -> children |> List.map nodeIdToString |> List.toArray
    ChildrenState = 
        match node.Children with
        | NotProcessed -> "NotProcessed"
        | NoChildren -> "NoChildren"
        | Parent children -> $"Parent({children.Length})"
    HasSymbol = node.Symbol.IsSome
    IsCorrelated = node.Symbol.IsSome
    // Soft-delete fields
    IsReachable = node.IsReachable
    EliminationReason = node.EliminationReason
    EliminationPass = node.EliminationPass
    ReachabilityDistance = node.ReachabilityDistance
    // Enhanced debugging fields
    SymbolIsFunction = 
        node.Symbol |> Option.map (fun s ->
            match s with
            | :? FSharpMemberOrFunctionOrValue as mfv -> mfv.IsFunction || mfv.IsMember
            | _ -> false) |> Option.defaultValue false
    LibraryCategory = 
        node.Symbol |> Option.map (fun s ->
            let fullName = s.FullName
            match fullName with
            | name when name.StartsWith("Alloy.") -> "AlloyLibrary"
            | name when name.StartsWith("FSharp.Core.") || name.StartsWith("Microsoft.FSharp.") -> "FSharpCore"
            | name when name.StartsWith("Examples.") -> "UserCode"
            | _ -> "UserCode") |> Option.defaultValue "Unknown"
    // Context tracking fields for Phase 1
    ContextRequirement =
        node.ContextRequirement |> Option.map (fun cr ->
            match cr with
            | Pure -> "Pure"
            | AsyncBoundary -> "AsyncBoundary"
            | ResourceAccess -> "ResourceAccess"
            | Parameter idx -> sprintf "Parameter(%d)" idx)
    ComputationPattern = 
        node.ComputationPattern |> Option.map (fun cp ->
            match cp with
            | DataDriven -> "DataDriven"
            | DemandDriven -> "DemandDriven")
|}

/// Enhanced PSGEdge serialization
let private preparePSGEdgeForJson (edge: PSGEdge) = {|
    Source = nodeIdToString edge.Source
    Target = nodeIdToString edge.Target
    Kind = edge.Kind.ToString()
|}

/// Generate symbol correlation analysis for debugging
let generateSymbolCorrelationAnalysis (psg: ProgramSemanticGraph) (reachableSymbols: Set<string>) (outputDir: string) =
    try
        // Extract all symbols from PSG nodes for analysis
        let psgSymbols = 
            psg.Nodes
            |> Map.toSeq
            |> Seq.choose (fun (_, node) -> 
                node.Symbol |> Option.map (fun s -> s, node))
            |> Seq.toArray
        
        // Create correlation analysis
        let correlationAnalysis = {|
            Summary = {|
                TotalPSGNodes = psg.Nodes.Count
                NodesWithSymbols = psgSymbols.Length
                NodesWithoutSymbols = psg.Nodes.Count - psgSymbols.Length
                ReachableSymbolTargets = Set.count reachableSymbols
                ReachableNodesFound = psg.Nodes |> Map.filter (fun _ node -> node.IsReachable) |> Map.count
            |}
            ReachableTargets = reachableSymbols |> Set.toArray |> Array.sort
            PSGSymbolSample = 
                psgSymbols 
                |> Array.take (min 20 psgSymbols.Length)
                |> Array.map (fun (symbol, node) -> {|
                    FullName = symbol.FullName
                    DisplayName = symbol.DisplayName
                    NodeId = nodeIdToString node.Id
                    SyntaxKind = node.SyntaxKind
                    IsReachable = node.IsReachable
                    SymbolType = symbol.GetType().Name
                    MatchesAnyTarget = reachableSymbols |> Set.exists (fun target ->
                        symbol.FullName = target || 
                        symbol.DisplayName = target ||
                        symbol.FullName.EndsWith("." + target) ||
                        target.EndsWith("." + symbol.FullName))
                |})
            SymbolMatching = 
                reachableSymbols
                |> Set.toArray
                |> Array.map (fun target -> {|
                    Target = target
                    ExactMatches = 
                        psgSymbols 
                        |> Array.filter (fun (symbol, _) -> symbol.FullName = target)
                        |> Array.map (fun (symbol, node) -> {|
                            FullName = symbol.FullName
                            NodeId = nodeIdToString node.Id
                            IsReachable = node.IsReachable
                        |})
                    PartialMatches = 
                        psgSymbols 
                        |> Array.filter (fun (symbol, _) -> 
                            symbol.FullName <> target && (
                                symbol.FullName.Contains(target) || 
                                target.Contains(symbol.FullName) ||
                                symbol.DisplayName = target ||
                                symbol.FullName.EndsWith("." + target)))
                        |> Array.take 5  // Limit to prevent huge output
                        |> Array.map (fun (symbol, node) -> {|
                            FullName = symbol.FullName
                            DisplayName = symbol.DisplayName
                            NodeId = nodeIdToString node.Id
                            IsReachable = node.IsReachable
                            MatchReason = 
                                if symbol.DisplayName = target then "DisplayName exact"
                                elif symbol.FullName.EndsWith("." + target) then "FullName suffix"
                                elif symbol.FullName.Contains(target) then "FullName contains"
                                elif target.Contains(symbol.FullName) then "Target contains"
                                else "Unknown"
                        |})
                |})
        |}
        
        let correlationPath = Path.Combine(outputDir, "symbol_correlation_analysis.json")
        File.WriteAllText(correlationPath, JsonSerializer.Serialize(correlationAnalysis, jsonOptions))
    with _ -> ()

/// Generate complete PSG debug output with enhanced symbol tracking
let generatePSGDebugOutput (psg: ProgramSemanticGraph) (outputDir: string) =
    try
        Directory.CreateDirectory(outputDir) |> ignore
        
        // Enhanced node serialization
        let nodes = 
            psg.Nodes
            |> Map.toSeq
            |> Seq.map (snd >> preparePSGNodeForJson)
            |> Seq.toArray
        
        let nodesPath = Path.Combine(outputDir, "psg_nodes.json")
        File.WriteAllText(nodesPath, JsonSerializer.Serialize(nodes, jsonOptions))
        
        // Edge serialization
        let edges = 
            psg.Edges
            |> List.map preparePSGEdgeForJson
            |> List.toArray
        
        let edgesPath = Path.Combine(outputDir, "psg_edges.json")
        File.WriteAllText(edgesPath, JsonSerializer.Serialize(edges, jsonOptions))
        
        // Enhanced reachability analysis
        let reachabilityAnalysis = {|
            TotalNodes = nodes.Length
            ReachableNodes = nodes |> Array.filter (fun n -> n.IsReachable) |> Array.length
            UnreachableNodes = nodes |> Array.filter (fun n -> not n.IsReachable) |> Array.length
            EliminationRate = 
                let unreachable = nodes |> Array.filter (fun n -> not n.IsReachable) |> Array.length
                if nodes.Length > 0 then 
                    Math.Round(float unreachable / float nodes.Length * 100.0, 2)
                else 0.0
            EliminationReasons = 
                nodes 
                |> Array.filter (fun n -> not n.IsReachable && n.EliminationReason.IsSome)
                |> Array.groupBy (fun n -> n.EliminationReason.Value)
                |> Array.map (fun (reason, nodes) -> {| Reason = reason; Count = nodes.Length |})
            ReachabilityDistanceDistribution = 
                nodes
                |> Array.filter (fun n -> n.IsReachable && n.ReachabilityDistance.IsSome)
                |> Array.groupBy (fun n -> n.ReachabilityDistance.Value)
                |> Array.map (fun (distance, nodes) -> {| Distance = distance; Count = nodes.Length |})
                |> Array.sortBy (fun entry -> entry.Distance)
            SymbolCorrelationStats = {|
                NodesWithSymbols = nodes |> Array.filter (fun n -> n.HasSymbol) |> Array.length
                NodesWithoutSymbols = nodes |> Array.filter (fun n -> not n.HasSymbol) |> Array.length
                ReachableWithSymbols = nodes |> Array.filter (fun n -> n.IsReachable && n.HasSymbol) |> Array.length
                ReachableWithoutSymbols = nodes |> Array.filter (fun n -> n.IsReachable && not n.HasSymbol) |> Array.length
                UnreachableWithSymbols = nodes |> Array.filter (fun n -> not n.IsReachable && n.HasSymbol) |> Array.length
            |}
            LibraryCategoryBreakdown = 
                nodes
                |> Array.filter (fun n -> n.HasSymbol)
                |> Array.groupBy (fun n -> n.LibraryCategory)
                |> Array.map (fun (category, categoryNodes) -> {|
                    Category = category
                    Total = categoryNodes.Length
                    Reachable = categoryNodes |> Array.filter (fun n -> n.IsReachable) |> Array.length
                    Unreachable = categoryNodes |> Array.filter (fun n -> not n.IsReachable) |> Array.length
                |})
        |}
        
        let reachabilityPath = Path.Combine(outputDir, "reachability_analysis.json")
        File.WriteAllText(reachabilityPath, JsonSerializer.Serialize(reachabilityAnalysis, jsonOptions))
        
        // Generate PSG summary with entry point analysis
        let psgSummary = {|
            CompilationOrder = psg.CompilationOrder |> List.map Path.GetFileName
            NodeCount = nodes.Length
            EdgeCount = edges.Length
            FileCount = psg.SourceFiles.Count
            SymbolCount = nodes |> Array.filter (fun n -> n.HasSymbol) |> Array.length
            EntryPoints = 
                psg.EntryPoints 
                |> List.choose (fun entryId ->
                    Map.tryFind entryId.Value psg.Nodes
                    |> Option.bind (fun node -> node.Symbol)
                    |> Option.map (fun symbol -> {|
                        NodeId = entryId.Value
                        SymbolFullName = symbol.FullName
                        SymbolDisplayName = symbol.DisplayName
                        IsMarkedReachable = 
                            Map.tryFind entryId.Value psg.Nodes
                            |> Option.map (fun n -> n.IsReachable)
                            |> Option.defaultValue false
                    |}))
        |}
        
        let summaryPath = Path.Combine(outputDir, "psg.summary.json")
        File.WriteAllText(summaryPath, JsonSerializer.Serialize(psgSummary, jsonOptions))
    with _ -> ()

/// Generate enhanced correlation debug output
let generateCorrelationDebugOutput (correlations: (range * FSharpSymbol)[]) (outputDir: string) =
    try
        Directory.CreateDirectory(outputDir) |> ignore
        
        let correlationData = 
            correlations
            |> Array.map (fun (range, symbol) -> {|
                Range = rangeToJson range
                SymbolName = symbol.DisplayName
                SymbolFullName = symbol.FullName
                SymbolKind = symbol.GetType().Name
                SymbolHash = symbol.GetHashCode()
                IsFunction = 
                    match symbol with
                    | :? FSharpMemberOrFunctionOrValue as mfv -> mfv.IsFunction || mfv.IsMember
                    | _ -> false
                LibraryCategory = 
                    let fullName = symbol.FullName
                    match fullName with
                    | name when name.StartsWith("Alloy.") -> "AlloyLibrary"
                    | name when name.StartsWith("FSharp.Core.") || name.StartsWith("Microsoft.FSharp.") -> "FSharpCore"
                    | _ -> "UserCode"
                HasTypeInformation = 
                    match symbol with
                    | :? FSharpMemberOrFunctionOrValue -> true
                    | :? FSharpEntity -> true
                    | :? FSharpField -> true
                    | _ -> false
                TypeName = 
                    match symbol with
                    | :? FSharpMemberOrFunctionOrValue as mfv -> 
                        Some (try mfv.FullType.Format(FSharpDisplayContext.Empty) with _ -> "unknown_type")
                    | :? FSharpEntity as entity -> 
                        if entity.IsFSharpRecord || entity.IsFSharpUnion || entity.IsClass then
                            Some (try entity.AsType().Format(FSharpDisplayContext.Empty) with _ -> "unknown_type")
                        else None
                    | :? FSharpField as field -> 
                        Some (try field.FieldType.Format(FSharpDisplayContext.Empty) with _ -> "unknown_type")
                    | _ -> None
            |})
        
        let correlationsPath = Path.Combine(outputDir, "psg_correlations.json")
        File.WriteAllText(correlationsPath, JsonSerializer.Serialize(correlationData, jsonOptions))
    with _ -> ()

/// Generate reachability analysis debug output with symbol correlation
let generateReachabilityDebugOutput (result: LibraryAwareReachability) (outputDir: string) =
    try
        Directory.CreateDirectory(outputDir) |> ignore
        
        let reachabilityData = {|
            summary = {|
                computationTimeMs = result.PruningStatistics.ComputationTimeMs
                originalSymbolCount = result.PruningStatistics.TotalSymbols
                reachableSymbolCount = result.PruningStatistics.ReachableSymbols
                eliminatedCount = result.PruningStatistics.EliminatedSymbols
                eliminationRate = 
                    if result.PruningStatistics.TotalSymbols > 0 then
                        float result.PruningStatistics.EliminatedSymbols / float result.PruningStatistics.TotalSymbols * 100.0
                    else 0.0
            |}
            entryPoints = result.BasicResult.EntryPoints |> List.map (fun ep -> ep.FullName)
            callGraph = result.BasicResult.CallGraph |> Map.toArray
            psgMetadata = {|
                totalNodes = result.MarkedPSG.Nodes.Count
                totalEdges = result.MarkedPSG.Edges.Length
                sourceFiles = result.MarkedPSG.SourceFiles.Keys |> Seq.map Path.GetFileName |> Seq.sort |> Seq.toArray
                compilationOrder = result.MarkedPSG.CompilationOrder |> List.map Path.GetFileName
            |}
            reachableSymbols = result.BasicResult.ReachableSymbols |> Set.toArray |> Array.sort
            libraryBoundaries = 
                result.LibraryCategories
                |> Map.toSeq
                |> Seq.groupBy snd
                |> Seq.map (fun (category, symbols) -> {|
                    category = category.ToString()
                    symbols = symbols |> Seq.map fst |> Seq.toArray
                    count = symbols |> Seq.length
                |})
                |> Seq.toArray
        |}
        
        let reachabilityPath = Path.Combine(outputDir, "reachability.analysis.json")
        File.WriteAllText(reachabilityPath, JsonSerializer.Serialize(reachabilityData, jsonOptions))
    with _ -> ()

/// Enhanced debug output generation with symbol correlation analysis
let generateDebugOutputs 
    (graph: ProgramSemanticGraph) 
    (correlations: (range * FSharpSymbol)[]) 
    (stats: CorrelationStats) 
    (outputDir: string) =
    
    generatePSGDebugOutput graph outputDir
    generateCorrelationDebugOutput correlations outputDir
    
    // Extract reachable symbols from the graph for correlation analysis
    let reachableSymbols = 
        graph.Nodes
        |> Map.toSeq
        |> Seq.choose (fun (_, node) -> 
            if node.IsReachable then
                node.Symbol |> Option.map (fun s -> s.FullName)
            else None)
        |> Set.ofSeq
    
    generateSymbolCorrelationAnalysis graph reachableSymbols outputDir
    
    // Generate summary
    let summaryStats = {|
        PSGStructure = {|
            TotalNodes = graph.Nodes.Count
            TotalEdges = graph.Edges.Length
            EntryPoints = graph.EntryPoints.Length
            SourceFiles = graph.SourceFiles.Count
        |}
        SoftDeleteAnalysis = {|
            ReachableNodes = graph.Nodes |> Map.filter (fun _ node -> node.IsReachable) |> Map.count
            UnreachableNodes = graph.Nodes |> Map.filter (fun _ node -> not node.IsReachable) |> Map.count
            EliminationRate = 
                let unreachable = graph.Nodes |> Map.filter (fun _ node -> not node.IsReachable) |> Map.count
                if graph.Nodes.Count > 0 then 
                    Math.Round(float unreachable / float graph.Nodes.Count * 100.0, 2)
                else 0.0
        |}
        CorrelationStats = {|
            TotalCorrelations = correlations.Length
            UniqueSymbols = correlations |> Array.map (snd >> fun s -> s.FullName) |> Array.distinct |> Array.length
        |}
        SymbolCorrelationIssues = {|
            ReachableSymbolsExtracted = Set.count reachableSymbols
            PSGNodesMarkedReachable = graph.Nodes |> Map.filter (fun _ node -> node.IsReachable) |> Map.count
            CorrelationGap = Set.count reachableSymbols > 0 && (graph.Nodes |> Map.filter (fun _ node -> node.IsReachable) |> Map.count) = 0
        |}
    |}
    
    let summaryPath = Path.Combine(outputDir, "debug_summary.json")
    File.WriteAllText(summaryPath, JsonSerializer.Serialize(summaryStats, jsonOptions))

/// Generate all debug outputs (alternative API)
let generateCompleteDebugOutput (psg: ProgramSemanticGraph) (correlations: (range * FSharpSymbol)[]) (outputDir: string) =
    generatePSGDebugOutput psg outputDir
    generateCorrelationDebugOutput correlations outputDir

    let reachableSymbols =
        psg.Nodes
        |> Map.toSeq
        |> Seq.choose (fun (_, node) ->
            if node.IsReachable then
                node.Symbol |> Option.map (fun s -> s.FullName)
            else None)
        |> Set.ofSeq

    generateSymbolCorrelationAnalysis psg reachableSymbols outputDir