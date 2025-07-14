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

/// Enhanced node representation with complete metadata including type information
type private EnhancedNodeJson = {
    Id: string
    Kind: string
    Symbol: string option
    SymbolFullName: string option
    Type: string option
    TypeFullName: string option
    HasTypeInformation: bool
    Range: {| File: string; StartLine: int; StartColumn: int; EndLine: int; EndColumn: int |}
    ParentId: string option
    Children: string array
    ChildrenState: string
    ChildrenCount: int
    HasSymbol: bool
    IsCorrelated: bool
}

/// Enhanced edge representation with comprehensive metadata
type private EnhancedEdgeJson = {
    Id: string
    Source: string
    Target: string
    Kind: string
    SourceNodeKind: string option
    TargetNodeKind: string option
    SourceHasSymbol: bool
    TargetHasSymbol: bool
    SourceHasType: bool
    TargetHasType: bool
}

/// Consolidated correlation analysis with source range information
type private EnhancedCorrelationJson = {
    Range: {| File: string; StartLine: int; StartColumn: int; EndLine: int; EndColumn: int |}
    SymbolName: string
    SymbolFullName: string
    SymbolKind: string
    SymbolHash: int
    IsFunction: bool
    LibraryCategory: string
    HasTypeInformation: bool
    TypeName: string option
}

/// Convert PSG node to enhanced JSON representation with complete type analysis
let private nodeToEnhancedJson (node: PSGNode) : EnhancedNodeJson =
    let childrenList = ChildrenStateHelpers.getChildrenList node
    let (childrenStateDescription, childrenCount) = 
        match node.Children with
        | NotProcessed -> ("NotProcessed", 0)
        | Leaf -> ("Leaf", 0)
        | Parent children -> ("Parent", children.Length)
    
    let (typeDisplayName, typeFullName, hasTypeInfo) = 
        match node.Type with
        | Some fsharpType -> 
            let displayName = fsharpType.Format(FSharpDisplayContext.Empty)
            let fullName = fsharpType.Format(FSharpDisplayContext.Empty)
            (Some displayName, Some fullName, true)
        | None -> (None, None, false)
    
    {
        Id = node.Id.Value
        Kind = node.SyntaxKind
        Symbol = node.Symbol |> Option.map (fun s -> s.DisplayName)
        SymbolFullName = node.Symbol |> Option.map (fun s -> s.FullName)
        Type = typeDisplayName
        TypeFullName = typeFullName
        HasTypeInformation = hasTypeInfo
        Range = {| 
            File = Path.GetFileName(node.SourceFile)
            StartLine = node.Range.Start.Line
            StartColumn = node.Range.Start.Column
            EndLine = node.Range.End.Line
            EndColumn = node.Range.End.Column 
        |}
        ParentId = node.ParentId |> Option.map (fun id -> id.Value)
        Children = childrenList |> List.map (fun id -> id.Value) |> List.toArray
        ChildrenState = childrenStateDescription
        ChildrenCount = childrenCount
        HasSymbol = node.Symbol.IsSome
        IsCorrelated = node.Symbol.IsSome
    }

/// Convert PSG edge to enhanced JSON representation with type metadata
let private edgeToEnhancedJson (edge: PSGEdge) (psg: ProgramSemanticGraph) : EnhancedEdgeJson =
    let sourceNode = Map.tryFind edge.Source.Value psg.Nodes
    let targetNode = Map.tryFind edge.Target.Value psg.Nodes
    
    {
        Id = sprintf "%s_to_%s_%s" edge.Source.Value edge.Target.Value (edge.Kind.ToString())
        Source = edge.Source.Value
        Target = edge.Target.Value
        Kind = edge.Kind.ToString()
        SourceNodeKind = sourceNode |> Option.map (fun n -> n.SyntaxKind)
        TargetNodeKind = targetNode |> Option.map (fun n -> n.SyntaxKind)
        SourceHasSymbol = sourceNode |> Option.map (fun n -> n.Symbol.IsSome) |> Option.defaultValue false
        TargetHasSymbol = targetNode |> Option.map (fun n -> n.Symbol.IsSome) |> Option.defaultValue false
        SourceHasType = sourceNode |> Option.map (fun n -> n.Type.IsSome) |> Option.defaultValue false
        TargetHasType = targetNode |> Option.map (fun n -> n.Type.IsSome) |> Option.defaultValue false
    }

/// Convert correlation data to enhanced JSON with comprehensive symbol and type analysis
let private correlationToEnhancedJson (range: range) (symbol: FSharpSymbol) : EnhancedCorrelationJson =
    let isFunction = 
        match symbol with
        | :? FSharpMemberOrFunctionOrValue as mfv -> mfv.IsFunction || mfv.IsMember
        | _ -> false
    
    let libraryCategory = 
        let fullName = symbol.FullName
        match fullName with
        | name when name.StartsWith("Alloy.") -> "AlloyLibrary"
        | name when name.StartsWith("FSharp.Core.") || name.StartsWith("Microsoft.FSharp.") -> "FSharpCore"
        | _ -> "UserCode"
    
    let (hasTypeInfo, typeName) = 
        match symbol with
        | :? FSharpMemberOrFunctionOrValue as mfv -> 
            (true, Some (mfv.FullType.Format(FSharpDisplayContext.Empty)))
        | :? FSharpEntity as entity -> 
            if entity.IsFSharpRecord || entity.IsFSharpUnion || entity.IsClass then
                (true, Some (entity.AsType().Format(FSharpDisplayContext.Empty)))
            else (false, None)
        | :? FSharpField as field -> 
            (true, Some (field.FieldType.Format(FSharpDisplayContext.Empty)))
        | _ -> (false, None)
    
    {
        Range = {|
            File = Path.GetFileName(range.FileName)
            StartLine = range.Start.Line
            StartColumn = range.Start.Column
            EndLine = range.End.Line
            EndColumn = range.End.Column
        |}
        SymbolName = symbol.DisplayName
        SymbolFullName = symbol.FullName
        SymbolKind = symbol.GetType().Name
        SymbolHash = symbol.GetHashCode()
        IsFunction = isFunction
        LibraryCategory = libraryCategory
        HasTypeInformation = hasTypeInfo
        TypeName = typeName
    }

/// Generate comprehensive PSG debug output with unified symbol and type analysis
let generatePSGDebugOutput (psg: ProgramSemanticGraph) (outputDir: string) =
    try
        Directory.CreateDirectory(outputDir) |> ignore
        
        let enhancedNodes = 
            psg.Nodes
            |> Map.toSeq
            |> Seq.map (snd >> nodeToEnhancedJson)
            |> Seq.toArray
        
        let nodesPath = Path.Combine(outputDir, "psg_nodes_enhanced.json")
        File.WriteAllText(nodesPath, JsonSerializer.Serialize(enhancedNodes, jsonOptions))
        
        let enhancedEdges = 
            psg.Edges
            |> List.map (fun edge -> edgeToEnhancedJson edge psg)
            |> List.toArray
        
        let edgesPath = Path.Combine(outputDir, "psg_edges_enhanced.json")
        File.WriteAllText(edgesPath, JsonSerializer.Serialize(enhancedEdges, jsonOptions))
        
        let typeAnalysisSummary = 
            let totalNodes = enhancedNodes.Length
            let nodesWithTypes = enhancedNodes |> Array.filter (fun n -> n.HasTypeInformation) |> Array.length
            let nodesWithSymbols = enhancedNodes |> Array.filter (fun n -> n.HasSymbol) |> Array.length
            let edgesWithSourceTypes = enhancedEdges |> Array.filter (fun e -> e.SourceHasType) |> Array.length
            let edgesWithTargetTypes = enhancedEdges |> Array.filter (fun e -> e.TargetHasType) |> Array.length
            
            {|
                Summary = {|
                    TotalNodes = totalNodes
                    NodesWithTypeInformation = nodesWithTypes
                    NodesWithSymbols = nodesWithSymbols
                    TypeCorrelationRate = if totalNodes > 0 then Math.Round(float nodesWithTypes / float totalNodes * 100.0, 2) else 0.0
                    SymbolCorrelationRate = if totalNodes > 0 then Math.Round(float nodesWithSymbols / float totalNodes * 100.0, 2) else 0.0
                |}
                EdgeTypeAnalysis = {|
                    TotalEdges = enhancedEdges.Length
                    EdgesWithSourceTypes = edgesWithSourceTypes
                    EdgesWithTargetTypes = edgesWithTargetTypes
                    SourceTypeRate = if enhancedEdges.Length > 0 then Math.Round(float edgesWithSourceTypes / float enhancedEdges.Length * 100.0, 2) else 0.0
                    TargetTypeRate = if enhancedEdges.Length > 0 then Math.Round(float edgesWithTargetTypes / float enhancedEdges.Length * 100.0, 2) else 0.0
                |}
                TypeDistribution = 
                    enhancedNodes 
                    |> Array.filter (fun n -> n.HasTypeInformation)
                    |> Array.groupBy (fun n -> n.Type |> Option.defaultValue "Unknown")
                    |> Array.map (fun (typeName, nodes) -> {| TypeName = typeName; Count = nodes.Length |})
                    |> Array.sortByDescending (fun entry -> entry.Count)
            |}
        
        let typeAnalysisPath = Path.Combine(outputDir, "psg_type_analysis.json")
        File.WriteAllText(typeAnalysisPath, JsonSerializer.Serialize(typeAnalysisSummary, jsonOptions))
        
        let childrenStateAnalysis = 
            psg.Nodes
            |> Map.toSeq
            |> Seq.map (fun (_, node) ->
                match node.Children with
                | NotProcessed -> "NotProcessed"
                | Leaf -> "Leaf"
                | Parent children -> sprintf "Parent_%d" children.Length)
            |> Seq.countBy id
            |> Seq.map (fun (state, count) -> {| State = state; Count = count |})
            |> Seq.toArray
        
        let stateAnalysisPath = Path.Combine(outputDir, "psg_children_state_analysis.json")
        File.WriteAllText(stateAnalysisPath, JsonSerializer.Serialize(childrenStateAnalysis, jsonOptions))
        
        printfn "Enhanced PSG debug output generated successfully in: %s" outputDir
        printfn "Files created:"
        printfn "  - psg_nodes_enhanced.json (%d nodes)" enhancedNodes.Length
        printfn "  - psg_edges_enhanced.json (%d edges)" enhancedEdges.Length
        printfn "  - psg_type_analysis.json (comprehensive type distribution)"
        printfn "  - psg_children_state_analysis.json (comprehensive state distribution)"
        
    with
    | ex -> 
        printfn "Error generating enhanced PSG debug output: %s" ex.Message

/// Generate comprehensive correlation debug output with enhanced symbol and type analysis
let generateCorrelationDebugOutput (context: CorrelationContext) (outputDir: string) =
    try
        Directory.CreateDirectory(outputDir) |> ignore
        
        let enhancedCorrelations = 
            context.SymbolUses
            |> Array.map (fun symbolUse -> correlationToEnhancedJson symbolUse.Range symbolUse.Symbol)
        
        let correlationsPath = Path.Combine(outputDir, "correlations_enhanced.json")
        File.WriteAllText(correlationsPath, JsonSerializer.Serialize(enhancedCorrelations, jsonOptions))
        
        let correlationStats = {|
            TotalSymbolUses = context.SymbolUses.Length
            UniqueSymbols = context.SymbolUses |> Array.map (fun su -> su.Symbol.FullName) |> Array.distinct |> Array.length
            FunctionCount = enhancedCorrelations |> Array.filter (fun c -> c.IsFunction) |> Array.length
            SymbolsWithTypeInfo = enhancedCorrelations |> Array.filter (fun c -> c.HasTypeInformation) |> Array.length
            TypeCorrelationRate = 
                let totalSymbols = enhancedCorrelations.Length
                if totalSymbols > 0 then
                    let withTypes = enhancedCorrelations |> Array.filter (fun c -> c.HasTypeInformation) |> Array.length
                    Math.Round(float withTypes / float totalSymbols * 100.0, 2)
                else 0.0
            LibraryDistribution = 
                enhancedCorrelations 
                |> Array.groupBy (fun c -> c.LibraryCategory)
                |> Array.map (fun (category, items) -> {| Category = category; Count = items.Length |})
            TypeDistribution = 
                enhancedCorrelations 
                |> Array.filter (fun c -> c.HasTypeInformation)
                |> Array.groupBy (fun c -> c.TypeName |> Option.defaultValue "Unknown")
                |> Array.map (fun (typeName, items) -> {| TypeName = typeName; Count = items.Length |})
                |> Array.sortByDescending (fun entry -> entry.Count)
                |> Array.take (Math.Min(20, enhancedCorrelations.Length))
        |}
        
        let statsPath = Path.Combine(outputDir, "correlation_statistics.json")
        File.WriteAllText(statsPath, JsonSerializer.Serialize(correlationStats, jsonOptions))
        
        printfn "Enhanced correlation debug output generated: %s" correlationsPath
        printfn "Type correlation rate: %.1f%% (%d/%d symbols)" 
            correlationStats.TypeCorrelationRate correlationStats.SymbolsWithTypeInfo context.SymbolUses.Length
        
    with
    | ex -> 
        printfn "Error generating enhanced correlation debug output: %s" ex.Message

/// Generate unified debug outputs with consistent symbol measurement methodology
let generateDebugOutputs 
    (graph: ProgramSemanticGraph) 
    (correlations: (range * FSharpSymbol)[]) 
    (stats: CorrelationStats) 
    (outputDir: string) =
    
    Directory.CreateDirectory(outputDir) |> ignore
    
    let meaningfulSymbols = UnifiedSymbolExtraction.extractMeaningfulSymbols graph
    let correlatedNodes = UnifiedSymbolExtraction.extractCorrelatedNodes graph
    let unifiedStats = {|
        TotalMeaningfulSymbols = meaningfulSymbols.Length
        CorrelatedNodeCount = correlatedNodes.Length
    |}
    
    let enhancedNodes = 
        graph.Nodes
        |> Map.toSeq
        |> Seq.map (fun (_, node) -> 
            let childrenList = ChildrenStateHelpers.getChildrenList node
            let childrenStateDescription = 
                match node.Children with
                | NotProcessed -> "NotProcessed"
                | Leaf -> "Leaf"
                | Parent children -> sprintf "Parent(%d)" children.Length
            
            {|
                Id = node.Id.Value
                Kind = node.SyntaxKind
                Symbol = node.Symbol |> Option.map (fun s -> s.DisplayName)
                SymbolFullName = node.Symbol |> Option.map (fun s -> s.FullName)
                Range = {| 
                    StartLine = node.Range.Start.Line
                    StartColumn = node.Range.Start.Column
                    EndLine = node.Range.End.Line
                    EndColumn = node.Range.End.Column 
                |}
                SourceFile = Path.GetFileName(node.SourceFile)
                ParentId = node.ParentId |> Option.map (fun id -> id.Value)
                Children = childrenList |> List.map (fun id -> id.Value) |> List.toArray
                ChildrenState = childrenStateDescription
                HasSymbol = node.Symbol.IsSome
                IsCorrelated = node.Symbol.IsSome
            |})
        |> Seq.toArray
    
    let nodesPath = Path.Combine(outputDir, "psg_nodes.json")
    File.WriteAllText(nodesPath, JsonSerializer.Serialize(enhancedNodes, jsonOptions))
    
    let enhancedEdges = 
        graph.Edges 
        |> List.map (fun edge -> {|
            Source = edge.Source.Value
            Target = edge.Target.Value
            Kind = edge.Kind.ToString()
            SourceNodeExists = Map.containsKey edge.Source.Value graph.Nodes
            TargetNodeExists = Map.containsKey edge.Target.Value graph.Nodes
        |})
        |> List.toArray
    
    let edgesPath = Path.Combine(outputDir, "psg_edges.json")
    File.WriteAllText(edgesPath, JsonSerializer.Serialize(enhancedEdges, jsonOptions))
    
    let childrenStateSummary = 
        graph.Nodes
        |> Map.toSeq
        |> Seq.map (fun (_, node) ->
            match node.Children with
            | NotProcessed -> "NotProcessed"
            | Leaf -> "Leaf"
            | Parent children -> "Parent")
        |> Seq.countBy id
        |> Seq.map (fun (state, count) -> {| State = state; Count = count |})
        |> Seq.toArray
    
    let statePath = Path.Combine(outputDir, "psg_children_state_summary.json")
    File.WriteAllText(statePath, JsonSerializer.Serialize(childrenStateSummary, jsonOptions))
    
    let enhancedCorrelations = 
        correlations
        |> Array.map (fun (range, symbol) -> correlationToEnhancedJson range symbol)
    
    let correlationPath = Path.Combine(outputDir, "psg_correlations.json")
    File.WriteAllText(correlationPath, JsonSerializer.Serialize(enhancedCorrelations, jsonOptions))
    
    let unifiedStatsReport = {|
        UnifiedSymbolAnalysis = {|
            TotalMeaningfulSymbols = unifiedStats.TotalMeaningfulSymbols
            CorrelatedNodeCount = unifiedStats.CorrelatedNodeCount
            SymbolCorrelationRate = 
                if unifiedStats.TotalMeaningfulSymbols > 0 then
                    (float unifiedStats.CorrelatedNodeCount / float unifiedStats.TotalMeaningfulSymbols) * 100.0
                else 0.0
        |}
        PSGStructureAnalysis = {|
            TotalNodes = graph.Nodes.Count
            TotalEdges = graph.Edges.Length
            NodesWithSymbols = enhancedNodes |> Array.filter (fun n -> n.HasSymbol) |> Array.length
            EntryPointsDeclared = graph.EntryPoints.Length
        |}
        ChildrenStateDistribution = childrenStateSummary
        CorrelationAnalysis = {|
            TotalCorrelations = enhancedCorrelations.Length
            FunctionCorrelations = enhancedCorrelations |> Array.filter (fun c -> c.IsFunction) |> Array.length
            LibraryDistribution = 
                enhancedCorrelations 
                |> Array.groupBy (fun c -> c.LibraryCategory)
                |> Array.map (fun (category, items) -> {| Category = category; Count = items.Length |})
        |}
    |}
    
    let unifiedStatsPath = Path.Combine(outputDir, "unified_statistics_report.json")
    File.WriteAllText(unifiedStatsPath, JsonSerializer.Serialize(unifiedStatsReport, jsonOptions))
    
    printfn "PSG debug outputs generated successfully in: %s" outputDir
    printfn "Files created:"
    printfn "  - psg_nodes.json (%d nodes)" enhancedNodes.Length
    printfn "  - psg_edges.json (%d edges)" enhancedEdges.Length
    printfn "  - psg_children_state_summary.json (state distribution)"
    printfn "  - psg_correlations.json (%d correlations)" enhancedCorrelations.Length
    printfn "  - unified_statistics_report.json (comprehensive analysis)"
    printfn "Unified Symbol Analysis: %d meaningful symbols, %d correlated nodes" 
        unifiedStats.TotalMeaningfulSymbols unifiedStats.CorrelatedNodeCount

/// Generate comprehensive debugging output coordinated with reachability analysis results
let generateComprehensiveDebugOutput (psg: ProgramSemanticGraph) (context: CorrelationContext) (outputDir: string) =
    let timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss")
    let debugDir = Path.Combine(outputDir, sprintf "psg_debug_%s" timestamp)
    
    generatePSGDebugOutput psg debugDir
    generateCorrelationDebugOutput context debugDir
    
    let summaryPath = Path.Combine(debugDir, "debug_summary.txt")
    let meaningfulSymbols = UnifiedSymbolExtraction.extractMeaningfulSymbols psg
    let correlatedNodes = UnifiedSymbolExtraction.extractCorrelatedNodes psg
    let unifiedStats = {|
        TotalMeaningfulSymbols = meaningfulSymbols.Length
        CorrelatedNodeCount = correlatedNodes.Length
    |}
    
    let lines = [
        "PSG Debug Summary - Generated: " + System.DateTime.Now.ToString()
        ""
        "=== Unified Symbol Analysis ==="
        "Meaningful Symbols: " + unifiedStats.TotalMeaningfulSymbols.ToString()
        "Correlated Nodes: " + unifiedStats.CorrelatedNodeCount.ToString()
        "Symbol Correlation Rate: " + sprintf "%.1f%%" (if unifiedStats.TotalMeaningfulSymbols > 0 then (float unifiedStats.CorrelatedNodeCount / float unifiedStats.TotalMeaningfulSymbols) * 100.0 else 0.0)
        ""
        "=== PSG Structure Analysis ==="
        "Total Nodes: " + psg.Nodes.Count.ToString()
        "Total Edges: " + psg.Edges.Length.ToString()
        "Entry Points: " + psg.EntryPoints.Length.ToString()
        "Source Files: " + psg.SourceFiles.Count.ToString()
        "Symbol Uses (Context): " + context.SymbolUses.Length.ToString()
        ""
        "=== ChildrenState Distribution ==="
    ]
    
    let childrenStateLines = 
        psg.Nodes 
        |> Map.toSeq 
        |> Seq.map (fun (_, node) -> 
            match node.Children with
            | NotProcessed -> "NotProcessed"
            | Leaf -> "Leaf" 
            | Parent children -> "Parent")
        |> Seq.countBy id
        |> Seq.map (fun (state, count) -> "  " + state + ": " + count.ToString())
        |> Seq.toList
    
    let edgeTypeLines = 
        [""; "=== Edge Type Distribution ==="] @
        (psg.Edges 
         |> List.map (fun edge -> edge.Kind.ToString())
         |> List.countBy id
         |> List.map (fun (kind, count) -> "  " + kind + ": " + count.ToString()))
    
    let allLines = lines @ childrenStateLines @ edgeTypeLines
    let summaryContent = String.concat "\n" allLines
    
    File.WriteAllText(summaryPath, summaryContent)
    printfn "Consolidated debug summary generated: %s" summaryPath