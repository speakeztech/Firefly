module Core.PSG.DebugOutput

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization
open FSharp.Compiler.Symbols
open FSharp.Compiler.Text
open Core.PSG.Types
open Core.PSG.Correlation
open Core.FCS.Helpers
open Core.Utilities.IntermediateWriter

// Define JSON types locally since they're only needed for serialization
type private NodeJson = {
    Id: string
    Kind: string
    Symbol: string option
    Range: {| StartLine: int; StartColumn: int; EndLine: int; EndColumn: int |}
    SourceFile: string
    ParentId: string option
    Children: string[]
}

type private EdgeJson = {
    Source: string
    Target: string
    Kind: string
}

type private CorrelationJson = {
    Range: {| File: string; StartLine: int; StartColumn: int; EndLine: int; EndColumn: int |}
    SymbolName: string
    SymbolKind: string
    SymbolHash: int
}

// Configure JSON options for consistent serialization
let private jsonOptions = 
    let opts = JsonSerializerOptions(WriteIndented = true)
    opts.Converters.Add(JsonFSharpConverter())
    opts

/// Convert PSG node to JSON representation
let private nodeToJson (node: PSGNode) : NodeJson =
    {
        Id = node.Id.Value
        Kind = node.SyntaxKind
        Symbol = node.Symbol |> Option.map (fun s -> s.DisplayName)
        Range = {| 
            StartLine = node.Range.StartLine
            StartColumn = node.Range.StartColumn
            EndLine = node.Range.EndLine
            EndColumn = node.Range.EndColumn 
        |}
        SourceFile = Path.GetFileName(node.SourceFile)
        ParentId = node.ParentId |> Option.map (fun id -> id.Value)
        Children = node.Children |> List.map (fun id -> id.Value) |> List.toArray
    }

/// Convert edge to JSON representation
let private edgeToJson (edge: PSGEdge) : EdgeJson =
    {
        Source = edge.Source.Value
        Target = edge.Target.Value
        Kind = 
            match edge.Kind with
            | ChildOf -> "ChildOf"
            | SymRef -> "SymRef"
            | TypeOf -> "TypeOf"
            | FunctionCall -> "FunctionCall"
            | Instantiates -> "Instantiates"
            | SymbolDef -> "SymbolDef"
            | SymbolUse -> "SymbolUse"
            | TypeInstantiation _ -> "TypeInstantiation"
            | ControlFlow kind -> sprintf "ControlFlow:%A" kind
            | DataDependency -> "DataDependency"
            | ModuleContainment -> "ModuleContainment"
            | TypeMembership -> "TypeMembership"
    }

/// Generate correlation map for debugging
let generateCorrelationMap 
    (correlations: (range * FSharpSymbol)[]) 
    (outputPath: string) =
    
    let correlationData = 
        correlations
        |> Array.map (fun (range, symbol) ->
            {
                Range = {| 
                    File = Path.GetFileName(range.FileName)
                    StartLine = range.StartLine
                    StartColumn = range.StartColumn
                    EndLine = range.EndLine
                    EndColumn = range.EndColumn 
                |}
                SymbolName = symbol.DisplayName
                SymbolKind = 
                    match symbol with
                    | :? FSharpMemberOrFunctionOrValue -> "MemberOrFunctionOrValue"
                    | :? FSharpEntity -> "Entity"
                    | :? FSharpGenericParameter -> "GenericParameter"
                    | :? FSharpParameter -> "Parameter"
                    | :? FSharpUnionCase -> "UnionCase"
                    | :? FSharpField -> "Field"
                    | :? FSharpActivePatternCase -> "ActivePatternCase"
                    | _ -> "Other"
                SymbolHash = symbol.GetHashCode()
            } : CorrelationJson
        )
    
    let json = JsonSerializer.Serialize(correlationData, jsonOptions)
    writeFileToPath outputPath json
    printfn "  Wrote correlation map (%d entries, %d bytes)" correlations.Length json.Length

/// Generate PSG node graph for visualization
let generateNodeGraph 
    (graph: ProgramSemanticGraph) 
    (outputPath: string) =
    
    let nodesData = 
        graph.Nodes
        |> Map.toArray
        |> Array.map (fun (_, node) -> nodeToJson node)
    
    let edgesData = 
        graph.Edges
        |> List.map edgeToJson
        |> List.toArray
    
    let graphData = {| 
        Nodes = nodesData
        Edges = edgesData
        NodeCount = nodesData.Length
        EdgeCount = edgesData.Length
        EntryPoints = graph.EntryPoints |> List.map (fun id -> id.Value) |> List.toArray
    |}
    
    let json = JsonSerializer.Serialize(graphData, jsonOptions)
    writeFileToPath outputPath json
    printfn "  Wrote node graph (%d nodes, %d edges, %d bytes)" 
        nodesData.Length edgesData.Length json.Length

/// Generate symbol table dump
let generateSymbolTable 
    (graph: ProgramSemanticGraph) 
    (outputPath: string) =
    
    let symbolData = 
        graph.SymbolTable
        |> Map.toArray
        |> Array.map (fun (name, symbol) ->
            {| 
                Name = name
                FullName = symbol.FullName
                Hash = symbol.GetHashCode()
                Kind = 
                    match symbol with
                    | :? FSharpMemberOrFunctionOrValue as mfv ->
                        if mfv.IsProperty then "Property"
                        elif mfv.IsEvent then "Event"
                        elif mfv.IsMember then "Member"
                        elif mfv.IsFunction then "Function"
                        elif mfv.IsModuleValueOrMember then "ModuleValue"
                        else "Value"
                    | :? FSharpEntity as entity ->
                        if entity.IsNamespace then "Namespace"
                        elif entity.IsFSharpModule then "Module"
                        elif entity.IsClass then "Class"
                        elif entity.IsInterface then "Interface"
                        elif entity.IsFSharpRecord then "Record"
                        elif entity.IsFSharpUnion then "Union"
                        elif entity.IsEnum then "Enum"
                        elif entity.IsDelegate then "Delegate"
                        elif entity.IsFSharpAbbreviation then "Abbreviation"
                        elif entity.IsArrayType then "Array"
                        else "Type"
                    | :? FSharpField -> "Field"
                    | :? FSharpUnionCase -> "UnionCase"
                    | :? FSharpParameter -> "Parameter"
                    | :? FSharpGenericParameter -> "GenericParameter"
                    | :? FSharpActivePatternCase -> "ActivePattern"
                    | _ -> "Other"
                Assembly = symbol.Assembly.SimpleName
                IsFromDefinition = true
            |}
        )
    
    let exportData = {|
        Symbols = symbolData
        Count = symbolData.Length
        HashToName = symbolData |> Array.map (fun s -> s.Hash, s.FullName) |> Map.ofArray
    |}
    
    let json = JsonSerializer.Serialize(exportData, jsonOptions)
    writeFileToPath outputPath json
    printfn "  Wrote symbol table (%d symbols, %d bytes)" symbolData.Length json.Length

/// Generate all debug outputs
let generateDebugOutputs 
    (graph: ProgramSemanticGraph) 
    (correlations: (range * FSharpSymbol)[]) 
    (stats: CorrelationStats) 
    (outputDir: string) =
    
    printfn "[PSG Debug] Generating debug outputs..."
    
    // Ensure output directory exists
    Directory.CreateDirectory(outputDir) |> ignore
    
    // Generate correlation map
    let corrPath = Path.Combine(outputDir, "psg.corr.json")
    generateCorrelationMap correlations corrPath
    
    // Generate node graph
    let nodesPath = Path.Combine(outputDir, "psg.nodes.json")
    generateNodeGraph graph nodesPath
    
    // Generate symbol table
    let symbolsPath = Path.Combine(outputDir, "psg.symbols.json")
    generateSymbolTable graph symbolsPath
    
    // Generate statistics
    let statsData = {|
        TotalSymbols = stats.TotalSymbols
        CorrelatedNodes = stats.CorrelatedNodes
        UncorrelatedNodes = stats.UncorrelatedNodes
        CorrelationRate = float stats.CorrelatedNodes / float stats.TotalSymbols
        SymbolsByKind = stats.SymbolsByKind |> Map.toArray
        FileCoverage = 
            stats.FileCoverage 
            |> Map.toArray 
            |> Array.map (fun (file, coverage) -> 
                {| File = Path.GetFileName(file); Coverage = coverage |})
    |}
    
    let statsPath = Path.Combine(outputDir, "psg.stats.json")
    let statsJson = JsonSerializer.Serialize(statsData, jsonOptions)
    writeFileToPath statsPath statsJson
    printfn "  Wrote correlation statistics (%d bytes)" statsJson.Length
    
    // Generate summary report
    let summary = 
        sprintf """PSG Build Summary
=================
Total Nodes: %d
Total Edges: %d
Entry Points: %d
Symbol Correlation Rate: %.2f%%

Files Processed: %d
Average File Coverage: %.2f%%

Symbol Distribution:
%s
"""
            graph.Nodes.Count
            graph.Edges.Length
            graph.EntryPoints.Length
            (if stats.TotalSymbols > 0 then float stats.CorrelatedNodes / float stats.TotalSymbols * 100.0 else 0.0)
            stats.FileCoverage.Count
            (if stats.FileCoverage.Count > 0 then 
                stats.FileCoverage |> Map.toSeq |> Seq.map snd |> Seq.average
             else 0.0)
            (stats.SymbolsByKind 
             |> Map.toSeq 
             |> Seq.map (fun (kind, count) -> sprintf "  %s: %d" kind count)
             |> String.concat "\n")
    
    let summaryPath = Path.Combine(outputDir, "psg.summary.txt")
    File.WriteAllText(summaryPath, summary)
    printfn "  Wrote summary report"
    
    printfn "[PSG Debug] Debug outputs complete"