module Core.Analysis.Reachability

open System
open FSharp.Compiler.Symbols
open Core.PSG.Types
open Core.Analysis.CouplingCohesion

/// Reachability analysis result
type ReachabilityResult = {
    EntryPoints: FSharpSymbol list
    ReachableSymbols: Set<string>
    UnreachableSymbols: Set<string>
    CallGraph: Map<string, string list>
}

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

/// Classify symbol by library boundary
let classifySymbol (symbol: FSharpSymbol) : LibraryCategory =
    let fullName = symbol.FullName
    match fullName with
    | name when name.StartsWith("Alloy.") -> AlloyLibrary
    | name when name.StartsWith("FSharp.Core.") || name.StartsWith("Microsoft.FSharp.") -> FSharpCore
    | _ -> UserCode

/// Extract function calls by analyzing Application nodes and their References edges
let extractFunctionCalls (psg: ProgramSemanticGraph) =
    // Find all Application nodes (function calls)
    let applicationNodes = 
        psg.Nodes
        |> Map.toSeq
        |> Seq.filter (fun (_, node) -> node.SyntaxKind = "Application")
        |> Seq.map fst
        |> Set.ofSeq
    
    // Find References edges from Application nodes to function symbols
    psg.Edges
    |> List.choose (fun edge ->
        match edge.Kind with
        | SymRef when Set.contains edge.Source.Value applicationNodes ->
            // This is a reference from a function call site
            let sourceNode = Map.tryFind edge.Source.Value psg.Nodes
            let targetNode = Map.tryFind edge.Target.Value psg.Nodes
            
            match sourceNode, targetNode with
            | Some src, Some tgt ->
                match src.Symbol, tgt.Symbol with
                | Some srcSymbol, Some tgtSymbol ->
                    let srcName = srcSymbol.FullName
                    let tgtName = tgtSymbol.FullName
                    
                    // Include calls to library functions and user functions
                    if (tgtName.StartsWith("Alloy.") || 
                        tgtName.StartsWith("FSharp.Core.") ||
                        tgtName.StartsWith("Examples.")) &&
                       srcName <> tgtName then
                        Some (srcName, tgtName)
                    else None
                | _ -> None
            | _ -> None
        | _ -> None
    )

/// Build call graph from function relationships
let buildCallGraph (relationships: (string * string) list) =
    relationships
    |> List.groupBy fst
    |> List.map (fun (caller, calls) ->
        caller, calls |> List.map snd |> List.distinct
    )
    |> Map.ofList

/// Extract entry points from PSG
let extractEntryPoints (psg: ProgramSemanticGraph) =
    psg.EntryPoints
    |> List.choose (fun entryPointId ->
        match entryPointId with
        | SymbolNode(_, symbolName) ->
            Map.tryFind symbolName psg.SymbolTable
        | RangeNode(file, sl, sc, el, ec) ->
            psg.Nodes
            |> Map.toSeq
            |> Seq.tryPick (fun (_, node) ->
                if node.Range.Start.Line = sl && 
                   node.Range.Start.Column = sc &&
                   node.Range.End.Line = el &&
                   node.Range.End.Column = ec &&
                   node.SourceFile.EndsWith(System.IO.Path.GetFileName(file))
                then node.Symbol
                else None
            )
    )

/// Compute reachable symbols using breadth-first traversal
let computeReachableSymbols (entryPoints: FSharpSymbol list) (callGraph: Map<string, string list>) (allSymbols: Set<string>) =
    let rec traverse (visited: Set<string>) (queue: string list) =
        match queue with
        | [] -> visited
        | current :: remaining ->
            if Set.contains current visited then
                traverse visited remaining
            else
                let newVisited = Set.add current visited
                let callees = Map.tryFind current callGraph |> Option.defaultValue []
                let newQueue = callees @ remaining
                traverse newVisited newQueue
    
    let entryPointNames = entryPoints |> List.map (fun ep -> ep.FullName)
    let initialQueue = entryPointNames |> List.filter (fun name -> Set.contains name allSymbols)
    
    traverse Set.empty initialQueue

/// Enhanced reachability analysis using Application nodes
let analyzeReachabilityWithBoundaries (psg: ProgramSemanticGraph) : LibraryAwareReachability =
    let startTime = DateTime.UtcNow
    
    printfn "[REACHABILITY] === PSG-Based Analysis Start ==="
    printfn "[REACHABILITY] PSG nodes: %d" (Map.count psg.Nodes)
    printfn "[REACHABILITY] PSG edges: %d" psg.Edges.Length
    printfn "[REACHABILITY] Entry points: %d" psg.EntryPoints.Length
    
    // Count Application nodes for debugging
    let applicationCount = 
        psg.Nodes 
        |> Map.toSeq 
        |> Seq.filter (fun (_, node) -> node.SyntaxKind = "Application") 
        |> Seq.length
    
    printfn "[REACHABILITY] Application nodes (function calls): %d" applicationCount
    
    // Extract all meaningful symbols
    let allSymbols = 
        psg.Nodes
        |> Map.toSeq
        |> Seq.choose (fun (_, node) -> 
            match node.Symbol with
            | Some symbol -> 
                let name = symbol.FullName
                if not (name.Contains("@") || name.Contains("$") || name.Length < 3) then
                    Some symbol
                else None
            | None -> None
        )
        |> Seq.toArray
    
    let allSymbolNames = allSymbols |> Array.map (fun s -> s.FullName) |> Set.ofArray
    
    printfn "[REACHABILITY] Total meaningful symbols: %d" allSymbols.Length
    
    // Extract entry points
    let entryPoints = extractEntryPoints psg
    
    printfn "[REACHABILITY] === Entry Point Analysis ==="
    printfn "[REACHABILITY] Entry points found: %d" entryPoints.Length
    
    entryPoints |> List.iter (fun ep ->
        printfn "  Entry Point: %s (%s)" ep.DisplayName ep.FullName)
    
    // Extract function calls from Application nodes
    let functionCalls = extractFunctionCalls psg
    
    printfn "[REACHABILITY] === Function Call Analysis ==="
    printfn "[REACHABILITY] Function relationships extracted: %d" functionCalls.Length
    
    if functionCalls.Length > 0 then
        printfn "[REACHABILITY] Sample function calls:"
        functionCalls 
        |> List.take (min 10 functionCalls.Length)
        |> List.iter (fun (src, tgt) ->
            printfn "  %s -> %s" src tgt)
    else
        printfn "[REACHABILITY] ⚠️  NO FUNCTION CALLS found"
    
    // Build call graph
    let callGraph = buildCallGraph functionCalls
    
    printfn "[REACHABILITY] Call graph entries: %d" (Map.count callGraph)
    
    // Compute reachable symbols
    let reachableSymbols = computeReachableSymbols entryPoints callGraph allSymbolNames
    let unreachableSymbols = Set.difference allSymbolNames reachableSymbols
    
    printfn "[REACHABILITY] === Reachability Results ==="
    printfn "[REACHABILITY] Reachable symbols: %d" (Set.count reachableSymbols)
    printfn "[REACHABILITY] Unreachable symbols: %d" (Set.count unreachableSymbols)
    
    if Set.count reachableSymbols > 0 then
        printfn "[REACHABILITY] Sample reachable symbols:"
        reachableSymbols 
        |> Set.toArray 
        |> Array.take (min 15 (Set.count reachableSymbols))
        |> Array.iter (fun symbolId -> printfn "  ✓ %s" symbolId)
    
    // Generate library categorization
    let libraryCategories = 
        allSymbols
        |> Array.map (fun symbol -> symbol.FullName, classifySymbol symbol)
        |> Map.ofArray
    
    let endTime = DateTime.UtcNow
    let computationTime = (endTime - startTime).TotalMilliseconds |> int64
    
    let basicResult = {
        EntryPoints = entryPoints
        ReachableSymbols = reachableSymbols
        UnreachableSymbols = unreachableSymbols
        CallGraph = callGraph
    }
    
    let pruningStats = {
        TotalSymbols = allSymbols.Length
        ReachableSymbols = Set.count reachableSymbols
        EliminatedSymbols = allSymbols.Length - Set.count reachableSymbols
        ComputationTimeMs = computationTime
    }
    
    let eliminationRate = 
        if allSymbols.Length > 0 then 
            (float pruningStats.EliminatedSymbols / float allSymbols.Length) * 100.0 
        else 0.0
    
    printfn "[REACHABILITY] === Summary ==="
    printfn "[REACHABILITY] Analysis completed in %dms" computationTime
    printfn "[REACHABILITY] Elimination rate: %.2f%%" eliminationRate
    printfn "[REACHABILITY] ==========================="
    
    {
        BasicResult = basicResult
        LibraryCategories = libraryCategories
        PruningStatistics = pruningStats
    }