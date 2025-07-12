module Core.Analysis.Reachability

open System
open FSharp.Compiler.Symbols
open FSharp.Compiler.Text
open Core.PSG.Types

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

/// Check if symbol represents a function
let isFunction (symbol: FSharpSymbol) : bool =
    match symbol with
    | :? FSharpMemberOrFunctionOrValue as mfv -> mfv.IsFunction || mfv.IsMember
    | _ -> false

/// Find enclosing function for a node by traversing containment edges upward
let rec findEnclosingFunction (psg: ProgramSemanticGraph) (nodeId: string) (depth: int) : string option =
    printfn "[ENCLOSING] Depth %d: Searching from node %s" depth nodeId
    if depth > 10 then 
        printfn "[ENCLOSING] Max depth reached"
        None
    else
        match Map.tryFind nodeId psg.Nodes with
        | Some node when node.Symbol.IsSome ->
            let symbol = node.Symbol.Value
            printfn "[ENCLOSING] Found symbol: %s (IsFunction: %b)" symbol.FullName (isFunction symbol)
            if isFunction symbol then
                Some symbol.FullName
            else
                let parentEdges = psg.Edges |> List.filter (fun edge -> edge.Target.Value = nodeId && edge.Kind = ChildOf)
                printfn "[ENCLOSING] Found %d parent edges" parentEdges.Length
                psg.Edges
                |> List.tryPick (fun edge ->
                    if edge.Target.Value = nodeId && edge.Kind = ChildOf then
                        printfn "[ENCLOSING] Following parent edge to %s" edge.Source.Value
                        findEnclosingFunction psg edge.Source.Value (depth + 1)
                    else None
                )
        | Some node ->
            printfn "[ENCLOSING] Node has no symbol, checking parents"
            let parentEdges = psg.Edges |> List.filter (fun edge -> edge.Target.Value = nodeId && edge.Kind = ChildOf)
            printfn "[ENCLOSING] Found %d parent edges" parentEdges.Length
            psg.Edges
            |> List.tryPick (fun edge ->
                if edge.Target.Value = nodeId && edge.Kind = ChildOf then
                    printfn "[ENCLOSING] Following parent edge to %s" edge.Source.Value
                    findEnclosingFunction psg edge.Source.Value (depth + 1)
                else None
            )
        | None ->
            printfn "[ENCLOSING] Node not found: %s" nodeId
            None
            
/// Extract function calls using PSG structure and edge relationships
let extractFunctionCallsFromPSG (psg: ProgramSemanticGraph) =
    let nodeIdToSymbolName = 
        psg.Nodes
        |> Map.toSeq
        |> Seq.choose (fun (nodeId, node) -> 
            match node.Symbol with
            | Some symbol -> Some (nodeId, symbol.FullName)
            | None -> None)
        |> Map.ofSeq

    // Extract calls by finding edges between nodes with symbols
    psg.Edges
    |> List.choose (fun edge ->
        match edge.Kind with
        | SymRef | FunctionCall ->  // CHANGED from CallsFunction
            let sourceSymbol = Map.tryFind edge.Source.Value nodeIdToSymbolName
            let targetSymbol = Map.tryFind edge.Target.Value nodeIdToSymbolName
            
            match sourceSymbol, targetSymbol with
            | Some src, Some tgt -> Some (src, tgt)
            | _ -> None
        | _ -> None
    )

/// Enhanced function call extraction using multiple PSG traversal strategies
let extractComprehensiveFunctionCalls (psg: ProgramSemanticGraph) =
    printfn "[CALL EXTRACTION] === Comprehensive Function Call Analysis ==="
    
    // Strategy 1: Find Application nodes and their function targets via CallsFunction edges
    let applicationCalls = 
        psg.Edges
        |> List.choose (fun edge ->
            if edge.Kind = FunctionCall then
                let sourceNode = Map.tryFind edge.Source.Value psg.Nodes
                let targetNode = Map.tryFind edge.Target.Value psg.Nodes
                
                match sourceNode, targetNode with
                | Some src, Some tgt when src.SyntaxKind = "Application" && tgt.Symbol.IsSome ->
                    // Find the calling context (parent function)
                    let callingContext = findEnclosingFunction psg edge.Source.Value 0
                    match callingContext with
                    | Some caller -> Some (caller, tgt.Symbol.Value.FullName)
                    | None -> Some ("TopLevel", tgt.Symbol.Value.FullName)
                | _ -> None
            else None
        )
    
    printfn "[CALL EXTRACTION] Application-based calls found: %d" applicationCalls.Length
    
    // Strategy 2: Direct function-to-function references  
    let directFunctionCalls = 
        psg.Edges
        |> List.choose (fun edge ->
            match edge.Kind with
            | SymRef ->
                let sourceNode = Map.tryFind edge.Source.Value psg.Nodes
                let targetNode = Map.tryFind edge.Target.Value psg.Nodes
                
                match sourceNode, targetNode with
                | Some src, Some tgt when src.Symbol.IsSome && tgt.Symbol.IsSome ->
                    let srcSymbol = src.Symbol.Value
                    let tgtSymbol = tgt.Symbol.Value
                    
                    // Only include if target is a function and source is different
                    if isFunction tgtSymbol && srcSymbol.FullName <> tgtSymbol.FullName then
                        Some (srcSymbol.FullName, tgtSymbol.FullName)
                    else None
                | _ -> None
            | _ -> None
        )
    
    printfn "[CALL EXTRACTION] Direct function calls found: %d" directFunctionCalls.Length
    
    // Combine and deduplicate
    let allCalls = 
        [ applicationCalls; directFunctionCalls ]
        |> List.concat
        |> List.distinct
        |> List.filter (fun (src, tgt) -> src <> tgt) // Remove self-references
    
    printfn "[CALL EXTRACTION] Total unique calls after filtering: %d" allCalls.Length
    
    if allCalls.Length > 0 then
        printfn "[CALL EXTRACTION] Sample function calls:"
        allCalls 
        |> List.take (min 10 allCalls.Length)
        |> List.iter (fun (src, tgt) -> printfn "  %s -> %s" src tgt)
    
    allCalls

/// Build call graph from function relationships with enhanced entry point correlation
let buildEnhancedCallGraph (entryPoints: FSharpSymbol list) (relationships: (string * string) list) =
    let entryPointNames = entryPoints |> List.map (fun ep -> ep.FullName) |> Set.ofList
    
    printfn "[CALL GRAPH] === Enhanced Call Graph Construction ==="
    printfn "[CALL GRAPH] Entry points: %A" (Set.toList entryPointNames)
    
    // Group by caller and build basic call graph
    let basicCallGraph = 
        relationships
        |> List.groupBy fst
        |> List.map (fun (caller, calls) ->
            caller, calls |> List.map snd |> List.distinct)
        |> Map.ofList
    
    // Add entry points to call graph even if they don't appear as callers
    let enhancedCallGraph = 
        entryPointNames
        |> Set.fold (fun graph entryPoint ->
            if not (Map.containsKey entryPoint graph) then
                // Find any calls that might originate from this entry point
                let implicitCalls = 
                    relationships
                    |> List.choose (fun (caller, target) ->
                        if caller = "ModuleScope" || caller = "UnknownCaller" then
                            Some target
                        else None)
                    |> List.distinct
                
                Map.add entryPoint implicitCalls graph
            else graph
        ) basicCallGraph
    
    printfn "[CALL GRAPH] Basic call graph entries: %d" (Map.count basicCallGraph)
    printfn "[CALL GRAPH] Enhanced call graph entries: %d" (Map.count enhancedCallGraph)
    
    enhancedCallGraph

/// Extract entry points from PSG with enhanced symbol resolution
let extractEnhancedEntryPoints (psg: ProgramSemanticGraph) =
    printfn "[ENTRY POINTS] === Enhanced Entry Point Extraction ==="
    
    let directEntryPoints = 
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
    
    // Also look for main functions by pattern matching
    let mainFunctionEntryPoints = 
        psg.Nodes
        |> Map.toSeq
        |> Seq.choose (fun (_, node) ->
            match node.Symbol with
            | Some symbol when symbol.DisplayName = "main" || symbol.FullName.EndsWith(".main") ->
                Some symbol
            | _ -> None
        )
        |> Seq.toList
    
    let allEntryPoints = 
        [ directEntryPoints; mainFunctionEntryPoints ]
        |> List.concat
        |> List.distinct
    
    printfn "[ENTRY POINTS] Direct entry points: %d" directEntryPoints.Length
    printfn "[ENTRY POINTS] Main function entry points: %d" mainFunctionEntryPoints.Length
    printfn "[ENTRY POINTS] Total unique entry points: %d" allEntryPoints.Length
    
    allEntryPoints |> List.iter (fun ep ->
        printfn "[ENTRY POINTS]   %s (%s)" ep.DisplayName ep.FullName)
    
    allEntryPoints

/// Compute reachable symbols using enhanced breadth-first traversal
let computeEnhancedReachableSymbols (entryPoints: FSharpSymbol list) (callGraph: Map<string, string list>) (allSymbols: Set<string>) =
    let rec traverse (visited: Set<string>) (queue: string list) =
        match queue with
        | [] -> visited
        | current :: remaining ->
            if Set.contains current visited then
                traverse visited remaining
            else
                let newVisited = Set.add current visited
                let callees = Map.tryFind current callGraph |> Option.defaultValue []
                let filteredCallees = callees |> List.filter (fun callee -> not (Set.contains callee newVisited))
                let newQueue = filteredCallees @ remaining
                traverse newVisited newQueue
    
    let entryPointNames = entryPoints |> List.map (fun ep -> ep.FullName)
    printfn "[REACHABILITY] Starting traversal from entry points: %A" entryPointNames
    
    let initialQueue = entryPointNames |> List.filter (fun name -> Set.contains name allSymbols)
    printfn "[REACHABILITY] Valid entry points in symbol set: %d" initialQueue.Length
    
    let reachable = traverse Set.empty initialQueue
    
    printfn "[REACHABILITY] Traversal completed, found %d reachable symbols" (Set.count reachable)
    if Set.count reachable > 0 then
        printfn "[REACHABILITY] Sample reachable symbols:"
        reachable |> Set.toArray |> Array.take (min 10 (Set.count reachable)) 
        |> Array.iter (fun s -> printfn "  ✓ %s" s)
    
    reachable

/// Enhanced reachability analysis using comprehensive PSG traversal
let analyzeReachabilityWithBoundaries (psg: ProgramSemanticGraph) : LibraryAwareReachability =
    let startTime = DateTime.UtcNow
    
    printfn "[REACHABILITY] === Enhanced PSG-Based Analysis Start ==="
    printfn "[REACHABILITY] PSG nodes: %d" (Map.count psg.Nodes)
    printfn "[REACHABILITY] PSG edges: %d" psg.Edges.Length
    printfn "[REACHABILITY] Entry points: %d" psg.EntryPoints.Length
    
    // Extract all meaningful symbols with enhanced filtering
    let allSymbols = 
        psg.Nodes
        |> Map.toSeq
        |> Seq.choose (fun (_, node) -> 
            match node.Symbol with
            | Some symbol -> 
                let name = symbol.FullName
                if not (name.Contains("@") || name.Contains("$") || name.Length < 3) 
                   && not (name.StartsWith("op_") || name.Contains(".cctor")) then
                    Some symbol
                else None
            | None -> None
        )
        |> Seq.toArray
    
    let allSymbolNames = allSymbols |> Array.map (fun s -> s.FullName) |> Set.ofArray
    
    printfn "[REACHABILITY] Total meaningful symbols: %d" allSymbols.Length
    
    // Debug: Check if entry points are in symbol set
    printfn "[REACHABILITY] Symbol set sample:"
    allSymbolNames |> Set.toArray |> Array.take (min 10 (Set.count allSymbolNames))
    |> Array.iter (fun s -> printfn "  - %s" s)
    
    // Extract entry points with enhanced resolution
    let entryPoints = extractEnhancedEntryPoints psg
    
    // Ensure entry points are included in symbol set (critical fix)
    let allSymbolsWithEntryPoints = 
        let entryPointSet = entryPoints |> List.map (fun ep -> ep.FullName) |> Set.ofList
        let currentSymbolSet = allSymbolNames
        let missingEntryPoints = Set.difference entryPointSet currentSymbolSet
        
        if not (Set.isEmpty missingEntryPoints) then
            printfn "[REACHABILITY] ⚠️  Adding missing entry points to symbol set:"
            missingEntryPoints |> Set.iter (fun ep -> printfn "  + %s" ep)
            Set.union currentSymbolSet missingEntryPoints
        else 
            printfn "[REACHABILITY] ✅ All entry points found in symbol set"
            currentSymbolSet
    
    printfn "[REACHABILITY] Final symbol count: %d (including entry points)" (Set.count allSymbolsWithEntryPoints)
    
    // Extract function calls using comprehensive strategy
    let functionCalls = extractComprehensiveFunctionCalls psg
    
    // Build enhanced call graph
    let callGraph = buildEnhancedCallGraph entryPoints functionCalls
    
    // Compute reachable symbols with enhanced traversal
    let reachableSymbols = computeEnhancedReachableSymbols entryPoints callGraph allSymbolsWithEntryPoints
    let unreachableSymbols = Set.difference allSymbolNames reachableSymbols
    
    printfn "[REACHABILITY] === Final Results ==="
    printfn "[REACHABILITY] Reachable symbols: %d" (Set.count reachableSymbols)
    printfn "[REACHABILITY] Unreachable symbols: %d" (Set.count unreachableSymbols)
    
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