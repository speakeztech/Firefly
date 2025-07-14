module Core.PSG.Reachability

open System
open FSharp.Compiler.Symbols
open FSharp.Compiler.Text
open Core.PSG.Types

/// Reachability analysis result with comprehensive symbol tracking
type ReachabilityResult = {
    EntryPoints: FSharpSymbol list
    ReachableSymbols: Set<string>
    UnreachableSymbols: Set<string>
    CallGraph: Map<string, string list>
}

/// Library classification for symbol boundary analysis
type LibraryCategory =
    | UserCode
    | AlloyLibrary  
    | FSharpCore
    | Other of libraryName: string

/// Enhanced reachability result with library boundary information and statistics
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

/// Unified symbol extraction using identical filtering criteria as DebugOutput.fs
module UnifiedSymbolExtraction =
    
    /// Extract meaningful symbols using standardized filtering logic
    let extractMeaningfulSymbols (psg: ProgramSemanticGraph) : FSharpSymbol array =
        psg.Nodes
        |> Map.toSeq
        |> Seq.choose (fun (_, node) -> 
            match node.Symbol with
            | Some symbol -> 
                let name = symbol.FullName
                // Apply identical filtering criteria as DebugOutput.fs
                if not (name.Contains("@") || name.Contains("$") || name.Length < 3) 
                   && not (name.StartsWith("op_") || name.Contains(".cctor") || name.Contains("..ctor"))
                   && not (name.Contains("get_") || name.Contains("set_")) then
                    Some symbol
                else None
            | None -> None
        )
        |> Seq.distinctBy (fun s -> s.FullName)
        |> Seq.toArray
    
    /// Extract all nodes with valid symbol correlations
    let extractCorrelatedNodes (psg: ProgramSemanticGraph) : (string * PSGNode) array =
        psg.Nodes
        |> Map.toSeq
        |> Seq.filter (fun (_, node) -> node.Symbol.IsSome)
        |> Seq.toArray

/// Classify symbol by library boundary for analysis purposes
let classifySymbol (symbol: FSharpSymbol) : LibraryCategory =
    let fullName = symbol.FullName
    match fullName with
    | name when name.StartsWith("Alloy.") -> AlloyLibrary
    | name when name.StartsWith("FSharp.Core.") || name.StartsWith("Microsoft.FSharp.") -> FSharpCore
    | _ -> UserCode

/// Determine if symbol represents a callable function or method
let isFunction (symbol: FSharpSymbol) : bool =
    match symbol with
    | :? FSharpMemberOrFunctionOrValue as mfv -> mfv.IsFunction || mfv.IsMember
    | _ -> false

/// Find the containing function for a given node
let findContainingFunction (psg: ProgramSemanticGraph) (nodeId: string) : string option =
    let visited = System.Collections.Generic.HashSet<string>()
    
    let rec traverse currentId =
        if visited.Contains(currentId) then
            printfn "[CYCLE] Detected cycle at node: %s" currentId
            None
        else
            visited.Add(currentId) |> ignore
            match Map.tryFind currentId psg.Nodes with
            | Some node ->
                match node.Symbol with
                | Some (:? FSharpMemberOrFunctionOrValue as mfv) when mfv.IsFunction ->
                    Some mfv.FullName
                | _ ->
                    match node.ParentId with
                    | Some parentId -> 
                        traverse parentId.Value
                    | None -> 
                        None
            | None -> 
                None
    
    traverse nodeId

/// Enhanced entry point detection using multiple comprehensive strategies
let extractEntryPoints (psg: ProgramSemanticGraph) =
    printfn "[ENTRY POINTS] === Entry Point Extraction ==="
    
    let meaningfulSymbols = UnifiedSymbolExtraction.extractMeaningfulSymbols psg
    let correlatedNodes = UnifiedSymbolExtraction.extractCorrelatedNodes psg
    
    printfn "[ENTRY POINTS] Analyzing %d meaningful symbols and %d correlated nodes" meaningfulSymbols.Length correlatedNodes.Length
    
    // Strategy 1: Search meaningful symbols for main functions and EntryPoint attributes
    let symbolBasedEntryPoints = 
        meaningfulSymbols
        |> Array.choose (fun symbol ->
            match symbol with
            | symbol when symbol.DisplayName = "main" ->
                printfn "[ENTRY POINTS] Found main function: %s" symbol.FullName
                Some symbol
            | :? FSharpMemberOrFunctionOrValue as mfv ->
                let hasEntryPoint = 
                    try
                        mfv.Attributes
                        |> Seq.exists (fun attr ->
                            let attrName = attr.AttributeType.DisplayName
                            attrName = "EntryPoint" || attrName = "EntryPointAttribute")
                    with
                    | _ -> false
                
                if hasEntryPoint then
                    printfn "[ENTRY POINTS] Found EntryPoint attribute: %s" symbol.FullName
                    Some symbol
                else None
            | _ -> None
        )
        |> Array.toList
    
    // Strategy 2: Search PSG nodes directly for entry point patterns
    let nodeBasedEntryPoints = 
        correlatedNodes
        |> Array.choose (fun (_, node) ->
            match node.Symbol with
            | Some symbol ->
                let name = symbol.DisplayName.ToLowerInvariant()
                if name = "main" || name.Contains("main") then
                    printfn "[ENTRY POINTS] Found main-pattern in node: %s" symbol.FullName
                    Some symbol
                elif node.SyntaxKind.Contains("Binding") && symbol.FullName.Contains("main") then
                    printfn "[ENTRY POINTS] Found main binding: %s" symbol.FullName
                    Some symbol
                else None
            | None -> None
        )
        |> Array.toList
    
    // Strategy 3: Search for HelloWorld or DirectMain patterns (project-specific)
    let projectSpecificEntryPoints = 
        meaningfulSymbols
        |> Array.choose (fun symbol ->
            let fullName = symbol.FullName
            if fullName.Contains("HelloWorld") || fullName.Contains("DirectMain") || fullName.Contains("hello") then
                printfn "[ENTRY POINTS] Found project-specific entry point pattern: %s" fullName
                Some symbol
            else None
        )
        |> Array.toList
    
    // Strategy 4: Check PSG explicit entry points collection
    let explicitEntryPoints = 
        psg.EntryPoints
        |> List.choose (fun entryId ->
            match Map.tryFind entryId.Value psg.Nodes with
            | Some node when node.Symbol.IsSome ->
                printfn "[ENTRY POINTS] Found explicit PSG entry point: %s" node.Symbol.Value.FullName
                Some node.Symbol.Value
            | _ -> None
        )
    
    // Combine and deduplicate all entry points
    let allEntryPoints = 
        [symbolBasedEntryPoints; nodeBasedEntryPoints; projectSpecificEntryPoints; explicitEntryPoints]
        |> List.concat
        |> List.distinctBy (fun ep -> ep.FullName)
    
    printfn "[ENTRY POINTS] Total entry points detected: %d" allEntryPoints.Length
    
    if allEntryPoints.IsEmpty then
        printfn "[ENTRY POINTS] WARNING: No entry points detected using any strategy!"
        printfn "[ENTRY POINTS] Meaningful symbols examined: %d" meaningfulSymbols.Length
        printfn "[ENTRY POINTS] Correlated nodes examined: %d" correlatedNodes.Length
        printfn "[ENTRY POINTS] Sample meaningful symbols:"
        meaningfulSymbols 
        |> Array.take (min 10 meaningfulSymbols.Length)
        |> Array.iter (fun s -> printfn "  - %s (%s)" s.FullName s.DisplayName)
    else
        allEntryPoints |> List.iter (fun ep -> 
            printfn "[ENTRY POINTS] ✓ %s" ep.FullName)
    
    allEntryPoints

/// Extract function calls using PSG FunctionCall edges
let extractFunctionCalls (psg: ProgramSemanticGraph) =
    // printfn "[CALL EXTRACTION] === Function Call Analysis ==="
    
    let functionCalls = 
        psg.Edges
        |> List.choose (fun edge ->
            if edge.Kind = FunctionCall then
                let sourceNode = Map.tryFind edge.Source.Value psg.Nodes
                let targetNode = Map.tryFind edge.Target.Value psg.Nodes
                
                match sourceNode, targetNode with
                | Some src, Some tgt when tgt.Symbol.IsSome ->
                    let callingContext = findContainingFunction psg edge.Source.Value
                    match callingContext with
                    | Some caller -> 
                        Some (caller, tgt.Symbol.Value.FullName)
                    | None -> 
                        match src.Symbol with
                        | Some srcSymbol -> Some (srcSymbol.FullName, tgt.Symbol.Value.FullName)
                        | None -> 
                            // printfn "[CALL EXTRACTION] Cannot determine calling context for %s" tgt.Symbol.Value.FullName
                            None
                | _ -> None
            else None
        )
    
    // printfn "[CALL EXTRACTION] Function calls found: %d" functionCalls.Length
    (*
    if functionCalls.Length > 0 then
        printfn "[CALL EXTRACTION] Function calls:"
        functionCalls 
        |> List.iter (fun (src, tgt) -> printfn "  %s -> %s" src tgt)
    else
        printfn "[CALL EXTRACTION] WARNING: No function calls detected!"
        let functionCallEdges = psg.Edges |> List.filter (fun e -> e.Kind = FunctionCall) |> List.length
        printfn "[CALL EXTRACTION] FunctionCall edges available: %d" functionCallEdges
    *)
    functionCalls

/// Build comprehensive call graph with entry point integration
let buildCallGraph (functionCalls: (string * string) list) (entryPoints: FSharpSymbol list) =
    printfn "[CALL GRAPH] === Call Graph Construction ==="
    
    let callGraph = 
        functionCalls
        |> List.groupBy fst
        |> List.map (fun (caller, calls) ->
            caller, calls |> List.map snd |> List.distinct)
        |> Map.ofList
    
    let finalCallGraph = 
        entryPoints
        |> List.fold (fun graph ep ->
            let epName = ep.FullName
            if not (Map.containsKey epName graph) then
                Map.add epName [] graph
            else graph
        ) callGraph
    
    printfn "[CALL GRAPH] Call graph entries: %d" (Map.count finalCallGraph)
    
    if Map.isEmpty finalCallGraph then
        printfn "[CALL GRAPH] WARNING: Empty call graph constructed!"
    else
        let sampleEntries = finalCallGraph |> Map.toSeq |> Seq.take (min 5 (Map.count finalCallGraph))
        printfn "[CALL GRAPH] Sample entries:"
        sampleEntries |> Seq.iter (fun (caller, callees) ->
            printfn "  %s -> [%s]" caller (String.concat "; " callees))
    
    finalCallGraph

/// Compute reachable symbols using breadth-first traversal with comprehensive validation
let computeReachableSymbols (entryPoints: FSharpSymbol list) (callGraph: Map<string, string list>) (allSymbolNames: Set<string>) =
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
    printfn "[REACHABILITY] Starting from entry points: %A" entryPointNames
    
    let symbolsWithEntryPoints = 
        entryPointNames 
        |> List.fold (fun acc ep -> Set.add ep acc) allSymbolNames
    
    let reachable = traverse Set.empty entryPointNames
    
    printfn "[REACHABILITY] Found %d reachable symbols from %d total" (Set.count reachable) (Set.count symbolsWithEntryPoints)
    
    if Set.count reachable > 0 then
        printfn "[REACHABILITY] Sample reachable symbols:"
        reachable |> Set.toArray |> Array.take (min 5 (Set.count reachable))
        |> Array.iter (fun s -> printfn "  ✓ %s" s)
    else
        printfn "[REACHABILITY] WARNING: No symbols marked as reachable!"
        printfn "[REACHABILITY] Entry points provided: %d" entryPoints.Length
        printfn "[REACHABILITY] Call graph size: %d" (Map.count callGraph)
        printfn "[REACHABILITY] Available symbols: %d" (Set.count allSymbolNames)
    
    reachable

/// Main entry point for PSG-based reachability analysis with unified symbol extraction
let analyzeReachabilityWithBoundaries (psg: ProgramSemanticGraph) : LibraryAwareReachability =
    let startTime = DateTime.UtcNow
    
    printfn "[REACHABILITY] === PSG-Based Analysis Start ==="
    printfn "[REACHABILITY] PSG nodes: %d" (Map.count psg.Nodes)
    printfn "[REACHABILITY] PSG edges: %d" psg.Edges.Length
    printfn "[REACHABILITY] PSG entry points: %d" psg.EntryPoints.Length
    
    // Use unified symbol extraction for consistent measurements
    let meaningfulSymbols = UnifiedSymbolExtraction.extractMeaningfulSymbols psg
    let correlatedNodes = UnifiedSymbolExtraction.extractCorrelatedNodes psg
    let meaningfulSymbolNames = meaningfulSymbols |> Array.map (fun s -> s.FullName) |> Set.ofArray
    
    printfn "[REACHABILITY] Total meaningful symbols (unified): %d" meaningfulSymbols.Length
    printfn "[REACHABILITY] Total correlated nodes: %d" correlatedNodes.Length
    
    // Extract entry points using enhanced detection
    let entryPoints = extractEntryPoints psg
    
    // Extract function calls using ChildrenState-compatible analysis
    let functionCalls = extractFunctionCalls psg
    
    // Build comprehensive call graph
    let callGraph = buildCallGraph functionCalls entryPoints
    
    // Compute reachable symbols from entry points
    let reachableSymbols = computeReachableSymbols entryPoints callGraph meaningfulSymbolNames
    let unreachableSymbols = Set.difference meaningfulSymbolNames reachableSymbols
    
    // Generate library classifications using meaningful symbols
    let libraryCategories = 
        meaningfulSymbols
        |> Array.map (fun symbol -> symbol.FullName, classifySymbol symbol)
        |> Map.ofArray
    
    // Calculate accurate statistics using unified measurements
    let endTime = DateTime.UtcNow
    let computationTime = int64 (endTime - startTime).TotalMilliseconds
    
    let reachableCount = Set.count reachableSymbols
    let unreachableCount = Set.count unreachableSymbols
    let totalCount = meaningfulSymbols.Length
    
    let pruningStats = {
        TotalSymbols = totalCount
        ReachableSymbols = reachableCount
        EliminatedSymbols = unreachableCount
        ComputationTimeMs = computationTime
    }
    
    printfn "[REACHABILITY] === Results ==="
    printfn "[REACHABILITY] Reachable symbols: %d" reachableCount
    printfn "[REACHABILITY] Unreachable symbols: %d" unreachableCount
    printfn "[REACHABILITY] Analysis completed in %dms" computationTime
    
    let eliminationRate = 
        if totalCount > 0 then
            (float unreachableCount / float totalCount) * 100.0
        else 0.0
    
    printfn "[REACHABILITY] Elimination rate: %.2f%% (%d eliminated out of %d total)" eliminationRate unreachableCount totalCount
    
    // Construct comprehensive result with accurate measurements
    let basicResult = {
        EntryPoints = entryPoints
        ReachableSymbols = reachableSymbols
        UnreachableSymbols = unreachableSymbols
        CallGraph = callGraph
    }
    
    {
        BasicResult = basicResult
        LibraryCategories = libraryCategories
        PruningStatistics = pruningStats
    }