module Core.PSG.Reachability

open System
open FSharp.Compiler.Symbols
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
    
    // Soft-delete support
    MarkedPSG: ProgramSemanticGraph  // Contains all nodes with IsReachable flags
}

and PruningStatistics = {
    TotalSymbols: int
    ReachableSymbols: int
    EliminatedSymbols: int
    ComputationTimeMs: int64
}

/// Enhanced symbol extraction with better filtering
module UnifiedSymbolExtraction =
    
    /// Extract meaningful symbols using enhanced filtering logic
    let extractMeaningfulSymbols (psg: ProgramSemanticGraph) : FSharpSymbol array =
        psg.Nodes
        |> Map.toSeq
        |> Seq.choose (fun (_, node) -> 
            match node.Symbol with
            | Some symbol -> 
                let name = symbol.FullName
                // More permissive filtering - allow critical symbols through
                if not (name.Contains("@") || name.Contains("$")) 
                   && not (name.StartsWith("op_"))
                   && not (name.Contains(".cctor"))
                   && not (name.Contains("get_") && name.Contains("set_"))
                   && name.Length >= 2 then
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

/// Enhanced symbol name normalization for consistent matching
let normalizeSymbolName (symbolName: string) : string =
    // Handle common symbol name variations
    symbolName
        .Replace("`", "")
        .Replace("@", "")
        .Replace("$", "")
        .Trim()

/// Enhanced symbol matching with multiple strategies
let symbolMatches (psgSymbol: FSharpSymbol) (targetSymbolName: string) : bool =
    let psgFullName = normalizeSymbolName psgSymbol.FullName
    let psgDisplayName = normalizeSymbolName psgSymbol.DisplayName
    let normalizedTarget = normalizeSymbolName targetSymbolName
    
    // Strategy 1: Exact full name match
    if psgFullName = normalizedTarget then true
    // Strategy 2: Exact display name match  
    elif psgDisplayName = normalizedTarget then true
    // Strategy 3: Full name ends with target (for qualified names)
    elif psgFullName.EndsWith("." + normalizedTarget) then true
    // Strategy 4: Target ends with full name (for qualified targets)
    elif normalizedTarget.EndsWith("." + psgFullName) then true
    // Strategy 5: Both contain the same core identifier
    elif psgDisplayName.Length > 3 && normalizedTarget.Contains(psgDisplayName) then true
    elif normalizedTarget.Length > 3 && psgFullName.Contains(normalizedTarget) then true
    else false

/// Classify symbol by library boundary for analysis purposes
let classifySymbol (symbol: FSharpSymbol) : LibraryCategory =
    let fullName = symbol.FullName
    match fullName with
    | name when name.StartsWith("Alloy.") -> AlloyLibrary
    | name when name.StartsWith("FSharp.Core.") || name.StartsWith("Microsoft.FSharp.") -> FSharpCore
    | name when name.StartsWith("Examples.") || name.StartsWith("HelloWorld") -> UserCode
    | _ -> UserCode

/// Determine if symbol represents a callable function or method
let isFunction (symbol: FSharpSymbol) : bool =
    match symbol with
    | :? FSharpMemberOrFunctionOrValue as mfv -> mfv.IsFunction || mfv.IsMember
    | _ -> false

/// Enhanced entry point detection with comprehensive debugging
let extractEntryPoints (psg: ProgramSemanticGraph) =
    printfn "[ENTRY POINTS] === Enhanced Entry Point Extraction ==="
    
    let meaningfulSymbols = UnifiedSymbolExtraction.extractMeaningfulSymbols psg
    let correlatedNodes = UnifiedSymbolExtraction.extractCorrelatedNodes psg
    
    printfn "[ENTRY POINTS] Analyzing %d meaningful symbols and %d correlated nodes" meaningfulSymbols.Length correlatedNodes.Length
    
    // Strategy 1: Search meaningful symbols for main functions and EntryPoint attributes
    let symbolBasedEntryPoints = 
        meaningfulSymbols
        |> Array.choose (fun symbol ->
            match symbol with
            | symbol when symbol.DisplayName = "main" || symbol.DisplayName = "hello" ->
                printfn "[ENTRY POINTS] Found entry function: %s (Display: %s)" symbol.FullName symbol.DisplayName
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
                    printfn "[ENTRY POINTS] Found EntryPoint attribute: %s (Display: %s)" symbol.FullName symbol.DisplayName
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
                let syntaxKind = node.SyntaxKind
                if syntaxKind.Contains("EntryPoint") || syntaxKind.Contains("Main") || 
                   (syntaxKind.Contains("Binding") && symbol.DisplayName = "main") ||
                   symbol.FullName.Contains("hello") then
                    printfn "[ENTRY POINTS] Found entry point node: %s (%s)" symbol.FullName syntaxKind
                    Some symbol
                else None
            | None -> None
        )
        |> Array.toList
    
    // Strategy 3: Check PSG explicit entry points collection
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
        [symbolBasedEntryPoints; nodeBasedEntryPoints; explicitEntryPoints]
        |> List.concat
        |> List.distinctBy (fun ep -> ep.FullName)
    
    printfn "[ENTRY POINTS] Total entry points detected: %d" allEntryPoints.Length
    
    if allEntryPoints.IsEmpty then
        printfn "[ENTRY POINTS] WARNING: No entry points detected using any strategy!"
        printfn "[ENTRY POINTS] Available symbols sample:"
        meaningfulSymbols 
        |> Array.take (min 10 meaningfulSymbols.Length)
        |> Array.iter (fun s -> printfn "  - %s (%s)" s.FullName s.DisplayName)
    else
        allEntryPoints |> List.iter (fun ep -> 
            printfn "[ENTRY POINTS] ✓ %s (Display: %s)" ep.FullName ep.DisplayName)
    
    allEntryPoints

/// Enhanced function call extraction using improved PSG edge analysis
let rec extractFunctionCalls (psg: ProgramSemanticGraph) =
    printfn "[CALL EXTRACTION] === Enhanced Function Call Analysis ==="
    
    let functionCallEdges = psg.Edges |> List.filter (fun e -> e.Kind = FunctionCall)
    let symRefEdges = psg.Edges |> List.filter (fun e -> e.Kind = SymRef)
    
    printfn "[CALL EXTRACTION] Processing %d FunctionCall edges and %d SymRef edges" 
        functionCallEdges.Length symRefEdges.Length
    
    // Direct function call edges
    let directFunctionCalls = 
        functionCallEdges
        |> List.choose (fun edge ->
            let sourceNode = Map.tryFind edge.Source.Value psg.Nodes
            let targetNode = Map.tryFind edge.Target.Value psg.Nodes
            
            match sourceNode, targetNode with
            | Some src, Some tgt when tgt.Symbol.IsSome ->
                match src.Symbol with
                | Some srcSymbol -> 
                    printfn "[CALL EXTRACTION] Direct call: %s -> %s" srcSymbol.FullName tgt.Symbol.Value.FullName
                    Some (srcSymbol.FullName, tgt.Symbol.Value.FullName)
                | None -> 
                    findContainingFunction psg edge.Source.Value |> Option.map (fun caller ->
                        printfn "[CALL EXTRACTION] Context call: %s -> %s" caller tgt.Symbol.Value.FullName
                        (caller, tgt.Symbol.Value.FullName))
            | _ -> None
        )
    
    // Symbol reference edges
    let symbolRefCalls = 
        symRefEdges
        |> List.choose (fun edge ->
            let sourceNode = Map.tryFind edge.Source.Value psg.Nodes
            let targetNode = Map.tryFind edge.Target.Value psg.Nodes
            
            match sourceNode, targetNode with
            | Some src, Some tgt when src.Symbol.IsSome && tgt.Symbol.IsSome ->
                let srcSymbol = src.Symbol.Value
                let tgtSymbol = tgt.Symbol.Value
                
                if isFunction tgtSymbol then
                    printfn "[CALL EXTRACTION] Symbol ref call: %s -> %s" srcSymbol.FullName tgtSymbol.FullName
                    Some (srcSymbol.FullName, tgtSymbol.FullName)
                else None
            | _ -> None
        )
    
    let allFunctionCalls = [directFunctionCalls; symbolRefCalls] |> List.concat |> List.distinct
    
    printfn "[CALL EXTRACTION] Total function calls found: %d" allFunctionCalls.Length
    allFunctionCalls

/// Find the containing function for a given node
and findContainingFunction (psg: ProgramSemanticGraph) (nodeId: string) : string option =
    let visited = System.Collections.Generic.HashSet<string>()
    
    let rec traverse currentId =
        if visited.Contains(currentId) then
            None
        else
            visited.Add(currentId) |> ignore
            match Map.tryFind currentId psg.Nodes with
            | Some node ->
                match node.Symbol with
                | Some symbol ->
                    match symbol with
                    | :? FSharpMemberOrFunctionOrValue as mfv when mfv.IsFunction ->
                        Some symbol.FullName
                    | :? FSharpMemberOrFunctionOrValue as mfv when mfv.IsModuleValueOrMember && mfv.IsFunction ->
                        Some symbol.FullName
                    | _ ->
                        match node.ParentId with
                        | Some parentId -> traverse parentId.Value
                        | None -> None
                | None ->
                    match node.ParentId with
                    | Some parentId -> traverse parentId.Value
                    | None -> None
            | None -> None
    
    traverse nodeId

/// Build comprehensive call graph with entry point integration
let buildCallGraph (functionCalls: (string * string) list) (entryPoints: FSharpSymbol list) =
    printfn "[CALL GRAPH] === Enhanced Call Graph Construction ==="
    
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
    finalCallGraph

/// Compute reachable symbols using breadth-first traversal
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
        reachable |> Set.toArray |> Array.take (min 10 (Set.count reachable))
        |> Array.iter (fun s -> printfn "  ✓ %s" s)
    
    reachable

/// Deep traversal to mark all nodes reachable from entry points
let markReachabilityInPSGDeep (psg: ProgramSemanticGraph) (entryPoints: FSharpSymbol list) : ProgramSemanticGraph =
    printfn "[SOFT DELETE] === Deep Reachability Traversal ==="
    printfn "[SOFT DELETE] Starting from %d entry points in %d PSG nodes" entryPoints.Length (Map.count psg.Nodes)
    
    let mutable reachableNodeIds = Set.empty<string>
    let mutable processedNodeIds = Set.empty<string>
    
    // Extract the actual functions called from call graph
    let functionCalls = extractFunctionCalls psg
    let callGraph = buildCallGraph functionCalls entryPoints
    
    // Get all functions that are called (including those under TopLevel)
    let allCalledFunctions = 
        callGraph
        |> Map.toSeq
        |> Seq.collect (fun (_, callees) -> callees)
        |> Set.ofSeq
    
    // Combine with standard reachability
    let reachableFromEntryPoints = computeReachableSymbols entryPoints callGraph (Set.empty)
    let reachableSymbols = Set.union reachableFromEntryPoints allCalledFunctions
    
    printfn "[SOFT DELETE] Identified %d reachable symbols from call graph" (Set.count reachableSymbols)
    if Set.count reachableSymbols < 50 then
        printfn "[SOFT DELETE] Reachable symbols:"
        reachableSymbols |> Set.iter (fun s -> printfn "  - %s" s)
    
    // Find all nodes for reachable symbols
    let entryNodeIds = 
        psg.Nodes
        |> Map.toSeq
        |> Seq.choose (fun (nodeId, node) ->
            match node.Symbol with
            | Some symbol when 
                entryPoints |> List.exists (fun ep -> symbolMatches symbol ep.FullName) ||
                reachableSymbols |> Set.exists (fun rs -> symbolMatches symbol rs) ->
                Some nodeId
            | _ -> None)
        |> Set.ofSeq
    
    printfn "[SOFT DELETE] Found %d nodes to start traversal" (Set.count entryNodeIds)
    
    // Deep traversal function - selectively follow children
    let rec traverseNode (nodeId: string) (distance: int) (followChildren: bool) =
        if Set.contains nodeId processedNodeIds || distance > 50 then () // Prevent infinite recursion
        else
            processedNodeIds <- Set.add nodeId processedNodeIds
            reachableNodeIds <- Set.add nodeId reachableNodeIds
            
            match Map.tryFind nodeId psg.Nodes with
            | Some node ->
                // Only follow children if this is a reachable function
                let shouldFollowChildren = 
                    distance = 0 ||  // Entry points
                    (followChildren && distance < 10)  // Limit depth to prevent marking everything
                
                if shouldFollowChildren then
                    match node.Children with
                    | Parent children ->
                        for childId in children do
                            traverseNode childId.Value (distance + 1) shouldFollowChildren
                    | _ -> ()
                
                // Follow edges to mark dependencies as reachable
                let outgoingEdges = psg.Edges |> List.filter (fun e -> e.Source.Value = nodeId)
                for edge in outgoingEdges do
                    match edge.Kind with
                    | FunctionCall | SymRef | SymbolUse | TypeOf | TypeInstantiation _ | DataDependency ->
                        // These edges indicate dependencies that must be included for MLIR
                        traverseNode edge.Target.Value (distance + 1) false  // Don't follow children of dependencies
                    | _ -> ()
                
                // Follow incoming TypeOf edges to include type definitions
                let incomingTypeEdges = psg.Edges |> List.filter (fun e -> 
                    e.Target.Value = nodeId && e.Kind = TypeOf)
                for edge in incomingTypeEdges do
                    traverseNode edge.Source.Value (distance + 1) false
                
                // Don't follow symbol relationships blindly - this was causing everything to be marked
                ()
            | None -> ()
    
    // Start traversal from entry points
    for entryId in entryNodeIds do
        traverseNode entryId 0 true  // Follow children for entry points
    
    // Find nodes that are children of reachable nodes (for completeness)
    let additionalReachable = 
        reachableNodeIds
        |> Set.toSeq
        |> Seq.collect (fun nodeId ->
            match Map.tryFind nodeId psg.Nodes with
            | Some node ->
                match node.Children with
                | Parent children -> children |> List.map (fun c -> c.Value)
                | _ -> []
            | None -> [])
        |> Set.ofSeq
    
    reachableNodeIds <- Set.union reachableNodeIds additionalReachable
    
    // Mark literals and constants in reachable functions as reachable
    let literalNodes =
        psg.Nodes
        |> Map.toSeq
        |> Seq.filter (fun (nodeId, node) ->
            Set.contains nodeId reachableNodeIds ||
            (node.SyntaxKind.StartsWith("Const:") && 
             match node.ParentId with
             | Some pid -> Set.contains pid.Value reachableNodeIds
             | None -> false))
        |> Seq.map fst
        |> Set.ofSeq
    
    reachableNodeIds <- Set.union reachableNodeIds literalNodes
    
    printfn "[SOFT DELETE] Deep traversal found %d reachable nodes" (Set.count reachableNodeIds)
    
    // Update all nodes with reachability status
    let mutable reachableCount = 0
    let mutable unreachableCount = 0
    
    let updatedNodes = 
        psg.Nodes
        |> Map.map (fun nodeId node ->
            if Set.contains nodeId reachableNodeIds then
                reachableCount <- reachableCount + 1
                let distance = 
                    if Set.contains nodeId entryNodeIds then 0
                    else 1  // Simplified distance calculation
                ReachabilityHelpers.markReachable distance node
            else
                unreachableCount <- unreachableCount + 1
                ReachabilityHelpers.markUnreachable 1 "Not reachable from entry points" node)
    
    printfn "[SOFT DELETE] Marking complete: %d reachable, %d unreachable" reachableCount unreachableCount
    
    // Report on what types of nodes are reachable
    let reachableByKind = 
        updatedNodes
        |> Map.toSeq
        |> Seq.filter (fun (_, node) -> node.IsReachable)
        |> Seq.groupBy (fun (_, node) -> node.SyntaxKind.Split(':').[0])
        |> Seq.map (fun (kind, nodes) -> kind, Seq.length nodes)
        |> Seq.sortByDescending snd
        |> Seq.truncate 10  // Use truncate instead of take to handle fewer elements
        |> Seq.toList
    
    printfn "[SOFT DELETE] Top reachable node types:"
    reachableByKind |> List.iter (fun (kind, count) ->
        printfn "  %s: %d nodes" kind count)
    
    { psg with Nodes = updatedNodes }

/// Main entry point for enhanced PSG-based reachability analysis
let analyzeReachabilityWithBoundaries (psg: ProgramSemanticGraph) : LibraryAwareReachability =
    let startTime = DateTime.UtcNow
    
    printfn "[REACHABILITY] === Enhanced PSG-Based Analysis Start ==="
    printfn "[REACHABILITY] PSG nodes: %d" (Map.count psg.Nodes)
    printfn "[REACHABILITY] PSG edges: %d" psg.Edges.Length
    printfn "[REACHABILITY] PSG entry points: %d" psg.EntryPoints.Length
    
    // Use enhanced symbol extraction
    let meaningfulSymbols = UnifiedSymbolExtraction.extractMeaningfulSymbols psg
    let correlatedNodes = UnifiedSymbolExtraction.extractCorrelatedNodes psg
    let meaningfulSymbolNames = meaningfulSymbols |> Array.map (fun s -> s.FullName) |> Set.ofArray
    
    printfn "[REACHABILITY] Total meaningful symbols (enhanced): %d" meaningfulSymbols.Length
    printfn "[REACHABILITY] Total correlated nodes: %d" correlatedNodes.Length
    
    // Extract entry points using enhanced detection
    let entryPoints = extractEntryPoints psg
    
    // Extract function calls using enhanced analysis
    let functionCalls = extractFunctionCalls psg
    
    // Build comprehensive call graph
    let callGraph = buildCallGraph functionCalls entryPoints
    
    // Compute reachable symbols from entry points
    let reachableSymbols = computeReachableSymbols entryPoints callGraph meaningfulSymbolNames
    let unreachableSymbols = Set.difference meaningfulSymbolNames reachableSymbols
    
    // Use DEEP traversal for comprehensive reachability marking
    let markedPSG = markReachabilityInPSGDeep psg entryPoints
    
    // Generate library classifications
    let libraryCategories = 
        meaningfulSymbols
        |> Array.map (fun symbol -> symbol.FullName, classifySymbol symbol)
        |> Map.ofArray
    
    // Calculate statistics
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
    
    printfn "[REACHABILITY] === Enhanced Results ==="
    printfn "[REACHABILITY] Reachable symbols: %d" reachableCount
    printfn "[REACHABILITY] Unreachable symbols: %d" unreachableCount
    printfn "[REACHABILITY] Analysis completed in %dms" computationTime
    
    let eliminationRate = 
        if totalCount > 0 then
            (float unreachableCount / float totalCount) * 100.0
        else 0.0
    
    printfn "[REACHABILITY] Elimination rate: %.2f%% (%d eliminated out of %d total)" eliminationRate unreachableCount totalCount
    
    // Validate PSG node marking
    let actualReachableNodes = markedPSG.Nodes |> Map.filter (fun _ node -> node.IsReachable) |> Map.count
    printfn "[REACHABILITY] PSG node marking validation: %d nodes marked as reachable" actualReachableNodes
    
    if actualReachableNodes = 0 then
        printfn "[REACHABILITY] CRITICAL: No PSG nodes marked as reachable despite %d reachable symbols!" reachableCount
    
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
        MarkedPSG = markedPSG
    }