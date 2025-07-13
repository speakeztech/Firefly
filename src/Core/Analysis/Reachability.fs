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
let rec findEnclosingFunction (psg: ProgramSemanticGraph) (nodeId: string) (visited: Set<string>) : string option =
    if Set.contains nodeId visited then
        printfn "[ERROR] Circular parent structure detected at %s" nodeId
        printfn "[ERROR] Cycle path: %A" (Set.toList visited)
        failwith "Circular parent-child relationships in PSG - invalid containment structure"
    
    if Set.count visited > 15 then 
        printfn "[ERROR] Excessive depth reached, likely circular structure"
        failwith "PSG containment depth exceeded - possible circular relationships"
    
    let newVisited = Set.add nodeId visited
    
    match Map.tryFind nodeId psg.Nodes with
    | Some node when node.Symbol.IsSome ->
        let symbol = node.Symbol.Value
        if isFunction symbol then
            Some symbol.FullName
        else
            psg.Edges
            |> List.tryPick (fun edge ->
                if edge.Target.Value = nodeId && edge.Kind = ChildOf then
                    findEnclosingFunction psg edge.Source.Value newVisited
                else None
            )
    | Some node ->
        psg.Edges
        |> List.tryPick (fun edge ->
            if edge.Target.Value = nodeId && edge.Kind = ChildOf then
                findEnclosingFunction psg edge.Source.Value newVisited
            else None
        )
    | None ->
        None

/// Public entry point for enclosing function search
let findEnclosingFunctionSafe (psg: ProgramSemanticGraph) (nodeId: string) : string option =
    findEnclosingFunction psg nodeId Set.empty
            
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
let extractFunctionCalls (psg: ProgramSemanticGraph) =
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
                    let callingContext = findEnclosingFunctionSafe psg edge.Source.Value
                    match callingContext with
                    | Some caller -> Some (caller, tgt.Symbol.Value.FullName)
                    | None -> 
                        printfn "[ERROR] Cannot determine calling context for %s" tgt.Symbol.Value.FullName
                        failwith "Enclosing function detection failed - PSG parent structure broken"
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


/// Extract entry points from PSG with enhanced symbol resolution
let extractEntryPoints (psg: ProgramSemanticGraph) =
    printfn "[ENTRY POINTS] === Entry Point Extraction ==="
    
    // Search all PSG nodes for main functions and EntryPoint attributes
    let foundEntryPoints = 
        psg.Nodes
        |> Map.toSeq
        |> Seq.choose (fun (_, node) ->
            match node.Symbol with
            | Some symbol when symbol.DisplayName = "main" ->
                printfn "[ENTRY POINTS] Found main function: %s" symbol.FullName
                Some symbol
            | Some symbol ->
                match symbol with
                | :? FSharpMemberOrFunctionOrValue as mfv ->
                    let hasEntryPoint = 
                        mfv.Attributes |> Seq.exists (fun attr ->
                            let name = attr.AttributeType.DisplayName
                            name = "EntryPointAttribute" || 
                            name = "EntryPoint" ||
                            attr.AttributeType.FullName.EndsWith("EntryPointAttribute"))
                    if hasEntryPoint then
                        printfn "[ENTRY POINTS] Found EntryPoint function: %s" symbol.FullName
                        Some symbol
                    else None
                | _ -> None
            | None -> None
        )
        |> Seq.toList
    
    // If no symbols found, try PSG entry points list
    let psgEntryPoints = 
        if foundEntryPoints.IsEmpty then
            psg.EntryPoints
            |> List.choose (fun entryPointId ->
                match entryPointId with
                | SymbolNode(_, symbolName) ->
                    Map.tryFind symbolName psg.SymbolTable
                | _ -> None
            )
        else []
    
    let allEntryPoints = foundEntryPoints @ psgEntryPoints |> List.distinct
    
    printfn "[ENTRY POINTS] Total entry points: %d" allEntryPoints.Length
    allEntryPoints |> List.iter (fun ep ->
        printfn "[ENTRY POINTS]   %s" ep.FullName)
    
    allEntryPoints

// REPLACE buildEnhancedCallGraph in src/Core/Analysis/Reachability.fs  
let buildCallGraph (entryPoints: FSharpSymbol list) (relationships: (string * string) list) =
    printfn "[CALL GRAPH] === Call Graph Construction ==="
    
    let callGraph = 
        relationships
        |> List.groupBy fst
        |> List.map (fun (caller, calls) ->
            caller, calls |> List.map snd |> List.distinct)
        |> Map.ofList
    
    // Ensure all entry points are in the graph
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

// REPLACE computeEnhancedReachableSymbols in src/Core/Analysis/Reachability.fs
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
    printfn "[REACHABILITY] Starting from entry points: %A" entryPointNames
    
    // Include entry points even if not in symbol set
    let symbolsWithEntryPoints = 
        entryPointNames 
        |> List.fold (fun acc ep -> Set.add ep acc) allSymbols
    
    let reachable = traverse Set.empty entryPointNames
    
    printfn "[REACHABILITY] Found %d reachable symbols" (Set.count reachable)
    if Set.count reachable > 0 then
        printfn "[REACHABILITY] Sample reachable:"
        reachable |> Set.toArray |> Array.take (min 5 (Set.count reachable))
        |> Array.iter (fun s -> printfn "  ✓ %s" s)
    
    reachable

// REPLACE analyzeReachabilityWithBoundaries in src/Core/Analysis/Reachability.fs
let analyzeReachabilityWithBoundaries (psg: ProgramSemanticGraph) : LibraryAwareReachability =
    let startTime = DateTime.UtcNow
    
    printfn "[REACHABILITY] === PSG-Based Analysis Start ==="
    printfn "[REACHABILITY] PSG nodes: %d" (Map.count psg.Nodes)
    printfn "[REACHABILITY] PSG edges: %d" psg.Edges.Length
    printfn "[REACHABILITY] PSG entry points: %d" psg.EntryPoints.Length
    
    // Extract meaningful symbols
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
    
    // Extract entry points using fixed function
    let entryPoints = extractEntryPoints psg
    
    // Extract function calls
    let functionCalls = extractFunctionCalls psg
    
    // Build call graph
    let callGraph = buildCallGraph entryPoints functionCalls
    
    // Compute reachable symbols
    let reachableSymbols = computeReachableSymbols entryPoints callGraph allSymbolNames
    let unreachableSymbols = Set.difference allSymbolNames reachableSymbols
    
    printfn "[REACHABILITY] === Results ==="
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
    
    printfn "[REACHABILITY] Analysis completed in %dms" computationTime
    printfn "[REACHABILITY] Elimination rate: %.2f%%" eliminationRate
    
    {
        BasicResult = basicResult
        LibraryCategories = libraryCategories
        PruningStatistics = pruningStats
    }

