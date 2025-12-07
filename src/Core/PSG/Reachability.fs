module Core.PSG.Reachability

open System
open FSharp.Compiler.Symbols
open Core.PSG.Types

/// Semantic reachability context based on type-aware analysis
type SemanticReachabilityContext = {
    /// Functions that are actually called
    ReachableFunctions: Set<string>
    /// Type instantiations that are used (e.g., Tree<int> not Tree<'T>)
    TypeInstantiations: Map<string, Set<string>>
    /// Union cases that are constructed or matched
    UsedUnionCases: Map<string, Set<string>>
    /// Interface methods that are called
    CalledInterfaceMethods: Map<string, Set<string>>
    /// Entry point symbols
    EntryPoints: FSharpSymbol list
}

/// Result of semantic reachability analysis
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

/// Enhanced reachability result with library boundary information
type LibraryAwareReachability = {
    BasicResult: ReachabilityResult
    LibraryCategories: Map<string, LibraryCategory>
    PruningStatistics: PruningStatistics
    MarkedPSG: ProgramSemanticGraph  // PSG with nodes marked for reachability
}

and PruningStatistics = {
    TotalSymbols: int
    ReachableSymbols: int
    EliminatedSymbols: int
    ComputationTimeMs: int64
}

/// Classify library category based on symbol name
let private classifyLibraryCategory (symbolName: string) : LibraryCategory =
    if symbolName.StartsWith("Examples.") || symbolName.StartsWith("HelloWorld") then
        UserCode
    elif symbolName.StartsWith("Alloy.") then
        AlloyLibrary
    elif symbolName.StartsWith("Microsoft.FSharp.") || symbolName.StartsWith("FSharp.Core") then
        FSharpCore
    elif symbolName.StartsWith("System.") then
        Other "System"
    else
        UserCode

/// Build semantic call graph from PSG using principled graph analysis
let private buildSemanticCallGraph (psg: ProgramSemanticGraph) : Map<string, string list> =
    let mutable callGraph = Map.empty
    
    // Build a map from nodes to their containing function/method
    let nodeToFunction = 
        let mutable result = Map.empty
        
        // Walk the graph to find all function/method definitions
        for KeyValue(nodeId, node) in psg.Nodes do
            match node.SyntaxKind with
            | sk when sk.StartsWith("Binding") || sk.StartsWith("Member") ->
                match node.Symbol with
                | Some sym ->
                    // This node defines a function - mark all its descendants
                    // Use visited set to prevent infinite recursion on cyclic references
                    let visited = System.Collections.Generic.HashSet<string>()
                    let rec markDescendants nId funcName =
                        if visited.Add(nId) then  // Only process if not already visited
                            result <- Map.add nId funcName result
                            match Map.tryFind nId psg.Nodes with
                            | Some n ->
                                match n.Children with
                                | Parent children ->
                                    for child in children do
                                        markDescendants child.Value funcName
                                | _ -> ()
                            | None -> ()

                    markDescendants nodeId sym.FullName
                | None -> ()
            | _ -> ()
        
        result
    
    // Analyze edges to build call graph
    for edge in psg.Edges do
        match edge.Kind with
        | FunctionCall ->
            // Find the calling function and called function
            let callerOpt = Map.tryFind edge.Source.Value nodeToFunction
            let targetOpt = Map.tryFind edge.Target.Value psg.Nodes
            
            match callerOpt, targetOpt with
            | Some caller, Some targetNode ->
                match targetNode.Symbol with
                | Some targetSym ->
                    let calledFunc = targetSym.FullName
                    // Only add if it's a different function and not already in the list
                    if caller <> calledFunc then
                        let currentCalls = Map.tryFind caller callGraph |> Option.defaultValue []
                        if not (List.contains calledFunc currentCalls) then
                            callGraph <- Map.add caller (calledFunc :: currentCalls) callGraph
                | None -> ()
            | _ -> ()
        | MethodCall ->
            // Find the calling function and called function
            let callerOpt = Map.tryFind edge.Source.Value nodeToFunction
            let targetOpt = Map.tryFind edge.Target.Value psg.Nodes
            
            match callerOpt, targetOpt with
            | Some caller, Some targetNode ->
                match targetNode.Symbol with
                | Some targetSym ->
                    let calledFunc = targetSym.FullName
                    // Only add if it's a different function and not already in the list
                    if caller <> calledFunc then
                        let currentCalls = Map.tryFind caller callGraph |> Option.defaultValue []
                        if not (List.contains calledFunc currentCalls) then
                            callGraph <- Map.add caller (calledFunc :: currentCalls) callGraph
                | None -> ()
            | _ -> ()
        | Reference ->
            // Find the calling function and called function
            let callerOpt = Map.tryFind edge.Source.Value nodeToFunction
            let targetOpt = Map.tryFind edge.Target.Value psg.Nodes
            
            match callerOpt, targetOpt with
            | Some caller, Some targetNode ->
                match targetNode.Symbol with
                | Some targetSym ->
                    let calledFunc = targetSym.FullName
                    // Only add if it's a different function and not already in the list
                    if caller <> calledFunc then
                        let currentCalls = Map.tryFind caller callGraph |> Option.defaultValue []
                        if not (List.contains calledFunc currentCalls) then
                            callGraph <- Map.add caller (calledFunc :: currentCalls) callGraph
                | None -> ()
            | _ -> ()
        | _ -> ()
    
    // Also analyze application nodes for function calls not captured by edges
    for KeyValue(nodeId, node) in psg.Nodes do
        match node.SyntaxKind with
        | "App" | "App:FunctionCall" | "TypeApp" ->
            let callerOpt = Map.tryFind nodeId nodeToFunction
            
            match callerOpt with
            | Some caller ->
                // Look at children to find what's being called
                match node.Children with
                | Parent children ->
                    for childId in children do
                        match Map.tryFind childId.Value psg.Nodes with
                        | Some childNode ->
                            match childNode.Symbol with
                            | Some childSym when childSym.FullName <> caller ->
                                let currentCalls = Map.tryFind caller callGraph |> Option.defaultValue []
                                if not (List.contains childSym.FullName currentCalls) then
                                    callGraph <- Map.add caller (childSym.FullName :: currentCalls) callGraph
                            | _ -> ()
                        | None -> ()
                | _ -> ()
            | None -> ()
        | _ -> ()
    
    callGraph

/// Compute reachable symbols using semantic analysis
let private computeSemanticReachability 
    (entryPoints: FSharpSymbol list) 
    (callGraph: Map<string, string list>) : Set<string> =
    
    let mutable reachable = Set.empty
    let mutable toProcess = []
    
    // Start with entry points
    for ep in entryPoints do
        reachable <- Set.add ep.FullName reachable
        toProcess <- ep.FullName :: toProcess
    
    // Process call graph
    while not (List.isEmpty toProcess) do
        match toProcess with
        | current :: rest ->
            toProcess <- rest
            
            // Add all functions called by current
            match Map.tryFind current callGraph with
            | Some callees ->
                for callee in callees do
                    if not (Set.contains callee reachable) then
                        reachable <- Set.add callee reachable
                        toProcess <- callee :: toProcess
            | None -> ()
        | [] -> ()
    
    reachable

/// Mark PSG nodes based on semantic reachability
///
/// KEY PRINCIPLE: Reachability has two levels:
/// 1. Symbol-level: Which functions/bindings are called from entry points
/// 2. Node-level: ALL nodes within a reachable function body are reachable
///
/// The previous implementation only did symbol-level marking, leaving function
/// body nodes (Sequential, Const, Tuple, etc.) marked as unreachable even when
/// their containing function was reachable. This broke emission.
let private markNodesForSemanticReachability
    (psg: ProgramSemanticGraph)
    (reachableSymbols: Set<string>) : ProgramSemanticGraph =

    printfn "[SEMANTIC] Marking nodes for %d reachable symbols" (Set.count reachableSymbols)

    // Step 1: Find all binding nodes for reachable symbols
    let reachableBindingNodeIds =
        psg.Nodes
        |> Map.toSeq
        |> Seq.choose (fun (nodeId, node) ->
            match node.Symbol with
            | Some symbol when Set.contains symbol.FullName reachableSymbols ->
                match node.SyntaxKind with
                | sk when sk.StartsWith("Binding") || sk.StartsWith("Member") ->
                    Some nodeId
                | _ -> None
            | _ -> None)
        |> Set.ofSeq

    // Step 2: For each reachable binding, mark ALL descendant nodes as reachable
    // This ensures function bodies are fully emitted
    let mutable reachableNodeIds = Set.empty

    let rec markDescendants nodeId =
        if not (Set.contains nodeId reachableNodeIds) then
            reachableNodeIds <- Set.add nodeId reachableNodeIds
            match Map.tryFind nodeId psg.Nodes with
            | Some node ->
                match node.Children with
                | Parent children ->
                    for childId in children do
                        markDescendants childId.Value
                | _ -> ()
            | None -> ()

    // Mark all descendants of reachable bindings
    for bindingId in reachableBindingNodeIds do
        markDescendants bindingId

    // Step 3: Also mark parent chain up to module level for structural completeness
    let markParentChain nodeId =
        let mutable currentId = Some nodeId
        while currentId.IsSome do
            let nId = currentId.Value
            reachableNodeIds <- Set.add nId reachableNodeIds
            match Map.tryFind nId psg.Nodes with
            | Some node ->
                currentId <- node.ParentId |> Option.map (fun p -> p.Value)
            | None ->
                currentId <- None

    for bindingId in reachableBindingNodeIds do
        markParentChain bindingId

    // Step 4: Mark nodes that call reachable functions (App nodes with reachable targets)
    for KeyValue(nodeId, node) in psg.Nodes do
        match node.SyntaxKind with
        | sk when sk.StartsWith("App") ->
            match node.Symbol with
            | Some symbol when Set.contains symbol.FullName reachableSymbols ->
                markDescendants nodeId
            | _ -> ()
        | _ -> ()

    // Step 5: Apply reachability to all nodes
    let finalNodes =
        psg.Nodes
        |> Map.map (fun nodeId node ->
            let isReachable = Set.contains nodeId reachableNodeIds
            { node with
                IsReachable = isReachable
                EliminationReason = if isReachable then None else Some "Not semantically reachable"
                EliminationPass = if isReachable then None else Some 1 }
        )

    let reachableNodeCount = Set.count reachableNodeIds

    printfn "[SEMANTIC] Marked %d nodes as reachable (%.1f%% of %d total)"
        reachableNodeCount
        (float reachableNodeCount / float (Map.count finalNodes) * 100.0)
        (Map.count finalNodes)

    { psg with Nodes = finalNodes }

/// Perform semantic reachability analysis
let analyzeReachability (psg: ProgramSemanticGraph) : LibraryAwareReachability =
    let startTime = DateTime.UtcNow
    printfn "[REACHABILITY] Starting semantic reachability analysis"
    
    // Find entry points
    let entryPoints = 
        psg.SymbolTable
        |> Map.toSeq
        |> Seq.choose (fun (name, symbol) ->
            match symbol with
            | :? FSharpMemberOrFunctionOrValue as mfv when 
                name = "main" || 
                mfv.Attributes |> Seq.exists (fun a -> a.AttributeType.DisplayName = "EntryPoint") ->
                Some symbol
            | _ -> None)
        |> Seq.toList
    
    // Also check for EntryPointAttribute itself
    let entryPointAttribute = 
        psg.SymbolTable
        |> Map.tryFind "EntryPointAttribute"
        |> Option.toList
    
    let allEntryPoints = entryPoints @ entryPointAttribute
    
    printfn "[REACHABILITY] Found %d entry points: %A" 
        allEntryPoints.Length 
        (allEntryPoints |> List.map (fun ep -> ep.DisplayName))
    
    // Build semantic call graph
    let callGraph = buildSemanticCallGraph psg
    printfn "[REACHABILITY] Built call graph with %d entries" (Map.count callGraph)
    
    // Compute semantic reachability
    let reachableSymbols = computeSemanticReachability allEntryPoints callGraph
    printfn "[REACHABILITY] Found %d semantically reachable symbols" (Set.count reachableSymbols)
    
    // All symbols in the PSG
    let allSymbols = 
        psg.SymbolTable
        |> Map.toSeq
        |> Seq.map (fun (_, sym) -> sym.FullName)
        |> Set.ofSeq
    
    let unreachableSymbols = Set.difference allSymbols reachableSymbols
    
    // Mark nodes based on semantic reachability
    let markedPSG = markNodesForSemanticReachability psg reachableSymbols
    
    // Calculate statistics
    let endTime = DateTime.UtcNow
    let computationTime = int64 (endTime - startTime).TotalMilliseconds
    
    let stats = {
        TotalSymbols = Set.count allSymbols
        ReachableSymbols = Set.count reachableSymbols
        EliminatedSymbols = Set.count unreachableSymbols
        ComputationTimeMs = computationTime
    }
    
    // Classify libraries
    let libraryCategories =
        allSymbols
        |> Set.toSeq
        |> Seq.map (fun sym -> sym, classifyLibraryCategory sym)
        |> Map.ofSeq
    
    let result = {
        EntryPoints = allEntryPoints
        ReachableSymbols = reachableSymbols
        UnreachableSymbols = unreachableSymbols
        CallGraph = callGraph
    }
    
    printfn "[REACHABILITY] Semantic analysis complete: %d/%d symbols reachable (%.1f%% eliminated)"
        stats.ReachableSymbols
        stats.TotalSymbols
        (float stats.EliminatedSymbols / float stats.TotalSymbols * 100.0)
    
    {
        BasicResult = result
        LibraryCategories = libraryCategories
        PruningStatistics = stats
        MarkedPSG = markedPSG
    }

/// Entry point for reachability analysis
let performReachabilityAnalysis (psg: ProgramSemanticGraph) : LibraryAwareReachability =
    analyzeReachability psg