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
        // Only top-level function bindings start new "containing function" scopes
        // Local let bindings (for variables) should NOT start new scopes
        for KeyValue(nodeId, node) in psg.Nodes do
            match node.SyntaxKind with
            | sk when sk.StartsWith("Binding") || sk.StartsWith("Member") ->
                match node.Symbol with
                | Some sym ->
                    // Only mark as a function boundary if this is a module-level function
                    // (not a local let binding for a variable)
                    let mfvOpt =
                        match sym with
                        | :? FSharpMemberOrFunctionOrValue as mfv -> Some mfv
                        | _ -> None

                    let isTopLevelFunction =
                        match mfvOpt with
                        | Some mfv -> mfv.IsModuleValueOrMember && mfv.IsFunction
                        | None -> false

                    if isTopLevelFunction then
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

    // Helper: Check if a symbol represents a callable function/method (not a parameter or local)
    // Key insight: Parameters and local bindings are NOT module-level values/members
    let isCallableSymbol (sym: FSharpSymbol) : bool =
        match sym with
        | :? FSharpMemberOrFunctionOrValue as mfv ->
            // Must be a module-level value/member (not a parameter or local binding)
            // AND must be a function or member (not just a value constant)
            mfv.IsModuleValueOrMember && (mfv.IsFunction || mfv.IsMember)
        | _ -> false

    // Helper: Try to add a call edge to the graph
    let tryAddCallEdge caller targetSym =
        if isCallableSymbol targetSym then
            let calledFunc = targetSym.FullName
            if caller <> calledFunc then
                let currentCalls = Map.tryFind caller callGraph |> Option.defaultValue []
                if not (List.contains calledFunc currentCalls) then
                    callGraph <- Map.add caller (calledFunc :: currentCalls) callGraph

    // Analyze edges to build call graph
    for edge in psg.Edges do
        match edge.Kind with
        | FunctionCall | SymRef ->
            let callerOpt = Map.tryFind edge.Source.Value nodeToFunction
            let targetOpt = Map.tryFind edge.Target.Value psg.Nodes
            match callerOpt, targetOpt with
            | Some caller, Some targetNode ->
                match targetNode.Symbol with
                | Some targetSym -> tryAddCallEdge caller targetSym
                | None -> ()
            | _ -> ()
        | _ -> ()
    
    // Also analyze application nodes for function calls not captured by edges
    // KEY FIX: Only consider actual function/method symbols, not parameters or local bindings
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
                            | Some sym -> tryAddCallEdge caller sym
                            | None -> ()
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

    { psg with Nodes = finalNodes }

/// Perform semantic reachability analysis
let analyzeReachability (psg: ProgramSemanticGraph) : LibraryAwareReachability =
    let startTime = DateTime.UtcNow

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

    // Build semantic call graph
    let callGraph = buildSemanticCallGraph psg

    // Compute semantic reachability
    let reachableSymbols = computeSemanticReachability allEntryPoints callGraph
    
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

    {
        BasicResult = result
        LibraryCategories = libraryCategories
        PruningStatistics = stats
        MarkedPSG = markedPSG
    }

/// Entry point for reachability analysis (requires typed PSG with symbols)
let performReachabilityAnalysis (psg: ProgramSemanticGraph) : LibraryAwareReachability =
    analyzeReachability psg

// ============================================================================
// STRUCTURAL REACHABILITY (Phase 1 - works without type checking)
// ============================================================================

/// Structural reachability result (no symbol information)
type StructuralReachabilityResult = {
    EntryPointNodes: Set<string>
    ReachableFunctions: Set<string>
    ReachableNodes: Set<string>
    ProvisionalCallGraph: Map<string, string list>
}

/// Find entry points from structural PSG (no symbols needed)
let private findStructuralEntryPoints (psg: ProgramSemanticGraph) : Set<string> * Set<string> =
    // Returns (entry point node IDs, entry point function names)
    let mutable nodeIds = Set.empty
    let mutable funcNames = Set.empty

    for KeyValue(nodeId, node) in psg.Nodes do
        match node.SyntaxKind with
        | "Binding:EntryPoint" ->
            nodeIds <- Set.add nodeId nodeIds
            // Extract function name from binding (next part of syntax kind would have it)
            // For now use node ID as the name key
            funcNames <- Set.add "main" funcNames  // EntryPoint typically means main
        | "Binding:Main" ->
            nodeIds <- Set.add nodeId nodeIds
            funcNames <- Set.add "main" funcNames
        | sk when sk.StartsWith("Binding:") && sk.EndsWith(":main") ->
            nodeIds <- Set.add nodeId nodeIds
            funcNames <- Set.add "main" funcNames
        | _ -> ()

    // Also look for binding nodes that contain "main" in their syntax kind
    if Set.isEmpty funcNames then
        for KeyValue(nodeId, node) in psg.Nodes do
            if node.SyntaxKind.Contains(":main") then
                nodeIds <- Set.add nodeId nodeIds
                funcNames <- Set.add "main" funcNames

    nodeIds, funcNames

/// Compute reachable functions from structural call graph
let private computeStructuralReachability
    (entryFunctions: Set<string>)
    (callGraph: Map<string, string list>) : Set<string> =

    let mutable reachable = entryFunctions
    let mutable toProcess = Set.toList entryFunctions

    while not (List.isEmpty toProcess) do
        match toProcess with
        | current :: rest ->
            toProcess <- rest
            match Map.tryFind current callGraph with
            | Some callees ->
                for callee in callees do
                    if not (Set.contains callee reachable) then
                        reachable <- Set.add callee reachable
                        toProcess <- callee :: toProcess
            | None -> ()
        | [] -> ()

    reachable

/// Mark all nodes within reachable functions as reachable
let private markStructuralReachability
    (psg: ProgramSemanticGraph)
    (reachableFunctions: Set<string>)
    (entryPointNodes: Set<string>) : ProgramSemanticGraph =

    // Build a map of function names to their binding node IDs
    let functionToNodes =
        psg.Nodes
        |> Map.toSeq
        |> Seq.choose (fun (nodeId, node) ->
            // Extract function name from syntax kind like "Binding:functionName"
            if node.SyntaxKind.StartsWith("Binding:") then
                let funcName =
                    let suffix = node.SyntaxKind.Substring(8)
                    // Handle special cases
                    if suffix = "EntryPoint" || suffix = "Main" then "main"
                    elif suffix.StartsWith("Mutable:") then suffix.Substring(8)
                    elif suffix = "Use" then "" // Skip use bindings for now
                    else suffix
                if funcName <> "" then Some (funcName, nodeId)
                else None
            else None)
        |> Seq.groupBy fst
        |> Seq.map (fun (name, pairs) -> name, pairs |> Seq.map snd |> Set.ofSeq)
        |> Map.ofSeq

    // Find all node IDs for reachable functions
    let reachableBindingNodes =
        reachableFunctions
        |> Set.toSeq
        |> Seq.collect (fun funcName ->
            match Map.tryFind funcName functionToNodes with
            | Some nodeIds -> Set.toSeq nodeIds
            | None -> Seq.empty)
        |> Set.ofSeq
        |> Set.union entryPointNodes

    // Mark all descendants of reachable bindings
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

    for bindingId in reachableBindingNodes do
        markDescendants bindingId

    // Also mark parent chain for structural completeness
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

    for bindingId in reachableBindingNodes do
        markParentChain bindingId

    // Apply reachability marks to all nodes
    let finalNodes =
        psg.Nodes
        |> Map.map (fun nodeId node ->
            let isReachable = Set.contains nodeId reachableNodeIds
            { node with
                IsReachable = isReachable
                EliminationReason = if isReachable then None else Some "Not structurally reachable"
                EliminationPass = if isReachable then None else Some 0 }
        )

    { psg with Nodes = finalNodes }

/// Perform structural reachability analysis (Phase 1 - before type checking)
/// Uses provisional call graph built from syntax, not symbols
let performStructuralReachabilityAnalysis (psg: ProgramSemanticGraph) : StructuralReachabilityResult * ProgramSemanticGraph =
    // Import provisional call graph builder
    let callGraph = Core.PSG.Nanopass.ProvisionalCallGraph.buildProvisionalCallGraph psg

    // Find entry points from syntax
    let entryPointNodes, entryFunctions = findStructuralEntryPoints psg

    // Compute reachable functions
    let reachableFunctions = computeStructuralReachability entryFunctions callGraph

    // Mark nodes
    let markedPSG = markStructuralReachability psg reachableFunctions entryPointNodes

    // Collect reachable node IDs
    let reachableNodes =
        markedPSG.Nodes
        |> Map.toSeq
        |> Seq.filter (fun (_, node) -> node.IsReachable)
        |> Seq.map fst
        |> Set.ofSeq

    let result = {
        EntryPointNodes = entryPointNodes
        ReachableFunctions = reachableFunctions
        ReachableNodes = reachableNodes
        ProvisionalCallGraph = callGraph
    }

    result, markedPSG

/// Get the set of files that contain reachable code
let getReachableFiles (psg: ProgramSemanticGraph) : Set<string> =
    psg.Nodes
    |> Map.toSeq
    |> Seq.filter (fun (_, node) -> node.IsReachable)
    |> Seq.map (fun (_, node) -> node.SourceFile)
    |> Set.ofSeq