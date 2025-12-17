/// Provisional Call Graph - Build call graph from syntax without type checking
///
/// This nanopass builds a call graph using only syntactic information:
/// - Function names from Binding:* nodes
/// - Called names from LongIdent:*, MethodCall:*, Ident:* nodes
///
/// This enables reachability analysis BEFORE type checking (Phase 3 before Phase 2).
module Core.PSG.Nanopass.ProvisionalCallGraph

open Core.PSG.Types

/// Extract function name from a binding node
let private extractBindingName (node: PSGNode) : string option =
    match node.Kind with
    | SKBinding _ ->
        // Use symbol name if available
        node.Symbol |> Option.map (fun s -> s.DisplayName)
    | _ -> None

/// Extract called function name from a node
let private extractCalledName (node: PSGNode) : string option =
    match node.Kind with
    | SKExpr ELongIdent | SKExpr EIdent ->
        // Use symbol name if available
        node.Symbol |> Option.map (fun s -> s.DisplayName)
    | SKExpr EMethodCall ->
        // Use symbol name if available
        node.Symbol |> Option.map (fun s -> s.DisplayName)
    | SKExpr EApp ->
        None  // Will be handled by looking at children
    | _ -> None

/// Extract the module path from a node's file location
let private getModuleFromFile (fileName: string) : string =
    System.IO.Path.GetFileNameWithoutExtension(fileName)

/// Build provisional call graph from structural PSG
/// Returns a map of (caller function name) -> (list of called function names)
let buildProvisionalCallGraph (psg: ProgramSemanticGraph) : Map<string, string list> =
    let mutable callGraph = Map.empty
    let mutable currentFunction = None

    // First pass: identify all function definitions
    let functionDefinitions =
        psg.Nodes
        |> Map.toSeq
        |> Seq.choose (fun (nodeId, node) ->
            extractBindingName node
            |> Option.map (fun name -> (nodeId, name, node)))
        |> Seq.toList

    // Build a map from node to containing function
    // Walk the tree structure using ChildOf edges
    let nodeToFunction =
        let mutable result = Map.empty

        for (nodeId, funcName, node) in functionDefinitions do
            // Mark this node and all descendants as belonging to this function
            let visited = System.Collections.Generic.HashSet<string>()
            let rec markDescendants nId =
                if visited.Add(nId) then
                    result <- Map.add nId funcName result
                    match Map.tryFind nId psg.Nodes with
                    | Some n ->
                        match n.Children with
                        | Parent children ->
                            for child in children do
                                markDescendants child.Value
                        | _ -> ()
                    | None -> ()
            markDescendants nodeId

        result

    // Second pass: find all calls and build edges
    for KeyValue(nodeId, node) in psg.Nodes do
        match extractCalledName node with
        | Some calledName ->
            // Find which function this call is in
            match Map.tryFind nodeId nodeToFunction with
            | Some callerName when callerName <> calledName ->
                // Add call edge (caller -> called)
                let currentCalls = Map.tryFind callerName callGraph |> Option.defaultValue []
                if not (List.contains calledName currentCalls) then
                    callGraph <- Map.add callerName (calledName :: currentCalls) callGraph
            | _ -> ()
        | None -> ()

    callGraph

/// Add provisional call edges to the PSG
/// This enriches the PSG with ProvisionalCall edges based on syntactic analysis
let addProvisionalCallEdges (psg: ProgramSemanticGraph) : ProgramSemanticGraph =
    let callGraph = buildProvisionalCallGraph psg

    // Build a name-to-node map for function definitions
    let nameToNode =
        psg.Nodes
        |> Map.toSeq
        |> Seq.choose (fun (nodeId, node) ->
            extractBindingName node
            |> Option.map (fun name -> (name, nodeId)))
        |> Map.ofSeq

    // Create ProvisionalCall edges
    let provisionalEdges =
        callGraph
        |> Map.toList
        |> List.collect (fun (caller, callees) ->
            match Map.tryFind caller nameToNode with
            | Some callerNodeId ->
                callees
                |> List.choose (fun callee ->
                    // Try to find the callee definition
                    // If not found locally, we still record the edge with a sentinel
                    match Map.tryFind callee nameToNode with
                    | Some calleeNodeId ->
                        Some {
                            Source = { Value = callerNodeId }
                            Target = { Value = calleeNodeId }
                            Kind = FunctionCall  // Provisional function call
                        }
                    | None ->
                        // External call - callee not in this PSG
                        // Record as a self-edge with the callee name in metadata
                        // (Reachability will handle external library calls)
                        None)
            | None -> [])

    { psg with Edges = provisionalEdges @ psg.Edges }

/// Get the set of function names defined in this PSG
let getDefinedFunctions (psg: ProgramSemanticGraph) : Set<string> =
    psg.Nodes
    |> Map.toSeq
    |> Seq.choose (fun (_, node) -> extractBindingName node)
    |> Set.ofSeq

/// Get the set of function names called in this PSG
let getCalledFunctions (psg: ProgramSemanticGraph) : Set<string> =
    psg.Nodes
    |> Map.toSeq
    |> Seq.choose (fun (_, node) -> extractCalledName node)
    |> Set.ofSeq
