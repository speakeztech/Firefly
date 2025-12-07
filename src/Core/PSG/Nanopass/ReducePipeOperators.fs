/// ReducePipeOperators - Nanopass to reduce F# pipe operators to direct application
///
/// The pipe operators are defined in the F# spec as library functions:
///   let inline (|>) x f = f x
///   let inline (<|) f x = f x
///
/// This nanopass performs beta reduction on pipe applications, transforming:
///   App [|>]              →  App [f]
///     LongIdent:(|>)           <function>
///     <value>                  <value>
///     <function>
///
/// This is a pure structural transformation requiring no type information.
/// Arithmetic/comparison operators are NOT reduced here - they are left for
/// Alex/XParsec to handle with full type context, preserving MLIR optimization
/// opportunities.
///
/// Run AFTER FlattenApplications, BEFORE DefUseEdges.
///
/// Reference: F# Language Spec §1734-1742 (operator syntax translation)
/// Reference: Nanopass Framework (Sarkar, Waddell, Dybvig, Keep)
module Core.PSG.Nanopass.ReducePipeOperators

open Core.PSG.Types

// ═══════════════════════════════════════════════════════════════════════════
// Pipe Operator Detection
// ═══════════════════════════════════════════════════════════════════════════

/// Get child nodes for a given node
let private getChildNodes (psg: ProgramSemanticGraph) (node: PSGNode) : PSGNode list =
    match node.Children with
    | Parent childIds ->
        childIds
        |> List.choose (fun id -> Map.tryFind id.Value psg.Nodes)
    | _ -> []

/// Check if a node is a forward pipe operator (|>)
let private isForwardPipe (node: PSGNode) : bool =
    node.SyntaxKind.Contains("op_PipeRight") ||
    (node.SyntaxKind.StartsWith("LongIdent:") && node.SyntaxKind.Contains("(|>)"))

/// Check if a node is a backward pipe operator (<|)
let private isBackwardPipe (node: PSGNode) : bool =
    node.SyntaxKind.Contains("op_PipeLeft") ||
    (node.SyntaxKind.StartsWith("LongIdent:") && node.SyntaxKind.Contains("(<|)"))

/// Check if a node is any pipe operator
let private isPipeOperator (node: PSGNode) : bool =
    isForwardPipe node || isBackwardPipe node

/// Check if an App node is a pipe application
let private isPipeApp (psg: ProgramSemanticGraph) (node: PSGNode) : bool =
    if not (node.SyntaxKind.StartsWith("App:")) then false
    else
        let children = getChildNodes psg node
        match children with
        | first :: _ -> isPipeOperator first
        | _ -> false

// ═══════════════════════════════════════════════════════════════════════════
// Beta Reduction
// ═══════════════════════════════════════════════════════════════════════════

/// Reduce a pipe application to direct function application.
///
/// For forward pipe (|>):
///   x |> f  ≡  (|>) x f  →  f x
///   Children: [op, value, func] → [func, value]
///
/// For backward pipe (<|):
///   f <| x  ≡  (<|) f x  →  f x
///   Children: [op, func, value] → [func, value]
let private reducePipeApp (psg: ProgramSemanticGraph) (node: PSGNode) : PSGNode =
    let children = getChildNodes psg node
    match children with
    | pipeOp :: arg1 :: arg2 :: rest when isPipeOperator pipeOp ->
        let (funcNode, valueNode) =
            if isForwardPipe pipeOp then
                // x |> f: arg1=value, arg2=func
                (arg2, arg1)
            else
                // f <| x: arg1=func, arg2=value
                (arg1, arg2)

        // Build new children: function first, then value, then any additional args
        let newChildren = funcNode :: valueNode :: rest
        let newChildIds = newChildren |> List.map (fun n -> n.Id)

        // Update node with reduced structure
        // Symbol changes to the function being applied
        { node with
            Children = Parent newChildIds
            Symbol = funcNode.Symbol }
    | _ ->
        // Malformed pipe application - leave unchanged
        node

// ═══════════════════════════════════════════════════════════════════════════
// Parent Reference Consistency
// ═══════════════════════════════════════════════════════════════════════════

/// Rebuild parent references from Children relationships.
/// After reduction, child order changes but parent refs may be stale.
let private rebuildParentRefs (psg: ProgramSemanticGraph) : ProgramSemanticGraph =
    // Build map: childId -> parentId from all Children relationships
    let parentMap =
        psg.Nodes
        |> Map.toSeq
        |> Seq.collect (fun (_, node) ->
            match node.Children with
            | Parent childIds ->
                childIds |> List.map (fun childId -> (childId.Value, node.Id))
            | _ -> [])
        |> Map.ofSeq

    // Update each node's ParentId if it differs
    let updatedNodes =
        psg.Nodes
        |> Map.map (fun nodeId node ->
            match Map.tryFind nodeId parentMap with
            | Some newParentId when node.ParentId <> Some newParentId ->
                { node with ParentId = Some newParentId }
            | _ -> node)

    { psg with Nodes = updatedNodes }

// ═══════════════════════════════════════════════════════════════════════════
// Main Nanopass Entry Point
// ═══════════════════════════════════════════════════════════════════════════

/// Reduce all pipe operator applications in the PSG.
///
/// This nanopass performs beta reduction on |> and <| operators,
/// transforming them to direct function application.
///
/// Composition operators (>>, <<) are NOT handled here as they
/// introduce lambdas and require different treatment.
let reducePipeOperators (psg: ProgramSemanticGraph) : ProgramSemanticGraph =
    // Find all pipe application nodes
    let pipeApps =
        psg.Nodes
        |> Map.toList
        |> List.map snd
        |> List.filter (isPipeApp psg)

    if List.isEmpty pipeApps then
        printfn "[NANOPASS] ReducePipeOperators: No pipe applications found"
        psg
    else
        printfn "[NANOPASS] ReducePipeOperators: Reducing %d pipe applications" (List.length pipeApps)

        // Reduce each pipe application
        let reducedNodes =
            pipeApps
            |> List.map (fun node -> (node.Id.Value, reducePipeApp psg node))
            |> Map.ofList

        // Update PSG with reduced nodes
        let updatedNodes =
            psg.Nodes
            |> Map.map (fun nodeId node ->
                match Map.tryFind nodeId reducedNodes with
                | Some reducedNode -> reducedNode
                | None -> node)

        let psgWithReductions = { psg with Nodes = updatedNodes }

        // Ensure parent references are consistent
        let finalPsg = rebuildParentRefs psgWithReductions

        printfn "[NANOPASS] ReducePipeOperators: Reduction complete"
        finalPsg
