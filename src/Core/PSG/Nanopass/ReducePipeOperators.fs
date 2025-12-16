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
/// Run BEFORE FlattenApplications (pipe reduction creates curried structures that need flattening).
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

/// Check if an App node is a pipe application (order-independent)
/// Looks for an inner App child that contains a pipe operator.
let private isPipeApp (psg: ProgramSemanticGraph) (node: PSGNode) : bool =
    if not (node.SyntaxKind.StartsWith("App:")) then false
    else
        let children = getChildNodes psg node
        // Find an inner App that CONTAINS a pipe operator (not just any App)
        let hasPipeContainingApp =
            children
            |> List.exists (fun c ->
                c.SyntaxKind.StartsWith("App:") &&
                (let innerChildren = getChildNodes psg c
                 innerChildren |> List.exists isPipeOperator))
        if hasPipeContainingApp then true
        else
            // Direct structure: check if any child is a pipe operator
            children |> List.exists isPipeOperator

// ═══════════════════════════════════════════════════════════════════════════
// Beta Reduction
// ═══════════════════════════════════════════════════════════════════════════

/// Reduce a pipe application to direct function application.
///
/// FCS represents x |> f as nested curried application: App(App(|>, x), f)
/// - Outer App has children: [innerApp, f]
/// - Inner App has children: [|>, x]
///
/// We transform this to: App(f, x)
///
/// For backward pipe (<|): App(App(<|, f), x) → App(f, x)
/// NOTE: Children order is not guaranteed, so we identify by SyntaxKind, not position.
let private reducePipeApp (psg: ProgramSemanticGraph) (node: PSGNode) : PSGNode =
    let children = getChildNodes psg node

    // Find the inner App that CONTAINS the pipe operator (not just any App)
    let innerAppOpt =
        children
        |> List.tryFind (fun c ->
            c.SyntaxKind.StartsWith("App:") &&
            (let innerChildren = getChildNodes psg c
             innerChildren |> List.exists isPipeOperator))
    let otherChildren = children |> List.filter (fun c -> not (c.SyntaxKind.StartsWith("App:")) ||
                                                           innerAppOpt |> Option.map (fun ia -> ia.Id <> c.Id) |> Option.defaultValue true)

    match innerAppOpt with
    | Some innerApp ->
        let innerChildren = getChildNodes psg innerApp
        // Find pipe operator and value in innerChildren (order-independent)
        let pipeOpOpt = innerChildren |> List.tryFind isPipeOperator
        let valueNodes = innerChildren |> List.filter (fun c -> not (isPipeOperator c))

        match pipeOpOpt, valueNodes with
        | Some pipeOp, [valueNode] when isForwardPipe pipeOp ->
            // x |> f: valueNode is x, otherChildren[0] should be f
            match otherChildren with
            | funcNode :: rest ->
                let newChildren = funcNode :: valueNode :: rest
                let newChildIds = newChildren |> List.map (fun n -> n.Id)
                { node with
                    Children = Parent newChildIds
                    Symbol = funcNode.Symbol }
            | _ -> node
        | Some pipeOp, [funcInner] when isBackwardPipe pipeOp ->
            // f <| x: funcInner is f, otherChildren[0] should be x
            match otherChildren with
            | valueNode :: rest ->
                let newChildren = funcInner :: valueNode :: rest
                let newChildIds = newChildren |> List.map (fun n -> n.Id)
                { node with
                    Children = Parent newChildIds
                    Symbol = funcInner.Symbol }
            | _ -> node
        | _ -> node
    | None ->
        // No inner App - check for legacy direct structure
        let pipeOpOpt = children |> List.tryFind isPipeOperator
        let otherNodes = children |> List.filter (fun c -> not (isPipeOperator c))
        match pipeOpOpt, otherNodes with
        | Some pipeOp, [arg1; arg2] ->
            let (funcNode, valueNode) =
                if isForwardPipe pipeOp then (arg2, arg1) else (arg1, arg2)
            let newChildren = [funcNode; valueNode]
            let newChildIds = newChildren |> List.map (fun n -> n.Id)
            { node with
                Children = Parent newChildIds
                Symbol = funcNode.Symbol }
        | _ -> node

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
        psg
    else
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
        rebuildParentRefs psgWithReductions
