/// FlattenApplications - Nanopass to normalize curried function applications
///
/// F# uses curried application semantics, so `f a b c` parses as `((f a) b) c`.
/// This creates nested App nodes in the PSG:
///   App [f]
///     App [f]
///       App [f]
///         Ident:f
///         arg_a
///       arg_b
///     arg_c
///
/// This nanopass flattens these into a single App node with all arguments:
///   App [f]
///     Ident:f
///     arg_a
///     arg_b
///     arg_c
///
/// This normalization:
/// 1. Simplifies downstream passes (they see flat structure)
/// 2. Makes pattern matching in emitters straightforward
/// 3. Follows nanopass principle: normalize early, process simple forms later
///
/// Run AFTER ReducePipeOperators (flattens curried structures from pipe reduction).
/// Run BEFORE DefUseEdges so edges are built on the flattened structure.
///
/// Reference: Nanopass Framework (Sarkar, Waddell, Dybvig, Keep)
module Core.PSG.Nanopass.FlattenApplications

open Core.PSG.Types
open Core.PSG.NavigationUtils

// ═══════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Check if a node is an App (function application)
let private isAppNode (node: PSGNode) : bool =
    node.SyntaxKind.StartsWith("App:")

/// Check if a node is an Ident or LongIdent (function reference)
let private isFunctionRef (node: PSGNode) : bool =
    node.SyntaxKind.StartsWith("Ident:") ||
    node.SyntaxKind.StartsWith("LongIdent:") ||
    node.SyntaxKind.StartsWith("Value:")

// ═══════════════════════════════════════════════════════════════════════════
// Flattening Algorithm
// ═══════════════════════════════════════════════════════════════════════════

/// Collect all arguments from a chain of curried applications.
/// Returns (functionNode, argumentNodes) where argumentNodes are in application order.
///
/// For `((f a) b) c`:
///   - functionNode = Ident:f
///   - argumentNodes = [a, b, c]
let rec private collectCurriedArgs (psg: ProgramSemanticGraph) (node: PSGNode) : PSGNode * PSGNode list =
    let children = getChildNodes psg node
    match children with
    | [] ->
        // Leaf node (shouldn't happen for well-formed App)
        (node, [])
    | [single] ->
        // Single child - could be the function itself
        if isAppNode single then
            collectCurriedArgs psg single
        else
            (single, [])
    | first :: rest ->
        // First child is either nested App or the function
        if isAppNode first then
            // Nested application - recurse to collect inner args
            let (funcNode, innerArgs) = collectCurriedArgs psg first
            (funcNode, innerArgs @ rest)
        else
            // First child is the function, rest are arguments
            (first, rest)

/// Check if a node is a curried application (has nested App as first child)
let private isCurriedApp (psg: ProgramSemanticGraph) (node: PSGNode) : bool =
    if not (isAppNode node) then false
    else
        let children = getChildNodes psg node
        match children with
        | first :: _ -> isAppNode first
        | _ -> false

/// Flatten a single curried application node.
/// Updates the node's Children to include all collected arguments.
let private flattenNode (psg: ProgramSemanticGraph) (node: PSGNode) : PSGNode =
    if not (isCurriedApp psg node) then node
    else
        let (funcNode, allArgs) = collectCurriedArgs psg node
        // Build new children list: function followed by all arguments
        let newChildIds =
            (funcNode :: allArgs)
            |> List.map (fun n -> n.Id)
        { node with Children = Parent newChildIds }

// ═══════════════════════════════════════════════════════════════════════════
// Parent ID Updates
// ═══════════════════════════════════════════════════════════════════════════

/// After flattening, some nodes that were children of inner Apps are now
/// children of the outer App. Update their ParentId accordingly.
let private updateParentIds (psg: ProgramSemanticGraph) : ProgramSemanticGraph =
    // Build a map of nodeId -> parentId from the Children relationships
    let parentMap =
        psg.Nodes
        |> Map.toSeq
        |> Seq.collect (fun (_, node) ->
            match node.Children with
            | Parent childIds ->
                childIds |> List.map (fun childId -> (childId.Value, node.Id))
            | _ -> [])
        |> Map.ofSeq

    // Update each node's ParentId based on the map
    let updatedNodes =
        psg.Nodes
        |> Map.map (fun nodeId node ->
            match Map.tryFind nodeId parentMap with
            | Some newParentId ->
                if node.ParentId <> Some newParentId then
                    { node with ParentId = Some newParentId }
                else node
            | None -> node)

    { psg with Nodes = updatedNodes }

// ═══════════════════════════════════════════════════════════════════════════
// Main Nanopass Entry Point
// ═══════════════════════════════════════════════════════════════════════════

/// Flatten all curried applications in the PSG.
/// This is the main entry point for the nanopass.
let flattenApplications (psg: ProgramSemanticGraph) : ProgramSemanticGraph =
    // Find all curried App nodes
    let curriedNodes =
        psg.Nodes
        |> Map.toList
        |> List.map snd
        |> List.filter (isCurriedApp psg)

    if List.isEmpty curriedNodes then
        psg
    else
        // Flatten each curried node
        let flattenedNodes =
            curriedNodes
            |> List.map (fun node -> (node.Id.Value, flattenNode psg node))
            |> Map.ofList

        // Update the PSG with flattened nodes
        let updatedNodes =
            psg.Nodes
            |> Map.map (fun nodeId node ->
                match Map.tryFind nodeId flattenedNodes with
                | Some flatNode -> flatNode
                | None -> node)

        let psgWithFlatNodes = { psg with Nodes = updatedNodes }

        // Update parent IDs to reflect new structure
        updateParentIds psgWithFlatNodes
