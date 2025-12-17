/// ReduceAlloyOperators - Nanopass to reduce Alloy library operators to direct application
///
/// Alloy defines the ($) operator as direct function application:
///   let inline ($) f x = f x
///
/// This is semantically equivalent to (<|) but comes from the Alloy library,
/// not from F# Core. It's used throughout Alloy for ergonomic function application.
///
/// This nanopass performs beta reduction on $ applications, transforming:
///   App [$]                →  App [f]
///     LongIdent:op_Dollar       <function>
///     <function>                <value>
///     <value>
///
/// Run AFTER FlattenApplications, AFTER ReducePipeOperators, BEFORE DefUseEdges.
///
/// Reference: Alloy/Core.fs - operator definition
/// Reference: Nanopass Framework (Sarkar, Waddell, Dybvig, Keep)
module Core.PSG.Nanopass.ReduceAlloyOperators

open Core.PSG.Types
open Core.PSG.NavigationUtils

// ═══════════════════════════════════════════════════════════════════════════
// Alloy Operator Detection
// ═══════════════════════════════════════════════════════════════════════════

/// Check if a node is Alloy's $ (direct application) operator
/// The $ operator is defined as: let inline ($) f x = f x
let private isDollarOperator (node: PSGNode) : bool =
    node.SyntaxKind.Contains("op_Dollar")

/// Check if an App node is a $ application
let private isDollarApp (psg: ProgramSemanticGraph) (node: PSGNode) : bool =
    if not (node.SyntaxKind.StartsWith("App:")) then false
    else
        let children = getChildNodes psg node
        match children with
        | first :: _ -> isDollarOperator first
        | _ -> false

// ═══════════════════════════════════════════════════════════════════════════
// Beta Reduction
// ═══════════════════════════════════════════════════════════════════════════

/// Reduce a $ application to direct function application.
///
/// For $ operator:
///   f $ x  ≡  ($) f x  →  f x
///   Children: [op, func, value] → [func, value]
let private reduceDollarApp (psg: ProgramSemanticGraph) (node: PSGNode) : PSGNode =
    let children = getChildNodes psg node
    match children with
    | dollarOp :: funcNode :: valueNode :: rest when isDollarOperator dollarOp ->
        // $ f x: func is arg1, value is arg2 (same as <|)
        // Build new children: function first, then value, then any additional args
        let newChildren = funcNode :: valueNode :: rest
        let newChildIds = newChildren |> List.map (fun n -> n.Id)

        // Update node with reduced structure
        // Symbol changes to the function being applied
        { node with
            Children = Parent newChildIds
            Symbol = funcNode.Symbol }
    | _ ->
        // Malformed $ application - leave unchanged
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

/// Reduce all Alloy $ operator applications in the PSG.
///
/// This nanopass performs beta reduction on $ operators,
/// transforming them to direct function application.
let reduceAlloyOperators (psg: ProgramSemanticGraph) : ProgramSemanticGraph =
    // Find all $ application nodes
    let dollarApps =
        psg.Nodes
        |> Map.toList
        |> List.map snd
        |> List.filter (isDollarApp psg)

    if List.isEmpty dollarApps then
        psg
    else
        // Reduce each $ application
        let reducedNodes =
            dollarApps
            |> List.map (fun node -> (node.Id.Value, reduceDollarApp psg node))
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
