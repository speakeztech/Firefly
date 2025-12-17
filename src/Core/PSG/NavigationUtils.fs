/// NavigationUtils - Shared PSG navigation helpers
///
/// Common utility functions for navigating the PSG structure.
/// Used by nanopasses, Baker, and Alex for traversing parent-child relationships.
///
/// IMPORTANT: Edge semantics for ChildOf:
///   ChildOf edges go FROM parent TO child: Source = parent, Target = child
///   This is the AUTHORITATIVE source of truth for graph structure.
module Core.PSG.NavigationUtils

open Core.PSG.Types

/// Get child nodes for a given node using ChildOf edges (authoritative)
/// ChildOf edges: Source = parent, Target = child
let getChildNodes (psg: ProgramSemanticGraph) (node: PSGNode) : PSGNode list =
    psg.Edges
    |> List.choose (fun edge ->
        match edge.Kind with
        | ChildOf when edge.Source = node.Id ->
            Map.tryFind edge.Target.Value psg.Nodes
        | _ -> None)
    |> List.sortBy (fun n -> n.Range.StartLine, n.Range.StartColumn)

/// Get child node IDs using ChildOf edges (authoritative)
/// Returns IDs in source order (sorted by range)
let getChildIds (psg: ProgramSemanticGraph) (node: PSGNode) : NodeId list =
    psg.Edges
    |> List.choose (fun edge ->
        match edge.Kind with
        | ChildOf when edge.Source = node.Id ->
            // Look up node to get range for sorting
            Map.tryFind edge.Target.Value psg.Nodes
            |> Option.map (fun n -> (edge.Target, n.Range.StartLine, n.Range.StartColumn))
        | _ -> None)
    |> List.sortBy (fun (_, line, col) -> (line, col))
    |> List.map (fun (id, _, _) -> id)

/// Get parent node if exists
let getParentNode (psg: ProgramSemanticGraph) (node: PSGNode) : PSGNode option =
    node.ParentId
    |> Option.bind (fun pid -> Map.tryFind pid.Value psg.Nodes)

/// Get all descendants of a node (depth-first traversal)
let getDescendants (psg: ProgramSemanticGraph) (node: PSGNode) : PSGNode list =
    let rec collect (n: PSGNode) =
        let children = getChildNodes psg n
        children @ (children |> List.collect collect)
    collect node

/// Find node by ID value (resolves int to string key)
let tryFindNode (psg: ProgramSemanticGraph) (id: int) : PSGNode option =
    Map.tryFind (string id) psg.Nodes

/// Find node by NodeId
let tryFindNodeById (psg: ProgramSemanticGraph) (nodeId: NodeId) : PSGNode option =
    Map.tryFind (string nodeId.Value) psg.Nodes
