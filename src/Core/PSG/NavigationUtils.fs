/// NavigationUtils - Shared PSG navigation helpers
///
/// Common utility functions for navigating the PSG structure.
/// Used by nanopasses that need to traverse parent-child relationships.
module Core.PSG.NavigationUtils

open Core.PSG.Types

/// Get child nodes for a given node by resolving child IDs
let getChildNodes (psg: ProgramSemanticGraph) (node: PSGNode) : PSGNode list =
    match node.Children with
    | Parent childIds ->
        childIds
        |> List.choose (fun id -> Map.tryFind id.Value psg.Nodes)
    | _ -> []

/// Get child node IDs without resolving
let getChildIds (node: PSGNode) : NodeId list =
    match node.Children with
    | Parent childIds -> childIds
    | _ -> []

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
