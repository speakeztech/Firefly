/// ConstantPropagation Nanopass
/// Evaluates constant expressions at compile time, setting ConstantValue on nodes.
///
/// Currently handles:
/// - PropertyAccess:Length on string literals → Int32Value
///
/// This is critical for freestanding compilation where runtime functions
/// like strlen are not available.
module Core.PSG.Nanopass.ConstantPropagation

open Core.PSG.Types

/// Follow def-use edges to find the defining node for an identifier
let private findDefinition (psg: ProgramSemanticGraph) (nodeId: NodeId) : PSGNode option =
    // Look for a SymbolUse edge where this node is the source
    psg.Edges
    |> List.tryPick (fun edge ->
        if edge.Source = nodeId && edge.Kind = SymbolUse then
            Map.tryFind edge.Target.Value psg.Nodes
        else
            None)

/// Get the ConstantValue for a node, following def-use edges if needed
let private resolveConstantValue (psg: ProgramSemanticGraph) (node: PSGNode) : ConstantValue option =
    match node.ConstantValue with
    | Some cv -> Some cv  // Direct constant
    | None ->
        // Check if this is an identifier that references a constant
        if node.SyntaxKind.StartsWith("Ident:") || node.SyntaxKind.StartsWith("LongIdent:") then
            // Follow def-use edge to find definition
            match findDefinition psg node.Id with
            | Some defNode -> defNode.ConstantValue
            | None -> None
        else
            None

/// Get children of a node
let private getChildren (psg: ProgramSemanticGraph) (node: PSGNode) : PSGNode list =
    psg.Edges
    |> List.choose (fun edge ->
        if edge.Kind = ChildOf && edge.Target = node.Id then
            Map.tryFind edge.Source.Value psg.Nodes
        else
            None)

/// Propagate constants through PropertyAccess:Length nodes
let private propagateStringLength (psg: ProgramSemanticGraph) : ProgramSemanticGraph =
    let updatedNodes =
        psg.Nodes
        |> Map.map (fun nodeIdVal node ->
            // Check for PropertyAccess:Length
            if node.SyntaxKind = "PropertyAccess:Length" then
                // Get the receiver (child node)
                let children = getChildren psg node
                match children with
                | [receiver] ->
                    // Resolve the receiver's constant value
                    match resolveConstantValue psg receiver with
                    | Some (StringValue s) ->
                        // String literal! Fold the length at compile time
                        { node with ConstantValue = Some (Int32Value s.Length) }
                    | _ -> node  // Not a constant string
                | _ -> node  // Multiple or no children
            else
                node)
    { psg with Nodes = updatedNodes }

/// Main entry point for the ConstantPropagation nanopass
let propagateConstants (psg: ProgramSemanticGraph) : ProgramSemanticGraph =
    propagateStringLength psg
