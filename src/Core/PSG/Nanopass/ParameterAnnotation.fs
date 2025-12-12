/// ParameterAnnotation - Nanopass to annotate function parameters in the PSG
///
/// This nanopass transforms a PSG by annotating Pattern:Named nodes that represent
/// function parameters with their parameter index in the ContextRequirement field.
///
/// After this pass, parameter nodes will have:
///   ContextRequirement = Some "Parameter(N)"
/// where N is the 0-based parameter index.
///
/// This allows the emitter to identify parameters and record their SSA values
/// without needing to traverse PSG structure at emission time.
///
/// Reference: Nanopass Framework (Sarkar, Waddell, Dybvig, Keep)
module Core.PSG.Nanopass.ParameterAnnotation

open Core.PSG.Types

// ═══════════════════════════════════════════════════════════════════════════
// Parameter Discovery
// ═══════════════════════════════════════════════════════════════════════════

/// Find all Pattern:Named nodes that are children of Pattern:LongIdent nodes
/// which are themselves children of Binding nodes. These are function parameters.
let private findParameterNodes (psg: ProgramSemanticGraph) : (PSGNode * int) list =
    psg.Nodes
    |> Map.toList
    |> List.map snd
    |> List.filter (fun node -> node.SyntaxKind.StartsWith("Binding"))
    |> List.collect (fun bindingNode ->
        // Get children of binding node
        match bindingNode.Children with
        | Parent childIds ->
            childIds
            |> List.choose (fun childId -> Map.tryFind childId.Value psg.Nodes)
            |> List.filter (fun n -> n.SyntaxKind.StartsWith("Pattern:LongIdent"))
            |> List.collect (fun longIdentNode ->
                // Get Pattern:Named children (parameters)
                match longIdentNode.Children with
                | Parent paramIds ->
                    paramIds
                    |> List.choose (fun paramId -> Map.tryFind paramId.Value psg.Nodes)
                    |> List.filter (fun n -> n.SyntaxKind.StartsWith("Pattern:Named"))
                    |> List.mapi (fun i node -> (node, i))
                | _ -> [])
        | _ -> [])

// ═══════════════════════════════════════════════════════════════════════════
// Node Annotation
// ═══════════════════════════════════════════════════════════════════════════

/// Annotate a parameter node with its index
let private annotateParameter (node: PSGNode) (index: int) : PSGNode =
    { node with ContextRequirement = Some (Parameter index) }

/// Apply parameter annotations to the PSG
let annotateParameters (psg: ProgramSemanticGraph) : ProgramSemanticGraph =
    let parameterNodes = findParameterNodes psg

    // Build a map of node ID -> annotated node
    let annotations =
        parameterNodes
        |> List.map (fun (node, idx) -> (node.Id.Value, annotateParameter node idx))
        |> Map.ofList

    // Update nodes in the PSG
    let updatedNodes =
        psg.Nodes
        |> Map.map (fun nodeId node ->
            match Map.tryFind nodeId annotations with
            | Some annotatedNode -> annotatedNode
            | None -> node)

    { psg with Nodes = updatedNodes }
