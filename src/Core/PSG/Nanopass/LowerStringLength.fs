/// LowerStringLength Nanopass
/// Transforms PropertyAccess:Length on runtime strings to SemanticPrimitive:fidelity_strlen.
///
/// This nanopass runs AFTER ConstantPropagation, so it only handles PropertyAccess:Length
/// nodes that weren't constant-folded (i.e., where the target is a runtime string, not a literal).
///
/// The transformation changes SyntaxKind from "PropertyAccess:Length" to "SemanticPrimitive:fidelity_strlen",
/// which code generation handles by emitting inline assembly for strlen.
module Core.PSG.Nanopass.LowerStringLength

open Core.PSG.Types

/// Check if a node's symbol indicates it's System.String.Length
let private isStringLengthSymbol (node: PSGNode) : bool =
    match node.Symbol with
    | Some sym ->
        try
            sym.FullName = "System.String.Length"
        with _ -> false
    | None -> false

/// Lower PropertyAccess:Length nodes that are System.String.Length
let private lowerStringLengthNodes (psg: ProgramSemanticGraph) : ProgramSemanticGraph =
    let updatedNodes =
        psg.Nodes
        |> Map.map (fun nodeIdVal node ->
            // Check for PropertyAccess:Length that is System.String.Length and wasn't constant-folded
            if node.SyntaxKind = "PropertyAccess:Length" &&
               node.ConstantValue.IsNone &&
               isStringLengthSymbol node then
                printfn "[NANOPASS] LowerStringLength: Lowering node %A to SemanticPrimitive:fidelity_strlen" node.Id
                { node with SyntaxKind = "SemanticPrimitive:fidelity_strlen" }
            else
                node)
    { psg with Nodes = updatedNodes }

/// Main entry point for the LowerStringLength nanopass
let lowerStringLength (psg: ProgramSemanticGraph) : ProgramSemanticGraph =
    printfn "[NANOPASS] LowerStringLength: Starting string length lowering"

    let result = lowerStringLengthNodes psg

    // Count how many nodes were lowered
    let loweredCount =
        result.Nodes
        |> Map.filter (fun id node ->
            match psg.Nodes |> Map.tryFind id with
            | Some original ->
                original.SyntaxKind = "PropertyAccess:Length" &&
                node.SyntaxKind = "SemanticPrimitive:fidelity_strlen"
            | None -> false)
        |> Map.count

    printfn "[NANOPASS] LowerStringLength: Lowered %d PropertyAccess:Length nodes" loweredCount
    result
