/// LowerInterpolatedStrings - Nanopass to transform F# interpolated strings to String.concat calls
///
/// F# interpolated strings like $"Hello, {name}!" compile to InterpolatedString AST nodes.
/// This nanopass transforms them into calls to Alloy.String.concat* functions,
/// which are normal function calls that code generation handles without special cases.
///
/// Transformation:
///   InterpolatedString                    →  App:StringConcat
///     InterpolatedStringPart:String:hash       Const:String:hash
///     InterpolatedStringPart:Fill              {fill expression}
///     InterpolatedStringPart:String:hash       Const:String:hash
///
/// The transformation preserves the semantic meaning while converting to a structure
/// that code generation can emit as normal function calls.
///
/// Run AFTER ClassifyOperations to ensure proper operation classification.
module Core.PSG.Nanopass.LowerInterpolatedStrings

open Core.PSG.Types

// ═══════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Get child nodes for a given node
let private getChildNodes (psg: ProgramSemanticGraph) (node: PSGNode) : PSGNode list =
    match node.Children with
    | Parent childIds ->
        childIds
        |> List.choose (fun id -> Map.tryFind id.Value psg.Nodes)
    | _ -> []

/// Get child node IDs
let private getChildIds (node: PSGNode) : NodeId list =
    match node.Children with
    | Parent childIds -> childIds
    | _ -> []

/// Extract hash from InterpolatedStringPart:String:hash syntax kind
let private extractStringHash (syntaxKind: string) : uint32 option =
    if syntaxKind.StartsWith("InterpolatedStringPart:String:") then
        let hashStr = syntaxKind.Substring("InterpolatedStringPart:String:".Length)
        match System.UInt32.TryParse(hashStr) with
        | true, hash -> Some hash
        | _ -> None
    else
        None

/// Check if a node is an InterpolatedString
let private isInterpolatedString (node: PSGNode) : bool =
    node.SyntaxKind = "InterpolatedString"

/// Check if a node is an InterpolatedStringPart:String
let private isStringPart (node: PSGNode) : bool =
    node.SyntaxKind.StartsWith("InterpolatedStringPart:String:")

/// Check if a node is an InterpolatedStringPart:Fill
let private isFillPart (node: PSGNode) : bool =
    node.SyntaxKind = "InterpolatedStringPart:Fill"

// ═══════════════════════════════════════════════════════════════════════════
// Lowering Strategy
// ═══════════════════════════════════════════════════════════════════════════

/// Select the concat operation based on argument count
let private selectConcatOp (argCount: int) : NativeStrOp =
    match argCount with
    | 2 -> StrConcat2
    | 3 -> StrConcat3
    | _ -> StrConcat3  // For now, handle 2-3 args; more would need chaining

/// Transform an InterpolatedStringPart:String node to a Const:String node
let private transformStringPart (psg: ProgramSemanticGraph) (node: PSGNode) : PSGNode =
    match extractStringHash node.SyntaxKind with
    | Some hash ->
        // Look up the actual string content from StringLiterals
        let stringValue = Map.tryFind hash psg.StringLiterals
        // Transform to Const:String with the actual string value
        { node with
            SyntaxKind = sprintf "Const:String %s" (stringValue |> Option.defaultValue "")
            Kind = SKExpr EConst
            Operation = None
            ConstantValue = stringValue |> Option.map StringValue }
    | None ->
        // Shouldn't happen, but preserve the node if we can't parse
        node

/// Transform an InterpolatedStringPart:Fill node - extract the child expression
/// The Fill node wraps an expression; we want the expression itself as the argument
let private transformFillPart (psg: ProgramSemanticGraph) (node: PSGNode) : PSGNode option =
    let children = getChildNodes psg node
    match children with
    | [child] -> Some child  // Return the wrapped expression
    | _ -> None  // Unexpected structure

/// Transform an InterpolatedString node into a String.concat call structure
///
/// The transformation:
/// 1. Transforms string parts to Const:String nodes
/// 2. Extracts fill expressions from their wrappers
/// 3. Creates an App node with StringConcat operation and the arguments as children
let private lowerInterpolatedString (psg: ProgramSemanticGraph) (node: PSGNode) : PSGNode * Map<string, PSGNode> =
    let children = getChildNodes psg node

    // Transform each child part
    let transformedArgs =
        children
        |> List.choose (fun child ->
            if isStringPart child then
                Some (transformStringPart psg child)
            elif isFillPart child then
                transformFillPart psg child
            else
                // Unknown part type - keep as-is
                Some child)

    let argCount = List.length transformedArgs
    let concatOp = selectConcatOp argCount

    // Create new child IDs for the transformed arguments
    let newChildIds = transformedArgs |> List.map (fun n -> n.Id)

    // Transform the parent node to an App:StringConcat
    let transformedNode =
        { node with
            SyntaxKind = "App:StringConcat"
            Kind = SKExpr EApp
            Operation = Some (NativeStr concatOp)
            Children = Parent newChildIds }

    // Build map of additional/updated nodes (the transformed arguments)
    let additionalNodes =
        transformedArgs
        |> List.map (fun n -> (n.Id.Value, n))
        |> Map.ofList

    (transformedNode, additionalNodes)

// ═══════════════════════════════════════════════════════════════════════════
// Main Nanopass Entry Point
// ═══════════════════════════════════════════════════════════════════════════

/// Lower all InterpolatedString nodes in the PSG to String.concat calls.
///
/// This nanopass transforms F# interpolated strings like $"Hello, {name}!"
/// into normal function call structures that code generation can emit.
let lowerInterpolatedStrings (psg: ProgramSemanticGraph) : ProgramSemanticGraph =
    // Find all InterpolatedString nodes
    let interpStrings =
        psg.Nodes
        |> Map.toList
        |> List.map snd
        |> List.filter isInterpolatedString

    if List.isEmpty interpStrings then
        psg
    else
        // Transform each interpolated string
        let transformations =
            interpStrings
            |> List.map (fun node ->
                let (transformed, additionalNodes) = lowerInterpolatedString psg node
                (node.Id.Value, transformed, additionalNodes))

        // Apply transformations to PSG
        let updatedNodes =
            transformations
            |> List.fold (fun nodes (nodeId, transformed, additional) ->
                let nodes' = Map.add nodeId transformed nodes
                // Add/update transformed argument nodes
                Map.fold (fun acc k v -> Map.add k v acc) nodes' additional
            ) psg.Nodes

        { psg with Nodes = updatedNodes }
