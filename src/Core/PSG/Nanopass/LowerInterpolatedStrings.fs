/// LowerInterpolatedStrings - Nanopass to transform F# interpolated strings to NativeStr operations
///
/// F# interpolated strings like $"Hello, {name}!" compile to InterpolatedString AST nodes.
/// This nanopass transforms them into calls to NativeStr semantic primitives (concat2, concat3, etc.)
/// which Alex can then emit as target-optimal code.
///
/// Transformation:
///   InterpolatedString                    →  App [NativeStr.concat*]
///     InterpolatedStringPart:String           outputBuffer
///     InterpolatedStringPart:Fill             part1
///     InterpolatedStringPart:String           part2
///     ...                                     ...
///
/// The semantic primitives (concat2, concat3) contain bounded memory operations (for loops)
/// that are NOT emitted as control flow. Alex recognizes them and emits target-optimal code
/// (memcpy, SIMD, unrolled loops) based on the target hardware.
///
/// Run AFTER ClassifyOperations to ensure proper operation classification of the generated calls.
module Core.PSG.Nanopass.LowerInterpolatedStrings

open FSharp.Compiler.Text
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

/// Create a new node ID based on a parent node and suffix
let private createNodeId (parentNode: PSGNode) (suffix: string) : NodeId =
    { Value = sprintf "%s_%s" parentNode.Id.Value suffix }

/// Create a new PSG node
let private createNode (syntaxKind: string) (range: range) (fileName: string) (parentId: NodeId option) (operation: OperationKind option) : PSGNode =
    {
        Id = NodeId.FromRange(fileName, range)
        SyntaxKind = syntaxKind
        Symbol = None
        Type = None
        Constraints = None
        Range = range
        SourceFile = fileName
        ParentId = parentId
        Children = NotProcessed
        IsReachable = false
        EliminationPass = None
        EliminationReason = None
        ReachabilityDistance = None
        ContextRequirement = None
        ComputationPattern = None
        Operation = operation
        ConstantValue = None
        Kind = SKExpr EApp  // Lowered interpolated strings become App nodes
        SRTPResolution = None
    }

// ═══════════════════════════════════════════════════════════════════════════
// Interpolated String Detection
// ═══════════════════════════════════════════════════════════════════════════

/// Check if a node is an InterpolatedString
let private isInterpolatedString (node: PSGNode) : bool =
    node.SyntaxKind = "InterpolatedString"

/// Extract string content from an InterpolatedStringPart:String node
let private extractStringContent (node: PSGNode) : string option =
    // The string content might be in a child Const:String node or in the SyntaxKind
    // For now, we'll need to look at the child nodes
    failwithf "extractStringContent: Not yet implemented for node %s (Id: %s)" node.SyntaxKind node.Id.Value

/// Count interpolated string parts
let private countParts (psg: ProgramSemanticGraph) (node: PSGNode) : int =
    let children = getChildNodes psg node
    children.Length

// ═══════════════════════════════════════════════════════════════════════════
// Lowering Strategy
// ═══════════════════════════════════════════════════════════════════════════

/// Determine which concat function to use based on part count
let private selectConcatOp (partCount: int) : NativeStrOp =
    match partCount with
    | 2 -> StrConcat2
    | 3 -> StrConcat3
    | _ ->
        // For more than 3 parts, we'd need to chain concat2/concat3 calls
        // For now, default to concat3 and handle others later
        StrConcat3

/// Transform an InterpolatedString node into a NativeStr.concat* call
///
/// The transformation creates:
/// 1. A stackalloc for the output buffer
/// 2. A call to concat2/concat3 with the parts
/// 3. The result is a NativeStr pointing to the concatenated output
let private lowerInterpolatedString (psg: ProgramSemanticGraph) (node: PSGNode) : PSGNode * Map<string, PSGNode> =
    let children = getChildNodes psg node
    let partCount = children.Length

    // Select the appropriate concat operation
    let concatOp = selectConcatOp partCount

    // Transform the node to an App node calling NativeStr.concat*
    // The children become arguments to the concat function
    //
    // Note: This is a simplified transformation. A complete implementation would:
    // 1. Insert stackalloc for output buffer
    // 2. Convert string literals to NativeStr.fromStatic calls
    // 3. Properly wire up the arguments
    //
    // For now, we mark the node as lowered and let the emitter handle it
    // with the Operation field set appropriately.

    let transformedNode =
        { node with
            SyntaxKind = "App:LoweredInterpolatedString"
            Operation = Some (NativeStr concatOp) }

    // Return the transformed node with no additional nodes for now
    (transformedNode, Map.empty)

// ═══════════════════════════════════════════════════════════════════════════
// Main Nanopass Entry Point
// ═══════════════════════════════════════════════════════════════════════════

/// Lower all InterpolatedString nodes in the PSG to NativeStr.concat* calls.
///
/// This nanopass transforms F# interpolated strings like $"Hello, {name}!"
/// into calls to semantic primitives that Alex can emit as target-optimal code.
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
                // Add any additional nodes created during transformation
                Map.fold (fun acc k v -> Map.add k v acc) nodes' additional
            ) psg.Nodes

        { psg with Nodes = updatedNodes }
