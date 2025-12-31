/// DetectInlineFunctions - Nanopass to mark inline function bindings in the PSG
///
/// This nanopass scans for PSG nodes whose symbols are F# inline functions
/// and sets their IsInlineFunction field to true.
///
/// Inline functions should NOT be compiled as standalone functions - they are
/// inlined at call sites. This marker allows Transfer.fs to exclude them from
/// the list of functions that need standalone MLIR generation.
///
/// Run AFTER reachability to only process reachable nodes.
module Core.PSG.Nanopass.DetectInlineFunctions

open FSharp.Native.Compiler.Symbols
open Core.PSG.Types

// ═══════════════════════════════════════════════════════════════════════════
// Inline Function Detection
// ═══════════════════════════════════════════════════════════════════════════

/// Check if a node represents an inline function
let private isInlineFunction (node: PSGNode) : bool =
    // Only process reachable binding nodes
    if not node.IsReachable then
        false
    elif not (SyntaxKindT.isBinding node.Kind) then
        false
    else
        match node.Symbol with
        | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
            // FCS InlineAnnotation enum:
            // - AlwaysInline = explicitly marked with 'inline' keyword or [<InlineIfLambda>]
            // - OptionalInline = compiler MAY inline (default for most functions)
            // - NeverInline = never inline
            //
            // CRITICAL: Only skip AlwaysInline functions. OptionalInline functions
            // are regular functions that the compiler might optimize - they still
            // need standalone MLIR generation for Firefly.
            try mfv.InlineAnnotation = FSharpInlineAnnotation.AlwaysInline with _ -> false
        | _ -> false

/// Process a single node - if it's an inline function, mark it
let private processNode (node: PSGNode) : PSGNode =
    if isInlineFunction node then
        { node with IsInlineFunction = true }
    else
        node

// ═══════════════════════════════════════════════════════════════════════════
// Main Nanopass Entry Point
// ═══════════════════════════════════════════════════════════════════════════

/// Detect inline functions and set IsInlineFunction on PSG nodes.
///
/// This nanopass scans for F# inline functions and marks them so that
/// Transfer.fs can exclude them from standalone function generation.
/// Inline function bodies are inlined at call sites, not compiled as
/// separate functions.
///
/// Run AFTER reachability to only process reachable nodes.
let detectInlineFunctions (psg: ProgramSemanticGraph) : ProgramSemanticGraph =
    let updatedNodes =
        psg.Nodes
        |> Map.map (fun _ node -> processNode node)

    { psg with Nodes = updatedNodes }
