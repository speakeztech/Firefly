/// DetectPlatformBindings - Nanopass to mark platform binding declarations in the PSG
///
/// This nanopass scans for PSG nodes whose symbols are functions in the
/// Alloy.Platform.Bindings module and sets their PlatformBinding field.
///
/// The PlatformBinding field allows Alex (MLIR generation) to recognize platform bindings
/// without name-string matching - it simply checks node.PlatformBinding.IsSome.
///
/// Run AFTER reachability to only process reachable nodes.
module Core.PSG.Nanopass.DetectPlatformBindings

open FSharp.Compiler.Symbols
open Core.PSG.Types

// ═══════════════════════════════════════════════════════════════════════════
// Platform Binding Detection
// ═══════════════════════════════════════════════════════════════════════════

/// The module path that indicates a platform binding (BCL-free pattern)
let private platformBindingsModule = "Alloy.Platform.Bindings"

/// Try to extract platform binding info from a function/value symbol
/// Returns Some if the symbol is a Platform.Bindings function
let private tryExtractPlatformBindingInfo (mfv: FSharpMemberOrFunctionOrValue) : PlatformBindingInfo option =
    try
        let fullName = mfv.FullName
        if fullName.StartsWith(platformBindingsModule + ".") then
            // Extract function name as entry point
            // e.g., "writeBytes" from "Alloy.Platform.Bindings.writeBytes"
            let entryPoint = fullName.Substring(platformBindingsModule.Length + 1)
            Some {
                Library = "platform"          // Marker indicating platform-provided binding
                EntryPoint = entryPoint
                CallingConvention = "Cdecl"   // Default for platform bindings
            }
        else
            None
    with _ -> None

/// Process a single node - if it's a platform binding, set PlatformBinding
let private processNode (node: PSGNode) : PSGNode =
    // Only process reachable nodes
    if not node.IsReachable then
        node
    else
        match node.Symbol with
        | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
            match tryExtractPlatformBindingInfo mfv with
            | Some info -> { node with PlatformBinding = Some info }
            | None -> node
        | _ -> node

// ═══════════════════════════════════════════════════════════════════════════
// Main Nanopass Entry Point
// ═══════════════════════════════════════════════════════════════════════════

/// Detect platform bindings and set PlatformBinding on PSG nodes.
///
/// This nanopass scans for Alloy.Platform.Bindings function calls and
/// marks them with PlatformBinding so that Alex can recognize them without
/// name-string matching.
///
/// Run AFTER reachability to only process reachable nodes.
let detectPlatformBindings (psg: ProgramSemanticGraph) : ProgramSemanticGraph =
    let updatedNodes =
        psg.Nodes
        |> Map.map (fun _ node -> processNode node)

    { psg with Nodes = updatedNodes }
