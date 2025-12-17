/// Nanopass: ValidateNativeTypes
/// Validates that only native-compatible types appear in the PSG.
/// This is a CRITICAL validation - non-native types require .NET runtime and break freestanding compilation.
///
/// ALLOWED (native-compatible types):
///   - nativeptr<T>, nativeint, voption, etc.
///   - FSharp.Core.Operators (arithmetic, comparison, etc.)
///   - Microsoft.FSharp.NativeInterop.*
///   - Primitive types (int, byte, bool, etc.)
///   - Alloy-provided implementations
///
/// DISALLOWED (types requiring .NET runtime):
///   - System.Object.* (except when shadowed by Alloy)
///   - System.String.* (except when shadowed by Alloy)
///   - System.Console.*
///   - System.Collections.*
///   - Any System.* that requires runtime allocation/GC
///
/// When non-native types are found, this indicates either:
///   1. Alloy is missing a native implementation
///   2. User code is using BCL directly (not through Alloy)
///   3. FCS resolved to BCL instead of Alloy's replacement
///
/// All of these are ARCHITECTURAL FAILURES that must be fixed upstream, not worked around.
module Core.PSG.Nanopass.ValidateNativeTypes

open FSharp.Compiler.Symbols
open Core.PSG.Types

// ═══════════════════════════════════════════════════════════════════
// Native Type Validation Configuration
// ═══════════════════════════════════════════════════════════════════

/// Disallowed types that require .NET runtime (non-native)
let private disallowedNonNativePrefixes = [
    "System.Object.ReferenceEquals"
    "System.Object.Equals"
    "System.Object.GetHashCode"
    "System.Object.GetType"
    "System.Console."
    "System.Collections."
    "System.Text.Encoding"
    "System.IO."
    "System.Runtime.CompilerServices.RuntimeHelpers"
    "System.GC."
    "System.Reflection."
    "System.Threading."
    // Add more as discovered
]

/// Explicitly allowed native-compatible types
let private allowedNativePrefixes = [
    "Microsoft.FSharp.Core.Operators."
    "Microsoft.FSharp.Core.LanguagePrimitives."
    "Microsoft.FSharp.NativeInterop."
    "Microsoft.FSharp.Core.ExtraTopLevelOperators."
    "Microsoft.FSharp.Collections.Array."  // Pure F# array operations
    "Microsoft.FSharp.Core.Option."
    "Microsoft.FSharp.Core.ValueOption."
    "Microsoft.FSharp.Core.Result."
    // Primitive type operations
    "System.Int32."
    "System.Int64."
    "System.Byte."
    "System.Boolean."
    "System.Char."
    "System.IntPtr."
    "System.UIntPtr."
    // String.Length is allowed (property access, not method)
    "System.String.get_Length"
    "System.String.get_Chars"
]

// ═══════════════════════════════════════════════════════════════════
// Validation Logic
// ═══════════════════════════════════════════════════════════════════

/// Native type validation error
type NativeTypeValidationError = {
    SymbolName: string
    NodeId: string
    SyntaxKind: string
    Message: string
}

/// Check if a symbol name is a disallowed non-native type
let private isDisallowedNonNative (fullName: string) : bool =
    // First check if explicitly allowed
    let isAllowed = allowedNativePrefixes |> List.exists (fun prefix -> fullName.StartsWith(prefix))
    if isAllowed then false
    else
        // Then check if disallowed
        disallowedNonNativePrefixes |> List.exists (fun prefix ->
            fullName.StartsWith(prefix) || fullName = prefix.TrimEnd('.'))

/// Validate a single symbol
let private validateSymbol (nodeId: string) (syntaxKind: string) (symbol: FSharpSymbol) : NativeTypeValidationError option =
    let fullName =
        match symbol with
        | :? FSharpMemberOrFunctionOrValue as mfv ->
            try mfv.FullName with _ -> mfv.DisplayName
        | :? FSharpEntity as entity ->
            entity.TryFullName |> Option.defaultValue entity.DisplayName
        | _ ->
            try symbol.FullName with _ -> symbol.DisplayName

    if isDisallowedNonNative fullName then
        Some {
            SymbolName = fullName
            NodeId = nodeId
            SyntaxKind = syntaxKind
            Message = sprintf "NON-NATIVE TYPE: '%s' requires .NET runtime. Alloy must provide a native implementation." fullName
        }
    else
        None

/// Validate all symbols in a PSG node
let private validateNode (node: PSGNode) : NativeTypeValidationError list =
    match node.Symbol with
    | Some symbol ->
        validateSymbol node.Id.Value (SyntaxKindT.toString node.Kind) symbol
        |> Option.toList
    | None -> []

// ═══════════════════════════════════════════════════════════════════
// Nanopass Entry Point
// ═══════════════════════════════════════════════════════════════════

/// Result of native type validation
type ValidationResult = {
    Errors: NativeTypeValidationError list
    HasErrors: bool
}

/// Run native type validation on the PSG
/// Returns a list of errors. If any errors exist, compilation MUST halt.
let validate (graph: ProgramSemanticGraph) : ValidationResult =
    let errors =
        graph.Nodes
        |> Map.toList
        |> List.collect (fun (_, node) -> validateNode node)

    { Errors = errors; HasErrors = not (List.isEmpty errors) }

/// Validate only reachable nodes (more efficient for large codebases)
let validateReachable (graph: ProgramSemanticGraph) : ValidationResult =
    let errors =
        graph.Nodes
        |> Map.toList
        |> List.filter (fun (_, node) -> node.IsReachable)
        |> List.collect (fun (_, node) -> validateNode node)

    { Errors = errors; HasErrors = not (List.isEmpty errors) }
