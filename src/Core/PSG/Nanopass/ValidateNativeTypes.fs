/// Nanopass: ValidateNativeTypes
/// Validates that only native-compatible types appear in the PSG.
/// This is a CRITICAL validation - non-native types require .NET runtime and break freestanding compilation.
///
/// NATIVE-COMPATIBLE TYPES (from FSharp.Core prim-types.fs):
///   - IL primitives: nativeptr<T>, voidptr, ilsigptr<T> (defined via (# ... #) syntax)
///   - Value types that map to LLVM: int, int64, byte, float, bool, char, nativeint, unativeint
///   - F# struct types: voption<T>/ValueOption<T>, user [<Struct>] types
///   - Alloy types: NativeStr, and all Alloy.* implementations
///   - unit (void)
///
/// BCL TYPES REQUIRING RUNTIME (DISALLOWED):
///   - Microsoft.FSharp.Core.string (= System.String) - heap-allocated, GC-managed
///   - option<T> - heap-allocated discriminated union
///   - list<T> - heap-allocated linked list
///   - System.Object - reference type base
///   - System.Console, System.IO, System.Collections, etc.
///
/// When non-native types are found in the PSG TypeName field, this indicates:
///   1. Alloy is missing a native implementation
///   2. User code is using BCL directly (not through Alloy)
///   3. Type resolution failed to map to Alloy's native types
///
/// All of these are ARCHITECTURAL FAILURES that must be fixed upstream, not worked around.
module Core.PSG.Nanopass.ValidateNativeTypes

open FSharp.Compiler.Symbols
open Core.PSG.Types

// ═══════════════════════════════════════════════════════════════════
// Native Type Classification (based on FSharp.Core/prim-types.fs)
// ═══════════════════════════════════════════════════════════════════

/// Native-compatible type names (exact matches or prefixes)
/// These map directly to LLVM types without runtime support.
let private nativeCompatibleTypes = [
    // F# primitive value types (from prim-types-prelude.fs)
    // These are type aliases to System.* but are VALUE TYPES with no GC
    "Microsoft.FSharp.Core.int"
    "Microsoft.FSharp.Core.int32"
    "Microsoft.FSharp.Core.int64"
    "Microsoft.FSharp.Core.int16"
    "Microsoft.FSharp.Core.int8"
    "Microsoft.FSharp.Core.byte"
    "Microsoft.FSharp.Core.sbyte"
    "Microsoft.FSharp.Core.uint"
    "Microsoft.FSharp.Core.uint32"
    "Microsoft.FSharp.Core.uint64"
    "Microsoft.FSharp.Core.uint16"
    "Microsoft.FSharp.Core.uint8"
    "Microsoft.FSharp.Core.float"
    "Microsoft.FSharp.Core.float32"
    "Microsoft.FSharp.Core.single"
    "Microsoft.FSharp.Core.double"
    "Microsoft.FSharp.Core.bool"
    "Microsoft.FSharp.Core.char"
    "Microsoft.FSharp.Core.unit"
    "Microsoft.FSharp.Core.nativeint"
    "Microsoft.FSharp.Core.unativeint"

    // IL pointer types (from prim-types-prelude.fs lines 122-126)
    // These compile to raw pointers - no runtime needed
    "Microsoft.FSharp.Core.nativeptr"
    "Microsoft.FSharp.Core.voidptr"
    "Microsoft.FSharp.Core.ilsigptr"
    "Microsoft.FSharp.NativeInterop.nativeptr"
    "Microsoft.FSharp.NativeInterop.voidptr"

    // Value option - struct-based, stack-allocated (from prim-types.fs line 4057)
    "Microsoft.FSharp.Core.ValueOption"
    "Microsoft.FSharp.Core.voption"
    "Microsoft.FSharp.Core.FSharpValueOption"

    // System value types (these ARE native - no heap allocation)
    "System.Int32"
    "System.Int64"
    "System.Int16"
    "System.SByte"
    "System.Byte"
    "System.UInt32"
    "System.UInt64"
    "System.UInt16"
    "System.Single"
    "System.Double"
    "System.Boolean"
    "System.Char"
    "System.IntPtr"
    "System.UIntPtr"
    "System.Void"
]

/// Type name prefixes that are always native-compatible
let private nativeCompatiblePrefixes = [
    // All Alloy types are native by design
    "Alloy."

    // NativeInterop module functions/types
    "Microsoft.FSharp.NativeInterop."

    // Operators produce native values
    "Microsoft.FSharp.Core.Operators."
    "Microsoft.FSharp.Core.LanguagePrimitives."
    "Microsoft.FSharp.Core.ExtraTopLevelOperators."
]

/// BCL types that REQUIRE runtime (heap allocation, GC, etc.)
/// These are ALWAYS errors - no exceptions.
let private bclTypesRequiringRuntime = [
    // String is THE critical one - F# string = System.String
    "Microsoft.FSharp.Core.string"
    "System.String"

    // Reference-based option (heap-allocated DU)
    "Microsoft.FSharp.Core.option"
    "Microsoft.FSharp.Core.FSharpOption"

    // List is heap-allocated
    "Microsoft.FSharp.Collections.list"
    "Microsoft.FSharp.Collections.FSharpList"

    // Object and its methods
    "System.Object"

    // Console I/O (must use Alloy.Console)
    "System.Console"

    // Collections (must use Alloy equivalents)
    "System.Collections."

    // I/O (must use Alloy equivalents)
    "System.IO."

    // Text encoding (must use Alloy.Utf8)
    "System.Text.Encoding"
    "System.Text.StringBuilder"

    // Threading (not supported in freestanding)
    "System.Threading."

    // Reflection (not supported in freestanding)
    "System.Reflection."

    // GC operations (no GC in freestanding)
    "System.GC"

    // Runtime helpers
    "System.Runtime.CompilerServices.RuntimeHelpers"
]

/// Disallowed symbol operations (for backward compat with existing validation)
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
]

/// Explicitly allowed native-compatible symbols
let private allowedNativePrefixes = [
    "Microsoft.FSharp.Core.Operators."
    "Microsoft.FSharp.Core.LanguagePrimitives."
    "Microsoft.FSharp.NativeInterop."
    "Microsoft.FSharp.Core.ExtraTopLevelOperators."
    "Microsoft.FSharp.Collections.Array."
    "Microsoft.FSharp.Core.Option."
    "Microsoft.FSharp.Core.ValueOption."
    "Microsoft.FSharp.Core.Result."
    "System.Int32."
    "System.Int64."
    "System.Byte."
    "System.Boolean."
    "System.Char."
    "System.IntPtr."
    "System.UIntPtr."
    "System.String.get_Length"
    "System.String.get_Chars"
]

// ═══════════════════════════════════════════════════════════════════
// Validation Logic
// ═══════════════════════════════════════════════════════════════════

/// Native type validation error
type NativeTypeValidationError = {
    NodeId: string
    SyntaxKind: string
    TypeName: string option
    SymbolName: string option
    Message: string
}

/// Check if a type name is native-compatible
let private isNativeCompatibleType (typeName: string) : bool =
    // Check exact matches first
    if nativeCompatibleTypes |> List.exists (fun t -> typeName.StartsWith(t)) then
        true
    // Check prefixes
    elif nativeCompatiblePrefixes |> List.exists (fun p -> typeName.StartsWith(p)) then
        true
    // User-defined types in the current project (Examples.*)
    elif typeName.StartsWith("Examples.") then
        true
    else
        false

/// Check if a type name is a known BCL type requiring runtime
let private isBclTypeRequiringRuntime (typeName: string) : bool =
    bclTypesRequiringRuntime |> List.exists (fun bcl ->
        typeName.StartsWith(bcl) || typeName = bcl)

/// Validate a type name from the PSG
/// Returns Some error if the type is BCL/requires runtime
let private validateTypeName (nodeId: string) (syntaxKind: string) (typeName: string) : NativeTypeValidationError option =
    // Skip empty or null type names
    if System.String.IsNullOrWhiteSpace(typeName) then
        None
    // First check if it's explicitly native-compatible
    elif isNativeCompatibleType typeName then
        None
    // Then check if it's a known BCL type
    elif isBclTypeRequiringRuntime typeName then
        Some {
            NodeId = nodeId
            SyntaxKind = syntaxKind
            TypeName = Some typeName
            SymbolName = None
            Message = sprintf "BCL TYPE IN PSG: '%s' requires .NET runtime. Type resolution should map to Alloy native types." typeName
        }
    // Unknown types - not in allow list and not in deny list
    // These need investigation but are not hard errors yet
    else
        None

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
            NodeId = nodeId
            SyntaxKind = syntaxKind
            TypeName = None
            SymbolName = Some fullName
            Message = sprintf "NON-NATIVE SYMBOL: '%s' requires .NET runtime. Alloy must provide a native implementation." fullName
        }
    else
        None

/// Recursively collect all type names from an FSharpType
/// This handles function types, generic types, arrays, etc.
let rec private collectTypeNames (fsharpType: FSharpType) : string list =
    try
        if fsharpType.IsFunctionType then
            // Function type: collect from both argument and return types
            let argTypes = fsharpType.GenericArguments |> Seq.toList
            argTypes |> List.collect collectTypeNames
        elif fsharpType.IsGenericParameter then
            // Generic parameter like 'T - no concrete type
            []
        elif fsharpType.IsTupleType then
            // Tuple: collect from all element types
            fsharpType.GenericArguments |> Seq.toList |> List.collect collectTypeNames
        elif fsharpType.HasTypeDefinition then
            let baseName = fsharpType.TypeDefinition.FullName
            // Also check generic arguments
            let genericNames =
                if fsharpType.GenericArguments.Count > 0 then
                    fsharpType.GenericArguments |> Seq.toList |> List.collect collectTypeNames
                else
                    []
            baseName :: genericNames
        else
            // Fallback to formatted name
            [fsharpType.Format(FSharp.Compiler.Symbols.FSharpDisplayContext.Empty)]
    with _ ->
        [fsharpType.Format(FSharp.Compiler.Symbols.FSharpDisplayContext.Empty)]

/// Extract the primary type name from an FSharpType (for error messages)
let private getTypeName (fsharpType: FSharpType) : string =
    try
        fsharpType.Format(FSharp.Compiler.Symbols.FSharpDisplayContext.Empty)
    with _ ->
        "?"

/// Validate all aspects of a PSG node (both Type and Symbol)
let private validateNode (node: PSGNode) : NativeTypeValidationError list =
    let errors = ResizeArray<NativeTypeValidationError>()

    // DEBUG: Check if reachable nodes have Type = None
    if node.IsReachable && node.Type.IsNone then
        // Only log first few
        ()

    // Validate Type field (FSharpType) - check ALL types in the type expression
    match node.Type with
    | Some fsharpType ->
        let allTypeNames = collectTypeNames fsharpType
        // DEBUG: Print what we're checking for String types
        if node.IsReachable && allTypeNames |> List.exists (fun t -> t.Contains("String") || t.Contains("string")) then
            eprintfn "[TYPE-VAL] Node %s (reachable=%b)" node.Id.Value node.IsReachable
            eprintfn "           Types: %A" allTypeNames
            // Check if validation catches it
            for typeName in allTypeNames do
                let isBcl = isBclTypeRequiringRuntime typeName
                let isNative = isNativeCompatibleType typeName
                eprintfn "           -> '%s' isBCL=%b isNative=%b" typeName isBcl isNative
        for typeName in allTypeNames do
            match validateTypeName node.Id.Value (SyntaxKindT.toString node.Kind) typeName with
            | Some err -> errors.Add(err)
            | None -> ()
    | None -> ()

    // Validate Symbol
    match node.Symbol with
    | Some symbol ->
        match validateSymbol node.Id.Value (SyntaxKindT.toString node.Kind) symbol with
        | Some err -> errors.Add(err)
        | None -> ()
    | None -> ()

    List.ofSeq errors

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
    let reachableNodes =
        graph.Nodes
        |> Map.toList
        |> List.filter (fun (_, node) -> node.IsReachable)

    // DEBUG: Count nodes with Type populated
    let withType = reachableNodes |> List.filter (fun (_, n) -> n.Type.IsSome) |> List.length
    let total = List.length reachableNodes
    System.Console.Error.WriteLine(sprintf "[TYPE-VAL] Reachable nodes: %d, with Type: %d" total withType)
    System.Console.Error.Flush()

    let errors =
        reachableNodes
        |> List.collect (fun (_, node) -> validateNode node)

    { Errors = errors; HasErrors = not (List.isEmpty errors) }

// ═══════════════════════════════════════════════════════════════════
// EARLY VALIDATION (runs on FSharpCheckProjectResults, before PSG)
// ═══════════════════════════════════════════════════════════════════
// This validation runs IMMEDIATELY after FCS type-checking, scanning
// the typed tree for BCL types. Much faster than waiting for PSG.

open FSharp.Compiler.CodeAnalysis

/// Early validation error (before PSG exists)
type EarlyValidationError = {
    File: string
    Line: int
    Column: int
    TypeName: string
    Message: string
}

/// Exception thrown when BCL type is detected - FAIL FAST
exception BclTypeDetectedException of file: string * line: int * column: int * typeName: string

/// Scan typed tree for BCL types - FAIL FAST on first error
let private scanTypedTreeForBclTypes (checkResults: FSharpCheckProjectResults) : unit =
    // Get all symbol uses from the project
    let symbolUses = checkResults.GetAllUsesOfAllSymbols()

    for symbolUse in symbolUses do
        let symbol = symbolUse.Symbol
        match symbol with
        | :? FSharpMemberOrFunctionOrValue as mfv ->
            try
                let fullType = mfv.FullType
                let typeNames = collectTypeNames fullType
                for typeName in typeNames do
                    if isBclTypeRequiringRuntime typeName then
                        // FAIL FAST - raise exception immediately
                        raise (BclTypeDetectedException(
                            symbolUse.FileName,
                            symbolUse.Range.StartLine,
                            symbolUse.Range.StartColumn,
                            typeName))
            with
            | :? BclTypeDetectedException -> reraise()
            | _ -> ()
        | _ -> ()

/// Run early validation on FCS results - FAILS FAST with exception
/// Returns None if valid, or throws BclTypeDetectedException
let validateEarlyFailFast (checkResults: FSharpCheckProjectResults) : unit =
    scanTypedTreeForBclTypes checkResults
