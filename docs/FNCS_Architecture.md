# FNCS: F# Native Compiler Services

## Executive Summary

FNCS (F# Native Compiler Services) is a minimal, surgical fork of F# Compiler Services (FCS) that provides native-first type resolution for the Fidelity framework. Rather than fighting FCS's BCL-centric assumptions downstream in Firefly, FNCS addresses the fundamental problem at its source: the type system itself.

FNCS is **not** a full rewrite of FCS. It is a targeted modification that:

1. Defines native primitive types (`NativeStr`, `NativePtr`, etc.) as first-class citizens
2. Types string literals as `NativeStr` instead of `System.String`
3. Resolves SRTP against a native witness hierarchy
4. Enforces null-free semantics where BCL would allow nulls
5. Maintains the same API surface as FCS for seamless Firefly integration

## The Problem: FCS's BCL-Centric Type Universe

FCS was designed for .NET compilation. Its type system assumes BCL primitives:

```fsharp
// In FCS CheckExpressions.fs, line ~7342
| false, LiteralArgumentType.Inline ->
    TcPropagatingExprLeafThenConvert cenv overallTy g.string_ty env m (fun () ->
        mkString g m s, tpenv)
```

When you write `"Hello"` in F# code, FCS **always** types it as `Microsoft.FSharp.Core.string` (which is `System.String`). This is hardcoded. No amount of type shadows, namespace tricks, or downstream transformations can change this fundamental behavior.

### Why Type Shadows Don't Work

Alloy attempted to shadow BCL types:

```fsharp
// In Alloy - this was the attempted solution
type string = NativeStr
```

But type shadows only affect **type annotations**, not **literal inference**:

```fsharp
let x: string = value  // Shadow works here: x is NativeStr
let y = "Hello"        // Shadow IGNORED: y is System.String from FCS
```

This asymmetry creates an impossible situation. User code looks correct (`Console.Write "Hello"`), but FCS has already decided that `"Hello"` is BCL string before any Alloy code runs.

### The Downstream Consequence

Because FCS outputs BCL types, every downstream component must either:

1. **Reject BCL types** - Firefly's `ValidateNativeTypes` does this, but then valid-looking code fails
2. **Transform BCL types** - Violates "no lowering" principle, creates semantic mismatches
3. **Accept BCL types** - Requires BCL runtime, defeats purpose of native compilation

None of these options are acceptable. The fix must happen at the source: the type system.

## The Solution: Native Type Universe in FNCS

FNCS modifies FCS to use a **native type universe** where Fidelity's primitives are first-class:

```
┌─────────────────────────────────────────────────────────────────────┐
│  FCS Type Universe (BCL-centric)                                    │
│                                                                     │
│  g.string_ty    → System.String                                     │
│  g.int32_ty     → System.Int32                                      │
│  g.option_ty    → Microsoft.FSharp.Core.option<T>  (nullable!)      │
│  String literal → System.String (hardcoded)                         │
│  SRTP           → Searches BCL method tables                        │
└─────────────────────────────────────────────────────────────────────┘

                              ↓ FNCS Fork ↓

┌─────────────────────────────────────────────────────────────────────┐
│  FNCS Type Universe (Native-centric)                                │
│                                                                     │
│  g.string_ty    → NativeStr                                         │
│  g.int32_ty     → int32  (same, but with native semantics)          │
│  g.option_ty    → voption<T>  (never null, stack-allocated)         │
│  String literal → NativeStr (fat pointer: ptr + length)             │
│  SRTP           → Searches native witness hierarchy                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Architectural Layering with FNCS

FNCS changes the fundamental layering of the Fidelity stack:

### Before (Current - FCS)

```
User Code: Console.Write "Hello"
    ↓
FCS: Types "Hello" as System.String (BCL)
FCS: SRTP searches BCL method tables
    ↓
Firefly/Baker: Receives BCL-typed tree
Firefly: ValidateNativeTypes FAILS (BCL detected)
    ↓
ERROR: BCL types in native compilation
```

### After (FNCS)

```
User Code: Console.Write "Hello"
    ↓
FNCS: Types "Hello" as NativeStr (native primitive)
FNCS: SRTP searches native witness hierarchy
    ↓
Firefly/Baker: Receives native-typed tree
Firefly: ValidateNativeTypes PASSES (only native types)
    ↓
Alex: Generates MLIR directly
    ↓
Native binary
```

## What FNCS Contains

FNCS is a focused modification. It contains:

### 1. Native Primitive Type Definitions

The core type universe redefinitions:

```fsharp
// Conceptual - actual implementation in TcGlobals.fs
type FNCSGlobals = {
    // String is NativeStr, not System.String
    string_ty: NativeStr  // {Pointer: nativeptr<byte>, Length: int}

    // Option is voption, not heap-allocated option
    option_tcr: voption<'T>  // struct, never null

    // Native pointers are first-class
    nativeptr_tcr: nativeptr<'T>
    nativeint_tcr: nativeint

    // Spans for memory views
    span_tcr: NativeSpan<'T>

    // Arrays are native
    array_tcr: NativeArray<'T>
}
```

### 2. String Literal Type Resolution

The key modification in `CheckExpressions.fs`:

```fsharp
// FNCS modification - string literals type as NativeStr
| false, LiteralArgumentType.Inline ->
    TcPropagatingExprLeafThenConvert cenv overallTy g.nativestr_ty env m (fun () ->
        mkNativeString g m s, tpenv)  // Creates NativeStr, not System.String
```

### 3. Native SRTP Witness Resolution

SRTP resolution searches native witnesses instead of BCL method tables:

```fsharp
// FNCS SRTP resolution
let resolveTraitCall (traitInfo: TraitConstraintInfo) =
    // Search native witness hierarchy, not BCL
    match traitInfo.MemberName with
    | "op_Addition" -> searchNativeWitness NumericOps.Add
    | "op_Dollar" -> searchNativeWitness WritableString.op_Dollar
    | "get_Length" -> searchNativeWitness Measurable.Length
    // ... native witnesses for all SRTP-resolvable operations
```

### 4. Null-Free Semantics

FNCS enforces null-free semantics where BCL would allow nulls:

```fsharp
// FNCS null handling
let checkNullAssignment targetType sourceExpr =
    match targetType with
    | NativeStr | NativeArray _ | voption _ ->
        // These types are NEVER null - error if null assigned
        if isNullLiteral sourceExpr then
            error "Native types cannot be null"
    | _ -> ()
```

### 5. BCL-Sympathetic API Surface

Despite native semantics, the API surface remains familiar:

```fsharp
// User code looks exactly like BCL F#
module Console =
    let Write (s: string) = ...      // string = NativeStr in FNCS
    let WriteLine (s: string) = ...
    let ReadLine () : string = ...

module String =
    let length (s: string) : int = ...
    let concat (sep: string) (strings: string seq) : string = ...
```

## What FNCS Does NOT Contain

FNCS is intentionally minimal. It does NOT include:

### Platform Bindings

Platform-specific operations remain in Firefly/Alex:

```fsharp
// NOT in FNCS - stays in Alloy/Firefly
module Platform.Bindings =
    let writeBytes fd buffer count : int = ...
    let readBytes fd buffer maxCount : int = ...
```

FNCS defines types; Firefly implements operations on those types.

### Runtime Implementations

FNCS provides type resolution, not runtime code:

```fsharp
// NOT in FNCS - runtime implementation in Alloy
let inline concat2 (dest: nativeptr<byte>) (s1: NativeStr) (s2: NativeStr) : NativeStr =
    // Actual byte-copying implementation
    ...
```

### Code Generation

FNCS produces typed trees. Code generation is Firefly/Alex's domain:

```fsharp
// NOT in FNCS - stays in Alex
let emitNativeStr (ctx: EmissionContext) (str: NativeStr) : MLIR =
    // Place bytes in data section, emit struct construction
    ...
```

## Integration with Firefly

FNCS produces **FSharp.Native.Compiler.Service.dll** - a distinct library with native-first type resolution.

### Repository Structure

```
dotnet/fsharp (upstream reference)
    │
    └──→ SpeakEZ/fsnative (pure divergence)
              │
              ├── main          ← Initial fork point (reference only)
              │
              └── fsnative      ← Active development branch
                                   All FNCS modifications here
```

**Key decisions:**
- **Pure divergence** - No ongoing merge from upstream. `dotnet/fsharp` is reference only.
- **Distinct identity** - Output is `FSharp.Native.Compiler.Service.dll`, not a drop-in replacement
- **Companion spec** - `SpeakEZ/fsnative-spec` documents the native F# dialect

### Current (FCS from NuGet)

```xml
<!-- Firefly.fsproj -->
<PackageReference Include="FSharp.Compiler.Service" Version="43.8.x" />
```

### With FNCS (Local Build)

```xml
<!-- Firefly.fsproj -->
<Reference Include="FSharp.Native.Compiler.Service">
  <HintPath>../fsnative/artifacts/bin/FSharp.Native.Compiler.Service/Release/netstandard2.0/FSharp.Native.Compiler.Service.dll</HintPath>
</Reference>
```

### API Compatibility

FNCS maintains API compatibility with FCS. Existing Firefly code requires minimal changes:

```fsharp
// Change: open FSharp.Compiler.* → open FSharp.Native.Compiler.*
open FSharp.Native.Compiler.CodeAnalysis
open FSharp.Native.Compiler.Symbols

let checker = FSharpChecker.Create()
let results = checker.ParseAndCheckFileInProject(...)
let typedTree = results.TypedTree  // Contains NativeStr instead of System.String
```

The API shape is the same; the namespace and output types differ.

## The Resulting Layer Separation

With FNCS, the Fidelity stack has clean layer separation:

| Layer | Responsibility |
|-------|---------------|
| **FNCS** | Type universe, literal typing, SRTP resolution, null-free semantics |
| **Alloy** | Library implementations using FNCS types (Console, String, etc.) |
| **Firefly/PSG** | Semantic graph construction from FNCS output |
| **Firefly/Baker** | Type overlay, SRTP extraction (now trivial - FNCS already resolved) |
| **Firefly/Alex** | Platform-aware MLIR generation |
| **Platform Bindings** | Syscall implementations per platform |

Each layer has a single responsibility. No layer needs to "work around" another layer's assumptions.

## Impact on Existing Components

### ValidateNativeTypes

With FNCS, `ValidateNativeTypes` becomes simpler:

```fsharp
// Before: Complex classification, many edge cases
// After: Simple - FNCS guarantees native types
let validateNode (node: PSGNode) =
    // FNCS already ensured all types are native
    // This pass becomes a sanity check, not a gatekeeper
    ()
```

### Baker/TypedTreeZipper

Baker's job becomes easier:

```fsharp
// Before: Extract types, handle BCL/native mismatches
// After: Types are already correct from FNCS
let overlayTypes (node: PSGNode) (fsharpExpr: FSharpExpr) =
    // Types from fsharpExpr are already native
    // No translation needed
    node.Type <- fsharpExpr.Type
```

### Alloy Type Shadows

Type shadows become unnecessary:

```fsharp
// Before: Attempted shadowing (didn't work for literals)
type string = NativeStr

// After: Not needed - FNCS string_ty IS NativeStr
// Alloy just uses the types FNCS provides
```

### Console.Write and SRTP

SRTP resolution becomes straightforward:

```fsharp
// Before: SRTP resolved against BCL, then we tried to redirect
// After: SRTP resolves against native witnesses directly

type WritableString =
    | WritableString
    static member inline ($) (WritableString, s: NativeStr) = writeNativeStr s
    // No BCL string overload needed - FNCS never produces BCL string
```

## Implementation Roadmap

### Phase 0: Repository Setup (DONE)

- **`SpeakEZ/fsnative`** - Fork of `dotnet/fsharp` on GitHub
- **`SpeakEZ/fsnative-spec`** - Fork of `fsharp/fslang-spec` on GitHub
- **Branch**: `fsnative` for all development (main frozen at fork point)
- **Pure divergence** - `dotnet/fsharp` is reference only, kept at `~/repos/fsharp`

### Phase 1: Assembly Identity and Build Infrastructure

1. Rename assembly to `FSharp.Native.Compiler.Service`
2. Update namespaces: `FSharp.Compiler.*` → `FSharp.Native.Compiler.*`
3. Verify build produces correctly named DLL
4. Create test harness for validating type resolution

### Phase 2: Native Type Definitions

1. Define `NativeStr` type in FNCS type system
2. Define `voption<'T>` as the default option type
3. Define `NativeArray<'T>`, `NativeSpan<'T>`, `NativePtr<'T>`
4. Wire these into `TcGlobals.fs`

### Phase 3: Literal Type Resolution

1. Modify `TcConstStringExpr` to produce `NativeStr`
2. Update string literal handling throughout checker
3. Ensure string operations type-check against `NativeStr`

### Phase 4: SRTP Native Witnesses

1. Define native witness hierarchy in FNCS
2. Modify constraint solver to search native witnesses
3. Ensure common operations (`+`, `$`, `Length`, etc.) resolve correctly

### Phase 5: Null-Free Enforcement

1. Add null checks where BCL would allow null
2. Emit errors for null assignments to native types
3. Ensure `voption` is used instead of nullable `option`

### Phase 6: Integration and Testing

1. Update Firefly to reference FNCS DLL
2. Verify HelloWorld samples compile
3. Run full test suite
4. Document any API differences

## FCS Files Requiring Modification

Based on the "From Bridged to Self Hosted" analysis and FCS structure:

| File | Modification |
|------|-------------|
| `src/Compiler/Checking/TcGlobals.fs` | Native type definitions |
| `src/Compiler/Checking/CheckExpressions.fs` | String literal typing (~line 7342) |
| `src/Compiler/Checking/ConstraintSolver.fs` | SRTP native witness resolution |
| `src/Compiler/TypedTree/TypedTree.fs` | Native type representations |
| `src/Compiler/TypedTree/TypedTreeBasics.fs` | `mkNativeString` helper |

The modification surface is intentionally small - surgical changes to type resolution, not a rewrite.

## Relationship to Self-Hosting Roadmap

FNCS is an **intermediate step** on the path to full self-hosting:

```
Current:     FCS (BCL) → Firefly → Native Binary
                ↓
FNCS:        FNCS (Native) → Firefly → Native Binary  ← WE ARE HERE
                ↓
Extracted:   Firefly.Syntax (Native) → Firefly → Native Binary
                ↓
Self-Hosted: Native Firefly → Native Firefly
```

FNCS provides immediate relief from BCL contamination while maintaining .NET tooling for Firefly itself. The full extraction (as described in "From Bridged to Self Hosted") remains the long-term goal, but FNCS unblocks current development.

## Success Criteria

FNCS is successful when:

1. `Console.Write "Hello"` compiles without BCL type errors
2. String literals in the typed tree are `NativeStr`, not `System.String`
3. SRTP resolves against native witnesses
4. `ValidateNativeTypes` passes without special cases
5. HelloWorld samples execute correctly
6. No changes required to Alloy namespace structure
7. No "lowering" or "transformation" passes needed in Firefly

## Conclusion

FNCS represents a pragmatic middle ground between fighting FCS's assumptions downstream and undertaking a full compiler extraction. By making targeted modifications to FCS's type universe, we get:

- **Correctness**: Types are right from the start
- **Simplicity**: No downstream workarounds
- **Compatibility**: Same FCS API, same Firefly integration
- **Progress**: Immediate unblocking of current development
- **Foundation**: Clear path to full self-hosting

The key insight is that the problem was never in Firefly, Baker, or Alloy - it was in FCS's fundamental assumption that all F# code targets .NET. FNCS corrects that assumption at the source.
