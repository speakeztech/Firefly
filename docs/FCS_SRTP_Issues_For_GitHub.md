# F# Compiler Service Issues for GitHub Discussion

This document outlines two issues with FCS that affect AOT compilation scenarios where SRTP (Statically Resolved Type Parameters) need to be resolved at compile time.

## Issue 1: TraitConstraintInfo.Solution Not Exposed in Public API

### Summary

The `FSharpExprPatterns.TraitCall` active pattern exposes trait call information but does NOT expose the resolved implementation (the `Solution` property of `TraitConstraintInfo`).

### Background

When using SRTP in F# code like:

```fsharp
type WritableString =
    | WritableString
    static member inline ($) (WritableString, s: string) = writeString s

let inline Write s = WritableString $ s
```

The `$` operator is an SRTP-dispatched call. At compile time, FCS resolves which implementation to use based on the argument types.

### Current Behavior

The public API `FSharpExprPatterns.TraitCall` provides:
- `sourceTypes: FSharpType list` - The types that support the trait
- `traitName: string` - The member name (e.g., "op_Dollar")
- `memberFlags: SynMemberFlags`
- `paramTypes: FSharpType list`
- `retTypes: FSharpType list`
- `traitArgs: FSharpExpr list`

**Missing**: The resolved implementation (which concrete method was selected).

### Internal State

Internally, `FSharp.Compiler.TypedTree.TraitConstraintInfo` has a `Solution` property of type `FSharpOption<TraitConstraintSln>` which can be:
- `FSMethSln` - Resolved to an F# method
- `FSRecdFieldSln` - Resolved to a record field
- `FSAnonRecdFieldSln` - Resolved to an anonymous record field
- `ILMethSln` - Resolved to an IL method
- `ClosedExprSln` - Resolved to a closed expression
- `BuiltInSln` - Resolved to a built-in operator

### Requested Enhancement

Expose the `Solution` property in the public API, either:
1. Add `solution: TraitConstraintSolution option` to the `TraitCall` active pattern
2. Add a new active pattern `TraitCallResolution` that includes the solution
3. Add a method to `FSharpExpr` to get SRTP resolution info

### Use Case

AOT compilers (like Firefly for native compilation) need to know the resolved implementation to:
1. Generate the correct call target
2. Inline the resolved implementation
3. Follow the call graph for reachability analysis

### Workaround

We can infer the resolution by looking up members on the first source type:
```fsharp
match sourceTypes with
| firstType :: _ when firstType.HasTypeDefinition ->
    let entity = firstType.TypeDefinition
    entity.MembersFunctionsAndValues
    |> Seq.tryFind (fun m -> m.LogicalName = traitName)
```

However, this is fragile and doesn't handle all cases (extension methods, inherited members, etc.).

---

## Issue 2: Symbol Correlation for Static Member Definitions

### Summary

When using `GetSymbolUseAtLocation` or iterating `FSharpCheckFileResults.GetSymbolUsesAtLocation`, static member DEFINITIONS are not included in the symbol uses. Only USE sites are tracked.

### Example

```fsharp
type WritableString =
    | WritableString
    static member inline ($) (WritableString, s: string) = writeString s
                        ^^^--- Definition NOT included in symbol uses
```

### Impact

When building a PSG (Program Semantic Graph) or any AST representation that needs to correlate syntax nodes with semantic symbols, static member definitions cannot be correlated with their `FSharpMemberOrFunctionOrValue` symbol.

This results in:
```
[BUILDER] Warning: Pattern 'op_Dollar' at (242,29--242,63) has no symbol correlation
```

### Requested Enhancement

Include definition locations in symbol uses for:
1. Static members on DUs
2. Static members on records
3. Instance members defined via explicit `member` declarations

### Current Workaround

Pattern-match on the `SyntaxKind` of child nodes:
```fsharp
// Look for bindings whose child pattern contains the method name
psg.Nodes
|> Seq.tryFind (fun n ->
    n.SyntaxKind.StartsWith("Binding") &&
    match n.Children with
    | Parent childIds ->
        childIds |> List.exists (fun childId ->
            patNode.SyntaxKind.Contains("op_Dollar"))
    | _ -> false)
```

This is fragile because it relies on syntax structure rather than semantic information.

---

## Context

These issues were discovered while developing [Firefly](https://github.com/speakeztech/Firefly), an AOT F# compiler targeting native binaries without the .NET runtime. The compiler uses FCS for parsing and type checking, then builds a semantic graph for code generation.

For SRTP specifically, we need to:
1. Identify all SRTP calls
2. Determine the resolved implementation for each
3. Follow the call graph through the resolved implementations
4. Generate target code for the resolved calls

The current FCS API makes step 2 difficult and step 3 impossible without workarounds.
