# The Tale of Two Identifiers: Navigating F# Compiler Services' Dual Identity Crisis

*A deep dive into why your `List.map` works in some places but not others when processing F# AST identifiers*

## The Discovery

While building the Firefly compiler (an F# to native compiler via MLIR), I encountered a puzzling type error that revealed an interesting design quirk in the F# Compiler Services (FCS). The same pattern for extracting identifier names worked perfectly in some AST nodes but failed in others. The culprit? FCS uses two different representations for multi-part identifiers depending on the context.

## The Two Faces of Long Identifiers

F# Compiler Services provides two types for representing dotted identifiers like `System.Collections.Generic`:

### 1. LongIdent - The Simple List
```fsharp
type LongIdent = Ident list  // Just a type alias!
```

This is straightforward - it's simply a list of `Ident` records. You can directly map over it:

```fsharp
// Works perfectly!
longId |> List.map (fun id -> id.idText) |> String.concat "."
```

### 2. SynLongIdent - The Rich Wrapper
```fsharp
type SynLongIdent = 
    | SynLongIdent of id: LongIdent * dotRanges: range list * trivia: IdentTrivia option list
```

This type extension preserves source formatting information. It wraps the actual identifier list with metadata about dot positions and formatting trivia (like backticks or parentheses). You need to destructure it first:

```fsharp
// Must pattern match to extract the inner list!
match synLongIdent with
| SynLongIdent(identList, _, _) ->
    identList |> List.map (fun id -> id.idText) |> String.concat "."
```

## Where Each Type Appears

Through debugging Firefly's ingestion pipeline, I mapped out where each type appears:

### Uses Plain `LongIdent`:
- `SynModuleOrNamespace` - module/namespace names
- `SynComponentInfo` - type and module component names  
- `SynTypeDefn` - type definition names

### Uses Rich `SynLongIdent`:
- `SynOpenDeclTarget.ModuleOrNamespace` - open declarations
- `SynExpr.LongIdent` - long identifier expressions
- `SynPat.LongIdent` - long identifier patterns
- Most expression and pattern contexts

## Why This Design?

This dual approach serves specific purposes:

1. **Performance**: Simple contexts that don't need formatting preservation use the lightweight `LongIdent`
2. **Source Fidelity**: Contexts where exact source representation matters use `SynLongIdent`
3. **Historical Evolution**: The API evolved over time, with `SynLongIdent` (formerly `LongIdentWithDots`) added later for better tooling support

## Practical Impact

This design has real implications for AST processing tools:

```fsharp
// Processing different declaration types requires different patterns
match declaration with
| SynModuleDecl.NestedModule(componentInfo, _, _, _, _, _) ->
    match componentInfo with
    | SynComponentInfo(_, _, _, longId, _, _, _, _) ->
        // longId is LongIdent - direct list operations work
        longId |> List.map (fun id -> id.idText) |> String.concat "."

| SynModuleDecl.Open(target, _) ->
    match target with
    | SynOpenDeclTarget.ModuleOrNamespace(longId, _) ->
        // longId is SynLongIdent - must destructure first!
        match longId with
        | SynLongIdent(identList, _, _) ->
            identList |> List.map (fun id -> id.idText) |> String.concat "."
```

## Best Practices

When working with FCS 43.9.300 and later:

1. **Always check the exact type** - Don't assume all `longId` parameters are the same type
2. **Use pattern matching** - Destructure `SynLongIdent` inline when possible:
   ```fsharp
   | SynOpenDeclTarget.ModuleOrNamespace(SynLongIdent(ids, _, _), _) ->
       ids |> List.map (fun id -> id.idText) |> String.concat "."
   ```
3. **Preserve trivia when needed** - If building a formatter or refactoring tool, don't discard the dot ranges and trivia
4. **Consider the context** - Expression/pattern contexts almost always use the rich types

## Conclusion

This inconsistency in the FCS API initially seems like a frustration, but it actually reflects thoughtful design decisions about when source formatting matters. For the Firefly compiler, which aims to generate comprehensive debug information, understanding these nuances is crucial for accurately representing F# code structure throughout the compilation pipeline.

The lesson? When working with compiler APIs, the devil is in the type details. What looks like the same conceptual thing (a dotted identifier) may have different representations based on compilation phase and fidelity requirements.

---

*This discovery was made while building [Firefly](https://github.com/example/firefly), an experimental F# to native compiler using MLIR. Special thanks to the F# Compiler Services team for their comprehensive (if occasionally surprising) AST design.*
