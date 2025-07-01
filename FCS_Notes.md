# FSharp.Compiler.Service 43.9.300 API Reference Guide

FSharp.Compiler.Service version 43.9.300, released as part of F# 9.0, introduces significant AST enhancements with improved trivia support and structural changes for better tooling experiences. This guide provides the exact constructor signatures and usage patterns for key AST components.

## Constructor signatures for core AST types

The following sections detail the precise constructor signatures with complete type information for each requested AST component in version 43.9.300.

### SynPat.Named constructor requires four arguments

```fsharp
SynPat.Named: ident: SynIdent * isThisVal: bool * accessibility: SynAccess option * range: range -> SynPat
```

The Named constructor takes a **SynIdent** (not plain Ident) as its first parameter, followed by a boolean indicating if this represents a 'this' variable, an optional access modifier, and a source range. The key change in 43.9.300 is the use of SynIdent instead of Ident, which preserves formatting trivia:

```fsharp
let namedPattern = SynPat.Named(
    SynIdent(Ident.Create("x"), None),  // SynIdent with optional trivia
    false,                              // not 'this' variable
    None,                               // no access modifier
    range.Zero                          // source location
)
```

### SynPat.LongIdent takes six parameters with enhanced structure

```fsharp
SynPat.LongIdent: longDotId: SynLongIdent * extraId: Ident option * typarDecls: SynValTyparDecls option * argPats: SynArgPats * accessibility: SynAccess option * range: range -> SynPat
```

This constructor handles qualified identifiers and constructor patterns. The **SynLongIdent** type (renamed from LongIdentWithDots) includes dot positions and trivia information:

```fsharp
let longIdentPattern = SynPat.LongIdent(
    SynLongIdent([Ident.Create("Some")], [], [None]),  // identifier, dot ranges, trivia
    None,                                               // extra identifier
    None,                                               // type parameters
    SynArgPats.Pats([SynPat.Wild(range.Zero)]),       // argument patterns
    None,                                               // accessibility
    range.Zero                                          // range
)
```

### SynSimplePats vs SynPat serve different compilation phases

**SynSimplePats** represents simplified, flattened patterns used internally by the compiler after initial parsing. It has two cases: SimplePats and Typed. In contrast, **SynPat** preserves the full source structure with complete trivia information.

Version 43.9.300 made significant changes to their interaction:
- `SyntaxVisitorBase<'T>.VisitSimplePats` now takes `SynPat` instead of `SynSimplePat list`
- `implicitCtorSynPats` in type definitions changed from `SynSimplePats option` to `SynPat option`

For AST manipulation tools, always prefer **SynPat** as it maintains source fidelity. Use SynSimplePats only when working with compiler-pruned representations in lambda expressions.

### LongIdent is a simple list while SynLongIdent preserves formatting

**LongIdent** is a type alias for `Ident list`, providing basic dotted identifier representation. **SynLongIdent** (formerly LongIdentWithDots) is a full AST node:

```fsharp
type LongIdent = Ident list
type SynLongIdent = SynLongIdent of id: LongIdent * dotRanges: range list * trivia: IdentTrivia option list
```

Use **LongIdent** for simple identifier operations where formatting doesn't matter. Use **SynLongIdent** in all AST constructors that require preserving source formatting, which includes most syntax tree operations in 43.9.300.

### SynIdent wraps Ident with trivia for source preservation

**Ident** provides basic identifier representation:
```fsharp
type Ident = { idText: string; idRange: range }
```

**SynIdent** enhances this with trivia support:
```fsharp
type SynIdent = SynIdent of ident: Ident * trivia: IdentTrivia option
```

The trivia preserves formatting information like parentheses, backticks, and other source-level details. In 43.9.300, most AST constructors require **SynIdent** rather than plain Ident to maintain formatting fidelity.

### ParsedInput handling requires pattern matching on file type

Working with `FSharpParseFileResults.ParseTree` involves matching on the ParsedInput type:

```fsharp
let processParseTree (parseResults: FSharpParseFileResults) =
    match parseResults.ParseTree with
    | ParsedInput.ImplFile(implFile) ->
        let (ParsedImplFileInput(contents = modules)) = implFile
        // Process F# implementation file (.fs, .fsx)
        modules |> List.iter processModule
    | ParsedInput.SigFile(sigFile) ->
        let (ParsedSigFileInput(contents = modules)) = sigFile
        // Process F# signature file (.fsi)
        modules |> List.iter processSignatureModule
```

The ParsedInput structure provides access to the complete AST, with modules containing declarations that can be recursively processed.

### SynExpr.Lambda includes parsedData for original pattern preservation

```fsharp
SynExpr.Lambda: fromMethod: bool * inLambdaSeq: bool * args: SynSimplePats * body: SynExpr * parsedData: (SynPat list * SynExpr) option * range: range * trivia: SynExprLambdaTrivia -> SynExpr
```

The **parsedData** field, new in 43.9.300, preserves the original parsed patterns before compiler transformations. This is crucial for tooling that needs source-accurate representations:

```fsharp
let lambdaExpr = SynExpr.Lambda(
    false,                          // not from method
    false,                          // not in lambda sequence
    SynSimplePats.SimplePats(       // simplified args
        [SynSimplePat.Id(Ident.Create("x"))], 
        range.Zero
    ),
    SynExpr.Ident(Ident.Create("x")),  // body
    Some(                              // original parsed data
        [SynPat.Named(SynIdent(Ident.Create("x"), None), false, None, range.Zero)], 
        SynExpr.Ident(Ident.Create("x"))
    ),
    range.Zero,                        // range
    { ArrowRange = Some range.Zero }   // trivia with arrow position
)
```

### SynUnionCase constructor supports full discriminated union cases

```fsharp
SynUnionCase: attributes: SynAttributes * ident: SynIdent * caseType: SynUnionCaseKind * xmlDoc: PreXmlDoc * accessibility: SynAccess option * range: range * trivia: SynUnionCaseTrivia -> SynUnionCase
```

This constructor handles discriminated union cases with attributes, documentation, and accessibility modifiers. The **SynUnionCaseKind** can be either Fields (for simple cases) or FullType (for cases with explicit type annotations).

### SynField defines record and class field structures

```fsharp
SynField: attributes: SynAttributes * isStatic: bool * idOpt: Ident option * fieldType: SynType * isMutable: bool * xmlDoc: PreXmlDoc * accessibility: SynAccess option * range: range * trivia: SynFieldTrivia -> SynField
```

The constructor supports both record fields and class fields, with options for mutability, static fields, and accessibility. Note that **idOpt** uses plain Ident rather than SynIdent, as field names don't typically require trivia preservation.

## Migration strategies and best practices

When upgrading to 43.9.300, the most significant changes involve namespace reorganization and enhanced trivia support. Code previously using `FSharp.Compiler.SourceCodeServices` should now import `FSharp.Compiler.CodeAnalysis`. The `LongIdentWithDots` type has been renamed to `SynLongIdent` with enhanced capabilities.

For lambda expressions, always check for **parsedData** when available, as it preserves the original source structure before compiler transformations. This is particularly important for formatting tools and linters that need to maintain source fidelity.

The removal of `CompileToDynamicAssembly` requires using `FSharpChecker.Compile` with appropriate output handling. Most AST constructors now include trivia parameters that should be properly initialized to maintain formatting information.

When pattern matching on AST nodes, handle all new fields appropriately, especially trivia and parsedData fields. Tools that traverse the AST should be updated to process these additional fields to maintain full source accuracy.

## Practical implementation patterns

For effective AST manipulation in 43.9.300, establish clear patterns for handling the enhanced type system. When creating AST nodes programmatically, always use SynIdent instead of plain Ident for identifiers that appear in patterns or expressions. Initialize trivia fields appropriately, even if just with default values.

The enhanced lambda representation with parsedData enables more accurate source transformations. Tools should prefer the parsed data when available, falling back to simplified patterns only when necessary. This dual representation supports both compiler optimization needs and tooling accuracy requirements.

Integration with the F# compiler service now requires careful attention to the multi-layered identifier types. Use the appropriate level of abstraction for your use case: Ident for basic operations, SynIdent for AST construction, and full pattern types for comprehensive source analysis.