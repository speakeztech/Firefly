# XParsec as Foundation for PSG-to-Alex Transformation

## Problem Statement

The Firefly compiler needs a principled approach to PSG traversal and MLIR generation. The solution is:

1. **XParsec** (`XParsec` package) - The canonical parser combinator library
2. **PSGZipper** - Bidirectional tree traversal with context
3. **Bindings** - Platform-specific MLIR generation

The user directive is clear:
> "I REALLY REALLY REALLY want to standardize on XParsec... If there needs to be a more sophisticated tree traversing parser it should be compositionally built up from XParsec primitives."

## The Zipper: Non-Negotiable Core Component

**The PSGZipper is not optional. It is the traversal engine for the PSG.**

From "Hyping Hypergraphs":
> "The bidirectional zipper that traverses the hypergraph becomes a powerful tool for maintaining proofs during transformation. As it navigates through the graph, it carries both the current focus and the proof context."

From "Speed And Safety With Graph Coloring":
> "Our bidirectional zipper traversal enables sophisticated pattern matching across both control flow and data flow"

From "Proof-Aware Compilation":
> "The bidirectional zipper that traverses the hypergraph becomes a powerful tool for maintaining proofs during transformation... Forward navigation accumulates proof obligations... Backward navigation propagates postconditions... Lateral movement discovers related proofs... Context preservation ensures transformations maintain required properties"

The zipper provides:
- **Focus**: The current node being examined
- **Path**: The trail of "breadcrumbs" back to root (parent chain with sibling context)
- **Graph Context**: Access to the full PSG for node lookups
- **Generation State**: SSA counter, locals, types - threaded through traversal
- **Bidirectional Navigation**: up/down/left/right movement with context preservation

### PSGZipper Structure

```fsharp
type PSGPath =
    | Top
    | BindingChild of parent: PSGNode * childIndex: int * siblings: NodeId list * path: PSGPath
    | AppArg of func: PSGNode * argIndex: int * otherArgs: NodeId list * path: PSGPath
    | SequenceItem of preceding: NodeId list * following: NodeId list * parent: PSGNode * path: PSGPath
    | MatchCase of scrutinee: PSGNode * caseIndex: int * otherCases: NodeId list * path: PSGPath
    | Child of parent: PSGNode * childIndex: int * siblings: NodeId list * path: PSGPath

type PSGZipper = {
    Focus: PSGNode
    Path: PSGPath
    Graph: ProgramSemanticGraph
    State: GenerationState
}
```

### Zipper Operations

```fsharp
val down: PSGZipper -> NavResult<PSGZipper>      // Move to first child
val up: PSGZipper -> NavResult<PSGZipper>        // Move to parent
val left: PSGZipper -> NavResult<PSGZipper>      // Move to previous sibling
val right: PSGZipper -> NavResult<PSGZipper>     // Move to next sibling
val downTo: int -> PSGZipper -> NavResult<PSGZipper>  // Move to nth child
val children: PSGZipper -> NodeId list           // Get child IDs
val childNodes: PSGZipper -> PSGNode list        // Get child nodes
val ancestors: PSGZipper -> PSGNode list         // Get ancestor chain
```

## The Architectural Insight: XParsec Operates Within the Zipper

XParsec is designed for sequential input parsing. The PSG is a tree. But **at each zipper position, children ARE sequential**. This is the key insight.

**XParsec's job is NOT to traverse the tree. The zipper does that.**

**XParsec's job IS to provide principled combinators for:**
1. Matching patterns at the current focus
2. Matching patterns across the sequential list of children
3. Composing these matches monadically
4. Threading state and handling failure/backtracking

## Architecture: XParsec Layered on Zipper

### Layer 1: XParsec Integration Types

```fsharp
/// The "input" for XParsec when parsing PSG children
[<Struct>]
type PSGChildSlice(nodes: PSGNode array, start: int, length: int) =
    interface IReadable<PSGNode, PSGChildSlice> with
        member _.Item index = nodes.[start + int index]
        member _.TryItem index =
            let i = start + int index
            if i < start + length then ValueSome nodes.[i]
            else ValueNone
        member _.Length = int64 length
        member _.Slice(newStart, newLength) =
            PSGChildSlice(nodes, start + int newStart, int newLength)
        member _.SpanSlice(index, count) =
            ReadOnlySpan(nodes, start + int index, count)

/// State threaded through XParsec parsing
type PSGParseState = {
    /// The zipper provides navigation context
    Zipper: PSGZipper
    /// Additional context for generation
    AdditionalContext: Map<string, obj>
}

/// XParsec parser specialized for PSG child sequences
type PSGChildParser<'T> =
    XParsec.Parser<'T, PSGNode, PSGParseState, PSGChildSlice, PSGChildSlice>
```

### Layer 2: Bridging Zipper and XParsec

The zipper navigates the tree. When we want to match against children, we use XParsec:

```fsharp
module PSGXParsec =
    open XParsec
    open XParsec.Parsers
    open XParsec.Combinators

    /// Run an XParsec parser on the children of the current zipper focus
    let parseChildren (parser: PSGChildParser<'T>) (zipper: PSGZipper) : Result<'T, string> =
        let children = PSGZipper.childNodes zipper |> Array.ofList
        let input = PSGChildSlice(children, 0, children.Length)
        let state = { Zipper = zipper; AdditionalContext = Map.empty }
        let reader = XParsec.Reader(input, state, 0L)

        match parser reader with
        | Ok success -> Ok success.Parsed
        | Error err -> Error (sprintf "Parse failed at position %d" err.Position.Index)

    /// Match if current child satisfies predicate
    let satisfyChild (pred: PSGNode -> bool) : PSGChildParser<PSGNode> =
        XParsec.Parsers.satisfy pred

    /// Match child with specific syntax kind prefix
    let childKind (prefix: string) : PSGChildParser<PSGNode> =
        satisfyChild (fun n -> n.SyntaxKind.StartsWith(prefix))

    /// Skip a child and continue
    let skipChild : PSGChildParser<unit> =
        satisfyChild (fun _ -> true) >>% ()

    /// Get remaining children
    let remainingChildren : PSGChildParser<PSGNode list> =
        XParsec.Combinators.many (satisfyChild (fun _ -> true))
        |>> List.ofSeq
```

### Layer 3: Pattern Type Becomes XParsec-Based

Instead of `Pattern<'T> = PSGZipper -> 'T option`, patterns become:

```fsharp
/// A pattern that matches at the current zipper focus
/// Uses XParsec result type for consistency
type FocusPattern<'T> = PSGZipper -> XParsec.ParseResult<'T, PSGNode, PSGParseState>

/// A pattern that matches against children using XParsec
type ChildrenPattern<'T> = PSGChildParser<'T>

module FocusPattern =
    open XParsec

    /// Succeed with value
    let succeed (value: 'T) : FocusPattern<'T> =
        fun _ -> Ok { Parsed = value }

    /// Fail with message
    let fail (msg: string) : FocusPattern<'T> =
        fun z ->
            let state = { Zipper = z; AdditionalContext = Map.empty }
            Error { Position = { Id = ReaderId 0L; Index = 0L; State = state }
                    Errors = Message msg }

    /// Match if focus satisfies predicate
    let satisfies (pred: PSGNode -> bool) : FocusPattern<PSGNode> =
        fun z ->
            if pred z.Focus then Ok { Parsed = z.Focus }
            else fail "Focus does not satisfy predicate" z

    /// Match syntax kind prefix
    let syntaxKind (prefix: string) : FocusPattern<PSGNode> =
        satisfies (fun n -> n.SyntaxKind.StartsWith(prefix))

    /// Bind (monadic composition) - uses XParsec's bind semantics
    let bind (f: 'T -> FocusPattern<'U>) (p: FocusPattern<'T>) : FocusPattern<'U> =
        fun z ->
            match p z with
            | Ok success -> f success.Parsed z
            | Error e -> Error e

    let (>>=) p f = bind f p
    let (|>>) p f = bind (fun x -> succeed (f x)) p
    let (>>.) p1 p2 = p1 >>= fun _ -> p2
    let (.>>) p1 p2 = p1 >>= fun x -> p2 |>> fun _ -> x
    let (<|>) p1 p2 = fun z -> match p1 z with | Ok r -> Ok r | Error _ -> p2 z

    /// Navigate down and apply pattern to children using XParsec
    let withChildren (childParser: ChildrenPattern<'T>) : FocusPattern<'T> =
        fun z ->
            match PSGXParsec.parseChildren childParser z with
            | Ok result -> Ok { Parsed = result }
            | Error msg -> fail msg z
```

### Layer 4: MLIR Generation Uses XParsec State Threading

```fsharp
/// MLIR generation is now a PSG parser that produces MLIR fragments
type MLIRGen<'T> = PSGChildParser<'T>

module MLIRGen =
    open XParsec
    open XParsec.Parsers
    open XParsec.Combinators

    /// Access the zipper from parse state
    let getZipper : MLIRGen<PSGZipper> =
        fun reader -> preturn reader.State.Zipper reader

    /// Update generation state in zipper
    let updateState (f: GenerationState -> GenerationState) : MLIRGen<unit> =
        fun reader ->
            let newZipper = PSGZipper.mapState f reader.State.Zipper
            reader.State <- { reader.State with Zipper = newZipper }
            preturn () reader

    /// Generate fresh SSA name
    let freshSSA : MLIRGen<string> =
        fun reader ->
            let z = reader.State.Zipper
            let newZ, name = PSGZipper.nextSSA z
            reader.State <- { reader.State with Zipper = newZ }
            preturn name reader

    /// Generate MLIR line (side effect through builder in state)
    let line (text: string) : MLIRGen<unit> =
        fun reader ->
            // Actual generation goes through MLIRBuilder
            preturn () reader
```

### Layer 5: Expression Processing as XParsec Parser

```fsharp
/// Main expression processing - an XParsec parser over PSG children
let rec pExpr : MLIRGen<ExprResult> =
    // Use XParsec's choice combinator
    XParsec.Combinators.choiceL [
        // Function calls - dispatch to bindings
        pFunctionCall

        // Arithmetic expressions
        pArithmetic

        // Constants
        pConst

        // Variables
        pIdent

        // Sequential expressions
        pSequential

        // Fallback
        pUnhandled
    ] "Expected expression"

and pFunctionCall : MLIRGen<ExprResult> =
    // Match: App node with function and arguments
    PSGXParsec.childKind "App" >>= fun appNode ->
    // ... pattern continues - dispatch to platform bindings
    MLIRGen.getZipper >>= fun z ->
    // Navigate into app and process
    match PSGZipper.downTo 0 z with
    | NavOk innerZ ->
        // Use XParsec on inner's children
        PSGXParsec.parseChildren pCallTarget innerZ
        |> function
        | Ok callInfo -> processCall callInfo  // Dispatches to Bindings
        | Error _ -> XParsec.Parsers.pzero
    | NavFail _ -> XParsec.Parsers.pzero

and pArithmetic : MLIRGen<ExprResult> =
    // Match curried binary op: App[App[op, left], right]
    PSGXParsec.childKind "App" >>= fun innerApp ->
    // ... pattern continues
    MLIRGen.getZipper >>= fun z ->
    match PSGZipper.downTo 0 z with
    | NavOk innerZ ->
        PSGXParsec.parseChildren pBinaryOp innerZ
        |> function
        | Ok opInfo -> generateBinaryOp opInfo
        | Error _ -> XParsec.Parsers.pzero
    | NavFail _ -> XParsec.Parsers.pzero
```

## The Key Principle: Separation of Concerns

1. **PSGZipper** - Tree navigation, context preservation, state threading
   - Moves through the PSG: up, down, left, right
   - Preserves path context for backtracking
   - Carries generation state (SSA counter, locals, etc.)

2. **XParsec** - Sequential combinators over child lists
   - Provides `>>=`, `|>>`, `<|>`, `many`, `choice`, etc.
   - Handles backtracking within child sequences
   - Provides error reporting infrastructure

3. **FocusPattern** - Matching at current zipper position
   - Built on XParsec result types
   - Uses zipper for focus access
   - Bridges to child parsing via `withChildren`

4. **Bindings** - Platform-specific MLIR generation
   - Looked up by PSG node structure (not library names)
   - Organized by platform (Linux_x86_64, etc.)
   - Generate MLIR for syscalls, memory operations, etc.

## Migration Path

### Phase 1: Create XParsec Bridge Types
1. Implement `PSGChildSlice : IReadable<PSGNode, ...>`
2. Implement `PSGParseState` with zipper reference
3. Create `PSGChildParser<'T>` type alias

### Phase 2: Create Bridge Combinators
1. `PSGXParsec.parseChildren` - Run XParsec on zipper's children
2. Basic child matchers: `childKind`, `satisfyChild`
3. `remainingChildren`, `skipChild`, etc.

### Phase 3: Rebuild FocusPattern on XParsec Results
1. `FocusPattern<'T>` uses `ParseResult` not `option`
2. Operators (`>>=`, `|>>`, etc.) delegate to XParsec semantics
3. `withChildren` bridges to child parsing

### Phase 4: Remove Duplicate Operators from PSGPatterns
1. Delete custom `>>=`, `|>>`, `.>>`, `.>>.`, `<|>` definitions
2. Import XParsec operators
3. Rewrite patterns to use XParsec primitives

### Phase 5: Create Binding Infrastructure
1. `Alex/Bindings/BindingTypes.fs` - Core binding abstractions
2. `Alex/Bindings/PlatformRegistry.fs` - Platform detection & selection
3. Platform-specific bindings (Console, Time, Memory, etc.)

### Phase 6: Integrate Zipper + XParsec + Bindings
1. Main dispatcher uses `XParsec.choice`
2. Handlers use XParsec combinators
3. Zipper navigation for tree traversal
4. Bindings for platform-specific MLIR generation

## Files to Create/Modify

1. **New**: `Alex/Traversal/PSGXParsec.fs` - XParsec bridge for PSG
2. **Rewrite**: `Alex/Patterns/PSGPatterns.fs` - Remove duplicate operators, use XParsec
3. **Rewrite**: `Alex/Patterns/AlloyPatterns.fs` - Use XParsec-based patterns
4. **New**: `Alex/Bindings/BindingTypes.fs` - Binding abstractions
5. **New**: `Alex/Bindings/PlatformRegistry.fs` - Platform selection
6. **New**: `Alex/Bindings/Console/Linux.fs` - Linux console bindings
7. **New**: `Alex/Bindings/Time/Linux.fs` - Linux time bindings

## What This Achieves

1. **Single Combinator Infrastructure**: All `>>=`, `|>>`, `<|>` come from XParsec
2. **Zipper Preserved**: Tree navigation remains zipper-based
3. **Sequential Parsing via XParsec**: Child sequences parsed with XParsec
4. **Platform Bindings**: MLIR generation is platform-aware
5. **Composability**: New patterns built from XParsec primitives
6. **Type Safety**: XParsec's type system ensures correctness
7. **Backtracking**: XParsec handles position save/restore
8. **Error Messages**: XParsec's error infrastructure

## Conclusion

The zipper traverses the tree. XParsec parses the child sequences at each zipper position. Bindings generate platform-specific MLIR. This is the principled separation of concerns:

- **Zipper** = tree navigation with context
- **XParsec** = sequential combinators with backtracking
- **Bindings** = platform-specific MLIR generation

By building PSG pattern matching on XParsec primitives (not reimplementing them), we eliminate architectural debt while preserving the essential bidirectional zipper navigation that the blog posts describe as critical for proof-aware, context-preserving compilation.
