# NEGATIVE EXAMPLES: What NOT To Do

These are real mistakes made during development. **DO NOT REPEAT THEM.**

## Mistake 1: Adding Alloy-specific logic to MLIR generation

```fsharp
// WRONG - MLIR generation should not know about Alloy
match symbolName with
| Some name when name = "Alloy.Console.Write" ->
    generateConsoleWrite psg ctx node  // Special case!
| Some name when name = "Alloy.Console.WriteLine" ->
    generateConsoleWriteLine psg ctx node  // Another special case!
```

**Why this is wrong**: MLIR generation is now coupled to Alloy's namespace structure. If Alloy changes, the compiler breaks.

**The fix**: Alloy functions should have real implementations. The PSG should contain the full call graph. The Zipper walks the graph and Bindings generate MLIR based on node structure.

## Mistake 2: Stub implementations in Alloy

```fsharp
// WRONG - This is a stub that expects compiler magic
let inline WriteLine (s: string) : unit =
    () // Placeholder - Firefly compiler handles this
```

**Why this is wrong**: The PSG will show `Const:Unit` as the function body. There's no semantic structure for Alex to work with.

**The fix**: Real implementation that decomposes to primitives:
```fsharp
// RIGHT - Real implementation using lower-level functions
let inline WriteLine (s: string) : unit =
    writeln s  // Calls writeStrOut -> writeBytes (the actual syscall primitive)
```

## Mistake 3: Putting nanopass logic in MLIR generation

```fsharp
// WRONG - Importing nanopass modules into code generation
open Core.PSG.Nanopass.DefUseEdges

// WRONG - Building indices during MLIR generation
let defIndex = buildDefinitionIndex psg
```

**Why this is wrong**: Nanopasses run BEFORE MLIR generation. They enrich the PSG. Code generation should consume the enriched PSG, not run nanopass logic.

## Mistake 4: Adding mutable state tracking to code generation

```fsharp
// WRONG - Code generation tracking mutable bindings
type GenerationContext = {
    // ...
    MutableBindings: Map<string, Val>  // NO! This is transformation logic
}
```

**Why this is wrong**: Mutable variable handling should be resolved in the PSG via nanopasses. Code generation should just follow edges to find values.

## Mistake 5: Creating a Central Dispatch/Emitter/Scribe

```fsharp
// WRONG - Central dispatch registry
module PSGEmitter =
    let handlers = Dictionary<string, NodeHandler>()

    let registerHandler prefix handler =
        handlers.[prefix] <- handler

    let emit node =
        let prefix = getKindPrefix node.SyntaxKind
        match handlers.TryGetValue(prefix) with
        | true, handler -> handler node
        | _ -> defaultHandler node
```

**Why this is wrong**:
- This antipattern was removed TWICE (PSGEmitter, then PSGScribe)
- It collects "special case" routing too early in the pipeline
- It inevitably attracts library-aware logic ("if ConsoleWrite then...")
- The centralization belongs at OUTPUT (MLIR Builder), not at DISPATCH

**The fix**: NO central dispatcher. The Zipper folds over PSG structure. XParsec matches locally at each node. Bindings are looked up by extern primitive entry point. MLIR Builder accumulates the output.

## Mistake 6: String-based parsing or name matching

```fsharp
// WRONG - String matching on symbol names
if symbolName.Contains("Console.Write") then ...

// WRONG - Hardcoded library paths
| Some name when name.StartsWith("Alloy.") -> ...

// RIGHT - Pattern match on PSG node structure
match node.SyntaxKind with
| "App:FunctionCall" -> processCall zipper bindings
| "WhileLoop" -> processWhileLoop zipper bindings
```

## Mistake 7: Premature Centralization

Pooling decision-making logic too early in the pipeline.

**Wrong**: Creating a router/dispatcher that decides what to do with each node kind
**Right**: Let PSG structure drive emission; centralization only at MLIR output

The PSG, enriched by nanopasses, carries enough information that emission is deterministic. No routing decisions needed.

## Mistake 8: Silent Failures in Code Generation (CRITICAL)

```fsharp
// WRONG - Silently return Void when function not found
and emitInlinedCall (ctx: EmitContext) (funcNode: PSGNode) (argNodes: PSGNode list) : ExprResult =
    // ...
    match funcBinding with
    | Some binding -> // ... emit code
    | None ->
        printfn "[GEN] Function not found in PSG: %s" name  // Just prints!
        Void  // *** SILENT FAILURE - continues compilation ***
```

**What happened**: During HelloWorldDirect compilation, the output showed:
```
[GEN] Function not found in PSG: System.Object.ReferenceEquals
[GEN] Function not found in PSG: Microsoft.FSharp.Core.Operators.``not``
```

The code printed warnings but returned `Void` and continued. The result:
- Conditional check silently failed
- Only a newline was written (not "Hello, World!")
- Binary segfaulted on `ret` instruction

**Why this is wrong**: 
- Compilers exist to surface errors. Silent failures hide bugs behind more bugs.
- The root cause (unresolved function) manifested as a symptom (segfault)
- Hours were spent chasing the segfault instead of fixing the real issue

**The fix**:
```fsharp
// RIGHT - Return EmitError and propagate it
| None ->
    EmitError (sprintf "Function not found in PSG: %s - cannot generate code" name)
    // Caller MUST handle EmitError and fail compilation
```

**The principle**: When code generation cannot proceed, it MUST emit an error that halts compilation. Never swallow failures with `printfn` + `Void`.

**When you see this pattern**: STOP EVERYTHING. Do not run the binary. Do not chase symptoms. Fix the error propagation first.

---

## The Acid Test

Before committing any change, ask:

> "If someone deleted all the comments and looked only at what this code DOES, would they see library-specific logic in MLIR generation?"

If yes, you have violated the layer separation principle. Revert and fix upstream.

> "Am I creating a central dispatch mechanism?"

If yes, STOP. This is the antipattern that was removed twice.
