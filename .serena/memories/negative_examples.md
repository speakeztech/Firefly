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

## Mistake 9: Hardcoding Types Instead of Using Architectural Type Flow (CRITICAL)

This antipattern was discovered and fixed in January 2026 during the "fat pointer string" debugging session.

```fsharp
// WRONG - Hardcoding type mappings instead of using FNCS type information
let rec mapType (ty: NativeType) : MLIRType =
    match ty with
    | ...
    | "string" -> Pointer  // IGNORES that FNCS knows strings are fat pointers!
```

```fsharp
// WRONG - Hardcoding function signatures instead of using node.Type
| SemanticKind.Lambda _ ->
    let funcName = name
    let signature = "(!llvm.ptr) -> !llvm.ptr"  // IGNORES valueNode.Type!
    let zipper' = MLIRZipper.observeExternFunc funcName signature zipper
    zipper', TRValue ("@" + funcName, "!llvm.ptr")
```

**The Principled Architecture**:
- FNCS defines types with correct layouts: `stringTyCon = mkTypeConRef "string" 0 (TypeLayout.Inline(16, 8))` — fat pointer
- PSG nodes carry `Type: NativeType` with full type information
- `mapType` converts NativeType → MLIRType
- `Serialize.mlirType` converts MLIRType → string

**What Was Wrong**:
1. `mapType` returned `Pointer` for strings instead of the fat pointer struct `NativeStrType`
2. External function signatures were hardcoded as `"(!llvm.ptr) -> !llvm.ptr"` instead of derived from `valueNode.Type`
3. Four separate locations had the same hardcoded signature cruft
4. The type information from FNCS was being discarded at the Alex boundary

**Why This Pattern Emerges**:
- When something doesn't work, the instinct is to add a "fallback" or "default"
- Fallbacks accumulate as cruft that ignores the principled type flow
- Each patch makes the next problem harder to diagnose

**The Fix Pattern**:
1. **Trace upstream**: The type information exists in FNCS. Where is it discarded?
2. **Remove, don't fix**: Don't make the fallback use correct types. REMOVE the fallback.
3. **Trust the zipper**: The codata/pull model means the graph contains everything
4. **Use existing architecture**: `mapType` + `Serialize.mlirType` on actual `node.Type`

```fsharp
// RIGHT - Use the type information the architecture provides
| "string" -> NativeStrType  // Fat pointer {ptr: *u8, len: i64}

// RIGHT - Derive signature from actual node type
| SemanticKind.Lambda _ ->
    let funcName = name
    let signature =
        match valueNode.Type with
        | NativeType.TFun(paramTy, retTy) ->
            sprintf "(%s) -> %s"
                (Serialize.mlirType (mapType paramTy))
                (Serialize.mlirType (mapType retTy))
        | _ -> // Error, not fallback
    let zipper' = MLIRZipper.observeExternFunc funcName signature zipper
    ...
```

**The Principle**: The architecture provides type information at every layer. When code ignores this and hardcodes types, it's always a bug that will manifest as type mismatches downstream. The fix is never to "improve the hardcoding" but to REMOVE it and use what the architecture provides.

**Remediation Checklist**:
1. Consult Serena memories on architecture before any fix
2. Identify where type information flows from FNCS through PSG to Alex
3. Find where it's being discarded or ignored
4. Remove cruft entirely - don't patch it
5. Trust the zipper's attention mechanism over the PSG

---

## Mistake 10: Imperative "Push" Patterns vs Codata "Pull" Model

Related to Mistake 9, this antipattern involves adding imperative traversal logic and fallback paths instead of trusting the zipper's pull model.

```fsharp
// WRONG - "Wasn't traversed yet" fallback with imperative assumption
| None ->
    // DEFERRED RESOLUTION: The binding's value wasn't traversed yet
    // Check if the value is a Literal (constant) or Lambda (function)
    match SemanticGraph.tryGetNode valueNodeId graph with
    | Some valueNode ->
        // Create extern declaration as workaround...
```

**Why this is wrong**:
- The zipper provides "attention" to any part of the graph
- There's no "wasn't traversed yet" if you use the zipper correctly
- The graph contains everything; the zipper lets you navigate to it

**The Codata/Pull Principle**:
- Don't track what was "already traversed"
- Don't create fallbacks for "not yet seen" nodes
- The graph is complete; witness what you need when you need it
- The zipper carries accumulated observations; recall prior observations via state

**When you find yourself writing "if not traversed yet, then fallback"**:
STOP. You're not using the zipper correctly. The information is available.

---

## Mistake 11: Wrong Binding Layer (BCL Stubs for Platform Operations)

This antipattern was identified in January 2026 during the unified binding architecture analysis.

```fsharp
// WRONG - Platform.Bindings with BCL stubs
module Platform.Bindings =
    let writeBytes fd buffer count : int = Unchecked.defaultof<int>  // BCL!
    let readBytes fd buffer maxCount : int = Unchecked.defaultof<int>
```

**Why this is wrong**:
1. `Unchecked.defaultof` is a BCL function - violates BCL-free principle
2. These are FNCS intrinsics (`Sys.write`, `Sys.read`) - Alloy shouldn't re-declare them
3. Creates semantic vacuum - PSG shows stub body, not meaningful operation
4. Forces Alex to do name-based dispatch ("if Platform.Bindings.writeBytes then...")

**The Three-Layer Architecture**:

| Layer | What | Mechanism |
|-------|------|-----------|
| Layer 1 | FNCS Intrinsics | FNCS emits directly - `Sys.write`, `NativePtr.set` |
| Layer 2 | Binding Libraries | Quotation semantic carriers - Farscape-generated |
| Layer 3 | User Code | Uses Layer 1 & 2 - Alloy, applications |

**The Fix**:
```fsharp
// RIGHT - Alloy uses FNCS intrinsics
module Console =
    let Write (s: string) =
        let ptr = String.asPtr s
        let len = String.length s
        Sys.write 1 ptr len |> ignore  // FNCS intrinsic, not stub
```

**When to use each layer**:
- **Layer 1 (Intrinsics)**: Operations native to the type universe - `Sys.*`, `NativePtr.*`
- **Layer 2 (Binding Libraries)**: External bindings with rich metadata - GTK, CMSIS
- **Layer 3 (User Code)**: Everything else - uses Layer 1 & 2

**See**: Firefly `binding_architecture_unified` memory for complete architecture.

---

## Mistake 12: TVar → Pointer Default for Polymorphic Operators

Discovered in January 2026 during SCF dialect work when `concat2` (string concatenation) failed.

```fsharp
// In mapType:
| NativeType.TVar _ -> Pointer  // ALL type variables become pointers
```

```fsharp
// What happens with op_Addition : 'a -> 'a -> 'a
let len1 = s1.Length    // Type: int (i64)
let len2 = s2.Length    // Type: int (i64)
let total = len1 + len2 // Type: 'a -> 'a -> 'a instantiated to int
```

**The Problem**:
1. `op_Addition` has polymorphic type `'a -> 'a -> 'a` in PSG
2. When curried application creates a Lambda, param types come from `TVar`
3. `mapType(TVar _) = Pointer` → Lambda gets `(!llvm.ptr, !llvm.ptr) -> !llvm.ptr`
4. At call site, `i64` values get converted via `inttoptr` to match signature
5. Result is `!llvm.ptr`, used where `i64` expected → type error

**MLIR Output**:
```mlir
// WRONG - Generated lambda for curried (+)
llvm.func internal @lambda_13(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> !llvm.ptr {
    %v0 = llvm.call @op_Addition(%arg0) : (!llvm.ptr) -> !llvm.ptr
    ...
}

// Call site converts i64 lengths to pointers
%v0 = llvm.extractvalue %arg0[1] : !llvm.struct<(!llvm.ptr, i64)>  // i64
%v1 = llvm.inttoptr %v0 : i64 to !llvm.ptr   // WRONG
%v2 = llvm.call @lambda_13(%v1, ...) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
```

**Why This Is Wrong**:
- The SRTP resolution should have resolved `'a` to `int` at this call site
- Typed tree overlay should carry concrete instantiated types
- Code generation receives `TVar` instead of resolved concrete type

**The Architectural Fix (NOT YET IMPLEMENTED)**:
The typed tree (`FSharpExpr`) contains SRTP resolution information. When F# compiler resolves `+` on integers, it knows the concrete instantiation. This information must be captured in the PSG's typed tree overlay and used during code generation.

**Current Workaround** (partial):
- Added `tryEmitPrimitiveBinaryOp` for when both args are primitive types at a single Application
- Doesn't help curried applications where Lambda is created with wrong types

**When You See This Pattern**:
1. Type mismatch involving `!llvm.ptr` and primitive types
2. `inttoptr`/`ptrtoint` conversions in generated MLIR
3. Polymorphic operators being treated as external functions

**Investigation Path**:
1. Check PSG node's `Type` field - is it `TVar` or concrete?
2. If `TVar`, the typed tree overlay isn't working for this node
3. Trace to Baker (typed tree zipper) to see where resolution is lost

**See**: `srtp_resolution_findings` memory for typed tree overlay architecture.

---

## The Acid Test

Before committing any change, ask:

> "If someone deleted all the comments and looked only at what this code DOES, would they see library-specific logic in MLIR generation?"

If yes, you have violated the layer separation principle. Revert and fix upstream.

> "Am I creating a central dispatch mechanism?"

If yes, STOP. This is the antipattern that was removed twice.
