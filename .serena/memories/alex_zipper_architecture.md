# Alex/Zipper Architecture - The Non-Dispatch Model

## FOUNDATIONAL INSIGHT: SSA IS Functional Programming

From Appel's seminal 1998 paper "SSA is Functional Programming":

> F#'s functional structure maps **directly** to MLIR/SSA without reconstruction.

This means:
- **F# ≅ SSA** - The functional structure IS the compilation target
- **PSG captures intent** - Semantic primitives express WHAT, not HOW
- **Alex chooses implementation** - Target-aware decisions (x86 AVX, ARM NEON, embedded, WAMI)

## STATUS: DUAL-ZIPPER ARCHITECTURE IMPLEMENTED (Dec 2024)

**Alex IS a transformer** - it transforms PSG into MLIR compute graphs using the dual-zipper architecture.

**NEW: MLIRZipper & Transfer.fs**

The dual-zipper architecture is now implemented:
- `PSGZipper.fs` - Complete INPUT zipper (PSG navigation) ✓
- `MLIRZipper.fs` - NEW OUTPUT zipper (MLIR composition) ✓
- `Transfer.fs` - NEW principled transfer pipeline using dual zippers ✓
- `MLIRTransfer.fs` - Legacy (still used for compilation, to be replaced)

**MLIRZipper Design** (symmetric to PSGZipper):
```fsharp
type MLIRZipper = {
    Focus: MLIRFocus           // AtModule | InFunction | InBlock
    Path: MLIRPath             // Breadcrumbs for bidirectional navigation
    CurrentOps: MLIROp list    // Operations at current focus
    State: MLIRState           // SSA counters, NodeSSA map, string literals
    Globals: MLIRGlobal list   // Global declarations
}
```

**Transfer.fs Key Principle:**
> "Children are ALREADY processed when parent is visited" (post-order traversal)

- Uses `foldPostOrder` ONCE at entry point - no manual navigation
- Child SSAs are looked up from accumulated `NodeSSA` map
- NO recursive `transferNode` calls - fold handles traversal
- Dispatch on `node.Kind`, `node.Operation`, `node.PlatformBinding` - NOT names

**Current Progress:**
- `PSGZipper.fs` - Complete with foldPreOrder, foldPostOrder, navigation ✓
- `PSGXParsec.fs` - Complete with combinators (pBind, pMap, pOr, etc.) ✓
- `MLIRZipper.fs` - NEW: Complete with functional state threading ✓
- `Transfer.fs` - NEW: Principled dual-zipper pipeline ✓
- `Bindings/` - Platform-specific extern dispatch ✓
- All samples 01-04 compile and run ✓

**See `alex_remediation_plan` memory for the comprehensive migration plan.**

## CRITICAL: NO EMITTER OR SCRIBE LAYER

**The "emitter" and "scribe" are ANTIPATTERNS that were removed.** PSGEmitter and PSGScribe were both removed because they created central dispatch hubs that attracted library-aware logic.

An emitter/scribe sits between PSG and MLIR and makes routing decisions. This is wrong because:
1. It collects "special case" routing too early in the pipeline
2. It inevitably attracts library-aware logic ("if ConsoleWrite then...")
3. It circumvents the compositional XParsec + Bindings architecture
4. The centralization belongs at OUTPUT (MLIR Builder), not at DISPATCH

## The Correct Architecture (Non-Dispatch Model)

> **Key Insight: Centralization belongs at the OUTPUT (MLIR Builder), not at DISPATCH (traversal logic).**

```
PSG Entry Point
    ↓
Zipper.create(psg, entryNode)
    ↓
foldPostOrder (children before parents)
    ↓
At each node: pattern match → MLIR emission
    ↓
Extern primitive? → ExternDispatch.dispatch(primitive)
    ↓
MLIR Builder accumulates (mutable BuilderState)
    ↓
Output: Complete MLIR module
```

There is NO routing table. NO central dispatcher. The PSG structure itself drives emission.

## CRITICAL: Post-Order Traversal for Expression Evaluation

**Use `foldPostOrder`, NOT `foldPreOrder` for MLIR generation.**

SSA-based code generation requires that **children are evaluated before parents**:
- `Add(x, y)` needs SSA values for `x` and `y` before emitting `arith.addi`
- Post-order visits children first, then the parent
- When the parent node is visited, its children's SSA values are already in NodeSSA map

```fsharp
// Post-order: children emit SSAs first, parent consumes them
let emitNode (acc: EmitAcc) (zipper: PSGZipper) : EmitAcc =
    let node = zipper.Focus
    // Children already processed - their SSAs are in acc.NodeSSA
    match node.Operation with
    | Some (Arithmetic Add) ->
        let childSSAs = getChildSSAs node acc.NodeSSA  // Already available!
        let result = mlir { let! r = arith.addi childSSAs.[0] childSSAs.[1]; return r } acc.State
        { acc with NodeSSA = Map.add node.Id.Value result acc.NodeSSA }
    | ...
```

## The MLIR Monad: Eager Execution with Mutable State

The `MLIR<'T>` monad is defined as:
```fsharp
type MLIR<'T> = BuilderState -> 'T
```

This is a **Reader monad** with **mutable state**:
- `BuilderState` has `mutable SSACounter`, `mutable Indent`, `Output: StringBuilder`
- When you run `mlir { ... } state`, mutations happen immediately
- The monad composes operations; execution is eager when applied to state

**This is NOT lazy/deferred execution.** Each `mlir { }` block runs immediately when given a BuilderState. The "pull" model refers to:
1. **Structure-driven**: PSG structure determines what gets emitted (no routing table)
2. **Demand-driven naming**: We don't pre-allocate SSA names; they're generated as needed
3. **Local decisions**: Each node's emission logic is self-contained

### Fold Accumulator Design

```fsharp
type EmitAcc = {
    State: BuilderState           // Shared, mutated during fold
    NodeSSA: Map<string, Val>     // NodeId.Value -> emitted Val
    Graph: ProgramSemanticGraph   // For def-use edge lookups
}

let emitNode (acc: EmitAcc) (zipper: PSGZipper) : EmitAcc =
    let node = zipper.Focus
    // Run mlir { } with shared state - mutations accumulate
    let emittedVal = emitForNode node acc |> fun m -> m acc.State
    // Update NodeSSA for def-use resolution
    { acc with NodeSSA = Map.add node.Id.Value emittedVal acc.NodeSSA }
```

The `BuilderState` is threaded through the entire fold via the accumulator. Each node's emission mutates it (appending to Output, incrementing SSACounter).

### Component Roles

**The Zipper:**
- Provides "attention" - focus on current node with full context
- Purely navigational: `up`, `down`, `left`, `right`, `downTo`
- Carries state through traversal (SSA counters, emitted blocks)
- NO dispatch logic, NO routing decisions

**XParsec Combinators:**
- Match PSG node structure at each position
- Composable patterns via `pMap`, `pBind`, `pOr`, `pMany`, etc.
- Local decision-making based on node data
- NOT a routing table

**The Bindings:**
- Contain platform-specific MLIR generation
- Looked up by extern primitive entry point (e.g., `"fidelity_write_bytes"`)
- Are DATA (syscall numbers, calling conventions), not routing logic
- Organized by `(OSFamily, Architecture, EntryPoint)` tuple
- `ExternDispatch.dispatch` looks up and invokes the binding

**MLIR Builder:**
- The natural "pool" where emissions accumulate
- This is where centralization CORRECTLY occurs
- Type-safe MLIR construction via computation expression
- All emissions flow here - this is the single output point

### Why This Model Works

The PSG, enriched by nanopasses, carries enough information that emission is a **deterministic function of node structure**:
- Node's SyntaxKind determines basic emission pattern
- Node's data (symbol, type, children) provides parameters
- Extern primitives trigger Binding lookup by entry point
- No interpretation or routing decisions needed

The Zipper doesn't "decide" what to do with a node - it focuses attention. The emission logic is local to each node kind, composed via XParsec.

## What NOT To Do

```fsharp
// WRONG - Central dispatch registry
module PSGEmitter =
    let handlers = Dictionary<string, NodeHandler>()
    let emit node =
        match handlers.TryGetValue(getPrefix node) with
        | true, h -> h node
        | _ -> default node

// WRONG - Pattern matching on library names
match symbolName with
| "Alloy.Console.Write" -> handleConsoleWrite()

// WRONG - Special cases for specific functions
if node.SymbolName.Contains("Console") then ...
```

## Two Zippers, Two Purposes

Firefly uses zippers in two distinct contexts with symmetric architecture:

| Aspect | Baker's TypedTreeZipper | Alex's PSGZipper |
|--------|-------------------------|------------------|
| **Trees** | FSharpExpr + PSGNode (two-tree) | PSGNode only |
| **Purpose** | Correlation/enrichment | Code generation |
| **Direction** | Inward (gathering type info) | Outward (emitting MLIR) |
| **State** | Accumulated correlations, member bodies | Emitted MLIR, SSA counters |
| **Phase** | Phase 4 (typed tree overlay) | Final emission |

**Zipper Coherence**: Both zippers operate on the **same narrowed graph** - the reachable subset determined in Phase 3. SRTP resolutions found by Baker are exactly those needed by Alex.

Baker focuses *in* on the application graph (gathering); Alex fans *out* to MLIR (emitting). One gathers, the other generates - a lens into a prism across the nanopass boundary.

## Correlation Flow from Baker

Baker provides `CorrelationInfo` that flows into Alex's `EmitContext`:

```fsharp
// CompileCommand.fs calls:
let mlirResult = generateMLIR psg bakerResult.Correlations targetTriple

// MLIRGeneration.fs creates context:
let ctx = EmitContext.create psg correlations

// EmitContext provides lookups:
let fieldInfo = EmitContext.lookupFieldAccess ctx nodeId
```

`FieldAccessInfo` enables correct `llvm.extractvalue` emission for struct fields like `NativeStr.Length`.

## THE DEEPER "WHY": DCont/Inet Duality and Waypoints

This architecture is not arbitrary. It's laying groundwork for future capabilities that require discipline NOW.

### DCont/Inet Duality (from "The DCont/Inet Duality" blog post)

Computation expressions decompose into two fundamental patterns:
- **DCont (Delimited Continuations)**: Sequential effects where each step depends on previous
- **Inet (Interaction Nets)**: Parallel pure computation with no dependencies

The PSGZipper fold IS the DCont pattern:
- Each node visit is a "continuation point"
- Post-order traversal = natural evaluation order (children before parents)
- State threading through fold = explicit continuation threading

**Why this matters for the future:**
```fsharp
// Today: Simple MLIR generation via fold
let! result = PSGZipper.foldPostOrder emitNode initialAcc zipper

// Future: Async F# compiles to DCont MLIR dialect
dcont.shift { ... }   // Captures "the rest of the computation"
dcont.resume k value  // Delivers value to captured continuation
dcont.reset result    // Establishes boundary
```

When we add async/await support, the SAME zipper fold architecture naturally extends to emit DCont operations. The discipline now enables DCont later.

### Coeffects: Tracking Requirements, Not Products

From "Coeffects and Codata in Firefly":

> Effects track what code *does* to its environment.
> Coeffects track what code *needs* from its environment.

The EmitAcc accumulator IS coeffect tracking:
```fsharp
type EmitAcc = {
    State: BuilderState           // SSA counter = "need fresh names"
    NodeSSA: Map<string, Val>     // Def-use = "need values from definitions"
    Graph: ProgramSemanticGraph   // Structure = "need child relationships"
}
```

Compilation decisions depend on requirements, not products. This is why the fold accumulator tracks what each node NEEDS (child SSAs, SSA counter, graph structure).

### Codata: Demand-Driven Observation

The fold "observes" nodes as it traverses - this IS the codata pattern:
- Data = eagerly constructed, then examined
- Codata = defined by observation/consumption

The PSGZipper doesn't pre-construct MLIR trees. It observes nodes on demand:
```fsharp
// Codata: observation defines behavior
let emitNode (acc: EmitAcc) (zipper: PSGZipper) : EmitAcc =
    let node = zipper.Focus  // OBSERVE the current node
    // Emit based on what we observe, not what was pre-constructed
```

This aligns with "codata patterns compile to stack-based state machines" from the blog.

### Why Discipline NOW Enables DCont/Inet LATER

The current refactoring from `transferNodeDirect` (direct recursion) to `foldPostOrder` (zipper fold) is NOT just cleanup. It's establishing the architectural pattern that:

1. **Enables DCont lowering**: When we add async support, suspension points become `dcont.shift` operations. The fold's continuation structure maps directly.

2. **Enables Inet optimization**: Pure regions identified by coeffects can be parallelized via Inet dialect. The fold structure makes these regions explicit.

3. **Enables WAMI targeting**: WebAssembly stack switching requires preserved continuation structure. The zipper fold preserves it.

4. **Enables hardware diversity**: CGRAs, neuromorphic processors, etc. express computation as dataflow graphs. The explicit structure survives to these targets.

**This is the waypoint**: Today's disciplined fold → Tomorrow's DCont/Inet duality → Future hardware targets.

## Key Files

| File | Role |
|------|------|
| `Alex/Generation/MLIRTransfer.fs` | Legacy MLIR generation (still used, to be replaced) |
| `Alex/Generation/Transfer.fs` | **NEW**: Principled dual-zipper transfer pipeline |
| `Alex/Traversal/PSGZipper.fs` | INPUT zipper - Bidirectional PSG traversal |
| `Alex/Traversal/MLIRZipper.fs` | **NEW**: OUTPUT zipper - MLIR composition |
| `Alex/Traversal/PSGXParsec.fs` | XParsec combinators, EmitContext with correlations |
| `Alex/Bindings/BindingTypes.fs` | ExternDispatch registry, platform types |
| `Alex/Bindings/Console/ConsoleBindings.fs` | Console I/O platform bindings (data) |
| `Alex/CodeGeneration/MLIRBuilder.fs` | MLIR accumulation (correct centralization point) |
| `Alex/Patterns/PSGPatterns.fs` | Predicates and extractors for PSG nodes |

## Removed Files (Antipatterns)

- `Alex/Traversal/PSGEmitter.fs` - Removed: central dispatch hub
- `Alex/Traversal/PSGScribe.fs` - Removed: "clean" attempt that recreated the same antipattern

## Validation

The correct implementation must pass these tests:
1. NO handler registry or dispatch table
2. NO pattern matching on symbol/function names
3. Zipper provides traversal only
4. XParsec provides local pattern matching only
5. Bindings are looked up by extern entry point only
6. MLIR Builder is the single accumulation point
