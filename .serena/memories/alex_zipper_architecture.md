# Alex/Zipper Architecture - The Non-Dispatch Model

## FOUNDATIONAL INSIGHT: SSA IS Functional Programming

From Appel's seminal 1998 paper "SSA is Functional Programming":

> F#'s functional structure maps **directly** to MLIR/SSA without reconstruction.

This means:
- **F# ≅ SSA** - The functional structure IS the compilation target
- **PSG captures intent** - Semantic primitives express WHAT, not HOW
- **Alex chooses implementation** - Target-aware decisions (x86 AVX, ARM NEON, embedded, WAMI)

## STATUS: ARCHITECTURAL INTEGRITY REFACTORING (Dec 2024)

**Alex IS a transformer** - it transforms PSG into MLIR compute graphs using XParsec and parser combinator monads. The issue is that MLIRTransfer.fs has accumulated CRUFT (workarounds, ad-hoc lookups, name matching) that compensates for insufficient PSG enrichment.

**This cruft must be moved UP into proper places:**
- **PSG nanopasses** for enrichment (ExternInfo, StringInfo, CallTarget edges)
- **Alloy** for real implementations (not stubs)
- **Proper Alex architecture** (Zipper + XParsec + Bindings)

**Current Progress:**
- `PSGZipper.fs` - Complete with foldPreOrder, foldPostOrder, navigation ✓
- `PSGXParsec.fs` - Complete with combinators (pBind, pMap, pOr, etc.) ✓
- `Bindings/` - Platform-specific extern dispatch ✓
- DetectExternPrimitives nanopass - In progress (moving extern detection from MLIR to PSG)

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
Fold over structure (pre-order or post-order)
    ↓
At each node: XParsec pattern → MLIR emission
    ↓
Extern primitive? → ExternDispatch.dispatch(primitive)
    ↓
MLIR Builder accumulates
    ↓
Output: Complete MLIR module
```

There is NO routing table. NO central dispatcher. The PSG structure itself drives emission.

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

## Key Files

| File | Role |
|------|------|
| `Alex/Generation/MLIRGeneration.fs` | Single MLIR generation entry point |
| `Alex/Traversal/PSGZipper.fs` | Bidirectional PSG traversal (attention) |
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
