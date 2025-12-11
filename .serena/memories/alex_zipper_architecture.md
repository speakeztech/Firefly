# Alex/Zipper Architecture - The Non-Dispatch Model

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

## Key Files

| File | Role |
|------|------|
| `Alex/Traversal/PSGZipper.fs` | Bidirectional PSG traversal (attention) |
| `Alex/Traversal/PSGXParsec.fs` | XParsec combinators for local pattern matching |
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
