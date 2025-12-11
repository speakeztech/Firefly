# Architecture Principles

## CRITICAL: Consult Memories at Decision Points

When encountering issues "downstream" in the pipeline, ALWAYS consult Serena memories BEFORE attempting fixes. The memories encode lessons learned from past mistakes.

Key memories to consult:
- `alex_zipper_architecture` - The Non-Dispatch Model
- `negative_examples` - Real mistakes to avoid
- `native_binding_architecture` - Extern primitive flow

## The Non-Dispatch Model

> **Key Insight: Centralization belongs at the OUTPUT (MLIR Builder), not at DISPATCH (traversal logic).**

The correct model has NO central dispatcher:

```
PSG Entry Point
    ↓
Zipper.create(psg, entryNode)
    ↓
Fold over structure
    ↓
At each node: XParsec pattern → MLIR emission
    ↓
Extern primitive? → ExternDispatch.dispatch(primitive)
    ↓
MLIR Builder accumulates (correct centralization)
    ↓
Output: Complete MLIR module
```

The PSG structure drives emission. There's no routing table.

## The Layer Separation Principle

> **Each layer has ONE responsibility. Do not mix concerns across layers.**

| Layer | Responsibility | DOES NOT |
|-------|---------------|----------|
| **Alloy** | F# library implementations | Contain stubs expecting compiler magic |
| **FCS** | Parse, type-check, resolve | Transform or generate code |
| **PSG Builder** | Construct semantic graph | Make targeting decisions |
| **Nanopasses** | Enrich PSG | Generate MLIR or know targets |
| **Alex/Zipper** | Traverse PSG | Route or dispatch |
| **XParsec** | Local pattern matching | Central routing |
| **Bindings** | Platform MLIR generation | Know F# syntax or Alloy |
| **MLIR Builder** | Accumulate emissions | Route or dispatch |

## ANTIPATTERN: Central Dispatch (Emitter/Scribe)

**NEVER create:**
- Handler registries that route by node kind
- Central dispatch mechanisms
- Pattern matching on symbol/function names
- "Emitter" or "Scribe" abstraction layers

These were removed twice (PSGEmitter, PSGScribe). They collect routing logic too early and attract library-aware special cases.

## The PSG Nanopass Pipeline

PSG construction is a true nanopass pipeline:

```
Phase 1: Structural Construction    SynExpr → PSG nodes + ChildOf edges
Phase 2: Symbol Correlation         + FSharpSymbol attachments
Phase 3: Soft-Delete Reachability   + IsReachable marks (preserve structure!)
Phase 4: Typed Tree Overlay         + Type, Constraints, SRTP resolution
Phase 5+: Enrichment Nanopasses     + def-use edges, classifications, etc.
```

**Soft-delete** is critical - the typed tree zipper needs full structure.

## The Extern Primitive Surface

The ONLY acceptable "stubs" are extern declarations:

```fsharp
[<DllImport("__fidelity", EntryPoint = "fidelity_write_bytes")>]
extern int writeBytes(int fd, nativeptr<byte> buffer, int count)
```

The `"__fidelity"` library is a marker. `ExternDispatch` looks up bindings by entry point.

## Coeffects and Codata

From the SpeakEZ blog:
- **Coeffects** track what code NEEDS - carried in PSG, not computed during emission
- **Codata** is demand-driven - Zipper "observes" nodes as it traverses

The Zipper doesn't decide what to do. The PSG node carries enough information (via nanopass enrichment) that emission is deterministic.

## Zoom Out Before Fixing

When a symptom appears downstream:

1. **Consult Serena memories** on architecture
2. **Trace upstream** through the full pipeline
3. **Find root cause** at the earliest point
4. **Fix upstream** - don't patch symptoms
5. **Validate end-to-end**

The symptom location and fix location are usually different.
