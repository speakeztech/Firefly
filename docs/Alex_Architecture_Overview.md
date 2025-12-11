# Alex Architecture Overview

> **See `Architecture_Canonical.md` for the authoritative two-layer model.**

## The "Library of Alexandria" Model

Alex is Firefly's **multi-dimensional hardware targeting layer**. It consumes the PSG (Program Semantic Graph) and generates platform-optimized MLIR.

## Core Responsibility: The Non-Dispatch Model

> **Key Insight: Centralization belongs at the OUTPUT (MLIR Builder), not at DISPATCH (traversal logic).**

Alex generates MLIR through Zipper traversal and platform Bindings. There is **NO central dispatch hub**.

```
PSG Entry Point
    ↓
Zipper.create(psg, entryNode)     -- provides "attention"
    ↓
Fold over structure (pre-order/post-order)
    ↓
At each node: XParsec matches locally → MLIR emission
    ↓
Extern primitive? → ExternDispatch.dispatch(primitive)
    ↓
MLIR Builder accumulates           -- correct centralization
    ↓
Output: Complete MLIR module
```

Platform differences are **data** (syscall numbers, register conventions), not routing logic.

## Key Components

| Component | Purpose |
|-----------|---------|
| `Traversal/PSGZipper.fs` | Bidirectional PSG traversal ("attention") |
| `Traversal/PSGXParsec.fs` | Local pattern matching combinators |
| `Bindings/BindingTypes.fs` | ExternDispatch registry, platform types |
| `Bindings/*/` | Platform bindings (data, not routing) |
| `CodeGeneration/MLIRBuilder.fs` | MLIR accumulation (correct centralization) |
| `Pipeline/CompilationOrchestrator.fs` | Entry point |

**Note:** PSGEmitter.fs and PSGScribe.fs were removed - they were central dispatch antipatterns.

## External Tool Integration

Alex delegates to battle-tested infrastructure:
- `mlir-opt` for dialect conversion
- `mlir-translate` for LLVM IR generation
- `llc` for machine code generation
- System linker for final executable

## OutputKind

```fsharp
type OutputKind =
    | Console       // Uses libc, main entry point
    | Freestanding  // No libc, _start wrapper, direct syscalls
    | Embedded      // No OS, custom startup
    | Library       // No entry point, exported symbols
```

---

*For detailed architecture decisions, see `Architecture_Canonical.md`.*
