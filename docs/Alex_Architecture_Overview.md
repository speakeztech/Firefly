# Alex Architecture Overview

> **See `Architecture_Canonical.md` for the authoritative two-layer model.**

## The "Library of Alexandria" Model

Alex is Firefly's **multi-dimensional hardware targeting layer**. It consumes the PSG (Program Semantic Graph) and generates platform-optimized MLIR.

## Core Responsibility

Alex dispatches **extern primitives** (declared in Alloy via `DllImport("__fidelity")`) to platform-specific MLIR generation. Platform differences are **data** (syscall numbers, register conventions), not separate code files.

```
PSG with extern metadata → Alex dispatch → Platform MLIR → LLVM → Native
```

## Key Components

| Component | Purpose |
|-----------|---------|
| `Bindings/BindingTypes.fs` | Platform enum, dispatch types |
| `Bindings/ExternDispatch.fs` | Central (entry_point, platform) → MLIR router |
| `Traversal/PSGZipper.fs` | Bidirectional PSG traversal |
| `CodeGeneration/MLIRBuilder.fs` | MLIR computation expression (`mlir { }`) |
| `Pipeline/CompilationOrchestrator.fs` | End-to-end compilation |

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
