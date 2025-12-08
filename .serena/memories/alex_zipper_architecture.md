# Alex/Zipper Architecture - The Correct Model

## CRITICAL: NO EMITTER LAYER

**The "emitter" is an ANTIPATTERN that was removed.** See `docs/Emitter_Removal_Rebuild_Plan.md`.

An emitter sits between PSG and MLIR and makes interpretation decisions. This is wrong because:
1. It would need Alloy awareness (WRONG - violates layer separation)
2. It becomes a translation layer that accumulates special cases
3. It circumvents the compositional XParsec + Bindings architecture

## The Correct Architecture

```
PSG (full semantic graph from FCS) 
    → Zipper (XParsec pattern matching on PSG structure)
        → Bindings (platform-specific MLIR generation)
            → MLIR (may include target triple)
                → LLVM → Native binary
```

### Why Alloy CANNOT Know About Targets

The *same Alloy code* must compile to radically different MLIR structures:
- **x86_64 Linux**: alloca is fine, syscalls for I/O
- **ARM Cortex-M33 bare metal**: everything on stack, loops unrolled/flattened, no syscalls
- **WASM**: completely different memory model

Memory management decisions, loop flattening, stack layout - these are STRUCTURAL targeting decisions that LLVM cannot infer from a triple. They must be made during MLIR generation by Alex.

### Why Alex CANNOT Be Circumvented

Alex (via Zipper + XParsec + Bindings) is the ONLY place where targeting decisions live:
- **Zipper** traverses PSG nodes bidirectionally
- **XParsec combinators** pattern-match children at each position
- **Bindings** dispatch based on `(entry_point, target_platform)` from fidproj

When traversal hits an extern primitive (identified by PSG structure - the `DllImport("__fidelity")` attribute), Bindings generate target-appropriate MLIR.

### What Phase 2 Should Be

Enhancing the Zipper's traversal to use XParsec combinators for matching PSG patterns, with Bindings producing target-appropriate MLIR. 

**NOT** creating an intermediate "emitter" abstraction.

## Key Files

| File | Role |
|------|------|
| `Alex/Traversal/PSGZipper.fs` | Bidirectional PSG traversal |
| `Alex/Traversal/PSGXParsec.fs` | XParsec combinators for PSG children |
| `Alex/Bindings/BindingTypes.fs` | ExternDispatch registry, platform types |
| `Alex/Bindings/Console/ConsoleBindings.fs` | Console I/O platform bindings |
| `Alex/CodeGeneration/MLIRBuilder.fs` | MLIR primitives (arith, llvm, func dialects) |
| `Alex/Patterns/PSGPatterns.fs` | Predicates and extractors for PSG nodes |

## Validation Samples (DO NOT MODIFY)

| Sample | Tests |
|--------|-------|
| 01_HelloWorldDirect | Static strings, basic Console calls |
| 02_HelloWorldSaturated | Let bindings, string interpolation |
| 03_HelloWorldHalfCurried | Pipe operators, function values |
