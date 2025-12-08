# Firefly Project Overview

## Purpose
Firefly is an ahead-of-time (AOT) compiler for F# targeting native binary output without the .NET runtime. The project compiles F# source code through MLIR to LLVM IR and finally to native executables.

**Primary Goal**: Compile F# code to efficient, standalone native binaries that can run without any runtime dependencies (freestanding mode) or with minimal libc dependency (console mode).

## The "Fidelity" Mission
The framework is named "Fidelity" because it **preserves memory and type safety** through the entire compilation pipeline to native code:
- **Preserves type fidelity**: F# types map to precise native representations, never erased
- **Preserves memory fidelity**: Compiler-verified lifetimes, deterministic allocation (stack/arena)
- **No runtime**: The generated binary has the same safety properties as the source, enforced at compile time

## Tech Stack
- **Language**: F# (.NET 9.0)
- **Build System**: dotnet CLI / MSBuild
- **Core Dependencies**:
  - FSharp.Compiler.Service 43.9.300 (parsing, type checking, semantic analysis)
  - XParsec 0.1.0 (parser combinators)
  - Argu 6.1.1 (CLI argument parsing)
  - FSharp.SystemTextJson 1.2.42 (JSON serialization)
- **Output Toolchain**: MLIR → LLVM → Native binary

## Compilation Pipeline
```
F# Source → FCS → PSG → Nanopasses → Alex/Zipper → MLIR → LLVM → Native Binary
```

### Core Components
1. **FCS (F# Compiler Services)** - `/src/Core/FCS/` - Parsing, type checking, semantic analysis
2. **PSG (Program Semantic Graph)** - `/src/Core/PSG/` - Unified IR correlating syntax with semantics
3. **Nanopasses** - `/src/Core/PSG/Nanopass/` - Small, single-purpose PSG transformations
4. **Alex** - `/src/Alex/` - Multi-dimensional hardware targeting layer with Zipper traversal and Bindings

## External Dependencies
- **Alloy** - `/home/hhh/repos/Alloy/src/` - Self-contained F# standard library for native compilation
- Reference resources at `~/repos/fsharp`, `~/repos/fslang-spec`, `~/triton-cpu`, `~/repos/mlir-hs`
