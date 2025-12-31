# Firefly Project Overview

## Purpose
Firefly is an ahead-of-time (AOT) compiler for F# targeting native binary output without the .NET runtime. The project compiles F# source code through MLIR to LLVM IR and finally to native executables.

**Primary Goal**: Compile F# code to efficient, standalone native binaries that can run without any runtime dependencies (freestanding mode) or with minimal libc dependency (console mode).

## The "Fidelity" Mission

> **CORE PRINCIPLE**: The entire point of Fidelity is that **the F# compiler controls memory layout - not MLIR, not LLVM**.

The framework is named "Fidelity" because it **preserves type fidelity** through the entire compilation pipeline:

- **Types guide every decision**: Memory regions, access kinds, and other semantic types carry meaning through the ENTIRE pipeline
- **Fidelity makes ALL memory layout decisions**: MLIR/LLVM never determine layout - they implement what Fidelity dictates
- **Erasure at last lowering**: Types ARE erased, but at the LAST possible stage - after Fidelity has made all decisions
- **No runtime**: The generated binary has the same safety properties as the source, enforced at compile time

**Fidelity dictates; LLVM implements.**

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
- **FNCS** - `~/repos/fsnative` - F# Native Compiler Services (FCS fork with native type universe)
- Reference resources at `~/repos/fsharp`, `~/repos/fslang-spec`, `~/triton-cpu`, `~/repos/mlir-hs`

## The Native Type Universe (Critical Insight)

The entire Fidelity framework depends on a **clean native type universe**:
- UTF-8 strings (not System.String)
- Value-type options (not nullable references)
- No `null`, no `obj` base type
- No BCL dependencies

**Key Finding (December 2025)**: The IL import assumption in FCS permeates its entire type-checking layer (3.2MB, 59 files). FNCS cannot be created by pruning - the type checker must be rebuilt. This surface area reduction is essential for Fidelity's success and organic growth as a platform.

Reducing surface area to focus solely on the native type universe enables clean architectural decisions that ripple positively throughout the framework.
