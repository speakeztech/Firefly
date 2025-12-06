# Alex Architecture Overview

## The "Library of Alexandria" Model

Alex serves as Firefly's **multi-dimensional hardware targeting layer** - a comprehensive knowledge base that maps high-level F# semantics to platform-optimized machine code. Unlike a simple 1:1 mirror of Alloy, Alex implements a **fan-out architecture** where a single Alloy abstraction can emit different code patterns based on:

1. **Target Architecture** (x86-64, ARM Cortex-M/A, RISC-V, WASM)
2. **Operating System** (Linux, Windows, macOS, bare metal)
3. **Hardware Capabilities** (SIMD extensions, cache hierarchy, instruction set features)
4. **Coeffect Analysis** (Pure, AsyncBoundary, MemoryPattern, etc.)

## Core Philosophy

### Semantic Preservation Through Layers

Alex preserves semantic intent from F# source through to native code:

```
F# Source (Intent)
    ↓
Alloy Library (BCL-sympathetic API)
    ↓
PSG (Program Semantic Graph)
    ↓
Alex Bindings (Platform-aware patterns)
    ↓
MLIR (Target-specific dialects)
    ↓
LLVM IR (Architecture-specific)
    ↓
Native Code (Optimal for target)
```

### The Fan-Out Pattern

From a single Alloy call like `Time.currentTicks()`, Alex fans out to multiple emission strategies:

```
Alloy.Time.currentTicks()
    │
    ├─► Linux x86-64:    clock_gettime via syscall
    ├─► Linux ARM64:     clock_gettime via svc #0
    ├─► Windows x86-64:  QueryPerformanceCounter via kernel32
    ├─► macOS x86-64:    mach_absolute_time via libSystem
    ├─► macOS ARM64:     mach_absolute_time via libSystem
    ├─► ARM Cortex-M:    SysTick register access
    ├─► RISC-V:          mtime CSR read
    └─► WASM:            performance.now() import
```

## Current Directory Structure

```
/home/hhh/repos/Firefly/src/
├── Alex/
│   └── Pipeline/
│       ├── CompilationTypes.fs      # Phase enums, config, error types
│       ├── CompilationOrchestrator.fs # Pipeline orchestration
│       ├── FCSPipeline.fs           # F# Compiler Services integration
│       ├── LoweringPipeline.fs      # MLIR → LLVM lowering
│       └── OptimizationPipeline.fs  # LLVM optimization passes
│
├── Core/
│   ├── XParsec/
│   │   └── Foundation.fs            # Unified error types, CompilerResult<'T>
│   ├── PSG/
│   │   ├── Types.fs                 # PSGNode, PSGEdge, EdgeKind
│   │   ├── Builder.fs               # PSG construction
│   │   ├── Correlation.fs           # Symbol correlation
│   │   └── Reachability.fs          # Dead code elimination
│   ├── MLIR/
│   │   └── Emitter.fs               # Current monolithic emission
│   ├── Types/
│   │   ├── Dialects.fs              # MLIR dialect enums
│   │   └── MLIRTypes.fs             # MLIR type system, OutputKind
│   ├── FCS/
│   │   └── ProjectContext.fs        # Project loading
│   ├── Meta/
│   │   └── Parser.fs                # XML doc pattern hints
│   └── IngestionPipeline.fs         # Full analysis pipeline
```

## Planned Extension: Alex Bindings

The following structure will enable composable, platform-aware code generation:

```
/home/hhh/repos/Firefly/src/Alex/
├── Pipeline/                        # (existing)
├── Bindings/
│   ├── BindingTypes.fs              # Core binding abstractions
│   ├── PlatformRegistry.fs          # Platform detection & selection
│   ├── PatternMatcher.fs            # XParsec-based symbol matching
│   │
│   ├── Console/                     # Console I/O bindings
│   │   ├── ConsoleBindings.fs       # Common patterns
│   │   ├── Linux.fs                 # Linux write/read syscalls
│   │   ├── Windows.fs               # Windows WriteConsole
│   │   └── MacOS.fs                 # macOS write syscalls
│   │
│   ├── Time/                        # Time operation bindings
│   │   ├── TimeBindings.fs          # Common time patterns
│   │   ├── Linux.fs                 # clock_gettime, nanosleep
│   │   ├── Windows.fs               # QueryPerformanceCounter, Sleep
│   │   └── MacOS.fs                 # mach_absolute_time, nanosleep
│   │
│   ├── Memory/                      # Memory operation bindings
│   │   ├── MemoryBindings.fs        # Common patterns
│   │   └── [platform files]
│   │
│   └── Syscalls/                    # Low-level syscall emission
│       ├── SyscallTypes.fs          # Syscall abstractions
│       ├── Linux_x86_64.fs          # Linux AMD64 syscalls
│       ├── Linux_ARM64.fs           # Linux AArch64 syscalls
│       ├── Windows_x86_64.fs        # Windows x64 calls
│       ├── MacOS_x86_64.fs          # macOS Intel syscalls
│       └── MacOS_ARM64.fs           # macOS Apple Silicon syscalls
│
├── Transformations/
│   ├── ClosureElimination.fs        # Closure → parameter passing
│   └── StackAllocation.fs           # Heap → stack promotion
│
├── CodeGeneration/
│   ├── TypeMapping.fs               # F# → MLIR type mapping
│   └── EmissionContext.fs           # Emission state management
│
└── Integration/
    └── FargoIntegration.fs          # fidproj target resolution
```

## XParsec as Central Glue

XParsec Foundation provides the unified type system across all Alex modules:

### Error Handling
```fsharp
type CompilerResult<'T> =
    | Success of 'T
    | CompilerFailure of FireflyError list

// All pipeline phases return CompilerResult<'T>
// Enables: result1 |> bind phase2 |> bind phase3
```

### Error Classification
```fsharp
type FireflyError =
    | SyntaxError of position * message * context
    | ConversionError of phase * source * target * message
    | TypeCheckError of construct * message * location
    | InternalError of phase * message * details
    | ParseError of position * message
    | DependencyResolutionError of symbol * message
```

### Dependency Analysis
```fsharp
type DependencyType =
    | DirectCall          // Regular function call
    | AlloyLibraryCall    // Call into Alloy (triggers binding lookup)
    | ConstructorCall     // Type construction
    | ExternalCall        // External library call
```

## Key Architectural Decisions

### 1. PSG-Based Compilation
The Program Semantic Graph preserves semantic relationships that are lost in traditional AST-to-IR compilation. This enables:
- Symbol correlation (position → semantic info)
- Reachability analysis (dead code elimination)
- Pattern recognition for special emission

### 2. External Tool Integration
Rather than reimplementing MLIR/LLVM lowering, Alex delegates to:
- `mlir-opt` for dialect conversion
- `mlir-translate` for LLVM IR generation
- `opt` for LLVM optimization
- `llc` for machine code generation

This leverages battle-tested infrastructure while focusing Alex development on F# → MLIR transformation.

### 3. OutputKind-Driven Emission
```fsharp
type OutputKind =
    | Console       // Uses libc, main is entry point
    | Freestanding  // No libc, _start wrapper, direct syscalls
    | Embedded      // No OS, custom startup, memory-mapped I/O
    | Library       // No entry point, exported symbols
```

### 4. Immutable Context Threading
All transformation phases thread immutable context through pipeline stages, avoiding global mutable state and enabling parallel compilation in the future.

## Integration with fidproj

The `.fidproj` file serves as the "starting position" for hardware-aware compilation:

```toml
[package]
name = "my_app"
output-kind = "freestanding"

[build]
target = "x86_64-unknown-linux-gnu"

[target.'cfg(target_arch = "aarch64")'.build]
target = "aarch64-unknown-linux-gnu"

[target.'cfg(target_os = "windows")'.build]
target = "x86_64-pc-windows-msvc"
```

When `fargo build` or `fpm build` executes:
1. Parse fidproj for target configuration
2. Resolve target triple
3. Consult Alex's platform registry
4. Select appropriate binding modules
5. Generate platform-optimized MLIR

## Progression of Forcing Functions

Each proof-of-concept sample forces new capabilities in Alex:

| Sample | Forces | Alex Module |
|--------|--------|-------------|
| HelloWorld | String emission, Console syscalls | Bindings/Console/* |
| TimeLoop | Platform time APIs, sleep | Bindings/Time/* |
| Blinky | ARM Cortex-M, GPIO, interrupts | Bindings/Embedded/* |
| GPU kernels | SPIR-V, memory coalescing | Bindings/GPU/* |
| Actors | Zero-copy, cache placement | Bindings/Memory/* |

## Next Steps

1. **Create Binding Infrastructure** (Alex/Bindings/BindingTypes.fs)
2. **Implement Time Bindings** for Linux, Windows, macOS
3. **Refactor Emitter.fs** to use binding registry
4. **Add Platform Detection** via target triple parsing
5. **Document Coeffect Integration** for optimization hints

---

## References

### Core Architectural Documents
- **Hyping Hypergraphs** (`~/repos/SpeakEZ/hugo/content/blog/Hyping Hypergraphs.md`): Temporal Program Hypergraph vision, bidirectional zippers, recursion schemes for compilation
- **Why F# Is A Natural Fit for MLIR** (`~/repos/SpeakEZ/hugo/content/blog/Why FSHarp Is A Natural Fit for MLIR.md`): SSA is functional programming, delimited continuations, semantic preservation

### Zipper Implementations
- **Tomas Petricek's Zipper Query**: https://tomasp.net/blog/tree-zipper-query.aspx/ - Tree zipper with Path context for navigation
- **Polymorphic Maybe with Zipper**: https://fssnip.net/aX/title/Polymorphic-Maybe-monad-with-default-value - Zipper operations with computational fallback

### Related Blog Entries
- **Building Firefly With Alloy** (`~/repos/SpeakEZ/hugo/content/blog/Building Firefly With Alloy.md`)
- **Context Aware Compilation** (`~/repos/SpeakEZ/hugo/content/blog/Context Aware Compilation.md`)
- **Speed And Safety With Graph Coloring** (`~/repos/SpeakEZ/hugo/content/blog/Speed And Safety With Graph Coloring.md`)
- **Coeffects And Codata In Firefly** (`~/repos/SpeakEZ/hugo/content/blog/Coeffects And Codata In Firefly.md`)

### External Inspirations
- **mlir-hs** (`~/repos/mlir-hs`): Haskell MLIR bindings with Builder monad pattern and TableGen code generation
- **XParsec** (`~/repos/XParsec`): Parser combinator library - foundation for PSG pattern recognition

---

*Last Updated: December 2024*
