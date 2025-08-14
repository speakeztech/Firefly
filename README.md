# Firefly: F# to Native Compiler with Deterministic Memory Management

<table>
  <tr>
    <td align="center" width="100%">
      <strong>⚠️ Caution: Experimental ⚠️</strong><br>
      This project is in early development and <i>not</i> intended for production use.
    </td>
  </tr>
</table>

Firefly is a novel F# compiler that brings the expressiveness and safety of functional programming directly to native code without runtime dependencies or garbage collection. Built as an orchestrating .NET CLI tool (similar to [Fable](https://github.com/fable-compiler/Fable)), Firefly leverages [F# Compiler Services](https://fsharp.github.io/fsharp-compiler-docs/fcs/) for parsing and type checking, custom transformations to generate MLIR, and LLVM for native code generation. The compilation pipeline progressively lowers F# through MLIR dialects, producing efficient native executables with deterministic memory management and compile-time resolution of operations.

## 🎯 Vision

Firefly transforms F# from a managed runtime language into a true systems programming language with deterministic memory guarantees. By orchestrating compilation through MLIR, Firefly provides flexible memory management strategies - from zero-allocation stack-based code to arena-managed bulk operations and structured concurrency through actors. This enables developers to write everything from embedded firmware to high-performance services while preserving F#'s elegant syntax and type safety.

Central to Firefly's approach is the Program Semantic Graph (PSG) - a representation that combines syntactic structure with rich type information and optimization metadata. This enables comprehensive static analysis and allows the compiler to choose optimal memory strategies based on usage patterns.

**Key Innovations:** 
- **Flexible memory strategies** from zero-allocation to arena-based management
- **Deterministic resource management** through RAII principles and compile-time tracking
- **Type-preserving compilation** maintaining F#'s rich type system throughout the pipeline
- **Progressive lowering** through MLIR dialects with continuous verification
- **Platform-aware optimization** adapting to target hardware characteristics

## 🏗️ Architecture

```
F# Source Code
    ↓ (F# Compiler Services parses & type-checks)
Type-checked AST + Symbol Information
    ↓ (PSG construction with semantic analysis)
Program Semantic Graph (PSG)
    ↓ (Memory strategy selection & optimization)
Memory-Optimized PSG
    ↓ (Transformation to MLIR operations)
MLIR High-Level Dialects
    ↓ (Progressive lowering through dialects)
Target-specific Dialects
    ↓ (Translation to LLVM IR or WAMI)
Native Binary or WebAssembly
```

### Compilation Pipeline

Firefly operates as an intelligent compilation orchestrator that:

1. **Parses & analyzes** - F# Compiler Services builds a fully type-checked AST
2. **Constructs PSG** - Merges syntax tree with type information and semantic metadata
3. **Selects memory strategy** - Chooses appropriate memory management based on usage patterns
4. **Transforms progressively** - PSG → MLIR dialects → Target IR
5. **Optimizes aggressively** - Platform-specific optimizations while preserving safety
6. **Verifies continuously** - Memory safety and resource management guarantees

## Memory Management Philosophy

Unlike traditional approaches that force a single memory model, Firefly adapts to your code's needs:

### Zero-Allocation Baseline
For tight loops and performance-critical code, Firefly can compile to pure stack-based execution:

```fsharp
let processBuffer (data: Span<byte>) =
    use buffer = stackBuffer<byte> 256
    // All operations use stack memory
    // No heap allocations, no GC pressure
    data |> transformInPlace buffer
```

### Arena Memory for Bulk Operations
When dynamic allocation is needed, arena-based management provides deterministic cleanup:

```fsharp
let processDataset records = 
    use arena = Arena.create 10_000_000  // 10MB arena
    // All allocations within scope use the arena
    // Bulk deallocation at scope exit - O(1) cleanup
    records |> processWithArena arena
```

### RAII-Based Resource Management
The compiler automatically tracks resource lifetimes and inserts cleanup:

```fsharp
let processFile() = async {
    let! file = File.openAsync "data.txt"    // Compiler tracks this resource
    let! data = file.readAsync()
    return processData data
}   // File automatically closed here - no explicit disposal needed
```

### Future: Actor-Based Memory Management
*In development:* The Olivier actor model will provide structured concurrency with per-actor memory arenas, enabling efficient message passing and isolated memory management for concurrent systems.

## Hello World Examples

### Simple Direct Output
```fsharp
module HelloWorld

open Alloy.Console

[<EntryPoint>]
let main argv = 
    WriteLine "Hello, World!"
    0
```

### Interactive with Stack Memory
```fsharp
module HelloWorldInteractive

open Alloy
open Alloy.Console
open Alloy.Memory

let hello() =
    use buffer = stackBuffer<byte> 256
    Prompt "Enter your name: "
    
    let name = 
        match readInto buffer with
        | Ok length -> buffer.AsSpan(0, length) |> toString
        | Error _ -> "World"
    
    WriteLine $"Hello, {name}!"

[<EntryPoint>]
let main argv = 
    hello()
    0
```

Compile with:
```bash
firefly compile HelloWorld.fidproj --output hello
./hello
```

## 🎛️ Configuration

Firefly projects use ".fidproj" files with TOML configuration:

```toml
[package]
name = "my_app"
version = "1.0.0"

[compilation]
# Memory management strategy
memory_model = "mixed"  # "zero_alloc" | "arena" | "mixed"

# Stack size for deterministic stack usage
max_stack_size = 4096

# Enable arena memory pools
enable_arenas = true
arena_default_size = 1_000_000

[optimization]
inline_threshold = 100
eliminate_closures = true  # Convert to explicit parameters

[profiles.release]
lto = "full"
optimize = true
```

## 🔬 Development Workflow

```bash
# Build with default settings
firefly build

# Build with specific memory model
firefly build --memory-model zero_alloc

# Build for embedded target
firefly build --target thumbv7em-none-eabihf

# Analyze memory usage
firefly analyze --show-memory-layout
```

## 🎯 Guarantees and Trade-offs

### ✅ What Firefly Provides

- **Deterministic memory management** - No GC pauses, predictable performance
- **Flexible memory strategies** - Choose the right approach for each component
- **Type and memory safety** - Compile-time verification where possible
- **Native performance** - Direct compilation to machine code
- **Async without allocations** - Delimited continuations enable zero-allocation async

### 🔄 Design Choices

- **Not purely stack-based** - Uses appropriate memory strategies for different scenarios
- **RAII over GC** - Automatic but deterministic resource management
- **Explicit when needed** - Some patterns require explicit memory strategy choices
- **Platform-aware** - Different targets may use different memory implementations

## 📋 Roadmap

### Phase 1: Foundation (Current)
- ✅ Basic F# to MLIR pipeline
- ✅ Stack memory operations
- 🚧 Arena memory management
- 🚧 RAII-based resource tracking
- 🚧 Async via delimited continuations

### Phase 2: Advanced Memory Models
- 📋 Linear types for zero-copy operations
- 📋 Actor-based memory isolation (Olivier)
- 📋 Cross-process memory coordination (Prospero)
- 📋 Reference sentinels for distributed systems

### Phase 3: Platform Expansion
- 📋 WebAssembly via WAMI
- 📋 Embedded ARM Cortex-M
- 📋 GPU compute kernels
- 📋 Hardware accelerator support

## 🤝 Contributing

We will welcome contributions after establishing a solid baseline. Areas of particular interest:

- **Memory optimization patterns** - Novel approaches to deterministic memory management
- **MLIR dialect design** - Preserving F# semantics through compilation
- **Platform targets** - Backend support for new architectures
- **Verification** - Formal proofs of memory safety properties

## 📄 License

Dual Apache 2.0 and Commercial License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Don Syme and F# Contributors**: For creating an elegant functional language
- **MLIR Community**: For the multi-level IR infrastructure
- **LLVM Community**: For robust code generation
- **Rust Community**: For demonstrating zero-cost abstractions in systems programming
- **Fable Project**: For showing F# can target alternative environments