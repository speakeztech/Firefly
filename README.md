# Firefly: F# to Native Compiler with Zero-Allocation Guarantees

<table>
  <tr>
    <td align="center" width="100%">
      <strong>⚠️ Caution: Experimental ⚠️</strong><br>
      This project is in early development and <i>not</i> intended for production use.
    </td>
  </tr>
</table>

Firefly is a novel F# compiler that brings the expressiveness and safety of functional programming directly to native code without runtime dependencies or heap allocations. Built as an orchestrating .NET CLI tool (similar to [Fable](https://github.com/fable-compiler/Fable)), Firefly leverages [F# Compiler Services](https://fsharp.github.io/fsharp-compiler-docs/fcs/) for parsing and type checking, custom transformations using [XParsec](https://github.com/roboz0r/XParsec) to generate MLIR, and LLVM along with other compiler "backend" options for intelligent static library analysis and linking. The orchestration pipeline progressively lowers F# through MLIR dialects, producing efficient transforms to native executables while guaranteeing zero heap allocations and compile-time resolution of all operations.

## 🎯 Vision

Firefly transforms F# from a managed runtime language into a true systems programming language with hard real-time guarantees. By orchestrating compilation through MLIR, Firefly ensures all memory is stack-allocated, all function calls are statically resolved, and all types have fixed layouts - enabling developers to write everything from embedded firmware to high-performance compute kernels while preserving F#'s elegant syntax and type safety.

Central to Firefly's approach is the Program Semantic Graph (PSG) - a unified representation that combines syntactic structure with rich type information and MLIR mapping metadata. This representation enables comprehensive static analysis and allows library authors to hint at optimal MLIR translations through structured XML documentation.

**Key Innovations:** 
- **Program Semantic Graph (PSG)** unifying syntax, types, and MLIR metadata
- **Zero-allocation guarantee** through compile-time memory management
- **Type-preserving compilation** maintaining F#'s rich type system throughout the pipeline
- **Intelligent static linking** via PSG graph analysis and selective object extraction
- **Hybrid library binding** architecture allowing per-library static/dynamic decisions
- **Progressive lowering** through MLIR dialects with continuous verification

## 🏗️ Architecture

```
F# Source Code + XML Documentation
    ↓ (F# Compiler Services parses & type-checks)
Type-checked AST + Symbol Information
    ↓ (PSG construction merges syntax, types, and metadata)
Program Semantic Graph (PSG)
    ↓ (Type-aware reachability analysis & pruning)
Memory-Layout Analyzed PSG
    ↓ (Alex transforms to MLIR operations)
MLIR High-Level Dialects
    ↓ (Progressive lowering through dialects)
Target-specific Dialects
    ↓ (Translation to target IR)
Static Linking + Binding Metadata
    ↓ (analyze archives & links selectively where applicable)
Optimized Native Code
```

### Compilation Pipeline

Firefly operates as an intelligent compilation orchestrator that:

1. **Parses & analyzes** - F# Compiler Services builds a fully type-checked AST and symbol information
2. **Constructs PSG** - Merges syntax tree, type information, and MLIR metadata into a unified graph
3. **Preserves types** - Rich type information flows through the entire pipeline
4. **Computes layouts** - Memory layouts for all types determined at compile time
5. **Transforms progressively** - PSG → MLIR dialects → Target IR
6. **Analyzes statically** - All allocations and calls resolved at compile time
7. **Links selectively** - For LLVM, examine targeted library and extract only needed objects
8. **Optimizes aggressively** - LTO across F# and native library boundaries
9. **Verifies continuously** - Zero allocations, bounded stack, no dynamic dispatch

## Hello World (Stack-Only)

```fsharp
module Examples.HelloWorldDirect

open Alloy
open Alloy.Console
open Alloy.Text.UTF8
open Alloy.Memory

let hello() =
    use buffer = stackBuffer<byte> 256
    Prompt "Enter your name: "
    
    let name = 
        match readInto buffer with
        | Ok length -> spanToString (buffer.AsReadOnlySpan(0, length))
        | Error _ -> "Unknown Person"
    
    let message = sprintf "Hello, %s!" name 
    WriteLine message

[<EntryPoint>]
let main argv = 
    hello()
    0
```

Compile and verify zero allocations:
```bash
firefly compile HelloWorldDirect.fidproj --output hello
# compiler console output
./hello
```

### Library With MLIR Hints

Alloy library functions use XML documentation to provide MLIR mapping hints where necessary:

```fsharp
namespace Alloy.Memory

/// <summary>Allocates memory on the stack</summary>
/// <param name="count">Number of elements to allocate</param>
/// <returns>Pointer to allocated memory</returns>
/// <mlir:dialect>memref</mlir:dialect>
/// <mlir:op>alloca</mlir:op>
/// <mlir:params>element_type={T}</mlir:params>
let inline stackalloc<'T when 'T : unmanaged> (count: int) : nativeptr<'T> =
    NativePtr.stackalloc<'T> count
```

The compiler extracts these hints during PSG construction, enabling optimal MLIR code generation.

### Embedded Message Parser Example

```fsharp
module Examples.MessageParser

open Alloy
open System

type MessageType = 
    | Heartbeat 
    | Data 
    | Error

// Discriminated union compiles to fixed 8-byte stack value
[<Struct>]
type Message = {
    Type: MessageType
    Length: uint16
    Checksum: uint32
}

let parseMessage (buffer: ReadOnlySpan<byte>) =
    if buffer.Length < 8 then
        Error "Too short"
    else
        // All operations resolve to direct memory access
        let msgType = LanguagePrimitives.EnumOfValue<byte, MessageType> buffer.[0]
        let length = BinaryPrimitives.ReadUInt16BigEndian(buffer.Slice(2))
        let checksum = BinaryPrimitives.ReadUInt32BigEndian(buffer.Slice(4))
        Ok { Type = msgType; Length = length; Checksum = checksum }
```

## 🎛️ Configuration

Firefly projects use a ".fidproj" project file format with TOML for fine-grained compilation control:

```toml
[package]
name = "embedded_controller"
version = "1.0.0"
max_stack_size = 4096  # Enforce stack bounds

[dependencies.crypto_lib]
version = "0.2.0"
binding = "static"
# Only link required objects from archive
selective_linking = true

[binding]
default = "static"  # Prefer static for embedded

[compilation]
# All allocations must be provable at compile time
require_static_memory = true
# Closures converted to explicit parameters
eliminate_closures = true

[profiles.development]
# Keep intermediates for inspection
keep_intermediates = true
# Generate VSCode debug info
generate_psg_explorer = true
# Minimal optimization for faster builds
optimize = false

[profiles.release]
# Aggressive inlining for zero-cost abstractions
inline_threshold = 100
# Link-time optimization across F#/C boundaries
lto = "full"
# Profile-guided optimization
use_pgo = true
```

## 🔬 Development Workflow

### Build Pipeline

The Firefly compilation process leverages multiple tools in concert:

```bash
# Standard build invocation
firefly build --release --target thumbv7em-none-eabihf
```

## 🎯 Memory & Execution Guarantees

### ✅ Enforced at Compile Time

- **Zero heap allocations** - Everything on stack or in static data
- **Fixed-size types** - All types have compile-time known sizes
- **Static dispatch** - All function calls resolved at compile time
- **Bounded stack** - Maximum stack usage computed and verified
- **No hidden allocations** - Closures pruned to explicit parameters

### 🚧 Transformation Examples

```fsharp
// F# Source with apparent allocations
let data = [| 1; 2; 3; 4; 5 |]
let doubled = data |> Array.map ((*) 2)

// Firefly transforms to:
// - Stack allocated fixed array  
// - In-place transformation
// - Zero heap usage
```

```fsharp
// F# closure that captures variables
let createAdder x =
    fun y -> x + y

// Firefly transforms to:
// - Static function with explicit parameters
// - No allocation for closure environment
// - Direct function call at use sites
```

## 📋 Roadmap

### Phase 1: Foundation (Current)
- ✅ Basic F# to MLIR pipeline
- ✅ Stack-only transformations  
- 🚧 Program Semantic Graph construction
- 🚧 Type-preserving compilation
- 🚧 Memory layout analysis
- 🚧 Static library selective linking
- 🚧 Discriminated union compilation

### Phase 2: Platform Expansion
- 📋 ARM Cortex-M targeting
- 📋 RISC-V embedded support
- 📋 SIMD optimizations
- 📋 UI Framework (WPF + LVGL)
- 📋 Formal verification integration

### Phase 3: Advanced Features  
- 📋 Computation expression transforms
- 📋 Cross-compilation profiles
- 📋 GPU kernel generation
- 📋 VSCode integration with PSG explorer

## 🤝 Contributing

We will welcome contributions after a solid baseline is established. Areas of particular interest:

- **Program Semantic Graph**: Techniques for merging syntax and semantic information
- **Type-Preserving Transformations**: Techniques for maintaining F#'s rich type system
- **Memory Layout Algorithms**: Advanced layout strategies for complex types
- **Zero-Allocation Patterns**: Novel stack-based algorithms for F# constructs
- **MLIR Optimizations**: Passes for better stack frame merging
- **Platform Targets**: Backend support for embedded architectures
- **VSCode Features**: Debugging, profiling, and visualization tools
- **Verification**: Formal proofs of transformation correctness

## 📄 License

Apache 2.0 License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Don Syme and F# Language Contributors**: For creating an elegant and capable functional language
- **Chris Lattner and MLIR Contributors**: For pioneering multi-level IR compilation
- **LLVM Community**: For robust code generation infrastructure in TableGen
- **Rust Community**: For demonstrating zero-cost abstractions in systems programming
- **Fable Project**: For clearly showing how F# can target alternative environments
- **Ada/SPARK Community**: For inspiration on proven memory-safe systems programming