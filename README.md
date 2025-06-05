# Firefly: F# to Native Compiler with Zero-Allocation Guarantees

<table>
  <tr>
    <td align="center" width="100%">
      <strong>⚠️ Caution: Experimental ⚠️</strong><br>
      This project is in early development and <i>not</i> intended for production use.
    </td>
  </tr>
</table>

Firefly is a novel F# compiler that brings the expressiveness and safety of functional programming directly to native code without runtime dependencies or heap allocations. Built as an orchestrating .NET CLI tool (similar to [Fable](https://github.com/fable-compiler/Fable)), Firefly leverages [Fantomas](https://fsprojects.github.io/fantomas/) and [F# Compiler Services](https://fsharp.github.io/fsharp-compiler-docs/fcs/) for parsing, custom transformations using [XParsec](https://github.com/roboz0r/XParsec) to generate MLIR, and [LLVM.NET](https://github.com/UbiquityDotNET/Llvm.NET) for intelligent static library analysis and linking. The orchestration pipeline progressively lowers F# through MLIR dialects to LLVM IR, producing efficient native executables while guaranteeing zero heap allocations and compile-time resolution of all operations.

## 🎯 Vision

Firefly transforms F# from a managed runtime language into a true systems programming language with hard real-time guarantees. By orchestrating compilation through MLIR, Firefly ensures all memory is stack-allocated, all function calls are statically resolved, and all types have fixed layouts - enabling developers to write everything from embedded firmware to high-performance compute kernels while preserving F#'s elegant syntax and type safety.

Central to Firefly's approach is intelligent static library handling through LLVM.NET. Rather than traditional whole-archive linking that bloats executables, Firefly examines static library archives at build time, traces symbol dependencies from your F# code, and extracts only the required object files. This selective linking means a cryptography library containing 47 object files might contribute just 2-3 objects to your final executable. Combined with link-time optimization across F# and C/C++ boundaries, Firefly delivers both the safety of functional programming and the efficiency of hand-tuned systems code.

**Key Innovations:** 
- **Zero-allocation guarantee** through compile-time memory management
- **Intelligent static linking** via LLVM.NET archive analysis and selective object extraction
- **Hybrid library binding** architecture allowing per-library static/dynamic decisions
- **Progressive lowering** through MLIR dialects with continuous verification

## 🏗️ Architecture

```
F# Source Code
    ↓ (F# Compiler Services parses & type-checks)
F# AST / Oak AST  
    ↓ (Dabbit transforms to MLIR operations)
MLIR High-Level Dialects
    ↓ (Progressive lowering through dialects)
MLIR LLVM Dialect
    ↓ (Translation to LLVM IR)
LLVM IR + Binding Metadata
    ↓ (LLVM.NET analyzes archives & links selectively)
Optimized Native Code
```

### Compilation Pipeline

Firefly operates as an intelligent compilation orchestrator that:

1. **Transforms progressively** - F# → Oak AST → MLIR dialects → LLVM IR
2. **Analyzes statically** - All allocations and calls resolved at compile time
3. **Links selectively** - LLVM.NET examines archives and extracts only needed objects
4. **Optimizes aggressively** - LTO across F# and native library boundaries
5. **Verifies continuously** - Zero allocations, bounded stack, no dynamic dispatch

## 🚀 Quick Start

### Installation

```bash
# Install as global .NET tool
dotnet tool install -g Firefly

# Or build from source
git clone https://github.com/speakez-llc/firefly.git
cd firefly
dotnet build
```

### Hello World (Stack-Only)

Create `hello.fs`:
```fsharp
module Examples.HelloWorld

open Alloy

let hello() =
    // All string operations use stack buffers
    let buffer = NativePtr.stackalloc<byte> 256
    printf "Enter your name: "
    let length = Console.readLine buffer 256
    printfn "Hello, %s!" (Span<byte>(buffer, length))

[<EntryPoint>]
let main argv =
    hello()
    0 
```

Compile and verify zero allocations:
```bash
firefly compile hello.fs --output hello --target embedded
firefly verify hello --no-heap
./hello
```

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

## 📚 Project Structure

```
firefly/
├── src/
│   ├── Firefly.CLI/              # Orchestrating compiler CLI
│   │   ├── Program.fs            # CLI entry point
│   │   ├── Commands/             # Build, verify, profile commands
│   │   └── Configuration/        # TOML project configuration
│   │
│   ├── Firefly.Core/             # Core compilation pipeline
│   │   ├── Parsing/              # F# to Oak AST conversion
│   │   ├── StaticAnalysis/       # Allocation & binding analysis
│   │   ├── MLIRGeneration/       # XParsec-based MLIR builders
│   │   └── LLVMIntegration/      # LLVM.NET binding resolution
│   │
│   └── Dabbit/                   # AST to MLIR transformation
│       ├── StackTransforms/      # Heap → Stack conversions
│       ├── ClosureElimination/   # Closure → Explicit params
│       ├── UnionLayouts/         # Fixed-size union compilation
│       └── BindingMetadata/      # Static binding attributes
│     
├── tests/
│   ├── AllocationTests/          # Verify zero-heap guarantee
│   ├── StackBounds/              # Maximum stack usage tests
│   └── StaticResolution/         # Ensure no dynamic dispatch
│
└── docs/
    ├── zero-allocation/          # Memory management guide
    ├── mlir-patterns/            # Common F# → MLIR transforms
    └── static-linking/           # LLVM.NET integration guide
```

## 🎛️ Configuration

Firefly projects use TOML for fine-grained compilation control:

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
# Keep MLIR for inspection
keep_intermediates = true
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

# What happens internally:
# 1. F# Compiler Services → Type-checked AST
# 2. Fantomas/Oak → Normalized AST representation  
# 3. XParsec patterns → MLIR generation
# 4. MLIR passes → Progressive lowering
# 5. LLVM.NET → Archive analysis & selective linking
# 6. LLVM → Optimized native code
```

### Static Linking Intelligence

When Firefly encounters a static library dependency, LLVM.NET provides deep introspection:

```
Analyzing libcrypto.a:
  Total archive size: 892KB (47 object files)
  Tracing symbols from F# code...
  Required: crypto_init, crypto_process, crypto_free
  Found in: init.o (2KB), process.o (5KB), util.o (1KB)
  Linking 3 of 47 objects (8KB vs 892KB)
  Applying LTO across F#/C boundaries...
  Final contribution: 4KB after optimization
```

### Verification Commands

```bash
# Verify zero-allocation guarantee
firefly verify myapp --no-heap --max-stack 8192

# Analyze symbol dependencies
firefly analyze --show-symbol-deps

# Profile-guided optimization
firefly build --pgo-data trace.pgo
```

## 🎯 Memory & Execution Guarantees

### ✅ Enforced at Compile Time
- **Zero heap allocations** - Everything on stack or in static data
- **Fixed-size types** - All types have compile-time known sizes
- **Static dispatch** - All function calls resolved at compile time
- **Bounded stack** - Maximum stack usage computed and verified
- **No hidden allocations** - Closures transformed to explicit parameters

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
- ✅ VSCode integration with dual views
- 🚧 Static library selective linking
- 🚧 Discriminated union compilation

### Phase 2: Platform Expansion
- 📋 ARM Cortex-M targeting
- 📋 RISC-V embedded support
- 📋 GPU kernel generation
- 📋 SIMD optimizations

### Phase 3: Advanced Features  
- 📋 Computation expression transforms
- 📋 Type provider integration
- 📋 Cross-compilation profiles
- 📋 Formal verification integration

## 🤝 Contributing

We welcome contributions! Areas of particular interest:

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
- **LLVM Community**: For robust code generation infrastructure
- **Rust Community**: For demonstrating zero-cost abstractions in systems programming
- **Fable Project**: For clearly showing how F# can target alternative environments
- **Ada/SPARK Community**: For inspiration on proven memory-safe systems programming
