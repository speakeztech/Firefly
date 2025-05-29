# Firefly: F# to Native Compiler

<table>
  <tr>
    <td align="center" width="100%">
      <strong>⚠️ Caution: Experimental ⚠️</strong><br>
      This project is in early development and *not* intended for production use.
    </td>
  </tr>
</table>

> **Bringing F#'s elegance to systems programming through MLIR and LLVM**

Firefly is a revolutionary F# compiler that brings the expressiveness and safety of functional programming directly to native code without runtime dependencies. Built as a .NET CLI tool (similar to [Fable](https://github.com/fable-compiler/Fable)), Firefly uses [Fantomas](https://fsprojects.github.io/fantomas/) which leans into [F# Compiler Services](https://fsharp.github.io/fsharp-compiler-docs/fcs/) along with our own tranforms using [XParsec](https://github.com/roboz0r/XParsec) to bring functional code and memory into MLIR. That pipeline then takes over to produce LLVM and generates efficient native executables while preserving F#'s type safety, pattern matching, and functional composition.

## 🎯 Vision

Firefly transforms F# from a managed runtime language into a true systems programming language, enabling developers to write everything from embedded firmware to high-performance servers using the same elegant functional paradigms.

**Key Innovation:** Hybrid library binding architecture that allows per-library decisions between static and dynamic linking, all while maintaining a consistent F# development experience.

## 🏗️ Architecture

```
F# Source Code
    ↓ (F# Compiler Services parses & type-checks)
F# AST  
    ↓ (Firefly transforms "Oak AST" into MLIR)
MLIR Operations
    ↓ (MLIR progressive lowering - monitored by Firefly)
LLVM IR
    ↓ (Native code generation with hybrid linking)
Native Executable
```

### Core Components

- **🔥 Firefly**: Main compiler CLI tool - F# AST to MLIR transformation
- **🐰 Dabbit**: AST transformation engine using XParsec-style combinators  
- **🚀 Farscape**: C/C++ binding generator (future component)
- **⚡ Alloy**: Dependency-free base libraries for native F#
- **📡 BAREWire**: Zero-copy serialization system (future component)

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

### Hello World

Create `hello.fs`:
```fsharp
module Examples.HelloWorld

let hello() =
    printf "Enter your name: "
    let name = stdin.ReadLine()
    printfn "Hello, %s!" name

[<EntryPoint>]
let main argv =
    hello()
    0 
```

Compile to native:
```bash
firefly compile hello.fs --output hello(.exe) --target desktop
./hello(.exe)
```

### Time Loop Example

```fsharp
module Examples.TimeLoop

open Alloy

let displayTime() =  
    let mutable counter = 0
    
    while counter < 5 do
        let utcNow = Alloy.Time.now()
        printfn "Time %d: %A" counter utcNow
        Alloy.Time.sleep 1000
        counter <- counter + 1

displayTime()
```

## 📚 Project Structure

```
firefly/
├── src/
│   ├── Firefly.CLI/              # Main compiler CLI tool
│   │   ├── Program.fs            # CLI entry point
│   │   ├── Commands/             # CLI command implementations
│   │   └── Configuration/        # Project configuration parsing
│   │
│   ├── Firefly.Core/             # Core compiler functionality
│   │   ├── Parsing/              # F# source parsing
│   │   ├── TypeChecking/         # F# type system integration
│   │   └── Pipeline/             # Compilation pipeline orchestration
│   │
│   ├── Dabbit/                   # AST to MLIR transformation
│   │   ├── Transforms/           # XParsec-style transform combinators
│   │   ├── MLIR/                 # MLIR operation builders
│   │   └── Binding/              # Library binding strategy handling
│   │
│   └── Alloy/                    # Base libraries for native F#
│       ├── Core.fs               # Core operations and collections
│       ├── Numerics.fs           # Zero-dependency math operations
│       ├── Time/                 # Platform-specific time implementations
│       └── Memory/               # Memory management utilities
│     
├── tests/                        # Test suite
│   ├── Unit/                     # Unit tests for components
│   ├── Integration/              # End-to-end compilation tests
│   └── Examples/                 # Compiled example validation
│
└── docs/                         # Documentation
    ├── architecture/             # Architectural documentation
    ├── language-support/         # F# feature support matrix
    └── binding-strategies/       # Library binding guide

```

## 🎛️ Configuration

Fidelity framework projects use TOML configuration for build settings and binding strategies:

```toml
[package]
name = "my_application"
version = "1.0.0"

[dependencies]
# Static binding for security-critical components
crypto_lib = { version = "1.2.0", binding = "static" } 

# Dynamic binding for system components
system_ui = { version = "2.1.0", binding = "dynamic" }

[binding]
default = "dynamic"

[profiles.development]
binding.default = "dynamic"
optimize = false
keep_intermediates = true

[profiles.release]  
binding.default = "dynamic"
binding.overrides = { crypto_lib = "static" }
optimize = true
```

The project file extension is ".fidproj" to distinguish itself from ".fsproj" .NET/XML based project structure.

## 🧬 Hybrid Binding Architecture

Firefly's revolutionary approach allows fine-grained control over library integration:

### Consistent F# API
```fsharp
// Same code works with any binding strategy
open CryptoLibrary
let hash = Crypto.computeHash(data)
```

### Flexible Binding Configuration
```toml
# Per-library binding decisions
crypto_lib = { binding = "static" }   # Security-critical
ui_framework = { binding = "dynamic" } # Large, shared component  
```

### MLIR Output Adapts Automatically
```mlir
// Static binding - direct function reference
func.func private @crypto_computeHash(%arg0: !llvm.ptr<i8>) -> !llvm.ptr<i8>
    attributes { llvm.linkage = #llvm.linkage<external> }

// Dynamic binding - P/Invoke preserved  
func.func private @ui_createWindow(%arg0: !llvm.ptr<i8>) -> !llvm.ptr<i8>
    attributes { fidelity.dll_import = "ui_framework" }
```

## 🔬 XParsec-Style Transforms

Dabbit uses compositional transforms for clean F# to MLIR conversion:

```fsharp
// Compositional transform combinators
let transformBinding = 
    extractBindingInfo 
    >>= transformFunction
    >>= optimizeForTarget
    >>= addToModule

// Pattern-based transforms for F# constructs
let transformExpression : ASTTransform<SynExpr> = fun expr context ->
    match expr with
    | SynExpr.App(_, _, func, arg, _) -> 
        createFunctionCall context func arg
    | SynExpr.While(_, condition, body, _) ->
        createWhileLoop context condition body
    | SynExpr.Let(_, bindings, body, _) ->
        createLetBinding context bindings body
```

## 🎯 Supported F# Features

### ✅ Currently Supported
- Let bindings and basic functions
- Primitive types (int, float, string, bool)
- Basic I/O (printf, printfn, stdin)
- While loops and mutable variables
- Platform API calls (time, sleep, etc.)
- Pattern matching (basic)

### 🚧 In Progress  
- Function composition and piping
- Discriminated unions
- Record types
- More pattern matching scenarios
- Exception handling
- Initial memory mapping

### 📋 Planned
- Computation expressions
- Generic types and functions
- Advanced pattern matching
- Module system
- Async workflows (native, no Task/async)
- BAREWire memory pre-optimization and schema publishing

## 🏁 Performance Goals

- **Startup time**: Sub-millisecond for small programs
- **Memory usage**: Initially - *no* garbage collector overhead
- **Binary size**: Competitive with C/C++ for equivalent functionality
- **Compile time**: Fast incremental compilation via MLIR caching

## 🤝 Contributing

We welcome contributions! Areas of particular interest:

- **F# Language Features**: Help expand F# construct support
- **Platform Support**: Add Linux/macOS platform implementations  
- **MLIR Dialects**: Create greater XParsec combinator coverage for MLIR dialects of interest
- **Binding Generators**: Extend Farscape for more C/C++ scenarios
- **Optimization**: MLIR pass development and tuning

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Don Syme and F# Language Contributors**: For creating an elegant functional language
- **.NET Engineering**: For creating the foundation that Fable and Fidelity departs from
- **Mono and Xamarin**: For extending the vision of .NET to support MacOS and Linux platforms
- **Fable Project**: For demonstrating F# compilation to other targets
- **MLIR/LLVM Contributors**: For providing the compilation infrastructure  
- **Mojo Language**: For pioneering the "frontend to MLIR" approach

---

**Firefly: Where functional programming honors the true power and performance of systems programming** 🔥