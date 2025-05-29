# Firefly: F# to Native Compiler

<table>
  <tr>
    <td align="center" width="100%">
      <strong>⚠️ Caution: Experimental ⚠️</strong><br>
      This project is in early development and <i>not</i> intended for production use.
    </td>
  </tr>
</table>

Firefly is a novel F# compiler that brings the expressiveness and safety of functional programming directly to native code without runtime dependencies. Built as a .NET CLI tool (similar to [Fable](https://github.com/fable-compiler/Fable)), Firefly leverages [Fantomas](https://fsprojects.github.io/fantomas/) and [F# Compiler Services](https://fsharp.github.io/fsharp-compiler-docs/fcs/) alongside custom transformations using [XParsec](https://github.com/roboz0r/XParsec) to compile functional code into MLIR. The MLIR pipeline then generates LLVM IR and produces efficient native executables while preserving F#'s type safety guarantees. This approach delivers the performance of systems programming with functional programming's developer ergonomics and precision.

## 🎯 Vision

Firefly will transform F# from a managed runtime language into a true systems programming language, enabling developers to write everything from embedded firmware to high-performance distributed workloads across standard and accelerated hardware. The goal is to preserve, as much as possible, the ergonomics that have made F# renowned as an elegant and powerful language.

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

[dependencies.crypto_lib]
version = "0.2.0"
binding = "static"

[binding]
default = "dynamic"

[profiles.development]
optimize = false
keep_intermediates = true

[profiles.release]
optimize = true
```

The project file extension is ".fidproj" to distinguish itself from ".fsproj" .NET/XML based project structure.

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
- Async workflows
- BAREWire memory pre-optimization

## 🤝 Contributing

We welcome contributions! Areas of particular interest:

- **F# Language Features**: Help expand F# construct support
- **Platform Support**: Add platform extensions around OS APIs (Alloy)
- **MLIR Dialects**: Increase XParsec combinator coverage for MLIR dialects (Dabbit)
- **Binding Generators**: Extend Farscape for more C/C++ scenarios
- **Optimization**: MLIR pass development and tuning

## 📄 License

Apache 2.0 License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Don Syme and F# Language Contributors**: For creating an elegant and capable functional language
- **.NET Engineering**: For creating a robust enterprise-grade runtime platform that gave runway to F# ✈️
- **[Mono Project](https://www.mono-project.com/)**: For its original vision to support iOS, Android, MacOS and Linux platforms in .NET
- **Fable Project**: For demonstrating F# compilation to targets beyond the .NET runtime
- **MLIR/LLVM Ecosystem**: For establishing powerful compiler foundations with a huge contributor base  
- **Mojo Language**: For pioneering the "frontend to MLIR" approach

---

**Firefly: Bringing the power of full-precision functional programming to modern systems** 🔥