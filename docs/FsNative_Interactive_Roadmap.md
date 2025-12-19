# F# Native Interactive (FSNI) Roadmap

## Overview

This document explores approaches to providing an interactive, REPL-like experience for F# Native (Fidelity). Unlike standard FSI which relies on .NET's JIT and runtime, FSNI must work within native compilation constraints while still providing the exploratory programming experience F# developers expect.

## The Fundamental Challenge

Standard FSI operates within the .NET runtime:

```
F# Source → FCS → IL → CLR JIT → Execute in managed heap
                         ↑
              Runtime handles everything:
              - Memory management (GC)
              - Type reification
              - Dynamic loading
              - Expression evaluation
```

Fidelity targets native execution:

```
F# Source → FNCS → PSG → Alex → MLIR → LLVM → Native binary
                                         ↑
                     No runtime - everything resolved at compile time:
                     - Deterministic memory (stack/arena)
                     - Types erased after compilation
                     - Static linking
                     - AOT execution
```

The question: How do we provide interactivity without a managed runtime?

## Reference Implementations

### Evcxr (Rust REPL)

[Evcxr](https://github.com/evcxr/evcxr) provides Rust REPL and Jupyter kernel support.

**Approach**: Compilation + Dynamic Loading
- Each evaluation compiles a new shared library (`.so`/`.dylib`)
- Libraries are dynamically loaded into the running process
- State persists across evaluations through careful symbol management
- "It really does invoke rustc and then loads the code with dynamic linking"

**Pros**:
- Uses the real compiler (rustc), not an interpreter
- Full language support
- Works with existing Cargo ecosystem

**Cons**:
- Compilation latency on each evaluation
- Complex state management
- Dynamic linking overhead

### Clang-Repl (C++ Interactive)

[Clang-Repl](https://clang.llvm.org/docs/ClangRepl.html) is an interactive C++ interpreter built on LLVM.

**Approach**: Incremental JIT Compilation
- Uses LLVM's ORC JIT infrastructure
- Compiles C++ to LLVM IR, then JIT-compiles to native code
- Maintains state through `ExecutionSession`
- Supports incremental module addition

**Key architecture**:
```
Input → AST → AST Transform → LLVM IR → JIT → Execute
                   ↓                      ↓
          Value synthesis        ORC ExecutionSession
          (result capture)       (state management)
```

**Pros**:
- Tight compiler integration
- Fast incremental compilation
- Access to full LLVM optimization pipeline

**Cons**:
- Requires deep compiler integration
- Complex architecture

### Swift REPL

[Swift REPL](https://www.swift.org/documentation/lldb/) is built on the LLDB debugger.

**Approach**: Debugger-Integrated JIT
- REPL is actually a debugger session
- Expressions compiled using embedded Swift compiler
- JIT execution within debugger context
- Breakpoints and debugging "for free"

**Key insight**: "Several motivating factors contributed to the decision to use the Swift debugger as a foundation for the Swift REPL. The most obvious benefit is that the Swift REPL is also a full-featured debugger."

**Pros**:
- REPL + Debugger in one tool
- Natural breakpoint/inspection support
- Tight compiler coupling ensures consistency

**Cons**:
- "Developers must use a matched pair of compiler and debugger"
- Heavy infrastructure requirement

## Proposed FSNI Architecture

Given Fidelity's existing use of MLIR and LLVM, the most natural approach combines aspects of Clang-Repl and Evcxr:

### Architecture: Incremental MLIR JIT

```
                    ┌─────────────────────────────────────────────────────┐
                    │                 FSNI Process                         │
                    ├─────────────────────────────────────────────────────┤
                    │                                                      │
                    │  ┌─────────────┐    ┌─────────────────────────────┐ │
                    │  │ Input       │    │ Evaluation Context           │ │
User Input ────────►│  │ Handler     │───►│                              │ │
                    │  │             │    │  ┌─────────────────────────┐ │ │
                    │  └─────────────┘    │  │ Symbol Table            │ │ │
                    │                     │  │ - Bindings: Map<name,val>│ │ │
                    │  ┌─────────────┐    │  │ - Types: Map<name,type>  │ │ │
                    │  │ FNCS        │    │  │ - Functions              │ │ │
                    │  │ Type Check  │◄───┤  └─────────────────────────┘ │ │
                    │  │ + PSG Build │    │                              │ │
                    │  └──────┬──────┘    │  ┌─────────────────────────┐ │ │
                    │         │           │  │ ORC JIT Session          │ │ │
                    │         ▼           │  │ (LLVM ExecutionSession)  │ │ │
                    │  ┌─────────────┐    │  │                          │ │ │
                    │  │ Alex        │    │  │  ┌──────────────────┐   │ │ │
                    │  │ MLIR Gen    │    │  │  │ JITDylib         │   │ │ │
                    │  └──────┬──────┘    │  │  │ (symbol storage) │   │ │ │
                    │         │           │  │  └──────────────────┘   │ │ │
                    │         ▼           │  │                          │ │ │
                    │  ┌─────────────┐    │  └─────────────────────────┘ │ │
                    │  │ MLIR        │────►│                              │ │
                    │  │ → LLVM IR   │    └───────────────────────────────┘ │
                    │  │ → ORC JIT   │                 │                    │
                    │  └─────────────┘                 │                    │
                    │         │                        │                    │
                    │         ▼                        │                    │
                    │  ┌─────────────┐                 │                    │
◄───────────────────│  │ Execute +   │◄────────────────┘                    │
   Output           │  │ Display     │                                      │
                    │  └─────────────┘                                      │
                    │                                                       │
                    └───────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Evaluation Context

The persistent state across REPL evaluations:

```fsharp
type EvaluationContext = {
    /// Symbols defined in this session
    Bindings: Map<string, BoundValue>

    /// Type definitions created in this session
    Types: Map<string, TypeDefinition>

    /// LLVM ORC ExecutionSession
    JitSession: OrcExecutionSession

    /// Current JITDylib for symbol storage
    MainDylib: JITDylib

    /// Accumulated FNCS state (for incremental type checking)
    FncsState: FNCSIncrementalState
}
```

#### 2. Incremental FNCS Mode

FNCS (fsnative) needs an incremental mode for REPL usage:

```fsharp
/// Incremental type checking for REPL
module FNCSIncremental =

    /// Parse and type-check a single expression/declaration
    let checkFragment (context: FNCSIncrementalState) (input: string)
        : Result<TypedFragment * FNCSIncrementalState, FNCSError> =
        // Parse fragment
        // Resolve against existing context
        // Return typed fragment + updated state
        ...

    /// Extend context with new binding
    let addBinding (context: FNCSIncrementalState) (name: string) (value: TypedValue)
        : FNCSIncrementalState =
        ...
```

#### 3. MLIR JIT Integration

Firefly already uses MLIR. The JIT path extends this:

```
Standard Firefly:  MLIR → LLVM IR → Object File → Linker → Executable
FSNI JIT:          MLIR → LLVM IR → ORC JIT → Direct Execution
                                        ↓
                                  Memory-resident code
                                  (no file I/O needed)
```

LLVM's ORC JIT provides:
- `ExecutionSession` - Central state management
- `JITDylib` - Symbol storage and lookup
- Lazy compilation support
- Concurrent execution safety

#### 4. Value Display

Unlike FSI which can use .NET reflection, FSNI needs native value display:

```fsharp
/// Native value rendering
module ValueDisplay =

    /// Render a native value for REPL display
    let display (value: nativeptr<byte>) (typ: NativeType) : string =
        match typ with
        | NativeType.Int32 -> sprintf "%d" (NativePtr.read<int32> value)
        | NativeType.NativeStr ->
            let str = NativePtr.toNativeStr value
            sprintf "\"%s\"" (str.ToString())
        | NativeType.Struct fields ->
            // Render struct fields
            ...
        | NativeType.Function _ ->
            "<function>"
```

This is similar to Clang-Repl's "value synthesis" where expressions are wrapped to capture and display results.

### The Background Firefly Insight

Your observation about running Firefly in the background is key:

```
┌─────────────────────────────────────────────────────────────────┐
│                    FSNI Architecture                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────┐     ┌──────────────────────────────┐  │
│  │ FSNI Shell           │     │ Background Firefly Service    │  │
│  │ (user interaction)   │────►│ (warm compiler state)         │  │
│  │                      │◄────│                               │  │
│  │ - Input handling     │     │ - FNCS ready to type-check   │  │
│  │ - Display formatting │     │ - Alex ready to generate     │  │
│  │ - History            │     │ - MLIR passes cached         │  │
│  │ - Completion         │     │ - ORC JIT session active     │  │
│  └──────────────────────┘     └──────────────────────────────┘  │
│                                          │                       │
│                                          ▼                       │
│                               ┌──────────────────────────────┐  │
│                               │ Incremental Compilation       │  │
│                               │                               │  │
│                               │ Fragment → PSG → MLIR → JIT  │  │
│                               │ (milliseconds, not seconds)   │  │
│                               └──────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Why this matters**:
- Compiler initialization is expensive (parsing Alloy, building symbol tables)
- A warm compiler service amortizes this cost across evaluations
- Similar to how TypeScript's `tsc --watch` keeps the compiler hot
- Aligns with how FSNAC (LSP server) will already maintain compiler state

### Leveraging LLVM/MLIR Directly

Since Firefly already targets MLIR→LLVM, we can leverage:

1. **MLIR ExecutionEngine** - Built-in eager JIT compilation
2. **LLVM ORC JIT** - The same infrastructure Clang-Repl uses
3. **Existing lowering passes** - MLIR→LLVM IR→Machine code

The JIT path is almost "free" given the existing pipeline.

## Implementation Phases

### Phase 0: Foundation (Prerequisite)

**Dependencies**: FNCS stability, Alex maturity

Before FSNI can begin, these must be stable:
- FNCS incremental API
- Alex MLIR generation for all required constructs
- Platform bindings for I/O (display output)

### Phase 1: Minimal Evaluator

**Goal**: Execute single expressions, display results

**Tasks**:
1. Create `fsni` CLI tool skeleton
2. Implement LLVM ORC JIT integration layer
3. Add FNCS incremental fragment parsing
4. Basic expression evaluation (`1 + 1`, `"hello".Length`)
5. Simple value display for primitives

**Validation**:
```
fsni> 1 + 1
val it: int = 2

fsni> "hello"
val it: NativeStr = "hello"
```

### Phase 2: Binding Persistence

**Goal**: Let bindings persist across evaluations

**Tasks**:
1. Implement symbol table for session bindings
2. Extend FNCS incremental state management
3. Handle `let` declarations with proper scoping
4. Support mutable bindings (`let mutable`)

**Validation**:
```
fsni> let x = 42
val x: int = 42

fsni> x * 2
val it: int = 84

fsni> let mutable counter = 0
val mutable counter: int = 0

fsni> counter <- counter + 1
val it: unit = ()

fsni> counter
val it: int = 1
```

### Phase 3: Function Definitions

**Goal**: Define and call functions interactively

**Tasks**:
1. Support `let` function definitions
2. Handle recursive functions
3. Support `inline` functions with SRTP
4. Add function signatures to symbol table

**Validation**:
```
fsni> let square x = x * x
val square: int -> int

fsni> square 5
val it: int = 25

fsni> let rec fib n = if n <= 1 then n else fib (n-1) + fib (n-2)
val fib: int -> int

fsni> fib 10
val it: int = 55
```

### Phase 4: Type Definitions

**Goal**: Define types interactively

**Tasks**:
1. Support record type definitions
2. Support discriminated union definitions
3. Handle type references across evaluations
4. Memory layout calculation for display

**Validation**:
```
fsni> type Person = { Name: NativeStr; Age: int }
type Person

fsni> let john = { Name = "John"; Age = 30 }
val john: Person = { Name = "John"; Age = 30 }

fsni> type Result<'T> = Ok of 'T | Error of NativeStr
type Result<'T>
```

### Phase 5: Alloy Integration

**Goal**: Use Alloy library functions interactively

**Tasks**:
1. Auto-import Alloy.Core
2. Support `open` declarations
3. Handle SRTP resolution in interactive context
4. Platform binding execution

**Validation**:
```
fsni> open Alloy.Console
fsni> WriteLine "Hello from FSNI!"
Hello from FSNI!
val it: unit = ()

fsni> open Alloy.Math
fsni> sqrt 2.0
val it: float = 1.4142135623730951
```

### Phase 6: Jupyter Kernel

**Goal**: Notebook-style interaction

**Tasks**:
1. Create `fsni-jupyter` kernel
2. Implement Jupyter protocol
3. Add rich output support (tables, charts via platform bindings)
4. Documentation integration

**Validation**:
- Run FSNI in Jupyter notebooks
- Display native data structures with formatting

### Phase 7: Debugger Integration (Future)

**Goal**: Swift-style REPL/debugger hybrid

**Tasks**:
1. Investigate LLDB integration patterns
2. Implement breakpoint support
3. Expression evaluation in debug context
4. Variable inspection

## Memory Model Considerations

Interactive evaluation introduces memory model questions:

### Arena-Based Session Memory

```fsharp
/// Session-scoped arena for REPL allocations
type SessionArena = {
    /// Main allocation arena
    Arena: Arena

    /// Reset between evaluations (optional)
    mutable EvaluationArena: Arena
}

/// Allocations within an evaluation
let allocate (session: SessionArena) (size: int) : nativeptr<byte> =
    Arena.alloc session.EvaluationArena size

/// Optional: Reset evaluation arena between inputs
let resetEvaluationArena (session: SessionArena) =
    Arena.reset session.EvaluationArena
```

### Value Persistence

Values bound in the session need to outlive individual evaluations:

```
Evaluation 1: let x = [| 1; 2; 3 |]
  → Array allocated in session arena
  → x registered in symbol table with pointer

Evaluation 2: x.[0]
  → x looked up in symbol table
  → Pointer still valid (session arena not reset)
```

## Comparison with FSI

| Feature | FSI | FSNI |
|---------|-----|------|
| Compilation | .NET JIT | LLVM ORC JIT |
| Type system | .NET types | Native types |
| Memory model | GC | Arena/Stack |
| Startup time | ~1s | ~100ms (warm) |
| String type | System.String | NativeStr |
| Interop | .NET assemblies | Platform bindings |
| Reflection | Full | None (compile-time) |
| Scripting | `.fsx` files | `.fsnx` files |

## Alternative Approaches Considered

### 1. Interpreter Mode

**Idea**: Build a pure interpreter that doesn't use native compilation

**Rejected because**:
- Wouldn't exercise native semantics
- Different behavior from compiled code
- Maintenance burden of two execution models

### 2. Dynamic Library Approach (Evcxr-style)

**Idea**: Compile each evaluation to a shared library, dynamically load

**Considered but deferred**:
- Higher latency than JIT
- File I/O overhead
- Symbol management complexity
- Could be fallback if JIT proves difficult

### 3. Full Firefly Recompilation

**Idea**: Recompile entire session on each input

**Rejected because**:
- Latency unacceptable for interactive use
- Doesn't scale with session size

## Related Resources

| Resource | Purpose |
|----------|---------|
| [Evcxr](https://github.com/evcxr/evcxr) | Rust REPL reference |
| [Clang-Repl](https://clang.llvm.org/docs/ClangRepl.html) | C++ JIT REPL reference |
| [Swift REPL](https://www.swift.org/documentation/lldb/) | Debugger-based REPL reference |
| [ORC JIT](https://llvm.org/docs/ORCv2.html) | LLVM JIT infrastructure |
| [MLIR ExecutionEngine](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/) | MLIR JIT tutorial |
| [jank](https://jank-lang.org/) | Clojure dialect with LLVM JIT |
| [LFortran](https://lfortran.org/) | Fortran with REPL via LLVM JIT |

## Summary

FSNI is achievable by leveraging:

1. **Existing infrastructure**: FNCS, Alex, MLIR pipeline
2. **LLVM ORC JIT**: Battle-tested JIT compilation framework
3. **Background compiler service**: Warm compiler state for low latency
4. **Incremental compilation**: Fragment-by-fragment type checking

The implementation builds naturally on Fidelity's existing architecture. The key insight is that **Firefly already has most of the pieces** - FSNI adds an incremental mode and JIT backend rather than requiring a fundamentally new architecture.

**Priority**: Phase 1-3 should follow FSNAC (the LSP server), as they share the incremental FNCS requirement. A warm FSNAC can potentially serve as the background compiler service for FSNI.

---

*Interactive exploration of native F#. The expressiveness of FSI meets the determinism of native compilation.*
