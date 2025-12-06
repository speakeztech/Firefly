# Emitter.fs Technical Debt Remediation Plan

## Overview

This document formalizes the plan for remediating the technical debt in `Core/MLIR/Emitter.fs` by extracting its useful components and re-engineering them into the principled Alex architecture using XParsec-style combinators and zipper-based PSG traversal.

**Current State**: 2,483 lines of string-matching spaghetti
**Target State**: Slim dispatcher (or elimination) with principled Alex components
**Validation Target**: TimeLoop sample compiles and runs correctly through the new infrastructure

---

## Remediation Phases

### Phase 1: Infrastructure Foundation

**Goal**: Create the core infrastructure that all subsequent phases will use.

#### 1.1 PSG Zipper Implementation
Create `Alex/Traversal/PSGZipper.fs`:
```fsharp
type PSGPath =
    | Top
    | BindingChild of parent: PSGNode * siblings: PSGNode list * path: PSGPath
    | AppArg of func: PSGNode * argIndex: int * otherArgs: PSGNode list * path: PSGPath
    | SequenceItem of preceding: PSGNode list * following: PSGNode list * path: PSGPath
    // ... other contexts

type PSGZipper = {
    Focus: PSGNode
    Path: PSGPath
    State: EmissionState
}

// Navigation
val left: PSGZipper -> PSGZipper option
val right: PSGZipper -> PSGZipper option
val up: PSGZipper -> PSGZipper option
val down: PSGZipper -> PSGZipper option
val top: PSGZipper -> PSGZipper
```

#### 1.2 Emission Context
Migrate from Emitter.fs to `Alex/CodeGeneration/EmissionContext.fs`:
- `SSAContext` (name generation, local tracking, type tracking)
- `MLIRBuilder` (indentation, output building)
- Add zipper integration

#### 1.3 Type Mapping
Migrate from Emitter.fs to `Alex/CodeGeneration/TypeMapping.fs`:
- `fsharpTypeToMLIR`
- `getFunctionReturnType`
- `getFunctionParamTypes`

**Deliverable**: Foundation infrastructure compiles and passes unit tests.

---

### Phase 2: Binding Registry

**Goal**: Create the platform-aware binding system that replaces string-matching dispatch.

#### 2.1 Binding Types
Create `Alex/Bindings/BindingTypes.fs`:
```fsharp
type BindingKey = {
    Namespace: string
    Function: string
    Arity: int
}

type BindingResult =
    | EmitInline of (EmissionContext -> PSGZipper -> MLIRFragment)
    | EmitCall of functionName: string * argMapper: (PSGNode list -> string list)
    | NotBound

type IBinding =
    abstract CanHandle: BindingKey -> bool
    abstract Emit: EmissionContext -> PSGZipper -> BindingResult
```

#### 2.2 Platform Registry
Create `Alex/Bindings/PlatformRegistry.fs`:
```fsharp
type Platform = Linux_x86_64 | Linux_ARM64 | Windows_x86_64 | MacOS_x86_64 | MacOS_ARM64

val detectPlatform: TargetTriple -> Platform
val getBindings: Platform -> IBinding list
val lookupBinding: Platform -> BindingKey -> IBinding option
```

#### 2.3 Console Bindings
Migrate syscall emission to `Alex/Bindings/Console/Linux.fs`:
- `emitSyscallWrite` → `LinuxConsoleBinding.Write`
- `emitWriteInt32`, `emitWriteInt64` → `LinuxConsoleBinding.WriteInt`
- String literal handling

#### 2.4 Time Bindings
Complete `Alex/Bindings/Time/Linux.fs`:
- `emitClockGettime` → `LinuxTimeBinding.GetTicks`
- `emitNanosleep` → `LinuxTimeBinding.Sleep`

**Deliverable**: HelloWorld sample compiles using binding registry instead of Emitter.fs string matching.

---

### Phase 3: XParsec-Style PSG Pattern Recognition

**Goal**: Replace string-based pattern matching with typed combinators.

#### 3.1 PSG Patterns Module
Create `Alex/Patterns/PSGPatterns.fs`:
```fsharp
// Pattern type
type PSGPattern<'T> = PSGZipper -> ('T * PSGZipper) option

// Basic patterns
val psgSymbol: string -> PSGPattern<FSharpSymbol>
val psgCall: string -> PSGPattern<PSGNode * PSGNode list>
val psgBinding: PSGPattern<string * PSGNode>
val psgConst: PSGPattern<obj * FSharpType>

// Combinators
val (<|>): PSGPattern<'T> -> PSGPattern<'T> -> PSGPattern<'T>
val (>>=): PSGPattern<'T> -> ('T -> PSGPattern<'U>) -> PSGPattern<'U>
val (|>>): PSGPattern<'T> -> ('T -> 'U) -> PSGPattern<'U>
val many: PSGPattern<'T> -> PSGPattern<'T list>
```

#### 3.2 Alloy Pattern Library
Create `Alex/Patterns/AlloyPatterns.fs`:
```fsharp
// Recognize Alloy.Console operations
val consoleWritePattern: PSGPattern<ConsoleWriteOp>
val consoleWriteLinePattern: PSGPattern<ConsoleWriteLineOp>

// Recognize Alloy.Time operations
val timeCurrentTicksPattern: PSGPattern<unit>
val timeSleepPattern: PSGPattern<PSGNode>
val timeCurrentDateTimeStringPattern: PSGPattern<unit>

// Recognize control flow
val whileLoopPattern: PSGPattern<PSGNode * PSGNode>
val ifThenElsePattern: PSGPattern<PSGNode * PSGNode * PSGNode option>
```

**Deliverable**: Pattern recognizers work on TimeLoop PSG and correctly identify all Alloy calls.

---

### Phase 4: Emission Combinators

**Goal**: Create composable emission using XParsec-style monadic composition.

#### 4.1 Emission Monad
Create `Alex/CodeGeneration/EmissionMonad.fs`:
```fsharp
type Emission<'T> = EmissionContext -> PSGZipper -> Result<'T * EmissionContext, EmissionError>

// Builder
type EmissionBuilder() =
    member _.Bind(m, f) = ...
    member _.Return(x) = ...
    member _.Zero() = ...

val emit: EmissionBuilder
```

#### 4.2 MLIR Emission Combinators
Create `Alex/CodeGeneration/MLIRCombinators.fs`:
```fsharp
val emitConst: obj -> FSharpType -> Emission<string>
val emitArith: string -> string -> string -> Emission<string>
val emitCall: string -> string list -> string -> Emission<string>
val emitBlock: string -> Emission<unit> -> Emission<unit>
val emitFunc: string -> (string * string) list -> string -> Emission<unit> -> Emission<unit>
```

#### 4.3 Pattern-Driven Emission
Create `Alex/CodeGeneration/PatternEmission.fs`:
```fsharp
val emitFromPattern: PSGPattern<'T> -> ('T -> Emission<string>) -> Emission<string option>

// Composed emission for known patterns
val emitConsoleWrite: Emission<string option>
val emitTimeOperation: Emission<string option>
val emitArithmeticOp: Emission<string option>
```

**Deliverable**: Individual MLIR fragments can be emitted using combinators.

---

### Phase 5: Zipper-Based Expression Walker

**Goal**: Replace the 1,400-line `emitExpression` with zipper traversal.

#### 5.1 Expression Emitter
Create `Alex/CodeGeneration/ExpressionEmitter.fs`:
```fsharp
val emitExpression: Emission<string option>

// Internally uses:
// 1. Pattern matching via PSGPatterns
// 2. Binding lookup via PlatformRegistry
// 3. Zipper navigation for context
// 4. Emission combinators for MLIR generation
```

#### 5.2 Function Emitter
Create `Alex/CodeGeneration/FunctionEmitter.fs`:
```fsharp
val emitFunction: bool -> Emission<unit>  // isEntryPoint flag
val emitModule: Emission<string>
```

**Deliverable**: Expression emission works through zipper traversal, not string matching.

---

### Phase 6: Dispatcher Refactor (or Elimination)

**Goal**: Determine if Emitter.fs survives as a slim dispatcher or is eliminated entirely.

#### 6.1 Analyze Remaining Emitter.fs
After Phases 1-5, audit what remains in Emitter.fs:
- If only orchestration: rename to `Alex/Pipeline/MLIROrchestrator.fs`
- If nothing unique: delete Emitter.fs entirely

#### 6.2 Integration Point
Either in the surviving orchestrator or directly in `CompilationOrchestrator.fs`:
```fsharp
let generateMLIR (psg: ProgramSemanticGraph) (config: CompilationConfig) =
    let platform = PlatformRegistry.detectPlatform config.Target
    let bindings = PlatformRegistry.getBindings platform
    let zipper = PSGZipper.create psg
    let context = EmissionContext.create config bindings

    ExpressionEmitter.emitModule
    |> Emission.run context zipper
```

**Deliverable**: Emitter.fs is either slim (<200 lines) or deleted.

---

### Phase 7: Validation

**Goal**: Prove the refactoring succeeded with TimeLoop compilation.

#### 7.1 TimeLoop Compilation Test
```bash
# Compile TimeLoop through new infrastructure
Firefly compile ~/repos/Firefly/samples/console/TimeLoop/TimeLoop.fidproj

# Verify MLIR output
mlir-opt --verify TimeLoop.mlir

# Complete lowering pipeline
mlir-opt --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm \
         --reconcile-unrealized-casts TimeLoop.mlir | mlir-translate --mlir-to-llvmir > TimeLoop.ll

# Compile to native
clang -O2 TimeLoop.ll -o timeloop -nostdlib -static

# Run and verify output
./timeloop
# Expected: 5 datetime strings printed, 1 second apart
```

#### 7.2 Regression Tests
- HelloWorld still compiles
- All samples in `/samples/console/` still work
- Performance regression test (compile time should improve)

**Deliverable**: TimeLoop runs correctly, proving the architecture works.

---

## Task Breakdown

### Immediate Tasks (Phase 1)

| Task | Description | Est. Lines | Priority |
|------|-------------|-----------|----------|
| 1.1a | Create `Alex/Traversal/` directory | - | P0 |
| 1.1b | Implement `PSGZipper.fs` with navigation | ~150 | P0 |
| 1.2a | Extract `SSAContext` from Emitter.fs | ~60 | P0 |
| 1.2b | Extract `MLIRBuilder` from Emitter.fs | ~40 | P0 |
| 1.2c | Create `EmissionContext.fs` combining both | ~100 | P0 |
| 1.3 | Create `TypeMapping.fs` from Emitter.fs | ~80 | P0 |

### Phase 2 Tasks

| Task | Description | Est. Lines | Priority |
|------|-------------|-----------|----------|
| 2.1 | Create `BindingTypes.fs` | ~80 | P1 |
| 2.2 | Create `PlatformRegistry.fs` | ~100 | P1 |
| 2.3a | Create `Console/Linux.fs` | ~150 | P1 |
| 2.3b | Migrate `emitWriteInt32/64` | ~200 | P1 |
| 2.4 | Complete `Time/Linux.fs` | ~100 | P1 |

### Phase 3 Tasks

| Task | Description | Est. Lines | Priority |
|------|-------------|-----------|----------|
| 3.1 | Create `PSGPatterns.fs` | ~200 | P1 |
| 3.2 | Create `AlloyPatterns.fs` | ~150 | P1 |

### Phase 4-5 Tasks

| Task | Description | Est. Lines | Priority |
|------|-------------|-----------|----------|
| 4.1 | Create `EmissionMonad.fs` | ~80 | P2 |
| 4.2 | Create `MLIRCombinators.fs` | ~150 | P2 |
| 4.3 | Create `PatternEmission.fs` | ~200 | P2 |
| 5.1 | Create `ExpressionEmitter.fs` | ~300 | P2 |
| 5.2 | Create `FunctionEmitter.fs` | ~150 | P2 |

### Phase 6-7 Tasks

| Task | Description | Est. Lines | Priority |
|------|-------------|-----------|----------|
| 6.1 | Audit remaining Emitter.fs | - | P3 |
| 6.2 | Refactor/delete Emitter.fs | - | P3 |
| 7.1 | TimeLoop validation | - | P3 |
| 7.2 | Regression testing | - | P3 |

---

## Success Criteria

1. **Emitter.fs < 200 lines** (or deleted entirely)
2. **No string matching on SyntaxKind** in new code
3. **All PSG traversal uses zipper** with context preservation
4. **Binding lookup uses registry** not if-chains
5. **TimeLoop compiles and runs correctly**
6. **Compile time equal or faster** than current implementation

---

## References

- Alex_Architecture_Overview.md - Directory structure
- Alex_Architecture_Memo.md - Architectural vision
- Hyping Hypergraphs - Zipper and recursion scheme theory
- XParsec - Combinator patterns
- Tomas Petricek's Zipper - F# zipper implementation

---

*Created: December 2024*
*Status: Planning Complete - Ready for Execution*
