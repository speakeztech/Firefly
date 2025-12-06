# Firefly Compiler - Claude Context

## Project Overview

Firefly is an ahead-of-time (AOT) compiler for F# targeting native binary output without the .NET runtime. The project compiles F# source code through MLIR to LLVM IR and finally to native executables.

**Primary Goal**: Compile F# code to efficient, standalone native binaries that can run without any runtime dependencies (freestanding mode) or with minimal libc dependency (console mode).

**Key Constraint**: "Compiles" means a working native binary that executes correctly, not just successful parsing or IR generation.

## Architecture Overview

The compilation pipeline flows through these major phases:

```
F# Source → FCS → PSG → Alex/Emission → MLIR → LLVM → Native Binary
```

### Core Components

1. **FCS (F# Compiler Services)** - `/src/Core/FCS/`
   - Provides parsing, type checking, and semantic analysis
   - Single source of truth for F# semantics
   - See `docs/FCS_*.md` for detailed documentation

2. **PSG (Program Semantic Graph)** - `/src/Core/PSG/`
   - Unified intermediate representation correlating syntax with semantics
   - Enables reachability analysis for dead code elimination
   - Fan-out traversal from entry points
   - See `docs/PSG_*.md` for architecture decisions

3. **Alex** - `/src/Alex/`
   - Multi-dimensional hardware targeting layer ("Library of Alexandria")
   - Maps F# semantics to platform-optimized code patterns
   - Contains:
     - `Pipeline/` - Orchestration, FCS integration, lowering, optimization
     - `Bindings/` - Platform-aware code emission patterns
     - `Emission/` - MLIR code generation
     - `CodeGeneration/` - Type mapping, emission context

4. **Alloy Library** - External at `/home/hhh/repos/Alloy/src/`
   - Self-contained F# standard library for native compilation
   - BCL-sympathetic API without .NET runtime dependency
   - Core modules: Core.fs, Math.fs, Memory.fs, Text.fs, Console.fs

## Critical Working Principle: Zoom Out Before Fixing

> **THIS IS A COMPILER. Compiler development has fundamentally different requirements than typical application development. The pipeline is a directed, multi-stage transformation where upstream decisions have cascading downstream effects. You MUST understand this before making any changes.**

### The Cardinal Rule

**When encountering ANY adverse finding that is NOT a baseline F# syntax error, you MUST stop and review the ENTIRE compiler pipeline before attempting a fix.**

### Why This Matters

**DO NOT "patch in place."** The instinct to fix a problem where it manifests is almost always wrong in compiler development. A symptom appearing in MLIR emission may have its root cause in:
- Missing or stub implementations in Alloy
- Incorrect symbol capture in FCS ingestion
- Failed correlation in PSG construction
- Wrong reachability decisions
- Missing binding patterns in Alex

**Patching downstream creates technical debt that compounds.** A "fix" in the emitter that works around a PSG deficiency:
- Masks the real problem
- Creates implicit dependencies on broken behavior
- Makes future fixes harder as the codebase grows
- Violates the architectural separation of concerns

### The Correct Approach

1. **Observe the symptom** - Note exactly what's wrong (wrong output, missing symbol, incorrect behavior)

2. **Trace upstream** - Walk backwards through the pipeline:
   ```
   Native Binary ← LLVM ← MLIR ← Alex/Emission ← PSG ← FCS ← Alloy ← F# Source
   ```

3. **Find the root cause** - The fix belongs at the EARLIEST point in the pipeline where the defect exists

4. **Fix upstream** - Correct the root cause, then verify the fix propagates correctly through all downstream stages

5. **Validate end-to-end** - Confirm the native binary behaves correctly

### Example: The Wrong Way vs. The Right Way

**Scenario**: HelloWorld binary prints output but doesn't wait for user input.

**WRONG**: Add special-case handling in ExpressionEmitter.fs to emit a read syscall when it sees a Console.readInto pattern.

**RIGHT**:
1. Check Alloy's Console.fs - Is `readInto` actually implemented or is it a stub?
2. Check FCS ingestion - Is the `readInto` symbol being captured?
3. Check PSG - Is `readInto` marked as reachable?
4. Check Alex bindings - Does a binding pattern exist for Console I/O?
5. Fix at the appropriate level (in this case, Alloy had stub implementations)

### Forcing Functions

Before proposing any fix for a non-syntax issue, answer these questions:

1. "Have I read the relevant docs in `/docs/`?"
2. "Have I traced this issue through the full pipeline?"
3. "Am I fixing the ROOT CAUSE or patching a SYMPTOM?"
4. "Is my fix at the earliest possible point in the pipeline?"
5. "Will this fix work correctly as the compiler evolves, or am I creating hidden coupling?"

If you cannot confidently answer all five questions, you have not yet understood the problem well enough to fix it.

### Pipeline Review Checklist

When a non-syntax issue arises:

1. **Alloy Library Level**
   - Is the required function actually implemented (not a stub)?
   - Does the function signature match what's expected?
   - Is the module properly exported?

2. **FCS Ingestion Level**
   - Is the symbol being captured correctly?
   - Is the type information complete and accurate?
   - Are all dependencies being resolved?

3. **PSG Construction Level**
   - Is the function reachable from the entry point?
   - Are call edges being created correctly?
   - Is symbol correlation working?

4. **Alex Emission Level**
   - Is the binding pattern matching correctly?
   - Is the platform-specific code being selected?
   - Are function calls being emitted (not skipped)?

5. **MLIR/LLVM Level**
   - Is the generated IR valid?
   - Are external declarations correct?
   - Is the calling convention appropriate?

## XParsec: The Critical Glue Layer

> **XParsec-based parser combinators are the ONLY acceptable mechanism for building the Alex transformation layer. This is non-negotiable.**

### Why XParsec Is Foundational

The transformation from PSG to MLIR (and beyond) requires pattern matching against typed semantic structures. XParsec provides:

1. **Type-Safe Composition** - Combinators carry type information through the entire transformation pipeline. The F# type system ensures that pattern matches are exhaustive and transformations are well-typed.

2. **Composability** - Small, focused parsers combine into complex pattern recognition without losing type safety or semantic precision.

3. **Semantic Preservation** - Unlike string-based approaches, XParsec operates on the typed PSG structure directly, preserving the semantic information that FCS provided.

4. **Future-Proof Architecture** - The same combinator infrastructure scales to MLIR dialect transformations, optimization passes, and future IR layers.

### What NOT To Do

**DO NOT fall back to string-based parsing or pattern matching on stringified representations.**

Past attempts to use string manipulation for AST/IR transformation have been:
- **Wasteful** - Significant development time invested in approaches that had to be ripped out
- **Fragile** - String patterns break silently when upstream representations change
- **Type-Unsafe** - Loses the type information that makes the pipeline reliable
- **Unmaintainable** - Ad-hoc string parsing cannot be composed or reasoned about

### The Correct Pattern

```
PSG Node → XParsec Pattern Match → Typed Intermediate → MLIR Emission
```

Every transformation in Alex should:
1. Accept typed PSG nodes as input
2. Use XParsec combinators to recognize patterns
3. Produce typed intermediate representations
4. Emit MLIR through typed builders

### Example: The Wrong Way vs. The Right Way

**WRONG** - String-based pattern matching:
```fsharp
// DO NOT DO THIS
let emitCall (node: PSGNode) =
    let text = node.ToString()
    if text.Contains("Console.Write") then
        emitWriteSyscall()
    elif text.Contains("Console.readInto") then
        emitReadSyscall()
```

**RIGHT** - XParsec combinator-based:
```fsharp
// Typed pattern recognition
let consoleWritePattern =
    psgCall "Alloy.Console" "Write"
    |>> fun args -> ConsoleWrite args

let consoleReadPattern =
    psgCall "Alloy.Console" "readInto"
    |>> fun args -> ConsoleRead args

let consolePattern = consoleWritePattern <|> consoleReadPattern
```

### Key XParsec Locations

| File | Purpose |
|------|---------|
| `/src/Core/XParsec/Foundation.fs` | Core types, CompilerResult<'T> |
| `/src/Alex/Patterns/` | PSG pattern matchers |
| `/src/Alex/Bindings/PatternMatcher.fs` | Symbol matching combinators |

### The Composability Guarantee

XParsec combinators guarantee that:
- If a pattern matches, the types are correct
- Partial matches can be combined without losing type information
- Backtracking preserves semantic state
- Error messages include full type context

This composability is what enables the Firefly pipeline to evolve. New Alloy functions, new platform targets, new optimization passes - all can be added by composing existing combinators with new ones, without breaking existing transformations.

**String parsing provides none of these guarantees. It is not acceptable in this codebase.**

## Essential Documentation

Before making changes, review these documents in `/docs/`:

| Document | Purpose |
|----------|---------|
| `PSG_architecture.md` | PSG design decisions, node identity, reachability |
| `Alex_Architecture_Overview.md` | Alex fan-out pattern, binding structure |
| `FCS_Ingestion_Architecture.md` | FCS integration approach |
| `HelloWorld_Sample_Goals.md` | Sample validation requirements |
| `HelloWorld_Lessons_Learned.md` | Common pitfalls and solutions |

## Sample Projects

Located in `/samples/console/`:

- `HelloWorld/` - Minimal validation sample
- `FidelityHelloWorld/` - Progressive complexity samples:
  - `01_HelloWorldDirect/` - Direct module calls
  - `02_HelloWorldSaturated/` - Saturated function calls
  - `03_HelloWorldHalfCurried/` - Pipe operators, partial application
  - `04_HelloWorldFullCurried/` - Full currying, Result.map, lambdas

## Build and Test Commands

```bash
# Build the compiler
cd /home/hhh/repos/Firefly/src
dotnet build

# Compile a sample
/home/hhh/repos/Firefly/src/bin/Debug/net9.0/Firefly compile HelloWorld.fidproj

# With verbose output
Firefly compile HelloWorld.fidproj --verbose

# Keep intermediate files for debugging
Firefly compile HelloWorld.fidproj -k

# Emit MLIR only (for inspection)
Firefly compile HelloWorld.fidproj --emit-mlir
```

## Key Files

| File | Purpose |
|------|---------|
| `/src/Firefly.fsproj` | Main compiler project |
| `/src/Core/IngestionPipeline.fs` | Pipeline orchestration |
| `/src/Core/PSG/Builder.fs` | PSG construction |
| `/src/Core/PSG/Reachability.fs` | Dead code elimination |
| `/src/Alex/Emission/ExpressionEmitter.fs` | MLIR emission |
| `/src/Alex/Pipeline/CompilationOrchestrator.fs` | Full compilation |

## Common Pitfalls

1. **Stub Functions**: Alloy functions that compile but do nothing at runtime. Always verify the implementation, not just the signature.

2. **Skipped Alloy Functions**: The emitter intentionally skips Alloy library functions expecting them to be handled specially. Ensure binding patterns exist.

3. **Symbol Correlation**: FCS symbol correlation can fail silently. Check `[BUILDER] Warning:` messages in verbose output.

4. **Platform Bindings**: Console I/O, time operations, and memory management require platform-specific implementations in Alex bindings.

## Project Configuration

Projects use `.fidproj` files (TOML format):

```toml
[package]
name = "ProjectName"

[compilation]
memory_model = "stack_only"
target = "native"

[dependencies]
alloy = { path = "/home/hhh/repos/Alloy/src" }

[build]
sources = ["Main.fs"]
output = "binary_name"
output_kind = "freestanding"  # or "console"
```

## When in Doubt: The Zoom-Out Protocol

**Stop. Do not write code yet.**

1. **Read the docs** - Review all relevant documentation in `/docs/` completely
2. **Trace the pipeline** - Follow data flow from F# source to native binary
3. **Identify the layer** - Determine which pipeline stage contains the ROOT CAUSE:
   - Alloy (library implementation)
   - FCS (parsing, type checking, symbol resolution)
   - PSG (semantic graph, reachability, correlation)
   - Alex (bindings, emission patterns, platform targeting)
   - MLIR/LLVM (IR generation, lowering, optimization)
4. **Fix upstream** - Apply the fix at the earliest point where the defect exists
5. **Validate downstream** - Verify the fix propagates correctly through all stages to produce a working binary

**Remember**: In compiler development, the symptom location and the fix location are usually different. Resist the temptation to patch where you see the problem. Find where the problem originates and fix it there.
