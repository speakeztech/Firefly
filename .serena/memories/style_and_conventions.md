# Style and Conventions

## F# Code Style
- Standard F# naming conventions (PascalCase for types/modules, camelCase for values/functions)
- Functional-first approach with immutable data structures
- Heavy use of discriminated unions and pattern matching
- Type annotations where clarity is needed

## Architecture Principles

### Layer Separation Principle
Each layer in the pipeline has ONE responsibility. Do not mix concerns across layers:

| Layer | Responsibility | DOES NOT |
|-------|---------------|----------|
| **Alloy** | Provide F# implementations of library functions | Contain stubs expecting compiler magic |
| **FCS** | Parse, type-check, resolve symbols | Transform or generate code |
| **PSG Builder** | Construct semantic graph from FCS output | Make targeting decisions |
| **Nanopasses** | Enrich PSG with edges, classifications | Generate MLIR or know about targets |
| **Alex/Zipper** | Traverse PSG, generate MLIR via bindings | Pattern-match on library names |
| **Bindings** | Platform-specific MLIR generation | Know about F# syntax or Alloy namespaces |

### Critical Anti-Patterns to AVOID
1. **Adding Alloy-specific logic to MLIR generation** - MLIR generation should not know about Alloy
2. **Stub implementations in Alloy** - All functions must decompose to real primitives
3. **Putting nanopass logic in MLIR generation** - Nanopasses run BEFORE, not during MLIR generation
4. **Library-specific special cases** - Never pattern-match on function names like "Alloy.Console.Write"

### The Extern Primitive Surface
The ONLY acceptable "stubs" are **extern declarations** using `DllImport("__fidelity")`:
```fsharp
[<DllImport("__fidelity", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "fidelity_write_bytes")>]
extern int writeBytes(int fd, nativeptr<byte> buffer, int count)
```

## Development Philosophy

### Deliberate, Didactic Approach
- **Use Agents Before Acting** - Explore reference materials before making changes
- **Confirm Intent** - Clarify ambiguity rather than making assumptions
- **Trace Full Pipeline** - Trace issues through complete compilation pipeline
- **Understand Before Implementing** - Read documentation first

### Zoom Out Before Fixing
When encountering ANY adverse finding that is NOT a baseline F# syntax error:
1. Stop and review the ENTIRE compiler pipeline
2. Trace upstream through: Native Binary ← LLVM ← MLIR ← Alex/Zipper ← Nanopasses ← PSG ← FCS ← Alloy ← F# Source
3. Fix at the EARLIEST point in the pipeline where the defect exists
4. Validate end-to-end with a working native binary

### Forcing Questions Before Any Fix
1. "Have I read the relevant docs in `/docs/`?"
2. "Have I traced this issue through the full pipeline?"
3. "Am I fixing the ROOT CAUSE or patching a SYMPTOM?"
4. "Is my fix at the earliest possible point in the pipeline?"
5. "Am I adding library-specific logic to a layer that shouldn't know about libraries?"
