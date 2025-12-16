# Firefly Architecture: Canonical Reference

## The Pipeline Model

```
┌─────────────────────────────────────────────────────────┐
│  Alloy (Platform-Agnostic Library)                      │
│  - Pure F# + FSharp.NativeInterop                       │
│  - BCL-sympathetic API surface                          │
│  - Platform Bindings: Module convention + marker type   │
│  - ZERO platform knowledge, ZERO BCL dependencies       │
└─────────────────────────────────────────────────────────┘
                          │
                          │ FCS parses & type-checks
                          ▼
┌─────────────────────────────────────────────────────────┐
│  PSG Construction: Phases 1-3                           │
│  - Phase 1: Structural (full library)                   │
│  - Phase 2: Symbol correlation (full library)           │
│  - Phase 3: Reachability (NARROWS to application scope) │
└─────────────────────────────────────────────────────────┘
                          │
                          │ Narrowed compute graph
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Baker (Type Resolution Layer) [Phase 4]                │
│  - Operates on NARROWED graph (post-reachability)       │
│  - Correlates typed tree with PSG structure             │
│  - Extracts SRTP resolutions (TraitCall → member)       │
│  - Maps static member bodies for inlining               │
│  - Overlays resolved types onto PSG nodes               │
└─────────────────────────────────────────────────────────┘
                          │
                          │ Enriched PSG (narrowed scope)
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Alex (Compiler Targeting Layer)                        │
│  - Consumes NARROWED, ENRICHED PSG                      │
│  - Same scope as Baker - zipper coherence               │
│  - Dispatches externs by (entry_point, platform)        │
│  - Platform differences are DATA, not separate files    │
│  - Generates platform-optimized MLIR                    │
└─────────────────────────────────────────────────────────┘
```

**Symmetric Component Libraries:** Baker and Alex are consolidation component libraries on opposite sides of the enriched PSG:
- **Baker**: Consolidates type-level transforms, focuses IN on narrowed application graph
- **Alex**: Consolidates code-level transforms, fans OUT to platform targets

**Zipper Coherence:** Both Baker and Alex use zippers to traverse the same narrowed graph:
- Baker's TypedTreeZipper correlates FSharpExpr with PSG nodes
- Alex's PSGZipper traverses enriched PSG to emit MLIR
- Same narrowed scope ensures no mismatch between type resolution and code generation

## Alloy: What It Is

Alloy provides F# implementations that decompose to **platform bindings** - functions that Alex provides platform-specific implementations for.

### The Platform Binding Pattern (BCL-Free)

Platform bindings are declared using a module convention (OCaml homage) combined with a marker type:

```fsharp
// Core marker type - signals "platform provides this"
[<Struct>]
type PlatformProvided = PlatformProvided

// Primary approach: Functions in Platform.Bindings module
// Alex recognizes this module structure and provides implementations
module Platform.Bindings =
    let writeBytes fd buffer count : int = Unchecked.defaultof<int>
    let readBytes fd buffer maxCount : int = Unchecked.defaultof<int>
    let getCurrentTicks () : int64 = Unchecked.defaultof<int64>
    let sleep milliseconds : unit = ()

// Console.fs - Real F# that decomposes to platform bindings
let inline write (s: NativeStr) =
    Platform.Bindings.writeBytes STDOUT s.Pointer s.Length |> ignore
```

**Why BCL-Free?** The `DllImportAttribute` from `System.Runtime.InteropServices` is a BCL dependency. Fidelity deliberately avoids ALL BCL dependencies to maintain a clean compilation path to native code. The Platform Binding pattern achieves the same goal (marking functions for platform-specific implementation) without BCL pollution.

**Alloy does NOT:**
- Know about Linux/Windows/macOS
- Contain `#if PLATFORM` conditionals
- Have platform-specific directories
- Make syscalls directly
- Use ANY BCL types (including DllImportAttribute)

## Alex: The Non-Dispatch Model

> **Key Insight: Centralization belongs at the OUTPUT (MLIR Builder), not at DISPATCH (traversal logic).**

Alex generates MLIR through Zipper traversal and platform Bindings. There is **NO central dispatch hub**.

```
PSG Entry Point
    ↓
Zipper.create(psg, entryNode)     -- provides "attention"
    ↓
Fold over structure (pre-order/post-order)
    ↓
At each node: XParsec matches locally → MLIR emission
    ↓
Extern primitive? → ExternDispatch.dispatch(primitive)
    ↓
MLIR Builder accumulates           -- correct centralization
    ↓
Output: Complete MLIR module
```

**Component Roles:**

- **Zipper**: Purely navigational - provides focus with context, carries state (SSA counters)
- **XParsec**: Local pattern matching - composable patterns, NOT a routing table
- **Bindings**: Platform-specific MLIR - looked up by extern entry point, are DATA not routing
- **MLIR Builder**: Where centralization correctly occurs - the single accumulation point

**Bindings are DATA, not routing logic:**

```fsharp
// Syscall numbers as data
module SyscallData =
    let linuxSyscalls = Map [
        "write", 1L
        "read", 0L
        "clock_gettime", 228L
    ]
    let macosSyscalls = Map [
        "write", 0x2000004L  // BSD offset
        "read", 0x2000003L
    ]

// Bindings registered by (OS, Arch, EntryPoint)
ExternDispatch.register Linux X86_64 "fidelity_write_bytes"
    (fun ext -> bindWriteBytes TargetPlatform.linux_x86_64 ext)
```

**NO central dispatch match statement. Bindings are looked up by entry point.**

## The Fidelity Mission

Unlike Fable (AST→AST, delegates memory to target runtime), Fidelity:
- **Preserves type fidelity**: F# types → precise native representations
- **Preserves memory fidelity**: Compiler-verified lifetimes, deterministic allocation
- **PSG carries proofs**: Not just syntax, but semantic guarantees about memory, types, ownership

The generated native binary has the same safety properties as the source F#.

## Platform Binding Contract

Alex recognizes functions in the `Platform.Bindings` module and provides platform-specific implementations:

| Binding Function | MLIR Mapping | Signature |
|-----------------|--------------|-----------|
| writeBytes | write syscall | (i32, ptr, i32) → i32 |
| readBytes | read syscall | (i32, ptr, i32) → i32 |
| getCurrentTicks | clock_gettime | () → i64 |
| getMonotonicTicks | clock_gettime(MONOTONIC) | () → i64 |
| getTickFrequency | constant (platform-specific) | () → i64 |
| sleep | nanosleep/Sleep | (i32) → void |

Alex provides implementations for each `(binding, platform)` pair. The module structure (`Platform.Bindings`) serves as the recognition marker - no attributes required.

## File Organization

```
Alloy/src/
├── Platform.fs        # PlatformProvided marker + Platform.Bindings module
├── Console.fs         # Decomposes to Platform.Bindings
├── Time.fs            # Decomposes to Platform.Bindings
└── ...                # NO platform directories, NO BCL dependencies

Firefly/src/Baker/
├── Baker.fs               # Main entry point, orchestration
├── Types.fs               # Baker-specific types (SRTPResolution, etc.)
├── TypedTreeZipper.fs     # Two-tree zipper (FSharpExpr ↔ PSGNode)
├── SRTPResolver.fs        # TraitCall → resolved member extraction
├── MemberBodyMapper.fs    # Static member → body mapping
└── TypeOverlay.fs         # Resolved types → PSG nodes

Firefly/src/Alex/
├── Traversal/
│   ├── PSGZipper.fs       # Bidirectional traversal (attention)
│   └── PSGXParsec.fs      # Local pattern matching combinators
├── Bindings/
│   ├── BindingTypes.fs    # ExternDispatch registry, platform types
│   ├── Console/ConsoleBindings.fs  # Console extern bindings (data)
│   ├── Time/TimeBindings.fs        # Time extern bindings (data)
│   └── Process/ProcessBindings.fs  # Process extern bindings (data)
├── CodeGeneration/
│   ├── MLIRBuilder.fs     # MLIR accumulation (correct centralization)
│   └── TypeMapping.fs     # F# → MLIR type mapping
├── Patterns/
│   └── PSGPatterns.fs     # Predicates and extractors
└── Pipeline/
    └── CompilationOrchestrator.fs  # Entry point
```

**Note:** PSGEmitter.fs and PSGScribe.fs were removed - they were antipatterns.

## Anti-Patterns (DO NOT DO)

```fsharp
// WRONG: BCL dependencies in Alloy (including DllImportAttribute!)
open System.Runtime.InteropServices
[<DllImport("__fidelity")>]
extern int writeBytes(...)  // NO! This pollutes entire pipeline with BCL

// WRONG: Platform code in Alloy
#if LINUX
let write fd buf len = syscall 1 fd buf len
#endif

// WRONG: Separate platform files in Alex
Alex/Bindings/Console/Linux.fs
Alex/Bindings/Console/Windows.fs
Alex/Bindings/Console/MacOS.fs

// WRONG: Pattern matching on library names
match symbolName with
| "Alloy.Console.Write" -> ...  // NO!

// WRONG: Alloy functions that are stubs expecting compiler magic
let Write (s: string) = ()  // placeholder

// WRONG: Central dispatch hub (the "emitter" or "scribe" antipattern)
module PSGEmitter =
    let handlers = Dictionary<string, NodeHandler>()
    let emit node =
        match handlers.TryGetValue(getPrefix node) with
        | true, h -> h node
        | _ -> default node
// This was removed TWICE. Centralization belongs at MLIR Builder output,
// not at traversal dispatch.
```

**The correct model:**
- Platform bindings in `Platform.Bindings` module (no attributes, just structure)
- Zipper provides attention (focus + context)
- XParsec provides local pattern matching
- Bindings looked up by module structure, not by attributes
- MLIR Builder accumulates (correct centralization point)

## PSG Construction: True Nanopass Pipeline

**See: `docs/PSG_Nanopass_Architecture.md` for complete details.**
**See: `docs/Baker_Architecture.md` for Phase 4 details.**

PSG construction is a **nanopass pipeline**, not a monolithic operation. Each phase does ONE thing:

```
Phase 1: Structural Construction    SynExpr → PSG with nodes + ChildOf edges
Phase 2: Symbol Correlation         + FSharpSymbol attachments (via FCS)
Phase 3: Soft-Delete Reachability   + IsReachable marks (structure preserved!)
Phase 4: Typed Tree Overlay [BAKER] + Type, Constraints, SRTP resolution, member bodies
Phase 5+: Enrichment Nanopasses     + def-use edges, operation classification, etc.
```

**Critical Principles:**

1. **Soft-delete reachability** - Mark unreachable nodes but preserve structure. The typed tree zipper needs full structure for navigation.

2. **Typed tree overlay via zipper** - A zipper correlates `FSharpExpr` (typed tree) with PSG nodes by range, capturing:
   - Resolved types (after inference)
   - Resolved constraints (after solving)
   - **SRTP resolution** (TraitCall → resolved member)

3. **Each phase is inspectable** - Intermediate PSGs can be examined independently.

### Why This Matters

Without the typed tree overlay, we miss SRTP resolution. For example:

```fsharp
// Alloy/Console.fs
let inline Write s = WritableString $ s  // $ is SRTP-dispatched!
```

- Syntax sees: `App [op_Dollar, WritableString, s]`
- Typed tree knows: `TraitCall` resolving `$` to `WritableString.op_Dollar` → `writeSystemString`

The typed tree zipper captures this resolution INTO the PSG. Downstream passes don't have to re-resolve.

## Validation Samples

These samples must compile WITHOUT modification:
- `01_HelloWorldDirect` - Console.Write, Console.WriteLine
- `02_HelloWorldSaturated` - Console.ReadLine, interpolated strings
- `03_HelloWorldHalfCurried` - Pipe operators, NativeStr

The samples use Alloy's BCL-sympathetic API. Firefly compiles them via the nanopass PSG pipeline:
1. Phase 1-4: Build PSG with full type/SRTP resolution
2. Phase 5+: Enrich with def-use edges, classify operations
3. Reachability prunes dead code
4. Alex/Zipper traverses enriched PSG → MLIR
5. MLIR → LLVM → native binary
