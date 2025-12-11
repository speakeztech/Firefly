# Firefly Architecture: Canonical Reference

## The Two-Layer Model

```
┌─────────────────────────────────────────────────────────┐
│  Alloy (Platform-Agnostic Library)                      │
│  - Pure F# + FSharp.NativeInterop                       │
│  - BCL-sympathetic API surface                          │
│  - Extern primitives: DllImport("__fidelity")           │
│  - ZERO platform knowledge                              │
└─────────────────────────────────────────────────────────┘
                          │
                          │ PSG captures extern metadata
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Alex (Compiler Targeting Layer)                        │
│  - Consumes PSG with semantic proofs                    │
│  - Dispatches externs by (entry_point, platform)        │
│  - Platform differences are DATA, not separate files    │
│  - Generates platform-optimized MLIR                    │
└─────────────────────────────────────────────────────────┘
```

## Alloy: What It Is

Alloy provides F# implementations that decompose to **abstract primitives**:

```fsharp
// Primitives.fs - THE ONLY acceptable "stubs"
[<DllImport("__fidelity", EntryPoint = "fidelity_write_bytes")>]
extern int writeBytes(int fd, nativeptr<byte> buffer, int count)

// Console.fs - Real F# that decomposes to primitives
let inline writeBytes fd buffer count = Primitives.writeBytes(fd, buffer, count)
let inline write (s: NativeStr) = writeBytes STDOUT s.Pointer s.Length |> ignore
```

**Alloy does NOT:**
- Know about Linux/Windows/macOS
- Contain `#if PLATFORM` conditionals
- Have platform-specific directories
- Make syscalls directly

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

## Extern Primitive Contract

| Primitive | Entry Point | Signature |
|-----------|-------------|-----------|
| writeBytes | fidelity_write_bytes | (i32, ptr, i32) → i32 |
| readBytes | fidelity_read_bytes | (i32, ptr, i32) → i32 |
| getCurrentTicks | fidelity_get_current_ticks | () → i64 |
| getMonotonicTicks | fidelity_get_monotonic_ticks | () → i64 |
| getTickFrequency | fidelity_get_tick_frequency | () → i64 |
| sleep | fidelity_sleep | (i32) → void |

Alex provides implementations for each (primitive, platform) pair.

## File Organization

```
Alloy/src/
├── Primitives.fs      # Extern declarations only
├── Console.fs         # Decomposes to Primitives
├── Time.fs            # Decomposes to Primitives
└── ...                # NO platform directories

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

**The correct model has NO central dispatch:**
- Zipper provides attention (focus + context)
- XParsec provides local pattern matching
- Bindings are looked up by extern entry point
- MLIR Builder accumulates (correct centralization point)

## PSG Construction: True Nanopass Pipeline

**See: `docs/PSG_Nanopass_Architecture.md` for complete details.**

PSG construction is a **nanopass pipeline**, not a monolithic operation. Each phase does ONE thing:

```
Phase 1: Structural Construction    SynExpr → PSG with nodes + ChildOf edges
Phase 2: Symbol Correlation         + FSharpSymbol attachments (via FCS)
Phase 3: Soft-Delete Reachability   + IsReachable marks (structure preserved!)
Phase 4: Typed Tree Overlay         + Type, Constraints, SRTP resolution (Zipper)
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
