# Firefly Architecture: Canonical Reference

> **Memory Architecture**: For hardware targets (CMSIS, embedded), see [Quotation_Based_Memory_Architecture.md](./Quotation_Based_Memory_Architecture.md)
> which describes the quotation + active pattern infrastructure spanning fsnative, BAREWire, and Farscape.
>
> **Desktop UI Stack**: For WebView-based desktop applications, see [WebView_Desktop_Architecture.md](./WebView_Desktop_Architecture.md)
> which describes Partas.Solid frontend + Firefly native backend with system webview rendering.

## The Pipeline Model

**ARCHITECTURE UPDATE (January 2026)**: FNCS now builds the PSG. Firefly consumes it.

```
┌─────────────────────────────────────────────────────────┐
│  Alloy (Platform-Agnostic Library)                      │
│  - Pure F# + FSharp.NativeInterop                       │
│  - BCL-sympathetic API surface                          │
│  - Platform Bindings: Module convention + marker type   │
│  - ZERO platform knowledge, ZERO BCL dependencies       │
└─────────────────────────────────────────────────────────┘
                          │
                          │ Compiled by FNCS
                          ▼
┌─────────────────────────────────────────────────────────┐
│  FNCS (F# Native Compiler Services)                     │
│  - Parses F# source (SynExpr, SynModule)                │
│  - Type checking with native types (no BCL)             │
│  - SRTP resolution during type checking                 │
│  - PSG CONSTRUCTION (moved from Firefly)                │
│  - Editor services preserved for design-time tooling    │
│                                                         │
│  OUTPUT: PSG with native types, SRTP resolved           │
│          Full symbol info for navigation                │
└─────────────────────────────────────────────────────────┘
                          │
                          │ PSG (correct by construction)
                          ▼
┌─────────────────────────────────────────────────────────┐
│  FIREFLY (Consumes PSG from FNCS)                       │
├─────────────────────────────────────────────────────────┤
│  Lowering Nanopasses (if needed)                        │
│  - FlattenApplications, ReducePipeOperators             │
│  - DetectPlatformBindings                               │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Alex (Compiler Targeting Layer)                        │
│  - Consumes PSG as "correct by construction"            │
│  - NO type checking needed - trusts FNCS                │
│  - Zipper traversal + XParsec pattern matching          │
│  - Platform-specific MLIR via Bindings                  │
│  - Generates platform-optimized MLIR                    │
└─────────────────────────────────────────────────────────┘
```

**FNCS-First Architecture:** With PSG construction moved to FNCS, Firefly's role simplifies:
- **FNCS**: Builds PSG with types attached, SRTP resolved, symbol info preserved
- **Alex**: Traverses PSG → generates MLIR → LLVM → native binary
- **Baker absorbed into FNCS**: Baker's work (type correlation, SRTP resolution) happens DURING PSG construction in FNCS, not as a post-hoc phase in Firefly

**Zipper Coherence:** Alex uses PSGZipper to traverse the PSG from FNCS:
- PSG comes from FNCS with all type information attached
- Alex's PSGZipper traverses PSG to emit MLIR
- No typed tree correlation needed in Firefly - FNCS does it during construction

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
// Note: string has native semantics (Pointer, Length members) via FNCS
let inline write (s: string) =
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
| createWebview | webview_create | (i32, ptr) → ptr |
| setWebviewHtml | webview_set_html | (ptr, ptr) → i32 |
| runWebview | webview_run | (ptr) → i32 |

Alex provides implementations for each `(binding, platform)` pair. The module structure (`Platform.Bindings`) serves as the recognition marker - no attributes required.

> **Note**: Webview bindings call library functions (WebKitGTK, WebView2, WKWebView) rather than syscalls. See [WebView_Desktop_Architecture.md](./WebView_Desktop_Architecture.md) for the full desktop UI stack architecture.

## File Organization

```
Alloy/src/
├── Platform.fs        # PlatformProvided marker + Platform.Bindings module
├── Console.fs         # Decomposes to Platform.Bindings
├── Time.fs            # Decomposes to Platform.Bindings
└── ...                # NO platform directories, NO BCL dependencies

fsnative/src/Compiler/Checking.Native/  # PSG CONSTRUCTION NOW HERE
├── NativeService.fs        # Public API for FNCS
├── NativeTypes.fs          # Native type representation
├── NativeGlobals.fs        # Built-in types (string=UTF-8, etc.)
├── CheckExpressions.fs     # Type checking with PSG construction
├── SemanticGraph.fs        # PSG data structures
├── SRTPResolution.fs       # SRTP resolution during type checking
└── NameResolution.fs       # Compositional name resolution

Firefly/src/Core/PSG/Nanopass/  # LOWERING PASSES (post-FNCS)
├── FlattenApplications.fs
├── ReducePipeOperators.fs
├── DetectPlatformBindings.fs
└── ...

Firefly/src/Alex/
├── Traversal/
│   ├── PSGZipper.fs       # Bidirectional traversal (attention)
│   └── PSGXParsec.fs      # Local pattern matching combinators
├── Bindings/
│   ├── BindingTypes.fs    # ExternDispatch registry, platform types
│   ├── Console/ConsoleBindings.fs  # Console extern bindings (data)
│   ├── Time/TimeBindings.fs        # Time extern bindings (data)
│   └── ...
├── CodeGeneration/
│   ├── MLIRBuilder.fs     # MLIR accumulation (correct centralization)
│   └── TypeMapping.fs     # F# → MLIR type mapping
└── Pipeline/
    └── CompilationOrchestrator.fs  # Entry point
```

**Note:** Baker functionality is ABSORBED into FNCS - type correlation happens during PSG construction, not post-hoc.
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

## PSG Construction: Handled by FNCS

**ARCHITECTURE UPDATE**: PSG construction has moved to FNCS.

See: `docs/PSG_Nanopass_Architecture.md` for nanopass principles.
See: `docs/FNCS_Architecture.md` for how FNCS builds the PSG.

FNCS builds the PSG with:
- Native types attached during type checking
- SRTP resolved during type checking (not post-hoc)
- Full symbol information preserved for design-time tooling

Firefly receives the completed PSG and applies **lowering nanopasses** if needed:
- FlattenApplications
- ReducePipeOperators  
- DetectPlatformBindings
- etc.

### Why FNCS Builds PSG

With FNCS handling PSG construction:
- SRTP is resolved during type checking, not in a separate pass
- Types are attached to nodes during construction, not overlaid later
- No separate typed tree correlation needed
- Symbol information flows directly from type checker to PSG

## Validation Samples

These samples must compile WITHOUT modification:
- `01_HelloWorldDirect` - Console.Write, Console.WriteLine
- `02_HelloWorldSaturated` - Console.ReadLine, interpolated strings
- `03_HelloWorldHalfCurried` - Pipe operators, string formatting

The samples use Alloy's BCL-sympathetic API. Compilation flow:
1. FNCS: Parse, type check with native types, build PSG with SRTP resolved
2. Firefly: Receive PSG from FNCS ("correct by construction")
3. Firefly: Apply lowering nanopasses (flatten apps, reduce pipes, etc.)
4. Alex/Zipper: Traverse PSG → generate MLIR
5. MLIR → LLVM → native binary

---

## Cross-References

### Core Architecture
- [FNCS_Architecture.md](./FNCS_Architecture.md) - FNCS and PSG construction (PRIMARY)
- [PSG_Nanopass_Architecture.md](./PSG_Nanopass_Architecture.md) - Nanopass principles
- [Quotation_Based_Memory_Architecture.md](./Quotation_Based_Memory_Architecture.md) - Memory model for embedded targets
- Note: Baker_Architecture.md is deprecated - FNCS now handles type correlation

### Desktop UI Stack
- [WebView_Desktop_Architecture.md](./WebView_Desktop_Architecture.md) - Partas.Solid + webview architecture
- [WebView_Build_Integration.md](./WebView_Build_Integration.md) - Firefly as unified build orchestrator
- [WebView_Desktop_Design.md](./WebView_Desktop_Design.md) - Implementation details (callbacks, IPC)

### QuantumCredential Demo
- [QC_Demo/](./QC_Demo/) - Demo documentation folder
- [QC_Demo/01_Demo_Strategy_Integrated.md](./QC_Demo/01_Demo_Strategy_Integrated.md) - Integrated demo strategy (desktop + embedded)

### Platform Bindings
- [Native_Library_Binding_Architecture.md](./Native_Library_Binding_Architecture.md) - Platform binding patterns
