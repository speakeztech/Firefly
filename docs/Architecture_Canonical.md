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

## Alex: What It Is

Alex routes extern primitives to platform-specific MLIR generation:

```fsharp
// Dispatch by (entry_point, platform) -> MLIR generator
let dispatch (entryPoint: string) (platform: Platform) (args: Val list) : MLIR<Val> =
    match entryPoint, platform.OS with
    | "fidelity_write_bytes", Linux -> emitLinuxWriteSyscall args
    | "fidelity_write_bytes", Windows -> emitWindowsWriteFile args
    | "fidelity_write_bytes", MacOS -> emitMacOSWriteSyscall args
    | ...
```

**Platform differences are DATA:**

```fsharp
// Syscall numbers as data, not separate files
let syscallNumbers = Map [
    (Linux, "write"), 1L
    (Linux, "read"), 0L
    (Linux, "clock_gettime"), 228L
    (MacOS, "write"), 0x2000004L  // with BSD offset
    ...
]
```

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
├── Bindings/
│   ├── BindingTypes.fs    # Platform enum, dispatch types
│   ├── ExternDispatch.fs  # Central dispatch table
│   ├── ConsoleBindings.fs # MLIR for console externs (all platforms)
│   └── TimeBindings.fs    # MLIR for time externs (all platforms)
└── ...
```

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
```

## Validation Samples

These samples must compile WITHOUT modification:
- `01_HelloWorldDirect` - Console.Write, Console.WriteLine
- `02_HelloWorldSaturated` - Console.ReadLine, interpolated strings
- `03_HelloWorldHalfCurried` - Pipe operators, NativeStr

The samples use Alloy's BCL-sympathetic API. Firefly compiles them by:
1. Building PSG from F# source + Alloy
2. PSG captures extern primitive metadata from DllImport
3. Alex dispatches externs to platform MLIR generators
4. MLIR → LLVM → native binary
