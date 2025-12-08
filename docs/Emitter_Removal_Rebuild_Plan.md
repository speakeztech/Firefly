# Emitter Removal and Rebuild Plan

> **See `Architecture_Canonical.md` for the authoritative architecture.**

## Context

The "emitter" was removed because it violated layer separation - it pattern-matched on library names like `"Alloy.Console.Write"`. This couples layers that should be independent.

## What Was Done

1. **Deleted emitter code** from Alex
2. **Cleaned Alloy** - removed all platform-specific code (Memory/Linux.fs, Time/Linux.fs, etc.)
3. **Created `Primitives.fs`** with extern declarations using `DllImport("__fidelity")`
4. **Updated Console.fs and Time.fs** to delegate to Primitives externs

## The Extern Primitive Pattern

```fsharp
// Alloy/Primitives.fs - declarative, no implementation
[<DllImport("__fidelity", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "fidelity_write_bytes")>]
extern int writeBytes(int fd, nativeptr<byte> buffer, int count)
```

The `"__fidelity"` library name is a marker telling Alex "this is an abstract primitive."

## Remaining Work

1. **Fix Firefly compiler** - update binding files to use new architecture
2. **Consolidate Alex bindings** - platform as data, not separate files
3. **Verify PSG captures DllImport metadata** - entry point, library, calling convention
4. **Validate samples** - 01, 02, 03 HelloWorld without modification

## Validation Samples (DO NOT MODIFY)

| Sample | Tests |
|--------|-------|
| 01_HelloWorldDirect | Console.Write, Console.WriteLine |
| 02_HelloWorldSaturated | Console.ReadLine, interpolated strings |
| 03_HelloWorldHalfCurried | Pipe operators, NativeStr parameter |

## Success Criteria

```bash
cd /home/hhh/repos/Firefly/samples/console/FidelityHelloWorld/01_HelloWorldDirect
/home/hhh/repos/Firefly/src/bin/Debug/net9.0/Firefly compile 01_HelloWorldDirect.fidproj
./hello  # Output: Hello, World!
```
