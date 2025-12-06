# Alex Time Bindings Design

## Overview

This document describes the architecture for emitting platform-specific time operations from Alloy's Time module. The design follows Alex's "Library of Alexandria" model, providing a fan-out from high-level F# time operations to optimized native code for each target platform.

## Alloy Time API Surface

The Time module in Alloy exposes these operations that require platform-specific emission:

```fsharp
module Alloy.Time
    // High-resolution timing
    val currentTicks: unit -> int64
    val highResolutionTicks: unit -> int64
    val tickFrequency: unit -> int64

    // Unix timestamp
    val currentUnixTimestamp: unit -> int64
    val currentTimestamp: unit -> Timestamp

    // Sleep
    val sleep: int -> unit

    // Timezone
    val getCurrentTimezoneOffsetMinutes: unit -> int
```

## Platform-Specific Implementations

### Linux x86-64

| Operation | Implementation | Syscall/Function |
|-----------|----------------|------------------|
| `currentTicks` | clock_gettime(CLOCK_REALTIME) | syscall #228 |
| `highResolutionTicks` | clock_gettime(CLOCK_MONOTONIC) | syscall #228 |
| `tickFrequency` | Constant 10000000L (100ns ticks) | N/A |
| `sleep` | nanosleep | syscall #35 |
| `getTimezoneOffset` | Parse /etc/localtime or return 0 | N/A |

**MLIR Emission Pattern:**
```mlir
// clock_gettime(CLOCK_MONOTONIC, &timespec)
%timespec = llvm.alloca %c1 x !llvm.struct<(i64, i64)> : (i32) -> !llvm.ptr
%clock_id = arith.constant 1 : i64      // CLOCK_MONOTONIC
%syscall = arith.constant 228 : i64     // clock_gettime
%result = llvm.inline_asm has_side_effects
    "syscall", "=r,{rax},{rdi},{rsi}"
    %syscall, %clock_id, %timespec : (i64, i64, !llvm.ptr) -> i64

// Extract seconds and nanoseconds
%sec_ptr = llvm.getelementptr %timespec[0, 0] : !llvm.ptr
%nsec_ptr = llvm.getelementptr %timespec[0, 1] : !llvm.ptr
%sec = llvm.load %sec_ptr : !llvm.ptr -> i64
%nsec = llvm.load %nsec_ptr : !llvm.ptr -> i64

// Convert to ticks (100-nanosecond intervals)
%ticks_per_sec = arith.constant 10000000 : i64
%sec_ticks = arith.muli %sec, %ticks_per_sec : i64
%nsec_div = arith.constant 100 : i64
%nsec_ticks = arith.divui %nsec, %nsec_div : i64
%total_ticks = arith.addi %sec_ticks, %nsec_ticks : i64
```

### Windows x86-64

| Operation | Implementation | Function |
|-----------|----------------|----------|
| `currentTicks` | GetSystemTimeAsFileTime | kernel32.dll |
| `highResolutionTicks` | QueryPerformanceCounter | kernel32.dll |
| `tickFrequency` | QueryPerformanceFrequency | kernel32.dll |
| `sleep` | Sleep | kernel32.dll |
| `getTimezoneOffset` | GetTimeZoneInformation | kernel32.dll |

**MLIR Emission Pattern:**
```mlir
// QueryPerformanceCounter(&counter)
%counter = llvm.alloca %c1 x i64 : (i32) -> !llvm.ptr
%qpc = llvm.mlir.addressof @QueryPerformanceCounter : !llvm.ptr
%result = llvm.call %qpc(%counter) : (!llvm.ptr) -> i32
%ticks = llvm.load %counter : !llvm.ptr -> i64
```

**External Function Declaration:**
```mlir
llvm.func @QueryPerformanceCounter(!llvm.ptr) -> i32 attributes {sym_visibility = "private"}
llvm.func @QueryPerformanceFrequency(!llvm.ptr) -> i32 attributes {sym_visibility = "private"}
llvm.func @Sleep(i32) attributes {sym_visibility = "private"}
llvm.func @GetSystemTimeAsFileTime(!llvm.ptr) attributes {sym_visibility = "private"}
```

### macOS x86-64 and ARM64

| Operation | Implementation | Function |
|-----------|----------------|----------|
| `currentTicks` | gettimeofday | libSystem |
| `highResolutionTicks` | mach_absolute_time | libSystem |
| `tickFrequency` | mach_timebase_info | libSystem |
| `sleep` | nanosleep | libSystem |
| `getTimezoneOffset` | localtime_r | libSystem |

**MLIR Emission Pattern:**
```mlir
// mach_absolute_time()
%mach_abs_time = llvm.mlir.addressof @mach_absolute_time : !llvm.ptr
%ticks = llvm.call %mach_abs_time() : () -> i64

// Convert to nanoseconds using mach_timebase_info
%timebase = llvm.alloca %c1 x !llvm.struct<(i32, i32)> : (i32) -> !llvm.ptr
%mach_timebase = llvm.mlir.addressof @mach_timebase_info : !llvm.ptr
llvm.call %mach_timebase(%timebase) : (!llvm.ptr) -> i32

%numer_ptr = llvm.getelementptr %timebase[0, 0] : !llvm.ptr
%denom_ptr = llvm.getelementptr %timebase[0, 1] : !llvm.ptr
%numer = llvm.load %numer_ptr : !llvm.ptr -> i32
%denom = llvm.load %denom_ptr : !llvm.ptr -> i32

// ticks_ns = ticks * numer / denom
%numer64 = arith.extui %numer : i32 to i64
%denom64 = arith.extui %denom : i32 to i64
%scaled = arith.muli %ticks, %numer64 : i64
%ns = arith.divui %scaled, %denom64 : i64
```

**External Function Declaration:**
```mlir
llvm.func @mach_absolute_time() -> i64 attributes {sym_visibility = "private"}
llvm.func @mach_timebase_info(!llvm.ptr) -> i32 attributes {sym_visibility = "private"}
llvm.func @nanosleep(!llvm.ptr, !llvm.ptr) -> i32 attributes {sym_visibility = "private"}
llvm.func @gettimeofday(!llvm.ptr, !llvm.ptr) -> i32 attributes {sym_visibility = "private"}
```

## Binding Architecture

### BindingTypes.fs

```fsharp
module Alex.Bindings.BindingTypes

open Core.XParsec.Foundation
open Core.PSG.Types
open Core.MLIR.Emitter

/// Target platform identification
type TargetPlatform =
    | Linux_x86_64
    | Linux_ARM64
    | Linux_RISCV64
    | Windows_x86_64
    | MacOS_x86_64
    | MacOS_ARM64
    | Embedded_CortexM
    | WASM

/// Binding match result
type BindingMatch = {
    SymbolPattern: string
    Platform: TargetPlatform
    Priority: int  // Higher = more specific match
}

/// Binding emission function signature
type BindingEmitter =
    MLIRBuilder -> SSAContext -> ProgramSemanticGraph -> PSGNode -> string option

/// Complete binding definition
type Binding = {
    Name: string
    SymbolPatterns: string list        // e.g., ["Alloy.Time.currentTicks"; "*.Time.currentTicks"]
    SupportedPlatforms: TargetPlatform list
    Emit: TargetPlatform -> BindingEmitter
    ExternalDeclarations: TargetPlatform -> string list  // External function declarations needed
}

/// Binding registry for pattern lookup
type BindingRegistry = {
    Bindings: Binding list
    CurrentPlatform: TargetPlatform
}
```

### TimeBindings.fs

```fsharp
module Alex.Bindings.Time.TimeBindings

open Alex.Bindings.BindingTypes
open Core.MLIR.Emitter

/// Time-related binding patterns
let timePatterns = [
    "Alloy.Time.currentTicks"
    "Alloy.Time.highResolutionTicks"
    "Alloy.Time.tickFrequency"
    "Alloy.Time.sleep"
    "Alloy.Time.currentUnixTimestamp"
    "Alloy.TimeApi.currentTicks"  // AutoOpen module variant
    "Alloy.TimeApi.highResolutionTicks"
    "Alloy.TimeApi.tickFrequency"
    "Alloy.TimeApi.sleep"
]

/// Match a PSG node against time patterns
let matchTimeOperation (node: PSGNode) : string option =
    match node.Symbol with
    | Some sym ->
        let fullName = sym.FullName
        timePatterns
        |> List.tryFind (fun pattern ->
            fullName.EndsWith(pattern.Split('.').[pattern.Split('.').Length - 1]))
        |> Option.map (fun _ ->
            fullName.Split('.').[fullName.Split('.').Length - 1])
    | None -> None

/// Dispatch to platform-specific emitter
let emitTimeOperation (platform: TargetPlatform) (operation: string)
                      (builder: MLIRBuilder) (ctx: SSAContext)
                      (psg: ProgramSemanticGraph) (node: PSGNode) : string option =
    match platform with
    | Linux_x86_64 | Linux_ARM64 ->
        Linux.emitTimeOperation operation builder ctx psg node
    | Windows_x86_64 ->
        Windows.emitTimeOperation operation builder ctx psg node
    | MacOS_x86_64 | MacOS_ARM64 ->
        MacOS.emitTimeOperation operation builder ctx psg node
    | _ ->
        // Fallback to portable implementation
        None
```

### Linux.fs (Time)

```fsharp
module Alex.Bindings.Time.Linux

open Core.MLIR.Emitter
open Core.PSG.Types

/// Linux syscall numbers for x86-64
module Syscalls =
    let clock_gettime = 228
    let nanosleep = 35

/// Clock IDs
module ClockId =
    let REALTIME = 0
    let MONOTONIC = 1

/// Emit clock_gettime syscall
let emitClockGettime (builder: MLIRBuilder) (ctx: SSAContext) (clockId: int) : string =
    // Allocate timespec on stack
    let timespec = SSAContext.nextValue ctx
    let one = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i32" one)
    MLIRBuilder.line builder (sprintf "%s = llvm.alloca %s x !llvm.struct<(i64, i64)> : (i32) -> !llvm.ptr" timespec one)

    // Prepare syscall arguments
    let clockIdVal = SSAContext.nextValue ctx
    let syscallNum = SSAContext.nextValue ctx
    let result = SSAContext.nextValue ctx

    MLIRBuilder.line builder (sprintf "%s = arith.constant %d : i64" clockIdVal clockId)
    MLIRBuilder.line builder (sprintf "%s = arith.constant %d : i64" syscallNum Syscalls.clock_gettime)
    MLIRBuilder.line builder (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"=r,{rax},{rdi},{rsi}\" %s, %s, %s : (i64, i64, !llvm.ptr) -> i64"
        result syscallNum clockIdVal timespec)

    // Extract and convert to ticks
    let secPtr = SSAContext.nextValue ctx
    let nsecPtr = SSAContext.nextValue ctx
    let sec = SSAContext.nextValue ctx
    let nsec = SSAContext.nextValue ctx

    MLIRBuilder.line builder (sprintf "%s = llvm.getelementptr %s[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)>" secPtr timespec)
    MLIRBuilder.line builder (sprintf "%s = llvm.getelementptr %s[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)>" nsecPtr timespec)
    MLIRBuilder.line builder (sprintf "%s = llvm.load %s : !llvm.ptr -> i64" sec secPtr)
    MLIRBuilder.line builder (sprintf "%s = llvm.load %s : !llvm.ptr -> i64" nsec nsecPtr)

    // Convert to 100-nanosecond ticks
    let ticksPerSec = SSAContext.nextValue ctx
    let secTicks = SSAContext.nextValue ctx
    let nsecDiv = SSAContext.nextValue ctx
    let nsecTicks = SSAContext.nextValue ctx
    let totalTicks = SSAContext.nextValue ctx

    MLIRBuilder.line builder (sprintf "%s = arith.constant 10000000 : i64" ticksPerSec)
    MLIRBuilder.line builder (sprintf "%s = arith.muli %s, %s : i64" secTicks sec ticksPerSec)
    MLIRBuilder.line builder (sprintf "%s = arith.constant 100 : i64" nsecDiv)
    MLIRBuilder.line builder (sprintf "%s = arith.divui %s, %s : i64" nsecTicks nsec nsecDiv)
    MLIRBuilder.line builder (sprintf "%s = arith.addi %s, %s : i64" totalTicks secTicks nsecTicks)

    totalTicks

/// Emit nanosleep syscall
let emitNanosleep (builder: MLIRBuilder) (ctx: SSAContext) (milliseconds: string) : unit =
    // Convert milliseconds to seconds and nanoseconds
    let thousand = SSAContext.nextValue ctx
    let million = SSAContext.nextValue ctx
    let msExtended = SSAContext.nextValue ctx
    let seconds = SSAContext.nextValue ctx
    let remainder = SSAContext.nextValue ctx
    let nanoseconds = SSAContext.nextValue ctx

    MLIRBuilder.line builder (sprintf "%s = arith.constant 1000 : i64" thousand)
    MLIRBuilder.line builder (sprintf "%s = arith.constant 1000000 : i64" million)
    MLIRBuilder.line builder (sprintf "%s = arith.extsi %s : i32 to i64" msExtended milliseconds)
    MLIRBuilder.line builder (sprintf "%s = arith.divui %s, %s : i64" seconds msExtended thousand)
    MLIRBuilder.line builder (sprintf "%s = arith.remui %s, %s : i64" remainder msExtended thousand)
    MLIRBuilder.line builder (sprintf "%s = arith.muli %s, %s : i64" nanoseconds remainder million)

    // Allocate timespec
    let one = SSAContext.nextValue ctx
    let reqTimespec = SSAContext.nextValue ctx
    let remTimespec = SSAContext.nextValue ctx

    MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i32" one)
    MLIRBuilder.line builder (sprintf "%s = llvm.alloca %s x !llvm.struct<(i64, i64)> : (i32) -> !llvm.ptr" reqTimespec one)
    MLIRBuilder.line builder (sprintf "%s = llvm.alloca %s x !llvm.struct<(i64, i64)> : (i32) -> !llvm.ptr" remTimespec one)

    // Store seconds and nanoseconds
    let secPtr = SSAContext.nextValue ctx
    let nsecPtr = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.getelementptr %s[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)>" secPtr reqTimespec)
    MLIRBuilder.line builder (sprintf "%s = llvm.getelementptr %s[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)>" nsecPtr reqTimespec)
    MLIRBuilder.line builder (sprintf "llvm.store %s, %s : i64, !llvm.ptr" seconds secPtr)
    MLIRBuilder.line builder (sprintf "llvm.store %s, %s : i64, !llvm.ptr" nanoseconds nsecPtr)

    // Call nanosleep syscall
    let syscallNum = SSAContext.nextValue ctx
    let result = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant %d : i64" syscallNum Syscalls.nanosleep)
    MLIRBuilder.line builder (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"=r,{rax},{rdi},{rsi}\" %s, %s, %s : (i64, !llvm.ptr, !llvm.ptr) -> i64"
        result syscallNum reqTimespec remTimespec)

/// Main dispatch for Linux time operations
let emitTimeOperation (operation: string) (builder: MLIRBuilder) (ctx: SSAContext)
                      (psg: ProgramSemanticGraph) (node: PSGNode) : string option =
    match operation with
    | "currentTicks" ->
        Some (emitClockGettime builder ctx ClockId.REALTIME)
    | "highResolutionTicks" ->
        Some (emitClockGettime builder ctx ClockId.MONOTONIC)
    | "tickFrequency" ->
        let freq = SSAContext.nextValue ctx
        MLIRBuilder.line builder (sprintf "%s = arith.constant 10000000 : i64" freq)
        Some freq
    | "sleep" ->
        // Get milliseconds argument from node children
        // For now, emit with placeholder - actual implementation needs argument extraction
        None  // TODO: Extract argument from PSG
    | _ -> None
```

## Integration with Emitter.fs

The existing `emitExpression` function in Emitter.fs will be modified to check the binding registry before falling back to generic emission:

```fsharp
let rec emitExpression (builder: MLIRBuilder) (ctx: SSAContext) (psg: ProgramSemanticGraph)
                       (node: PSGNode) (platform: TargetPlatform) : string option =
    // First, check binding registry for platform-specific emission
    match BindingRegistry.tryEmit platform node builder ctx psg with
    | Some result -> Some result
    | None ->
        // Fall back to generic emission
        match node.SyntaxKind with
        | sk when sk.StartsWith("Const:String:") -> ...
        | "App" | "Application" -> ...
        // ... rest of existing logic
```

## External Declarations

Each platform requires different external function declarations in the MLIR module:

### Linux (using syscalls - no external declarations needed)
Syscalls are emitted inline via `llvm.inline_asm`.

### Windows
```mlir
llvm.func @QueryPerformanceCounter(!llvm.ptr) -> i32
llvm.func @QueryPerformanceFrequency(!llvm.ptr) -> i32
llvm.func @Sleep(i32)
llvm.func @GetSystemTimeAsFileTime(!llvm.ptr)
```

### macOS
```mlir
llvm.func @mach_absolute_time() -> i64
llvm.func @mach_timebase_info(!llvm.ptr) -> i32
llvm.func @nanosleep(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @gettimeofday(!llvm.ptr, !llvm.ptr) -> i32
```

## Testing Strategy

1. **Unit Tests**: Each platform emitter tested in isolation
2. **Integration Tests**: Full compilation through TimeLoop sample
3. **Platform Tests**: Actual execution on Linux, Windows, macOS
4. **Cross-Compilation Tests**: Build from one platform for another

## Future Extensions

### ARM Cortex-M (Embedded)
- SysTick register access for timing
- No syscalls - direct memory-mapped I/O
- Interrupt-based sleep

### WASM
- `performance.now()` import
- `setTimeout` for sleep (async)

### RISC-V
- `rdtime` instruction for cycle counter
- `mtime` CSR for real-time counter

---

*Design Document for Alex Time Bindings - December 2024*
