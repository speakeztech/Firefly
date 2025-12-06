# Alloy BCL Compatibility Design

## Overview

Alloy is the core library for the Fidelity Framework, providing platform-abstracted, native-compilable implementations of essential functionality. A key design principle is **BCL API surface compatibility**: Alloy should mimic the .NET Base Class Library (BCL) structure and API signatures on the surface level, while implementing Fidelity-appropriate SRTP-based "deterministic memory" machinery under the covers.

This document establishes the design principles for achieving BCL compatibility in Alloy, with a focus on the `DateTime` and related time APIs as the initial case study.

## Design Principles

### 1. Surface-Level BCL Mimicry

Alloy APIs should look familiar to F# developers coming from .NET:

```fsharp
// BCL/.NET way:
let now = DateTime.Now
let formatted = now.ToString()

// Alloy way (should be identical surface API):
let now = DateTime.Now
let formatted = now.ToString()
```

The goal is **zero cognitive load** for developers migrating from .NET. They should not need to learn new naming conventions or API patterns.

### 2. Implementation Independence

While the surface API matches BCL, the implementation is completely independent:

- **No .NET runtime dependencies** - compiles to native code via MLIR/LLVM
- **SRTP-based polymorphism** - uses Statically Resolved Type Parameters for static dispatch
- **Platform-specific implementations** - Linux, macOS, Windows each have syscall-based implementations
- **Deterministic memory** - no garbage collector, predictable allocation patterns

### 3. Composition Over Complexity

From the blog post "ByRef Resolved":
> "Instead of exposing complexity and asking developers to manage it correctly, the framework aims to hide complexity behind intuitive composition patterns."

Alloy achieves this by:
- Providing familiar BCL-like APIs at the top level
- Hiding platform-specific syscall details in internal modules
- Using SRTP to eliminate runtime dispatch overhead

## DateTime API Design

### Target BCL API Surface

The following BCL `DateTime` members should be supported:

```fsharp
// Static properties
DateTime.Now        : DateTime
DateTime.UtcNow     : DateTime
DateTime.Today      : DateTime
DateTime.MinValue   : DateTime
DateTime.MaxValue   : DateTime

// Instance properties
member Year         : int
member Month        : int
member Day          : int
member Hour         : int
member Minute       : int
member Second       : int
member Millisecond  : int
member DayOfWeek    : DayOfWeek
member DayOfYear    : int
member Ticks        : int64

// Instance methods
member ToString()           : string
member ToString(format)     : string
member ToShortDateString()  : string
member ToShortTimeString()  : string
member ToLongDateString()   : string
member ToLongTimeString()   : string
member AddDays(days)        : DateTime
member AddHours(hours)      : DateTime
member AddMinutes(minutes)  : DateTime
member AddSeconds(seconds)  : DateTime
member Subtract(other)      : TimeSpan
```

### Current Alloy Time API (Non-BCL Compatible)

The current implementation in `~/repos/Alloy/src/Time.fs` uses non-standard names:

```fsharp
// Current (non-BCL) API:
currentUnixTimestamp()      // Should be: DateTime.UtcNow (internally)
currentDateTimeString()     // Should be: DateTime.Now.ToString()
currentTicks()              // Should be: DateTime.Now.Ticks
sleep(ms)                   // Should be: Thread.Sleep(ms) or Task.Delay(ms)
```

### Proposed Alloy DateTime Implementation

```fsharp
namespace Alloy

/// BCL-compatible DateTime type for Fidelity native compilation
[<Struct>]
type DateTime =
    val private ticks: int64

    new(ticks: int64) = { ticks = ticks }
    new(year, month, day) = ...
    new(year, month, day, hour, minute, second) = ...

    // Static members
    static member Now : DateTime
    static member UtcNow : DateTime
    static member Today : DateTime

    // Instance members
    member this.Year : int
    member this.Month : int
    member this.Day : int
    member this.Hour : int
    member this.Minute : int
    member this.Second : int
    member this.Ticks : int64

    // Formatting
    member this.ToString() : string
    member this.ToString(format: string) : string

    // Arithmetic
    member this.AddDays(value: float) : DateTime
    member this.AddHours(value: float) : DateTime
    member this.Subtract(other: DateTime) : TimeSpan
```

### Platform Implementation Strategy

From "Source-Level Dependency Resolution":
> "The same high-level operation maps to different low-level implementations that are optimized for each platform's conventions and available system services."

```fsharp
// Alloy/Time/Platform.fs - Platform abstraction
module internal Alloy.Time.Platform

#if LINUX
let getCurrentTimeTicks() : int64 =
    // syscall 228 (clock_gettime) with CLOCK_REALTIME
    ...
#elif MACOS
let getCurrentTimeTicks() : int64 =
    // syscall with clock_gettime equivalent
    ...
#elif WINDOWS
let getCurrentTimeTicks() : int64 =
    // GetSystemTimeAsFileTime via inline asm
    ...
#endif
```

## TimeLoop Sample Migration

### Current TimeLoop.fs (Non-BCL)

```fsharp
open Alloy

let displayTimeLoop iterations =
    Console.WriteLine "TimeLoop - Current DateTime Demo"
    let mutable counter = 0
    while counter < iterations do
        let now = currentDateTimeString()  // Non-BCL
        Console.WriteLine now
        Time.sleep 1000                     // Non-BCL
        counter <- counter + 1
    Console.WriteLine "Done."
```

### Target TimeLoop.fs (BCL-Compatible)

```fsharp
open Alloy

let displayTimeLoop iterations =
    Console.WriteLine "TimeLoop - Current DateTime Demo"
    let mutable counter = 0
    while counter < iterations do
        let now = DateTime.Now              // BCL-compatible
        Console.WriteLine (now.ToString())  // BCL-compatible
        Threading.Thread.Sleep 1000         // BCL-compatible
        counter <- counter + 1
    Console.WriteLine "Done."
```

## Integration with Alex/Firefly

### Pattern Recognition in AlloyPatterns.fs

Alex should recognize BCL-style patterns:

```fsharp
/// Time operation patterns matching BCL API
type TimeOp =
    | DateTimeNow              // DateTime.Now
    | DateTimeUtcNow           // DateTime.UtcNow
    | DateTimeToString         // DateTime.ToString()
    | ThreadSleep              // Thread.Sleep(ms)
    | TaskDelay                // Task.Delay(ms)
```

### Emission in Time/Linux.fs

```fsharp
/// Emit DateTime.Now for Linux
let emitDateTimeNow (ctx: SSAContext) : string =
    // Emit clock_gettime syscall
    // Convert to DateTime ticks
    // Return SSA value
    ...

/// Emit DateTime.ToString() for Linux
let emitDateTimeToString (ctx: SSAContext) (dateTimeSSA: string) : string =
    // Format ticks as "YYYY-MM-DD HH:MM:SS"
    // Return string pointer SSA
    ...
```

## Reconciliation: FidelityHelloWorld vs ~/repos/Alloy

Two Alloy implementations currently exist:

1. **~/repos/Alloy/src/** - The canonical Alloy library
2. **~/repos/FidelityHelloWorld/lib/Alloy/** - Sample-specific Alloy

### Action Items

1. **Audit both implementations** - identify differences
2. **Migrate to BCL-compatible API** in ~/repos/Alloy
3. **Update FidelityHelloWorld** to use canonical Alloy
4. **Update TimeLoop** to use canonical Alloy with BCL API
5. **Ensure Firefly samples reference ~/repos/Alloy** properly

## References

### Blog Posts

1. **"From Dotnet To Fidelity Concurrency"** - Establishes Alloy as automatic static resolution library
   > "Alloy analyzes your code during compilation and automatically applies static resolution."

2. **"ByRef Resolved"** - Discusses Alloy's role in memory safety
   > "Good library design should support gradual depth."

3. **"Beyond Zero-Allocation"** - Alloy as foundation for memory model
   > "Fidelity starts with stack-only allocation to illustrate that functional programming doesn't need managed runtimes."

4. **"Source-Level Dependency Resolution"** - Alloy as source-level library
   > "Alloy exists as F# source code in your project's `lib` directory."

5. **"HKTs in Fidelity"** - Alloy's SRTP-based approach
   > "Push complexity into the tooling, not onto the developer."

6. **"RAII in Olivier and Prospero"** - Alloy integration with actors
   > "The Firefly compiler identifies actor state, determines allocation patterns, and generates appropriate RAII semantics."

### External References

- .NET BCL DateTime documentation
- POSIX `clock_gettime` specification
- MLIR dialect design patterns

## Implementation Timeline

1. **Phase 1**: Design DateTime struct with BCL-compatible surface
2. **Phase 2**: Implement platform-specific time retrieval
3. **Phase 3**: Implement ToString() formatting
4. **Phase 4**: Update TimeLoop sample
5. **Phase 5**: Validate end-to-end compilation to native binary

## Success Criteria

TimeLoop compiles and runs as a native binary:
```bash
$ ./timeloop
TimeLoop - Current DateTime Demo
2025-12-06 14:30:45
2025-12-06 14:30:46
2025-12-06 14:30:47
2025-12-06 14:30:48
2025-12-06 14:30:49
Done.
```
