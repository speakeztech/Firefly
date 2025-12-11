module Alex.Bindings.Time.TimeBindings

open Alex.CodeGeneration.MLIRBuilder
open Alex.Bindings.BindingTypes

// ===================================================================
// Platform Data: Syscall numbers and clock IDs
// ===================================================================

module SyscallData =
    /// Linux x86-64 syscall numbers
    let linuxSyscalls = Map [
        "clock_gettime", 228L
        "nanosleep", 35L
    ]

    /// macOS uses library calls, not direct syscalls for time
    /// These are placeholder - actual implementation uses function calls

    /// Clock IDs (POSIX standard)
    let CLOCK_REALTIME = 0L
    let CLOCK_MONOTONIC = 1L

// ===================================================================
// Common Types
// ===================================================================

let timespecType = Struct [Integer I64; Integer I64]  // { tv_sec: i64, tv_nsec: i64 }

// ===================================================================
// Linux Time Implementation
// ===================================================================

/// Emit clock_gettime syscall for Linux
let emitLinuxClockGettime (clockId: int64) : MLIR<Val> = mlir {
    // Allocate timespec on stack
    let! one = arith.constant 1L I64
    let! timespec = llvm.alloca one timespecType

    // Prepare syscall arguments
    let! clockIdVal = arith.constant clockId I64
    let! syscallNum = arith.constant 228L I64  // clock_gettime

    // Execute syscall
    let! _result = llvm.inlineAsm "syscall" "={rax},{rax},{rdi},{rsi},~{rcx},~{r11},~{memory}"
                    [{ SSA = syscallNum.SSA; Type = Integer I64 }
                     { SSA = clockIdVal.SSA; Type = Integer I64 }
                     timespec] (Integer I64)

    // Extract seconds and nanoseconds
    let! zeroIdx = arith.constant 0L I64
    let! oneIdx = arith.constant 1L I64
    let! secPtr = llvm.getelementptr timespec timespecType [zeroIdx; zeroIdx]
    let! nsecPtr = llvm.getelementptr timespec timespecType [zeroIdx; oneIdx]
    let! sec = llvm.load (Integer I64) secPtr
    let! nsec = llvm.load (Integer I64) nsecPtr

    // Convert to 100-nanosecond ticks
    let! ticksPerSec = arith.constant 10000000L I64
    let! secTicks = arith.muli sec ticksPerSec
    let! nsecDiv = arith.constant 100L I64
    let! nsecTicks = arith.divsi nsec nsecDiv
    let! totalTicks = arith.addi secTicks nsecTicks

    return totalTicks
}

/// Emit nanosleep syscall for Linux
let emitLinuxNanosleep (milliseconds: Val) : MLIR<unit> = mlir {
    // Convert milliseconds to seconds and nanoseconds
    let! thousand = arith.constant 1000L I64
    let! million = arith.constant 1000000L I64

    let! msExtended =
        match milliseconds.Type with
        | Integer I32 -> arith.extsi milliseconds I64
        | _ -> mlir { return milliseconds }

    let! seconds = arith.divsi msExtended thousand
    let! remainder = arith.remsi msExtended thousand
    let! nanoseconds = arith.muli remainder million

    // Allocate timespec structs
    let! one = arith.constant 1L I64
    let! reqTimespec = llvm.alloca one timespecType
    let! remTimespec = llvm.alloca one timespecType

    // Store seconds and nanoseconds
    let! zeroIdx = arith.constant 0L I64
    let! oneIdx = arith.constant 1L I64
    let! secPtr = llvm.getelementptr reqTimespec timespecType [zeroIdx; zeroIdx]
    let! nsecPtr = llvm.getelementptr reqTimespec timespecType [zeroIdx; oneIdx]
    do! llvm.store seconds secPtr
    do! llvm.store nanoseconds nsecPtr

    // Call nanosleep syscall
    let! syscallNum = arith.constant 35L I64  // nanosleep
    let! _result = llvm.inlineAsm "syscall" "={rax},{rax},{rdi},{rsi},~{rcx},~{r11},~{memory}"
                    [{ SSA = syscallNum.SSA; Type = Integer I64 }
                     reqTimespec
                     remTimespec] (Integer I64)
    return ()
}

// ===================================================================
// Extern Primitive Bindings (Platform-Dispatched)
// ===================================================================

/// fidelity_get_current_ticks - get current time in .NET ticks format
let bindGetCurrentTicks (platform: TargetPlatform) (_prim: ExternPrimitive) : MLIR<EmissionResult> = mlir {
    match platform.OS with
    | Linux ->
        let! ticks = emitLinuxClockGettime SyscallData.CLOCK_REALTIME
        // Add Unix epoch offset to convert to .NET ticks (since 0001-01-01)
        let! epochOffset = arith.constant 621355968000000000L I64
        let! result = arith.addi ticks epochOffset
        return Emitted result
    | MacOS ->
        // TODO: macOS implementation using gettimeofday
        return NotSupported "macOS time not yet implemented"
    | Windows ->
        // TODO: Windows implementation using GetSystemTimeAsFileTime
        return NotSupported "Windows time not yet implemented"
    | _ ->
        return NotSupported $"Time not supported on {platform.OS}"
}

/// fidelity_get_monotonic_ticks - get high-resolution monotonic ticks
let bindGetMonotonicTicks (platform: TargetPlatform) (_prim: ExternPrimitive) : MLIR<EmissionResult> = mlir {
    match platform.OS with
    | Linux ->
        let! ticks = emitLinuxClockGettime SyscallData.CLOCK_MONOTONIC
        return Emitted ticks
    | MacOS ->
        return NotSupported "macOS monotonic time not yet implemented"
    | Windows ->
        return NotSupported "Windows monotonic time not yet implemented"
    | _ ->
        return NotSupported $"Monotonic time not supported on {platform.OS}"
}

/// fidelity_get_tick_frequency - get ticks per second
let bindGetTickFrequency (_platform: TargetPlatform) (_prim: ExternPrimitive) : MLIR<EmissionResult> = mlir {
    // All platforms use 100-nanosecond ticks = 10,000,000 per second
    let! freq = arith.constant 10000000L I64
    return Emitted freq
}

/// fidelity_sleep - sleep for specified milliseconds
let bindSleep (platform: TargetPlatform) (prim: ExternPrimitive) : MLIR<EmissionResult> = mlir {
    match prim.Args with
    | [msArg] ->
        match platform.OS with
        | Linux ->
            do! emitLinuxNanosleep msArg
            return EmittedVoid
        | MacOS ->
            return NotSupported "macOS sleep not yet implemented"
        | Windows ->
            return NotSupported "Windows sleep not yet implemented"
        | _ ->
            return NotSupported $"Sleep not supported on {platform.OS}"
    | _ ->
        return NotSupported "sleep requires milliseconds argument"
}

// ===================================================================
// Registration
// ===================================================================

/// Register all time bindings for all platforms
let registerBindings () =
    // Linux bindings
    ExternDispatch.register Linux X86_64 "fidelity_get_current_ticks"
        (fun ext -> bindGetCurrentTicks TargetPlatform.linux_x86_64 ext)
    ExternDispatch.register Linux X86_64 "fidelity_get_monotonic_ticks"
        (fun ext -> bindGetMonotonicTicks TargetPlatform.linux_x86_64 ext)
    ExternDispatch.register Linux X86_64 "fidelity_get_tick_frequency"
        (fun ext -> bindGetTickFrequency TargetPlatform.linux_x86_64 ext)
    ExternDispatch.register Linux X86_64 "fidelity_sleep"
        (fun ext -> bindSleep TargetPlatform.linux_x86_64 ext)

    // Linux ARM64 (same implementation, different arch tag)
    ExternDispatch.register Linux ARM64 "fidelity_get_current_ticks"
        (fun ext -> bindGetCurrentTicks { TargetPlatform.linux_x86_64 with Arch = ARM64 } ext)
    ExternDispatch.register Linux ARM64 "fidelity_get_monotonic_ticks"
        (fun ext -> bindGetMonotonicTicks { TargetPlatform.linux_x86_64 with Arch = ARM64 } ext)
    ExternDispatch.register Linux ARM64 "fidelity_get_tick_frequency"
        (fun ext -> bindGetTickFrequency { TargetPlatform.linux_x86_64 with Arch = ARM64 } ext)
    ExternDispatch.register Linux ARM64 "fidelity_sleep"
        (fun ext -> bindSleep { TargetPlatform.linux_x86_64 with Arch = ARM64 } ext)

    // macOS bindings (placeholder - will return NotSupported until implemented)
    ExternDispatch.register MacOS X86_64 "fidelity_get_current_ticks"
        (fun ext -> bindGetCurrentTicks TargetPlatform.macos_x86_64 ext)
    ExternDispatch.register MacOS X86_64 "fidelity_sleep"
        (fun ext -> bindSleep TargetPlatform.macos_x86_64 ext)
    ExternDispatch.register MacOS ARM64 "fidelity_get_current_ticks"
        (fun ext -> bindGetCurrentTicks TargetPlatform.macos_arm64 ext)
    ExternDispatch.register MacOS ARM64 "fidelity_sleep"
        (fun ext -> bindSleep TargetPlatform.macos_arm64 ext)
