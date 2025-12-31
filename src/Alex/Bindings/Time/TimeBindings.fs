/// TimeBindings - Platform-specific time bindings (witness-based)
///
/// ARCHITECTURAL FOUNDATION (December 2025):
/// Uses the codata accumulator pattern from MLIRZipper.
/// Bindings are witness functions that take primitive info and zipper,
/// returning an updated zipper with the witnessed MLIR operations.
module Alex.Bindings.Time.TimeBindings

open Alex.CodeGeneration.MLIRTypes
open Alex.Traversal.MLIRZipper
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

    /// Clock IDs (POSIX standard)
    let CLOCK_REALTIME = 0L
    let CLOCK_MONOTONIC = 1L

// ===================================================================
// Common Types
// ===================================================================

let timespecType = Struct [Integer I64; Integer I64]  // { tv_sec: i64, tv_nsec: i64 }

// ===================================================================
// Helper Witness Functions (Memory Operations)
// ===================================================================

/// Witness stack allocation
let witnessAlloca (elemType: MLIRType) (zipper: MLIRZipper) : string * MLIRZipper =
    // First witness constant 1 for the count
    let oneSSA, zipper1 = MLIRZipper.witnessConstant 1L I64 zipper
    let resultSSA, zipper2 = MLIRZipper.yieldSSA zipper1
    let typeStr = Serialize.mlirType elemType
    let text = sprintf "%s = llvm.alloca %s x %s : (i64) -> !llvm.ptr" resultSSA oneSSA typeStr
    resultSSA, MLIRZipper.witnessOpWithResult text resultSSA Pointer zipper2

/// Witness GEP (get element pointer)
let witnessGEP (base_: string) (baseType: MLIRType) (indices: string list) (zipper: MLIRZipper) : string * MLIRZipper =
    let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
    let typeStr = Serialize.mlirType baseType
    let indicesStr = String.concat ", " indices
    let text = sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i64, i64) -> !llvm.ptr, %s"
                   resultSSA base_ indicesStr typeStr
    resultSSA, MLIRZipper.witnessOpWithResult text resultSSA Pointer zipper'

/// Witness load from memory
let witnessLoad (resultType: MLIRType) (ptr: string) (zipper: MLIRZipper) : string * MLIRZipper =
    let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
    let typeStr = Serialize.mlirType resultType
    let text = sprintf "%s = llvm.load %s : !llvm.ptr -> %s" resultSSA ptr typeStr
    resultSSA, MLIRZipper.witnessOpWithResult text resultSSA resultType zipper'

/// Witness store to memory
let witnessStore (value: string) (valueType: MLIRType) (ptr: string) (zipper: MLIRZipper) : MLIRZipper =
    let typeStr = Serialize.mlirType valueType
    let text = sprintf "llvm.store %s, %s : %s, !llvm.ptr" value ptr typeStr
    MLIRZipper.witnessVoidOp text zipper

/// Witness integer multiplication
let witnessMuli (lhs: string) (rhs: string) (ty: MLIRType) (zipper: MLIRZipper) : string * MLIRZipper =
    MLIRZipper.witnessArith "arith.muli" lhs rhs ty zipper

/// Witness signed integer division
let witnessDivsi (lhs: string) (rhs: string) (ty: MLIRType) (zipper: MLIRZipper) : string * MLIRZipper =
    MLIRZipper.witnessArith "arith.divsi" lhs rhs ty zipper

/// Witness signed integer remainder
let witnessRemsi (lhs: string) (rhs: string) (ty: MLIRType) (zipper: MLIRZipper) : string * MLIRZipper =
    MLIRZipper.witnessArith "arith.remsi" lhs rhs ty zipper

/// Witness integer addition
let witnessAddi (lhs: string) (rhs: string) (ty: MLIRType) (zipper: MLIRZipper) : string * MLIRZipper =
    MLIRZipper.witnessArith "arith.addi" lhs rhs ty zipper

/// Witness sign extension if needed
let witnessExtSIIfNeeded (ssaName: string) (fromType: MLIRType) (toWidth: IntegerBitWidth) (zipper: MLIRZipper) : string * MLIRZipper =
    match fromType with
    | Integer fromWidth when fromWidth <> toWidth ->
        let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
        let fromStr = Serialize.integerBitWidth fromWidth
        let toStr = Serialize.integerBitWidth toWidth
        let text = sprintf "%s = arith.extsi %s : %s to %s" resultSSA ssaName fromStr toStr
        resultSSA, MLIRZipper.witnessOpWithResult text resultSSA (Integer toWidth) zipper'
    | _ ->
        ssaName, zipper

// ===================================================================
// Linux Time Implementation (Witness-Based)
// ===================================================================

/// Witness clock_gettime syscall for Linux
/// Returns ticks (100-nanosecond intervals)
let witnessLinuxClockGettime (clockId: int64) (zipper: MLIRZipper) : string * MLIRZipper =
    // Allocate timespec on stack
    let timespecSSA, zipper1 = witnessAlloca timespecType zipper

    // Prepare syscall arguments
    let clockIdSSA, zipper2 = MLIRZipper.witnessConstant clockId I64 zipper1
    let sysNumSSA, zipper3 = MLIRZipper.witnessConstant 228L I64 zipper2

    // Execute syscall (clock_gettime)
    let args = [
        (clockIdSSA, "i64")
        (timespecSSA, "!llvm.ptr")
    ]
    let _resultSSA, zipper4 = MLIRZipper.witnessSyscall sysNumSSA args (Integer I64) zipper3

    // Extract seconds and nanoseconds via GEP
    let zeroSSA, zipper5 = MLIRZipper.witnessConstant 0L I64 zipper4
    let oneSSA, zipper6 = MLIRZipper.witnessConstant 1L I64 zipper5
    let secPtrSSA, zipper7 = witnessGEP timespecSSA timespecType [zeroSSA; zeroSSA] zipper6
    let nsecPtrSSA, zipper8 = witnessGEP timespecSSA timespecType [zeroSSA; oneSSA] zipper7
    let secSSA, zipper9 = witnessLoad (Integer I64) secPtrSSA zipper8
    let nsecSSA, zipper10 = witnessLoad (Integer I64) nsecPtrSSA zipper9

    // Convert to 100-nanosecond ticks
    let ticksPerSecSSA, zipper11 = MLIRZipper.witnessConstant 10000000L I64 zipper10
    let secTicksSSA, zipper12 = witnessMuli secSSA ticksPerSecSSA (Integer I64) zipper11
    let nsecDivSSA, zipper13 = MLIRZipper.witnessConstant 100L I64 zipper12
    let nsecTicksSSA, zipper14 = witnessDivsi nsecSSA nsecDivSSA (Integer I64) zipper13
    let totalTicksSSA, zipper15 = witnessAddi secTicksSSA nsecTicksSSA (Integer I64) zipper14

    totalTicksSSA, zipper15

/// Witness nanosleep syscall for Linux
let witnessLinuxNanosleep (msSSA: string) (msType: MLIRType) (zipper: MLIRZipper) : MLIRZipper =
    // Convert milliseconds to seconds and nanoseconds
    let thousandSSA, zipper1 = MLIRZipper.witnessConstant 1000L I64 zipper
    let millionSSA, zipper2 = MLIRZipper.witnessConstant 1000000L I64 zipper1

    // Extend ms to i64 if needed
    let msExtSSA, zipper3 = witnessExtSIIfNeeded msSSA msType I64 zipper2

    let secondsSSA, zipper4 = witnessDivsi msExtSSA thousandSSA (Integer I64) zipper3
    let remainderSSA, zipper5 = witnessRemsi msExtSSA thousandSSA (Integer I64) zipper4
    let nanosecondsSSA, zipper6 = witnessMuli remainderSSA millionSSA (Integer I64) zipper5

    // Allocate timespec structs
    let reqTimespecSSA, zipper7 = witnessAlloca timespecType zipper6
    let remTimespecSSA, zipper8 = witnessAlloca timespecType zipper7

    // Store seconds and nanoseconds
    let zeroSSA, zipper9 = MLIRZipper.witnessConstant 0L I64 zipper8
    let oneSSA, zipper10 = MLIRZipper.witnessConstant 1L I64 zipper9
    let secPtrSSA, zipper11 = witnessGEP reqTimespecSSA timespecType [zeroSSA; zeroSSA] zipper10
    let nsecPtrSSA, zipper12 = witnessGEP reqTimespecSSA timespecType [zeroSSA; oneSSA] zipper11
    let zipper13 = witnessStore secondsSSA (Integer I64) secPtrSSA zipper12
    let zipper14 = witnessStore nanosecondsSSA (Integer I64) nsecPtrSSA zipper13

    // Call nanosleep syscall
    let sysNumSSA, zipper15 = MLIRZipper.witnessConstant 35L I64 zipper14
    let args = [
        (reqTimespecSSA, "!llvm.ptr")
        (remTimespecSSA, "!llvm.ptr")
    ]
    let _resultSSA, zipper16 = MLIRZipper.witnessSyscall sysNumSSA args (Integer I64) zipper15
    zipper16

// ===================================================================
// Platform Primitive Bindings (Witness-Based)
// ===================================================================

/// getCurrentTicks - get current time in .NET ticks format
/// Witness binding from Alloy.Platform.Bindings.getCurrentTicks
let witnessGetCurrentTicks (platform: TargetPlatform) (_prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match platform.OS with
    | Linux ->
        let ticksSSA, zipper1 = witnessLinuxClockGettime SyscallData.CLOCK_REALTIME zipper
        // Add Unix epoch offset to convert to .NET ticks (since 0001-01-01)
        let epochOffsetSSA, zipper2 = MLIRZipper.witnessConstant 621355968000000000L I64 zipper1
        let resultSSA, zipper3 = witnessAddi ticksSSA epochOffsetSSA (Integer I64) zipper2
        zipper3, WitnessedValue (resultSSA, Integer I64)
    | MacOS ->
        zipper, NotSupported "macOS time not yet implemented"
    | Windows ->
        zipper, NotSupported "Windows time not yet implemented"
    | _ ->
        zipper, NotSupported (sprintf "Time not supported on %A" platform.OS)

/// getMonotonicTicks - get high-resolution monotonic ticks
/// Witness binding from Alloy.Platform.Bindings.getMonotonicTicks
let witnessGetMonotonicTicks (platform: TargetPlatform) (_prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match platform.OS with
    | Linux ->
        let ticksSSA, zipper1 = witnessLinuxClockGettime SyscallData.CLOCK_MONOTONIC zipper
        zipper1, WitnessedValue (ticksSSA, Integer I64)
    | MacOS ->
        zipper, NotSupported "macOS monotonic time not yet implemented"
    | Windows ->
        zipper, NotSupported "Windows monotonic time not yet implemented"
    | _ ->
        zipper, NotSupported (sprintf "Monotonic time not supported on %A" platform.OS)

/// getTickFrequency - get ticks per second
/// Witness binding from Alloy.Platform.Bindings.getTickFrequency
let witnessGetTickFrequency (_platform: TargetPlatform) (_prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    // All platforms use 100-nanosecond ticks = 10,000,000 per second
    let freqSSA, zipper1 = MLIRZipper.witnessConstant 10000000L I64 zipper
    zipper1, WitnessedValue (freqSSA, Integer I64)

/// sleep - sleep for specified milliseconds
/// Witness binding from Alloy.Platform.Bindings.sleep
let witnessSleep (platform: TargetPlatform) (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(msSSA, msType)] ->
        match platform.OS with
        | Linux ->
            let zipper1 = witnessLinuxNanosleep msSSA msType zipper
            zipper1, WitnessedVoid
        | MacOS ->
            zipper, NotSupported "macOS sleep not yet implemented"
        | Windows ->
            zipper, NotSupported "Windows sleep not yet implemented"
        | _ ->
            zipper, NotSupported (sprintf "Sleep not supported on %A" platform.OS)
    | _ ->
        zipper, NotSupported "sleep requires milliseconds argument"

// ===================================================================
// Registration (Witness-Based)
// ===================================================================

/// Register all time bindings for all platforms
/// Entry points match Platform.Bindings function names
let registerBindings () =
    // Linux bindings
    PlatformDispatch.register Linux X86_64 "getCurrentTicks"
        (fun prim zipper -> witnessGetCurrentTicks TargetPlatform.linux_x86_64 prim zipper)
    PlatformDispatch.register Linux X86_64 "getMonotonicTicks"
        (fun prim zipper -> witnessGetMonotonicTicks TargetPlatform.linux_x86_64 prim zipper)
    PlatformDispatch.register Linux X86_64 "getTickFrequency"
        (fun prim zipper -> witnessGetTickFrequency TargetPlatform.linux_x86_64 prim zipper)
    PlatformDispatch.register Linux X86_64 "sleep"
        (fun prim zipper -> witnessSleep TargetPlatform.linux_x86_64 prim zipper)

    // Linux ARM64 (same implementation, different arch tag)
    PlatformDispatch.register Linux ARM64 "getCurrentTicks"
        (fun prim zipper -> witnessGetCurrentTicks { TargetPlatform.linux_x86_64 with Arch = ARM64 } prim zipper)
    PlatformDispatch.register Linux ARM64 "getMonotonicTicks"
        (fun prim zipper -> witnessGetMonotonicTicks { TargetPlatform.linux_x86_64 with Arch = ARM64 } prim zipper)
    PlatformDispatch.register Linux ARM64 "getTickFrequency"
        (fun prim zipper -> witnessGetTickFrequency { TargetPlatform.linux_x86_64 with Arch = ARM64 } prim zipper)
    PlatformDispatch.register Linux ARM64 "sleep"
        (fun prim zipper -> witnessSleep { TargetPlatform.linux_x86_64 with Arch = ARM64 } prim zipper)

    // macOS bindings (placeholder - will return NotSupported until implemented)
    PlatformDispatch.register MacOS X86_64 "getCurrentTicks"
        (fun prim zipper -> witnessGetCurrentTicks TargetPlatform.macos_x86_64 prim zipper)
    PlatformDispatch.register MacOS X86_64 "sleep"
        (fun prim zipper -> witnessSleep TargetPlatform.macos_x86_64 prim zipper)
    PlatformDispatch.register MacOS ARM64 "getCurrentTicks"
        (fun prim zipper -> witnessGetCurrentTicks TargetPlatform.macos_arm64 prim zipper)
    PlatformDispatch.register MacOS ARM64 "sleep"
        (fun prim zipper -> witnessSleep TargetPlatform.macos_arm64 prim zipper)
