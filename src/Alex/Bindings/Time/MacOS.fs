module Alex.Bindings.Time.MacOS

open Core.PSG.Types
open Alex.CodeGeneration.EmissionContext
open Alex.Bindings.BindingTypes

// ===================================================================
// macOS Time Functions
// Uses libSystem functions rather than raw syscalls
// ===================================================================

/// Emit mach_absolute_time() call for high-resolution timing
let emitMachAbsoluteTime (builder: MLIRBuilder) (ctx: SSAContext) : string =
    let result = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.call @mach_absolute_time() : () -> i64" result)
    result

/// Emit mach_timebase_info() to get conversion factors
/// Returns (numerator, denominator) for converting mach_absolute_time to nanoseconds
let emitMachTimebaseInfo (builder: MLIRBuilder) (ctx: SSAContext) : string * string =
    // Allocate mach_timebase_info_t: struct { uint32 numer; uint32 denom; }
    let one = SSAContext.nextValue ctx
    let timebase = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i32" one)
    MLIRBuilder.line builder (sprintf "%s = llvm.alloca %s x !llvm.struct<(i32, i32)> : (i32) -> !llvm.ptr" timebase one)

    // Call mach_timebase_info
    let resultCode = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.call @mach_timebase_info(%s) : (!llvm.ptr) -> i32" resultCode timebase)

    // Extract numerator and denominator
    let numerPtr = SSAContext.nextValue ctx
    let denomPtr = SSAContext.nextValue ctx
    let numer = SSAContext.nextValue ctx
    let denom = SSAContext.nextValue ctx

    MLIRBuilder.line builder (sprintf "%s = llvm.getelementptr %s[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32)>" numerPtr timebase)
    MLIRBuilder.line builder (sprintf "%s = llvm.getelementptr %s[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32)>" denomPtr timebase)
    MLIRBuilder.line builder (sprintf "%s = llvm.load %s : !llvm.ptr -> i32" numer numerPtr)
    MLIRBuilder.line builder (sprintf "%s = llvm.load %s : !llvm.ptr -> i32" denom denomPtr)

    (numer, denom)

/// Emit gettimeofday() call for wall-clock time
let emitGettimeofday (builder: MLIRBuilder) (ctx: SSAContext) : string =
    // Allocate timeval: struct { int64 tv_sec; int32 tv_usec; } (on 64-bit)
    // Note: On macOS, timeval uses time_t (64-bit) and suseconds_t (32-bit padded)
    let one = SSAContext.nextValue ctx
    let timeval = SSAContext.nextValue ctx
    let nullPtr = SSAContext.nextValue ctx

    MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i32" one)
    MLIRBuilder.line builder (sprintf "%s = llvm.alloca %s x !llvm.struct<(i64, i64)> : (i32) -> !llvm.ptr" timeval one)
    MLIRBuilder.line builder (sprintf "%s = llvm.mlir.zero : !llvm.ptr" nullPtr)

    // Call gettimeofday(timeval, NULL)
    let resultCode = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.call @gettimeofday(%s, %s) : (!llvm.ptr, !llvm.ptr) -> i32" resultCode timeval nullPtr)

    // Extract seconds and microseconds
    let secPtr = SSAContext.nextValue ctx
    let usecPtr = SSAContext.nextValue ctx
    let sec = SSAContext.nextValue ctx
    let usec = SSAContext.nextValue ctx

    MLIRBuilder.line builder (sprintf "%s = llvm.getelementptr %s[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)>" secPtr timeval)
    MLIRBuilder.line builder (sprintf "%s = llvm.getelementptr %s[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)>" usecPtr timeval)
    MLIRBuilder.line builder (sprintf "%s = llvm.load %s : !llvm.ptr -> i64" sec secPtr)
    MLIRBuilder.line builder (sprintf "%s = llvm.load %s : !llvm.ptr -> i64" usec usecPtr)

    // Convert to 100-nanosecond ticks
    // ticks = sec * 10_000_000 + usec * 10
    let ticksPerSec = SSAContext.nextValue ctx
    let secTicks = SSAContext.nextValue ctx
    let usecMult = SSAContext.nextValue ctx
    let usecTicks = SSAContext.nextValue ctx
    let totalTicks = SSAContext.nextValue ctx

    MLIRBuilder.line builder (sprintf "%s = arith.constant 10000000 : i64" ticksPerSec)
    MLIRBuilder.line builder (sprintf "%s = arith.muli %s, %s : i64" secTicks sec ticksPerSec)
    MLIRBuilder.line builder (sprintf "%s = arith.constant 10 : i64" usecMult)
    MLIRBuilder.line builder (sprintf "%s = arith.muli %s, %s : i64" usecTicks usec usecMult)
    MLIRBuilder.line builder (sprintf "%s = arith.addi %s, %s : i64" totalTicks secTicks usecTicks)

    totalTicks

/// Emit nanosleep() call
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

    // Allocate timespec structs
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

    // Call nanosleep
    let resultCode = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.call @nanosleep(%s, %s) : (!llvm.ptr, !llvm.ptr) -> i32" resultCode reqTimespec remTimespec)

// ===================================================================
// Time Operation Dispatch
// ===================================================================

/// Check if a node matches a time operation pattern
let matchesTimePattern (node: PSGNode) : string option =
    match node.Symbol with
    | Some sym ->
        let fullName = sym.FullName
        if fullName.Contains("Time.currentTicks") || fullName.Contains("TimeApi.currentTicks") then
            Some "currentTicks"
        elif fullName.Contains("Time.highResolutionTicks") || fullName.Contains("TimeApi.highResolutionTicks") then
            Some "highResolutionTicks"
        elif fullName.Contains("Time.tickFrequency") || fullName.Contains("TimeApi.tickFrequency") then
            Some "tickFrequency"
        elif fullName.Contains("Time.sleep") || fullName.Contains("TimeApi.sleep") then
            Some "sleep"
        elif fullName.Contains("Time.currentUnixTimestamp") || fullName.Contains("TimeApi.currentUnixTimestamp") then
            Some "currentUnixTimestamp"
        else None
    | None -> None

/// Emit a time operation for macOS
let emitTimeOperation (operation: string) (builder: MLIRBuilder) (ctx: SSAContext)
                      (psg: ProgramSemanticGraph) (node: PSGNode) (args: string list) : EmissionResult =
    match operation with
    | "currentTicks" ->
        let ticks = emitGettimeofday builder ctx
        // Add Unix epoch offset to convert to .NET ticks
        let epochOffset = SSAContext.nextValue ctx
        let result = SSAContext.nextValue ctx
        MLIRBuilder.line builder (sprintf "%s = arith.constant 621355968000000000 : i64" epochOffset)
        MLIRBuilder.line builder (sprintf "%s = arith.addi %s, %s : i64" result ticks epochOffset)
        Emitted result

    | "highResolutionTicks" ->
        // Use mach_absolute_time for high-resolution monotonic timing
        let machTicks = emitMachAbsoluteTime builder ctx
        let (numer, denom) = emitMachTimebaseInfo builder ctx

        // Convert mach ticks to nanoseconds: ns = ticks * numer / denom
        let numer64 = SSAContext.nextValue ctx
        let denom64 = SSAContext.nextValue ctx
        let scaled = SSAContext.nextValue ctx
        let ns = SSAContext.nextValue ctx

        MLIRBuilder.line builder (sprintf "%s = arith.extui %s : i32 to i64" numer64 numer)
        MLIRBuilder.line builder (sprintf "%s = arith.extui %s : i32 to i64" denom64 denom)
        MLIRBuilder.line builder (sprintf "%s = arith.muli %s, %s : i64" scaled machTicks numer64)
        MLIRBuilder.line builder (sprintf "%s = arith.divui %s, %s : i64" ns scaled denom64)

        // Convert nanoseconds to 100-nanosecond ticks
        let hundred = SSAContext.nextValue ctx
        let result = SSAContext.nextValue ctx
        MLIRBuilder.line builder (sprintf "%s = arith.constant 100 : i64" hundred)
        MLIRBuilder.line builder (sprintf "%s = arith.divui %s, %s : i64" result ns hundred)

        Emitted result

    | "tickFrequency" ->
        // Return 10,000,000 (100-nanosecond ticks per second)
        let freq = SSAContext.nextValue ctx
        MLIRBuilder.line builder (sprintf "%s = arith.constant 10000000 : i64" freq)
        Emitted freq

    | "currentUnixTimestamp" ->
        let ticks = emitGettimeofday builder ctx
        let ticksPerSec = SSAContext.nextValue ctx
        let result = SSAContext.nextValue ctx
        MLIRBuilder.line builder (sprintf "%s = arith.constant 10000000 : i64" ticksPerSec)
        MLIRBuilder.line builder (sprintf "%s = arith.divui %s, %s : i64" result ticks ticksPerSec)
        Emitted result

    | "sleep" ->
        match args with
        | [msArg] ->
            emitNanosleep builder ctx msArg
            EmittedVoid
        | _ ->
            NotSupported "sleep requires milliseconds argument"

    | _ -> DeferToGeneric

// ===================================================================
// Platform Binding Registration
// ===================================================================

/// Create the macOS time binding
let createBinding () : PlatformBinding =
    {
        Name = "MacOS.Time"
        SymbolPatterns = [
            "Alloy.Time.currentTicks"
            "Alloy.Time.highResolutionTicks"
            "Alloy.Time.tickFrequency"
            "Alloy.Time.sleep"
            "Alloy.Time.currentUnixTimestamp"
            "Alloy.TimeApi.currentTicks"
            "Alloy.TimeApi.highResolutionTicks"
            "Alloy.TimeApi.tickFrequency"
            "Alloy.TimeApi.sleep"
            "Alloy.TimeApi.currentUnixTimestamp"
        ]
        SupportedPlatforms = [
            TargetPlatform.macos_x86_64
            TargetPlatform.macos_arm64
        ]
        Matches = fun node -> matchesTimePattern node |> Option.isSome
        Emit = fun _platform builder ctx psg node ->
            match matchesTimePattern node with
            | Some operation -> emitTimeOperation operation builder ctx psg node []
            | None -> DeferToGeneric
        GetExternalDeclarations = fun _platform ->
            // macOS requires external function declarations for libSystem
            [
                { Name = "mach_absolute_time"; Signature = "() -> i64"; Library = Some "libSystem.B.dylib" }
                { Name = "mach_timebase_info"; Signature = "(!llvm.ptr) -> i32"; Library = Some "libSystem.B.dylib" }
                { Name = "nanosleep"; Signature = "(!llvm.ptr, !llvm.ptr) -> i32"; Library = Some "libSystem.B.dylib" }
                { Name = "gettimeofday"; Signature = "(!llvm.ptr, !llvm.ptr) -> i32"; Library = Some "libSystem.B.dylib" }
            ]
    }
