module Alex.Bindings.Time.Linux

open Core.PSG.Types
open Alex.CodeGeneration.EmissionContext
open Alex.Bindings.BindingTypes

// ===================================================================
// Linux Syscall Numbers (x86-64)
// ===================================================================

module Syscalls =
    let clock_gettime = 228
    let nanosleep = 35

module ClockId =
    let REALTIME = 0
    let MONOTONIC = 1

// ===================================================================
// MLIR Emission Helpers
// ===================================================================

/// Emit clock_gettime syscall and return ticks (100-nanosecond intervals)
let emitClockGettime (builder: MLIRBuilder) (ctx: SSAContext) (clockId: int) : string =
    // Allocate timespec on stack: struct { int64 tv_sec; int64 tv_nsec; }
    let one = SSAContext.nextValue ctx
    let timespec = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i32" one)
    MLIRBuilder.line builder (sprintf "%s = llvm.alloca %s x !llvm.struct<(i64, i64)> : (i32) -> !llvm.ptr" timespec one)

    // Prepare syscall arguments
    let clockIdVal = SSAContext.nextValue ctx
    let syscallNum = SSAContext.nextValue ctx
    let result = SSAContext.nextValue ctx

    MLIRBuilder.line builder (sprintf "%s = arith.constant %d : i64" clockIdVal clockId)
    MLIRBuilder.line builder (sprintf "%s = arith.constant %d : i64" syscallNum Syscalls.clock_gettime)

    // Execute syscall: clock_gettime(clock_id, &timespec)
    MLIRBuilder.line builder (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"=r,{rax},{rdi},{rsi}\" %s, %s, %s : (i64, i64, !llvm.ptr) -> i64"
        result syscallNum clockIdVal timespec)

    // Extract seconds and nanoseconds from timespec
    let secPtr = SSAContext.nextValue ctx
    let nsecPtr = SSAContext.nextValue ctx
    let sec = SSAContext.nextValue ctx
    let nsec = SSAContext.nextValue ctx

    MLIRBuilder.line builder (sprintf "%s = llvm.getelementptr %s[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)>" secPtr timespec)
    MLIRBuilder.line builder (sprintf "%s = llvm.getelementptr %s[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)>" nsecPtr timespec)
    MLIRBuilder.line builder (sprintf "%s = llvm.load %s : !llvm.ptr -> i64" sec secPtr)
    MLIRBuilder.line builder (sprintf "%s = llvm.load %s : !llvm.ptr -> i64" nsec nsecPtr)

    // Convert to 100-nanosecond ticks (.NET DateTime ticks format)
    // ticks = sec * 10_000_000 + nsec / 100
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
    // seconds = ms / 1000
    // nanoseconds = (ms % 1000) * 1_000_000
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

    // Allocate request and remainder timespec structs
    let one = SSAContext.nextValue ctx
    let reqTimespec = SSAContext.nextValue ctx
    let remTimespec = SSAContext.nextValue ctx

    MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i32" one)
    MLIRBuilder.line builder (sprintf "%s = llvm.alloca %s x !llvm.struct<(i64, i64)> : (i32) -> !llvm.ptr" reqTimespec one)
    MLIRBuilder.line builder (sprintf "%s = llvm.alloca %s x !llvm.struct<(i64, i64)> : (i32) -> !llvm.ptr" remTimespec one)

    // Store seconds and nanoseconds into request timespec
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

/// Emit currentDateTimeString - formats unix timestamp as "YYYY-MM-DD HH:MM:SS"
/// Returns pointer to stack-allocated string buffer
let emitCurrentDateTimeString (builder: MLIRBuilder) (ctx: SSAContext) : string =
    // Step 1: Get current unix timestamp (seconds since epoch)
    let ticks = emitClockGettime builder ctx ClockId.REALTIME
    let ticksPerSec = SSAContext.nextValue ctx
    let unixSecs = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 10000000 : i64" ticksPerSec)
    MLIRBuilder.line builder (sprintf "%s = arith.divui %s, %s : i64" unixSecs ticks ticksPerSec)

    // Step 2: Allocate buffer for formatted string (20 bytes: "YYYY-MM-DD HH:MM:SS\0")
    let one = SSAContext.nextValue ctx
    let bufSize = SSAContext.nextValue ctx
    let buf = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i32" one)
    MLIRBuilder.line builder (sprintf "%s = arith.constant 20 : i64" bufSize)
    MLIRBuilder.line builder (sprintf "%s = llvm.alloca %s x i8 : (i64) -> !llvm.ptr" buf bufSize)

    // Step 3: Extract time components
    let c60 = SSAContext.nextValue ctx
    let c3600 = SSAContext.nextValue ctx
    let c86400 = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 60 : i64" c60)
    MLIRBuilder.line builder (sprintf "%s = arith.constant 3600 : i64" c3600)
    MLIRBuilder.line builder (sprintf "%s = arith.constant 86400 : i64" c86400)

    // Time of day calculation
    let daySeconds = SSAContext.nextValue ctx
    let hour = SSAContext.nextValue ctx
    let remaining = SSAContext.nextValue ctx
    let minute = SSAContext.nextValue ctx
    let second = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.remsi %s, %s : i64" daySeconds unixSecs c86400)
    MLIRBuilder.line builder (sprintf "%s = arith.divsi %s, %s : i64" hour daySeconds c3600)
    MLIRBuilder.line builder (sprintf "%s = arith.remsi %s, %s : i64" remaining daySeconds c3600)
    MLIRBuilder.line builder (sprintf "%s = arith.divsi %s, %s : i64" minute remaining c60)
    MLIRBuilder.line builder (sprintf "%s = arith.remsi %s, %s : i64" second remaining c60)

    // Calculate total days since epoch for date
    let totalDays = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.divsi %s, %s : i64" totalDays unixSecs c86400)

    // Simplified year/month/day calculation
    // For a proper implementation, would need leap year handling
    // For now, emit a call to a helper function
    MLIRBuilder.line builder (sprintf "// DateTime formatting: calling @__formatDateTime helper")
    let formatResult = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = func.call @__formatDateTime(%s, %s) : (i64, !llvm.ptr) -> i32" formatResult unixSecs buf)

    buf

// ===================================================================
// Time Operation Dispatch
// ===================================================================

/// Check if a node matches a time operation pattern
let matchesTimePattern (node: PSGNode) : string option =
    match node.Symbol with
    | Some sym ->
        let fullName = try sym.FullName with _ -> ""
        // Match various Time module patterns
        if fullName.Contains("Time.currentTicks") || fullName.Contains("TimeApi.currentTicks") then
            Some "currentTicks"
        elif fullName.Contains("Time.highResolutionTicks") || fullName.Contains("TimeApi.highResolutionTicks") then
            Some "highResolutionTicks"
        elif fullName.Contains("Time.tickFrequency") || fullName.Contains("TimeApi.tickFrequency") then
            Some "tickFrequency"
        elif fullName.Contains("Time.sleep") || fullName.Contains("TimeApi.sleep") ||
             fullName.EndsWith(".sleep") then
            Some "sleep"
        elif fullName.Contains("Time.currentUnixTimestamp") || fullName.Contains("TimeApi.currentUnixTimestamp") ||
             fullName.EndsWith(".currentUnixTimestamp") then
            Some "currentUnixTimestamp"
        elif fullName.Contains("currentDateTimeString") || fullName.EndsWith(".currentDateTimeString") then
            Some "currentDateTimeString"
        elif fullName.Contains("currentDateTime") || fullName.EndsWith(".currentDateTime") then
            Some "currentDateTime"
        else None
    | None -> None

/// Emit a time operation for Linux
let emitTimeOperation (operation: string) (builder: MLIRBuilder) (ctx: SSAContext)
                      (psg: ProgramSemanticGraph) (node: PSGNode) (args: string list) : EmissionResult =
    match operation with
    | "currentTicks" ->
        let ticks = emitClockGettime builder ctx ClockId.REALTIME
        // Add Unix epoch offset to convert to .NET ticks
        let epochOffset = SSAContext.nextValue ctx
        let result = SSAContext.nextValue ctx
        MLIRBuilder.line builder (sprintf "%s = arith.constant 621355968000000000 : i64" epochOffset)
        MLIRBuilder.line builder (sprintf "%s = arith.addi %s, %s : i64" result ticks epochOffset)
        Emitted result

    | "highResolutionTicks" ->
        let ticks = emitClockGettime builder ctx ClockId.MONOTONIC
        Emitted ticks

    | "tickFrequency" ->
        // Linux clock_gettime uses nanoseconds, converted to 100ns ticks
        // Frequency is 10,000,000 ticks per second
        let freq = SSAContext.nextValue ctx
        MLIRBuilder.line builder (sprintf "%s = arith.constant 10000000 : i64" freq)
        Emitted freq

    | "currentUnixTimestamp" ->
        // Get current time and convert to Unix timestamp (seconds since epoch)
        let ticks = emitClockGettime builder ctx ClockId.REALTIME
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
            // Need milliseconds argument
            NotSupported "sleep requires milliseconds argument"

    | "currentDateTimeString" ->
        let buf = emitCurrentDateTimeString builder ctx
        Emitted buf

    | "currentDateTime" ->
        // For now, currentDateTime returns same as currentDateTimeString
        // Eventually this should return a DateTime struct
        let buf = emitCurrentDateTimeString builder ctx
        Emitted buf

    | _ -> DeferToGeneric

// ===================================================================
// Platform Binding Registration
// ===================================================================

/// Create the Linux time binding
let createBinding () : PlatformBinding =
    {
        Name = "Linux.Time"
        SymbolPatterns = [
            "Alloy.Time.currentTicks"
            "Alloy.Time.highResolutionTicks"
            "Alloy.Time.tickFrequency"
            "Alloy.Time.sleep"
            "Alloy.Time.currentUnixTimestamp"
            "Alloy.Time.currentDateTimeString"
            "Alloy.Time.currentDateTime"
            "Alloy.TimeApi.currentTicks"
            "Alloy.TimeApi.highResolutionTicks"
            "Alloy.TimeApi.tickFrequency"
            "Alloy.TimeApi.sleep"
            "Alloy.TimeApi.currentUnixTimestamp"
            "Alloy.TimeApi.currentDateTimeString"
            "Alloy.TimeApi.currentDateTime"
        ]
        SupportedPlatforms = [
            TargetPlatform.linux_x86_64
            { TargetPlatform.linux_x86_64 with Arch = ARM64; Triple = "aarch64-unknown-linux-gnu" }
        ]
        Matches = fun node -> matchesTimePattern node |> Option.isSome
        Emit = fun _platform builder ctx psg node ->
            match matchesTimePattern node with
            | Some operation -> emitTimeOperation operation builder ctx psg node []
            | None -> DeferToGeneric
        GetExternalDeclarations = fun _platform ->
            // Linux uses inline syscalls, no external declarations needed
            []
    }
