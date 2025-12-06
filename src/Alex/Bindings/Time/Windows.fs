module Alex.Bindings.Time.Windows

open Core.PSG.Types
open Alex.CodeGeneration.EmissionContext
open Alex.Bindings.BindingTypes

// ===================================================================
// Windows Time Functions
// Uses kernel32.dll functions
// ===================================================================

/// Emit QueryPerformanceCounter call for high-resolution timing
let emitQueryPerformanceCounter (builder: MLIRBuilder) (ctx: SSAContext) : string =
    // Allocate LARGE_INTEGER for the counter value
    let one = SSAContext.nextValue ctx
    let counter = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i32" one)
    MLIRBuilder.line builder (sprintf "%s = llvm.alloca %s x i64 : (i32) -> !llvm.ptr" counter one)

    // Call QueryPerformanceCounter
    let resultCode = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.call @QueryPerformanceCounter(%s) : (!llvm.ptr) -> i32" resultCode counter)

    // Load and return the counter value
    let result = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.load %s : !llvm.ptr -> i64" result counter)
    result

/// Emit QueryPerformanceFrequency call
let emitQueryPerformanceFrequency (builder: MLIRBuilder) (ctx: SSAContext) : string =
    // Allocate LARGE_INTEGER for the frequency
    let one = SSAContext.nextValue ctx
    let freq = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i32" one)
    MLIRBuilder.line builder (sprintf "%s = llvm.alloca %s x i64 : (i32) -> !llvm.ptr" freq one)

    // Call QueryPerformanceFrequency
    let resultCode = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.call @QueryPerformanceFrequency(%s) : (!llvm.ptr) -> i32" resultCode freq)

    // Load and return the frequency value
    let result = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.load %s : !llvm.ptr -> i64" result freq)
    result

/// Emit GetSystemTimeAsFileTime call for wall-clock time
let emitGetSystemTimeAsFileTime (builder: MLIRBuilder) (ctx: SSAContext) : string =
    // Allocate FILETIME: struct { uint32 dwLowDateTime; uint32 dwHighDateTime; }
    let one = SSAContext.nextValue ctx
    let fileTime = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i32" one)
    MLIRBuilder.line builder (sprintf "%s = llvm.alloca %s x !llvm.struct<(i32, i32)> : (i32) -> !llvm.ptr" fileTime one)

    // Call GetSystemTimeAsFileTime (no return value)
    MLIRBuilder.line builder (sprintf "llvm.call @GetSystemTimeAsFileTime(%s) : (!llvm.ptr) -> ()" fileTime)

    // Extract low and high parts
    let lowPtr = SSAContext.nextValue ctx
    let highPtr = SSAContext.nextValue ctx
    let low = SSAContext.nextValue ctx
    let high = SSAContext.nextValue ctx

    MLIRBuilder.line builder (sprintf "%s = llvm.getelementptr %s[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32)>" lowPtr fileTime)
    MLIRBuilder.line builder (sprintf "%s = llvm.getelementptr %s[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32)>" highPtr fileTime)
    MLIRBuilder.line builder (sprintf "%s = llvm.load %s : !llvm.ptr -> i32" low lowPtr)
    MLIRBuilder.line builder (sprintf "%s = llvm.load %s : !llvm.ptr -> i32" high highPtr)

    // Combine into 64-bit value: result = (high << 32) | low
    let low64 = SSAContext.nextValue ctx
    let high64 = SSAContext.nextValue ctx
    let thirtyTwo = SSAContext.nextValue ctx
    let highShifted = SSAContext.nextValue ctx
    let result = SSAContext.nextValue ctx

    MLIRBuilder.line builder (sprintf "%s = arith.extui %s : i32 to i64" low64 low)
    MLIRBuilder.line builder (sprintf "%s = arith.extui %s : i32 to i64" high64 high)
    MLIRBuilder.line builder (sprintf "%s = arith.constant 32 : i64" thirtyTwo)
    MLIRBuilder.line builder (sprintf "%s = arith.shli %s, %s : i64" highShifted high64 thirtyTwo)
    MLIRBuilder.line builder (sprintf "%s = arith.ori %s, %s : i64" result highShifted low64)

    result

/// Emit Sleep call
let emitSleep (builder: MLIRBuilder) (ctx: SSAContext) (milliseconds: string) : unit =
    // Sleep takes DWORD (32-bit) milliseconds
    MLIRBuilder.line builder (sprintf "llvm.call @Sleep(%s) : (i32) -> ()" milliseconds)

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

/// Emit a time operation for Windows
let emitTimeOperation (operation: string) (builder: MLIRBuilder) (ctx: SSAContext)
                      (psg: ProgramSemanticGraph) (node: PSGNode) (args: string list) : EmissionResult =
    match operation with
    | "currentTicks" ->
        // GetSystemTimeAsFileTime returns Windows FILETIME (100-nanosecond intervals since 1601)
        // Need to convert to .NET ticks (100-nanosecond intervals since 0001)
        let fileTime = emitGetSystemTimeAsFileTime builder ctx

        // Windows file time epoch offset: ticks from 0001-01-01 to 1601-01-01
        // This is 504911232000000000 ticks
        let epochOffset = SSAContext.nextValue ctx
        let result = SSAContext.nextValue ctx
        MLIRBuilder.line builder (sprintf "%s = arith.constant 504911232000000000 : i64" epochOffset)
        MLIRBuilder.line builder (sprintf "%s = arith.addi %s, %s : i64" result fileTime epochOffset)
        Emitted result

    | "highResolutionTicks" ->
        let ticks = emitQueryPerformanceCounter builder ctx
        Emitted ticks

    | "tickFrequency" ->
        let freq = emitQueryPerformanceFrequency builder ctx
        Emitted freq

    | "currentUnixTimestamp" ->
        // Get file time and convert to Unix timestamp
        let fileTime = emitGetSystemTimeAsFileTime builder ctx

        // Windows file time to Unix: subtract epoch diff and divide by 10,000,000
        // Epoch diff: 116444736000000000 (100-nanosecond intervals from 1601 to 1970)
        let epochDiff = SSAContext.nextValue ctx
        let adjusted = SSAContext.nextValue ctx
        let ticksPerSec = SSAContext.nextValue ctx
        let result = SSAContext.nextValue ctx

        MLIRBuilder.line builder (sprintf "%s = arith.constant 116444736000000000 : i64" epochDiff)
        MLIRBuilder.line builder (sprintf "%s = arith.subi %s, %s : i64" adjusted fileTime epochDiff)
        MLIRBuilder.line builder (sprintf "%s = arith.constant 10000000 : i64" ticksPerSec)
        MLIRBuilder.line builder (sprintf "%s = arith.divui %s, %s : i64" result adjusted ticksPerSec)
        Emitted result

    | "sleep" ->
        match args with
        | [msArg] ->
            emitSleep builder ctx msArg
            EmittedVoid
        | _ ->
            NotSupported "sleep requires milliseconds argument"

    | _ -> DeferToGeneric

// ===================================================================
// Platform Binding Registration
// ===================================================================

/// Create the Windows time binding
let createBinding () : PlatformBinding =
    {
        Name = "Windows.Time"
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
            TargetPlatform.windows_x86_64
        ]
        Matches = fun node -> matchesTimePattern node |> Option.isSome
        Emit = fun _platform builder ctx psg node ->
            match matchesTimePattern node with
            | Some operation -> emitTimeOperation operation builder ctx psg node []
            | None -> DeferToGeneric
        GetExternalDeclarations = fun _platform ->
            // Windows requires external function declarations for kernel32
            [
                { Name = "QueryPerformanceCounter"; Signature = "(!llvm.ptr) -> i32"; Library = Some "kernel32.dll" }
                { Name = "QueryPerformanceFrequency"; Signature = "(!llvm.ptr) -> i32"; Library = Some "kernel32.dll" }
                { Name = "Sleep"; Signature = "(i32) -> ()"; Library = Some "kernel32.dll" }
                { Name = "GetSystemTimeAsFileTime"; Signature = "(!llvm.ptr) -> ()"; Library = Some "kernel32.dll" }
            ]
    }
