module Alex.Lite.TimeEmitter

open System
open System.Text
open Core.PSG.Types

/// Platform detection for Time library
type TimePlatform = 
    | Windows
    | Linux
    | MacOS
    | Portable

/// Detect target platform from PSG or config
let detectPlatform (psg: ProgramSemanticGraph) =
    // Look for platform registration calls in PSG
    let platformNodes = 
        psg.Nodes 
        |> Map.filter (fun _ node -> 
            node.Symbol.IsSome && 
            node.Symbol.Value.FullName.Contains("Time.Windows") ||
            node.Symbol.Value.FullName.Contains("Time.Linux") ||
            node.Symbol.Value.FullName.Contains("Time.MacOS"))
    
    if Map.isEmpty platformNodes then Portable
    elif Map.exists (fun _ n -> n.Symbol.Value.FullName.Contains("Windows")) platformNodes then Windows
    elif Map.exists (fun _ n -> n.Symbol.Value.FullName.Contains("Linux")) platformNodes then Linux
    elif Map.exists (fun _ n -> n.Symbol.Value.FullName.Contains("MacOS")) platformNodes then MacOS
    else Portable

/// Generate LLVM dialect for Time.highResolutionTicks()
let emitHighResolutionTicks (platform: TimePlatform) (resultVar: string) =
    match platform with
    | Windows ->
        // Windows: Use QueryPerformanceCounter
        sprintf """
    %%ticks_ptr = llvm.alloca i64 : (i64) -> !llvm.ptr<i64>
    %%qpc_result = llvm.call @QueryPerformanceCounter(%%ticks_ptr) : (!llvm.ptr<i64>) -> i32
    %s = llvm.load %%ticks_ptr : !llvm.ptr<i64> -> i64""" resultVar
        
    | Linux | MacOS ->
        // POSIX: Use clock_gettime with CLOCK_MONOTONIC
        sprintf """
    %%timespec = llvm.alloca !llvm.array<2 x i64> : (i32) -> !llvm.ptr<!llvm.array<2 x i64>>
    %%clock_id = llvm.mlir.constant(1 : i32) : i32  // CLOCK_MONOTONIC
    %%result = llvm.call @clock_gettime(%%clock_id, %%timespec) : (i32, !llvm.ptr<i8>) -> i32
    %%sec_ptr = llvm.getelementptr %%timespec[0, 0] : (!llvm.ptr<!llvm.array<2 x i64>>, i32, i32) -> !llvm.ptr<i64>
    %%nsec_ptr = llvm.getelementptr %%timespec[0, 1] : (!llvm.ptr<!llvm.array<2 x i64>>, i32, i32) -> !llvm.ptr<i64>
    %%sec = llvm.load %%sec_ptr : !llvm.ptr<i64> -> i64
    %%nsec = llvm.load %%nsec_ptr : !llvm.ptr<i64> -> i64
    %%billion = llvm.mlir.constant(1000000000 : i64) : i64
    %%sec_as_nsec = llvm.mul %%sec, %%billion : i64
    %s = llvm.add %%sec_as_nsec, %%nsec : i64""" resultVar
        
    | Portable ->
        // Fallback: Just return a counter
        sprintf """
    // Portable fallback - incrementing counter
    %%counter_addr = llvm.mlir.addressof @time_counter : !llvm.ptr<i64>
    %%current = llvm.load %%counter_addr : !llvm.ptr<i64> -> i64
    %%one = llvm.mlir.constant(1 : i64) : i64
    %%next = llvm.add %%current, %%one : i64
    llvm.store %%next, %%counter_addr : i64, !llvm.ptr<i64>
    %s = llvm.add %%current, %%zero : i64  // Copy""" resultVar

/// Generate LLVM dialect for Time.tickFrequency()
let emitTickFrequency (platform: TimePlatform) (resultVar: string) =
    match platform with
    | Windows ->
        sprintf """
    %%freq_ptr = llvm.alloca i64 : (i64) -> !llvm.ptr<i64>
    %%qpf_result = llvm.call @QueryPerformanceFrequency(%%freq_ptr) : (!llvm.ptr<i64>) -> i32
    %s = llvm.load %%freq_ptr : !llvm.ptr<i64> -> i64""" resultVar
        
    | Linux | MacOS ->
        // POSIX clock_gettime uses nanoseconds
        sprintf """
    %s = llvm.mlir.constant(1000000000 : i64) : i64  // 1 billion nanoseconds per second""" resultVar
        
    | Portable ->
        sprintf """
    %s = llvm.mlir.constant(1000 : i64) : i64  // Arbitrary frequency for portable mode""" resultVar

/// Generate LLVM dialect for Time.sleep(milliseconds)
let emitSleep (platform: TimePlatform) (msVar: string) =
    match platform with
    | Windows ->
        sprintf """
    llvm.call @Sleep(%s) : (i32) -> !llvm.void""" msVar
        
    | Linux | MacOS ->
        sprintf """
    // Convert milliseconds to timespec
    %%ms_as_i64 = llvm.sext %s : i32 to i64
    %%thousand = llvm.mlir.constant(1000 : i64) : i64
    %%million = llvm.mlir.constant(1000000 : i64) : i64
    %%sec = llvm.sdiv %%ms_as_i64, %%thousand : i64
    %%ms_remainder = llvm.srem %%ms_as_i64, %%thousand : i64
    %%nsec = llvm.mul %%ms_remainder, %%million : i64
    
    %%timespec = llvm.alloca !llvm.array<2 x i64> : (i32) -> !llvm.ptr<!llvm.array<2 x i64>>
    %%sec_ptr = llvm.getelementptr %%timespec[0, 0] : (!llvm.ptr<!llvm.array<2 x i64>>, i32, i32) -> !llvm.ptr<i64>
    %%nsec_ptr = llvm.getelementptr %%timespec[0, 1] : (!llvm.ptr<!llvm.array<2 x i64>>, i32, i32) -> !llvm.ptr<i64>
    llvm.store %%sec, %%sec_ptr : i64, !llvm.ptr<i64>
    llvm.store %%nsec, %%nsec_ptr : i64, !llvm.ptr<i64>
    
    %%null = llvm.mlir.null : !llvm.ptr<i8>
    %%result = llvm.call @nanosleep(%%timespec, %%null) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32""" msVar
        
    | Portable ->
        sprintf """
    // Portable: Busy-wait (not ideal but works everywhere)
    %%target = llvm.add %%current_ticks, %s : i64
    llvm.br ^busy_wait
    ^busy_wait:
        %%now = llvm.call @get_ticks() : () -> i64
        %%done = llvm.icmp "sge" %%now, %%target : i64
        llvm.cond_br %%done, ^done, ^busy_wait
    ^done:""" msVar

/// Generate LLVM dialect for Time.now() - returns DateTime struct
let emitNow (platform: TimePlatform) (resultVar: string) =
    match platform with
    | Windows ->
        sprintf """
    // Get Windows FILETIME (100-nanosecond intervals since 1601)
    %%filetime = llvm.alloca i64 : (i64) -> !llvm.ptr<i64>
    llvm.call @GetSystemTimeAsFileTime(%%filetime) : (!llvm.ptr<i64>) -> !llvm.void
    %%ticks = llvm.load %%filetime : !llvm.ptr<i64> -> i64
    
    // Convert to Unix epoch and extract components
    %%epoch_diff = llvm.mlir.constant(116444736000000000 : i64) : i64
    %%unix_ticks = llvm.sub %%ticks, %%epoch_diff : i64
    
    // Extract date/time components (simplified - real impl needs more arithmetic)
    %%ticks_per_sec = llvm.mlir.constant(10000000 : i64) : i64
    %%seconds = llvm.sdiv %%unix_ticks, %%ticks_per_sec : i64
    
    // Pack into DateTime struct
    %s = llvm.mlir.undef : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)>
    // ... insert components ...""" resultVar
        
    | Linux | MacOS ->
        sprintf """
    // Get current time using clock_gettime with CLOCK_REALTIME
    %%timespec = llvm.alloca !llvm.array<2 x i64> : (i32) -> !llvm.ptr<!llvm.array<2 x i64>>
    %%clock_id = llvm.mlir.constant(0 : i32) : i32  // CLOCK_REALTIME
    %%result = llvm.call @clock_gettime(%%clock_id, %%timespec) : (i32, !llvm.ptr<i8>) -> i32
    
    // Extract seconds since Unix epoch
    %%sec_ptr = llvm.getelementptr %%timespec[0, 0] : (!llvm.ptr<!llvm.array<2 x i64>>, i32, i32) -> !llvm.ptr<i64>
    %%seconds = llvm.load %%sec_ptr : !llvm.ptr<i64> -> i64
    
    // Convert to DateTime components (simplified)
    %s = llvm.mlir.undef : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)>
    // ... convert Unix timestamp to date/time ...""" resultVar
        
    | Portable ->
        sprintf """
    // Portable: Return a fixed date/time
    %s = llvm.mlir.constant(dense<[2024, 1, 1, 0, 0, 0, 0]> : tensor<7xi32>) : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)>""" resultVar

/// Map Time library calls to LLVM dialect
let mapTimeCall (node: PSGNode) (platform: TimePlatform) =
    match node.Symbol with
    | Some sym when sym.FullName.Contains("highResolutionTicks") ->
        emitHighResolutionTicks platform (sprintf "%%ticks_%d" node.Id)
    | Some sym when sym.FullName.Contains("tickFrequency") ->
        emitTickFrequency platform (sprintf "%%freq_%d" node.Id)
    | Some sym when sym.FullName.Contains("sleep") ->
        // Extract milliseconds argument from node
        emitSleep platform (sprintf "%%ms_%d" node.Id)
    | Some sym when sym.FullName.Contains("now") ->
        emitNow platform (sprintf "%%datetime_%d" node.Id)
    | _ -> ""