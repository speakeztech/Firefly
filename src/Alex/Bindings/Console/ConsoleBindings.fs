module Alex.Bindings.Console.ConsoleBindings

open Alex.CodeGeneration.MLIRBuilder
open Alex.Bindings.BindingTypes

// ===================================================================
// Platform Data: Syscall numbers and conventions
// ===================================================================

module SyscallData =
    /// Linux x86-64 syscall numbers
    let linuxSyscalls = Map [
        "read", 0L
        "write", 1L
    ]

    /// macOS syscall numbers (with BSD 0x2000000 offset for x86-64)
    let macosSyscalls = Map [
        "read", 0x2000003L
        "write", 0x2000004L
    ]

    let getSyscallNumber (os: OSFamily) (name: string) : int64 option =
        match os with
        | Linux -> Map.tryFind name linuxSyscalls
        | MacOS -> Map.tryFind name macosSyscalls
        | _ -> None

// ===================================================================
// MLIR Generation for Console Primitives
// ===================================================================

/// Emit write syscall for Unix-like systems (Linux, macOS)
let emitUnixWriteSyscall (syscallNum: int64) (fd: Val) (buf: Val) (count: Val) : MLIR<Val> = mlir {
    // Extend fd from i32 to i64 if needed
    let! fdExt =
        match fd.Type with
        | Integer I32 -> arith.extsi fd I64
        | _ -> mlir { return fd }

    // Extend count to i64 if needed
    let! countExt =
        match count.Type with
        | Integer I32 -> arith.extsi count I64
        | _ -> mlir { return count }

    // Syscall number
    let! sysNum = arith.constant syscallNum I64

    // Execute syscall: write(fd, buf, count)
    // rax = syscall number, rdi = fd, rsi = buf, rdx = count
    let! result = llvm.inlineAsm "syscall" "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}"
                    [{ SSA = sysNum.SSA; Type = Integer I64 }
                     fdExt
                     buf
                     countExt] (Integer I64)

    return result
}

/// Emit read syscall for Unix-like systems (Linux, macOS)
let emitUnixReadSyscall (syscallNum: int64) (fd: Val) (buf: Val) (count: Val) : MLIR<Val> = mlir {
    // Extend fd from i32 to i64 if needed
    let! fdExt =
        match fd.Type with
        | Integer I32 -> arith.extsi fd I64
        | _ -> mlir { return fd }

    // Extend count to i64 if needed
    let! countExt =
        match count.Type with
        | Integer I32 -> arith.extsi count I64
        | _ -> mlir { return count }

    // Syscall number
    let! sysNum = arith.constant syscallNum I64

    // Execute syscall: read(fd, buf, count)
    let! result = llvm.inlineAsm "syscall" "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}"
                    [{ SSA = sysNum.SSA; Type = Integer I64 }
                     fdExt
                     buf
                     countExt] (Integer I64)

    return result
}

// ===================================================================
// Platform Bindings (BCL-Free Pattern)
// ===================================================================

/// writeBytes - write bytes to file descriptor
/// Bound from Alloy.Platform.Bindings.writeBytes
let bindWriteBytes (platform: TargetPlatform) (prim: ExternPrimitive) : MLIR<EmissionResult> = mlir {
    match prim.Args with
    | [fd; buf; count] ->
        match platform.OS with
        | Linux ->
            let! result = emitUnixWriteSyscall 1L fd buf count
            let! truncated = arith.trunci result I32
            return Emitted truncated
        | MacOS ->
            let! result = emitUnixWriteSyscall 0x2000004L fd buf count
            let! truncated = arith.trunci result I32
            return Emitted truncated
        | Windows ->
            // TODO: Windows uses WriteFile API call
            return NotSupported "Windows console not yet implemented"
        | _ ->
            return NotSupported $"Console not supported on {platform.OS}"
    | _ ->
        return NotSupported "writeBytes requires (fd, buffer, count) arguments"
}

/// readBytes - read bytes from file descriptor
/// Bound from Alloy.Platform.Bindings.readBytes
let bindReadBytes (platform: TargetPlatform) (prim: ExternPrimitive) : MLIR<EmissionResult> = mlir {
    match prim.Args with
    | [fd; buf; maxCount] ->
        match platform.OS with
        | Linux ->
            let! result = emitUnixReadSyscall 0L fd buf maxCount
            let! truncated = arith.trunci result I32
            return Emitted truncated
        | MacOS ->
            let! result = emitUnixReadSyscall 0x2000003L fd buf maxCount
            let! truncated = arith.trunci result I32
            return Emitted truncated
        | Windows ->
            return NotSupported "Windows console not yet implemented"
        | _ ->
            return NotSupported $"Console not supported on {platform.OS}"
    | _ ->
        return NotSupported "readBytes requires (fd, buffer, maxCount) arguments"
}

// ===================================================================
// Registration
// ===================================================================

/// Register all console bindings for all platforms
/// Entry points match Platform.Bindings function names (e.g., "writeBytes", "readBytes")
let registerBindings () =
    // Register for Linux
    ExternDispatch.register Linux X86_64 "writeBytes"
        (fun ext -> bindWriteBytes TargetPlatform.linux_x86_64 ext)
    ExternDispatch.register Linux X86_64 "readBytes"
        (fun ext -> bindReadBytes TargetPlatform.linux_x86_64 ext)
    ExternDispatch.register Linux ARM64 "writeBytes"
        (fun ext -> bindWriteBytes { TargetPlatform.linux_x86_64 with Arch = ARM64 } ext)
    ExternDispatch.register Linux ARM64 "readBytes"
        (fun ext -> bindReadBytes { TargetPlatform.linux_x86_64 with Arch = ARM64 } ext)

    // Register for macOS
    ExternDispatch.register MacOS X86_64 "writeBytes"
        (fun ext -> bindWriteBytes TargetPlatform.macos_x86_64 ext)
    ExternDispatch.register MacOS X86_64 "readBytes"
        (fun ext -> bindReadBytes TargetPlatform.macos_x86_64 ext)
    ExternDispatch.register MacOS ARM64 "writeBytes"
        (fun ext -> bindWriteBytes TargetPlatform.macos_arm64 ext)
    ExternDispatch.register MacOS ARM64 "readBytes"
        (fun ext -> bindReadBytes TargetPlatform.macos_arm64 ext)
