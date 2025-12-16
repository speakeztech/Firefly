/// ProcessBindings - Platform-specific MLIR generation for process operations (exit, etc.)
module Alex.Bindings.Process.ProcessBindings

open Alex.CodeGeneration.MLIRBuilder
open Alex.Bindings.BindingTypes

// ===================================================================
// Platform Data: Syscall numbers
// ===================================================================

module SyscallData =
    /// Linux x86-64 syscall numbers
    let linuxSyscalls = Map [
        "exit", 60L
        "exit_group", 231L
    ]

    /// macOS syscall numbers (with BSD 0x2000000 offset for x86-64)
    let macosSyscalls = Map [
        "exit", 0x2000001L
    ]

    let getSyscallNumber (os: OSFamily) (name: string) : int64 option =
        match os with
        | Linux -> Map.tryFind name linuxSyscalls
        | MacOS -> Map.tryFind name macosSyscalls
        | _ -> None

// ===================================================================
// MLIR Generation for Process Primitives
// ===================================================================

/// Emit exit syscall for Unix-like systems (Linux, macOS)
let emitUnixExitSyscall (syscallNum: int64) (exitCode: Val) : MLIR<unit> = mlir {
    // Extend exit code to i64 if needed
    let! codeExt =
        match exitCode.Type with
        | Integer I32 -> arith.extsi exitCode I64
        | _ -> mlir { return exitCode }

    // Syscall number
    let! sysNum = arith.constant syscallNum I64

    // Execute syscall: exit(code)
    // rax = syscall number, rdi = exit code
    let! _ = llvm.inlineAsm "syscall" "=r,{rax},{rdi},~{rcx},~{r11},~{memory}"
                [{ SSA = sysNum.SSA; Type = Integer I64 }
                 codeExt] (Integer I64)

    // Mark as unreachable (exit never returns)
    do! llvm.unreachable

    return ()
}

// ===================================================================
// Platform Primitive Bindings
// ===================================================================

/// exit - terminate process with exit code
/// Bound from Alloy.Platform.Bindings.exit
let bindExit (platform: TargetPlatform) (prim: PlatformPrimitive) : MLIR<EmissionResult> = mlir {
    match prim.Args with
    | [exitCode] ->
        match platform.OS with
        | Linux ->
            do! emitUnixExitSyscall 60L exitCode
            return EmittedVoid
        | MacOS ->
            do! emitUnixExitSyscall 0x2000001L exitCode
            return EmittedVoid
        | Windows ->
            // TODO: Windows uses ExitProcess API call
            return NotSupported "Windows exit not yet implemented"
        | _ ->
            return NotSupported $"Exit not supported on {platform.OS}"
    | _ ->
        return NotSupported "exit requires (exitCode) argument"
}

// ===================================================================
// Registration
// ===================================================================

/// Register all process bindings for all platforms
/// Entry points match Platform.Bindings function names
let registerBindings () =
    // Register for Linux
    PlatformDispatch.register Linux X86_64 "exit"
        (fun ext -> bindExit TargetPlatform.linux_x86_64 ext)
    PlatformDispatch.register Linux ARM64 "exit"
        (fun ext -> bindExit { TargetPlatform.linux_x86_64 with Arch = ARM64 } ext)

    // Register for macOS
    PlatformDispatch.register MacOS X86_64 "exit"
        (fun ext -> bindExit TargetPlatform.macos_x86_64 ext)
    PlatformDispatch.register MacOS ARM64 "exit"
        (fun ext -> bindExit TargetPlatform.macos_arm64 ext)
