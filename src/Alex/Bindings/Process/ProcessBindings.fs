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
// Extern Primitive Bindings
// ===================================================================

/// fidelity_exit - terminate process with exit code
let bindExit (platform: TargetPlatform) (prim: ExternPrimitive) : MLIR<EmissionResult> = mlir {
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
let registerBindings () =
    // Register for Linux
    ExternDispatch.register Linux X86_64 "fidelity_exit"
        (fun ext -> bindExit TargetPlatform.linux_x86_64 ext)
    ExternDispatch.register Linux ARM64 "fidelity_exit"
        (fun ext -> bindExit { TargetPlatform.linux_x86_64 with Arch = ARM64 } ext)

    // Register for macOS
    ExternDispatch.register MacOS X86_64 "fidelity_exit"
        (fun ext -> bindExit TargetPlatform.macos_x86_64 ext)
    ExternDispatch.register MacOS ARM64 "fidelity_exit"
        (fun ext -> bindExit TargetPlatform.macos_arm64 ext)
