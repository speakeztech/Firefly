/// String primitive bindings for Fidelity native compilation
/// Provides platform-specific implementations for string operations
module Alex.Bindings.String.StringBindings

open Alex.Bindings.Types
open Alex.CodeGeneration.MLIRBuilder

/// Emit inline strlen loop - counts bytes until null terminator
/// This is a pure computation, no syscalls needed
let private emitInlineStrlen (strPtr: SSAValue) : MLIR<SSAValue> = mlir {
    // strlen algorithm:
    // 1. Initialize counter to 0
    // 2. Loop: load byte, if zero exit, else increment counter and continue
    // 3. Return counter

    let! zero = arith.constant 0L I64
    let! one = arith.constant 1L I64

    // Create loop structure with scf.while
    // Initial value for counter
    let! initCounter = arith.constant 0L I64

    // We'll use a simple loop: while (*ptr != 0) { ptr++; len++; }
    // Entry block sets up, then branches to condition check

    // For simplicity, emit a scf.while loop
    // before: yield condition
    // after: yield next values

    // Use LLVM-style loop with blocks
    let! loopEntry = newLabel "strlen_entry"
    let! loopBody = newLabel "strlen_body"
    let! loopExit = newLabel "strlen_exit"

    // Entry: initialize
    do! emitLabel loopEntry
    let! ptrInit = llvm.bitcast strPtr LLVMPtr
    do! emitBranch loopBody

    // Body: check and increment
    do! emitLabel loopBody
    // PHI for pointer and counter would go here
    // For now, use a simpler approach with llvm loop intrinsics

    // Actually, let's just emit a call to our own inline asm strlen
    // This is cleaner than trying to build the loop in MLIR

    // Simple approach: emit inline assembly for strlen
    let! result = llvm.inlineAsm
        "xor %rcx, %rcx\n.Lloop%=: cmpb $$0, (%rdi,%rcx)\n je .Ldone%=\n inc %rcx\n jmp .Lloop%=\n.Ldone%=:"
        "={rcx},{rdi}"
        [strPtr]
        I64
        true  // has side effects (reads memory)

    return result
}

/// fidelity_strlen - get length of null-terminated string
let bindStrlen (platform: TargetPlatform) (prim: ExternPrimitive) : MLIR<EmissionResult> = mlir {
    match prim.Args with
    | [strPtr] ->
        // All platforms use the same inline implementation
        let! result = emitInlineStrlen strPtr
        let! truncated = arith.trunci result I32
        return Emitted truncated
    | _ ->
        return NotSupported "strlen requires (str) argument"
}

/// Register string bindings with the extern dispatch
let registerBindings () =
    // Register for Linux
    ExternDispatch.register Linux X86_64 "fidelity_strlen"
        (fun ext -> bindStrlen TargetPlatform.linux_x86_64 ext)
    ExternDispatch.register Linux ARM64 "fidelity_strlen"
        (fun ext -> bindStrlen { TargetPlatform.linux_x86_64 with Arch = ARM64 } ext)

    // Register for macOS
    ExternDispatch.register MacOS X86_64 "fidelity_strlen"
        (fun ext -> bindStrlen TargetPlatform.macos_x86_64 ext)
    ExternDispatch.register MacOS ARM64 "fidelity_strlen"
        (fun ext -> bindStrlen TargetPlatform.macos_arm64 ext)
