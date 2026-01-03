/// ProcessBindings - Platform-specific process bindings (witness-based)
///
/// ARCHITECTURAL FOUNDATION (December 2025):
/// Uses the codata accumulator pattern from MLIRZipper.
/// Bindings are witness functions that take primitive info and zipper,
/// returning an updated zipper with the witnessed MLIR operations.
module Alex.Bindings.Process.ProcessBindings

open Alex.CodeGeneration.MLIRTypes
open Alex.Traversal.MLIRZipper
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
// Helper Witness Functions
// ===================================================================

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
// MLIR Generation for Process Primitives (Witness-Based)
// ===================================================================

/// Witness exit syscall for Unix-like systems (Linux, macOS)
let witnessUnixExitSyscall (syscallNum: int64) (exitCodeSSA: string) (exitCodeType: MLIRType) (zipper: MLIRZipper) : MLIRZipper =
    // Extend exit code to i64 if needed
    let codeExtSSA, zipper1 = witnessExtSIIfNeeded exitCodeSSA exitCodeType I64 zipper

    // Syscall number
    let sysNumSSA, zipper2 = MLIRZipper.witnessConstant syscallNum I64 zipper1

    // Execute syscall: exit(code)
    // rax = syscall number, rdi = exit code
    let args = [(codeExtSSA, "i64")]
    let _resultSSA, zipper3 = MLIRZipper.witnessSyscall sysNumSSA args (Integer I64) zipper2

    // Mark as unreachable (exit never returns)
    MLIRZipper.witnessUnreachable zipper3

// ===================================================================
// Platform Primitive Bindings (Witness-Based)
// ===================================================================

/// exit - terminate process with exit code
/// Witness binding from Alloy.Platform.Bindings.exit
let witnessExit (platform: TargetPlatform) (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(exitCodeSSA, exitCodeType)] ->
        match platform.OS with
        | Linux ->
            let zipper1 = witnessUnixExitSyscall 60L exitCodeSSA exitCodeType zipper
            zipper1, WitnessedVoid
        | MacOS ->
            let zipper1 = witnessUnixExitSyscall 0x2000001L exitCodeSSA exitCodeType zipper
            zipper1, WitnessedVoid
        | Windows ->
            zipper, NotSupported "Windows exit not yet implemented"
        | _ ->
            zipper, NotSupported (sprintf "Exit not supported on %A" platform.OS)
    | _ ->
        zipper, NotSupported "exit requires (exitCode) argument"

// ===================================================================
// Registration (Witness-Based)
// ===================================================================

/// Register all process bindings for all platforms
/// Entry points match Platform.Bindings function names AND FNCS Sys intrinsics
let registerBindings () =
    // Register for Linux x86_64
    PlatformDispatch.register Linux X86_64 "exit"
        (fun prim zipper -> witnessExit TargetPlatform.linux_x86_64 prim zipper)
    // FNCS Sys intrinsic - same implementation as exit
    PlatformDispatch.register Linux X86_64 "Sys.exit"
        (fun prim zipper -> witnessExit TargetPlatform.linux_x86_64 prim zipper)

    // Register for Linux ARM64
    PlatformDispatch.register Linux ARM64 "exit"
        (fun prim zipper -> witnessExit { TargetPlatform.linux_x86_64 with Arch = ARM64 } prim zipper)
    PlatformDispatch.register Linux ARM64 "Sys.exit"
        (fun prim zipper -> witnessExit { TargetPlatform.linux_x86_64 with Arch = ARM64 } prim zipper)

    // Register for macOS x86_64
    PlatformDispatch.register MacOS X86_64 "exit"
        (fun prim zipper -> witnessExit TargetPlatform.macos_x86_64 prim zipper)
    PlatformDispatch.register MacOS X86_64 "Sys.exit"
        (fun prim zipper -> witnessExit TargetPlatform.macos_x86_64 prim zipper)

    // Register for macOS ARM64
    PlatformDispatch.register MacOS ARM64 "exit"
        (fun prim zipper -> witnessExit TargetPlatform.macos_arm64 prim zipper)
    PlatformDispatch.register MacOS ARM64 "Sys.exit"
        (fun prim zipper -> witnessExit TargetPlatform.macos_arm64 prim zipper)
