/// ConsoleBindings - Platform-specific console I/O bindings (witness-based)
///
/// ARCHITECTURAL FOUNDATION (December 2025):
/// Uses the codata accumulator pattern from MLIRZipper.
/// Bindings are witness functions that take primitive info and zipper,
/// returning an updated zipper with the witnessed MLIR operations.
module Alex.Bindings.Console.ConsoleBindings

open Alex.CodeGeneration.MLIRTypes
open Alex.Traversal.MLIRZipper
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
// Helper Witness Functions
// ===================================================================

/// Witness sign extension (extsi) if needed
let witnessExtSIIfNeeded (ssaName: string) (fromType: MLIRType) (toWidth: IntegerBitWidth) (zipper: MLIRZipper) : string * MLIRZipper =
    match fromType with
    | Integer fromWidth when fromWidth <> toWidth ->
        let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
        let fromStr = Serialize.integerBitWidth fromWidth
        let toStr = Serialize.integerBitWidth toWidth
        let text = sprintf "%s = arith.extsi %s : %s to %s" resultSSA ssaName fromStr toStr
        resultSSA, MLIRZipper.witnessOpWithResult text resultSSA (Integer toWidth) zipper'
    | _ ->
        // No extension needed
        ssaName, zipper

/// Witness truncation (trunci)
let witnessTruncI (ssaName: string) (fromType: MLIRType) (toWidth: IntegerBitWidth) (zipper: MLIRZipper) : string * MLIRZipper =
    let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
    let fromStr = Serialize.mlirType fromType
    let toStr = Serialize.integerBitWidth toWidth
    let text = sprintf "%s = arith.trunci %s : %s to %s" resultSSA ssaName fromStr toStr
    resultSSA, MLIRZipper.witnessOpWithResult text resultSSA (Integer toWidth) zipper'

// ===================================================================
// MLIR Generation for Console Primitives (Witness-Based)
// ===================================================================

/// Witness write syscall for Unix-like systems (Linux, macOS)
let witnessUnixWriteSyscall (syscallNum: int64) (fd: string) (fdType: MLIRType) (buf: string) (bufType: MLIRType) (count: string) (countType: MLIRType) (zipper: MLIRZipper) : string * MLIRZipper =
    // Extend fd to i64 if needed
    let fdExt, zipper1 = witnessExtSIIfNeeded fd fdType I64 zipper

    // Extend count to i64 if needed
    let countExt, zipper2 = witnessExtSIIfNeeded count countType I64 zipper1

    // Witness syscall number constant
    let sysNumSSA, zipper3 = MLIRZipper.witnessConstant syscallNum I64 zipper2

    // Witness syscall: write(fd, buf, count)
    // rax = syscall number, rdi = fd, rsi = buf, rdx = count
    // Buffer type can be !llvm.ptr or i64 (after ptrtoint conversion)
    let bufTypeStr = Serialize.mlirType bufType
    let args = [
        (fdExt, "i64")
        (buf, bufTypeStr)
        (countExt, "i64")
    ]
    MLIRZipper.witnessSyscall sysNumSSA args (Integer I64) zipper3

/// Witness read syscall for Unix-like systems (Linux, macOS)
let witnessUnixReadSyscall (syscallNum: int64) (fd: string) (fdType: MLIRType) (buf: string) (bufType: MLIRType) (maxCount: string) (countType: MLIRType) (zipper: MLIRZipper) : string * MLIRZipper =
    // Extend fd to i64 if needed
    let fdExt, zipper1 = witnessExtSIIfNeeded fd fdType I64 zipper

    // Extend count to i64 if needed
    let countExt, zipper2 = witnessExtSIIfNeeded maxCount countType I64 zipper1

    // Witness syscall number constant
    let sysNumSSA, zipper3 = MLIRZipper.witnessConstant syscallNum I64 zipper2

    // Witness syscall: read(fd, buf, count)
    // Buffer type can be !llvm.ptr or i64 (after ptrtoint conversion)
    let bufTypeStr = Serialize.mlirType bufType
    let args = [
        (fdExt, "i64")
        (buf, bufTypeStr)
        (countExt, "i64")
    ]
    MLIRZipper.witnessSyscall sysNumSSA args (Integer I64) zipper3

// ===================================================================
// Platform Bindings (Witness-Based Pattern)
// ===================================================================

/// writeBytes - write bytes to file descriptor
/// Witness binding from Alloy.Platform.Bindings.writeBytes
let witnessWriteBytes (platform: TargetPlatform) (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(fd, fdType); (buf, bufType); (count, countType)] ->
        match platform.OS with
        | Linux ->
            let resultSSA, zipper1 = witnessUnixWriteSyscall 1L fd fdType buf bufType count countType zipper
            let truncSSA, zipper2 = witnessTruncI resultSSA (Integer I64) I32 zipper1
            zipper2, WitnessedValue (truncSSA, Integer I32)
        | MacOS ->
            let resultSSA, zipper1 = witnessUnixWriteSyscall 0x2000004L fd fdType buf bufType count countType zipper
            let truncSSA, zipper2 = witnessTruncI resultSSA (Integer I64) I32 zipper1
            zipper2, WitnessedValue (truncSSA, Integer I32)
        | Windows ->
            zipper, NotSupported "Windows console not yet implemented"
        | _ ->
            zipper, NotSupported (sprintf "Console not supported on %A" platform.OS)
    | _ ->
        zipper, NotSupported "writeBytes requires (fd, buffer, count) arguments"

/// readBytes - read bytes from file descriptor
/// Witness binding from Alloy.Platform.Bindings.readBytes
let witnessReadBytes (platform: TargetPlatform) (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
    match prim.Args with
    | [(fd, fdType); (buf, bufType); (maxCount, countType)] ->
        match platform.OS with
        | Linux ->
            let resultSSA, zipper1 = witnessUnixReadSyscall 0L fd fdType buf bufType maxCount countType zipper
            let truncSSA, zipper2 = witnessTruncI resultSSA (Integer I64) I32 zipper1
            zipper2, WitnessedValue (truncSSA, Integer I32)
        | MacOS ->
            let resultSSA, zipper1 = witnessUnixReadSyscall 0x2000003L fd fdType buf bufType maxCount countType zipper
            let truncSSA, zipper2 = witnessTruncI resultSSA (Integer I64) I32 zipper1
            zipper2, WitnessedValue (truncSSA, Integer I32)
        | Windows ->
            zipper, NotSupported "Windows console not yet implemented"
        | _ ->
            zipper, NotSupported (sprintf "Console not supported on %A" platform.OS)
    | _ ->
        zipper, NotSupported "readBytes requires (fd, buffer, maxCount) arguments"

// ===================================================================
// Registration (Witness-Based)
// ===================================================================

/// Register all console bindings for all platforms
/// Entry points match Platform.Bindings function names AND FNCS Sys intrinsics
let registerBindings () =
    // Register for Linux x86_64
    PlatformDispatch.register Linux X86_64 "writeBytes"
        (fun prim zipper -> witnessWriteBytes TargetPlatform.linux_x86_64 prim zipper)
    PlatformDispatch.register Linux X86_64 "readBytes"
        (fun prim zipper -> witnessReadBytes TargetPlatform.linux_x86_64 prim zipper)
    // FNCS Sys intrinsics - same implementation as writeBytes/readBytes
    PlatformDispatch.register Linux X86_64 "Sys.write"
        (fun prim zipper -> witnessWriteBytes TargetPlatform.linux_x86_64 prim zipper)
    PlatformDispatch.register Linux X86_64 "Sys.read"
        (fun prim zipper -> witnessReadBytes TargetPlatform.linux_x86_64 prim zipper)

    // Register for Linux ARM64
    PlatformDispatch.register Linux ARM64 "writeBytes"
        (fun prim zipper -> witnessWriteBytes { TargetPlatform.linux_x86_64 with Arch = ARM64 } prim zipper)
    PlatformDispatch.register Linux ARM64 "readBytes"
        (fun prim zipper -> witnessReadBytes { TargetPlatform.linux_x86_64 with Arch = ARM64 } prim zipper)
    PlatformDispatch.register Linux ARM64 "Sys.write"
        (fun prim zipper -> witnessWriteBytes { TargetPlatform.linux_x86_64 with Arch = ARM64 } prim zipper)
    PlatformDispatch.register Linux ARM64 "Sys.read"
        (fun prim zipper -> witnessReadBytes { TargetPlatform.linux_x86_64 with Arch = ARM64 } prim zipper)

    // Register for macOS x86_64
    PlatformDispatch.register MacOS X86_64 "writeBytes"
        (fun prim zipper -> witnessWriteBytes TargetPlatform.macos_x86_64 prim zipper)
    PlatformDispatch.register MacOS X86_64 "readBytes"
        (fun prim zipper -> witnessReadBytes TargetPlatform.macos_x86_64 prim zipper)
    PlatformDispatch.register MacOS X86_64 "Sys.write"
        (fun prim zipper -> witnessWriteBytes TargetPlatform.macos_x86_64 prim zipper)
    PlatformDispatch.register MacOS X86_64 "Sys.read"
        (fun prim zipper -> witnessReadBytes TargetPlatform.macos_x86_64 prim zipper)

    // Register for macOS ARM64
    PlatformDispatch.register MacOS ARM64 "writeBytes"
        (fun prim zipper -> witnessWriteBytes TargetPlatform.macos_arm64 prim zipper)
    PlatformDispatch.register MacOS ARM64 "readBytes"
        (fun prim zipper -> witnessReadBytes TargetPlatform.macos_arm64 prim zipper)
    PlatformDispatch.register MacOS ARM64 "Sys.write"
        (fun prim zipper -> witnessWriteBytes TargetPlatform.macos_arm64 prim zipper)
    PlatformDispatch.register MacOS ARM64 "Sys.read"
        (fun prim zipper -> witnessReadBytes TargetPlatform.macos_arm64 prim zipper)
