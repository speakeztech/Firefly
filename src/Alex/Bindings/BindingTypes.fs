/// BindingTypes - Platform binding types for witness-based MLIR generation
///
/// ARCHITECTURAL FOUNDATION (December 2025):
/// This module uses the codata accumulator pattern from MLIRZipper.
/// Bindings are witness functions that take primitive info and zipper,
/// returning an updated zipper with the witnessed MLIR operations.
///
/// Key vocabulary (from Coeffects & Codata):
/// - **witness** - Record observation of computation
/// - **observe** - Note context requirement (coeffect)
/// - **yield** - Produce on demand (codata production)
module Alex.Bindings.BindingTypes

open Alex.CodeGeneration.MLIRTypes
open Alex.Traversal.MLIRZipper

// ===================================================================
// Target Platform Identification
// ===================================================================

/// Operating system family
type OSFamily =
    | Linux
    | Windows
    | MacOS
    | FreeBSD
    | BareMetal
    | WASM

/// Processor architecture
type Architecture =
    | X86_64
    | ARM64
    | ARM32_Thumb
    | RISCV64
    | RISCV32
    | WASM32

/// Complete platform identification
type TargetPlatform = {
    OS: OSFamily
    Arch: Architecture
    Triple: string
    Features: Set<string>  // e.g., "avx2", "neon", "thumb"
}

module TargetPlatform =
    /// Parse LLVM target triple to TargetPlatform
    let parseTriple (triple: string) : TargetPlatform option =
        let parts = triple.ToLowerInvariant().Split('-')
        let archOsOpt =
            match parts with
            | [| arch; _vendor; os |] -> Some (arch, os)
            | [| arch; _vendor; os; _env |] -> Some (arch, os)
            | _ -> None
        match archOsOpt with
        | None -> None
        | Some (arch, os) ->
            let architecture =
                match arch with
                | "x86_64" | "amd64" -> Some X86_64
                | "aarch64" | "arm64" -> Some ARM64
                | a when a.StartsWith("armv7") || a.StartsWith("thumb") -> Some ARM32_Thumb
                | "riscv64" -> Some RISCV64
                | "riscv32" -> Some RISCV32
                | "wasm32" -> Some WASM32
                | _ -> None

            let osFamily =
                match os with
                | o when o.StartsWith("linux") -> Some Linux
                | o when o.StartsWith("windows") -> Some Windows
                | o when o.StartsWith("darwin") || o.StartsWith("macos") -> Some MacOS
                | o when o.StartsWith("freebsd") -> Some FreeBSD
                | "none" | "unknown" when arch.StartsWith("thumb") -> Some BareMetal
                | "unknown" when arch = "wasm32" -> Some WASM
                | _ -> None

            match architecture, osFamily with
            | Some arch, Some os ->
                Some { OS = os; Arch = arch; Triple = triple; Features = Set.empty }
            | _ -> None

    /// Default platforms for quick reference
    let linux_x86_64 = { OS = Linux; Arch = X86_64; Triple = "x86_64-unknown-linux-gnu"; Features = Set.empty }
    let windows_x86_64 = { OS = Windows; Arch = X86_64; Triple = "x86_64-pc-windows-msvc"; Features = Set.empty }
    let macos_x86_64 = { OS = MacOS; Arch = X86_64; Triple = "x86_64-apple-darwin"; Features = Set.empty }
    let macos_arm64 = { OS = MacOS; Arch = ARM64; Triple = "aarch64-apple-darwin"; Features = Set.empty }

    /// Detect host platform at runtime
    let detectHost () : TargetPlatform =
        let os =
            if System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
                System.Runtime.InteropServices.OSPlatform.Linux) then Linux
            elif System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
                System.Runtime.InteropServices.OSPlatform.Windows) then Windows
            elif System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
                System.Runtime.InteropServices.OSPlatform.OSX) then MacOS
            else failwith "detectCurrentPlatform: Unsupported OS - not Linux, Windows, or MacOS"

        let arch =
            match System.Runtime.InteropServices.RuntimeInformation.OSArchitecture with
            | System.Runtime.InteropServices.Architecture.X64 -> X86_64
            | System.Runtime.InteropServices.Architecture.Arm64 -> ARM64
            | System.Runtime.InteropServices.Architecture.Arm -> ARM32_Thumb
            | other -> failwithf "detectCurrentPlatform: Unsupported architecture %A" other

        let triple =
            match os, arch with
            | Linux, X86_64 -> "x86_64-unknown-linux-gnu"
            | Linux, ARM64 -> "aarch64-unknown-linux-gnu"
            | Windows, X86_64 -> "x86_64-pc-windows-msvc"
            | MacOS, X86_64 -> "x86_64-apple-darwin"
            | MacOS, ARM64 -> "aarch64-apple-darwin"
            | _ -> "x86_64-unknown-linux-gnu"

        { OS = os; Arch = arch; Triple = triple; Features = Set.empty }

// ===================================================================
// Binding Strategy (Static vs Dynamic)
// ===================================================================

/// How a library binding should be linked
/// Static: library code merged into binary at link time (unikernel, embedded, security-critical)
/// Dynamic: symbol reference resolved at runtime by dynamic linker (desktop, server, plugins)
type BindingStrategy =
    | Static   // Link library code directly into binary (no runtime dependency)
    | Dynamic  // Record symbol reference, resolve via PLT/GOT at runtime

// ===================================================================
// Emission Result (Witness-Based Pattern)
// ===================================================================

/// Result of witnessing a platform binding
type EmissionResult =
    | WitnessedValue of ssaName: string * mlirType: MLIRType
    | WitnessedVoid
    | NotSupported of reason: string

// ===================================================================
// Platform Primitive Types
// ===================================================================

/// Represents a platform primitive call extracted from FNCS SemanticGraph
/// Platform primitives are functions in Alloy.Platform.Bindings that
/// Alex provides platform-specific implementations for
type PlatformPrimitive = {
    /// The entry point name (e.g., "writeBytes")
    EntryPoint: string
    /// The library name (e.g., "platform")
    Library: string
    /// Calling convention
    CallingConvention: string
    /// Argument SSA names and types (already witnessed by post-order)
    Args: (string * MLIRType) list
    /// Return type
    ReturnType: MLIRType
    /// Binding strategy (from project configuration)
    BindingStrategy: BindingStrategy
}

/// External function declaration needed by a binding
type ExternalDeclaration = {
    Name: string
    Signature: string
    Library: string option
}

// ===================================================================
// Witness-Based Binding Signature
// ===================================================================

/// A witness-based binding takes a platform primitive and zipper,
/// returns updated zipper and emission result
/// This replaces the old monad-based: PlatformPrimitive -> MLIR<EmissionResult>
type WitnessBinding = PlatformPrimitive -> MLIRZipper -> MLIRZipper * EmissionResult

// ===================================================================
// Platform Dispatch Registry (Witness-Based)
// ===================================================================

/// Registry for platform primitive bindings
/// Dispatches based on (platform, entry_point) to platform-specific witness functions
module PlatformDispatch =

    /// Binding registration: (platform, entry_point) -> witness binding function
    let mutable private bindings: Map<(OSFamily * Architecture * string), WitnessBinding> = Map.empty
    let mutable private currentPlatform: TargetPlatform option = None

    /// Register a witness binding for a specific platform and entry point
    let register (os: OSFamily) (arch: Architecture) (entryPoint: string) (binding: WitnessBinding) =
        let key = (os, arch, entryPoint)
        bindings <- Map.add key binding bindings

    /// Register a binding for all architectures of an OS
    let registerForOS (os: OSFamily) (entryPoint: string) (binding: WitnessBinding) =
        // Register for common architectures
        register os X86_64 entryPoint binding
        register os ARM64 entryPoint binding

    /// Set the current target platform
    let setTargetPlatform (platform: TargetPlatform) =
        currentPlatform <- Some platform

    /// Get current target platform
    let getTargetPlatform () =
        currentPlatform |> Option.defaultValue (TargetPlatform.detectHost())

    /// Dispatch a platform primitive to its witness binding
    /// Returns updated zipper and emission result
    let dispatch (prim: PlatformPrimitive) (zipper: MLIRZipper) : MLIRZipper * EmissionResult =
        let platform = getTargetPlatform()
        let key = (platform.OS, platform.Arch, prim.EntryPoint)
        match Map.tryFind key bindings with
        | Some binding ->
            binding prim zipper
        | None ->
            // Try without architecture specificity (OS-level binding)
            let fallbackKey = (platform.OS, X86_64, prim.EntryPoint)
            match Map.tryFind fallbackKey bindings with
            | Some binding -> binding prim zipper
            | None -> zipper, NotSupported $"No binding for {prim.EntryPoint} on {platform.OS}/{platform.Arch}"

    /// Check if an entry point has a registered binding
    let hasBinding (entryPoint: string) : bool =
        let platform = getTargetPlatform()
        let key = (platform.OS, platform.Arch, entryPoint)
        Map.containsKey key bindings ||
        Map.containsKey (platform.OS, X86_64, entryPoint) bindings

    /// Clear all registered bindings (for testing)
    let clear () =
        bindings <- Map.empty
        currentPlatform <- None

    /// Get all registered entry points for current platform
    let getRegisteredEntryPoints () : string list =
        let platform = getTargetPlatform()
        bindings
        |> Map.toList
        |> List.filter (fun ((os, arch, _), _) -> os = platform.OS && (arch = platform.Arch || arch = X86_64))
        |> List.map (fun ((_, _, ep), _) -> ep)
        |> List.distinct
