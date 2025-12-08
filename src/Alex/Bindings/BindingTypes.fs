module Alex.Bindings.BindingTypes

open Core.PSG.Types
open Alex.CodeGeneration.MLIRBuilder

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
            else Linux  // Default fallback

        let arch =
            match System.Runtime.InteropServices.RuntimeInformation.OSArchitecture with
            | System.Runtime.InteropServices.Architecture.X64 -> X86_64
            | System.Runtime.InteropServices.Architecture.Arm64 -> ARM64
            | System.Runtime.InteropServices.Architecture.Arm -> ARM32_Thumb
            | _ -> X86_64

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
// Extern Primitive Types
// ===================================================================

/// Represents an extern primitive call extracted from PSG
type ExternPrimitive = {
    /// The entry point name (e.g., "fidelity_write_bytes")
    EntryPoint: string
    /// The library name (e.g., "__fidelity")
    Library: string
    /// Calling convention
    CallingConvention: string
    /// Argument values (already evaluated)
    Args: Val list
    /// Return type
    ReturnType: Ty
}

/// Result of attempting to emit a binding
type EmissionResult =
    | Emitted of resultVal: Val
    | EmittedVoid
    | NotSupported of reason: string

/// External function declaration needed by a binding
type ExternalDeclaration = {
    Name: string
    Signature: string
    Library: string option
}

// ===================================================================
// Binding Function Signature
// ===================================================================

/// A binding is a function that takes an extern primitive and returns MLIR
/// The MLIR<Val> monad handles all state threading
type ExternBinding = ExternPrimitive -> MLIR<EmissionResult>

// ===================================================================
// Extern Dispatch Registry
// ===================================================================

/// Registry for extern primitive bindings
/// Dispatches based on (platform, entry_point) to platform-specific MLIR generators
module ExternDispatch =

    /// Binding registration: (platform, entry_point) -> binding function
    let mutable private bindings: Map<(OSFamily * Architecture * string), ExternBinding> = Map.empty
    let mutable private currentPlatform: TargetPlatform option = None

    /// Register a binding for a specific platform and entry point
    let register (os: OSFamily) (arch: Architecture) (entryPoint: string) (binding: ExternBinding) =
        let key = (os, arch, entryPoint)
        bindings <- Map.add key binding bindings

    /// Register a binding for all architectures of an OS
    let registerForOS (os: OSFamily) (entryPoint: string) (binding: ExternBinding) =
        // Register for common architectures
        register os X86_64 entryPoint binding
        register os ARM64 entryPoint binding

    /// Set the current target platform
    let setTargetPlatform (platform: TargetPlatform) =
        currentPlatform <- Some platform

    /// Get current target platform
    let getTargetPlatform () =
        currentPlatform |> Option.defaultValue (TargetPlatform.detectHost())

    /// Dispatch an extern primitive to its platform binding
    let dispatch (prim: ExternPrimitive) : MLIR<EmissionResult> = mlir {
        let platform = getTargetPlatform()
        let key = (platform.OS, platform.Arch, prim.EntryPoint)
        match Map.tryFind key bindings with
        | Some binding ->
            return! binding prim
        | None ->
            // Try without architecture specificity (OS-level binding)
            let fallbackKey = (platform.OS, X86_64, prim.EntryPoint)
            match Map.tryFind fallbackKey bindings with
            | Some binding -> return! binding prim
            | None -> return NotSupported $"No binding for {prim.EntryPoint} on {platform.OS}/{platform.Arch}"
    }

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
