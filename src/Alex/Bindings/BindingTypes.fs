module Alex.Bindings.BindingTypes

open Core.PSG.Types
open Alex.CodeGeneration.EmissionContext

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
// Binding Emission Types
// ===================================================================

/// Result of attempting to emit a binding
type EmissionResult =
    | Emitted of resultSSA: string
    | EmittedVoid
    | NotSupported of reason: string
    | DeferToGeneric

/// External function declaration needed by a binding
type ExternalDeclaration = {
    Name: string
    Signature: string
    Library: string option
}

/// A binding that can match and emit platform-specific code
type PlatformBinding = {
    /// Unique name for this binding
    Name: string
    /// Symbol patterns this binding handles (e.g., "Alloy.Time.currentTicks")
    SymbolPatterns: string list
    /// Platforms this binding supports
    SupportedPlatforms: TargetPlatform list
    /// Check if this binding matches a given PSG node
    Matches: PSGNode -> bool
    /// Emit MLIR for this binding (returns SSA value or None)
    Emit: TargetPlatform -> MLIRBuilder -> SSAContext
          -> ProgramSemanticGraph -> PSGNode -> EmissionResult
    /// Get external declarations needed by this binding
    GetExternalDeclarations: TargetPlatform -> ExternalDeclaration list
}

// ===================================================================
// Binding Registry
// ===================================================================

/// Global registry for platform bindings
module BindingRegistry =
    let mutable private bindings: PlatformBinding list = []
    let mutable private currentPlatform: TargetPlatform option = None

    /// Register a binding
    let register (binding: PlatformBinding) =
        bindings <- binding :: bindings

    /// Set the current target platform
    let setTargetPlatform (platform: TargetPlatform) =
        currentPlatform <- Some platform

    /// Get current target platform
    let getTargetPlatform () =
        currentPlatform |> Option.defaultValue (TargetPlatform.detectHost())

    /// Find a binding that matches a PSG node for the current platform
    let findBinding (node: PSGNode) : PlatformBinding option =
        let platform = getTargetPlatform()
        bindings
        |> List.tryFind (fun b ->
            b.Matches node &&
            b.SupportedPlatforms |> List.exists (fun p -> p.OS = platform.OS && p.Arch = platform.Arch))

    /// Try to emit using a platform binding
    let tryEmit (node: PSGNode) (builder: MLIRBuilder)
                (ctx: SSAContext) (psg: ProgramSemanticGraph)
                : string option =
        let platform = getTargetPlatform()
        match findBinding node with
        | Some binding ->
            match binding.Emit platform builder ctx psg node with
            | Emitted result -> Some result
            | EmittedVoid -> Some ""  // Void return, no SSA value
            | NotSupported _ | DeferToGeneric -> None
        | None -> None

    /// Get all external declarations needed for current platform
    let getAllExternalDeclarations () : ExternalDeclaration list =
        let platform = getTargetPlatform()
        bindings
        |> List.filter (fun b ->
            b.SupportedPlatforms |> List.exists (fun p -> p.OS = platform.OS && p.Arch = platform.Arch))
        |> List.collect (fun b -> b.GetExternalDeclarations platform)
        |> List.distinctBy (fun d -> d.Name)

    /// Clear all registered bindings (for testing)
    let clear () =
        bindings <- []
        currentPlatform <- None
