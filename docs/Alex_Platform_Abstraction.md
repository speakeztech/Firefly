# Alex Platform Abstraction Layer

## Purpose

This document describes how Alex abstracts platform differences for consistent code generation across operating systems and architectures. The platform abstraction layer enables the "Library of Alexandria" vision where high-level F# operations compile optimally for any target.

## Target Platform Identification

### Target Triple Format

Following LLVM conventions, targets are identified by triples:

```
<arch>-<vendor>-<os>[-<environment>]

Examples:
- x86_64-unknown-linux-gnu     (Linux AMD64)
- x86_64-pc-windows-msvc       (Windows x64)
- x86_64-apple-darwin          (macOS Intel)
- aarch64-apple-darwin         (macOS Apple Silicon)
- thumbv7em-none-eabihf        (ARM Cortex-M4F bare metal)
- wasm32-unknown-unknown       (WebAssembly)
```

### Platform Enumeration

```fsharp
module Alex.Bindings.PlatformRegistry

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
    | ARM32
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

/// Parse target triple to platform
let parseTriple (triple: string) : TargetPlatform option =
    let parts = triple.Split('-')
    match parts with
    | [| arch; _vendor; os |] | [| arch; _vendor; os; _env |] ->
        let architecture =
            match arch with
            | "x86_64" | "amd64" -> Some X86_64
            | "aarch64" | "arm64" -> Some ARM64
            | a when a.StartsWith("armv7") || a.StartsWith("thumb") -> Some ARM32
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
    | _ -> None
```

## Platform-Specific Operation Categories

### 1. Syscall Mechanisms

Different platforms use different syscall conventions:

| Platform | Mechanism | Syscall Instruction | Syscall Number Register |
|----------|-----------|---------------------|------------------------|
| Linux x86-64 | Inline syscall | `syscall` | rax |
| Linux ARM64 | Inline syscall | `svc #0` | x8 |
| Windows x64 | External call | N/A (call kernel32) | N/A |
| macOS x86-64 | Inline syscall | `syscall` | rax (with 0x2000000 offset) |
| macOS ARM64 | Inline syscall | `svc #0x80` | x16 |
| Bare metal | N/A | Direct memory access | N/A |

### 2. External Library Dependencies

```fsharp
/// External libraries needed per platform
type ExternalDependencies = {
    Libraries: string list
    Functions: (string * string) list  // (function name, signature)
}

let getTimeDependencies (platform: TargetPlatform) : ExternalDependencies =
    match platform.OS with
    | Linux -> {
        Libraries = []  // Uses inline syscalls
        Functions = []
      }
    | Windows -> {
        Libraries = ["kernel32.dll"]
        Functions = [
            ("QueryPerformanceCounter", "(!llvm.ptr) -> i32")
            ("QueryPerformanceFrequency", "(!llvm.ptr) -> i32")
            ("Sleep", "(i32) -> ()")
            ("GetSystemTimeAsFileTime", "(!llvm.ptr) -> ()")
        ]
      }
    | MacOS -> {
        Libraries = ["libSystem.B.dylib"]
        Functions = [
            ("mach_absolute_time", "() -> i64")
            ("mach_timebase_info", "(!llvm.ptr) -> i32")
            ("nanosleep", "(!llvm.ptr, !llvm.ptr) -> i32")
            ("gettimeofday", "(!llvm.ptr, !llvm.ptr) -> i32")
        ]
      }
    | BareMetal -> {
        Libraries = []
        Functions = []  // Direct register access
      }
    | _ -> { Libraries = []; Functions = [] }
```

### 3. Memory Layout Considerations

Different architectures have different alignment and layout requirements:

```fsharp
/// Architecture-specific memory layout
type MemoryLayout = {
    PointerSize: int       // 4 or 8 bytes
    Alignment: int         // Default alignment
    StackGrowsDown: bool
    LittleEndian: bool
}

let getMemoryLayout (arch: Architecture) : MemoryLayout =
    match arch with
    | X86_64 | ARM64 | RISCV64 ->
        { PointerSize = 8; Alignment = 8; StackGrowsDown = true; LittleEndian = true }
    | ARM32 | RISCV32 | WASM32 ->
        { PointerSize = 4; Alignment = 4; StackGrowsDown = true; LittleEndian = true }
```

## Binding Emission Interface

### Common Emission Pattern

All platform-specific bindings follow a common interface:

```fsharp
/// Binding emission result
type EmissionResult =
    | Emitted of resultSSA: string
    | EmittedVoid  // For void-returning operations like sleep
    | NotSupported of reason: string
    | DeferToGeneric  // Fall back to generic emission

/// Platform binding interface
type IPlatformBinding =
    abstract member Matches: PSGNode -> bool
    abstract member Emit: MLIRBuilder -> SSAContext -> ProgramSemanticGraph -> PSGNode -> EmissionResult
    abstract member ExternalDeclarations: unit -> string list
```

### Binding Registration

```fsharp
/// Global binding registry
module BindingRegistry =
    let mutable private bindings: Map<TargetPlatform, IPlatformBinding list> = Map.empty

    let register (platform: TargetPlatform) (binding: IPlatformBinding) =
        let existing = bindings |> Map.tryFind platform |> Option.defaultValue []
        bindings <- Map.add platform (binding :: existing) bindings

    let tryEmit (platform: TargetPlatform) (node: PSGNode)
                (builder: MLIRBuilder) (ctx: SSAContext) (psg: ProgramSemanticGraph) =
        match Map.tryFind platform bindings with
        | Some platformBindings ->
            platformBindings
            |> List.tryPick (fun binding ->
                if binding.Matches node then
                    match binding.Emit builder ctx psg node with
                    | Emitted result -> Some (Some result)
                    | EmittedVoid -> Some None
                    | NotSupported _ | DeferToGeneric -> None
                else None)
        | None -> None

    let getExternalDeclarations (platform: TargetPlatform) =
        match Map.tryFind platform bindings with
        | Some platformBindings ->
            platformBindings
            |> List.collect (fun b -> b.ExternalDeclarations())
            |> List.distinct
        | None -> []
```

## MLIR Dialect Selection

Different operations map to different MLIR dialects based on target:

```fsharp
/// Select appropriate MLIR operation for target
type OperationDialect =
    | ArithDialect of operation: string
    | LLVMDialect of operation: string
    | FuncDialect of operation: string
    | SCFDialect of operation: string
    | MemRefDialect of operation: string

let selectDialect (platform: TargetPlatform) (operation: string) : OperationDialect =
    match platform.OS, operation with
    // Syscalls use LLVM inline_asm on Linux
    | Linux, "syscall" -> LLVMDialect "inline_asm"
    // External calls use func.call
    | Windows, "external_call" -> FuncDialect "call"
    // Memory operations typically use llvm dialect
    | _, "alloca" -> LLVMDialect "alloca"
    | _, "load" -> LLVMDialect "load"
    | _, "store" -> LLVMDialect "store"
    // Arithmetic is dialect-agnostic
    | _, "add" | _, "mul" | _, "div" -> ArithDialect operation
    | _ -> LLVMDialect operation
```

## Cross-Compilation Support

### Host vs Target Platform

```fsharp
/// Compilation context includes both host and target
type CompilationContext = {
    HostPlatform: TargetPlatform   // Where compiler runs
    TargetPlatform: TargetPlatform  // Where code will execute
    CrossCompiling: bool
}

let createContext (hostTriple: string) (targetTriple: string) =
    let host = parseTriple hostTriple |> Option.get
    let target = parseTriple targetTriple |> Option.get
    {
        HostPlatform = host
        TargetPlatform = target
        CrossCompiling = hostTriple <> targetTriple
    }
```

### Platform Detection at Compile Time

When building on a platform, detect the host:

```fsharp
let detectHostPlatform () : TargetPlatform =
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
        | System.Runtime.InteropServices.Architecture.Arm -> ARM32
        | _ -> X86_64

    { OS = os; Arch = arch; Triple = "detected"; Features = Set.empty }
```

## Integration with fidproj

The `.fidproj` file specifies target platforms:

```toml
[build]
# Default target (same as host if not specified)
target = "x86_64-unknown-linux-gnu"

# Platform-specific overrides
[target.'cfg(target_os = "windows")']
target = "x86_64-pc-windows-msvc"

[target.'cfg(target_os = "macos")']
target = "x86_64-apple-darwin"

# Cross-compilation example
[target.'cfg(target_arch = "aarch64")']
target = "aarch64-unknown-linux-gnu"
```

### Parsing Platform Configuration

```fsharp
module FidprojPlatform =
    let parseTargetConfig (fidproj: FidprojFile) : TargetPlatform =
        let defaultTriple = fidproj.Build.Target |> Option.defaultValue (detectHostTriple())

        // Check for platform-specific overrides
        let hostOS = detectHostPlatform().OS
        let override =
            fidproj.TargetConfigs
            |> List.tryFind (fun cfg -> matchesCfg cfg.Condition hostOS)
            |> Option.bind (fun cfg -> cfg.Target)

        parseTriple (override |> Option.defaultValue defaultTriple)
        |> Option.defaultWith (fun () -> detectHostPlatform())
```

## Coeffect-Driven Optimization

Platform abstraction integrates with coeffect analysis for optimization:

```fsharp
/// Coeffects inform platform-specific optimizations
type PlatformOptimization =
    | UseSIMD of width: int
    | UsePrefetch of distance: int
    | UseNonTemporal  // Bypass cache for streaming
    | UseInlineAsm    // Prefer inline assembly

let getOptimizations (platform: TargetPlatform) (coeffects: Set<Coeffect>) =
    match platform.Arch, coeffects with
    | X86_64, cs when cs.Contains(MemoryPattern Sequential) && platform.Features.Contains("avx2") ->
        [UseSIMD 256; UsePrefetch 64]
    | ARM64, cs when cs.Contains(MemoryPattern Sequential) && platform.Features.Contains("neon") ->
        [UseSIMD 128]
    | _, cs when cs.Contains(MemoryPattern Streaming) ->
        [UseNonTemporal]
    | _ -> []
```

## Error Handling for Unsupported Platforms

```fsharp
type PlatformError =
    | UnsupportedPlatform of triple: string * reason: string
    | MissingFeature of feature: string * platform: TargetPlatform
    | IncompatibleOperation of operation: string * platform: TargetPlatform

let validatePlatformSupport (platform: TargetPlatform) (operation: string) : Result<unit, PlatformError> =
    match platform.OS, operation with
    | WASM, "directSyscall" ->
        Error (IncompatibleOperation ("Direct syscalls not supported in WASM", platform))
    | BareMetal, "sleep" when not (platform.Features.Contains "systick") ->
        Error (MissingFeature ("SysTick timer required for sleep on bare metal", platform))
    | _ -> Ok ()
```

---

*Platform Abstraction Design for Alex - December 2024*
