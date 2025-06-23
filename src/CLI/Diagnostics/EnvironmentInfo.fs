module CLI.Diagnostics.EnvironmentInfo

open System
open System.IO
open System.Runtime.InteropServices

/// Environment information for diagnostics
type EnvironmentInfo = {
    OperatingSystem: string
    Architecture: string
    RuntimeVersion: string
    RuntimeIdentifier: string
    IsWindows: bool
    IsLinux: bool
    IsMacOS: bool
    MSYS2Environment: MSYS2Info option
    EnvironmentVariables: Map<string, string>
    ToolchainPaths: string list
}

/// MSYS2 specific information
and MSYS2Info = {
    MSysType: string      // MSYS, MINGW64, MINGW32, etc.
    MSysRoot: string      // Root installation path
    MSysPrefix: string    // Active prefix path
    IsMinGW64: bool
    IsMinGW32: bool
}

/// Get current runtime identifier
let getRuntimeIdentifier() =
    let arch = 
        match RuntimeInformation.OSArchitecture with
        | Architecture.X64 -> "x64"
        | Architecture.X86 -> "x86"
        | Architecture.Arm -> "arm"
        | Architecture.Arm64 -> "arm64"
        | _ -> "unknown"
    
    let os = 
        if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then "win"
        elif RuntimeInformation.IsOSPlatform(OSPlatform.Linux) then "linux"
        elif RuntimeInformation.IsOSPlatform(OSPlatform.OSX) then "osx"
        else "unknown"
    
    sprintf "%s-%s" os arch

/// Detect MSYS2 environment on Windows
let detectMSYS2() =
    if not (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) then
        None
    else
        let msystem = Environment.GetEnvironmentVariable("MSYSTEM")
        let msysRoot = Environment.GetEnvironmentVariable("MSYSTEM_PREFIX")
        let mingwPrefix = Environment.GetEnvironmentVariable("MINGW_PREFIX")
        
        match msystem with
        | null | "" -> None
        | msysType ->
            let prefix = 
                if not (String.IsNullOrEmpty(msysRoot)) then msysRoot
                elif not (String.IsNullOrEmpty(mingwPrefix)) then mingwPrefix
                else 
                    // Try to detect from PATH
                    let path = Environment.GetEnvironmentVariable("PATH")
                    if String.IsNullOrEmpty(path) then ""
                    else
                        path.Split(';')
                        |> Array.tryFind (fun p -> p.Contains("mingw64") || p.Contains("mingw32"))
                        |> Option.map (fun p -> p.Substring(0, p.IndexOf("bin")))
                        |> Option.defaultValue ""
            
            Some {
                MSysType = msysType
                MSysRoot = if String.IsNullOrEmpty(msysRoot) then prefix else msysRoot
                MSysPrefix = prefix
                IsMinGW64 = msysType = "MINGW64"
                IsMinGW32 = msysType = "MINGW32"
            }

/// Get relevant environment variables
let getRelevantEnvironmentVariables() =
    let relevantVars = [
        "PATH"
        "MSYSTEM"
        "MSYSTEM_PREFIX"
        "MINGW_PREFIX"
        "CC"
        "CXX"
        "LD"
        "LLVM_HOME"
        "MLIR_HOME"
        "FSHARP_COMPILER_BIN"
    ]
    
    relevantVars
    |> List.choose (fun var ->
        match Environment.GetEnvironmentVariable(var) with
        | null | "" -> None
        | value -> Some (var, value))
    |> Map.ofList

/// Extract toolchain paths from PATH
let extractToolchainPaths() =
    let path = Environment.GetEnvironmentVariable("PATH")
    if String.IsNullOrEmpty(path) then
        []
    else
        path.Split(Path.PathSeparator)
        |> Array.filter (fun p -> 
            not (String.IsNullOrWhiteSpace(p)) &&
            (p.Contains("mingw") || p.Contains("llvm") || p.Contains("gcc") || 
             p.Contains("clang") || p.Contains("bin")))
        |> Array.toList

/// Gather complete environment information
let gatherEnvironmentInfo() : EnvironmentInfo =
    {
        OperatingSystem = RuntimeInformation.OSDescription
        Architecture = RuntimeInformation.OSArchitecture.ToString()
        RuntimeVersion = RuntimeInformation.FrameworkDescription
        RuntimeIdentifier = getRuntimeIdentifier()
        IsWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
        IsLinux = RuntimeInformation.IsOSPlatform(OSPlatform.Linux)
        IsMacOS = RuntimeInformation.IsOSPlatform(OSPlatform.OSX)
        MSYS2Environment = detectMSYS2()
        EnvironmentVariables = getRelevantEnvironmentVariables()
        ToolchainPaths = extractToolchainPaths()
    }

/// Format environment info for display
let formatEnvironmentInfo (info: EnvironmentInfo) =
    let sb = System.Text.StringBuilder()
    
    sb.AppendLine("=== Environment Information ===") |> ignore
    sb.AppendLine(sprintf "OS: %s" info.OperatingSystem) |> ignore
    sb.AppendLine(sprintf "Architecture: %s" info.Architecture) |> ignore
    sb.AppendLine(sprintf "Runtime: %s" info.RuntimeVersion) |> ignore
    sb.AppendLine(sprintf "Runtime ID: %s" info.RuntimeIdentifier) |> ignore
    
    match info.MSYS2Environment with
    | Some msys ->
        sb.AppendLine() |> ignore
        sb.AppendLine("MSYS2 Environment:") |> ignore
        sb.AppendLine(sprintf "  Type: %s" msys.MSysType) |> ignore
        sb.AppendLine(sprintf "  Root: %s" msys.MSysRoot) |> ignore
        sb.AppendLine(sprintf "  Prefix: %s" msys.MSysPrefix) |> ignore
    | None ->
        if info.IsWindows then
            sb.AppendLine() |> ignore
            sb.AppendLine("MSYS2: Not detected (native Windows environment)") |> ignore
    
    if not (List.isEmpty info.ToolchainPaths) then
        sb.AppendLine() |> ignore
        sb.AppendLine("Toolchain Paths:") |> ignore
        info.ToolchainPaths |> List.iter (fun p ->
            sb.AppendLine(sprintf "  %s" p) |> ignore)
    
    sb.ToString()

/// Check if running in a suitable compilation environment
let validateCompilationEnvironment() =
    let info = gatherEnvironmentInfo()
    
    if info.IsWindows then
        match info.MSYS2Environment with
        | Some msys when msys.MSysType = "MSYS" ->
            Error "Running in MSYS environment. Please use MINGW64 for native compilation."
        | Some msys when msys.IsMinGW64 || msys.IsMinGW32 ->
            Ok info
        | _ ->
            Error "No suitable MSYS2 environment detected. Please install MSYS2 and use MINGW64."
    else
        Ok info