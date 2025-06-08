module CLI.ToolchainVerification

open System
open System.IO
open Core.XParsec.Foundation

/// Represents the status of a toolchain component
type ComponentStatus =
    | Found of version: string
    | Missing of hint: string
    | Error of message: string

/// Represents a required toolchain component
type ToolchainComponent = {
    Name: string
    Description: string
    CheckCommand: string option
    CheckFiles: string list
    InstallHint: string
    Required: bool
}

/// Platform-specific toolchain requirements
let getWindowsToolchainRequirements() : ToolchainComponent list =
    let msysPrefix = Environment.GetEnvironmentVariable("MSYSTEM_PREFIX")
    let mingwRoot = if String.IsNullOrEmpty(msysPrefix) then "/mingw64" else msysPrefix
    
    [
        {
            Name = "GCC Compiler"
            Description = "GNU Compiler Collection for compiling object files"
            CheckCommand = Some "gcc --version"
            CheckFiles = []
            InstallHint = "pacman -S mingw-w64-x86_64-gcc"
            Required = true
        }
        {
            Name = "MinGW-w64 CRT"
            Description = "C Runtime startup files (critical for linking)"
            CheckCommand = None
            CheckFiles = [
                Path.Combine(mingwRoot, "lib", "crt1.o")
                Path.Combine(mingwRoot, "lib", "crt2.o")
            ]
            InstallHint = "pacman -S mingw-w64-x86_64-crt-git"
            Required = true
        }
        {
            Name = "LLVM Tools"
            Description = "LLVM compiler infrastructure (llc)"
            CheckCommand = Some "llc --version"
            CheckFiles = []
            InstallHint = "pacman -S mingw-w64-x86_64-llvm"
            Required = true
        }
        {
            Name = "GNU Binutils"
            Description = "Binary utilities including linker"
            CheckCommand = Some "ld --version"
            CheckFiles = []
            InstallHint = "pacman -S mingw-w64-x86_64-binutils"
            Required = true
        }
        {
            Name = "LLD Linker"
            Description = "LLVM's native linker (alternative to GNU ld)"
            CheckCommand = Some "lld --version"
            CheckFiles = []
            InstallHint = "pacman -S mingw-w64-x86_64-lld"
            Required = false
        }
    ]

/// Checks if a command is available and returns its version
let checkCommand (command: string) : ComponentStatus =
    try
        let parts = command.Split(' ', 2)
        let exe = parts.[0]
        let args = if parts.Length > 1 then parts.[1] else ""
        
        let processInfo = System.Diagnostics.ProcessStartInfo()
        processInfo.FileName <- exe
        processInfo.Arguments <- args
        processInfo.UseShellExecute <- false
        processInfo.RedirectStandardOutput <- true
        processInfo.RedirectStandardError <- true
        processInfo.CreateNoWindow <- true
        
        use proc = System.Diagnostics.Process.Start(processInfo)
        let output = proc.StandardOutput.ReadToEnd()
        proc.WaitForExit(5000) |> ignore
        
        if proc.ExitCode = 0 then
            let firstLine = output.Split('\n').[0].Trim()
            Found firstLine
        else
            Missing "Command failed"
    with
    | ex -> Error ex.Message

/// Checks if required files exist
let checkFiles (files: string list) : ComponentStatus =
    let missing = files |> List.filter (not << File.Exists)
    if missing.IsEmpty then
        Found "All files present"
    else
        Missing (sprintf "Missing files: %s" (String.concat ", " missing))

/// Checks a single toolchain component
let checkComponent (component: ToolchainComponent) : ComponentStatus =
    match component.CheckCommand with
    | Some cmd -> checkCommand cmd
    | None ->
        if component.CheckFiles.IsEmpty then
            Error "No check method specified"
        else
            checkFiles component.CheckFiles

/// Detects the MSYS2 environment
let detectMSYS2Environment() : string option =
    match Environment.GetEnvironmentVariable("MSYSTEM") with
    | null | "" -> None
    | env -> Some env

/// Performs complete toolchain verification
let verifyToolchain (verbose: bool) : CompilerResult<unit> =
    printfn "Verifying Firefly toolchain requirements..."
    printfn "=========================================="
    
    // Detect environment
    let environment = 
        if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
            match detectMSYS2Environment() with
            | Some env -> sprintf "MSYS2 %s" env
            | None -> "Windows (native)"
        elif RuntimeInformation.IsOSPlatform(OSPlatform.Linux) then
            "Linux"
        elif RuntimeInformation.IsOSPlatform(OSPlatform.OSX) then
            "macOS"
        else
            "Unknown"
    
    printfn "Platform: %s" environment
    printfn ""
    
    // Check for MSYS2 MINGW64 environment specifically
    if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
        match detectMSYS2Environment() with
        | Some "MSYS" ->
            printfn "WARNING: You are in the MSYS environment."
            printfn "For native Windows executables, use MSYS2 MINGW64 instead!"
            printfn ""
        | _ -> ()
    
    // Get platform-specific requirements
    let requirements = 
        if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
            getWindowsToolchainRequirements()
        else
            [] // Add Linux/macOS requirements as needed
    
    // Check each component
    let results = 
        requirements 
        |> List.map (fun component ->
            let status = checkComponent component
            (component, status))
    
    // Display results
    let mutable hasErrors = false
    
    results |> List.iter (fun (component, status) ->
        let statusSymbol, statusText, isError = 
            match status with
            | Found version -> 
                "✓", version, false
            | Missing hint -> 
                "✗", sprintf "Missing - %s" hint, component.Required
            | Error msg -> 
                "!", sprintf "Error - %s" msg, component.Required
        
        if isError then hasErrors <- true
        
        printfn "%s %s: %s" statusSymbol component.Name statusText
        
        if verbose || isError then
            printfn "  %s" component.Description
            
        if isError && component.Required then
            printfn "  To install: %s" component.InstallHint
            printfn ""
    )
    
    printfn "=========================================="
    
    if hasErrors then
        printfn "ERROR: Missing required components!"
        printfn "Please install the missing components before using Firefly."
        CompilerFailure [CompilerError("toolchain", "Missing required toolchain components", None)]
    else
        printfn "All required components found!"
        Success ()

/// Quick check for critical components (used during compilation)
let quickVerifyToolchain() : bool =
    if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
        // Quick check for the most common issue: missing CRT files
        let msysPrefix = Environment.GetEnvironmentVariable("MSYSTEM_PREFIX")
        let mingwRoot = if String.IsNullOrEmpty(msysPrefix) then "/mingw64" else msysPrefix
        let crt1Path = Path.Combine(mingwRoot, "lib", "crt1.o")
        
        if not (File.Exists(crt1Path)) then
            printfn "WARNING: MinGW-w64 C runtime files not found!"
            printfn "This is likely why linking is failing."
            printfn "Install with: pacman -S mingw-w64-x86_64-crt-git"
            false
        else
            true
    else
        true // Assume OK on other platforms for now

/// Suggests fixes for common toolchain issues
let suggestToolchainFixes (error: string) : unit =
    if error.Contains("WinMain") then
        printfn ""
        printfn "This error typically indicates missing MinGW-w64 runtime files."
        printfn "Try running: pacman -S mingw-w64-x86_64-crt-git"
        printfn ""
        printfn "If you're not in MINGW64 environment, switch to it:"
        printfn "Close this terminal and open 'MSYS2 MINGW64' from Start Menu"
    elif error.Contains("gcc: command not found") then
        printfn ""
        printfn "GCC compiler not found. Install with:"
        printfn "pacman -S mingw-w64-x86_64-gcc"
    elif error.Contains("llc: command not found") then
        printfn ""
        printfn "LLVM tools not found. Install with:"
        printfn "pacman -S mingw-w64-x86_64-llvm"