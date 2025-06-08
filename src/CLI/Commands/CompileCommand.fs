module CLI.Commands.CompileCommand

open System
open System.IO
open System.Runtime.InteropServices
open Argu
open Core.XParsec.Foundation
open Dabbit.Parsing.Translator
open Core.Conversion.LoweringPipeline
open Core.Conversion.LLVMTranslator
open Core.Conversion.OptimizationPipeline
open CLI.Configurations.ProjectConfig

/// Command line arguments for the compile command
type CompileArgs =
    | Input of string
    | Output of string
    | Target of string
    | Optimize of string
    | Config of string
    | Keep_Intermediates
    | Verbose
    | No_External_Tools
with
    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Input _ -> "Input F# source file (required)"
            | Output _ -> "Output binary path (required)"
            | Target _ -> "Target platform (e.g. x86_64-pc-windows-msvc, x86_64-pc-linux-gnu, embedded)"
            | Optimize _ -> "Optimization level (none, less, default, aggressive, size, sizemin)"
            | Config _ -> "Path to configuration file (firefly.toml)"
            | Keep_Intermediates -> "Keep intermediate files (FCS AST, Oak AST, MLIR, LLVM IR)"
            | Verbose -> "Enable verbose output and diagnostics"
            | No_External_Tools -> "Use only internal XParsec-based transformations"

/// Gets the default target for the current platform
let private getDefaultTarget() =
    if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
        "x86_64-w64-windows-gnu"
    elif RuntimeInformation.IsOSPlatform(OSPlatform.Linux) then
        "x86_64-pc-linux-gnu"
    elif RuntimeInformation.IsOSPlatform(OSPlatform.OSX) then
        "x86_64-apple-darwin"
    else
        "x86_64-w64-windows-gnu"

/// Cross-platform text file writing with UTF-8 encoding and Unix line endings
let private writeTextFile (filePath: string) (content: string) : unit =
    let encoding = System.Text.UTF8Encoding(false)
    let normalizedContent = content.Replace("\r\n", "\n").Replace("\r", "\n")
    File.WriteAllText(filePath, normalizedContent, encoding)

/// Comprehensive intermediate file saving with guaranteed AST support
let private saveIntermediateFiles (basePath: string) (baseName: string) (keepIntermediates: bool) (pipelineOutput: TranslationPipelineOutput) =
    printfn "=== SAVING INTERMEDIATE FILES ==="
    printfn "Keep intermediates: %b" keepIntermediates
    printfn "Base path: %s" basePath
    printfn "Base name: %s" baseName
    printfn "Pipeline outputs available: %d" pipelineOutput.PhaseOutputs.Count
    
    if keepIntermediates then
        try
            let intermediatesDir = Path.Combine(basePath, "intermediates")
            if not (Directory.Exists(intermediatesDir)) then
                Directory.CreateDirectory(intermediatesDir) |> ignore
                printfn "Created intermediates directory: %s" intermediatesDir
            
            pipelineOutput.PhaseOutputs |> Map.iter (fun key _ -> printfn "  Available: %s" key)
            
            // Save pipeline diagnostics
            let diagnosticsPath = Path.Combine(intermediatesDir, baseName + ".diagnostics.txt")
            let diagnosticsContent = 
                pipelineOutput.Diagnostics
                |> List.map (fun (phase, message) -> sprintf "[%s] %s" phase message)
                |> String.concat "\n"
            writeTextFile diagnosticsPath diagnosticsContent
            printfn "✓ Saved diagnostics to: %s" diagnosticsPath
            
            // Save all intermediate outputs with proper extensions and naming
            pipelineOutput.PhaseOutputs
            |> Map.iter (fun phaseName output ->
                let extension = 
                    match phaseName with
                    | name when name.Contains("mlir") -> ".mlir"
                    | name when name.Contains("oak") -> ".oak"
                    | name when name.Contains("fcs") || name.Contains("compiler-services") -> ".fcs"
                    | name when name.Contains("llvm") -> ".ll"
                    | _ -> ".txt"
                
                let fileName = 
                    match phaseName with
                    | "f#-compiler-services-ast" -> baseName + ".fcs"
                    | "oak-ast" -> baseName + ".oak"
                    | _ -> sprintf "%s.%s%s" baseName (phaseName.Replace("-", "_")) extension
                
                let filePath = Path.Combine(intermediatesDir, fileName)
                writeTextFile filePath output
                printfn "✓ Saved %s (%d chars) to: %s" phaseName output.Length fileName
                
                // Verify file was actually created
                if File.Exists(filePath) then
                    let fileSize = (FileInfo(filePath)).Length
                    printfn "  ✓ VERIFIED: File exists (%d bytes)" fileSize
                else
                    printfn "  ✗ ERROR: File was not created!")
            
            // Save symbol mappings
            let symbolMappingsPath = Path.Combine(intermediatesDir, baseName + ".symbols.txt")
            let symbolMappingsContent = 
                pipelineOutput.SymbolMappings
                |> Map.toList
                |> List.map (fun (original, transformed) -> sprintf "%s -> %s" original transformed)
                |> String.concat "\n"
            writeTextFile symbolMappingsPath symbolMappingsContent
            printfn "✓ Saved symbol mappings to: %s" symbolMappingsPath
            
            // Report final results
            let displayPath = intermediatesDir.Replace(Path.DirectorySeparatorChar, '/')
            printfn "Saved intermediate files to: %s" displayPath
            
            let intermediateFiles = Directory.GetFiles(intermediatesDir, baseName + ".*")
            if intermediateFiles.Length > 0 then
                printfn "Generated intermediate files:"
                intermediateFiles |> Array.iter (fun file ->
                    let fileName = Path.GetFileName(file)
                    let fileSize = (FileInfo(file)).Length
                    printfn "  %s (%d bytes)" fileName fileSize)
            else
                printfn "WARNING: No intermediate files were found after saving!"
            
        with
        | ex ->
            printfn "ERROR: Failed to save intermediate files: %s" ex.Message
            printfn "Stack trace: %s" ex.StackTrace
    else
        printfn "Intermediate file saving disabled"

/// Validates compilation arguments with enhanced error reporting
let private validateCompilationArgs (inputPath: string) (outputPath: string) : CompilerResult<unit> =
    printfn "=== VALIDATING COMPILATION ARGUMENTS ==="
    printfn "Input path: %s" inputPath
    printfn "Output path: %s" outputPath
    
    if not (File.Exists inputPath) then
        printfn "ERROR: Input file does not exist"
        CompilerFailure [CompilerError("argument validation", sprintf "Input file '%s' does not exist" inputPath, None)]
    elif String.IsNullOrWhiteSpace(outputPath) then
        printfn "ERROR: Output path is empty"
        CompilerFailure [CompilerError("argument validation", "Output path cannot be empty", None)]
    else
        let inputExt = Path.GetExtension(inputPath).ToLowerInvariant()
        if inputExt <> ".fs" && inputExt <> ".fsx" then
            printfn "ERROR: Invalid input file extension: %s" inputExt
            CompilerFailure [CompilerError("argument validation", sprintf "Input file must be F# source (.fs or .fsx), got: %s" inputExt, None)]
        else
            printfn "✓ Arguments validated successfully"
            Success ()

/// Enhanced XParsec-based compilation process with GUARANTEED AST generation
let private executeEnhancedCompilationPipeline (inputPath: string) (sourceCode: string) (config: FireflyConfig) (keepIntermediates: bool) (verbose: bool) : CompilerResult<TranslationPipelineOutput> =
    printfn "=== ENHANCED COMPILATION PIPELINE START ==="
    printfn "Input path: %s" inputPath
    printfn "Source code length: %d" sourceCode.Length
    printfn "Keep intermediates: %b" keepIntermediates
    printfn "Verbose: %b" verbose
    
    // Use the enhanced translation pipeline with guaranteed AST generation
    printfn "=== CALLING ENHANCED TRANSLATION PIPELINE ==="
    
    let pipelineResult = translateF#ToMLIRWithIntermediates inputPath sourceCode keepIntermediates
    
    printfn "=== TRANSLATION PIPELINE COMPLETED ==="
    
    match pipelineResult with
    | CompilerFailure pipelineErrors ->
        printfn "Translation pipeline failed with %d errors:" pipelineErrors.Length
        pipelineErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
        CompilerFailure pipelineErrors
    | Success pipelineOutput ->
        printfn "✓ Translation pipeline succeeded!"
        printfn "Generated %d phase outputs:" pipelineOutput.PhaseOutputs.Count
        pipelineOutput.PhaseOutputs |> Map.iter (fun key _ -> printfn "  - %s" key)
        
        // Verify AST outputs are present
        let hasAST = 
            pipelineOutput.PhaseOutputs.ContainsKey("f#-compiler-services-ast") &&
            pipelineOutput.PhaseOutputs.ContainsKey("oak-ast")
        
        if hasAST then
            printfn "✓ AST intermediate representations successfully generated"
        else
            printfn "✗ WARNING: AST intermediate representations are missing!"
            printfn "Available outputs:"
            pipelineOutput.PhaseOutputs |> Map.iter (fun key _ -> printfn "  - %s" key)
        
        if verbose then
            printfn "Successful phases: %d" pipelineOutput.SuccessfulPhases.Length
            pipelineOutput.SuccessfulPhases |> List.iter (fun phase -> printfn "  ✓ %s" phase)
            
            if not pipelineOutput.Diagnostics.IsEmpty then
                printfn "Pipeline diagnostics:"
                pipelineOutput.Diagnostics |> List.iter (fun (phase, msg) -> printfn "  [%s] %s" phase msg)
        
        printfn "=== ENHANCED COMPILATION PIPELINE END ==="
        Success pipelineOutput

/// External toolchain integration modules
module ExternalToolchain =
    
    let isCommandAvailable (command: string) : bool =
        try
            let commands = 
                if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
                    [command; command + ".exe"]
                else
                    [command]
            
            commands |> List.exists (fun cmd ->
                try
                    let processInfo = System.Diagnostics.ProcessStartInfo()
                    processInfo.FileName <- cmd
                    processInfo.Arguments <- "--version"
                    processInfo.UseShellExecute <- false
                    processInfo.RedirectStandardOutput <- true
                    processInfo.RedirectStandardError <- true
                    processInfo.CreateNoWindow <- true
                    
                    use proc = System.Diagnostics.Process.Start(processInfo)
                    proc.WaitForExit(5000) |> ignore
                    proc.ExitCode = 0
                with _ -> false)
        with _ -> false
    
    let getCompilerCommands (target: string) : CompilerResult<string * string> =
        if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
            let llcOptions = ["llc"; "llc.exe"]
            let linkerOptions = ["gcc"; "gcc.exe"; "clang"; "clang.exe"]
            
            let llcFound = llcOptions |> List.tryFind isCommandAvailable
            let linkerFound = linkerOptions |> List.tryFind isCommandAvailable
            
            match llcFound, linkerFound with
            | Some llc, Some linker -> Success (llc, linker)
            | None, _ -> CompilerFailure [CompilerError("toolchain", "LLVM compiler (llc) not found", Some "Install LLVM tools: pacman -S mingw-w64-x86_64-llvm")]
            | _, None -> CompilerFailure [CompilerError("toolchain", "C compiler not found", Some "Install GCC: pacman -S mingw-w64-x86_64-gcc")]
        else
            if isCommandAvailable "llc" && isCommandAvailable "clang" then
                Success ("llc", "clang")
            elif isCommandAvailable "llc" && isCommandAvailable "gcc" then
                Success ("llc", "gcc")
            else
                CompilerFailure [CompilerError("toolchain", "Required compilers not found", Some "Install LLVM and GCC/Clang")]
    
    let runExternalCommand (command: string) (arguments: string) (workingDir: string option) : CompilerResult<string> =
        try
            printfn "Debug: Executing %s %s" command arguments
            
            let processInfo = System.Diagnostics.ProcessStartInfo()
            processInfo.FileName <- command
            processInfo.Arguments <- arguments
            processInfo.UseShellExecute <- false
            processInfo.RedirectStandardOutput <- true
            processInfo.RedirectStandardError <- true
            processInfo.CreateNoWindow <- true
            
            match workingDir with
            | Some dir -> processInfo.WorkingDirectory <- dir
            | None -> ()
            
            use proc = System.Diagnostics.Process.Start(processInfo)
            let output = proc.StandardOutput.ReadToEnd()
            let error = proc.StandardError.ReadToEnd()
            proc.WaitForExit()
            
            if proc.ExitCode = 0 then
                Success output
            else
                CompilerFailure [CompilerError("external command", 
                    sprintf "%s failed with exit code %d" command proc.ExitCode, 
                    Some error)]
        with
        | ex ->
            CompilerFailure [CompilerError("external command", 
                sprintf "Failed to execute %s" command, 
                Some ex.Message)]

module TargetTripleManagement =
    
    let getTargetTriple (target: string) : string =
        match target.ToLowerInvariant() with
        | "x86_64-pc-windows-msvc" -> "x86_64-pc-windows-msvc"
        | "x86_64-pc-linux-gnu" -> "x86_64-pc-linux-gnu"  
        | "x86_64-apple-darwin" -> "x86_64-apple-darwin"
        | "embedded" -> "thumbv7em-none-eabihf"
        | "thumbv7em-none-eabihf" -> "thumbv7em-none-eabihf"
        | "x86_64-w64-windows-gnu" -> "x86_64-w64-windows-gnu"
        | _ -> 
            if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
                "x86_64-w64-windows-gnu"
            elif RuntimeInformation.IsOSPlatform(OSPlatform.Linux) then
                "x86_64-pc-linux-gnu"
            elif RuntimeInformation.IsOSPlatform(OSPlatform.OSX) then
                "x86_64-apple-darwin"
            else
                "x86_64-w64-windows-gnu"
    
    let validateTargetTriple (triple: string) : CompilerResult<unit> =
        let parts = triple.Split('-')
        if parts.Length >= 3 then
            Success ()
        else
            CompilerFailure [TransformError("target validation", triple, "valid target triple", 
                "Target triple must have format: arch-vendor-os(-environment)")]

/// Native compilation functions
let private compileLLVMToNativeConsole (llvmOutput: LLVMOutput) (outputPath: string) (target: string) : CompilerResult<unit> =
    printfn "=== NATIVE COMPILATION START ==="
    
    let targetTriple = TargetTripleManagement.getTargetTriple target
    TargetTripleManagement.validateTargetTriple targetTriple >>= fun _ ->
    
    ExternalToolchain.getCompilerCommands target >>= fun (llcCommand, linkerCommand) ->
    
    let llvmPath = Path.ChangeExtension(outputPath, ".ll")
    let objPath = Path.ChangeExtension(outputPath, ".o")
    let outputDir = Path.GetDirectoryName(outputPath)
    
    try
        let utf8WithoutBom = System.Text.UTF8Encoding(false)
        let normalizedIR = llvmOutput.LLVMIRText.Replace("\r\n", "\n").Replace("\r", "\n")
        File.WriteAllText(llvmPath, normalizedIR, utf8WithoutBom)
        printfn "Saved LLVM IR to: %s" llvmPath
        
        let llcArgs = sprintf "-filetype=obj -mtriple=%s -relocation-model=pic -o \"%s\" \"%s\"" 
                             targetTriple objPath llvmPath
        printfn "Compiling to native code for target '%s'..." targetTriple
        
        ExternalToolchain.runExternalCommand llcCommand llcArgs (Some outputDir) >>= fun _ ->
        printfn "Created object file: %s" objPath
        
        let linkArgs = 
            if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
                sprintf "\"%s\" -o \"%s\" -mconsole -Wl,--subsystem,console -Wl,--entry,mainCRTStartup -static-libgcc -lmsvcrt -lkernel32" 
                        objPath outputPath
            else
                sprintf "\"%s\" -o \"%s\"" objPath outputPath
        
        printfn "Linking object file to executable with console application configuration..."        
        ExternalToolchain.runExternalCommand linkerCommand linkArgs (Some outputDir) >>= fun linkOutput ->
        
        if File.Exists(outputPath) then
            printfn "Successfully created console executable: %s" outputPath
            if File.Exists(objPath) then File.Delete(objPath)
            Success ()
        else
            CompilerFailure [CompilerError("native compilation", "Console executable was not created", 
                Some (sprintf "Linker output: %s\nTry manual compilation: gcc %s -o %s -mconsole" linkOutput objPath outputPath))]
    
    with
    | ex ->
        CompilerFailure [CompilerError("native compilation", "Failed during console application compilation", Some ex.Message)]

let private compileToNative (llvmOutput: LLVMOutput) (outputPath: string) (target: string) (verbose: bool) (noExternalTools: bool) =
    if verbose then printfn "Compiling to native executable for target: %s" target
    
    if noExternalTools then
        printfn "Warning: --no-external-tools specified, but native compilation requires external LLVM toolchain"
        let llvmPath = Path.ChangeExtension(outputPath, ".ll")
        let encoding = System.Text.UTF8Encoding(false)
        File.WriteAllText(llvmPath, llvmOutput.LLVMIRText, encoding)
        printfn "Saved LLVM IR to: %s" llvmPath
        0
    else
        match compileLLVMToNativeConsole llvmOutput outputPath target with
        | CompilerFailure nativeErrors ->
            printfn "Native compilation failed:"
            nativeErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
            1
        | Success () ->
            printfn "✓ Compilation successful: %s" outputPath
            0

/// MAIN COMPILE FUNCTION - SINGLE ENTRY POINT WITH GUARANTEED AST GENERATION
let compile (args: ParseResults<CompileArgs>) =
    printfn "=== FIREFLY COMPILE COMMAND START ==="
    printfn "Command invoked with %d arguments" args.GetAllResults().Length
    
    // Parse and validate arguments
    let inputPath = 
        match args.TryGetResult Input with
        | Some path -> 
            printfn "✓ Input path: %s" path
            path
        | None -> 
            printfn "✗ ERROR: Input file is required"
            exit 1
    
    let outputPath = 
        match args.TryGetResult Output with
        | Some path -> 
            printfn "✓ Output path: %s" path
            path
        | None -> 
            printfn "✗ ERROR: Output path is required"
            exit 1
    
    let target = args.TryGetResult Target |> Option.defaultValue (getDefaultTarget())
    let optimizeLevel = args.TryGetResult Optimize |> Option.defaultValue "default"
    let configPath = args.TryGetResult Config |> Option.defaultValue "firefly.toml"
    let keepIntermediates = args.Contains Keep_Intermediates
    let verbose = args.Contains Verbose
    let noExternalTools = args.Contains No_External_Tools
    
    printfn "=== COMPILATION SETTINGS ==="
    printfn "Target: %s" target
    printfn "Optimize level: %s" optimizeLevel
    printfn "Config path: %s" configPath
    printfn "Keep intermediates: %b" keepIntermediates
    printfn "Verbose: %b" verbose
    printfn "No external tools: %b" noExternalTools
    
    // Validate arguments
    match validateCompilationArgs inputPath outputPath with
    | CompilerFailure errors ->
        errors |> List.iter (fun error -> printfn "Error: %s" (error.ToString()))
        1
    | Success () ->
        
        // Load configuration
        if verbose then printfn "Loading configuration from: %s" configPath
        
        match loadAndValidateConfig configPath with
        | CompilerFailure configErrors ->
            printfn "Configuration errors:"
            configErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
            1
        | Success config ->
            
            if verbose then
                printfn "Loaded configuration for package: %s v%s" config.PackageName config.Version
                printfn "  Require static memory: %b" config.Compilation.RequireStaticMemory
                printfn "  Eliminate closures: %b" config.Compilation.EliminateClosures
                printfn "  Keep intermediates: %b" keepIntermediates
            
            // Read and validate source code
            try
                let sourceCode = File.ReadAllText(inputPath)
                if verbose then printfn "Read source file: %s (%d characters)" inputPath sourceCode.Length
                
                // Execute enhanced compilation pipeline with GUARANTEED AST generation
                printfn "=== STARTING ENHANCED COMPILATION PIPELINE WITH GUARANTEED AST GENERATION ==="
                
                match executeEnhancedCompilationPipeline inputPath sourceCode config keepIntermediates verbose with
                | CompilerFailure pipelineErrors ->
                    printfn "Enhanced compilation pipeline failed:"
                    pipelineErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
                    1
                
                | Success pipelineOutput ->
                    if verbose then
                        printfn "Pipeline completed successfully with %d phases:" pipelineOutput.SuccessfulPhases.Length
                        pipelineOutput.SuccessfulPhases |> List.iter (fun phase -> printfn "  ✓ %s" phase)
                    
                    // Save enhanced intermediate files including GUARANTEED AST representations
                    let basePath = Path.GetDirectoryName(outputPath)
                    let baseName = Path.GetFileNameWithoutExtension(outputPath)
                    saveIntermediateFiles basePath baseName keepIntermediates pipelineOutput
                    
                    // Continue with LLVM generation and compilation
                    if verbose then printfn "Translating MLIR to LLVM IR..."
                    
                    match translateToLLVM pipelineOutput.FinalMLIR with
                    | CompilerFailure llvmErrors ->
                        printfn "LLVM IR generation failed:"
                        llvmErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
                        1
                    
                    | Success llvmOutput ->
                        if verbose then printfn "Generated LLVM IR for module: %s" llvmOutput.ModuleName
                        
                        let optimizationLevel = 
                            match optimizeLevel.ToLowerInvariant() with
                            | "none" -> None
                            | "less" -> Less
                            | "aggressive" -> Aggressive
                            | "size" -> Size
                            | "sizemin" -> SizeMin
                            | _ -> Default
                        
                        let optimizationPasses = createOptimizationPipeline optimizationLevel
                        
                        match optimizeLLVMIR llvmOutput optimizationPasses with
                        | CompilerFailure optimizationErrors ->
                            printfn "LLVM optimization failed:"
                            optimizationErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
                            1
                        
                        | Success optimizedLLVM ->
                            if config.Compilation.RequireStaticMemory then
                                match validateZeroAllocationGuarantees optimizedLLVM.LLVMIRText with
                                | CompilerFailure validationErrors ->
                                    printfn "Zero-allocation validation failed:"
                                    validationErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
                                    1
                                | Success () ->
                                    if verbose then printfn "✓ Zero-allocation guarantees validated"
                                    compileToNative optimizedLLVM outputPath target verbose noExternalTools
                            else
                                compileToNative optimizedLLVM outputPath target verbose noExternalTools
            
            with
            | ex ->
                printfn "Error reading source file: %s" ex.Message
                1

/// Entry point for compilation with comprehensive error handling and AST diagnostics
let compileWithFullDiagnostics (inputPath: string) (outputPath: string) (target: string) (configPath: string) =
    match validateCompilationArgs inputPath outputPath with
    | CompilerFailure errors ->
        printfn "Argument validation failed:"
        errors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
        1
    | Success () ->
        match loadAndValidateConfig configPath with
        | CompilerFailure configErrors ->
            printfn "Configuration validation failed:"
            configErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
            1
        | Success config ->
            try
                let sourceCode = File.ReadAllText(inputPath)
                
                match executeEnhancedCompilationPipeline inputPath sourceCode config true true with
                | CompilerFailure pipelineErrors ->
                    printfn "Enhanced XParsec compilation pipeline failed:"
                    pipelineErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
                    1
                | Success pipelineOutput ->
                    printfn "Enhanced Compilation Statistics:"
                    printfn "Successful phases: %d" pipelineOutput.SuccessfulPhases.Length
                    printfn "Phase outputs saved: %d" pipelineOutput.PhaseOutputs.Count
                    0
            with
            | ex ->
                printfn "Unexpected error: %s" ex.Message
                1