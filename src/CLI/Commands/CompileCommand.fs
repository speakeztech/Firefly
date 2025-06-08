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

/// Saves all intermediate files from the compilation pipeline
let private saveIntermediateFiles (outputPath: string) (pipelineOutput: TranslationPipelineOutput) (llvmIR: string option) =
    let baseName = Path.GetFileNameWithoutExtension(outputPath)
    let dir = Path.GetDirectoryName(outputPath)
    
    printfn "[SAVE-DEBUG] Starting intermediate file save process"
    printfn "[SAVE-DEBUG] Output path: %s" outputPath
    printfn "[SAVE-DEBUG] Base name: %s" baseName
    printfn "[SAVE-DEBUG] Directory: %s" dir
    printfn "[SAVE-DEBUG] Pipeline output phase count: %d" pipelineOutput.PhaseOutputs.Count
    printfn "[SAVE-DEBUG] Available phase output keys: %A" (Map.keys pipelineOutput.PhaseOutputs |> Seq.toList)
    
    // Check each specific key in detail
    match pipelineOutput.PhaseOutputs.TryFind "f#-compiler-services-ast" with
    | Some content -> 
        printfn "[SAVE-DEBUG] Found FCS AST content: %d characters" content.Length
        if String.IsNullOrWhiteSpace(content) then
            printfn "[SAVE-DEBUG] WARNING: FCS AST content is null or whitespace"
        else
            printfn "[SAVE-DEBUG] FCS AST content preview: %s" (content.Substring(0, min 50 content.Length))
    | None -> 
        printfn "[SAVE-DEBUG] FCS AST content NOT FOUND in pipeline outputs"
    
    match pipelineOutput.PhaseOutputs.TryFind "oak-ast" with
    | Some content -> 
        printfn "[SAVE-DEBUG] Found Oak AST content: %d characters" content.Length
        if String.IsNullOrWhiteSpace(content) then
            printfn "[SAVE-DEBUG] WARNING: Oak AST content is null or whitespace"
        else
            printfn "[SAVE-DEBUG] Oak AST content preview: %s" (content.Substring(0, min 50 content.Length))
    | None -> 
        printfn "[SAVE-DEBUG] Oak AST content NOT FOUND in pipeline outputs"
    
    match pipelineOutput.PhaseOutputs.TryFind "mlir" with
    | Some content -> 
        printfn "[SAVE-DEBUG] Found MLIR content: %d characters" content.Length
    | None -> 
        printfn "[SAVE-DEBUG] MLIR content NOT FOUND in pipeline outputs"
    
    let filesToSave = [
        // AST files
        match pipelineOutput.PhaseOutputs.TryFind "f#-compiler-services-ast" with
        | Some content when not (String.IsNullOrWhiteSpace(content)) -> 
            printfn "[SAVE-DEBUG] Adding FCS AST file to save list"
            Some ("FCS AST", Path.Combine(dir, baseName + ".fcs"), content)
        | Some content -> 
            printfn "[SAVE-DEBUG] FCS AST content is empty, skipping file creation"
            None
        | None -> 
            printfn "[SAVE-DEBUG] No FCS AST content available"
            None
        
        match pipelineOutput.PhaseOutputs.TryFind "oak-ast" with
        | Some content when not (String.IsNullOrWhiteSpace(content)) -> 
            printfn "[SAVE-DEBUG] Adding Oak AST file to save list"
            Some ("Oak AST", Path.Combine(dir, baseName + ".oak"), content)
        | Some content -> 
            printfn "[SAVE-DEBUG] Oak AST content is empty, skipping file creation"
            None
        | None -> 
            printfn "[SAVE-DEBUG] No Oak AST content available"
            None
        
        // MLIR file
        match pipelineOutput.PhaseOutputs.TryFind "mlir" with
        | Some content when not (String.IsNullOrWhiteSpace(content)) -> 
            printfn "[SAVE-DEBUG] Adding MLIR file to save list"
            Some ("MLIR", Path.Combine(dir, baseName + ".mlir"), content)
        | Some content -> 
            printfn "[SAVE-DEBUG] MLIR content is empty, skipping file creation"
            None
        | None -> 
            printfn "[SAVE-DEBUG] No MLIR content available"
            None
        
        // LLVM IR file (if provided)
        match llvmIR with
        | Some content when not (String.IsNullOrWhiteSpace(content)) -> 
            printfn "[SAVE-DEBUG] Adding LLVM IR file to save list"
            Some ("LLVM IR", Path.Combine(dir, baseName + ".ll"), content)
        | Some content -> 
            printfn "[SAVE-DEBUG] LLVM IR content is empty, skipping file creation"
            None
        | None -> 
            printfn "[SAVE-DEBUG] No LLVM IR content provided"
            None
    ]
    |> List.choose id
    
    printfn "[SAVE-DEBUG] Total files to save: %d" filesToSave.Length
    printfn "[SAVE-DEBUG] Files to save: %A" (filesToSave |> List.map (fun (desc, path, _) -> sprintf "%s -> %s" desc (Path.GetFileName(path))))
    
    // Save all files
    filesToSave |> List.iter (fun (desc, path, content) ->
        try
            printfn "[SAVE-DEBUG] Attempting to save %s to %s (%d chars)" desc path content.Length
            writeTextFile path content
            printfn "[SAVE-DEBUG] Successfully saved %s" desc
        with
        | ex ->
            printfn "[SAVE-DEBUG] ERROR saving %s: %s" desc ex.Message
            printfn "Warning: Failed to save %s: %s" desc ex.Message
    )
    
    // Report what was saved
    if not filesToSave.IsEmpty then
        printfn "Saved intermediate files:"
        filesToSave |> List.iter (fun (desc, path, _) ->
            printfn "  %s: %s" desc (Path.GetFileName(path))
        )
    else
        printfn "[SAVE-DEBUG] WARNING: No intermediate files were saved!"

/// Validates compilation arguments
let private validateCompilationArgs (inputPath: string) (outputPath: string) : CompilerResult<unit> =
    if not (File.Exists inputPath) then
        CompilerFailure [CompilerError("argument validation", sprintf "Input file '%s' does not exist" inputPath, None)]
    elif String.IsNullOrWhiteSpace(outputPath) then
        CompilerFailure [CompilerError("argument validation", "Output path cannot be empty", None)]
    else
        let inputExt = Path.GetExtension(inputPath).ToLowerInvariant()
        if inputExt <> ".fs" && inputExt <> ".fsx" then
            CompilerFailure [CompilerError("argument validation", sprintf "Input file must be F# source (.fs or .fsx), got: %s" inputExt, None)]
        else
            Success ()

/// External toolchain integration
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

/// Target triple management
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
let private compileLLVMToNative (llvmOutput: LLVMOutput) (outputPath: string) (target: string) (verbose: bool) : CompilerResult<unit> =
    if verbose then printfn "Compiling to native executable for target: %s" target
    
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
        
        printfn "Linking object file to executable..."        
        ExternalToolchain.runExternalCommand linkerCommand linkArgs (Some outputDir) >>= fun linkOutput ->
        
        if File.Exists(outputPath) then
            printfn "Successfully created executable: %s" outputPath
            if File.Exists(objPath) then File.Delete(objPath)
            Success ()
        else
            CompilerFailure [CompilerError("native compilation", "Executable was not created", 
                Some (sprintf "Linker output: %s" linkOutput))]
    
    with
    | ex ->
        CompilerFailure [CompilerError("native compilation", "Failed during compilation", Some ex.Message)]

/// Main compile function - single path with clear flow
let compile (args: ParseResults<CompileArgs>) =
    // Parse arguments
    let inputPath = 
        match args.TryGetResult Input with
        | Some path -> path
        | None -> 
            printfn "Error: Input file is required"
            exit 1
    
    let outputPath = 
        match args.TryGetResult Output with
        | Some path -> path
        | None -> 
            printfn "Error: Output path is required"
            exit 1
    
    let target = args.TryGetResult Target |> Option.defaultValue (getDefaultTarget())
    let optimizeLevel = args.TryGetResult Optimize |> Option.defaultValue "default"
    let configPath = args.TryGetResult Config |> Option.defaultValue "firefly.toml"
    let keepIntermediates = args.Contains Keep_Intermediates
    let verbose = args.Contains Verbose
    let noExternalTools = args.Contains No_External_Tools
    
    printfn "[COMPILE-DEBUG] Starting compilation with keep intermediates: %b" keepIntermediates
    
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
            
            // Read source code
            try
                let sourceCode = File.ReadAllText(inputPath)
                if verbose then printfn "Read source file: %s (%d characters)" inputPath sourceCode.Length
                
                // Execute compilation pipeline
                printfn "[COMPILE-DEBUG] Calling translateFSharpToMLIRWithIntermediates with keepIntermediates: %b" keepIntermediates
                match translateFSharpToMLIRWithIntermediates inputPath sourceCode keepIntermediates with
                | CompilerFailure pipelineErrors ->
                    printfn "Error: Compilation failed"
                    pipelineErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
                    1
                
                | Success pipelineOutput ->
                    printfn "[COMPILE-DEBUG] Pipeline completed, phase outputs count: %d" pipelineOutput.PhaseOutputs.Count
                    
                    // Continue with LLVM generation
                    printfn "Translating to LLVM IR..."
                    match translateToLLVM pipelineOutput.FinalMLIR with
                    | CompilerFailure llvmErrors ->
                        printfn "LLVM IR generation failed:"
                        llvmErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
                        1
                    
                    | Success llvmOutput ->
                        if verbose then printfn "Generated LLVM IR for module: %s" llvmOutput.ModuleName
                        
                        // Apply optimizations
                        let optimizationLevel = 
                            match optimizeLevel.ToLowerInvariant() with
                            | "none" -> None
                            | "less" -> Less
                            | "aggressive" -> Aggressive
                            | "size" -> Size
                            | "sizemin" -> SizeMin
                            | _ -> Default
                        
                        printfn "Applying LLVM optimizations (%A)..." optimizationLevel
                        let optimizationPasses = createOptimizationPipeline optimizationLevel
                        
                        match optimizeLLVMIR llvmOutput optimizationPasses with
                        | CompilerFailure optimizationErrors ->
                            printfn "LLVM optimization failed:"
                            optimizationErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
                            1
                        
                        | Success optimizedLLVM ->
                            // Save all intermediate files if requested
                            if keepIntermediates then
                                printfn "[COMPILE-DEBUG] Saving intermediate files..."
                                saveIntermediateFiles outputPath pipelineOutput (Some optimizedLLVM.LLVMIRText)
                            else
                                printfn "[COMPILE-DEBUG] Skipping intermediate file save (keepIntermediates = false)"
                            
                            // Validate zero-allocation if required
                            if config.Compilation.RequireStaticMemory then
                                match validateZeroAllocationGuarantees optimizedLLVM.LLVMIRText with
                                | CompilerFailure validationErrors ->
                                    printfn "Zero-allocation validation failed:"
                                    validationErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
                                    1
                                | Success () ->
                                    if verbose then printfn "Zero-allocation guarantees validated"
                                    
                                    // Compile to native
                                    if noExternalTools then
                                        printfn "Warning: --no-external-tools specified, skipping native compilation"
                                        let llvmPath = Path.ChangeExtension(outputPath, ".ll")
                                        File.WriteAllText(llvmPath, optimizedLLVM.LLVMIRText)
                                        printfn "Saved LLVM IR to: %s" llvmPath
                                        0
                                    else
                                        match compileLLVMToNative optimizedLLVM outputPath target verbose with
                                        | CompilerFailure nativeErrors ->
                                            printfn "Linking failed: %s" (nativeErrors |> List.map (fun e -> e.ToString()) |> String.concat "; ")
                                            printfn "Error: Compilation failed"
                                            1
                                        | Success () ->
                                            printfn "Compilation successful: %s" outputPath
                                            0
                            else
                                // Compile without validation
                                if noExternalTools then
                                    printfn "Warning: --no-external-tools specified, skipping native compilation"
                                    let llvmPath = Path.ChangeExtension(outputPath, ".ll")
                                    File.WriteAllText(llvmPath, optimizedLLVM.LLVMIRText)
                                    printfn "Saved LLVM IR to: %s" llvmPath
                                    0
                                else
                                    match compileLLVMToNative optimizedLLVM outputPath target verbose with
                                    | CompilerFailure nativeErrors ->
                                        printfn "Linking failed: %s" (nativeErrors |> List.map (fun e -> e.ToString()) |> String.concat "; ")
                                        printfn "Error: Compilation failed"
                                        1
                                    | Success () ->
                                        printfn "Compilation successful: %s" outputPath
                                        0
            
            with
            | ex ->
                printfn "Error reading source file: %s" ex.Message
                if verbose then printfn "Stack trace: %s" ex.StackTrace
                1