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

/// Enhanced intermediate file saving with FCS and Oak AST support
let private saveIntermediateFiles (basePath: string) (baseName: string) (keepIntermediates: bool) (pipelineOutput: TranslationPipelineOutput) =
    if keepIntermediates then
        try
            let intermediatesDir = Path.Combine(basePath, "intermediates")
            if not (Directory.Exists(intermediatesDir)) then
                Directory.CreateDirectory(intermediatesDir) |> ignore
            
            // Cross-platform text file writing with consistent encoding
            // NOTE: Using UTF-8 without BOM for maximum compatibility across Windows, Linux, and macOS
            // Future platform considerations:
            // - All MLIR/LLVM tools expect UTF-8 encoding
            // - Unix line endings (\n) provide best cross-platform compatibility
            // - VSCode and most editors handle UTF-8 without BOM correctly on all platforms
            let writeTextFile (filePath: string) (content: string) =
                let encoding = System.Text.UTF8Encoding(false) // false = no BOM
                let normalizedContent = content.Replace("\r\n", "\n").Replace("\r", "\n")
                File.WriteAllText(filePath, normalizedContent, encoding)
            
            // Save pipeline diagnostics
            let diagnosticsPath = Path.Combine(intermediatesDir, baseName + ".diagnostics.txt")
            let diagnosticsContent = 
                pipelineOutput.Diagnostics
                |> List.map (fun (phase, message) -> sprintf "[%s] %s" phase message)
                |> String.concat "\n"
            writeTextFile diagnosticsPath diagnosticsContent
            
            // Save all intermediate outputs with proper extensions
            pipelineOutput.PhaseOutputs
            |> Map.iter (fun phaseName output ->
                let extension = 
                    match phaseName with
                    | name when name.Contains("mlir") -> ".mlir"
                    | name when name.Contains("oak") -> ".oak"
                    | name when name.Contains("fcs") -> ".fcs"
                    | name when name.Contains("llvm") -> ".ll"
                    | _ -> ".txt"
                let fileName = sprintf "%s.%s%s" baseName (phaseName.Replace("-", "_")) extension
                let filePath = Path.Combine(intermediatesDir, fileName)
                writeTextFile filePath output)
            
            // Save symbol mappings
            let symbolMappingsPath = Path.Combine(intermediatesDir, baseName + ".symbols.txt")
            let symbolMappingsContent = 
                pipelineOutput.SymbolMappings
                |> Map.toList
                |> List.map (fun (original, transformed) -> sprintf "%s -> %s" original transformed)
                |> String.concat "\n"
            writeTextFile symbolMappingsPath symbolMappingsContent
            
            // Report saved files with proper formatting for cross-platform paths
            let displayPath = intermediatesDir.Replace(Path.DirectorySeparatorChar, '/')
            printfn "Saved intermediate files to: %s" displayPath
            
            // List the specific files that were created for developer visibility
            let intermediateFiles = Directory.GetFiles(intermediatesDir, baseName + ".*")
            if intermediateFiles.Length > 0 then
                printfn "Generated intermediate files:"
                intermediateFiles |> Array.iter (fun file ->
                    let fileName = Path.GetFileName(file)
                    let fileSize = (FileInfo(file)).Length
                    printfn "  %s (%d bytes)" fileName fileSize)
            
        with
        | ex ->
            printfn "Warning: Failed to save intermediate files: %s" ex.Message

/// Validates compilation arguments with enhanced error reporting
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

/// Enhanced XParsec-based compilation process with intermediate AST generation
let private executeEnhancedCompilationPipeline (inputPath: string) (sourceCode: string) (config: FireflyConfig) (keepIntermediates: bool) (verbose: bool) : CompilerResult<TranslationPipelineOutput> =
    let basePath = Path.GetDirectoryName(inputPath)
    let baseName = Path.GetFileNameWithoutExtension(inputPath)
    
    if verbose then printfn "Starting enhanced XParsec-based compilation pipeline with AST intermediate generation..."
    
    // Generate F# Compiler Services and Oak AST representations
    printfn "Parsing F# source code..."
    let parsingResult = Dabbit.Parsing.AstConverter.parseAndConvertToOakAstWithIntermediate sourceCode
    
    if keepIntermediates then
        try
            let intermediatesDir = Path.Combine(basePath, "intermediates")
            if not (Directory.Exists(intermediatesDir)) then
                Directory.CreateDirectory(intermediatesDir) |> ignore
            
            // Save FCS AST using cross-platform compatible writing
            let fcsPath = Path.Combine(intermediatesDir, baseName + ".fcs")
            let encoding = System.Text.UTF8Encoding(false)
            let normalizedFCS = parsingResult.FCSAstText.Replace("\r\n", "\n").Replace("\r", "\n")
            File.WriteAllText(fcsPath, normalizedFCS, encoding)
            
            // Save Oak AST using cross-platform compatible writing
            let oakPath = Path.Combine(intermediatesDir, baseName + ".oak")
            let normalizedOak = parsingResult.OakAstText.Replace("\r\n", "\n").Replace("\r", "\n")
            File.WriteAllText(oakPath, normalizedOak, encoding)
            
            if verbose then
                printfn "Saved F# Compiler Services AST to: %s" (fcsPath.Replace(Path.DirectorySeparatorChar, '/'))
                printfn "Saved Oak AST to: %s" (oakPath.Replace(Path.DirectorySeparatorChar, '/'))
        
        with
        | ex ->
            printfn "Warning: Failed to save AST intermediate files: %s" ex.Message
    
    // Continue with existing compilation pipeline using the Oak program
    match translateF#ToMLIRWithDiagnostics inputPath sourceCode with
    | CompilerFailure pipelineErrors ->
        CompilerFailure pipelineErrors
    | Success pipelineOutput ->
        // Enhanced pipeline output that includes AST information
        let enhancedOutput = {
            pipelineOutput with
                PhaseOutputs = 
                    pipelineOutput.PhaseOutputs
                    |> Map.add "f#-compiler-services-ast" parsingResult.FCSAstText
                    |> Map.add "oak-ast" parsingResult.OakAstText
        }
        Success enhancedOutput

/// Executes the complete XParsec-based compilation process with enhanced AST diagnostics
let compile (args: ParseResults<CompileArgs>) =
    // Parse and validate arguments
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
    
    // Validate arguments using XParsec patterns
    match validateCompilationArgs inputPath outputPath with
    | CompilerFailure errors ->
        errors |> List.iter (fun error -> printfn "Error: %s" (error.ToString()))
        1
    | Success () ->
        
        // Load and validate configuration using XParsec-based parser
        if verbose then printfn "Loading configuration from: %s" configPath
        
        match loadAndValidateConfig configPath with
        | CompilerFailure configErrors ->
            printfn "Configuration errors:"
            configErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
            1
        | Success config ->
            
            if verbose then
                printfn "Loaded configuration for package: %s v%s" config.PackageName config.Version
                printfn "Compilation settings:"
                printfn "  Require static memory: %b" config.Compilation.RequireStaticMemory
                printfn "  Eliminate closures: %b" config.Compilation.EliminateClosures
                printfn "  Optimization level: %s" config.Compilation.OptimizationLevel
                printfn "  Use LTO: %b" config.Compilation.UseLTO
                printfn "  Keep intermediates: %b" keepIntermediates
            
            // Read and validate source code
            try
                let sourceCode = File.ReadAllText(inputPath)
                if verbose then printfn "Read source file: %s (%d characters)" inputPath sourceCode.Length
                
                // Execute enhanced compilation pipeline with AST generation
                if verbose then printfn "Starting enhanced XParsec-based compilation pipeline..."
                
                match executeEnhancedCompilationPipeline inputPath sourceCode config keepIntermediates verbose with
                | CompilerFailure pipelineErrors ->
                    printfn "Enhanced compilation pipeline failed:"
                    pipelineErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
                    1
                
                | Success pipelineOutput ->
                    if verbose then
                        printfn "Pipeline completed successfully with %d phases:" pipelineOutput.SuccessfulPhases.Length
                        pipelineOutput.SuccessfulPhases |> List.iter (fun phase -> printfn "  ✓ %s" phase)
                        
                        if not pipelineOutput.Diagnostics.IsEmpty then
                            printfn "Pipeline diagnostics:"
                            pipelineOutput.Diagnostics |> List.iter (fun (phase, msg) -> printfn "  [%s] %s" phase msg)
                    
                    // Save enhanced intermediate files including AST representations
                    let basePath = Path.GetDirectoryName(outputPath)
                    let baseName = Path.GetFileNameWithoutExtension(outputPath)
                    saveIntermediateFiles basePath baseName keepIntermediates pipelineOutput
                    
                    // Apply LLVM optimization pipeline
                    if verbose then printfn "Translating MLIR to LLVM IR..."
                    
                    match translateToLLVM pipelineOutput.FinalMLIR with
                    | CompilerFailure llvmErrors ->
                        printfn "LLVM IR generation failed:"
                        llvmErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
                        1
                    
                    | Success llvmOutput ->
                        if verbose then printfn "Generated LLVM IR for module: %s" llvmOutput.ModuleName
                        
                        // Apply optimizations based on configuration and arguments
                        let optimizationLevel = 
                            match optimizeLevel.ToLowerInvariant() with
                            | "none" -> None
                            | "less" -> Less
                            | "aggressive" -> Aggressive
                            | "size" -> Size
                            | "sizemin" -> SizeMin
                            | _ -> Default
                        
                        let optimizationPasses = createOptimizationPipeline optimizationLevel
                        
                        if verbose then 
                            printfn "Applying LLVM optimizations (level: %A)..." optimizationLevel
                            if optimizationPasses.IsEmpty then
                                printfn "  No optimization passes selected"
                            else
                                printfn "  Optimization passes: %A" optimizationPasses
                        
                        match optimizeLLVMIR llvmOutput optimizationPasses with
                        | CompilerFailure optimizationErrors ->
                            printfn "LLVM optimization failed:"
                            optimizationErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
                            1
                        
                        | Success optimizedLLVM ->
                            // Validate zero-allocation guarantees
                            if config.Compilation.RequireStaticMemory then
                                if verbose then printfn "Validating zero-allocation guarantees..."
                                
                                match validateZeroAllocationGuarantees optimizedLLVM.LLVMIRText with
                                | CompilerFailure validationErrors ->
                                    printfn "Zero-allocation validation failed:"
                                    validationErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
                                    printfn "Consider disabling require_static_memory in configuration if heap allocation is acceptable"
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

/// Enhanced native compilation with improved MinGW64 console application support
and private compileToNative (llvmOutput: LLVMOutput) (outputPath: string) (target: string) (verbose: bool) (noExternalTools: bool) =
    if verbose then printfn "Compiling to native executable for target: %s" target
    
    if noExternalTools then
        printfn "Warning: --no-external-tools specified, but native compilation requires external LLVM toolchain"
        printfn "The XParsec pipeline has generated optimized LLVM IR, but cannot complete native compilation"
        printfn "To complete compilation, remove --no-external-tools flag or use external LLVM tools"
        
        // Save the LLVM IR for manual compilation
        let llvmPath = Path.ChangeExtension(outputPath, ".ll")
        let encoding = System.Text.UTF8Encoding(false)
        File.WriteAllText(llvmPath, llvmOutput.LLVMIRText, encoding)
        printfn "Saved LLVM IR to: %s" llvmPath
        printfn "To compile manually: llc -filetype=obj %s -o %s.o && gcc %s.o -o %s -mconsole" 
                 llvmPath (Path.GetFileNameWithoutExtension(outputPath))
                 (Path.GetFileNameWithoutExtension(outputPath)) outputPath
        0
    else
        match compileLLVMToNativeConsole llvmOutput outputPath target with
        | CompilerFailure nativeErrors ->
            printfn "Native compilation failed:"
            nativeErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
            printfn ""
            printfn "Possible solutions:"
            printfn "  1. Install LLVM tools (llc, clang/gcc) in your PATH"
            printfn "  2. Use --no-external-tools to generate LLVM IR only"
            printfn "  3. Check that target '%s' is supported by your LLVM installation" target
            printfn "  4. Ensure you're using MSYS2 MINGW64 terminal on Windows"
            1
        
        | Success () ->
            printfn "✓ Compilation successful: %s" outputPath
            
            if verbose then
                try
                    let fileInfo = FileInfo(outputPath)
                    printfn "  Output size: %d bytes" fileInfo.Length
                    printfn "  Target: %s" target
                    printfn "  Uses zero-allocation Firefly runtime"
                    printfn "  Console application with proper main entry point"
                with
                | _ -> ()
            
            0

/// Enhanced native compilation specifically for console applications with MinGW64
and private compileLLVMToNativeConsole (llvmOutput: LLVMOutput) (outputPath: string) (target: string) : CompilerResult<unit> =
    let targetTriple = TargetTripleManagement.getTargetTriple target
    TargetTripleManagement.validateTargetTriple targetTriple >>= fun _ ->
    
    ExternalToolchain.getCompilerCommands target >>= fun (llcCommand, linkerCommand) ->
    
    let llvmPath = Path.ChangeExtension(outputPath, ".ll")
    let objPath = Path.ChangeExtension(outputPath, ".o")
    let outputDir = Path.GetDirectoryName(outputPath)
    
    try
        // Write LLVM IR to file with cross-platform compatible encoding
        let utf8WithoutBom = System.Text.UTF8Encoding(false)
        let normalizedIR = llvmOutput.LLVMIRText.Replace("\r\n", "\n").Replace("\r", "\n")
        File.WriteAllText(llvmPath, normalizedIR, utf8WithoutBom)
        printfn "Saved LLVM IR to: %s" llvmPath
        
        // Compile to object file with enhanced options
        let llcArgs = sprintf "-filetype=obj -mtriple=%s -relocation-model=pic -o \"%s\" \"%s\"" 
                             targetTriple objPath llvmPath
        printfn "Compiling to native code for target '%s'..." targetTriple
        
        ExternalToolchain.runExternalCommand llcCommand llcArgs (Some outputDir) >>= fun _ ->
        printfn "Created object file: %s" objPath
        
        // Enhanced linking with explicit console application configuration for MinGW64
        let linkArgs = 
            if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
                // Explicit console subsystem configuration to avoid WinMain requirement
                sprintf "\"%s\" -o \"%s\" -mconsole -Wl,--subsystem,console -Wl,--entry,mainCRTStartup -static-libgcc -lmsvcrt -lkernel32" 
                        objPath outputPath
            else
                sprintf "\"%s\" -o \"%s\"" objPath outputPath
        
        printfn "Linking object file to executable with console application configuration..."        
        ExternalToolchain.runExternalCommand linkerCommand linkArgs (Some outputDir) >>= fun linkOutput ->
        
        // Verify output file exists
        if File.Exists(outputPath) then
            printfn "Successfully created console executable: %s" outputPath
            // Cleanup intermediate files but keep LLVM IR for debugging
            if File.Exists(objPath) then File.Delete(objPath)
            Success ()
        else
            CompilerFailure [CompilerError(
                "native compilation", 
                "Console executable was not created", 
                Some (sprintf "Linker output: %s\nTry manual compilation: gcc %s -o %s -mconsole" linkOutput objPath outputPath))]
    
    with
    | ex ->
        CompilerFailure [CompilerError("native compilation", 
            "Failed during console application compilation", 
            Some ex.Message)]

/// Displays detailed compilation statistics including AST metrics
let private showEnhancedCompilationStats (pipelineOutput: TranslationPipelineOutput) =
    printfn ""
    printfn "Enhanced Compilation Statistics:"
    printfn "==============================="
    printfn "Successful phases: %d" pipelineOutput.SuccessfulPhases.Length
    printfn "Symbol mappings: %d" pipelineOutput.SymbolMappings.Count
    printfn "Phase outputs saved: %d" pipelineOutput.PhaseOutputs.Count
    
    // Report on intermediate representations
    let astOutputs = 
        pipelineOutput.PhaseOutputs 
        |> Map.filter (fun key _ -> key.Contains("ast") || key.Contains("fcs") || key.Contains("oak"))
    
    if not astOutputs.IsEmpty then
        printfn ""
        printfn "AST Representations Generated:"
        astOutputs |> Map.iter (fun phase content ->
            let lineCount = content.Split('\n').Length
            printfn "  %s: %d lines" phase lineCount)
    
    if not pipelineOutput.Diagnostics.IsEmpty then
        printfn ""
        printfn "Detailed Diagnostics:"
        pipelineOutput.Diagnostics 
        |> List.groupBy fst
        |> List.iter (fun (phase, messages) ->
            printfn "  [%s]:" phase
            messages |> List.iter (fun (_, msg) -> printfn "    %s" msg))

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
                    showEnhancedCompilationStats pipelineOutput
                    
                    match translateToLLVM pipelineOutput.FinalMLIR with
                    | CompilerFailure llvmErrors ->
                        printfn "LLVM translation failed:"
                        llvmErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
                        1
                    | Success llvmOutput ->
                        let optimizationPasses = createOptimizationPipeline Default
                        match optimizeLLVMIR llvmOutput optimizationPasses with
                        | CompilerFailure optimizationErrors ->
                            printfn "Optimization failed:"
                            optimizationErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
                            1
                        | Success optimizedLLVM ->
                            match compileLLVMToNativeConsole optimizedLLVM outputPath target with
                            | CompilerFailure nativeErrors ->
                                printfn "Native compilation failed:"
                                nativeErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
                                1
                            | Success () ->
                                printfn "✓ Complete enhanced XParsec-based compilation successful!"
                                0
            with
            | ex ->
                printfn "Unexpected error: %s" ex.Message
                1