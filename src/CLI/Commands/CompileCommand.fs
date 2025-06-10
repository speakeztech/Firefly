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
            | Keep_Intermediates -> "Keep intermediate files (Oak AST, MLIR, LLVM IR)"
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

/// Validates that the input file exists and is readable
let private validateInputFile (inputPath: string) : CompilerResult<unit> =
    printfn "Validating input file: %s" inputPath
    
    if String.IsNullOrWhiteSpace(inputPath) then
        printfn "Error: Input path is null or empty"
        CompilerFailure [SyntaxError(
            { Line = 0; Column = 0; File = ""; Offset = 0 },
            "Input path cannot be empty",
            ["argument validation"])]
    elif not (File.Exists(inputPath)) then
        printfn "Error: Input file does not exist at path: %s" inputPath
        CompilerFailure [SyntaxError(
            { Line = 0; Column = 0; File = inputPath; Offset = 0 },
            sprintf "Input file '%s' does not exist" inputPath,
            ["file validation"])]
    else
        let fileInfo = FileInfo(inputPath)
        printfn "Input file found: %s (%d bytes)" inputPath fileInfo.Length
        
        let inputExt = Path.GetExtension(inputPath).ToLowerInvariant()
        if inputExt <> ".fs" && inputExt <> ".fsx" then
            printfn "Error: Invalid file extension: %s" inputExt
            CompilerFailure [SyntaxError(
                { Line = 0; Column = 0; File = inputPath; Offset = 0 },
                sprintf "Input file must be F# source (.fs or .fsx), got: %s" inputExt,
                ["file validation"])]
        else
            printfn "File validation successful"
            Success ()

/// Reads the source file with comprehensive error handling
let private readSourceFile (inputPath: string) : CompilerResult<string> =
    printfn "Reading source file: %s" inputPath
    
    try
        let sourceCode = File.ReadAllText(inputPath)
        printfn "Source file read successfully: %d characters" sourceCode.Length
        
        if String.IsNullOrEmpty(sourceCode) then
            printfn "Warning: Source file is empty"
            CompilerFailure [SyntaxError(
                { Line = 0; Column = 0; File = inputPath; Offset = 0 },
                "Source file is empty",
                ["file reading"])]
        else
            printfn "Source content preview: %s..." (sourceCode.Substring(0, min 50 sourceCode.Length))
            Success sourceCode
    with
    | :? UnauthorizedAccessException as ex ->
        printfn "Error: Access denied reading file: %s" ex.Message
        CompilerFailure [SyntaxError(
            { Line = 0; Column = 0; File = inputPath; Offset = 0 },
            sprintf "Access denied: %s" ex.Message,
            ["file reading"])]
    | :? FileNotFoundException as ex ->
        printfn "Error: File not found: %s" ex.Message
        CompilerFailure [SyntaxError(
            { Line = 0; Column = 0; File = inputPath; Offset = 0 },
            sprintf "File not found: %s" ex.Message,
            ["file reading"])]
    | :? IOException as ex ->
        printfn "Error: IO exception reading file: %s" ex.Message
        CompilerFailure [SyntaxError(
            { Line = 0; Column = 0; File = inputPath; Offset = 0 },
            sprintf "IO error: %s" ex.Message,
            ["file reading"])]
    | ex ->
        printfn "Error: Unexpected exception reading file: %s" ex.Message
        CompilerFailure [SyntaxError(
            { Line = 0; Column = 0; File = inputPath; Offset = 0 },
            sprintf "Unexpected error: %s" ex.Message,
            ["file reading"])]

/// Validates the output path is writable
let private validateOutputPath (outputPath: string) : CompilerResult<unit> =
    printfn "Validating output path: %s" outputPath
    
    if String.IsNullOrWhiteSpace(outputPath) then
        printfn "Error: Output path is null or empty"
        CompilerFailure [SyntaxError(
            { Line = 0; Column = 0; File = ""; Offset = 0 },
            "Output path cannot be empty",
            ["argument validation"])]
    else
        let outputDir = Path.GetDirectoryName(outputPath)
        if not (String.IsNullOrEmpty(outputDir)) && not (Directory.Exists(outputDir)) then
            printfn "Creating output directory: %s" outputDir
            try
                Directory.CreateDirectory(outputDir) |> ignore
                printfn "Output directory created successfully"
                Success ()
            with
            | ex ->
                printfn "Error: Failed to create output directory: %s" ex.Message
                CompilerFailure [SyntaxError(
                    { Line = 0; Column = 0; File = outputPath; Offset = 0 },
                    sprintf "Cannot create output directory: %s" ex.Message,
                    ["output validation"])]
        else
            printfn "Output path validation successful"
            Success ()

/// Converts optimization level string to enum
let private parseOptimizationLevel (optimizeStr: string) : OptimizationLevel =
    match optimizeStr.ToLowerInvariant() with
    | "none" -> OptimizationLevel.None
    | "less" -> Less
    | "aggressive" -> Aggressive
    | "size" -> Size
    | "sizemin" -> SizeMin
    | _ -> Default

/// Saves intermediate compilation results for debugging
let private saveIntermediateFiles (basePath: string) (baseName: string) (keepIntermediates: bool) (pipelineOutput: TranslationPipelineOutput) =
    if keepIntermediates then
        printfn "Saving intermediate files..."
        try
            let intermediatesDir = Path.Combine(basePath, "intermediates")
            if not (Directory.Exists(intermediatesDir)) then
                Directory.CreateDirectory(intermediatesDir) |> ignore
            
            // Save Oak AST output immediately after successful pipeline completion
            try
                // Find the Oak AST content from phase outputs
                match pipelineOutput.PhaseOutputs.TryFind "oak-ast" with
                | Some oakContent ->
                    let oakFilePath = Path.Combine(intermediatesDir, baseName + ".oak")
                    File.WriteAllText(oakFilePath, oakContent, System.Text.Encoding.UTF8)
                    printfn "Oak AST written to: %s (%d characters)" oakFilePath oakContent.Length
                | Option.None ->
                    printfn "Warning: Oak AST content not found in pipeline outputs"
            with
            | ex ->
                printfn "Warning: Failed to write Oak AST file: %s" ex.Message
            
            // Save pipeline diagnostics
            let diagnosticsPath = Path.Combine(intermediatesDir, baseName + ".diagnostics.txt")
            let diagnosticsContent = 
                pipelineOutput.Diagnostics
                |> List.map (fun (phase, message) -> sprintf "[%s] %s" phase message)
                |> String.concat "\n"
            File.WriteAllText(diagnosticsPath, diagnosticsContent)
            
            // Save all intermediate outputs
            pipelineOutput.PhaseOutputs
            |> Map.iter (fun phaseName output ->
                let extension = 
                    match phaseName with
                    | name when name.Contains("mlir") -> ".mlir"
                    | name when name.Contains("oak") -> ".oak"
                    | name when name.Contains("llvm") -> ".ll"
                    | _ -> ".txt"
                let fileName = sprintf "%s.%s%s" baseName (phaseName.Replace("-", "_")) extension
                let filePath = Path.Combine(intermediatesDir, fileName)
                File.WriteAllText(filePath, output, System.Text.Encoding.UTF8))
            
            // Save symbol mappings
            let symbolMappingsPath = Path.Combine(intermediatesDir, baseName + ".symbols.txt")
            let symbolMappingsContent = 
                pipelineOutput.SymbolMappings
                |> Map.toList
                |> List.map (fun (original, transformed) -> sprintf "%s -> %s" original transformed)
                |> String.concat "\n"
            File.WriteAllText(symbolMappingsPath, symbolMappingsContent)
            
            printfn "Intermediate files saved to: %s" intermediatesDir
        with
        | ex ->
            printfn "Warning: Failed to save intermediate files: %s" ex.Message

/// Compiles LLVM IR to native executable
let private compileToNativeExecutable (llvmOutput: LLVMOutput) (outputPath: string) (target: string) (verbose: bool) (noExternalTools: bool) =
    printfn "Compiling to native executable for target: %s" target
    
    if noExternalTools then
        printfn "Warning: --no-external-tools specified, but native compilation requires external LLVM toolchain"
        printfn "The compilation pipeline has generated optimized LLVM IR, but cannot complete native compilation"
        printfn "To complete compilation, remove --no-external-tools flag or use external LLVM tools"
        
        // Save the LLVM IR for manual compilation
        let llvmPath = Path.ChangeExtension(outputPath, ".ll")
        File.WriteAllText(llvmPath, llvmOutput.LLVMIRText, System.Text.UTF8Encoding(false))
        printfn "Saved LLVM IR to: %s" llvmPath
        printfn "To compile manually: llc -filetype=obj %s -o %s.o && gcc %s.o -o %s" 
                 llvmPath (Path.GetFileNameWithoutExtension(outputPath))
                 (Path.GetFileNameWithoutExtension(outputPath)) outputPath
        0
    else
        printfn "Invoking LLVM toolchain for native compilation..."
        match compileLLVMToNative llvmOutput outputPath target with
        | CompilerFailure nativeErrors ->
            printfn "Native compilation failed:"
            nativeErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
            printfn ""
            printfn "Possible solutions:"
            printfn "  1. Install LLVM tools (llc, clang/gcc) in your PATH"
            printfn "  2. Use --no-external-tools to generate LLVM IR only"
            printfn "  3. Check that target '%s' is supported by your LLVM installation" target
            1
        
        | Success () ->
            printfn "✓ Compilation successful: %s" outputPath
            
            if verbose then
                try
                    let fileInfo = FileInfo(outputPath)
                    printfn "  Output size: %d bytes" fileInfo.Length
                    printfn "  Target: %s" target
                    printfn "  Uses zero-allocation Firefly runtime"
                with
                | _ -> ()
            
            0

/// Main compilation function with comprehensive error handling
let compile (args: ParseResults<CompileArgs>) =
    printfn "Firefly F# Compiler - Starting compilation process"
    
    // Parse and validate arguments with explicit output
    printfn "Parsing command line arguments..."
    let inputPath = 
        match args.TryGetResult Input with
        | Some path -> 
            printfn "Input file: %s" path
            path
        | Option.None -> 
            printfn "Error: Input file is required"
            printfn "Usage: firefly compile --input <file.fs> --output <executable>"
            exit 1
    
    let outputPath = 
        match args.TryGetResult Output with
        | Some path -> 
            printfn "Output file: %s" path
            path
        | Option.None -> 
            printfn "Error: Output path is required"
            printfn "Usage: firefly compile --input <file.fs> --output <executable>"
            exit 1
    
    let target = args.TryGetResult Target |> Option.defaultValue (getDefaultTarget())
    let optimizeStr = args.TryGetResult Optimize |> Option.defaultValue "default"
    let optimizeLevel = parseOptimizationLevel optimizeStr
    let configPath = args.TryGetResult Config |> Option.defaultValue "firefly.toml"
    let keepIntermediates = args.Contains Keep_Intermediates
    let verbose = args.Contains Verbose
    let noExternalTools = args.Contains No_External_Tools
    
    printfn "Configuration: target=%s, optimize=%s, keep-intermediates=%b, verbose=%b" 
            target optimizeStr keepIntermediates verbose
    
    // Step 1: Validate input file
    match validateInputFile inputPath with
    | CompilerFailure errors ->
        printfn "Input file validation failed:"
        errors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
        1
    | Success () ->
        
        // Step 2: Validate output path
        match validateOutputPath outputPath with
        | CompilerFailure errors ->
            printfn "Output path validation failed:"
            errors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
            1
        | Success () ->
            
            // Step 3: Load configuration
            printfn "Loading configuration from: %s" configPath
            match loadAndValidateConfig configPath with
            | CompilerFailure configErrors ->
                printfn "Configuration errors:"
                configErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
                1
            | Success config ->
                
                printfn "Loaded configuration for package: %s v%s" config.PackageName config.Version
                if verbose then
                    printfn "Compilation settings:"
                    printfn "  Require static memory: %b" config.Compilation.RequireStaticMemory
                    printfn "  Eliminate closures: %b" config.Compilation.EliminateClosures
                    printfn "  Optimization level: %s" config.Compilation.OptimizationLevel
                    printfn "  Use LTO: %b" config.Compilation.UseLTO
                
                // Step 4: Read source file
                match readSourceFile inputPath with
                | CompilerFailure readErrors ->
                    printfn "Source file reading failed:"
                    readErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
                    1
                | Success sourceCode ->
                    
                    // Step 5: Execute compilation pipeline
                    printfn "Starting Firefly compilation pipeline..."
                    printfn "Phase 1: F# to MLIR translation"
                    
                    match translateFsToMLIRWithDiagnostics inputPath sourceCode with
                    | CompilerFailure pipelineErrors ->
                        printfn "Compilation pipeline failed:"
                        pipelineErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
                        1
                    
                    | Success pipelineOutput ->
                        printfn "Phase 1 completed successfully with %d phases:" pipelineOutput.SuccessfulPhases.Length
                        pipelineOutput.SuccessfulPhases |> List.iter (fun phase -> printfn "  ✓ %s" phase)
                        
                        if not pipelineOutput.Diagnostics.IsEmpty then
                            printfn "Pipeline diagnostics:"
                            pipelineOutput.Diagnostics |> List.iter (fun (phase, msg) -> printfn "  [%s] %s" phase msg)
                        
                        // Save intermediate files if requested
                        let basePath = Path.GetDirectoryName(outputPath)
                        let baseName = Path.GetFileNameWithoutExtension(outputPath)
                        saveIntermediateFiles basePath baseName keepIntermediates pipelineOutput
                        
                        // Step 6: MLIR to LLVM IR translation
                        printfn "Phase 2: MLIR to LLVM IR translation"
                        
                        match translateToLLVM pipelineOutput.FinalMLIR with
                        | CompilerFailure llvmErrors ->
                            printfn "LLVM IR generation failed:"
                            llvmErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
                            1
                        
                        | Success llvmOutput ->
                            printfn "Generated LLVM IR for module: %s" llvmOutput.ModuleName
                            
                            // Step 7: Apply optimizations
                            printfn "Phase 3: LLVM optimization"
                            let optimizationPasses = createOptimizationPipeline optimizeLevel
                            
                            printfn "Applying LLVM optimizations (level: %A)..." optimizeLevel
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
                                
                                // Step 8: Validate zero-allocation guarantees
                                if config.Compilation.RequireStaticMemory then
                                    printfn "Phase 4: Zero-allocation validation"
                                    
                                    match validateZeroAllocationGuarantees optimizedLLVM.LLVMIRText with
                                    | CompilerFailure validationErrors ->
                                        printfn "Zero-allocation validation failed:"
                                        validationErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
                                        printfn "Consider disabling require_static_memory in configuration if heap allocation is acceptable"
                                        1
                                    | Success () ->
                                        printfn "✓ Zero-allocation guarantees validated"
                                        
                                        // Step 9: Native compilation
                                        printfn "Phase 5: Native code generation"
                                        compileToNativeExecutable optimizedLLVM outputPath target verbose noExternalTools
                                else
                                    // Step 9: Native compilation (skip validation)
                                    printfn "Phase 4: Native code generation"
                                    compileToNativeExecutable optimizedLLVM outputPath target verbose noExternalTools