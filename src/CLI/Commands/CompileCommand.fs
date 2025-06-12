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
    if String.IsNullOrWhiteSpace(inputPath) then
        CompilerFailure [SyntaxError(
            { Line = 0; Column = 0; File = ""; Offset = 0 },
            "Input path cannot be empty",
            ["argument validation"])]
    elif not (File.Exists(inputPath)) then
        CompilerFailure [SyntaxError(
            { Line = 0; Column = 0; File = inputPath; Offset = 0 },
            sprintf "Input file '%s' does not exist" inputPath,
            ["file validation"])]
    else
        let inputExt = Path.GetExtension(inputPath).ToLowerInvariant()
        if inputExt <> ".fs" && inputExt <> ".fsx" then
            CompilerFailure [SyntaxError(
                { Line = 0; Column = 0; File = inputPath; Offset = 0 },
                sprintf "Input file must be F# source (.fs or .fsx), got: %s" inputExt,
                ["file validation"])]
        else
            Success ()

/// Reads the source file with comprehensive error handling
let private readSourceFile (inputPath: string) : CompilerResult<string> =
    try
        let sourceCode = File.ReadAllText(inputPath)
        if String.IsNullOrEmpty(sourceCode) then
            CompilerFailure [SyntaxError(
                { Line = 0; Column = 0; File = inputPath; Offset = 0 },
                "Source file is empty",
                ["file reading"])]
        else
            Success sourceCode
    with
    | :? UnauthorizedAccessException as ex ->
        CompilerFailure [SyntaxError(
            { Line = 0; Column = 0; File = inputPath; Offset = 0 },
            sprintf "Access denied: %s" ex.Message,
            ["file reading"])]
    | :? FileNotFoundException as ex ->
        CompilerFailure [SyntaxError(
            { Line = 0; Column = 0; File = inputPath; Offset = 0 },
            sprintf "File not found: %s" ex.Message,
            ["file reading"])]
    | :? IOException as ex ->
        CompilerFailure [SyntaxError(
            { Line = 0; Column = 0; File = inputPath; Offset = 0 },
            sprintf "IO error: %s" ex.Message,
            ["file reading"])]
    | ex ->
        CompilerFailure [SyntaxError(
            { Line = 0; Column = 0; File = inputPath; Offset = 0 },
            sprintf "Unexpected error: %s" ex.Message,
            ["file reading"])]

/// Validates the output path is writable
let private validateOutputPath (outputPath: string) : CompilerResult<unit> =
    if String.IsNullOrWhiteSpace(outputPath) then
        CompilerFailure [SyntaxError(
            { Line = 0; Column = 0; File = ""; Offset = 0 },
            "Output path cannot be empty",
            ["argument validation"])]
    else
        let outputDir = Path.GetDirectoryName(outputPath)
        if not (String.IsNullOrEmpty(outputDir)) && not (Directory.Exists(outputDir)) then
            try
                Directory.CreateDirectory(outputDir) |> ignore
                Success ()
            with
            | ex ->
                CompilerFailure [SyntaxError(
                    { Line = 0; Column = 0; File = outputPath; Offset = 0 },
                    sprintf "Cannot create output directory: %s" ex.Message,
                    ["output validation"])]
        else
            Success ()

/// Converts optimization level string to enum
let private parseOptimizationLevel (optimizeStr: string) : OptimizationLevel =
    match optimizeStr.ToLowerInvariant() with
    | "none" -> OptimizationLevel.None
    | "less" -> OptimizationLevel.Less
    | "aggressive" -> OptimizationLevel.Aggressive
    | "size" -> OptimizationLevel.Size
    | "sizemin" -> OptimizationLevel.SizeMin
    | _ -> OptimizationLevel.Default

/// Saves intermediate compilation results with clean, simple naming
let private saveIntermediateFiles (basePath: string) (baseName: string) (keepIntermediates: bool) (pipelineOutput: TranslationPipelineOutput) =
    if keepIntermediates then
        printfn "Saving intermediate files..."
        try
            let intermediatesDir = Path.Combine(basePath, "intermediates")
            if not (Directory.Exists(intermediatesDir)) then
                Directory.CreateDirectory(intermediatesDir) |> ignore
            
            // Simple, clean file mapping
            let fileMap = [
                ("fsharp-ast", ".fcs")      // F# Compiler Services AST
                ("oak-ast", ".oak")         // Oak AST
                ("closure-transformed", ".closures") 
                ("layout-transformed", ".unions")
                ("mlir", ".mlir")
                ("lowered-mlir", ".lowered")
            ]
            
            // Save phase outputs with clean names
            for (phaseName, extension) in fileMap do
                match pipelineOutput.PhaseOutputs.TryFind phaseName with
                | Some content ->
                    let filePath = Path.Combine(intermediatesDir, baseName + extension)
                    File.WriteAllText(filePath, content, System.Text.Encoding.UTF8)
                    printfn "  %s → %s" phaseName (Path.GetFileName(filePath))
                | Option.None ->
                    printfn "  %s (not found)" phaseName
            
            // Save diagnostics with clean name
            let diagPath = Path.Combine(intermediatesDir, baseName + ".diag")
            let diagContent = 
                pipelineOutput.Diagnostics
                |> List.map (fun (phase, message) -> sprintf "[%s] %s" phase message)
                |> String.concat "\n"
            File.WriteAllText(diagPath, diagContent)
            
            // Save symbol mappings
            let symbolsPath = Path.Combine(intermediatesDir, baseName + ".symbols")
            let symbolsContent = 
                pipelineOutput.SymbolMappings
                |> Map.toList
                |> List.map (fun (orig, trans) -> sprintf "%s → %s" orig trans)
                |> String.concat "\n"
            File.WriteAllText(symbolsPath, symbolsContent)
            
            printfn "Intermediate files saved to: %s" (Path.GetFileName(intermediatesDir))
        with
        | ex ->
            printfn "Warning: Failed to save intermediate files: %s" ex.Message

/// Compiles LLVM IR to native executable
let private compileToNativeExecutable (llvmOutput: LLVMOutput) (outputPath: string) (target: string) (verbose: bool) (noExternalTools: bool) (intermediatesDir: string option) =
    if noExternalTools then
        printfn "Warning: --no-external-tools specified, but native compilation requires external LLVM toolchain"
        
        // Save to intermediates only if keeping intermediates
        match intermediatesDir with
        | Some dir ->
            let baseName = Path.GetFileNameWithoutExtension(outputPath)
            let llvmPath = Path.Combine(dir, baseName + ".ll")
            File.WriteAllText(llvmPath, llvmOutput.LLVMIRText, System.Text.UTF8Encoding(false))
            printfn "Saved LLVM IR to: %s" (Path.GetFileName(llvmPath))
        | None ->
            let llvmPath = Path.ChangeExtension(outputPath, ".ll")
            File.WriteAllText(llvmPath, llvmOutput.LLVMIRText, System.Text.UTF8Encoding(false))
            printfn "Saved LLVM IR to: %s" llvmPath
        0
    else
        match compileLLVMToNative llvmOutput outputPath target with
        | CompilerFailure nativeErrors ->
            printfn "Native compilation failed:"
            nativeErrors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
            1
        | Success () ->
            printfn "✓ Compilation successful: %s" outputPath
            if verbose then
                try
                    let fileInfo = FileInfo(outputPath)
                    printfn "  Output size: %d bytes" fileInfo.Length
                    printfn "  Target: %s" target
                with _ -> ()
            0

/// Main compilation function with clean, concise output
let compile (args: ParseResults<CompileArgs>) =
    printfn "Firefly F# Compiler"
    
    // Parse arguments with minimal output
    let inputPath = args.GetResult Input
    let outputPath = args.GetResult Output
    let target = args.TryGetResult Target |> Option.defaultValue (getDefaultTarget())
    let optimizeStr = args.TryGetResult Optimize |> Option.defaultValue "default"
    let optimizeLevel = parseOptimizationLevel optimizeStr
    let configPath = args.TryGetResult Config |> Option.defaultValue "firefly.toml"
    let keepIntermediates = args.Contains Keep_Intermediates
    let verbose = args.Contains Verbose
    let noExternalTools = args.Contains No_External_Tools
    
    if verbose then
        printfn "Input: %s → Output: %s" (Path.GetFileName(inputPath)) (Path.GetFileName(outputPath))
        printfn "Target: %s, Optimize: %s" target optimizeStr
    
    // Validation
    match validateInputFile inputPath with
    | CompilerFailure errors ->
        errors |> List.iter (fun error -> printfn "Error: %s" (error.ToString()))
        1
    | Success () ->
        
        match validateOutputPath outputPath with
        | CompilerFailure errors ->
            errors |> List.iter (fun error -> printfn "Error: %s" (error.ToString()))
            1
        | Success () ->
            
            // Load configuration quietly
            match loadAndValidateConfig configPath with
            | CompilerFailure configErrors ->
                configErrors |> List.iter (fun error -> printfn "Config Error: %s" (error.ToString()))
                1
            | Success config ->
                
                if verbose then
                    printfn "Configuration: %s v%s" config.PackageName config.Version
                
                // Read source file
                match readSourceFile inputPath with
                | CompilerFailure readErrors ->
                    readErrors |> List.iter (fun error -> printfn "Read Error: %s" (error.ToString()))
                    1
                | Success sourceCode ->
                    
                    // Execute compilation pipeline
                    match translateFsToMLIRWithDiagnostics inputPath sourceCode with
                    | CompilerFailure pipelineErrors ->
                        pipelineErrors |> List.iter (fun error -> printfn "Pipeline Error: %s" (error.ToString()))
                        1
                    
                    | Success pipelineOutput ->
                        
                        // Save all intermediate files in one place
                        if keepIntermediates then
                            let basePath = Path.GetDirectoryName(outputPath)
                            let baseName = Path.GetFileNameWithoutExtension(outputPath)
                            saveIntermediateFiles basePath baseName keepIntermediates pipelineOutput
                        
                        // Continue with LLVM translation
                        printfn "Phase 2: MLIR → LLVM IR"
                        match translateToLLVM pipelineOutput.FinalMLIR with
                        | CompilerFailure llvmErrors ->
                            llvmErrors |> List.iter (fun error -> printfn "LLVM Error: %s" (error.ToString()))
                            1
                        
                        | Success llvmOutput ->
                            printfn "Generated LLVM IR (%d chars)" llvmOutput.LLVMIRText.Length
                            
                            // Save unoptimized LLVM IR to intermediates if requested
                            if keepIntermediates then
                                let intermediatesDir = Path.Combine(Path.GetDirectoryName(outputPath), "intermediates")
                                let baseName = Path.GetFileNameWithoutExtension(outputPath)
                                let llvmPath = Path.Combine(intermediatesDir, baseName + ".ll")
                                File.WriteAllText(llvmPath, llvmOutput.LLVMIRText, System.Text.UTF8Encoding(false))
                                printfn "  llvm-ir → %s.ll" baseName
                            
                            // Apply optimizations
                            printfn "Phase 3: LLVM optimization (%s)" optimizeStr
                            let optimizationPasses = createOptimizationPipeline optimizeLevel
                            match optimizeLLVMIR llvmOutput optimizationPasses with
                            | CompilerFailure optimizationErrors ->
                                optimizationErrors |> List.iter (fun error -> printfn "Optimization Error: %s" (error.ToString()))
                                1
                            
                            | Success optimizedLLVM ->
                                printfn "✓ LLVM optimized (%d chars)" optimizedLLVM.LLVMIRText.Length
                                
                                // Save optimized LLVM IR to intermediates if requested
                                if keepIntermediates then
                                    let intermediatesDir = Path.Combine(Path.GetDirectoryName(outputPath), "intermediates")
                                    let baseName = Path.GetFileNameWithoutExtension(outputPath)
                                    let optimizedLlvmPath = Path.Combine(intermediatesDir, baseName + ".optimized.ll")
                                    File.WriteAllText(optimizedLlvmPath, optimizedLLVM.LLVMIRText, System.Text.UTF8Encoding(false))
                                    printfn "  optimized-llvm → %s.optimized.ll" baseName
                                
                                // Validate zero-allocation if required
                                if config.Compilation.RequireStaticMemory then
                                    printfn "Phase 4: Zero-allocation validation"
                                    match validateZeroAllocationGuarantees optimizedLLVM.LLVMIRText with
                                    | CompilerFailure validationErrors ->
                                        validationErrors |> List.iter (fun error -> printfn "Validation Error: %s" (error.ToString()))
                                        1
                                    | Success () ->
                                        printfn "✓ Zero-allocation guarantees validated"
                                        printfn "Phase 5: Native compilation"
                                        let intermediatesDir = if keepIntermediates then Some (Path.Combine(Path.GetDirectoryName(outputPath), "intermediates")) else None
                                        compileToNativeExecutable optimizedLLVM outputPath target verbose noExternalTools intermediatesDir
                                else
                                    printfn "Phase 4: Native compilation"
                                    let intermediatesDir = if keepIntermediates then Some (Path.Combine(Path.GetDirectoryName(outputPath), "intermediates")) else None
                                    compileToNativeExecutable optimizedLLVM outputPath target verbose noExternalTools intermediatesDir