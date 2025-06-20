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

/// Compilation context with all settings
type CompilationContext = {
    InputPath: string
    OutputPath: string
    Target: string
    OptimizeLevel: OptimizationLevel
    OptimizeStr: string
    Config: FireflyConfig
    KeepIntermediates: bool
    Verbose: bool
    NoExternalTools: bool
    LibraryPaths: string list  // New field for library paths
}

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

/// Converts optimization level string to enum
let private parseOptimizationLevel (optimizeStr: string) : OptimizationLevel =
    match optimizeStr.ToLowerInvariant() with
    | "none" -> OptimizationLevel.None
    | "less" -> OptimizationLevel.Less
    | "aggressive" -> OptimizationLevel.Aggressive
    | "size" -> OptimizationLevel.Size
    | "sizemin" -> OptimizationLevel.SizeMin
    | _ -> OptimizationLevel.Default

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

/// Saves intermediate compilation results
let private saveIntermediateFiles (compilationCtx: CompilationContext) (pipelineOutput: TranslationPipelineOutput) : unit =
    if compilationCtx.KeepIntermediates then
        printfn "Saving intermediate files..."
        try
            let basePath = Path.GetDirectoryName(compilationCtx.OutputPath)
            let baseName = Path.GetFileNameWithoutExtension(compilationCtx.OutputPath)
            let intermediatesDir = Path.Combine(basePath, "intermediates")
            
            if not (Directory.Exists(intermediatesDir)) then
                Directory.CreateDirectory(intermediatesDir) |> ignore
            
            let fileExtensions = [
                ("fsharp-ast", ".fcs")
                ("oak-ast", ".oak")
                ("tree-shaking-stats", ".treeshake.log")
                ("ra-oak", ".ra.oak")
                ("closure-transformed", ".closures") 
                ("layout-transformed", ".unions")
                ("mlir", ".mlir")
                ("lowered-mlir", ".lowered")
            ]
            
            // Process each phase output file
            for (phaseName, extension) in fileExtensions do
                match Map.tryFind phaseName pipelineOutput.PhaseOutputs with
                | Some content ->
                    let filePath = Path.Combine(intermediatesDir, baseName + extension)
                    File.WriteAllText(filePath, content, System.Text.Encoding.UTF8)
                    printfn "  %s → %s" phaseName (Path.GetFileName(filePath))
                | Option.None ->  // Explicitly use Option.None to avoid conflict with OptimizationLevel.None
                    printfn "  %s (not found)" phaseName
            
            let diagPath = Path.Combine(intermediatesDir, baseName + ".diag")
            let diagContent = 
                pipelineOutput.Diagnostics
                |> List.map (fun (phase, message) -> sprintf "[%s] %s" phase message)
                |> String.concat "\n"
            File.WriteAllText(diagPath, diagContent)
            
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

/// Saves LLVM IR file to intermediates
let private saveLLVMFile (ctx: CompilationContext) (suffix: string) (llvmText: string) : unit =
    if ctx.KeepIntermediates then
        let intermediatesDir = Path.Combine(Path.GetDirectoryName(ctx.OutputPath), "intermediates")
        let baseName = Path.GetFileNameWithoutExtension(ctx.OutputPath)
        let fileName = if String.IsNullOrEmpty(suffix) then sprintf "%s.ll" baseName else sprintf "%s.%s.ll" baseName suffix
        let llvmPath = Path.Combine(intermediatesDir, fileName)
        File.WriteAllText(llvmPath, llvmText, System.Text.UTF8Encoding(false))
        printfn "  llvm-ir → %s" fileName

/// Compiles to native executable
let private compileToNative (ctx: CompilationContext) (llvmOutput: LLVMOutput) : CompilerResult<int> =
    if ctx.NoExternalTools then
        printfn "Warning: --no-external-tools specified, but native compilation requires external LLVM toolchain"
        let llvmPath = 
            if ctx.KeepIntermediates then
                let intermediatesDir = Path.Combine(Path.GetDirectoryName(ctx.OutputPath), "intermediates")
                let baseName = Path.GetFileNameWithoutExtension(ctx.OutputPath)
                Path.Combine(intermediatesDir, baseName + ".ll")
            else
                Path.ChangeExtension(ctx.OutputPath, ".ll")
        
        File.WriteAllText(llvmPath, llvmOutput.LLVMIRText, System.Text.UTF8Encoding(false))
        printfn "Saved LLVM IR to: %s" (Path.GetFileName(llvmPath))
        Success 0
    else
        match compileLLVMToNative llvmOutput ctx.OutputPath ctx.Target with
        | CompilerFailure errors ->
            errors |> List.iter (fun error -> printfn "Native compilation failed: %s" (error.ToString()))
            Success 1
        | Success () ->
            printfn "✓ Compilation successful: %s" ctx.OutputPath
            if ctx.Verbose then
                try
                    let fileInfo = FileInfo(ctx.OutputPath)
                    printfn "  Output size: %d bytes" fileInfo.Length
                    printfn "  Target: %s" ctx.Target
                with _ -> ()
            Success 0

/// Result computation expression for chaining operations
type CompilerResultBuilder() =
    member _.Bind(result, f) =
        match result with
        | Success value -> f value
        | CompilerFailure errors -> CompilerFailure errors
    
    member _.Return(value) = Success value
    
    member _.ReturnFrom(result) = result
    
    member _.Zero() = Success ()
    
    member _.Delay(f) = f
    
    member _.Run(f) = f ()
    
    member _.Combine(a, b) =
        match a with
        | Success _ -> b ()
        | CompilerFailure errors -> CompilerFailure errors

let result = CompilerResultBuilder()

/// Main compilation pipeline
let private runCompilationPipeline (ctx: CompilationContext) (sourceCode: string) : CompilerResult<int> =
    result {
        // Phase 1: F# to MLIR
        let! pipelineOutput = translateFsToMLIRWithDiagnostics ctx.InputPath sourceCode
        
        // Save intermediate files
        saveIntermediateFiles ctx pipelineOutput
        
        // Phase 2: MLIR to LLVM IR
        printfn "Phase 2: MLIR → LLVM IR"
        let! llvmOutput = translateToLLVM pipelineOutput.FinalMLIR
        printfn "Generated LLVM IR (%d chars)" llvmOutput.LLVMIRText.Length
        
        // Save unoptimized LLVM IR
        saveLLVMFile ctx "" llvmOutput.LLVMIRText
        
        // Phase 3: LLVM Optimization
        printfn "Phase 3: LLVM optimization (%s)" ctx.OptimizeStr
        let optimizationPasses = createOptimizationPipeline ctx.OptimizeLevel
        let! optimizedLLVM = optimizeLLVMIR llvmOutput optimizationPasses
        printfn "✓ LLVM optimized (%d chars)" optimizedLLVM.LLVMIRText.Length
        
        // Save optimized LLVM IR
        saveLLVMFile ctx "optimized" optimizedLLVM.LLVMIRText
        
        // Phase 4: Zero-allocation validation (if required)
        let! validationResult = 
            if ctx.Config.Compilation.RequireStaticMemory then
                result {
                    printfn "Phase 4: Zero-allocation validation"
                    let! _ = validateZeroAllocationGuarantees optimizedLLVM.LLVMIRText
                    printfn "✓ Zero-allocation guarantees validated"
                    printfn "Phase 5: Native compilation"
                    return ()
                }
            else
                result {
                    printfn "Phase 4: Native compilation"
                    return ()
                }
        
        // Phase 5: Native compilation
        return! compileToNative ctx optimizedLLVM
    }

/// Creates compilation context from command line arguments
let private createCompilationContext (args: ParseResults<CompileArgs>) : CompilerResult<CompilationContext * string> =
    result {
        let inputPath = args.GetResult Input
        let outputPath = args.GetResult Output
        let target = args.TryGetResult Target |> Option.defaultValue (getDefaultTarget())
        let optimizeStr = args.TryGetResult Optimize |> Option.defaultValue "default"
        let optimizeLevel = parseOptimizationLevel optimizeStr
        let configPath = args.TryGetResult Config |> Option.defaultValue "firefly.toml"
        let keepIntermediates = args.Contains Keep_Intermediates
        let verbose = args.Contains Verbose
        let noExternalTools = args.Contains No_External_Tools
        
        // Add library paths to search for dependencies
        let libraryPaths = [
            Path.GetDirectoryName(inputPath);  // Same directory as input file
            Path.Combine(Path.GetDirectoryName(inputPath), "lib", "Alloy")  // Alloy library path
        ]
        
        // Validate inputs
        let! _ = validateInputFile inputPath
        let! _ = validateOutputPath outputPath
        
        // Load configuration
        let! config = loadAndValidateConfig configPath
        
        // Read source file
        let! sourceCode = readSourceFile inputPath
        
        // Handle verbose output
        if verbose then
            printfn "Input: %s → Output: %s" (Path.GetFileName(inputPath)) (Path.GetFileName(outputPath))
            printfn "Target: %s, Optimize: %s" target optimizeStr
            printfn "Configuration: %s v%s" config.PackageName config.Version
            printfn "Library search paths: %s" (String.concat ", " libraryPaths)
        
        let ctx = {
            InputPath = inputPath
            OutputPath = outputPath
            Target = target
            OptimizeLevel = optimizeLevel
            OptimizeStr = optimizeStr
            Config = config
            KeepIntermediates = keepIntermediates
            Verbose = verbose
            NoExternalTools = noExternalTools
            LibraryPaths = libraryPaths  // Add this field to CompilationContext
        }
        
        return (ctx, sourceCode)
    }

/// Main compilation function - clean and functional
let compile (args: ParseResults<CompileArgs>) =
    printfn "Firefly F# Compiler"
    
    match createCompilationContext args with
    | Success (ctx, sourceCode) ->
        // Create intermediates directory if keeping intermediates
        let intermediatesDir = 
            if ctx.KeepIntermediates then
                let dir = Path.Combine(Path.GetDirectoryName(ctx.OutputPath), "intermediates")
                if not (Directory.Exists(dir)) then
                    Directory.CreateDirectory(dir) |> ignore
                Some dir
            else
                None
        
        match runCompilationPipeline ctx sourceCode intermediatesDir with
        | Success exitCode -> exitCode
        | CompilerFailure errors ->
            errors |> List.iter (fun error -> printfn "Error: %s" (error.ToString()))
            1
    | CompilerFailure errors ->
        errors |> List.iter (fun error -> printfn "Error: %s" (error.ToString()))
        1