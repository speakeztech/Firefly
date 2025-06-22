module CLI.Commands.CompileCommand

open System
open System.IO
open Argu
open Core.XParsec.Foundation
open Dabbit.Parsing.Translator
open Core.Conversion.LoweringPipeline
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
with
    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Input _ -> "Input F# source file (required)"
            | Output _ -> "Output binary path (required)"
            | Target _ -> "Target platform (e.g. x86_64-pc-windows-msvc, x86_64-pc-linux-gnu)"
            | Optimize _ -> "Optimization level (none, less, default, aggressive, size)"
            | Config _ -> "Path to configuration file (firefly.toml)"
            | Keep_Intermediates -> "Keep intermediate files (Oak AST, MLIR, LLVM IR)"
            | Verbose -> "Enable verbose output and diagnostics"

/// Compilation context with settings
type CompilationContext = {
    InputPath: string
    OutputPath: string
    Target: string
    OptimizeLevel: string
    Config: FireflyConfig
    KeepIntermediates: bool
    Verbose: bool
    IntermediatesDir: string option
}

/// Gets default target for current platform
let private getDefaultTarget() =
    if Environment.OSVersion.Platform = PlatformID.Win32NT then
        "x86_64-pc-windows-msvc"
    elif Environment.OSVersion.Platform = PlatformID.Unix then
        "x86_64-pc-linux-gnu"
    else
        "x86_64-pc-windows-msvc"

/// Validates input file exists and is readable
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
        Success ()

/// Validates output path is writable
let private validateOutputPath (outputPath: string) : CompilerResult<unit> =
    if String.IsNullOrWhiteSpace(outputPath) then
        CompilerFailure [SyntaxError(
            { Line = 0; Column = 0; File = ""; Offset = 0 },
            "Output path cannot be empty",
            ["argument validation"])]
    else
        try
            let dir = Path.GetDirectoryName(outputPath)
            if not (String.IsNullOrEmpty(dir)) && not (Directory.Exists(dir)) then
                Directory.CreateDirectory(dir) |> ignore
            Success ()
        with ex ->
            CompilerFailure [SyntaxError(
                { Line = 0; Column = 0; File = outputPath; Offset = 0 },
                sprintf "Cannot create output directory: %s" ex.Message,
                ["output validation"])]

/// Reads source file with error handling
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
    with ex ->
        CompilerFailure [SyntaxError(
            { Line = 0; Column = 0; File = inputPath; Offset = 0 },
            sprintf "Error reading file: %s" ex.Message,
            ["file reading"])]

/// Compiles F# source to native executable
let compile (args: ParseResults<CompileArgs>) =
    // Parse command line arguments
    let inputPath = args.GetResult Input
    let outputPath = args.GetResult Output
    let target = args.TryGetResult Target |> Option.defaultValue (getDefaultTarget())
    let optimizeLevel = args.TryGetResult Optimize |> Option.defaultValue "default"
    let configPath = args.TryGetResult Config |> Option.defaultValue "firefly.toml"
    let keepIntermediates = args.Contains Keep_Intermediates
    let verbose = args.Contains Verbose
    
    // Create intermediates directory if needed
    let intermediatesDir = 
        if keepIntermediates then
            let dir = Path.Combine(Path.GetDirectoryName(outputPath), "intermediates")
            if not (Directory.Exists(dir)) then
                Directory.CreateDirectory(dir) |> ignore
            Some dir
        else
            None
    
    // Validate inputs
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
            // Load configuration
            match loadAndValidateConfig configPath with
            | CompilerFailure errors -> 
                errors |> List.iter (fun error -> printfn "Error: %s" (error.ToString()))
                1
            | Success config ->
                // Read source file
                match readSourceFile inputPath with
                | CompilerFailure errors -> 
                    errors |> List.iter (fun error -> printfn "Error: %s" (error.ToString()))
                    1
                | Success sourceCode ->
                    // Create compilation context
                    let ctx = {
                        InputPath = inputPath
                        OutputPath = outputPath
                        Target = target
                        OptimizeLevel = optimizeLevel
                        Config = config
                        KeepIntermediates = keepIntermediates
                        Verbose = verbose
                        IntermediatesDir = intermediatesDir
                    }
                    
                    // Start compilation pipeline
                    printfn "Compiling %s to %s" (Path.GetFileName(inputPath)) (Path.GetFileName(outputPath))
                    
                    // Phase 1: F# to MLIR
                    match translateFsToMLIRWithDiagnostics inputPath sourceCode intermediatesDir with
                    | CompilerFailure errors -> 
                        errors |> List.iter (fun error -> printfn "Error: %s" (error.ToString()))
                        1
                    | Success pipelineOutput ->
                        let mlirText = pipelineOutput.FinalMLIR
                        
                        // Phase 2: MLIR optimization
                        match applyMLIROptimization mlirText intermediatesDir with
                        | CompilerFailure errors -> 
                            errors |> List.iter (fun error -> printfn "Error: %s" (error.ToString()))
                            1
                        | Success optimizedMLIR ->
                            // Phase 3: MLIR to LLVM IR
                            match translateToLLVMIR optimizedMLIR intermediatesDir with
                            | CompilerFailure errors -> 
                                errors |> List.iter (fun error -> printfn "Error: %s" (error.ToString()))
                                1
                            | Success llvmIR ->
                                // Save LLVM IR if keeping intermediates
                                match intermediatesDir with
                                | Some dir ->
                                    let llvmPath = Path.Combine(dir, Path.GetFileNameWithoutExtension(inputPath) + ".ll")
                                    File.WriteAllText(llvmPath, llvmIR)
                                    printfn "Generated LLVM IR: %s" llvmPath
                                | None -> ()
                                
                                // Phase 4: LLVM to native (invoke external tools)
                                printfn "Compilation successful! (LLVM to native using external tools)"
                                // TODO: Implement calling of clang/lld for final linking
                                0