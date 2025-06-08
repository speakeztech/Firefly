module CLI.Commands.CompileCommand

open System
open System.IO
open System.Runtime.InteropServices
open Argu
open Dabbit.Parsing
open Dabbit.Closures
open Dabbit.UnionLayouts
open Core.MLIRGeneration
open Core.Conversion
open CLI.Configurations

/// Command line arguments for the compile command
type CompileArgs =
    | Input of string
    | Output of string
    | Target of string
    | Optimize of string
    | Config of string
    | Keep_Intermediates
with
    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Input _ -> "Input F# source file (required)"
            | Output _ -> "Output binary path (required)"
            | Target _ -> "Target platform (e.g. x86_64-pc-windows-msvc, x86_64-pc-linux-gnu, embedded)"
            | Optimize _ -> "Optimization level (none, less, default, aggressive, size)"
            | Config _ -> "Path to configuration file (firefly.toml)"
            | Keep_Intermediates -> "Keep intermediate files (MLIR, LLVM IR)"

/// Gets the default target for the current platform
let private getDefaultTarget() =
    if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
        "x86_64-w64-windows-gnu"  // Use GNU to match LLVM default
    elif RuntimeInformation.IsOSPlatform(OSPlatform.Linux) then
        "x86_64-pc-linux-gnu"
    elif RuntimeInformation.IsOSPlatform(OSPlatform.OSX) then
        "x86_64-apple-darwin"
    else
        "x86_64-w64-windows-gnu"  // Default fallback

/// Executes the compilation process
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

    // Load configuration
    let config = ProjectConfig.parseConfigFile configPath

    // Read source code
    if not (File.Exists inputPath) then
        printfn "Error: Input file '%s' does not exist" inputPath
        1
    else
        let sourceCode = File.ReadAllText inputPath

        // Process through Dabbit (F# to Oak AST to MLIR)
        printfn "Parsing F# source code..."
        let oakAst = AstConverter.parseAndConvertToOakAst sourceCode

        printfn "Transforming closures..."
        let transformedAst = ClosureTransformer.eliminateClosures oakAst

        printfn "Computing fixed layouts..."
        let fixedLayoutAst = FixedLayoutCompiler.compileFixedLayouts transformedAst

        printfn "Generating MLIR using XParsec..."
        let mlirOutput = XParsecMLIRGenerator.generateMLIR fixedLayoutAst
        let mlirText = Translator.generateMLIRModuleText mlirOutput

        // Apply MLIR lowering pipeline
        printfn "Applying MLIR lowering passes..."
        let loweredMLIR = LoweringPipeline.applyLoweringPipeline mlirText

        // Translate to LLVM IR
        printfn "Translating to LLVM IR..."
        let llvmOutput = LLVMTranslator.translateToLLVM loweredMLIR

        // Apply optimizations
        let optimizationLevel = 
            match optimizeLevel.ToLowerInvariant() with
            | "none" -> OptimizationPipeline.OptimizationLevel.None
            | "less" -> OptimizationPipeline.OptimizationLevel.Less
            | "aggressive" -> OptimizationPipeline.OptimizationLevel.Aggressive
            | "size" -> OptimizationPipeline.OptimizationLevel.Size
            | "sizemin" -> OptimizationPipeline.OptimizationLevel.SizeMin
            | _ -> OptimizationPipeline.OptimizationLevel.Default

        printfn "Applying LLVM optimizations (%A)..." optimizationLevel
        let optimizationPasses = OptimizationPipeline.createOptimizationPipeline optimizationLevel
        let optimizedLLVM = OptimizationPipeline.optimizeLLVMIR llvmOutput optimizationPasses

        // Save intermediate files if requested
        if keepIntermediates then
            let basePath = Path.GetDirectoryName(outputPath)
            let baseName = Path.GetFileNameWithoutExtension(outputPath)

            let mlirPath = Path.Combine(basePath, baseName + ".mlir")
            let llvmPath = Path.Combine(basePath, baseName + ".ll")

            File.WriteAllText(mlirPath, mlirText, System.Text.Encoding.UTF8)
            File.WriteAllText(llvmPath, optimizedLLVM.LLVMIRText, System.Text.Encoding.UTF8)

            printfn "Saved intermediate files:"
            printfn "  MLIR: %s" mlirPath
            printfn "  LLVM IR: %s" llvmPath

        // Compile LLVM IR to native code
        printfn "Compiling to native code for target '%s'..." target
        let success = LLVMTranslator.compileLLVMToNative optimizedLLVM outputPath target

        if success then
            printfn "Compilation successful: %s" outputPath
            0 // Success
        else
            printfn "Error: Compilation failed"
            1 // Error