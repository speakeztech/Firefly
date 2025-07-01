module CLI.Commands.CompileCommand

open System
open System.IO
open FSharp.Compiler.CodeAnalysis
open System.Runtime.InteropServices
open Argu
open Core.XParsec.Foundation
open Core.FCSIngestion.FileLoader
open Core.Utilities.IntermediateWriter
open CLI.Configurations.ProjectConfig
open Dabbit.Pipeline.CompilationTypes
open Dabbit.Pipeline.CompilationOrchestrator

/// Command line arguments for compile command
type CompileArgs =
    | Input of path: string
    | Output of path: string
    | Target of target: string
    | Config of path: string
    | Optimize of level: string
    | [<AltCommandLine("-k")>] Keep_Intermediates
    | [<AltCommandLine("-v")>] Verbose

    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Input _ -> "F# source file to compile"
            | Output _ -> "Output executable path"
            | Target _ -> "Target triple (default: auto-detect)"
            | Config _ -> "Configuration file path (default: firefly.toml)"
            | Optimize _ -> "Optimization level (none, default, aggressive)"
            | Keep_Intermediates -> "Keep intermediate files for debugging"
            | Verbose -> "Enable verbose output"

/// Get default target triple based on platform
let private getDefaultTarget() =
    if RuntimeInformation.IsOSPlatform OSPlatform.Windows then
        "x86_64-pc-windows-msvc"
    elif RuntimeInformation.IsOSPlatform OSPlatform.Linux then
        "x86_64-unknown-linux-gnu"
    elif RuntimeInformation.IsOSPlatform OSPlatform.OSX then
        "x86_64-apple-darwin"
    else
        "x86_64-unknown-unknown"

/// Progress reporter for console output
let private createProgressReporter (verbose: bool) : ProgressCallback =
    fun phase message ->
        if verbose then
            let phaseStr = 
                match phase with
                | Initialization -> "INIT"
                | Parsing -> "PARSE"
                | TypeChecking -> "CHECK"
                | SymbolExtraction -> "SYMBOL"
                | ReachabilityAnalysis -> "REACH"
                | ASTTransformation -> "TRANS"
                | MLIRGeneration -> "MLIR"
                | LLVMGeneration -> "LLVM"
                | Finalization -> "FINAL"
            printfn "[%s] %s" phaseStr message
        else
            match phase with
            | Initialization -> printfn "Initializing..."
            | Parsing -> printfn "Parsing source files..."
            | TypeChecking -> printfn "Type checking..."
            | SymbolExtraction -> printfn "Extracting symbols..."
            | ReachabilityAnalysis -> printfn "Analyzing reachability..."
            | ASTTransformation -> printfn "Transforming AST..."
            | MLIRGeneration -> printfn "Generating MLIR..."
            | LLVMGeneration -> printfn "Generating LLVM IR..."
            | Finalization -> printfn "Finalizing..."

/// Execute native toolchain to produce final executable
let private invokeNativeToolchain (llvmIR: string) (outputPath: string) (target: string) (intermediatesDir: string option) =
    // For now, just write the LLVM IR and print a message
    match intermediatesDir with
    | Some dir ->
        let llPath = Path.Combine(dir, Path.GetFileNameWithoutExtension(outputPath) + ".ll")
        File.WriteAllText(llPath, llvmIR)
        printfn "Wrote LLVM IR to %s" llPath
    | None -> ()
    
    printfn "TODO: Invoke clang/lld to produce %s" outputPath
    Success ()

/// Main entry point for compile command
let execute (args: ParseResults<CompileArgs>) =
    let inputPath = args.GetResult(Input)
    let outputPath = args.GetResult(Output)
    let target = args.TryGetResult(Target) |> Option.defaultValue (getDefaultTarget())
    let optimize = args.TryGetResult(Optimize) |> Option.defaultValue "default"
    let configPath = args.TryGetResult(Config) |> Option.defaultValue "firefly.toml"
    let keepIntermediates = args.Contains(Keep_Intermediates)
    let verbose = args.Contains(Verbose)
    
    // Validate input file exists
    if not (File.Exists inputPath) then
        printfn "Error: Input file '%s' not found" inputPath
        1
    else
        // Load configuration
        let config = 
            match parseConfigFile configPath with
            | Success cfg -> cfg
            | CompilerFailure _ -> defaultConfig
        
        // Setup intermediates directory
        let intermediatesDir = 
            if keepIntermediates then
                let dir = Path.Combine(Path.GetDirectoryName(outputPath), "intermediates")
                prepareIntermediatesDirectory (Some dir)
                Some dir
            else None
        
        // Create pipeline configuration
        let pipelineConfig = {
            EnableClosureElimination = config.Compilation.RequireStaticMemory
            EnableStackAllocation = config.Compilation.RequireStaticMemory
            EnableReachabilityAnalysis = true
            PreserveIntermediateASTs = keepIntermediates
            VerboseOutput = verbose
        }
        
        // Create progress reporter
        let progress = createProgressReporter verbose
        
        printfn "Firefly Compiler v0.2.0"
        printfn "Compiling %s -> %s" inputPath outputPath
        printfn "Target: %s, Optimization: %s" target optimize
        
        // Get project options
        let checker = FSharpChecker.Create()
        let (projectOptions, loadDiagnostics) = 
            loadProjectFiles inputPath checker intermediatesDir
            |> Async.RunSynchronously
        
        // Report any load diagnostics
        if loadDiagnostics.Length > 0 && verbose then
            printfn "Load diagnostics:"
            for diag in loadDiagnostics do
                printfn "  %s" diag.Message
        
        // Run compilation pipeline
        let compilationResult = 
            compileProject inputPath outputPath projectOptions pipelineConfig intermediatesDir progress
            |> Async.RunSynchronously
        
        // Report results
        if compilationResult.Success then
            printfn "✓ Compilation successful!"
            printfn "  Total symbols: %d" compilationResult.Statistics.TotalSymbols
            printfn "  Reachable symbols: %d" compilationResult.Statistics.ReachableSymbols
            printfn "  Eliminated: %.1f%%" 
                ((float compilationResult.Statistics.EliminatedSymbols / float compilationResult.Statistics.TotalSymbols) * 100.0)
            printfn "  Compilation time: %.2fs" (compilationResult.Statistics.CompilationTimeMs / 1000.0)
            
            // Invoke native toolchain if we have LLVM output
            match compilationResult.LLVMOutput with
            | Some llvmIR ->
                match invokeNativeToolchain llvmIR outputPath target intermediatesDir with
                | Success () -> 0
                | CompilerFailure errors ->
                    errors |> List.iter (printfn "Error: %A")
                    1
            | None ->
                printfn "Error: No LLVM output generated"
                1
        else
            printfn "✗ Compilation failed!"
            compilationResult.Diagnostics |> List.iter (printfn "Error: %A")
            1