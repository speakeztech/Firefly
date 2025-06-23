module CLI.Commands.CompileCommand

#nowarn "57" // Suppress experimental FCS API warnings

open System
open System.IO
open Argu
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Text
open FSharp.Compiler.Diagnostics
open FSharp.Compiler.Syntax
open Core.XParsec.Foundation
open Core.FCSProcessing.ASTTransformer
open Core.FCSProcessing.DependencyResolver
open Core.MLIRGeneration.DirectGenerator
open Core.MLIRGeneration.TypeMapping
open Core.Conversion.LoweringPipeline
open Core.Conversion.OptimizationPipeline
open CLI.Configurations.ProjectConfig
open Dabbit.Bindings.SymbolRegistry
open Dabbit.Analysis.ReachabilityAnalyzer
open Dabbit.Transformations.ClosureElimination

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

/// Write intermediate file if keeping intermediates
let private writeIntermediateFile (dir: string option) (baseName: string) (extension: string) (content: string) =
    match dir with
    | Some d ->
        let filePath = Path.Combine(d, baseName + extension)
        File.WriteAllText(filePath, content)
        printfn "  Wrote %s (%d bytes)" (Path.GetFileName(filePath)) content.Length
    | None -> ()

/// Main compilation pipeline
let private compilePipeline (ctx: CompilationContext) (sourceCode: string) : CompilerResult<unit> =
    let baseName = Path.GetFileNameWithoutExtension(ctx.InputPath)
    
    // Phase 1: Parse with FCS
    printfn "Phase 1: Parsing F# source..."
    let checker = FSharpChecker.Create()
    let sourceText = SourceText.ofString sourceCode
    let (projectOptions, scriptDiagnostics) = 
        checker.GetProjectOptionsFromScript(ctx.InputPath, sourceText) 
        |> Async.RunSynchronously
    
    // Create parsing options
    let parsingOptions = { 
        FSharpParsingOptions.Default with 
            SourceFiles = [| ctx.InputPath |]
            ConditionalDefines = []
            DiagnosticOptions = FSharpDiagnosticOptions.Default
            LangVersionText = "preview"
            IsInteractive = false
            CompilingFSharpCore = false
            IsExe = true
    }
    
    let parseResults = checker.ParseFile(ctx.InputPath, sourceText, parsingOptions) |> Async.RunSynchronously
    
    // Check for parse errors first
    if parseResults.ParseHadErrors then
        let errors = parseResults.Diagnostics |> Array.map (fun d -> d.Message) |> String.concat "\n"
        CompilerFailure [SyntaxError({ Line = 0; Column = 0; File = ctx.InputPath; Offset = 0 }, errors, ["FCS parsing"])]
    else
        // Extract parse tree
        let parseTreeOpt = parseResults.ParseTree
        match parseTreeOpt with
        | None -> 
            CompilerFailure [SyntaxError({ Line = 0; Column = 0; File = ctx.InputPath; Offset = 0 }, "No parse tree generated", ["FCS parsing"])]
        | Some parsedInput ->
            
            // Write FCS AST if keeping intermediates
            writeIntermediateFile ctx.IntermediatesDir baseName ".fcs" (sprintf "%A" parsedInput)
            
            // Phase 2: Type check
            printfn "Phase 2: Type checking..."
            let checkResult = checker.CheckFileInProject(parseResults, ctx.InputPath, 0, sourceText, projectOptions) |> Async.RunSynchronously
            
            match checkResult with
            | FSharpCheckFileAnswer.Aborted -> 
                CompilerFailure [InternalError("type checking", "Type checking was aborted", None)]
            | FSharpCheckFileAnswer.Succeeded checkResults ->
                
                // Phase 3: Transform AST
                printfn "Phase 3: Transforming AST..."
                let typeCtx = TypeContextBuilder.create()
                
                match RegistryConstruction.buildAlloyRegistry() with
                | CompilerFailure errors -> CompilerFailure errors
                | Success symbolRegistry ->
                    
                    // Analyze dependencies and reachability
                    let deps = buildDependencyMap checkResults
                    let entryPoints = findEntryPoints checkResults
                    let symbols = getAllSymbols checkResults
                    let reachability = { 
                        Reachable = findTransitiveDependencies deps entryPoints
                        UnionCases = Map.empty
                        Statistics = {
                            TotalSymbols = Map.count symbols
                            ReachableSymbols = Set.count entryPoints
                            EliminatedSymbols = Map.count symbols - Set.count entryPoints
                            ModuleBreakdown = Map.empty
                        }
                    }
                    
                    // Create transformation context
                    let transformCtx = {
                        TypeContext = typeCtx
                        SymbolRegistry = symbolRegistry
                        Reachability = reachability
                        ClosureState = { Counter = 0; Scope = Set.empty; Lifted = [] }
                    }
                    
                    // Transform the AST
                    let transformedAST = transformAST transformCtx parsedInput
                    
                    // Write Oak AST if keeping intermediates
                    writeIntermediateFile ctx.IntermediatesDir baseName ".oak" (sprintf "%A" transformedAST)
                    
                    // Phase 4: Generate MLIR
                    printfn "Phase 4: Generating MLIR..."
                    let mlirText = generateModule baseName typeCtx transformedAST
                    
                    // Write MLIR if keeping intermediates
                    writeIntermediateFile ctx.IntermediatesDir baseName ".mlir" mlirText
                    
                    // Phase 5: Apply lowering
                    printfn "Phase 5: Lowering MLIR..."
                    match applyLoweringPipeline mlirText with
                    | CompilerFailure errors -> CompilerFailure errors
                    | Success loweredMLIR ->
                        
                        // Write lowered MLIR if keeping intermediates
                        writeIntermediateFile ctx.IntermediatesDir baseName "_lowered.mlir" loweredMLIR
                        
                        // Phase 6: Optimize
                        printfn "Phase 6: Optimizing..."
                        let optLevel = 
                            match ctx.OptimizeLevel.ToLowerInvariant() with
                            | "none" -> OptimizationLevel.Zero
                            | "less" -> OptimizationLevel.Less
                            | "aggressive" -> OptimizationLevel.Aggressive
                            | "size" -> OptimizationLevel.Size
                            | _ -> OptimizationLevel.Default
                        
                        let llvmOutput = {
                            LLVMIRText = loweredMLIR
                            ModuleName = baseName
                            OptimizationLevel = optLevel
                            Metadata = Map.empty
                        }
                        
                        let passes = createOptimizationPipeline optLevel
                        match optimizeLLVMIR llvmOutput passes with
                        | CompilerFailure errors -> CompilerFailure errors
                        | Success optimizedOutput ->
                            
                            // Phase 7: Convert to LLVM IR
                            printfn "Phase 7: Converting to LLVM IR..."
                            match translateToLLVMIR optimizedOutput.LLVMIRText ctx.IntermediatesDir with
                            | CompilerFailure errors -> CompilerFailure errors
                            | Success llvmIR ->
                                
                                // Write LLVM IR if keeping intermediates
                                writeIntermediateFile ctx.IntermediatesDir baseName ".ll" llvmIR
                                
                                // Validate zero-allocation guarantees
                                match validateZeroAllocationGuarantees llvmIR with
                                | CompilerFailure errors -> CompilerFailure errors
                                | Success () ->
                                    printfn "✓ Zero-allocation guarantees verified"
                                    
                                    // Phase 8: Invoke external tools for final compilation
                                    printfn "Phase 8: Invoking external tools..."
                                    printfn "TODO: Call clang/lld to produce final executable"
                                    Success ()

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
                    printfn "Target: %s, Optimization: %s" target optimizeLevel
                    printfn ""
                    
                    match compilePipeline ctx sourceCode with
                    | Success () ->
                        printfn ""
                        printfn "Compilation successful!"
                        0
                    | CompilerFailure errors ->
                        printfn ""
                        printfn "Compilation failed:"
                        errors |> List.iter (fun error -> printfn "  %s" (error.ToString()))
                        1