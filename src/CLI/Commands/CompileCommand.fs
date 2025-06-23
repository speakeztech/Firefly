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

/// Compiler result builder for computation expressions
type CompilerResultBuilder() =
    member _.Return(x) = Success x
    member _.ReturnFrom(x: CompilerResult<_>) = x
    member _.Bind(x, f) = ResultHelpers.bind f x
    member _.Zero() = Success ()
    member _.Combine(a: CompilerResult<unit>, b: CompilerResult<'T>) : CompilerResult<'T> =
        match a with
        | Success () -> b
        | CompilerFailure errors -> CompilerFailure errors

let compilerResult = CompilerResultBuilder()

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

/// Phase 1: Parse F# source
let private parseSource (ctx: CompilationContext) (sourceCode: string) : CompilerResult<ParsedInput * FSharpChecker * FSharpProjectOptions> =
    printfn "Phase 1: Parsing F# source..."
    let checker = FSharpChecker.Create()
    let sourceText = SourceText.ofString sourceCode
    let (projectOptions, _) = 
        checker.GetProjectOptionsFromScript(ctx.InputPath, sourceText) 
        |> Async.RunSynchronously
    
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
    
    // Check for errors first
    if parseResults.ParseHadErrors then
        let errors = parseResults.Diagnostics |> Array.map (fun d -> d.Message) |> String.concat "\n"
        CompilerFailure [SyntaxError({ Line = 0; Column = 0; File = ctx.InputPath; Offset = 0 }, errors, ["FCS parsing"])]
    else
        // ParseTree is not optional in this version of FCS - it's always present
        let parsedInput = parseResults.ParseTree
        
        if ctx.KeepIntermediates then
            writeIntermediateFile ctx.IntermediatesDir (Path.GetFileNameWithoutExtension ctx.InputPath) ".fcs" (sprintf "%A" parsedInput)
        
        Success (parsedInput, checker, projectOptions)

/// Phase 2: Type check
let private typeCheck (ctx: CompilationContext) (parsedInput: ParsedInput, checker: FSharpChecker, projectOptions: FSharpProjectOptions) : CompilerResult<FSharpCheckFileResults * ParsedInput> =
    printfn "Phase 2: Type checking..."
    let sourceText = SourceText.ofString (File.ReadAllText ctx.InputPath)
    
    // We need to re-parse to get the parse results for CheckFileInProject
    let parsingOptions = { 
        FSharpParsingOptions.Default with 
            SourceFiles = [| ctx.InputPath |]
            LangVersionText = "preview"
            IsExe = true
    }
    let parseResults = checker.ParseFile(ctx.InputPath, sourceText, parsingOptions) |> Async.RunSynchronously
    
    let checkResult = checker.CheckFileInProject(parseResults, ctx.InputPath, 0, sourceText, projectOptions) |> Async.RunSynchronously
    
    match checkResult with
    | FSharpCheckFileAnswer.Aborted -> 
        CompilerFailure [InternalError("type checking", "Type checking was aborted", None)]
    | FSharpCheckFileAnswer.Succeeded checkResults ->
        Success (checkResults, parsedInput)

/// Phase 3: Transform AST
let private transformASTPhase (ctx: CompilationContext) (checkResults: FSharpCheckFileResults, parsedInput: ParsedInput) : CompilerResult<ParsedInput * TypeContext * SymbolRegistry> =
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
        
        let transformCtx = {
            TypeContext = typeCtx
            SymbolRegistry = symbolRegistry
            Reachability = reachability
            ClosureState = { Counter = 0; Scope = Set.empty; Lifted = [] }
        }
        
        let transformedAST = transformAST transformCtx parsedInput
        
        if ctx.KeepIntermediates then
            writeIntermediateFile ctx.IntermediatesDir (Path.GetFileNameWithoutExtension ctx.InputPath) ".oak" (sprintf "%A" transformedAST)
        
        Success (transformedAST, typeCtx, symbolRegistry)

/// Phase 4: Generate MLIR
let private generateMLIR (ctx: CompilationContext) (transformedAST: ParsedInput, typeCtx: TypeContext, symbolRegistry: SymbolRegistry) : CompilerResult<string> =
    printfn "Phase 4: Generating MLIR..."
    let baseName = Path.GetFileNameWithoutExtension(ctx.InputPath)
    let mlirText = generateModule baseName typeCtx transformedAST
    
    if ctx.KeepIntermediates then
        writeIntermediateFile ctx.IntermediatesDir baseName ".mlir" mlirText
    
    Success mlirText

/// Phase 5: Lower MLIR
let private lowerMLIR (ctx: CompilationContext) (mlirText: string) : CompilerResult<string> =
    printfn "Phase 5: Lowering MLIR..."
    match applyLoweringPipeline mlirText with
    | CompilerFailure errors -> CompilerFailure errors
    | Success loweredMLIR ->
        if ctx.KeepIntermediates then
            let baseName = Path.GetFileNameWithoutExtension(ctx.InputPath)
            writeIntermediateFile ctx.IntermediatesDir baseName "_lowered.mlir" loweredMLIR
        Success loweredMLIR

/// Phase 6: Optimize
let private optimizeCode (ctx: CompilationContext) (loweredMLIR: string) : CompilerResult<LLVMOutput> =
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
        ModuleName = Path.GetFileNameWithoutExtension(ctx.InputPath)
        OptimizationLevel = optLevel
        Metadata = Map.empty
    }
    
    let passes = createOptimizationPipeline optLevel
    optimizeLLVMIR llvmOutput passes

/// Phase 7: Convert to LLVM IR
let private convertToLLVMIR (ctx: CompilationContext) (optimizedOutput: LLVMOutput) : CompilerResult<string> =
    printfn "Phase 7: Converting to LLVM IR..."
    match translateToLLVMIR optimizedOutput.LLVMIRText ctx.IntermediatesDir with
    | CompilerFailure errors -> CompilerFailure errors
    | Success llvmIR ->
        if ctx.KeepIntermediates then
            let baseName = Path.GetFileNameWithoutExtension(ctx.InputPath)
            writeIntermediateFile ctx.IntermediatesDir baseName ".ll" llvmIR
        Success llvmIR

/// Phase 8: Validate and finalize
let private validateAndFinalize (ctx: CompilationContext) (llvmIR: string) : CompilerResult<unit> =
    match validateZeroAllocationGuarantees llvmIR with
    | CompilerFailure errors -> CompilerFailure errors
    | Success () ->
        printfn "✓ Zero-allocation guarantees verified"
        printfn "Phase 8: Invoking external tools..."
        printfn "TODO: Call clang/lld to produce final executable"
        Success ()

/// Main compilation pipeline - clean composition using computation expression
let private compilePipeline (ctx: CompilationContext) (sourceCode: string) : CompilerResult<unit> =
    compilerResult {
        let! parsed = parseSource ctx sourceCode
        let! typeChecked = typeCheck ctx parsed
        let! transformed = transformASTPhase ctx typeChecked
        let! mlir = generateMLIR ctx transformed
        let! lowered = lowerMLIR ctx mlir
        let! optimized = optimizeCode ctx lowered
        let! llvmIR = convertToLLVMIR ctx optimized
        return! validateAndFinalize ctx llvmIR
    }

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