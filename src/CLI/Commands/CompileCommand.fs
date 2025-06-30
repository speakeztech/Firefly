module CLI.Commands.CompileCommand

#nowarn "57" // Suppress experimental FCS API warnings

open System
open System.IO
open System.Runtime.InteropServices
open Argu
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Text
open FSharp.Compiler.Syntax
open Core.Utilities.IntermediateWriter
open Core.Utilities.RemoveIntermediates
open Core.Utilities.ReadSourceFile
open Core.XParsec.Foundation
open Core.FCSProcessing.DependencyResolver
open Core.FCSProcessing.TypeExtractor
open Dabbit.CodeGeneration.TypeMapping
open Dabbit.CodeGeneration.MLIRModuleGenerator
open Dabbit.Bindings.SymbolRegistry
open Dabbit.Bindings.PatternLibrary
open Dabbit.Integration.AlloyBindings
open Dabbit.Analysis.ReachabilityAnalyzer
open Dabbit.Analysis.AstPruner
open Dabbit.Transformations.ASTTransformer
open Dabbit.Pipeline.FCSPipeline
open Dabbit.Pipeline.LoweringPipeline
open Dabbit.Pipeline.OptimizationPipeline
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
            | Target _ -> "Target platform (e.g., x86_64-unknown-linux-gnu)"
            | Optimize _ -> "Optimization level: none|less|default|aggressive|size"
            | Config _ -> "Path to project configuration file (.toml)"
            | Keep_Intermediates -> "Keep intermediate files (.ast, .mlir, .ll)"
            | Verbose -> "Enable verbose output"

/// Compilation context
type CompilationContext = {
    InputPath: string
    OutputPath: string
    Target: string
    OptimizeLevel: string
    Config: FireflyConfig
    KeepIntermediates: bool
    Verbose: bool
    IntermediatesDir: string
}

/// Helper functions
module Helpers =
    

    let getDefaultTarget() =
        if RuntimeInformation.IsOSPlatform OSPlatform.Windows then
            "x86_64-pc-windows-msvc"
        elif RuntimeInformation.IsOSPlatform OSPlatform.Linux then
            "x86_64-unknown-linux-gnu"
        elif RuntimeInformation.IsOSPlatform OSPlatform.OSX then
            "x86_64-apple-darwin"
        else
            "x86_64-unknown-unknown"

/// Phase 1: Parse and check source
let private parseAndCheck (ctx: CompilationContext) (sourceCode: string) : Async<CompilerResult<FSharpCheckFileResults * ParsedInput>> =
    async {
        printfn "Phase 1: Parsing and type checking..."
        
        let checker = FSharpChecker.Create(keepAssemblyContents = true)
        let sourceText = SourceText.ofString sourceCode
        let projectOptions = {
            ProjectFileName = "firefly.fsproj"
            ProjectId = None
            SourceFiles = [| ctx.InputPath |]
            OtherOptions = [| 
                "--target:exe"
                "--noframework"
                "--nowin32manifest"
                "--define:ZERO_ALLOCATION"
                "--define:FIDELITY"
            |]
            ReferencedProjects = [||]
            IsIncompleteTypeCheckEnvironment = false
            UseScriptResolutionRules = false
            LoadTime = DateTime.Now
            UnresolvedReferences = None
            OriginalLoadReferences = []
            Stamp = None
        }
        
        let! parseResult = checker.ParseFile(ctx.InputPath, sourceText, projectOptions)
        
        if parseResult.ParseHadErrors then
            let errors = parseResult.Diagnostics |> Array.map (fun d ->
                SyntaxError({ Line = d.StartLine; Column = d.StartColumn; File = ctx.InputPath; Offset = 0 },
                           d.Message,
                           ["parsing"]))
            return CompilerFailure (Array.toList errors)
        else
            let! checkAnswer = checker.CheckFileInProject(parseResult, ctx.InputPath, 0, sourceText, projectOptions)
            match checkAnswer with
            | FSharpCheckFileAnswer.Succeeded checkResults ->
                return Success (checkResults, parseResult.ParseTree)
            | _ ->
                return CompilerFailure [SyntaxError(
                    { Line = 0; Column = 0; File = ctx.InputPath; Offset = 0 },
                    "Type checking failed",
                    ["type checking"])]
    }

/// Phase 2: Process compilation unit with transformations
let private processAndTransform (ctx: CompilationContext) (checkResults: FSharpCheckFileResults, ast: ParsedInput) : Async<CompilerResult<_>> =
    async {
        printfn "Phase 2: Processing and transforming AST..."
        
        // Create processing context
        let typeCtx = TypeContextBuilder.create()
        let symbolRegistry = 
            let registry = Registry.createRegistry alloyPatterns
            registerAlloySymbols registry
            
        let processingCtx = {
            Checker = FSharpChecker.Create()
            Options = checkResults.ProjectContext.ProjectOptions
            TypeCtx = typeCtx
            SymbolRegistry = symbolRegistry
        }
        
        // Process the compilation unit (includes transformations and reachability)
        let! processed = processCompilationUnit processingCtx ast
        
        if ctx.KeepIntermediates then
            let baseName = Path.GetFileNameWithoutExtension(ctx.InputPath)
            writeFile ctx.IntermediatesDir baseName ".processed.fcs" (sprintf "%A" processed.Input)
        
        return Success processed
    }

/// Phase 3: Generate MLIR
let private generateMLIR (ctx: CompilationContext) (processed: {| Input: ParsedInput; TypeContext: TypeContext; Reachability: ReachabilityResult; Symbols: Map<string,obj> |}) : CompilerResult<string> =
    printfn "Phase 3: Generating MLIR..."
    
    try
        let mlirText = generateModuleFromAST processed.Input processed.TypeContext processed.Reachability.Registry
        
        if ctx.KeepIntermediates then
            let baseName = Path.GetFileNameWithoutExtension(ctx.InputPath)
            writeFile ctx.IntermediatesDir baseName ".mlir" mlirText
        
        Success mlirText
    with ex ->
        CompilerFailure [ConversionError("mlir_generation", "ast", "mlir", ex.Message)]

/// Phase 4-7: Lower, optimize, and convert to LLVM IR
let private lowerAndOptimize (ctx: CompilationContext) (mlirText: string) : CompilerResult<string> =
    printfn "Phase 4-7: Lowering and optimizing..."
    
    // Lower MLIR
    match applyLoweringPipeline mlirText with
    | CompilerFailure errors -> CompilerFailure errors
    | Success lowered ->
        // Optimize
        let optLevel = 
            match ctx.OptimizeLevel.ToLowerInvariant() with
            | "none" -> OptimizationLevel.Zero
            | "less" -> OptimizationLevel.Less
            | "aggressive" -> OptimizationLevel.Aggressive
            | "size" -> OptimizationLevel.Size
            | _ -> OptimizationLevel.Default
        
        let llvmOutput = {
            LLVMIRText = lowered
            ModuleName = Path.GetFileNameWithoutExtension(ctx.InputPath)
            OptimizationLevel = optLevel
            Metadata = Map.empty
        }
        
        let passes = createOptimizationPipeline optLevel
        match optimizeLLVMIR llvmOutput passes with
        | CompilerFailure errors -> CompilerFailure errors
        | Success optimized ->
            // Convert to LLVM IR
            match translateToLLVMIR optimized.LLVMIRText  (Some ctx.IntermediatesDir) with
            | CompilerFailure errors -> CompilerFailure errors
            | Success llvmIR ->
                if ctx.KeepIntermediates then
                    let baseName = Path.GetFileNameWithoutExtension(ctx.InputPath)
                    writeFile ctx.IntermediatesDir baseName ".ll" llvmIR
                Success llvmIR

/// Phase 8: Validate and finalize
let private validateAndFinalize (ctx: CompilationContext) (llvmIR: string) : CompilerResult<unit> =
    printfn "Phase 8: Validating and finalizing..."
    
    // Simple zero-allocation validation
    if llvmIR.Contains("@malloc") || llvmIR.Contains("@calloc") || llvmIR.Contains("@_Znwm") then
        CompilerFailure [ConversionError("validation", "llvm_ir", "zero_allocation", "Heap allocation detected")]
    elif llvmIR.Contains("@GC_") then
        CompilerFailure [ConversionError("validation", "llvm_ir", "zero_allocation", "GC allocation detected")]
    else
        printfn "✓ Zero-allocation guarantees verified"
        printfn "TODO: Call clang/lld to produce final executable"
        Success ()

/// Main compilation pipeline
let private compilePipeline (ctx: CompilationContext) (sourceCode: string) : Async<CompilerResult<unit>> =
    async {
        // Phase 1: Parse and check
        let! parseResult = parseAndCheck ctx sourceCode
        match parseResult with
        | CompilerFailure errors -> return CompilerFailure errors
        | Success (checkResults, ast) ->
            // Phase 2: Process and transform
            let! processResult = processAndTransform ctx (checkResults, ast)
            match processResult with
            | CompilerFailure errors -> return CompilerFailure errors
            | Success processed ->
                // Phase 3: Generate MLIR
                match generateMLIR ctx processed with
                | CompilerFailure errors -> return CompilerFailure errors
                | Success mlirText ->
                    // Phases 4-7: Lower and optimize
                    match lowerAndOptimize ctx mlirText with
                    | CompilerFailure errors -> return CompilerFailure errors
                    | Success llvmIR ->
                        // Phase 8: Validate and finalize
                        return validateAndFinalize ctx llvmIR
    }

/// Execute the compile command
let execute (args: ParseResults<CompileArgs>) =
    let inputPath = args.GetResult(Input)
    let outputPath = args.GetResult(Output)
    let target = args.TryGetResult(Target) |> Option.defaultValue (Helpers.getDefaultTarget())
    let optimize = args.TryGetResult(Optimize) |> Option.defaultValue "default"
    let configPath = args.TryGetResult(Config) |> Option.defaultValue "firefly.toml"
    let keepIntermediates = args.Contains(Keep_Intermediates)
    let verbose = args.Contains(Verbose)
    
    // Load configuration
    let config = 
        match loadAndValidateConfig configPath with
        | Success cfg -> cfg
        | CompilerFailure _ -> defaultConfig
    
    let intermediatesDir = 
        if keepIntermediates then
            let dir = Path.Combine(Path.GetDirectoryName(outputPath), "intermediates")
            Directory.CreateDirectory(dir) |> ignore
            Some dir
        else None
    
    let ctx = {
        InputPath = inputPath
        OutputPath = outputPath
        Target = target
        OptimizeLevel = optimize
        Config = config
        KeepIntermediates = keepIntermediates
        Verbose = verbose
        IntermediatesDir = intermediatesDir
    }
    
    printfn "Firefly Compiler v0.1.0"
    printfn "Compiling %s -> %s" inputPath outputPath
    printfn "Target: %s, Optimization: %s" target optimize
    
    match readSourceFile inputPath with
    | CompilerFailure errors ->
        errors |> List.iter (printfn "Error: %A")
        1
    | Success sourceCode ->
        match compilePipeline ctx sourceCode |> Async.RunSynchronously with
        | CompilerFailure errors ->
            errors |> List.iter (printfn "Error: %A")
            1
        | Success () ->
            printfn "✓ Compilation successful!"
            0