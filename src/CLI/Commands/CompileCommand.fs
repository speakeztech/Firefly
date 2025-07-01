module CLI.Commands.CompileCommand

#nowarn "57" // Suppress experimental FCS API warnings

open System
open System.IO
open System.Runtime.InteropServices
open System.Text.Json
open System.Text.Json.Serialization
open Argu
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Text
open FSharp.Compiler.Syntax
open Core.Utilities.IntermediateWriter
open Core.Utilities.RemoveIntermediates
open Core.Utilities.ReadSourceFile
open Core.XParsec.Foundation
open Core.FCSIngestion.FileLoader
open Core.FCSIngestion.SymbolExtraction
open Core.FCSIngestion.AstMerger
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

let private jsonOptions =
    let options = JsonSerializerOptions()
    options.Converters.Add(JsonFSharpConverter())
    options.WriteIndented <- true
    options

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
    IntermediatesDir: string option
}

type CompilationUnit = {
    Order: int
    File: string
    ParseResult: FSharpParseFileResults
    CheckResults: FSharpCheckFileResults
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

/// Phase 1: Parse and check source with all dependencies
let private parseAndCheck (ctx: CompilationContext) (sourceCode: string) : Async<CompilerResult<CompilationUnit list>> =
    async {
        printfn "Phase 1: Parsing and type checking..."
        
        let checker = FSharpChecker.Create(keepAssemblyContents = true)
        
        // Get project options using FileLoader
        let! (projectOptions, loadDiagnostics) = 
            loadProjectFiles ctx.InputPath checker ctx.IntermediatesDir
        
        // Log diagnostics
        if loadDiagnostics.Length > 0 then
            printfn "  Load diagnostics:"
            for diag in loadDiagnostics do
                printfn "    %s" diag.Message
        
        // Create directory structure if keeping intermediates
        match ctx.IntermediatesDir with
        | Some dir ->
            let rawUnitsDir = Path.Combine(dir, "fcs", "initial", "compilation_units")
            let rawSymbolsDir = Path.Combine(dir, "fcs", "initial", "symbol_tables")
            let summaryDir = Path.Combine(dir, "fcs", "summary")
            Directory.CreateDirectory(rawUnitsDir) |> ignore
            Directory.CreateDirectory(rawSymbolsDir) |> ignore
            Directory.CreateDirectory(summaryDir) |> ignore
        | None -> ()
        
        // Get parsing options
        let (parsingOptions, _) = checker.GetParsingOptionsFromProjectOptions projectOptions
        
        // Parse all files
        let! parseResults = 
            projectOptions.SourceFiles
            |> Array.mapi (fun order file ->
                async {
                    let! sourceText = 
                        match readSourceFile file with
                        | Success text -> async { return text }
                        | CompilerFailure _ -> failwith $"Failed to read {file}"
                    
                    let! parseResult = 
                        checker.ParseFile(file, SourceText.ofString sourceText, parsingOptions)
                    
                    // Write initial AST if keeping intermediates
                    match ctx.IntermediatesDir with
                    | Some dir ->
                        let rawUnitsDir = Path.Combine(dir, "fcs", "initial", "compilation_units")
                        let baseName = sprintf "%02d_%s" order (Path.GetFileName(file))
                        
                        // Write full AST
                        writeFile rawUnitsDir baseName ".ast" (sprintf "%A" parseResult.ParseTree)
                        
                        // Write AST metadata as JSON
                        let astMeta = {|
                            FileName = file
                            Order = order
                            HasErrors = parseResult.ParseHadErrors
                            DiagnosticsCount = parseResult.Diagnostics.Length
                            Diagnostics = parseResult.Diagnostics |> Array.map (fun d -> 
                                {| Message = d.Message; Line = d.StartLine; Column = d.StartColumn |})
                        |}
                        writeFile rawUnitsDir baseName ".ast.json" (JsonSerializer.Serialize(astMeta, jsonOptions))
                    | None -> ()
                    
                    return (order, file, parseResult)
                })
            |> Async.Sequential
        
        // Check for parse errors
        let parseErrors = 
            parseResults 
            |> Array.collect (fun (_, _, r) -> 
                if r.ParseHadErrors then r.Diagnostics else [||])
        
        if parseErrors.Length > 0 then
            let errors = parseErrors |> Array.map (fun d ->
                SyntaxError(
                    { Line = d.StartLine; Column = d.StartColumn; File = d.FileName; Offset = 0 },
                    d.Message,
                    ["parsing"]))
            return CompilerFailure (Array.toList errors)
        else
            // Type check each file in order
            let mutable cumulativeSymbolCount = 0
            let mutable checkResultsList = []
            
            for (order, file, parseResult) in parseResults do
                let! checkAnswer = 
                    checker.CheckFileInProject(
                        parseResult,
                        file,
                        0,
                        SourceText.ofString (File.ReadAllText(file)),
                        projectOptions)
                
                match checkAnswer with
                | FSharpCheckFileAnswer.Succeeded checkResults ->
                    // Write symbols if keeping intermediates
                    match ctx.IntermediatesDir with
                    | Some dir ->
                        let baseName = sprintf "%02d_%s" order (Path.GetFileName(file))
                        
                        // Write defined symbols
                        let definedSymbols = extractDefinedSymbols checkResults file
                        let rawUnitsDir = Path.Combine(dir, "fcs", "initial", "compilation_units")
                        writeFile rawUnitsDir baseName ".symbols.json" (JsonSerializer.Serialize(definedSymbols, jsonOptions))
                        
                        // Write cumulative symbol table
                        let cumulativeSymbols = extractCumulativeSymbols checkResults
                        cumulativeSymbolCount <- cumulativeSymbols.TotalModules + cumulativeSymbols.TotalTypes + cumulativeSymbols.TotalFunctions
                        
                        let rawSymbolsDir = Path.Combine(dir, "fcs", "initial", "symbol_tables")
                        let symbolFileName = sprintf "after_%02d_%s" order (Path.GetFileNameWithoutExtension(file))
                        writeFile rawSymbolsDir symbolFileName ".json" (JsonSerializer.Serialize(cumulativeSymbols, jsonOptions))
                        
                        printfn "    %s: %d symbols defined, %d total symbols available" 
                            (Path.GetFileName(file)) 
                            (Array.length definedSymbols.Modules + Array.length definedSymbols.Types + Array.length definedSymbols.Functions)
                            cumulativeSymbolCount
                    | None -> ()
                    
                    checkResultsList <- checkResultsList @ [(order, file, parseResult, Some checkResults)]
                | _ ->
                    checkResultsList <- checkResultsList @ [(order, file, parseResult, None)]
            
            // Merge ASTs
            let successfulParseResults = 
                checkResultsList 
                |> List.choose (fun (_, _, parseResult, checkResult) -> 
                    match checkResult with 
                    | Some _ -> Some parseResult 
                    | None -> None)
                |> Array.ofList
                  
            // Write final compilation state
            match ctx.IntermediatesDir with
            | Some dir ->
                let summaryDir = Path.Combine(dir, "fcs", "summary")
                
                // Write compilation state
                let compilationState = {|
                    TotalFiles = projectOptions.SourceFiles.Length
                    SuccessfulFiles = successfulParseResults.Length
                    TotalSymbols = cumulativeSymbolCount
                    Timestamp = DateTime.UtcNow
                    Files = checkResultsList |> List.map (fun (order, file, _, checkResult) ->
                        {| 
                            Order = order
                            FileName = file
                            ShortName = Path.GetFileName(file : string)  // Type annotation fixes error 3
                            Success = Option.isSome checkResult
                        |})
                |}
                writeFile summaryDir "compilation_state" ".json" (JsonSerializer.Serialize(compilationState, jsonOptions))
            | None -> ()
            
            let compilationUnits = 
                checkResultsList 
                |> List.choose (fun (order, file, parseResult, checkResult) ->
                    match checkResult with
                    | Some cr -> Some { Order = order; File = file; ParseResult = parseResult; CheckResults = cr }
                    | None -> None)
            
            match compilationUnits with
            | [] ->
                return CompilerFailure [SyntaxError(
                    { Line = 0; Column = 0; File = ctx.InputPath; Offset = 0 },
                    "No files successfully type checked",
                    ["type checking"])]
            | units ->
                return Success units
    }

/// Phase 2: Process compilation unit with transformations
let private processAndTransform (ctx: CompilationContext) (compilationUnits: CompilationUnit list) : Async<CompilerResult<_>> =
    async {
        printfn "Phase 2: Processing and transforming AST..."
        printfn "  Processing %d compilation units" compilationUnits.Length
        
        // Get project options from the first compilation unit
        let projectOptions = 
            match compilationUnits with
            | [] -> failwith "No compilation units"
            | firstUnit :: _ -> firstUnit.CheckResults.ProjectContext.ProjectOptions
        
        // Create processing context
        let typeCtx = TypeContextBuilder.create()
        let symbolRegistry = 
            let registry = Registry.createRegistry alloyPatterns
            registerAlloySymbols registry |> ignore
            registry
            
        let processingCtx = {
            Checker = FSharpChecker.Create()
            Options = projectOptions
            TypeCtx = typeCtx
            SymbolRegistry = symbolRegistry
        }
        
        // Process each compilation unit with detailed error tracking
        let! processedUnits = 
            compilationUnits
            |> List.map (fun unit ->
                async {
                    try
                        printfn "  Processing unit: %s" (Path.GetFileName(unit.File))
                        let! processed = 
                            try
                                processCompilationUnit processingCtx unit.ParseResult.ParseTree
                            with
                            | :? MatchFailureException as ex ->
                                printfn "    ERROR: Match failure in %s" (Path.GetFileName(unit.File))
                                printfn "    Match failed at: %s:%d" ex.Data0 ex.Data1
                                printfn "    Message: %s" ex.Message
                                reraise()
                        printfn "    Successfully processed %s" (Path.GetFileName(unit.File))
                        return Some (unit.File, processed)
                    with
                    | :? MatchFailureException as ex ->
                        printfn "    ERROR: Match failure in %s" (Path.GetFileName(unit.File))
                        printfn "    Stack trace: %s" ex.StackTrace
                        printfn "    At: %s:%d" ex.Data0 ex.Data1
                        return None
                    | ex ->
                        printfn "    ERROR: %s in %s" ex.Message (Path.GetFileName(unit.File))
                        printfn "    Stack trace: %s" ex.StackTrace
                        return None
                })
            |> Async.Sequential
        
        let successfulUnits = processedUnits |> Array.choose id
        
        if successfulUnits.Length = 0 then
            return CompilerFailure [InternalError("processAndTransform", "All units failed to process", None)]
        else
            printfn "  Successfully processed %d/%d units" successfulUnits.Length compilationUnits.Length
            
            // Write intermediate results if keeping intermediates
            if ctx.KeepIntermediates then
                match ctx.IntermediatesDir with
                | Some dir ->
                    let prunedDir = Path.Combine(dir, "fcs", "pruned", "compilation_units")
                    Directory.CreateDirectory(prunedDir) |> ignore
                    
                    successfulUnits |> Array.iteri (fun i (file, processed) ->
                        let baseName = sprintf "%02d_%s" i (Path.GetFileName(file))
                        writeFile prunedDir baseName ".pruned.ast" (sprintf "%A" processed.Input)
                    )
                | None -> ()
            
            // For now, return the last processed unit
            match Array.tryLast successfulUnits with
            | Some (_, processed) -> return Success (processed, symbolRegistry)
            | None -> return CompilerFailure [InternalError("processAndTransform", "No units to return", None)]
    }

/// Phase 3: Generate MLIR
let private generateMLIR (ctx: CompilationContext) (processed: {| Input: ParsedInput; TypeContext: TypeContext; Reachability: ReachabilityResult; Symbols: Map<string,obj> |}, symbolRegistry) : CompilerResult<string> =
    printfn "Phase 3: Generating MLIR..."
    
    try
        let mlirText = generateModuleFromAST processed.Input processed.TypeContext symbolRegistry
        
        if ctx.KeepIntermediates then
            let dir = 
                match ctx.IntermediatesDir with
                | Some d -> d
                | None -> failwith "IntermediatesDir should be Some when KeepIntermediates is true"
            let baseName = Path.GetFileNameWithoutExtension(ctx.InputPath)
            writeFile dir baseName ".mlir" mlirText
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
            match translateToLLVMIR optimized.LLVMIRText ctx.IntermediatesDir with
            | CompilerFailure errors -> CompilerFailure errors
            | Success llvmIR ->
                if ctx.KeepIntermediates then
                        let dir = 
                            match ctx.IntermediatesDir with
                            | Some d -> d
                            | None -> failwith "IntermediatesDir should be Some when KeepIntermediates is true"
                        let baseName = Path.GetFileNameWithoutExtension(ctx.InputPath)
                        writeFile dir baseName ".ll" llvmIR
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
        | Success compilationUnits ->  // Changed from (checkResults, ast)
            // Phase 2: Process and transform
            let! processResult = processAndTransform ctx compilationUnits  // Pass the list
            match processResult with
            | CompilerFailure errors -> return CompilerFailure errors
            | Success (processed, symbolRegistry) ->
                // Phase 3: Generate MLIR
                match generateMLIR ctx (processed, symbolRegistry) with
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
    
    // Clear intermediates directory before starting
    prepareIntermediatesDirectory intermediatesDir
    
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