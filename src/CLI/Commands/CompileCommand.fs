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
open Dabbit.Analysis.DependencyGraphBuilder
open Dabbit.Analysis.AstPruner
open Dabbit.Transformations.ClosureElimination
open Dabbit.Transformations.StackAllocation

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
            | Keep_Intermediates -> "Keep intermediate files (.raw.fcs, .sm.fcs, .ra.fcs, MLIR, LLVM IR)"
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

/// Combined parsed inputs with dependency information
type ParsedProgram = {
    MainFile: string
    ParsedInputs: (string * ParsedInput) list  // (filepath, parsed AST)
    Dependencies: string list
    Checker: FSharpChecker
    ProjectOptions: FSharpProjectOptions
}

/// Refined AST after .NET construct removal
type RefinedProgram = {
    MainFile: string
    RefinedInputs: (string * ParsedInput) list
    RemovedConstructs: string list  // For diagnostics
}

/// Reachable program after tree-shaking
type ReachableProgram = {
    MainFile: string
    ReachableInputs: (string * ParsedInput) list
    ReachabilityStats: ReachabilityStats
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

/// Extract opened module names from parsed input
let private extractOpenedModules (input: ParsedInput) : string list =
    match input with
    | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
        modules 
        |> List.collect (fun (SynModuleOrNamespace(_, _, _, decls, _, _, _, _, _)) ->
            decls |> List.choose (function
                | SynModuleDecl.Open(target, _) ->
                    match target with
                    | SynOpenDeclTarget.ModuleOrNamespace(SynLongIdent(ids, _, _), _) ->
                        let modulePath = ids |> List.map (fun id -> id.idText) |> String.concat "."
                        Some modulePath
                    | _ -> None
                | _ -> None))
    | _ -> []

/// Resolve module name to file path
let private resolveModulePath (moduleName: string) : string option =
    let possiblePaths = [
        // Look in lib/Alloy for Alloy modules
        Path.Combine("lib", "Alloy", moduleName.Replace(".", Path.DirectorySeparatorChar.ToString()) + ".fs")
        // Look in lib for other modules
        Path.Combine("lib", moduleName.Replace(".", Path.DirectorySeparatorChar.ToString()) + ".fs")
        // Look in current directory
        moduleName.Replace(".", Path.DirectorySeparatorChar.ToString()) + ".fs"
        // Handle nested modules (e.g., Alloy.IO.Console -> Alloy/IO.fs)
        let parts = moduleName.Split('.')
        if parts.Length > 2 then
            Path.Combine("lib", parts.[0], String.concat (Path.DirectorySeparatorChar.ToString()) parts.[1..parts.Length-2] + ".fs")
        else
            ""
    ]
    
    possiblePaths 
    |> List.filter (fun p -> not (String.IsNullOrEmpty(p)))
    |> List.tryFind File.Exists

/// Phase 1: Parse F# source including all dependencies
let private parseSource (ctx: CompilationContext) (sourceCode: string) : CompilerResult<ParsedProgram> =
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
    
    // Parse main file
    let parseResults = checker.ParseFile(ctx.InputPath, sourceText, parsingOptions) |> Async.RunSynchronously
    
    if parseResults.ParseHadErrors then
        let errors = parseResults.Diagnostics |> Array.map (fun d -> d.Message) |> String.concat "\n"
        CompilerFailure [SyntaxError({ Line = 0; Column = 0; File = ctx.InputPath; Offset = 0 }, errors, ["FCS parsing"])]
    else
        let mainParsedInput = parseResults.ParseTree
        let dependencies = extractOpenedModules mainParsedInput
        
        // Parse each dependency
        let mutable allParsedInputs = [(ctx.InputPath, mainParsedInput)]
        let mutable failedDeps = []
        
        for dep in dependencies do
            match resolveModulePath dep with
            | Some filePath ->
                try
                    let depSource = File.ReadAllText(filePath)
                    let depSourceText = SourceText.ofString depSource
                    let depParseResults = checker.ParseFile(filePath, depSourceText, parsingOptions) |> Async.RunSynchronously
                    
                    if not depParseResults.ParseHadErrors then
                        allParsedInputs <- allParsedInputs @ [(filePath, depParseResults.ParseTree)]
                    else
                        failedDeps <- failedDeps @ [sprintf "%s: parse errors" dep]
                with ex ->
                    failedDeps <- failedDeps @ [sprintf "%s: %s" dep ex.Message]
            | None ->
                failedDeps <- failedDeps @ [sprintf "%s: file not found" dep]
        
        // Log any dependency issues but don't fail compilation
        if not failedDeps.IsEmpty && ctx.Verbose then
            printfn "  Warning: Could not parse some dependencies:"
            failedDeps |> List.iter (printfn "    - %s")
        
        let parsedProgram = {
            MainFile = ctx.InputPath
            ParsedInputs = allParsedInputs
            Dependencies = dependencies
            Checker = checker
            ProjectOptions = projectOptions
        }
        
        // Write raw FCS output with all dependencies
        if ctx.KeepIntermediates then
            writeIntermediateFile ctx.IntermediatesDir 
                (Path.GetFileNameWithoutExtension ctx.InputPath) 
                ".raw.fcs" 
                (sprintf "%A" parsedProgram)
        
        Success parsedProgram

/// Remove .NET-specific constructs from AST
let private removeNetConstructs (input: ParsedInput) : ParsedInput * string list =
    let mutable removedConstructs = []
    
    let rec transformExpr expr =
        match expr with
        // Remove array creation expressions that use .NET allocations
        | SynExpr.ArrayOrList(isArray, elements, range) when isArray ->
            removedConstructs <- "Array literal allocation" :: removedConstructs
            // Convert to stack allocation pattern if small enough
            if elements.Length <= 16 then
                SynExpr.App(
                    ExprAtomicFlag.NonAtomic, false,
                    SynExpr.Ident(Ident("stackalloc", range)),
                    SynExpr.Const(SynConst.Int32 elements.Length, range),
                    range)
            else
                SynExpr.Const(SynConst.Unit, range)  // Remove large allocations
        
        // Remove list expressions
        | SynExpr.ArrayOrList(isArray, elements, range) when not isArray ->
            removedConstructs <- "List allocation" :: removedConstructs
            SynExpr.Const(SynConst.Unit, range)
        
        // Remove object expressions - FCS 43.9.300 expects 8 arguments
        | SynExpr.ObjExpr(objType, argOpt, withKeyword, bindings, members, extraImpls, newExprRange, range) ->
            removedConstructs <- "Object expression" :: removedConstructs
            SynExpr.Const(SynConst.Unit, range)
        
        // Remove new expressions for reference types
        | SynExpr.New(isProtected, targetType, expr, range) ->
            removedConstructs <- "New expression" :: removedConstructs
            transformExpr expr  // Keep the argument expression
        
        // Recursively transform nested expressions
        | SynExpr.App(flag, isInfix, funcExpr, argExpr, range) ->
            SynExpr.App(flag, isInfix, transformExpr funcExpr, transformExpr argExpr, range)
        
        | SynExpr.LetOrUse(isRec, isUse, bindings, body, range, trivia) ->
            let transformedBindings = bindings |> List.map transformBinding
            SynExpr.LetOrUse(isRec, isUse, transformedBindings, transformExpr body, range, trivia)
        
        | SynExpr.Sequential(debugPoint, isTrueSeq, expr1, expr2, range, trivia) ->
            SynExpr.Sequential(debugPoint, isTrueSeq, transformExpr expr1, transformExpr expr2, range, trivia)
        
        | SynExpr.IfThenElse(ifExpr, thenExpr, elseExprOpt, spIfToThen, isFromTry, range, trivia) ->
            SynExpr.IfThenElse(
                transformExpr ifExpr,
                transformExpr thenExpr,
                Option.map transformExpr elseExprOpt,
                spIfToThen, isFromTry, range, trivia)
        
        | SynExpr.Match(spMatch, matchExpr, clauses, range, trivia) ->
            let transformedClauses = clauses |> List.map (fun (SynMatchClause(pat, whenExpr, resultExpr, range, spTarget, trivia)) ->
                SynMatchClause(pat, Option.map transformExpr whenExpr, transformExpr resultExpr, range, spTarget, trivia))
            SynExpr.Match(spMatch, transformExpr matchExpr, transformedClauses, range, trivia)
        
        | SynExpr.Lambda(fromMethod, inLambdaSeq, args, body, parsedData, range, trivia) ->
            SynExpr.Lambda(fromMethod, inLambdaSeq, args, transformExpr body, parsedData, range, trivia)
        
        | _ -> expr
    
    and transformBinding (SynBinding(access, kind, isInline, isMutable, attrs, xmlDoc, valData, pat, returnInfo, expr, range, sp, trivia)) =
        SynBinding(access, kind, isInline, isMutable, attrs, xmlDoc, valData, pat, returnInfo, transformExpr expr, range, sp, trivia)
    
    let rec transformDecl decl =
        match decl with
        | SynModuleDecl.Let(isRec, bindings, range) ->
            SynModuleDecl.Let(isRec, bindings |> List.map transformBinding, range)
        
        | SynModuleDecl.Types(typeDefs, range) ->
            // Remove class types, keep only records and unions
            let filteredTypeDefs = typeDefs |> List.filter (fun (SynTypeDefn(componentInfo, typeRepr, members, implicitCtor, range, trivia)) ->
                match typeRepr with
                | SynTypeDefnRepr.ObjectModel(kind, members, range) ->
                    match kind with
                    | SynTypeDefnKind.Class -> 
                        removedConstructs <- "Class definition" :: removedConstructs
                        false
                    | _ -> true
                | _ -> true)
            SynModuleDecl.Types(filteredTypeDefs, range)
        
        | SynModuleDecl.NestedModule(componentInfo, isRec, decls, range, trivia, moduleKeyword) ->
            SynModuleDecl.NestedModule(componentInfo, isRec, decls |> List.map transformDecl, range, trivia, moduleKeyword)
        
        | _ -> decl
    
    match input with
    | ParsedInput.ImplFile(ParsedImplFileInput(fileName, isScript, qualName, pragmas, directives, modules, isLast, trivia, ids)) ->
        let transformedModules = modules |> List.map (fun (SynModuleOrNamespace(longId, isRec, kind, decls, xmlDoc, attrs, access, range, trivia)) ->
            let transformedDecls = decls |> List.map transformDecl
            SynModuleOrNamespace(longId, isRec, kind, transformedDecls, xmlDoc, attrs, access, range, trivia))
        
        let transformedInput = ParsedInput.ImplFile(
            ParsedImplFileInput(fileName, isScript, qualName, pragmas, directives, transformedModules, isLast, trivia, ids))
        
        (transformedInput, removedConstructs)
    
    | sig_ -> (sig_, removedConstructs)

/// Phase 2a: Remove .NET constructs to create smaller AST
let private refineAST (ctx: CompilationContext) (parsedProgram: ParsedProgram) : CompilerResult<RefinedProgram> =
    printfn "Phase 2a: Removing .NET constructs..."
    
    let refinedInputs = 
        parsedProgram.ParsedInputs 
        |> List.map (fun (path, input) ->
            let (refined, removed) = removeNetConstructs input
            if ctx.Verbose && not removed.IsEmpty then
                printfn "  Removed from %s: %s" (Path.GetFileName(path)) (String.concat ", " (removed |> List.distinct))
            (path, refined))
    
    let allRemoved = 
        parsedProgram.ParsedInputs 
        |> List.collect (fun (_, input) -> snd (removeNetConstructs input))
        |> List.distinct
    
    let refinedProgram = {
        MainFile = parsedProgram.MainFile
        RefinedInputs = refinedInputs
        RemovedConstructs = allRemoved
    }
    
    // Write smaller FCS output
    if ctx.KeepIntermediates then
        writeIntermediateFile ctx.IntermediatesDir 
            (Path.GetFileNameWithoutExtension ctx.InputPath) 
            ".sm.fcs" 
            (sprintf "%A" refinedProgram)
    
    Success refinedProgram

/// Build dependency graph from refined AST (without FSharpSymbol objects)
let private buildDependencyGraph (refinedProgram: RefinedProgram) : CompilerResult<(Set<string> * Map<string, Set<string>> * Set<string>)> =
    let mutable allSymbols = Set.empty
    let mutable edges = Map.empty
    let mutable entryPoints = Set.empty
    
    // Process each input file
    for (filePath, input) in refinedProgram.RefinedInputs do
        match input with
        | ParsedInput.ImplFile(ParsedImplFileInput(_, _, qualName, _, _, modules, _, _, _)) ->
            for (SynModuleOrNamespace(longId, _, _, decls, _, _, _, _, _)) in modules do
                let moduleName = longId |> List.map (fun id -> id.idText) |> String.concat "."
                
                // Build graph from module declarations
                let moduleGraph = buildFromModule moduleName decls
                
                // Collect all symbols and merge edges
                for KeyValue(node, deps) in moduleGraph.Edges do
                    allSymbols <- Set.add node allSymbols
                    edges <- Map.add node deps edges
                    // Also add dependencies as symbols
                    for dep in deps do
                        allSymbols <- Set.add dep allSymbols
                
                // Find entry points
                for decl in decls do
                    match decl with
                    | SynModuleDecl.Let(_, bindings, _) ->
                        for binding in bindings do
                            let (SynBinding(_, _, _, _, attrs, _, _, pat, _, _, _, _, _)) = binding
                            let hasEntryPoint = attrs |> List.exists (fun attrList ->
                                attrList.Attributes |> List.exists (fun attr ->
                                    match attr.TypeName with
                                    | SynLongIdent([ident], _, _) -> ident.idText = "EntryPoint"
                                    | _ -> false))
                            
                            if hasEntryPoint then
                                match pat with
                                | SynPat.Named(SynIdent(ident, _), _, _, _) ->
                                    let fullName = sprintf "%s.%s" moduleName ident.idText
                                    entryPoints <- Set.add fullName entryPoints
                                    allSymbols <- Set.add fullName allSymbols
                                | SynPat.LongIdent(SynLongIdent(ids, _, _), _, _, _, _, _) ->
                                    let name = ids |> List.map (fun id -> id.idText) |> String.concat "."
                                    let fullName = sprintf "%s.%s" moduleName name
                                    entryPoints <- Set.add fullName entryPoints
                                    allSymbols <- Set.add fullName allSymbols
                                | _ -> ()
                    | _ -> ()
        | _ -> ()
    
    Success (allSymbols, edges, entryPoints)

/// Phase 2b: Perform reachability analysis (tree-shaking)
let private performReachabilityAnalysis (ctx: CompilationContext) (refinedProgram: RefinedProgram) : CompilerResult<ReachableProgram> =
    printfn "Phase 2b: Performing reachability analysis..."
    
    // Use the new reachability analyzer that works directly with parsed inputs
    let reachabilityResult = Dabbit.Analysis.ReachabilityAnalyzer.analyzeFromParsedInputs refinedProgram.RefinedInputs
    
    // Print statistics
    let stats = reachabilityResult.Statistics
    if ctx.Verbose then
        printfn "  Total symbols: %d" stats.TotalSymbols
        printfn "  Reachable symbols: %d" stats.ReachableSymbols
        printfn "  Eliminated symbols: %d" stats.EliminatedSymbols
    
    // Prune unreachable code from AST
    let reachableInputs = 
        refinedProgram.RefinedInputs
        |> List.map (fun (path, input) ->
            let pruned = prune reachabilityResult.Reachable input
            (path, pruned))
    
    let reachableProgram = {
        MainFile = refinedProgram.MainFile
        ReachableInputs = reachableInputs
        ReachabilityStats = stats
    }
    
    // Write reachability analysis result
    if ctx.KeepIntermediates then
        writeIntermediateFile ctx.IntermediatesDir 
            (Path.GetFileNameWithoutExtension ctx.InputPath) 
            ".ra.fcs" 
            (sprintf "%A" reachableProgram)
    
    Success reachableProgram

/// Phase 3: Transform AST 
let private transformASTPhase (ctx: CompilationContext) (reachableProgram: ReachableProgram, checker: FSharpChecker, projectOptions: FSharpProjectOptions) : CompilerResult<ReachableProgram * TypeContext * SymbolRegistry> =
    printfn "Phase 3: Transforming reachable AST..."
    let typeCtx = TypeContextBuilder.create()
    
    match RegistryConstruction.buildAlloyRegistry() with
    | CompilerFailure errors -> CompilerFailure errors
    | Success symbolRegistry ->
        // Transform all reachable inputs while preserving the structure
        let transformCtx = {
            TypeContext = typeCtx
            SymbolRegistry = symbolRegistry
            Reachability = {
                Reachable = reachableProgram.ReachabilityStats.ReachableSymbols |> Set.ofList
                UnionCases = Map.empty
                Statistics = reachableProgram.ReachabilityStats
            }
            ClosureState = { Counter = 0; Scope = Set.empty; Lifted = [] }
        }
        
        // Transform each reachable input
        let transformedInputs = 
            reachableProgram.ReachableInputs 
            |> List.map (fun (path, input) ->
                let transformedAST = transformAST transformCtx input
                (path, transformedAST)
            )
        
        // Create transformed reachable program
        let transformedReachableProgram = {
            MainFile = reachableProgram.MainFile
            ReachableInputs = transformedInputs
            ReachabilityStats = reachableProgram.ReachabilityStats
        }
        
        Success (transformedReachableProgram, typeCtx, symbolRegistry)

/// Phase 4: Generate MLIR
let private generateMLIR (ctx: CompilationContext) (reachableProgram: ReachableProgram, typeCtx: TypeContext, symbolRegistry: SymbolRegistry) : CompilerResult<string> =
    printfn "Phase 4: Generating MLIR..."
    let baseName = Path.GetFileNameWithoutExtension(ctx.InputPath)
    
    // Pass all reachable inputs to the generator
    let mlirText = Core.MLIRGeneration.DirectGenerator.generateProgram baseName typeCtx symbolRegistry reachableProgram.ReachableInputs
    
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
        let! refined = refineAST ctx parsed
        let! reachable = performReachabilityAnalysis ctx refined
        let! (transformedReachable, typeCtx, symbolRegistry) = transformASTPhase ctx (reachable, parsed.Checker, parsed.ProjectOptions)
        let! mlir = generateMLIR ctx (transformedReachable, typeCtx, symbolRegistry)
        
        // Phase 5: Lower MLIR
        let! lowered = 
            match Core.Conversion.LoweringPipeline.applyLoweringPipeline mlir with
            | Success result -> Success result
            | CompilerFailure errors -> CompilerFailure errors
        
        // Phase 6: Optimize
        let optLevel = 
            match ctx.OptimizeLevel.ToLowerInvariant() with
            | "none" -> Core.Conversion.OptimizationPipeline.OptimizationLevel.Zero
            | "less" -> Core.Conversion.OptimizationPipeline.OptimizationLevel.Less
            | "aggressive" -> Core.Conversion.OptimizationPipeline.OptimizationLevel.Aggressive
            | "size" -> Core.Conversion.OptimizationPipeline.OptimizationLevel.Size
            | _ -> Core.Conversion.OptimizationPipeline.OptimizationLevel.Default
        
        let llvmOutput = {
            Core.Conversion.OptimizationPipeline.LLVMIRText = lowered
            ModuleName = Path.GetFileNameWithoutExtension(ctx.InputPath)
            OptimizationLevel = optLevel
            Metadata = Map.empty
        }
        
        let! optimized = 
            let passes = Core.Conversion.OptimizationPipeline.createOptimizationPipeline optLevel
            Core.Conversion.OptimizationPipeline.optimizeLLVMIR llvmOutput passes
        
        // Phase 7: Convert to LLVM IR
        let! llvmIR = 
            match Core.Conversion.LoweringPipeline.translateToLLVMIR optimized.LLVMIRText ctx.IntermediatesDir with
            | Success result -> Success result
            | CompilerFailure errors -> CompilerFailure errors
        
        // Phase 8: Validate
        return! 
            match Core.Conversion.OptimizationPipeline.validateZeroAllocationGuarantees llvmIR with
            | Success () ->
                printfn "✓ Zero-allocation guarantees verified"
                printfn "Phase 8: Invoking external tools..."
                printfn "TODO: Call clang/lld to produce final executable"
                Success ()
            | CompilerFailure errors -> CompilerFailure errors
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