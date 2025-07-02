module Dabbit.Pipeline.CompilationOrchestrator

open System
open System.IO
open FSharp.Compiler.Syntax
open FSharp.Compiler.CodeAnalysis
open Core.XParsec.Foundation
open Core.Utilities.IntermediateWriter
open Core.FCSIngestion.SymbolExtraction
open Dabbit.Pipeline.CompilationTypes
open Dabbit.CodeGeneration.TypeMapping
open Dabbit.Bindings.SymbolRegistry
open Dabbit.Bindings.SymbolRegistry.Registry
open Dabbit.Bindings.PatternLibrary
open Dabbit.Integration.AlloyBindings
open Dabbit.Analysis.CompilationUnit
open Dabbit.Analysis.ReachabilityAnalyzer
open Dabbit.Analysis.AstPruner
open Dabbit.CodeGeneration.MLIRModuleGenerator
open FCSPipeline

/// Cumulative state maintained during Phase 1 (FCSIngestion)
type FCSIngestionState = {
    ParsedFiles: (int * string * ParsedInput) list
    CheckedFiles: (string * ParsedInput * FSharpCheckFileResults) list
    SymbolRegistry: SymbolRegistry
    TypeContext: TypeContext
    CumulativeProjectOptions: FSharpProjectOptions
    IntermediateStructure: IntermediateFileStructure option
    FileSymbols: Map<string, ExtractedSymbol list>
}


/// Detect script-style entry points
let private detectScriptEntryPoints (ast: ParsedInput) : Set<string> =
    match ast with
    | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
        modules 
        |> List.collect (fun (SynModuleOrNamespace(moduleIds, _, _, decls, _, _, _, _, _)) ->
            let modulePath = moduleIds |> List.map (fun id -> id.idText) |> String.concat "."
            decls |> List.collect (fun decl ->
                match decl with
                | SynModuleDecl.Let(_, bindings, _) ->
                    bindings 
                    |> List.choose (fun binding ->
                        match binding with
                        | SynBinding(_, _, _, _, _, _, valData, pat, _, _, _, _, _) ->
                            match valData with
                            | SynValData(_, SynValInfo([], _), _) ->
                                // No parameters = value binding = potential entry point
                                match pat with
                                | SynPat.Named(SynIdent(id, _), _, _, _) ->
                                    if modulePath = "" then Some id.idText
                                    else Some (sprintf "%s.%s" modulePath id.idText)
                                | SynPat.LongIdent(SynLongIdent(ids, _, _), _, _, _, _, _) ->
                                    let name = ids |> List.map (fun id -> id.idText) |> String.concat "."
                                    if modulePath = "" then Some name
                                    else Some (sprintf "%s.%s" modulePath name)
                                | _ -> None
                            | _ -> None)
                | SynModuleDecl.Expr(_, _) ->
                    // Top-level expression is an entry point (includes do-expressions)
                    [sprintf "%s.<script_expr>" modulePath]
                | _ -> []))
        |> Set.ofList
    | _ -> Set.empty

// Add these new functions after the existing imports in CompilationOrchestrator.fs

/// ===== 5-PASS PIPELINE IMPLEMENTATION =====

/// Pass 1: Symbol Discovery
let private executeSymbolDiscovery (compilationUnit: CompilationUnit) (progress: ProgressCallback) : Async<CompilerResult<SymbolDiscoveryResult>> =
    async {
        progress SymbolDiscovery "Starting symbol discovery across all files"
        
        let mutable fileSymbols : Map<string, ExtractedSymbol list> = Map.empty
        let mutable modulePaths = Map.empty
        let mutable entryPoints = Set.empty
        
        // Process each file
        for KeyValue(filePath, sourceFile) in compilationUnit.SourceFiles do
            progress SymbolDiscovery (sprintf "Discovering symbols in %s" (Path.GetFileName filePath))
            
            // Use existing symbol extraction
            let symbols = Core.FCSIngestion.SymbolExtraction.extractSymbolsFromParsedInput sourceFile.Ast
            fileSymbols <- Map.add filePath symbols fileSymbols
            
            // Extract module path
            let modulePath = extractModulePath sourceFile.Ast
            modulePaths <- Map.add filePath modulePath modulePaths
            
            // Find entry points - FIXED VERSION
            let fileEntryPoints = detectScriptEntryPoints sourceFile.Ast
            
            // For the main file, also check for script-style execution
            if filePath = compilationUnit.MainFile then
                // Find any function named "main" as an entry point
                let mainFunctions = 
                    symbols 
                    |> List.filter (fun s -> s.ShortName = "main")
                    |> List.map (fun s -> s.QualifiedName)
                    |> Set.ofList
                entryPoints <- Set.union entryPoints (Set.union fileEntryPoints mainFunctions)
            else
                entryPoints <- Set.union entryPoints fileEntryPoints
        
        progress SymbolDiscovery (sprintf "Discovered %d total symbols across %d files" 
            (fileSymbols |> Map.toSeq |> Seq.sumBy (fun (_, syms) -> List.length syms))
            (Map.count fileSymbols))
        
        return Success {
            ParsedFiles = compilationUnit.SourceFiles |> Map.map (fun _ sf -> sf.Ast)
            FileSymbols = fileSymbols
            ModulePaths = modulePaths
            EntryPoints = entryPoints
        }
    }

/// Pass 2: Import Resolution
let private executeImportResolution (discovery: SymbolDiscoveryResult) (progress: ProgressCallback) : Async<CompilerResult<ImportResolutionResult>> =
    async {
        progress ImportResolution "Starting import resolution and scope building"
        
        let mutable fileOpenedModules = Map.empty
        let mutable fileScopes = Map.empty
        
        // Build global symbol table first
        let globalSymbolTable = 
            discovery.FileSymbols
            |> Map.toSeq
            |> Seq.collect (fun (_, symbols) -> symbols)
            |> List.ofSeq
            |> groupSymbolsByModule
        
        // Process each file's imports
        for KeyValue(filePath, ast) in discovery.ParsedFiles do
            progress ImportResolution (sprintf "Resolving imports in %s" (Path.GetFileName filePath))
            
            match ast with
            | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
                for (SynModuleOrNamespace(_, _, _, decls, _, _, _, _, _)) in modules do
                    let openedModules = extractOpenedModules decls
                    fileOpenedModules <- Map.add filePath openedModules fileOpenedModules
                    
                    let modulePath = Map.tryFind filePath discovery.ModulePaths |> Option.defaultValue []
                    let scope = {
                        CurrentModule = modulePath
                        OpenedModules = openedModules
                    }
                    fileScopes <- Map.add filePath scope fileScopes
            | _ -> ()
        
        progress ImportResolution (sprintf "Resolved imports for %d files" (Map.count fileScopes))
        
        return Success {
            Discovery = discovery
            FileOpenedModules = fileOpenedModules
            FileScopes = fileScopes
            GlobalSymbolTable = globalSymbolTable
        }
    }

/// Pass 3: Type Resolution
let private executeTypeResolution (imports: ImportResolutionResult) (checker: FSharpChecker) (progress: ProgressCallback) : Async<CompilerResult<TypeResolutionResult>> =
    async {
        progress TypeResolution "Starting type resolution for all expressions"
        
        // TODO: Implement type resolution
        // This is where we'll walk the AST and use FCS type information
        
        progress TypeResolution "Type resolution not yet implemented"
        
        return Success {
            ImportContext = imports
            ExpressionTypes = Map.empty
            BindingTypes = Map.empty
            TypeMembers = Map.empty
            InterfaceImplementations = Map.empty
        }
    }

/// Pass 4: Reference Resolution
let private executeReferenceResolution (types: TypeResolutionResult) (progress: ProgressCallback) : Async<CompilerResult<ReferenceResolutionResult>> =
    async {
        progress ReferenceResolution "Starting reference resolution with type information"
        
        // Build the dependency graph using existing code from ReachabilityAnalyzer
        let parsedInputs = 
            types.ImportContext.Discovery.ParsedFiles 
            |> Map.toList
        
        // Use the existing buildDependencyGraph function
        let ctx, dependencies, _ = buildDependencyGraph parsedInputs
        
        progress ReferenceResolution (sprintf "Built dependency graph with %d nodes" (Map.count dependencies))
        
        // TODO: Build resolved references map (for now empty)
        let resolvedReferences = Map.empty
        let unresolved = []
        
        return Success {
            TypeContext = types
            ResolvedReferences = resolvedReferences
            DependencyGraph = dependencies
            UnresolvedReferences = unresolved
        }
    }

/// Pass 5: Reachability Analysis
let private executeReachabilityAnalysis (refs: ReferenceResolutionResult) (progress: ProgressCallback) : Async<CompilerResult<ReachabilityAnalysisResult>> =
    async {
        progress ReachabilityAnalysis "Starting reachability analysis from entry points"
        
        // Use existing computeTransitiveClosure
        let entryPoints = refs.TypeContext.ImportContext.Discovery.EntryPoints
        let reachable = computeTransitiveClosure refs.DependencyGraph entryPoints
        
        // Calculate per-file reachability
        let fileReachableSymbols =
            refs.TypeContext.ImportContext.Discovery.FileSymbols
            |> Map.map (fun filePath symbols ->
                symbols
                |> List.map (fun s -> s.QualifiedName)
                |> Set.ofList
                |> Set.intersect reachable
            )
        
        let totalSymbols = 
            refs.TypeContext.ImportContext.Discovery.FileSymbols 
            |> Map.toSeq 
            |> Seq.sumBy (fun (_, syms) -> List.length syms)
        
        let stats = {
            TotalSymbols = totalSymbols
            ReachableSymbols = Set.count reachable
            EliminatedSymbols = totalSymbols - Set.count reachable
            DependencyEdges = refs.DependencyGraph |> Map.toSeq |> Seq.sumBy (fun (_, deps) -> Set.count deps)
        }
        
        progress ReachabilityAnalysis (sprintf "Reachability complete: %d/%d symbols reachable (%.1f%% eliminated)"
            stats.ReachableSymbols
            stats.TotalSymbols
            (float stats.EliminatedSymbols / float stats.TotalSymbols * 100.0))
        
        return Success {
            ReferenceContext = refs
            ReachableSymbols = reachable
            FileReachableSymbols = fileReachableSymbols
            Statistics = stats
        }
    }

/// Execute the complete 5-pass pipeline
let private executeFivePassPipeline (compilationUnit: CompilationUnit) (state: CompilationState) (progress: ProgressCallback) : Async<CompilerResult<FivePassPipelineResult>> =
    async {
        // Pass 1: Symbol Discovery
        let! symbolResult = executeSymbolDiscovery compilationUnit progress
        match symbolResult with
        | CompilerFailure errors -> return CompilerFailure errors
        | Success symbols ->
            
        // Pass 2: Import Resolution
        let! importResult = executeImportResolution symbols progress
        match importResult with
        | CompilerFailure errors -> return CompilerFailure errors
        | Success imports ->
            
        // Pass 3: Type Resolution
        let! typeResult = executeTypeResolution imports state.Checker progress
        match typeResult with
        | CompilerFailure errors -> return CompilerFailure errors
        | Success types ->
            
        // Pass 4: Reference Resolution
        let! refResult = executeReferenceResolution types progress
        match refResult with
        | CompilerFailure errors -> return CompilerFailure errors
        | Success refs ->
            
        // Pass 5: Reachability Analysis
        let! reachResult = executeReachabilityAnalysis refs progress
        match reachResult with
        | CompilerFailure errors -> return CompilerFailure errors
        | Success reach ->
            
        return Success {
            SymbolDiscovery = symbols
            ImportResolution = imports
            TypeResolution = types
            ReferenceResolution = refs
            ReachabilityAnalysis = reach
        }
    }

// MARK FOR DELETION: The old processCompilationUnits function will be replaced by executeFivePassPipeline
// MARK FOR DELETION: The old buildDependencyGraph will be split across passes 2-4

/// Initialize compilation state with all necessary components
let private initializeCompilationState (projectOptions: FSharpProjectOptions) (config: PipelineConfiguration) (intermediatesDir: string option) =
    let typeCtx = TypeContextBuilder.create()
    let symbolRegistry = 
        let registry = createRegistry alloyPatterns
        registerAlloySymbols registry |> ignore
        registry
    
    {
        Checker = FSharpChecker.Create(keepAssemblyContents = true)
        ProjectOptions = projectOptions
        TypeContext = typeCtx
        SymbolRegistry = symbolRegistry
        Configuration = config
        IntermediatesDirectory = intermediatesDir
    }

/// Process a single file through parsing, symbol extraction, and type checking
let private processSingleFile 
    (checker: FSharpChecker) 
    (order: int) 
    (filePath: string) 
    (ingestionState: FCSIngestionState) 
    (config: PipelineConfiguration)
    (progress: ProgressCallback) =
    
    async {
        let fileName = Path.GetFileName(filePath)
        progress Parsing (sprintf "Parsing %s (file %d)" fileName order)
        
        // Read and parse the file
        let! sourceText = File.ReadAllTextAsync(filePath) |> Async.AwaitTask
        let sourceTextObj = FSharp.Compiler.Text.SourceText.ofString sourceText
        let parsingOptions, _ = checker.GetParsingOptionsFromProjectOptions(ingestionState.CumulativeProjectOptions)
        let! parseResult = checker.ParseFile(filePath, sourceTextObj, parsingOptions)
        
        // Write parsing results if keeping intermediates
        match ingestionState.IntermediateStructure with
        | Some structure ->
            writeParsingResults structure order filePath parseResult.ParseTree parseResult.Diagnostics
        | None -> ()
        
        // Check for parse errors
        if parseResult.ParseHadErrors then
            let errors = 
                parseResult.Diagnostics 
                |> Array.map (fun d -> 
                    SyntaxError({ Line = d.StartLine; Column = d.StartColumn; File = filePath; Offset = 0 }, d.Message, ["parsing"]))
                |> Array.toList
            return Error errors
        else
            // Extract symbols from the parsed AST
            progress SymbolDiscovery (sprintf "Extracting symbols from %s" fileName)
            let symbols = Core.FCSIngestion.SymbolExtraction.extractSymbolsFromParsedInput parseResult.ParseTree
            
            // NOTE: Symbol Registry update removed - this now happens in Pass 3 (Type Resolution)
            // The SymbolRegistry requires ResolvedSymbol which needs type information from FCS
            
            // Write symbol extraction results if keeping intermediates
            match ingestionState.IntermediateStructure with
            | Some structure ->
                let symbolData = {|
                    File = filePath
                    Order = order
                    SymbolCount = symbols.Length
                    Symbols = symbols |> List.map (fun s -> {|
                        QualifiedName = s.QualifiedName
                        ShortName = s.ShortName
                        ModulePath = s.ModulePath
                        Kind = s.Kind.ToString()
                    |})
                |}
                writeSymbolExtractionResults structure filePath symbolData (List.length ingestionState.ParsedFiles + symbols.Length)
            | None -> ()
            
            // Type check with accumulated context
            progress TypeChecking (sprintf "Type checking %s" fileName)
            let fileVersion = 0
            let! checkAnswer = checker.CheckFileInProject(parseResult, filePath, fileVersion, sourceTextObj, ingestionState.CumulativeProjectOptions)
            
            match checkAnswer with
            | FSharpCheckFileAnswer.Aborted ->
                return Error [InternalError("processSingleFile", sprintf "Type checking aborted for file: %s" filePath, None)]
            | FSharpCheckFileAnswer.Succeeded checkResult ->
                // Write type checking results if keeping intermediates
                match ingestionState.IntermediateStructure with
                | Some structure ->
                    let checkData = {|
                        File = filePath
                        Order = order
                        HasTypeErrors = checkResult.Diagnostics.Length > 0
                        DiagnosticsCount = checkResult.Diagnostics.Length
                        HasFullTypeCheckInfo = checkResult.HasFullTypeCheckInfo
                        Diagnostics = 
                            checkResult.Diagnostics 
                            |> Array.map (fun d -> {|
                                Message = d.Message
                                Severity = d.Severity.ToString()
                                Range = sprintf "(%d,%d)-(%d,%d)" 
                                    d.StartLine d.StartColumn 
                                    d.EndLine d.EndColumn
                            |})
                            |> Array.toList
                    |}
                    let baseName = sprintf "%02d_%s_typecheck" order (Path.GetFileNameWithoutExtension(filePath))
                    writeFile structure.CompilationUnitsDir baseName ".json" (System.Text.Json.JsonSerializer.Serialize(checkData))
                | None -> ()
                
                // Return updated state with extracted symbols
                let updatedState = {
                    ingestionState with
                        ParsedFiles = ingestionState.ParsedFiles @ [(order, filePath, parseResult.ParseTree)]
                        CheckedFiles = ingestionState.CheckedFiles @ [(filePath, parseResult.ParseTree, checkResult)]
                        FileSymbols = Map.add filePath symbols ingestionState.FileSymbols  // Store ExtractedSymbol list
                        // SymbolRegistry update removed - happens in Type Resolution pass
                }
                
                return Ok updatedState
    }

/// Phase 1: Complete FCSIngestion - parse, extract symbols, and type check all files in order
let private performFCSIngestion 
    (state: CompilationState) 
    (projectOptions: FSharpProjectOptions) 
    (intermediateStructure: IntermediateFileStructure option)
    (progress: ProgressCallback) =
    
    async {
        progress Initialization "Starting FCS ingestion phase"
        
        // Initialize ingestion state
        let initialIngestionState = {
            ParsedFiles = []
            CheckedFiles = []
            SymbolRegistry = state.SymbolRegistry
            TypeContext = state.TypeContext
            CumulativeProjectOptions = projectOptions
            IntermediateStructure = intermediateStructure
            FileSymbols = Map.empty
        }
        
        // Process each file sequentially in compilation order
        let! finalStateResult = 
            projectOptions.SourceFiles
            |> Array.mapi (fun i f -> (i, f))
            |> Array.fold (fun accAsync (order, filePath) ->
                async {
                    let! accResult = accAsync
                    match accResult with
                    | Error errors -> return Error errors
                    | Ok accState ->
                        let! processResult = processSingleFile state.Checker order filePath accState state.Configuration progress
                        match processResult with
                        | Error errors -> return Error errors
                        | Ok newState -> return Ok newState
                }) (async { return Ok initialIngestionState })
        
        match finalStateResult with
        | Error errors -> return CompilerFailure errors
        | Ok finalState ->
            // Write ingestion summary if keeping intermediates
            match intermediateStructure with
            | Some structure ->
                let summary = {|
                    Phase = "FCSIngestion"
                    TotalFiles = finalState.ParsedFiles.Length
                    TotalSymbols = 
                        finalState.ParsedFiles 
                        |> List.sumBy (fun (_, _, ast) -> 
                            extractSymbolsFromParsedInput ast |> List.length)
                    ParsedSuccessfully = finalState.ParsedFiles.Length
                    TypeCheckedSuccessfully = finalState.CheckedFiles.Length
                    CompilationOrder = 
                        finalState.ParsedFiles 
                        |> List.map (fun (order, file, _) -> {| Order = order; File = Path.GetFileName(file) |})
                |}
                writeCompilationSummary structure.SummaryDir summary
            | None -> ()
            
            return Success finalState
    }

/// Build compilation unit from FCS ingestion results using unified extraction
let private buildCompilationUnitFromIngestion (ingestionState: FCSIngestionState) (mainFile: string) (progress: ProgressCallback) =
    async {
        progress ReachabilityAnalysis "Building compilation unit from ingested files"
        
        // Use unified symbol extraction for all files
        let filesWithSymbols = 
            ingestionState.CheckedFiles
            |> List.map (fun (filePath, ast, _) ->
                let symbols = Core.FCSIngestion.SymbolExtraction.extractSymbolsFromParsedInput ast
                (filePath, ast, symbols))
        
        // Build source files with extracted symbols
        let sourceFiles = 
            filesWithSymbols
            |> List.map (fun (filePath, ast, symbols) ->
                let symbolSet = symbols |> List.map (fun s -> s.QualifiedName) |> Set.ofList
                
                let sourceFile = {
                    Path = filePath
                    Ast = ast
                    LoadedFiles = [] // Already resolved during project loading
                    ModulePath = Core.FCSIngestion.SymbolExtraction.extractModulePath ast
                    DefinedSymbols = symbolSet
                }
                (filePath, sourceFile))
            |> Map.ofList
        
        // Build the global symbol-to-file mapping using the unified function
        let filesAndSymbols = 
            filesWithSymbols 
            |> List.map (fun (path, _, symbols) -> (path, symbols))
        
        let symbolToFile = buildSymbolToFileMap filesAndSymbols
        
        // Build file dependencies
        let fileDependencies = 
            sourceFiles
            |> Map.map (fun _ sourceFile -> 
                sourceFile.LoadedFiles |> Set.ofList)
        
        // Log debugging information
        progress ReachabilityAnalysis (sprintf "Built compilation unit with %d files and %d total symbols" 
            (Map.count sourceFiles) (Map.count symbolToFile))
        
        // Create the compilation unit
        let compilationUnit = {
            MainFile = mainFile
            SourceFiles = sourceFiles
            SymbolToFile = symbolToFile
            FileDependencies = fileDependencies
        }
        
        return compilationUnit
    }

/// Detect EntryPoint attributes
let private detectEntryPointAttributes (ast: ParsedInput) : Set<string> =
    match ast with
    | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
        modules 
        |> List.collect (fun (SynModuleOrNamespace(moduleIds, _, _, decls, _, _, _, _, _)) ->
            let modulePath = moduleIds |> List.map (fun id -> id.idText) |> String.concat "."
            decls |> List.choose (fun decl ->
                match decl with
                | SynModuleDecl.Let(_, bindings, _) ->
                    bindings 
                    |> List.tryPick (fun binding ->
                        match binding with
                        | SynBinding(_, _, _, _, attrs, _, _, pat, _, _, _, _, _) ->
                            let hasEntryPoint = 
                                attrs 
                                |> List.exists (fun attrList ->
                                    attrList.Attributes |> List.exists (fun attr ->
                                        match attr.TypeName with
                                        | SynLongIdent([id], _, _) when id.idText = "EntryPoint" -> true
                                        | _ -> false))
                            
                            if hasEntryPoint then
                                match pat with
                                | SynPat.Named(SynIdent(id, _), _, _, _) ->
                                    if modulePath = "" then Some id.idText
                                    else Some (sprintf "%s.%s" modulePath id.idText)
                                | SynPat.LongIdent(SynLongIdent(ids, _, _), _, _, _, _, _) ->
                                    let name = ids |> List.map (fun id -> id.idText) |> String.concat "."
                                    if modulePath = "" then Some name
                                    else Some (sprintf "%s.%s" modulePath name)
                                | _ -> None
                            else None)
                | _ -> None))
        |> Set.ofList
    | _ -> Set.empty


/// Process compilation units through analysis and transformation
let private processCompilationUnits (compilationUnit: CompilationUnit) (ingestionState: FCSIngestionState) (state: CompilationState) (progress: ProgressCallback) =
    async {
        // Perform reachability analysis
        progress ReachabilityAnalysis "Performing global reachability analysis"
        let analysis = analyzeCompilationUnit compilationUnit
        
        // Report statistics
        progress ReachabilityAnalysis (sprintf "Analysis complete: %d/%d symbols reachable (%.1f%% eliminated)" 
            analysis.GlobalReachability.Statistics.ReachableSymbols
            analysis.GlobalReachability.Statistics.TotalSymbols
            ((float analysis.GlobalReachability.Statistics.EliminatedSymbols / float analysis.GlobalReachability.Statistics.TotalSymbols) * 100.0))
        
        // Write intermediate files if configured
        if state.Configuration.PreserveIntermediateASTs && state.IntermediatesDirectory.IsSome then
            // Get original file order from ingestion state
            let fileOrder = 
                ingestionState.ParsedFiles 
                |> List.map (fun (order, filePath, _) -> (filePath, order))
                |> Map.ofList
            
            // Write pruned ASTs in original parsing order
            analysis.PrunedAsts
            |> Map.toList
            |> List.sortBy (fun (filePath, _) -> 
                Map.tryFind filePath fileOrder |> Option.defaultValue 999)
            |> List.iter (fun (filePath, prunedAst) ->
                let order = Map.tryFind filePath fileOrder |> Option.defaultValue 999
                let fileName = Path.GetFileName(filePath)
                let baseName = sprintf "%02d_%s" order fileName
                let prunedDir = Path.Combine(state.IntermediatesDirectory.Value, "fcs", "pruned", "compilation_units")
                Directory.CreateDirectory(prunedDir) |> ignore
                
                // Log which file we're writing with its reachable symbol count
                let reachableCount = 
                    Map.tryFind filePath analysis.PerFileReachable 
                    |> Option.map Set.count 
                    |> Option.defaultValue 0
                progress ReachabilityAnalysis (sprintf "Writing pruned AST for %s (%d reachable symbols)" fileName reachableCount)
                
                writeFile prunedDir baseName ".pruned.ast" (sprintf "%A" prunedAst))
        
        // Build processing result
        let processedUnits = 
            analysis.PrunedAsts
            |> Map.map (fun filePath prunedAst ->
                let originalAst = 
                    ingestionState.CheckedFiles 
                    |> List.find (fun (p, _, _) -> p = filePath) 
                    |> (fun (_, ast, _) -> ast)
                let symbols = compilationUnit.SourceFiles.[filePath].DefinedSymbols
                let reachableSymbols = Map.tryFind filePath analysis.PerFileReachable |> Option.defaultValue Set.empty
                
                {
                    FilePath = filePath
                    OriginalAST = originalAst
                    TransformedAST = prunedAst
                    Symbols = symbols
                    ReachableSymbols = reachableSymbols
                })
        
        return Success {
            ProcessedUnits = processedUnits
            CompilationAnalysis = analysis
            TypeContext = ingestionState.TypeContext
            SymbolRegistry = ingestionState.SymbolRegistry
            GlobalReachability = analysis.GlobalReachability
        }
    }

/// Generate MLIR from processed results
let private generateMLIR (result: MultiFileProcessingResult) (state: CompilationState) (progress: ProgressCallback) =
    progress MLIRGeneration "Generating MLIR from transformed ASTs"
    
    // Generate MLIR from the main file's transformed AST
    let mainFile = result.CompilationAnalysis.Unit.MainFile
    match Map.tryFind mainFile result.ProcessedUnits with
    | None -> CompilerFailure [InternalError("generateMLIR", "Main file not found in processed units", None)]
    | Some processedUnit ->
        try
            let mlirText = generateModuleFromAST processedUnit.TransformedAST result.TypeContext result.SymbolRegistry
            
            // Write MLIR if configured
            if state.IntermediatesDirectory.IsSome then
                let mlirPath = Path.Combine(state.IntermediatesDirectory.Value, Path.GetFileNameWithoutExtension(mainFile) + ".mlir")
                File.WriteAllText(mlirPath, mlirText)
                progress MLIRGeneration (sprintf "Wrote MLIR to %s" mlirPath)
            
            Success mlirText
        with ex ->
            CompilerFailure [MLIRGenerationError("generateMLIR", ex.Message, Some mainFile)]

/// Main compilation pipeline orchestration
let compileProject (inputPath: string) 
                   (outputPath: string) 
                   (projectOptions: FSharpProjectOptions) 
                   (config: PipelineConfiguration) 
                   (intermediatesDir: string option)
                   (progress: ProgressCallback) =
    async {
        let startTime = DateTime.UtcNow
        progress Initialization "Initializing compilation pipeline"
        
        // Initialize compilation state
        let state = initializeCompilationState projectOptions config intermediatesDir
        
        // Create intermediate file structure if keeping intermediates
        let intermediateStructure = 
            match intermediatesDir with
            | Some dir when config.PreserveIntermediateASTs -> 
                Some (createIntermediateStructure dir)
            | _ -> None
        
        // Phase 1: Complete FCS Ingestion
        let! ingestionResult = performFCSIngestion state projectOptions intermediateStructure progress
        
        match ingestionResult with
        | CompilerFailure errors ->
            return {
                Success = false
                MLIROutput = None
                LLVMOutput = None
                Diagnostics = errors
                Statistics = {
                    TotalFiles = projectOptions.SourceFiles.Length
                    TotalSymbols = 0
                    ReachableSymbols = 0
                    EliminatedSymbols = 0
                    CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                }
            }
        | Success ingestionState ->
            // Build compilation unit from ingestion results
            let! compilationUnit = buildCompilationUnitFromIngestion ingestionState inputPath progress
            
            // Phase 2: Execute the new 5-pass pipeline
            let! pipelineResult = executeFivePassPipeline compilationUnit state progress
            
            match pipelineResult with
            | CompilerFailure errors ->
                return {
                    Success = false
                    MLIROutput = None
                    LLVMOutput = None
                    Diagnostics = errors
                    Statistics = {
                        TotalFiles = projectOptions.SourceFiles.Length
                        TotalSymbols = 0
                        ReachableSymbols = 0
                        EliminatedSymbols = 0
                        CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                    }
                }
            | Success pipeline ->
                // Phase 3: Transform ASTs based on reachability
                progress ASTTransformation "Transforming ASTs based on reachability analysis"
                
                // Apply transformations to each file
                let transformedFiles = 
                    pipeline.ReachabilityAnalysis.ReferenceContext.TypeContext.ImportContext.Discovery.ParsedFiles
                    |> Map.map (fun filePath ast ->
                        let fileReachable = 
                            Map.tryFind filePath pipeline.ReachabilityAnalysis.FileReachableSymbols 
                            |> Option.defaultValue Set.empty
                        
                        // Apply pruning
                        let prunedAst = prune fileReachable ast
                        
                        // Apply transformations if enabled
                        let transformedAst = 
                            if config.EnableClosureElimination || config.EnableStackAllocation then
                                applyTransformations prunedAst state.TypeContext fileReachable
                            else
                                prunedAst
                        
                        transformedAst
                    )
                
                // Write intermediate files if requested
                if config.PreserveIntermediateASTs && intermediateStructure.IsSome then
                    transformedFiles |> Map.iter (fun filePath ast ->
                        let fileName = Path.GetFileNameWithoutExtension(filePath)
                        let astText = sprintf "%A" ast
                        writeFile filePath fileName ".ra.fcs" astText
                    )
                
                // Phase 4: Generate MLIR from the main file
                progress MLIRGeneration "Generating MLIR from transformed ASTs"
                
                let mainFile = compilationUnit.MainFile
                match Map.tryFind mainFile transformedFiles with
                | None -> 
                    return {
                        Success = false
                        MLIROutput = None
                        LLVMOutput = None
                        Diagnostics = [InternalError("generateMLIR", "Main file not found in transformed files", Some mainFile)]
                        Statistics = {
                            TotalFiles = projectOptions.SourceFiles.Length
                            TotalSymbols = pipeline.ReachabilityAnalysis.Statistics.TotalSymbols
                            ReachableSymbols = pipeline.ReachabilityAnalysis.Statistics.ReachableSymbols
                            EliminatedSymbols = pipeline.ReachabilityAnalysis.Statistics.EliminatedSymbols
                            CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                        }
                    }
                | Some transformedMainAst ->
                    try
                        let mlirText = generateModuleFromAST transformedMainAst state.TypeContext state.SymbolRegistry
                        
                        // Write MLIR if configured
                        if intermediatesDir.IsSome then
                            let mlirPath = Path.Combine(intermediatesDir.Value, Path.GetFileNameWithoutExtension(mainFile) + ".mlir")
                            File.WriteAllText(mlirPath, mlirText)
                            progress MLIRGeneration (sprintf "Wrote MLIR to %s" mlirPath)
                        
                        // Write final compilation report if keeping intermediates
                        match intermediateStructure with
                        | Some structure ->
                            let report = {|
                                Success = true
                                TotalFiles = projectOptions.SourceFiles.Length
                                TotalSymbols = pipeline.ReachabilityAnalysis.Statistics.TotalSymbols
                                ReachableSymbols = pipeline.ReachabilityAnalysis.Statistics.ReachableSymbols
                                EliminatedSymbols = pipeline.ReachabilityAnalysis.Statistics.EliminatedSymbols
                                EliminationPercentage = 
                                    if pipeline.ReachabilityAnalysis.Statistics.TotalSymbols > 0 then
                                        (float pipeline.ReachabilityAnalysis.Statistics.EliminatedSymbols / 
                                         float pipeline.ReachabilityAnalysis.Statistics.TotalSymbols) * 100.0
                                    else 0.0
                                CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                                MLIRGenerated = true
                                NextStep = "Use mlir-opt and mlir-translate for lowering to LLVM IR"
                            |}
                            writeCompilationReport structure report
                        | None -> ()
                        
                        progress Finalization "MLIR generation complete"
                        return {
                            Success = true
                            MLIROutput = Some mlirText
                            LLVMOutput = None
                            Diagnostics = []
                            Statistics = {
                                TotalFiles = projectOptions.SourceFiles.Length
                                TotalSymbols = pipeline.ReachabilityAnalysis.Statistics.TotalSymbols
                                ReachableSymbols = pipeline.ReachabilityAnalysis.Statistics.ReachableSymbols
                                EliminatedSymbols = pipeline.ReachabilityAnalysis.Statistics.EliminatedSymbols
                                CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                            }
                        }
                    with ex ->
                        return {
                            Success = false
                            MLIROutput = None
                            LLVMOutput = None
                            Diagnostics = [MLIRGenerationError("generateMLIR", ex.Message, Some mainFile)]
                            Statistics = {
                                TotalFiles = projectOptions.SourceFiles.Length
                                TotalSymbols = pipeline.ReachabilityAnalysis.Statistics.TotalSymbols
                                ReachableSymbols = pipeline.ReachabilityAnalysis.Statistics.ReachableSymbols
                                EliminatedSymbols = pipeline.ReachabilityAnalysis.Statistics.EliminatedSymbols
                                CompilationTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
                            }
                        }
    }

/// Create a progress reporter based on verbosity settings
let createProgressReporter (verbose: bool) : ProgressCallback =
    if verbose then
        fun phase message -> printfn "[%A] %s" phase message
    else
        fun phase message -> 
            match phase with
            | Initialization | Finalization -> printfn "%s" message
            | _ -> ()