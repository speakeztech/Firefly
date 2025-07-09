module Core.FCS.ProjectContext

open System
open System.IO
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Text

/// Wraps FSharpChecker with intelligent caching and project-wide operations
type ProjectContext = {
    Checker: FSharpChecker
    ProjectOptions: FSharpProjectOptions
    CacheStrategy: CacheStrategy
}

and CacheStrategy =
    | Aggressive     // Maximum caching, higher memory usage
    | Balanced       // Default caching strategy
    | Conservative   // Minimal caching for memory-constrained environments

/// Create a properly configured FSharpChecker for the platform
let createChecker (strategy: CacheStrategy) =
    let cacheSize = 
        match strategy with
        | Aggressive -> 200
        | Balanced -> 50
        | Conservative -> 10
    
    FSharpChecker.Create(
        projectCacheSize = cacheSize,
        keepAssemblyContents = true,  // Always keep for PSG construction
        keepAllBackgroundResolutions = (strategy = Aggressive),
        enableBackgroundItemKeyStoreAndSemanticClassification = true
    )

/// Build project options from source files (FCS 43.9.300 doesn't have GetProjectOptionsFromProjectFile)
let buildProjectOptions (projectPath: string) (sourceFiles: string[]) =
    let projectDir = Path.GetDirectoryName(projectPath)
    
    // Build compiler options for Firefly
    let otherOptions = [|
        "--noframework"
        "--nostdlib"
        "--standalone"
        "--define:NO_MSCORLIB"
        "--define:NO_SYSTEM_REFERENCE"
        "--define:FIREFLY"
        "--define:ZERO_ALLOCATION"
        "--targetprofile:netcore"
        "--simpleresolution"
        "--nowin32manifest"
    |]
    
    {
        ProjectFileName = projectPath
        ProjectId = None
        SourceFiles = sourceFiles
        OtherOptions = otherOptions
        ReferencedProjects = [||]
        IsIncompleteTypeCheckEnvironment = false
        UseScriptResolutionRules = false
        LoadTime = DateTime.Now
        UnresolvedReferences = None
        OriginalLoadReferences = []
        Stamp = None
    }

/// Load project with optimal settings for compilation
let loadProject (projectPath: string) (strategy: CacheStrategy) = async {
    let checker = createChecker strategy
    
    // For now, use a simple file enumeration (in real implementation, parse the project file)
    let projectDir = Path.GetDirectoryName(projectPath)
    let sourceFiles = 
        Directory.GetFiles(projectDir, "*.fs", SearchOption.AllDirectories)
        |> Array.filter (fun f -> not (f.Contains("obj") || f.Contains("bin")))
        |> Array.sort
    
    let projectOptions = buildProjectOptions projectPath sourceFiles
    
    // Pre-parse all files to warm cache if aggressive strategy
    if strategy = Aggressive then
        let parsingOptions, _ = checker.GetParsingOptionsFromProjectOptions(projectOptions)
        
        for sourceFile in sourceFiles do
            let source = SourceText.ofString(File.ReadAllText(sourceFile))
            let! _ = checker.ParseFile(sourceFile, source, parsingOptions)
            ()
    
    return {
        Checker = checker
        ProjectOptions = projectOptions
        CacheStrategy = strategy
    }
}

/// Get complete project results with both ASTs
type ProjectResults = {
    CheckResults: FSharpCheckProjectResults
    ParseResults: FSharpParseFileResults[]
    SymbolUses: FSharpSymbolUse[]
    CompilationOrder: string[]
    Context: ProjectContext  // Add context to carry checker and options
}

/// Parse and check entire project in one operation
let getProjectResults (ctx: ProjectContext) = async {
    let! checkResults = ctx.Checker.ParseAndCheckProject(ctx.ProjectOptions)
    
    // Get parsing options from project options
    let parsingOptions, _ = ctx.Checker.GetParsingOptionsFromProjectOptions(ctx.ProjectOptions)
    
    // Get parse results for each file
    let! parseResults = 
        ctx.ProjectOptions.SourceFiles
        |> Array.map (fun file -> async {
            let source = SourceText.ofString(File.ReadAllText(file))
            return! ctx.Checker.ParseFile(file, source, parsingOptions)
        })
        |> Async.Parallel
    
    return {
        CheckResults = checkResults
        ParseResults = parseResults
        SymbolUses = checkResults.GetAllUsesOfAllSymbols()
        CompilationOrder = ctx.ProjectOptions.SourceFiles
        Context = ctx  // Store the context
    }
}

/// Get symbol uses at specific location 
let getSymbolAt (line: int) (col: int) (fileName: string) (results: ProjectResults) =
    // Find the parse results for this file
    let parseResultOpt = 
        results.ParseResults 
        |> Array.tryFind (fun pr -> pr.FileName = fileName)
    
    match parseResultOpt with
    | Some parseResult ->
        // Use CheckFileResults to get symbol at location
        async {
            let source = SourceText.ofString(File.ReadAllText(fileName))
            let! checkAnswer = 
                results.Context.Checker.CheckFileInProject(
                    parseResult,
                    fileName,
                    0,
                    source,
                    results.Context.ProjectOptions
                )
            
            match checkAnswer with
            | FSharpCheckFileAnswer.Succeeded(checkFileResults) ->
                return checkFileResults.GetSymbolUseAtLocation(line, col, "", [])
            | _ -> 
                return None
        } |> Async.RunSynchronously
    | None -> None

/// Get implementation file by name
let getImplementationFile (fileName: string) (results: ProjectResults) =
    results.CheckResults.AssemblyContents.ImplementationFiles
    |> List.tryFind (fun implFile -> 
        implFile.FileName.EndsWith(fileName)
    )