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

/// Load project with optimal settings for compilation
let loadProject (projectPath: string) (strategy: CacheStrategy) = async {
    let checker = createChecker strategy
    
    // Use FCS project cracking
    let! (options, diagnostics) = 
        checker.GetProjectOptionsFromProjectFile(projectPath)
    
    // Pre-parse all files to warm cache if aggressive strategy
    if strategy = Aggressive then
        for sourceFile in options.SourceFiles do
            let! _ = checker.ParseFile(
                sourceFile,
                SourceText.ofString(File.ReadAllText(sourceFile)),
                options
            )
            ()
    
    return {
        Checker = checker
        ProjectOptions = options
        CacheStrategy = strategy
    }
}

/// Get complete project results with both ASTs
type ProjectResults = {
    CheckResults: FSharpCheckProjectResults
    ParseResults: FSharpParseFileResults[]
    SymbolUses: FSharpSymbolUse[]
    CompilationOrder: string[]
}

/// Parse and check entire project in one operation
let getProjectResults (ctx: ProjectContext) = async {
    let! checkResults = ctx.Checker.ParseAndCheckProject(ctx.ProjectOptions)
    
    // Get parse results for each file
    let! parseResults = 
        ctx.ProjectOptions.SourceFiles
        |> Array.map (fun file -> async {
            let source = SourceText.ofString(File.ReadAllText(file))
            return! ctx.Checker.ParseFile(file, source, ctx.ProjectOptions)
        })
        |> Async.Parallel
    
    return {
        CheckResults = checkResults
        ParseResults = parseResults
        SymbolUses = checkResults.GetAllUsesOfAllSymbols()
        CompilationOrder = ctx.ProjectOptions.SourceFiles
    }
}

/// Get symbol at specific location using FCS built-in correlation
let getSymbolAt (line: int) (col: int) (fileName: string) (results: ProjectResults) =
    results.CheckResults.GetSymbolUseAtLocation(
        line, col, fileName, []
    )

/// Get implementation file by name
let getImplementationFile (fileName: string) (results: ProjectResults) =
    results.CheckResults.AssemblyContents.ImplementationFiles
    |> List.tryFind (fun implFile -> 
        implFile.FileName.EndsWith(fileName)
    )