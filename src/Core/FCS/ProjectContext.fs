module Core.FCS.ProjectContext

open System
open System.IO
open System.Xml.Linq
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

/// Project file data extracted from .fsproj
type ProjectFileData = {
    OutputType: string option
    TargetFramework: string option
    NoStdLib: bool
    DisableImplicitFSharpCoreReference: bool
    DisableImplicitSystemValueTupleReference: bool
    TreatWarningsAsErrors: bool
    OtherFlags: string list
    NoWarns: string list
    CompileItems: string list
    PackageReferences: (string * string) list
    ProjectReferences: string list
}

/// Parse the .fsproj file properly
let parseProjectFile (projectPath: string) =
    try
        let doc = XDocument.Load(projectPath)
        let ns = doc.Root.Name.Namespace
        let projectDir = Path.GetDirectoryName(projectPath)
        
        // Helper to get element value
        let getElementValue (name: string) =
            doc.Descendants(ns + name)
            |> Seq.tryHead
            |> Option.map (fun elem -> elem.Value)
        
        // Helper to get boolean property
        let getBoolProperty (name: string) =
            getElementValue name
            |> Option.map (fun v -> v.Equals("true", StringComparison.OrdinalIgnoreCase))
            |> Option.defaultValue false
        
        // Extract compile items
        let compileItems = 
            doc.Descendants(ns + "Compile")
            |> Seq.choose (fun elem -> 
                elem.Attribute(XName.Get("Include")) 
                |> Option.ofObj 
                |> Option.map (fun attr -> Path.GetFullPath(Path.Combine(projectDir, attr.Value))))
            |> Seq.toList
        
        // Extract package references
        let packageReferences =
            doc.Descendants(ns + "PackageReference")
            |> Seq.choose (fun elem ->
                let includeAttr = elem.Attribute(XName.Get("Include")) |> Option.ofObj
                let versionAttr = elem.Attribute(XName.Get("Version")) |> Option.ofObj
                match includeAttr, versionAttr with
                | Some inc, Some ver -> Some (inc.Value, ver.Value)
                | _ -> None)
            |> Seq.toList
        
        // Extract project references
        let projectReferences =
            doc.Descendants(ns + "ProjectReference")
            |> Seq.choose (fun elem ->
                elem.Attribute(XName.Get("Include"))
                |> Option.ofObj
                |> Option.map (fun attr -> Path.GetFullPath(Path.Combine(projectDir, attr.Value))))
            |> Seq.toList
        
        // Parse OtherFlags
        let otherFlags =
            getElementValue "OtherFlags"
            |> Option.map (fun flags -> flags.Split([|' '|], StringSplitOptions.RemoveEmptyEntries) |> List.ofArray)
            |> Option.defaultValue []
        
        // Parse NoWarn
        let noWarns =
            getElementValue "NoWarn"
            |> Option.map (fun warns -> warns.Split([|';'|], StringSplitOptions.RemoveEmptyEntries) |> List.ofArray)
            |> Option.defaultValue []
        
        Some {
            OutputType = getElementValue "OutputType"
            TargetFramework = getElementValue "TargetFramework"
            NoStdLib = getBoolProperty "NoStdLib"
            DisableImplicitFSharpCoreReference = getBoolProperty "DisableImplicitFSharpCoreReference"
            DisableImplicitSystemValueTupleReference = getBoolProperty "DisableImplicitSystemValueTupleReference"
            TreatWarningsAsErrors = getBoolProperty "TreatWarningsAsErrors"
            OtherFlags = otherFlags
            NoWarns = noWarns
            CompileItems = compileItems
            PackageReferences = packageReferences
            ProjectReferences = projectReferences
        }
    with ex ->
        printfn "[ProjectContext] Error parsing project file: %s" ex.Message
        None

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

/// Build project options from parsed project data
let buildProjectOptions (projectPath: string) (projData: ProjectFileData) =
    let projectDir = Path.GetDirectoryName(projectPath)
    
    // Build compiler options from project file data
    let otherOptions = [
        // Output type
        match projData.OutputType with
        | Some "Exe" -> yield "--target:exe"
        | Some "Library" -> yield "--target:library"
        | _ -> yield "--target:exe"
        
        // Framework defines
        match projData.TargetFramework with
        | Some fw when fw.StartsWith("net") && not (fw.Contains("framework")) ->
            yield "--targetprofile:netcore"
            yield "--define:NETCOREAPP"
        | _ -> ()
        
        // Firefly-specific defines
        yield "--define:FIREFLY"
        yield "--define:ZERO_ALLOCATION"
        
        // Warning handling
        if projData.TreatWarningsAsErrors then
            yield "--warnaserror+"
        
        for warn in projData.NoWarns do
            yield sprintf "--nowarn:%s" warn
        
        // Other flags from project file
        yield! projData.OtherFlags
        
        // For FCS to work, we need some basic references even if NoStdLib is true
        // We'll add minimal references to satisfy FCS type checking
        // The actual Firefly compiler will ignore these during code generation
        if projData.NoStdLib then
            // Add a minimal mscorlib reference just for FCS type checking
            // This allows FCS to resolve System.Array, System.Object, etc.
            let dotnetRoot = 
                match Environment.GetEnvironmentVariable("DOTNET_ROOT") with
                | null | "" -> 
                    if Environment.OSVersion.Platform = PlatformID.Win32NT then
                        @"C:\Program Files\dotnet"
                    else
                        "/usr/share/dotnet"
                | root -> root
            
            let runtimeDir = Path.Combine(dotnetRoot, "shared", "Microsoft.NETCore.App")
            if Directory.Exists(runtimeDir) then
                let versions = Directory.GetDirectories(runtimeDir)
                if versions.Length > 0 then
                    let latestVersion = versions |> Array.sort |> Array.last
                    let mscorlibPath = Path.Combine(latestVersion, "mscorlib.dll")
                    if File.Exists(mscorlibPath) then
                        yield sprintf "-r:%s" mscorlibPath
        
        // Add explicit FSharp.Core reference if found
        for (pkg, version) in projData.PackageReferences do
            if pkg = "FSharp.Core" then
                // Try to find FSharp.Core in NuGet packages
                let userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)
                let nugetPath = Path.Combine(userProfile, ".nuget", "packages", "fsharp.core", version, "lib")
                if Directory.Exists(nugetPath) then
                    let possibleTargets = 
                        match projData.TargetFramework with
                        | Some "net8.0" -> ["net8.0"; "net7.0"; "net6.0"; "netstandard2.1"; "netstandard2.0"]
                        | Some "net9.0" -> ["net9.0"; "net8.0"; "net7.0"; "net6.0"; "netstandard2.1"; "netstandard2.0"]
                        | Some fw -> [fw; "netstandard2.1"; "netstandard2.0"]
                        | None -> ["netstandard2.1"; "netstandard2.0"]
                    
                    let fsharpCorePath = 
                        possibleTargets
                        |> List.tryPick (fun target ->
                            let path = Path.Combine(nugetPath, target, "FSharp.Core.dll")
                            if File.Exists(path) then Some path else None)
                    
                    match fsharpCorePath with
                    | Some path -> yield sprintf "-r:%s" path
                    | None -> ()
    ]
    
    {
        ProjectFileName = projectPath
        ProjectId = None
        SourceFiles = projData.CompileItems |> List.toArray
        OtherOptions = otherOptions |> List.toArray
        ReferencedProjects = [||]  // Firefly doesn't support project references - all code must be in source form
        IsIncompleteTypeCheckEnvironment = false
        UseScriptResolutionRules = false
        LoadTime = DateTime.Now
        UnresolvedReferences = None
        OriginalLoadReferences = []
        Stamp = None
    }

/// Load project with optimal settings for compilation
let loadProject (projectPath: string) (strategy: CacheStrategy) = async {
    match parseProjectFile projectPath with
    | None ->
        return failwith $"Failed to parse project file: {projectPath}"
    | Some projData ->
        printfn "[ProjectContext] Parsed project file:"
        printfn "  OutputType: %A" projData.OutputType
        printfn "  TargetFramework: %A" projData.TargetFramework
        printfn "  NoStdLib: %b" projData.NoStdLib
        printfn "  Source files: %d" projData.CompileItems.Length
        
        let checker = createChecker strategy
        let projectOptions = buildProjectOptions projectPath projData
        
        printfn "[ProjectContext] Project options created:"
        printfn "  Source files: %d" projectOptions.SourceFiles.Length
        printfn "  Compiler options: %d" projectOptions.OtherOptions.Length
        for opt in projectOptions.OtherOptions do
            printfn "    %s" opt
        
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
    
    // Report any errors
    if checkResults.HasCriticalErrors then
        printfn "[ProjectContext] Critical errors detected:"
        for error in checkResults.Diagnostics do
            if error.Severity = FSharp.Compiler.Diagnostics.FSharpDiagnosticSeverity.Error then
                printfn "  ERROR: %s" error.Message
    
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