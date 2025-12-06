module CLI.Configurations.FidprojLoader

open System
open System.IO
open FSharp.Compiler.CodeAnalysis
open Core.XParsec.Foundation
open CLI.Configurations.ProjectConfig

/// Resolved dependency with its source files
type ResolvedDependency = {
    Name: string
    SourcePath: string
    SourceFiles: string list
}

/// Fully resolved .fidproj configuration
type ResolvedFidproj = {
    Name: string
    Version: string
    ProjectDir: string
    Sources: string list
    Dependencies: ResolvedDependency list
    AllSourcesInOrder: string list  // Dependencies first, then project sources
    Target: string option
    MemoryModel: string
    MaxStackSize: int option
    OutputName: string
}

/// Parse dependency table from TOML content
let private parseDependencies (content: string) (projectDir: string) : (string * string) list =
    // Look for [dependencies] section and parse key = { path = "..." } entries
    let lines = content.Split([|'\n'|], StringSplitOptions.None)
    let mutable inDepsSection = false
    let deps = ResizeArray<string * string>()

    for line in lines do
        let trimmed = line.Trim()
        if trimmed = "[dependencies]" then
            inDepsSection <- true
        elif trimmed.StartsWith("[") && trimmed.EndsWith("]") then
            inDepsSection <- false
        elif inDepsSection && trimmed.Contains("=") && not (trimmed.StartsWith("#")) then
            // Parse: name = { path = "..." }
            let eqIdx = trimmed.IndexOf('=')
            if eqIdx > 0 then
                let name = trimmed.Substring(0, eqIdx).Trim()
                let rest = trimmed.Substring(eqIdx + 1).Trim()
                // Extract path from { path = "..." }
                let pathPattern = System.Text.RegularExpressions.Regex(@"path\s*=\s*""([^""]+)""")
                let m = pathPattern.Match(rest)
                if m.Success then
                    let path = m.Groups.[1].Value
                    deps.Add((name, path))

    deps |> Seq.toList

/// Find source files in a dependency directory
/// Looks for .fidproj in the directory to get proper source ordering,
/// otherwise collects .fs files in a sensible order
let private findDependencySourceFiles (depPath: string) : string list =
    if not (Directory.Exists depPath) then
        []
    else
        // Check for .fidproj in dependency
        let fidprojs = Directory.GetFiles(depPath, "*.fidproj")
        if fidprojs.Length > 0 then
            // Parse the dependency's .fidproj for its source list
            let depContent = File.ReadAllText(fidprojs.[0])
            let pattern = @"sources\s*=\s*\[([^\]]*)\]"
            let regex = System.Text.RegularExpressions.Regex(pattern, System.Text.RegularExpressions.RegexOptions.Singleline)
            let m = regex.Match(depContent)
            if m.Success then
                m.Groups.[1].Value.Split([|','; '\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
                |> Array.map (fun s -> s.Trim().Trim('"'))
                |> Array.filter (fun s -> not (String.IsNullOrWhiteSpace(s)) && not (s.StartsWith("#")))
                |> Array.map (fun s -> Path.Combine(depPath, s))
                |> Array.filter File.Exists
                |> Array.toList
            else
                // Fallback: collect .fs files
                Directory.GetFiles(depPath, "*.fs", SearchOption.AllDirectories)
                |> Array.toList
        else
            // No .fidproj - collect .fs files, trying to order sensibly
            // Put files with "Core" or "Types" first, "Platform" files last
            let allFiles = Directory.GetFiles(depPath, "*.fs", SearchOption.AllDirectories) |> Array.toList
            allFiles
            |> List.sortBy (fun f ->
                let name = Path.GetFileNameWithoutExtension(f).ToLowerInvariant()
                let dir = Path.GetDirectoryName(f).ToLowerInvariant()
                if name.Contains("core") || name.Contains("types") then 0
                elif name.Contains("platform") || dir.Contains("platform") then 2
                else 1)

/// Resolve a dependency path, with fallback to repos directory
let private resolveDependencyPath (name: string) (relativePath: string) (projectDir: string) : string option =
    // First try the relative path from project
    let directPath = Path.GetFullPath(Path.Combine(projectDir, relativePath))
    if Directory.Exists directPath then
        Some directPath
    else
        // Try adding /src suffix
        let withSrc = Path.Combine(directPath, "src")
        if Directory.Exists withSrc then
            Some withSrc
        else
            // Fallback: look in ~/repos/<name>/src
            let homeDir = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)
            let reposPath = Path.Combine(homeDir, "repos", name, "src")
            if Directory.Exists reposPath then
                printfn "[FidprojLoader] Resolved '%s' via repos fallback: %s" name reposPath
                Some reposPath
            else
                // Try capitalized name
                let capitalizedName =
                    if name.Length > 0 then
                        Char.ToUpperInvariant(name.[0]).ToString() + name.Substring(1)
                    else name
                let capitalizedPath = Path.Combine(homeDir, "repos", capitalizedName, "src")
                if Directory.Exists capitalizedPath then
                    printfn "[FidprojLoader] Resolved '%s' via repos fallback (capitalized): %s" name capitalizedPath
                    Some capitalizedPath
                else
                    None

/// Parse sources array from TOML content
let private parseSources (content: string) : string list =
    let pattern = @"sources\s*=\s*\[([^\]]*)\]"
    let regex = System.Text.RegularExpressions.Regex(pattern, System.Text.RegularExpressions.RegexOptions.Singleline)
    let m = regex.Match(content)
    if m.Success then
        m.Groups.[1].Value.Split([|','; '\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
        |> Array.map (fun s -> s.Trim().Trim('"'))
        |> Array.filter (fun s -> not (String.IsNullOrWhiteSpace(s)) && not (s.StartsWith("#")))
        |> Array.toList
    else
        []

/// Load and resolve a .fidproj file
let loadFidproj (filePath: string) : Result<ResolvedFidproj, string> =
    try
        if not (File.Exists filePath) then
            Error (sprintf "Project file not found: %s" filePath)
        else
            let content = File.ReadAllText(filePath)
            let projectDir = Path.GetDirectoryName(Path.GetFullPath(filePath))
            let tomlMap = parseTOML content

            // Basic project info
            let name = getString tomlMap "package.name" "unnamed"
            let version = getString tomlMap "package.version" "0.1.0"
            let outputName = getString tomlMap "build.output" name
            let memoryModel = getString tomlMap "compilation.memory_model" "mixed"
            let maxStackSize = getInt tomlMap "compilation.max_stack_size"
            let target =
                match Map.tryFind "compilation.target" tomlMap with
                | Some (TOMLValue.String "native") -> None
                | Some (TOMLValue.String t) -> Some t
                | _ -> None

            // Parse project sources
            let projectSources =
                parseSources content
                |> List.map (fun s ->
                    if Path.IsPathRooted(s) then s
                    else Path.GetFullPath(Path.Combine(projectDir, s)))

            // If no sources specified, find .fs files in project dir
            let projectSources =
                if projectSources.IsEmpty then
                    Directory.GetFiles(projectDir, "*.fs") |> Array.toList
                else
                    projectSources

            // Resolve dependencies
            let depSpecs = parseDependencies content projectDir
            let resolvedDeps =
                depSpecs
                |> List.choose (fun (depName, depPath) ->
                    match resolveDependencyPath depName depPath projectDir with
                    | Some resolvedPath ->
                        let sources = findDependencySourceFiles resolvedPath
                        if sources.IsEmpty then
                            printfn "[FidprojLoader] Warning: No source files found for dependency '%s' at %s" depName resolvedPath
                            None
                        else
                            printfn "[FidprojLoader] Resolved dependency '%s': %d source files from %s" depName sources.Length resolvedPath
                            Some {
                                Name = depName
                                SourcePath = resolvedPath
                                SourceFiles = sources
                            }
                    | None ->
                        printfn "[FidprojLoader] Warning: Could not resolve dependency '%s' (path: %s)" depName depPath
                        None)

            // Combine: dependencies first (in order), then project sources
            let allSources =
                (resolvedDeps |> List.collect (fun d -> d.SourceFiles))
                @ projectSources

            Ok {
                Name = name
                Version = version
                ProjectDir = projectDir
                Sources = projectSources
                Dependencies = resolvedDeps
                AllSourcesInOrder = allSources
                Target = target
                MemoryModel = memoryModel
                MaxStackSize = maxStackSize
                OutputName = outputName
            }
    with ex ->
        Error (sprintf "Error loading project file: %s" ex.Message)

/// Find the .NET runtime directory for reference assemblies
let private findDotNetRefs () : string[] =
    // Find net9.0 reference assemblies (or fall back to net8.0)
    let dotnetShared = "/usr/share/dotnet/shared/Microsoft.NETCore.App"
    let versions =
        if Directory.Exists dotnetShared then
            Directory.GetDirectories(dotnetShared)
            |> Array.sortDescending  // Get newest first
            |> Array.filter (fun d ->
                let ver = Path.GetFileName(d)
                ver.StartsWith("9.") || ver.StartsWith("8."))
            |> Array.tryHead
        else
            None

    match versions with
    | Some runtimeDir ->
        // Get core assemblies needed for FCS type checking
        let coreAssemblies = [|
            "System.Runtime.dll"
            "System.Collections.dll"
            "System.Console.dll"
            "netstandard.dll"
        |]
        coreAssemblies
        |> Array.map (fun dll -> Path.Combine(runtimeDir, dll))
        |> Array.filter File.Exists
        |> Array.map (sprintf "-r:%s")
    | None -> [||]

/// Find FSharp.Core reference
let private findFSharpCore () : string option =
    let nugetPath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".nuget", "packages", "fsharp.core")

    if Directory.Exists nugetPath then
        // Find highest version
        Directory.GetDirectories(nugetPath)
        |> Array.sortDescending
        |> Array.tryHead
        |> Option.bind (fun versionDir ->
            // Prefer netstandard2.1, fall back to netstandard2.0
            let ns21 = Path.Combine(versionDir, "lib", "netstandard2.1", "FSharp.Core.dll")
            let ns20 = Path.Combine(versionDir, "lib", "netstandard2.0", "FSharp.Core.dll")
            if File.Exists ns21 then Some ns21
            elif File.Exists ns20 then Some ns20
            else None)
    else
        None

/// Create FSharpProjectOptions from resolved .fidproj
/// This bypasses the .fsproj/MSBuild path entirely
/// NOTE: We provide .NET references for FCS type-checking only.
/// Code generation ignores these and uses Alloy implementations.
let createProjectOptions (resolved: ResolvedFidproj) : FSharpProjectOptions =
    // Verify all source files exist
    for source in resolved.AllSourcesInOrder do
        if not (File.Exists source) then
            failwithf "Source file not found: %s" source

    // Find FSharp.Core for FCS
    let fsharpCoreRef =
        match findFSharpCore() with
        | Some path -> [| sprintf "-r:%s" path |]
        | None -> [||]

    // Find .NET runtime refs for FCS type checking
    let dotnetRefs = findDotNetRefs()

    // Build compiler options for Firefly
    // FCS needs these refs for type checking, but Firefly code gen uses Alloy
    let otherOptions = [|
        yield "--target:exe"
        yield "--targetprofile:netcore"
        yield "--define:FIREFLY"
        yield "--define:ZERO_ALLOCATION"
        yield "--nowarn:3391"  // Suppress FSharp.Core version warnings
        yield "--nowarn:FS3511" // Suppress experimental warnings
        yield "--nowarn:64"    // Suppress SRTP warnings
        yield "--nowarn:77"    // Suppress warning about implicit member constraints
        yield! fsharpCoreRef
        yield! dotnetRefs
    |]

    {
        ProjectFileName = Path.Combine(resolved.ProjectDir, resolved.Name + ".fidproj")
        ProjectId = None
        SourceFiles = resolved.AllSourcesInOrder |> List.toArray
        OtherOptions = otherOptions
        ReferencedProjects = [||]
        IsIncompleteTypeCheckEnvironment = false
        UseScriptResolutionRules = false
        LoadTime = DateTime.Now
        UnresolvedReferences = None
        OriginalLoadReferences = []
        Stamp = None
    }

/// Create an FSharpChecker configured for Firefly compilation
let createChecker () =
    FSharpChecker.Create(
        projectCacheSize = 50,
        keepAssemblyContents = true,
        keepAllBackgroundResolutions = false,
        enableBackgroundItemKeyStoreAndSemanticClassification = true
    )

/// Load project and get full check results
let loadAndCheckProject (filePath: string) = async {
    match loadFidproj filePath with
    | Error msg ->
        return Error msg
    | Ok resolved ->
        printfn "[FidprojLoader] Project '%s' resolved with %d source files" resolved.Name resolved.AllSourcesInOrder.Length

        let checker = createChecker()
        let projectOptions = createProjectOptions resolved

        printfn "[FidprojLoader] Parsing and type-checking..."
        let! checkResults = checker.ParseAndCheckProject(projectOptions)

        // Report any errors
        if checkResults.Diagnostics.Length > 0 then
            printfn "[FidprojLoader] Diagnostics:"
            for diag in checkResults.Diagnostics do
                let severity =
                    match diag.Severity with
                    | FSharp.Compiler.Diagnostics.FSharpDiagnosticSeverity.Error -> "ERROR"
                    | FSharp.Compiler.Diagnostics.FSharpDiagnosticSeverity.Warning -> "WARN"
                    | FSharp.Compiler.Diagnostics.FSharpDiagnosticSeverity.Info -> "INFO"
                    | _ -> "NOTE"
                printfn "  [%s] %s: %s" severity (Path.GetFileName(diag.FileName)) diag.Message

        if checkResults.HasCriticalErrors then
            return Error "Project has critical compilation errors"
        else
            // Get parse results for each file
            let parsingOptions, _ = checker.GetParsingOptionsFromProjectOptions(projectOptions)
            let! parseResults =
                projectOptions.SourceFiles
                |> Array.map (fun file -> async {
                    let source = FSharp.Compiler.Text.SourceText.ofString(File.ReadAllText(file))
                    return! checker.ParseFile(file, source, parsingOptions)
                })
                |> Async.Parallel

            return Ok (resolved, checkResults, parseResults, checker, projectOptions)
}
