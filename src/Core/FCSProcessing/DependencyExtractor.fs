module Core.FCSProcessing.DependencyExtractor

open System
open System.IO
open FSharp.Compiler.Syntax
open FSharp.Compiler.Text
open FSharp.Compiler.CodeAnalysis
open Core.XParsec.Foundation

/// Extracted script directive information
type ScriptDirective =
    | IncludePath of string          // #I directive
    | LoadFile of string             // #load directive
    | Reference of string            // #r directive

/// Extract string value from ParsedHashDirectiveArgument
let private extractArgString (arg: ParsedHashDirectiveArgument) : string option =
    match arg with
    | ParsedHashDirectiveArgument.String(s, _, _) -> Some s
    | ParsedHashDirectiveArgument.SourceIdentifier(_, s, _) -> Some s

/// Extract directives from hash directives in the AST
let extractDirectives (hashDirectives: ParsedHashDirective list) : ScriptDirective list =
    printfn "  Extracting directives from %d hash directives" hashDirectives.Length
    hashDirectives |> List.choose (fun directive ->
        match directive with
        | ParsedHashDirective(sourcePath, args, _) ->
            printfn "    Found directive: #%s" sourcePath
            match sourcePath with
            | "I" -> 
                // #I "path"
                args |> List.tryHead |> Option.bind extractArgString |> Option.map IncludePath
            | "load" -> 
                // #load "file.fsx"
                args |> List.tryHead |> Option.bind extractArgString |> Option.map LoadFile
            | "r" ->
                // #r "assembly"
                args |> List.tryHead |> Option.bind extractArgString |> Option.map Reference
            | _ -> None
    )

/// Resolve full paths based on include paths and base directory
let resolvePaths (baseDir: string) (directives: ScriptDirective list) : (string list * string list) =
    let mutable includePaths = [baseDir]
    let mutable filesToLoad = []
    
    printfn "  Resolving paths from base directory: %s" baseDir
    
    for directive in directives do
        match directive with
        | IncludePath path ->
            // Handle both forward and backward slashes
            let normalizedPath = path.Replace('\\', Path.DirectorySeparatorChar).Replace('/', Path.DirectorySeparatorChar)
            // Resolve relative to base directory
            let fullPath = 
                if Path.IsPathRooted(normalizedPath) then normalizedPath
                else Path.GetFullPath(Path.Combine(baseDir, normalizedPath))
            printfn "    Include path: %s -> %s" path fullPath
            includePaths <- fullPath :: includePaths
            
        | LoadFile file ->
            printfn "    Looking for file: %s" file
            // Try to resolve file in include paths
            let resolved = 
                includePaths 
                |> List.tryPick (fun includePath ->
                    let fullPath = Path.GetFullPath(Path.Combine(includePath, file))
                    printfn "      Checking: %s" fullPath
                    if File.Exists(fullPath) then 
                        printfn "      Found!" 
                        Some fullPath
                    else None)
            
            match resolved with
            | Some path -> 
                filesToLoad <- path :: filesToLoad
            | None -> 
                // Try relative to base directory
                let relativePath = Path.GetFullPath(Path.Combine(baseDir, file))
                if File.Exists(relativePath) then
                    printfn "      Found at: %s" relativePath
                    filesToLoad <- relativePath :: filesToLoad
                else
                    printfn "      ERROR: Could not resolve file: %s" file
                    failwithf "Could not resolve file: %s" file
                    
        | Reference _ ->
            // Ignore assembly references for now
            ()
    
    (List.rev includePaths, List.rev filesToLoad)

/// Recursively extract all dependencies from a script file
let rec extractAllDependencies (visitedFiles: Set<string>) (filePath: string) : string list =
    let normalizedPath = Path.GetFullPath(filePath)
    printfn "  Processing file: %s" normalizedPath
    
    if Set.contains normalizedPath visitedFiles then
        printfn "    Already visited, skipping"
        []
    else
        let visitedFiles' = Set.add normalizedPath visitedFiles
        let baseDir = Path.GetDirectoryName(normalizedPath)
        
        try
            // Parse the file to get directives
            let sourceText = File.ReadAllText(normalizedPath)
            let checker = FSharpChecker.Create()
            
            printfn "    Parsing file..."
            // Simple parsing to extract hash directives
            let parsingOptions = {
                FSharpParsingOptions.Default with 
                    SourceFiles = [| normalizedPath |]
                    IsInteractive = true  // Important for .fsx files!
                    LangVersionText = "preview"
            }
            
            // Simple parsing to extract hash directives
            let parseResult = 
                checker.ParseFile(
                    normalizedPath, 
                    SourceText.ofString sourceText,
                    parsingOptions)
                |> Async.RunSynchronously
            
            if parseResult.ParseHadErrors then
                printfn "    WARNING: Parse errors in %s" normalizedPath
                parseResult.Diagnostics |> Array.iter (fun d -> printfn "      %s" d.Message)
            
            match parseResult.ParseTree with
            | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, hashDirectives, _, _, _, _)) ->
                printfn "    Found %d hash directives" hashDirectives.Length
                let directives = extractDirectives hashDirectives
                let (_, files) = resolvePaths baseDir directives
                
                printfn "    Dependencies: %d files" files.Length
                files |> List.iter (fun f -> printfn "      - %s" f)
                
                // Get this file plus all its dependencies recursively
                normalizedPath :: 
                (files |> List.collect (extractAllDependencies visitedFiles'))
                
            | _ -> 
                printfn "    Not an implementation file"
                [normalizedPath]
                
        with ex ->
            printfn "    ERROR: %s" ex.Message
            // Even if we can't parse, include the file
            [normalizedPath]

/// Build complete project options with all dependencies
let buildProjectOptionsWithDependencies (mainFile: string) (checker: FSharpChecker) =
    printfn "  Building project options for: %s" mainFile
    
    let allFiles = 
        try
            extractAllDependencies Set.empty mainFile 
            |> List.distinct
            |> List.map Path.GetFullPath  // Normalize all paths
        with ex ->
            printfn "  ERROR extracting dependencies: %s" ex.Message
            // Fallback to just the main file
            [Path.GetFullPath(mainFile)]
    
    printfn "  Total files discovered: %d" allFiles.Length
    allFiles |> List.iteri (fun i f -> printfn "    %d: %s" i f)
    
    // Use a temporary project name based on the input file
    let projectName = Path.GetFileNameWithoutExtension(mainFile) + ".fireflyproj"
    
    {
        ProjectFileName = projectName
        ProjectId = None
        SourceFiles = allFiles |> Array.ofList
        OtherOptions = [| 
            "--target:exe"
            "--noframework"
            "--nowin32manifest"
            "--define:ZERO_ALLOCATION"
            "--define:FIDELITY"
        |]
        ReferencedProjects = [||]
        IsIncompleteTypeCheckEnvironment = false
        UseScriptResolutionRules = true  // Important for script files!
        LoadTime = DateTime.Now
        UnresolvedReferences = None
        OriginalLoadReferences = []
        Stamp = None
    }: FSharpProjectOptions