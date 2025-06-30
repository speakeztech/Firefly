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
    hashDirectives |> List.choose (fun directive ->
        match directive with
        | ParsedHashDirective(sourcePath, args, _) ->
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
    
    for directive in directives do
        match directive with
        | IncludePath path ->
            // Resolve relative to base directory
            let fullPath = 
                if Path.IsPathRooted(path) then path
                else Path.Combine(baseDir, path)
            includePaths <- fullPath :: includePaths
            
        | LoadFile file ->
            // Try to resolve file in include paths
            let resolved = 
                includePaths 
                |> List.tryPick (fun includePath ->
                    let fullPath = Path.Combine(includePath, file)
                    if File.Exists(fullPath) then Some fullPath
                    else None)
            
            match resolved with
            | Some path -> filesToLoad <- path :: filesToLoad
            | None -> 
                // Try relative to base directory
                let relativePath = Path.Combine(baseDir, file)
                if File.Exists(relativePath) then
                    filesToLoad <- relativePath :: filesToLoad
                else
                    failwithf "Could not resolve file: %s" file
                    
        | Reference _ ->
            // Ignore assembly references for now
            ()
    
    (List.rev includePaths, List.rev filesToLoad)

/// Recursively extract all dependencies from a script file
let rec extractAllDependencies (visitedFiles: Set<string>) (filePath: string) : string list =
    if Set.contains filePath visitedFiles then
        []
    else
        let visitedFiles' = Set.add filePath visitedFiles
        let baseDir = Path.GetDirectoryName(filePath)
        
        // Parse the file to get directives
        let sourceText = File.ReadAllText(filePath)
        let checker = FSharpChecker.Create()
        
        // Simple parsing to extract hash directives
        let parseResult = 
            checker.ParseFile(
                filePath, 
                SourceText.ofString sourceText,
                FSharpParsingOptions.Default)
            |> Async.RunSynchronously
        
        match parseResult.ParseTree with
        | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, hashDirectives, _, _, _, _)) ->
            let directives = extractDirectives hashDirectives
            let (_, files) = resolvePaths baseDir directives
            
            // Get this file plus all its dependencies recursively
            filePath :: 
            (files |> List.collect (extractAllDependencies visitedFiles'))
            
        | _ -> [filePath]

/// Build complete project options with all dependencies
let buildProjectOptionsWithDependencies (mainFile: string) (checker: FSharpChecker) =
    let allFiles = extractAllDependencies Set.empty mainFile |> List.distinct
    
    // Use a temporary project name based on the input file
    let projectName = Path.GetFileNameWithoutExtension(mainFile) + ".fidproj"
    
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