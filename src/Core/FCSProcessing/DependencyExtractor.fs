module Core.FCSProcessing.DependencyExtractor

open System
open System.IO
open FSharp.Compiler.Syntax
open FSharp.Compiler.Text
open FSharp.Compiler.CodeAnalysis
open Core.XParsec.Foundation

/// Extract string value from ParsedHashDirectiveArgument
let private extractArgString (arg: ParsedHashDirectiveArgument) : string option =
    match arg with
    | ParsedHashDirectiveArgument.String(s, _, _) -> Some s
    | ParsedHashDirectiveArgument.SourceIdentifier(_, s, _) -> Some s

/// Extract only explicit #load directives from the main file
let getExplicitLoadFiles (mainFile: string) : string list =
    printfn "    Parsing main file for directives..."
    
    // Create a simple checker just for parsing
    let checker = FSharpChecker.Create()
    
    // Read the file and look for #if INTERACTIVE block manually
    let lines = File.ReadAllLines(mainFile)
    let mutable inInteractiveBlock = false
    let mutable loadFiles = []
    let mutable includeDir = ""
    
    for line in lines do
        let trimmed = line.Trim()
        if trimmed = "#if INTERACTIVE" then
            inInteractiveBlock <- true
            printfn "    Found #if INTERACTIVE block"
        elif trimmed = "#endif" || trimmed = "#else" then
            inInteractiveBlock <- false
        elif inInteractiveBlock then
            if trimmed.StartsWith("#I ") then
                includeDir <- trimmed.Substring(3).Trim().Trim('"')
                printfn "    Found #I %s" includeDir
            elif trimmed.StartsWith("#load ") then
                let file = trimmed.Substring(6).Trim().Trim('"')
                printfn "    Found #load %s" file
                loadFiles <- file :: loadFiles
    
    // Resolve paths
    let baseDir = Path.GetDirectoryName(mainFile)
    let resolvedIncludeDir = 
        if Path.IsPathRooted(includeDir) then includeDir
        else Path.GetFullPath(Path.Combine(baseDir, includeDir))
    
    loadFiles 
    |> List.rev
    |> List.map (fun file ->
        let fullPath = Path.GetFullPath(Path.Combine(resolvedIncludeDir, file))
        printfn "    Resolved to: %s" fullPath
        fullPath)

/// Build complete project options with only explicitly loaded dependencies
let buildProjectOptionsWithDependencies (mainFile: string) (checker: FSharpChecker) =
    async {
        printfn "  Extracting explicit #load dependencies from: %s" mainFile
        
        // Get only the files explicitly loaded
        let loadedFiles = getExplicitLoadFiles mainFile
        
        // Build source file list: loaded files first, then main file
        let allFiles = loadedFiles @ [mainFile]
        
        printfn "  Explicit files to compile: %d" allFiles.Length
        allFiles |> List.iteri (fun i f -> 
            printfn "    %d: %s" i (Path.GetFileName f))
        
        // Create project options manually
        let projectName = Path.GetFileNameWithoutExtension(mainFile) + ".fireflyproj"
        
        return {
            ProjectFileName = projectName
            ProjectId = None
            SourceFiles = allFiles |> Array.ofList
            OtherOptions = [| 
                "--target:exe"
                "--noframework"
                "--nowin32manifest"
                "--define:ZERO_ALLOCATION"
                "--define:FIDELITY"
                "--define:INTERACTIVE"  // Always define for script compilation
            |]
            ReferencedProjects = [||]
            IsIncompleteTypeCheckEnvironment = false
            UseScriptResolutionRules = true
            LoadTime = DateTime.Now
            UnresolvedReferences = None
            OriginalLoadReferences = []
            Stamp = None
        }
    }