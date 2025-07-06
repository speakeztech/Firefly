module Core.FCSIngestion.ProjectOptionsLoader

open System
open System.IO
open FSharp.Compiler.CodeAnalysis
open XParsec
open XParsec.CharParsers

/// Project file data extracted from .fsproj
type ProjectFileData = {
    OutputType: string option
    TargetFramework: string option
    NoStdLib: bool
    TreatWarningsAsErrors: bool
    OtherFlags: string list
    CompileItems: string list
    NoWarns: string list
}

module ProjectFileParser =
    open XParsec.Parsers
    
    // Basic XML parsing helpers
    let pWhitespace = skipMany (anyOf " \t\r\n")
    
    let pUntilChar c = manyChars (noneOf [c])
    
    let pQuotedString = 
        between (pchar '"') (pchar '"') (manyChars (noneOf "\""))
    
    // Parse any XML tag (opening or self-closing)
    let pXmlTag =
        parser {
            let! _ = pchar '<'
            let! _ = pWhitespace
            let! tagContent = manyCharsTill anyChar (pchar '>')
            return tagContent |> fst
        }
    
    // Find and extract content of a specific tag
    let extractTagContent (tagName: string) (content: string) =
        // Pattern 1: <TagName>content</TagName>
        let pattern1 = 
            parser {
                let! _ = manyCharsTill anyChar (pstring $"<{tagName}>")
                let! content = manyCharsTill anyChar (pstring $"</{tagName}>")
                return content |> fst |> (fun s -> s.Trim())
            }
        
        // Pattern 2: <TagName />  (self-closing, means empty/default)
        let pattern2 = 
            parser {
                let! _ = manyCharsTill anyChar (pstring $"<{tagName}")
                let! _ = pWhitespace
                let! _ = pstring "/>"
                return ""
            }
        
        let p = pattern1 <|> pattern2
        
        match p (Reader.ofString content ()) with
        | Ok result -> Some result.Parsed
        | Error _ -> None
    
    // Extract all Compile Include items
    let extractCompileItems (content: string) =
        let pCompileInclude =
            parser {
                let! _ = pstring "<Compile"
                let! _ = spaces1
                let! _ = pstring "Include="
                let! path = pQuotedString
                let! _ = manyCharsTill anyChar (pchar '>')  // Handle both /> and ></Compile>
                return path
            }
        
        let pAllCompileItems = 
            many (parser {
                let! _ = manyCharsTill anyChar (lookAhead (pstring "<Compile"))
                return! pCompileInclude
            })
        
        match pAllCompileItems (Reader.ofString content ()) with
        | Ok result -> result.Parsed |> List.ofSeq
        | Error _ -> []
    
    // Parse the project file
    let parseProjectData (content: string) =
        let getTagValue tagName = extractTagContent tagName content
        
        let getBoolValue tagName =
            match getTagValue tagName with
            | Some "true" -> true
            | _ -> false
        
        let getListValue tagName (separator: char) =
            match getTagValue tagName with
            | Some value -> 
                value.Split(separator) 
                |> Array.map (fun s -> s.Trim())
                |> Array.filter (String.IsNullOrWhiteSpace >> not) 
                |> Array.toList
            | None -> []
        
        {
            OutputType = getTagValue "OutputType"
            TargetFramework = getTagValue "TargetFramework"
            NoStdLib = getBoolValue "NoStdLib"
            TreatWarningsAsErrors = getBoolValue "TreatWarningsAsErrors"
            OtherFlags = getListValue "OtherFlags" ' '
            CompileItems = extractCompileItems content
            NoWarns = getListValue "NoWarn" ';'
        }

/// Load a project file and create FCS options
let loadProject (projectFile: string) (checker: FSharpChecker) =
    async {
        printfn "[ProjectLoader] Loading project: %s" projectFile
        
        if not (File.Exists projectFile) then
            return Error (sprintf "Project file not found: %s" projectFile)
        else
            try
                let projectDir = Path.GetDirectoryName(projectFile)
                let projectContent = File.ReadAllText(projectFile)
                
                // Parse project file using XParsec
                let projData = ProjectFileParser.parseProjectData projectContent
                
                // Convert compile items to full paths
                let sourceFiles = 
                    projData.CompileItems 
                    |> List.map (fun item -> Path.GetFullPath(Path.Combine(projectDir, item)))
                    |> List.toArray
                
                printfn "[ProjectLoader] Parsed project data:"
                printfn "  Output type: %s" (projData.OutputType |> Option.defaultValue "Exe")
                printfn "  Target framework: %s" (projData.TargetFramework |> Option.defaultValue "net8.0")
                printfn "  NoStdLib: %b" projData.NoStdLib
                printfn "  Source files: %d" sourceFiles.Length
                sourceFiles |> Array.iteri (fun i f -> 
                    printfn "    [%02d] %s" i (Path.GetFileName f))
                
                // Build compiler options from parsed data
                let compilerOptions = [
                    if projData.NoStdLib then "--nostdlib"
                    if projData.TreatWarningsAsErrors then "--warnaserror+"
                    yield! projData.OtherFlags
                    yield! projData.NoWarns |> List.map (sprintf "--nowarn:%s")
                    "--noframework"  // Always for Firefly
                    "--standalone"   // Ensure standalone compilation
                    "--define:NO_MSCORLIB" 
                    "--define:NO_SYSTEM_REFERENCE"
                    "--define:FIREFLY"
                    "--define:ZERO_ALLOCATION"
                ]
                
                // Create FSharpProjectOptions
                let projectOptions : FSharpProjectOptions = {
                    ProjectFileName = projectFile
                    ProjectId = None
                    SourceFiles = sourceFiles
                    OtherOptions = compilerOptions |> List.toArray
                    ReferencedProjects = [||]
                    IsIncompleteTypeCheckEnvironment = false
                    UseScriptResolutionRules = false
                    LoadTime = DateTime.Now
                    UnresolvedReferences = None
                    OriginalLoadReferences = []
                    Stamp = None
                }
                
                return Ok projectOptions
                
            with ex ->
                return Error (sprintf "Failed to load project: %s" ex.Message)
    }

/// Analyze project options and report statistics
let analyzeProjectOptions (options: FSharpProjectOptions) =
    printfn "\n[ProjectLoader] Project Analysis:"
    printfn "  Project: %s" (Path.GetFileName options.ProjectFileName)
    printfn "  Source files: %d" options.SourceFiles.Length
    printfn "  Compiler options:"
    options.OtherOptions |> Array.iter (printfn "    %s")
    
    let hasNoStdLib = options.OtherOptions |> Array.exists ((=) "--nostdlib")
    let hasFirefly = options.OtherOptions |> Array.exists (fun o -> o.Contains("FIREFLY"))
    
    printfn "\n  Firefly compliance:"
    printfn "    ✓ No stdlib: %b" hasNoStdLib
    printfn "    ✓ Firefly mode: %b" hasFirefly