/// Project Loading for FNCS
/// Loads .fidproj files and invokes FNCS type checking.
module Core.FNCS.ProjectLoader

open System
open System.IO
open FSharp.Native.Compiler.Syntax
open FSharp.Native.Compiler.NativeService
open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes

/// Project configuration loaded from .fidproj
type ProjectConfig = {
    Name: string
    Sources: string list
    AlloyPath: string option
    OutputKind: string  // "freestanding" or "console"
    TargetTriple: string option
}

/// Result of loading and checking a project
type ProjectResult = {
    Config: ProjectConfig
    CheckResult: CheckResult
}

/// Parse a single F# source file
let parseSourceFile (filePath: string) : ParsedInput option =
    if not (File.Exists filePath) then
        printfn "[FNCS] Source file not found: %s" filePath
        None
    else
        try
            let source = File.ReadAllText filePath
            // TODO: Use FNCS parser when available
            // For now, we need to call into the parser
            // This is a placeholder - the actual parsing API needs to be exposed from FNCS
            printfn "[FNCS] Parsing: %s" filePath
            None  // Placeholder
        with ex ->
            printfn "[FNCS] Parse error in %s: %s" filePath ex.Message
            None

/// Check a parsed file and return the semantic graph
let checkFile (parsed: ParsedInput) : CheckResult =
    checkParsedInput parsed

/// Load project sources and check them
let loadAndCheck (config: ProjectConfig) : ProjectResult =
    printfn "[FNCS] Loading project: %s" config.Name
    printfn "[FNCS] Sources: %A" config.Sources

    // Parse all source files
    let parsedFiles =
        config.Sources
        |> List.choose parseSourceFile

    if List.isEmpty parsedFiles then
        printfn "[FNCS] Warning: No files were parsed successfully"
        {
            Config = config
            CheckResult = {
                Graph = SemanticGraph.empty
                Diagnostics = [{
                    Severity = DiagnosticSeverity.Error
                    Code = "FF0001"
                    Message = "No source files were parsed"
                    Range = dummyRange
                    RelatedNodes = []
                }]
            }
        }
    else
        // For now, check the first file (multi-file support TODO)
        let checkResult =
            parsedFiles
            |> List.head
            |> checkFile

        {
            Config = config
            CheckResult = checkResult
        }

/// Get entry points from the semantic graph
let findEntryPoints (graph: SemanticGraph) : NodeId list =
    graph.EntryPoints

/// Get all modules from the semantic graph
let getModules (graph: SemanticGraph) : Map<ModulePath, NodeId list> =
    graph.Modules

/// Summary of check result for logging
let summarize (result: ProjectResult) : string =
    let graph = result.CheckResult.Graph
    let nodeCount = Map.count graph.Nodes
    let entryCount = List.length graph.EntryPoints
    let errorCount = result.CheckResult.Diagnostics |> List.filter (fun d -> d.Severity = DiagnosticSeverity.Error) |> List.length
    let warnCount = result.CheckResult.Diagnostics |> List.filter (fun d -> d.Severity = DiagnosticSeverity.Warning) |> List.length

    sprintf "[FNCS] %s: %d nodes, %d entry points, %d errors, %d warnings"
        result.Config.Name nodeCount entryCount errorCount warnCount
