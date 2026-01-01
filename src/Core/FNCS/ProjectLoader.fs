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

/// Parse a single F# source file using FNCS
let parseSourceFile (filePath: string) : ParsedInput option =
    if not (File.Exists filePath) then
        printfn "[FNCS] Source file not found: %s" filePath
        None
    else
        try
            let source = File.ReadAllText filePath
            printfn "[FNCS] Parsing: %s" filePath
            match Core.FNCS.Integration.parseSource source filePath with
            | ParseSuccess input ->
                printfn "[FNCS] Parse successful: %s" filePath
                Some input
            | ParseError errors ->
                for err in errors do
                    printfn "[FNCS] Parse error: %s" err
                None
        with ex ->
            printfn "[FNCS] Parse exception in %s: %s" filePath ex.Message
            None

/// Parse and type-check a source file in one step
let parseAndCheckFile (filePath: string) : ParseAndCheckResult =
    if not (File.Exists filePath) then
        printfn "[FNCS] Source file not found: %s" filePath
        ParseFailure [$"File not found: {filePath}"]
    else
        try
            let source = File.ReadAllText filePath
            printfn "[FNCS] Parsing and checking: %s" filePath

            // Debug: try just parsing first to see the AST structure
            match Core.FNCS.Integration.parseSource source filePath with
            | FSharp.Native.Compiler.NativeService.ParseError errors ->
                printfn "[FNCS] Debug: Parse error before full check"
                for err in errors do
                    printfn "[FNCS] Debug:   %s" err
            | FSharp.Native.Compiler.NativeService.ParseSuccess parsedInput ->
                match parsedInput with
                | FSharp.Native.Compiler.Syntax.ParsedInput.ImplFile implFile ->
                    let (FSharp.Native.Compiler.Syntax.ParsedImplFileInput(_, _, qualName, _, contents, _, _, _)) = implFile
                    printfn "[FNCS] Debug: Parsed ImplFile with %d module(s)/namespace(s)" (List.length contents)
                    for modOrNs in contents do
                        let (FSharp.Native.Compiler.Syntax.SynModuleOrNamespace(longId, _, kind, decls, _, _, _, _, _)) = modOrNs
                        let moduleName = longId |> List.map (fun id -> id.idText) |> String.concat "."
                        let kindStr =
                            match kind with
                            | FSharp.Native.Compiler.Syntax.SynModuleOrNamespaceKind.NamedModule -> "NamedModule"
                            | FSharp.Native.Compiler.Syntax.SynModuleOrNamespaceKind.AnonModule -> "AnonModule"
                            | FSharp.Native.Compiler.Syntax.SynModuleOrNamespaceKind.DeclaredNamespace -> "DeclaredNamespace"
                            | FSharp.Native.Compiler.Syntax.SynModuleOrNamespaceKind.GlobalNamespace -> "GlobalNamespace"
                        printfn "[FNCS] Debug:   Module '%s' (kind=%s) with %d decls" moduleName kindStr (List.length decls)
                        for decl in decls do
                            let declStr =
                                match decl with
                                | FSharp.Native.Compiler.Syntax.SynModuleDecl.Let(_, bindings, _, _) ->
                                    sprintf "Let (bindings=%d)" (List.length bindings)
                                | FSharp.Native.Compiler.Syntax.SynModuleDecl.Expr(_, _) -> "Expr"
                                | FSharp.Native.Compiler.Syntax.SynModuleDecl.Open(_, _) -> "Open"
                                | FSharp.Native.Compiler.Syntax.SynModuleDecl.Types(types, _) ->
                                    sprintf "Types (count=%d)" (List.length types)
                                | FSharp.Native.Compiler.Syntax.SynModuleDecl.NestedModule(_, _, _, _, _, _) -> "NestedModule"
                                | _ -> "Other"
                            printfn "[FNCS] Debug:     - %s" declStr
                | FSharp.Native.Compiler.Syntax.ParsedInput.SigFile _ ->
                    printfn "[FNCS] Debug: Parsed SigFile (signature)"

            Core.FNCS.Integration.parseAndCheckSource source filePath
        with ex ->
            printfn "[FNCS] Exception in %s: %s" filePath ex.Message
            ParseFailure [$"Exception: {ex.Message}"]

/// Check a parsed file and return the semantic graph
let checkFile (parsed: ParsedInput) : CheckResult =
    checkParsedInput parsed

/// Load project sources and check them using FNCS parseAndCheck
let loadAndCheck (config: ProjectConfig) : ProjectResult =
    printfn "[FNCS] Loading project: %s" config.Name
    printfn "[FNCS] Sources: %A" config.Sources

    // Parse and check all source files
    let results =
        config.Sources
        |> List.map parseAndCheckFile

    // Collect successful results and errors
    let successes, failures =
        results
        |> List.fold (fun (succ, fail) result ->
            match result with
            | Success r -> (r :: succ, fail)
            | ParseFailure errs -> (succ, errs @ fail)
            | CheckFailure r ->
                let errs = r.Diagnostics |> List.map (fun d -> d.Message)
                (succ, errs @ fail)
        ) ([], [])

    if List.isEmpty successes then
        let errorMsgs = if List.isEmpty failures then ["No files were parsed"] else failures
        printfn "[FNCS] Warning: No files were checked successfully"
        for err in errorMsgs do
            printfn "[FNCS] Error: %s" err
        {
            Config = config
            CheckResult = {
                Graph = SemanticGraph.empty
                Diagnostics =
                    errorMsgs
                    |> List.map (fun msg -> {
                        Severity = DiagnosticSeverity.Error
                        Code = "FF0001"
                        Message = msg
                        Range = dummyRange
                        RelatedNodes = []
                    })
            }
        }
    else
        // For now, use the first successful result (multi-file merging TODO)
        let firstResult = List.head successes
        printfn "[FNCS] Successfully checked %d file(s)" (List.length successes)
        printfn "[FNCS] Graph has %d nodes, %d entry points"
            (Map.count firstResult.Graph.Nodes)
            (List.length firstResult.Graph.EntryPoints)
        {
            Config = config
            CheckResult = firstResult
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
