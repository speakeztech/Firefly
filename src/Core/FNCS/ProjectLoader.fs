/// Project Loading for FNCS
/// Loads .fidproj files and invokes FNCS type checking.
module Core.FNCS.ProjectLoader

open System
open System.IO
open FSharp.Native.Compiler.Syntax
open FSharp.Native.Compiler.NativeService
open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open FSharp.Native.Compiler.Project

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

/// Collect all F# source files from a directory (fallback only - prefer fidproj ordering)
let collectSourceFilesFallback (dirPath: string) : string list =
    if Directory.Exists dirPath then
        Directory.GetFiles(dirPath, "*.fs", SearchOption.AllDirectories)
        |> Array.filter (fun f ->
            let name = Path.GetFileName(f)
            // Skip signature files and hidden files
            not (name.EndsWith(".fsi")) &&
            not (name.StartsWith(".")))
        |> Array.toList
        |> List.sort
    else
        []

/// Load project sources and check them using FNCS with shared type environment.
/// This follows the FCS pattern: parse all files first, then check together.
/// Environment is threaded through files so earlier files' types/bindings are
/// visible to later files.
let loadAndCheck (config: ProjectConfig) : ProjectResult =
    printfn "[FNCS] Loading project: %s" config.Name
    printfn "[FNCS] Sources: %A" config.Sources

    // Collect Alloy sources using FNCS SourceResolver - respects Alloy.fidproj ordering
    // This is the PRINCIPLED approach: dependencies declare their own source ordering
    let alloySources =
        match config.AlloyPath with
        | Some path ->
            // Use SourceResolver to get sources in the order defined by Alloy.fidproj
            let sources = SourceResolver.getAlloySources path
            if List.isEmpty sources then
                // Fallback if no fidproj found (shouldn't happen with proper setup)
                printfn "[FNCS] Warning: No Alloy.fidproj found, using directory scan as fallback"
                collectSourceFilesFallback path
            else
                printfn "[FNCS] Loaded %d Alloy sources from: %s (ordered by Alloy.fidproj)" (List.length sources) path
                sources
        | None ->
            []

    // All sources: Alloy first (for type definitions), then project sources
    let allSources = alloySources @ config.Sources
    printfn "[FNCS] Total sources to check: %d" (List.length allSources)

    // STEP 1: Parse all files first (collect ParsedInputs)
    let parsedInputs, parseErrors =
        allSources
        |> List.fold (fun (parsed, errors) filePath ->
            match parseSourceFile filePath with
            | Some input -> (input :: parsed, errors)
            | None -> (parsed, sprintf "Failed to parse: %s" filePath :: errors)
        ) ([], [])

    // Parsed inputs need to be in order (Alloy first, then project)
    let parsedInputs = List.rev parsedInputs
    let parseErrors = List.rev parseErrors

    if not (List.isEmpty parseErrors) then
        printfn "[FNCS] Parse errors:"
        for err in parseErrors do
            printfn "[FNCS]   %s" err

    if List.isEmpty parsedInputs then
        printfn "[FNCS] No files were parsed successfully"
        {
            Config = config
            CheckResult = {
                Graph = SemanticGraph.empty
                Diagnostics =
                    parseErrors
                    |> List.map (fun msg -> {
                        Severity = NativeDiagnosticSeverity.Error
                        Code = "FF0001"
                        Message = msg
                        Range = dummyRange
                        RelatedNodes = []
                    })
            }
        }
    else
        printfn "[FNCS] Parsed %d file(s), checking with shared environment..." (List.length parsedInputs)

        // STEP 2: Check all files together with shared type environment
        // This is the FCS pattern - checkParsedInputs threads environment through files
        let checkResult = Core.FNCS.Integration.checkMultipleInputs parsedInputs

        printfn "[FNCS] Check complete: %d nodes, %d entry points"
            (Map.count checkResult.Graph.Nodes)
            (List.length checkResult.Graph.EntryPoints)

        let errorCount =
            checkResult.Diagnostics
            |> List.filter (fun d -> d.Severity = NativeDiagnosticSeverity.Error)
            |> List.length

        if errorCount > 0 then
            printfn "[FNCS] Type checking found %d error(s)" errorCount
            for diag in checkResult.Diagnostics do
                if diag.Severity = NativeDiagnosticSeverity.Error then
                    printfn "[FNCS]   ERROR: %s" diag.Message

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
    let errorCount = result.CheckResult.Diagnostics |> List.filter (fun d -> d.Severity = NativeDiagnosticSeverity.Error) |> List.length
    let warnCount = result.CheckResult.Diagnostics |> List.filter (fun d -> d.Severity = NativeDiagnosticSeverity.Warning) |> List.length

    sprintf "[FNCS] %s: %d nodes, %d entry points, %d errors, %d warnings"
        result.Config.Name nodeCount entryCount errorCount warnCount
