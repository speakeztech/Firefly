module CLI.Commands.CompileCommand

open System
open System.IO
open System.Runtime.InteropServices
open Argu
open CLI.Configurations.FidprojLoader
open Core.PSG.Builder
open Core.PSG.Reachability
open Alex.Pipeline.CompilationOrchestrator

/// Command line arguments for compile command
type CompileArgs =
    | [<MainCommand; Unique>] Project of path: string
    | [<AltCommandLine("-o")>] Output of path: string
    | [<AltCommandLine("-t")>] Target of target: string
    | [<AltCommandLine("-k")>] Keep_Intermediates
    | [<AltCommandLine("-v")>] Verbose
    | Emit_MLIR
    | Emit_LLVM

    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Project _ -> ".fidproj file or F# source file to compile"
            | Output _ -> "Output executable path"
            | Target _ -> "Target triple (default: host platform)"
            | Keep_Intermediates -> "Keep intermediate files (.mlir, .ll) for debugging"
            | Verbose -> "Enable verbose output"
            | Emit_MLIR -> "Emit MLIR and stop (don't generate executable)"
            | Emit_LLVM -> "Emit LLVM IR and stop (don't generate executable)"

/// Get default target triple based on platform
let private getDefaultTarget() =
    if RuntimeInformation.IsOSPlatform OSPlatform.Windows then
        "x86_64-pc-windows-gnu"
    elif RuntimeInformation.IsOSPlatform OSPlatform.Linux then
        "x86_64-unknown-linux-gnu"
    elif RuntimeInformation.IsOSPlatform OSPlatform.OSX then
        "x86_64-apple-darwin"
    else
        "x86_64-unknown-unknown"

/// Progress reporting for verbose mode
let private report (verbose: bool) (phase: string) (message: string) =
    if verbose then
        printfn "[%s] %s" phase message

/// Find actual function name from entry point binding
/// Entry point nodes may have the attribute as their symbol, but the actual
/// function name is in the binding pattern or can be found from the symbol table
let private findEntryPointFunctionName (psg: Core.PSG.Types.ProgramSemanticGraph) (node: Core.PSG.Types.PSGNode) : string =
    // If this is a Binding:EntryPoint, the function name is "main" by convention
    // or we look for it in the symbol table
    if node.SyntaxKind = "Binding:EntryPoint" || node.SyntaxKind = "Binding:Main" then
        // Look for "main" in the symbol table
        match Map.tryFind "main" psg.SymbolTable with
        | Some sym -> sym.DisplayName
        | None -> "main"  // Default to main
    else
        match node.Symbol with
        | Some sym ->
            // If the symbol is an attribute, try to find the actual function
            if sym.DisplayName.EndsWith("Attribute") then
                "main"
            else
                sym.DisplayName
        | None -> "main"

/// Generate MLIR from PSG using Alex emission pipeline
let private generateMLIRFromPSG (psg: Core.PSG.Types.ProgramSemanticGraph) (projectName: string) (targetTriple: string) (_outputKind: Core.Types.MLIRTypes.OutputKind) : string =
    // Use Alex.Pipeline.CompilationOrchestrator.generateMLIRViaAlex
    generateMLIRViaAlex psg projectName targetTriple

/// Generate LLVM IR from PSG (simplified)
let private generateLLVMFromPSG (psg: Core.PSG.Types.ProgramSemanticGraph) (projectName: string) (targetTriple: string) : string =
    let sb = System.Text.StringBuilder()

    sb.AppendLine(sprintf "; Firefly-generated LLVM IR for %s" projectName) |> ignore
    sb.AppendLine(sprintf "; PSG Nodes: %d" psg.Nodes.Count) |> ignore
    sb.AppendLine() |> ignore
    sb.AppendLine(sprintf "target triple = \"%s\"" targetTriple) |> ignore
    sb.AppendLine() |> ignore

    // Generate function for each entry point
    for entryPointId in psg.EntryPoints do
        match Map.tryFind entryPointId.Value psg.Nodes with
        | Some node ->
            let funcName =
                match node.Symbol with
                | Some sym -> sym.DisplayName
                | None -> "main"

            sb.AppendLine(sprintf "define i32 @%s() {" funcName) |> ignore
            sb.AppendLine("entry:") |> ignore
            sb.AppendLine("  ret i32 0") |> ignore
            sb.AppendLine("}") |> ignore
            sb.AppendLine() |> ignore
        | None -> ()

    // Default main if no entry points
    if psg.EntryPoints.IsEmpty then
        sb.AppendLine("define i32 @main() {") |> ignore
        sb.AppendLine("entry:") |> ignore
        sb.AppendLine("  ret i32 0") |> ignore
        sb.AppendLine("}") |> ignore

    sb.ToString()

/// Main entry point for compile command
let execute (args: ParseResults<CompileArgs>) =
    let projectPath =
        match args.TryGetResult(Project) with
        | Some p -> p
        | None ->
            // Look for .fidproj in current directory
            let fidprojs = Directory.GetFiles(".", "*.fidproj")
            if fidprojs.Length = 1 then
                fidprojs.[0]
            elif fidprojs.Length > 1 then
                printfn "Error: Multiple .fidproj files found. Please specify which one to compile."
                exit 1
            else
                printfn "Error: No .fidproj file found and no project specified."
                printfn "Usage: firefly compile <project.fidproj>"
                exit 1

    let verbose = args.Contains(Verbose)
    let keepIntermediates = args.Contains(Keep_Intermediates)
    let emitMLIR = args.Contains(Emit_MLIR)
    let emitLLVM = args.Contains(Emit_LLVM)

    report verbose "INIT" (sprintf "Loading project: %s" projectPath)

    // =========================================================================
    // PHASE 1: Load and parse .fidproj, resolve dependencies
    // =========================================================================

    printfn "Firefly Compiler v0.4.164"
    printfn "========================="
    printfn ""

    let loadResult = loadAndCheckProject projectPath |> Async.RunSynchronously

    match loadResult with
    | Error msg ->
        printfn "Error: %s" msg
        1
    | Ok (resolved, checkResults, parseResults, checker, projectOptions) ->

        let targetTriple =
            args.TryGetResult(Target)
            |> Option.orElse resolved.Target
            |> Option.defaultValue (getDefaultTarget())

        let outputPath =
            args.TryGetResult(Output)
            |> Option.defaultValue (Path.Combine(resolved.ProjectDir, resolved.OutputName))

        printfn "Project: %s" resolved.Name
        printfn "Sources: %d files (%d from dependencies)"
            resolved.AllSourcesInOrder.Length
            (resolved.AllSourcesInOrder.Length - resolved.Sources.Length)
        printfn "Target:  %s" targetTriple
        printfn "Output:  %s" outputPath
        printfn ""

        // Create intermediates directory if needed
        let intermediatesDir =
            if keepIntermediates || emitMLIR || emitLLVM then
                let dir = Path.Combine(resolved.ProjectDir, "build", "intermediates")
                Directory.CreateDirectory(dir) |> ignore
                Some dir
            else
                None

        // =========================================================================
        // PHASE 2: Build Program Semantic Graph
        // =========================================================================

        report verbose "PSG" "Building Program Semantic Graph..."

        let psg = buildProgramSemanticGraph checkResults parseResults

        printfn "[PSG] Built: %d nodes, %d edges, %d entry points, %d symbols"
            psg.Nodes.Count psg.Edges.Length psg.EntryPoints.Length psg.SymbolTable.Count

        // =========================================================================
        // PHASE 3: Reachability Analysis
        // =========================================================================

        report verbose "REACH" "Analyzing reachability..."

        let reachabilityResult = performReachabilityAnalysis psg

        printfn "[REACH] %d/%d symbols reachable (%.1f%% eliminated)"
            reachabilityResult.PruningStatistics.ReachableSymbols
            reachabilityResult.PruningStatistics.TotalSymbols
            (if reachabilityResult.PruningStatistics.TotalSymbols > 0 then
                (float reachabilityResult.PruningStatistics.EliminatedSymbols /
                 float reachabilityResult.PruningStatistics.TotalSymbols) * 100.0
             else 0.0)

        // =========================================================================
        // PHASE 4: MLIR Generation
        // =========================================================================

        report verbose "MLIR" "Generating MLIR..."

        let mlirOutput = generateMLIRFromPSG reachabilityResult.MarkedPSG resolved.Name targetTriple resolved.OutputKind

        match intermediatesDir with
        | Some dir ->
            let mlirPath = Path.Combine(dir, resolved.Name + ".mlir")
            File.WriteAllText(mlirPath, mlirOutput)
            printfn "[MLIR] Wrote: %s" mlirPath

            // Write PSG debug info with tree structure for reachable nodes
            // Use the marked PSG (after reachability analysis) for accurate IsReachable flags
            let markedPsg = reachabilityResult.MarkedPSG
            let psgInfoPath = Path.Combine(dir, resolved.Name + ".psg.txt")
            let psgInfo = System.Text.StringBuilder()
            psgInfo.AppendLine(sprintf "PSG Summary for %s" resolved.Name) |> ignore
            psgInfo.AppendLine(sprintf "Nodes: %d" markedPsg.Nodes.Count) |> ignore
            psgInfo.AppendLine(sprintf "Edges: %d" markedPsg.Edges.Length) |> ignore
            psgInfo.AppendLine(sprintf "Entry Points: %d" markedPsg.EntryPoints.Length) |> ignore
            psgInfo.AppendLine(sprintf "Symbols: %d" markedPsg.SymbolTable.Count) |> ignore

            // Count reachable nodes
            let reachableCount = markedPsg.Nodes |> Map.filter (fun _ n -> n.IsReachable) |> Map.count
            psgInfo.AppendLine(sprintf "Reachable Nodes: %d (%.1f%%)" reachableCount (100.0 * float reachableCount / float markedPsg.Nodes.Count)) |> ignore
            psgInfo.AppendLine() |> ignore

            // Entry Points section
            psgInfo.AppendLine("Entry Points:") |> ignore
            for ep in markedPsg.EntryPoints do
                match Map.tryFind ep.Value markedPsg.Nodes with
                | Some node ->
                    let name = node.Symbol |> Option.map (fun s -> s.FullName) |> Option.defaultValue "(unknown)"
                    psgInfo.AppendLine(sprintf "  - %s (%s)" name node.SyntaxKind) |> ignore
                | None -> ()
            psgInfo.AppendLine() |> ignore

            // Tree view of reachable nodes (DuckDB-ready format)
            // This structure aligns with PGQ graph traversal
            psgInfo.AppendLine("═══════════════════════════════════════════════════════════════════") |> ignore
            psgInfo.AppendLine("Reachable Node Tree (for emission debugging)") |> ignore
            psgInfo.AppendLine("═══════════════════════════════════════════════════════════════════") |> ignore

            // Helper to get children in source order
            let getChildren (node: Core.PSG.Types.PSGNode) =
                match node.Children with
                | Core.PSG.Types.ChildrenState.Parent childIds ->
                    childIds
                    |> List.rev  // Children stored in reverse order
                    |> List.choose (fun id -> Map.tryFind id.Value markedPsg.Nodes)
                | _ -> []

            // Recursive tree printer
            let rec printTree (node: Core.PSG.Types.PSGNode) (indent: string) (isLast: bool) =
                let prefix = if isLast then "└── " else "├── "
                let symbolInfo =
                    node.Symbol
                    |> Option.map (fun s -> sprintf " [%s]" s.DisplayName)
                    |> Option.defaultValue ""
                let typeInfo =
                    node.Type
                    |> Option.map (fun t ->
                        try sprintf " : %s" (t.Format(FSharp.Compiler.Symbols.FSharpDisplayContext.Empty))
                        with _ -> "")
                    |> Option.defaultValue ""
                let reachMark = if node.IsReachable then "" else " (UNREACHABLE)"
                psgInfo.AppendLine(sprintf "%s%s%s%s%s%s" indent prefix node.SyntaxKind symbolInfo typeInfo reachMark) |> ignore

                let children = getChildren node
                let childIndent = indent + (if isLast then "    " else "│   ")
                children |> List.iteri (fun i child ->
                    let isLastChild = (i = children.Length - 1)
                    printTree child childIndent isLastChild
                )

            // Print tree for each entry point and reachable function
            for ep in markedPsg.EntryPoints do
                match Map.tryFind ep.Value markedPsg.Nodes with
                | Some node ->
                    psgInfo.AppendLine() |> ignore
                    let name = node.Symbol |> Option.map (fun s -> s.FullName) |> Option.defaultValue ep.Value
                    psgInfo.AppendLine(sprintf "ENTRY: %s" name) |> ignore
                    printTree node "" true
                | None -> ()

            // Also print other reachable functions (not entry points)
            let entryIds = markedPsg.EntryPoints |> List.map (fun e -> e.Value) |> Set.ofList
            let reachableFunctions =
                markedPsg.Nodes
                |> Map.toList
                |> List.filter (fun (id, node) ->
                    node.IsReachable &&
                    not (Set.contains id entryIds) &&
                    (node.SyntaxKind = "Binding" || node.SyntaxKind.StartsWith("LetBinding")))
                |> List.filter (fun (_, node) ->
                    match node.Symbol with
                    | Some (:? FSharp.Compiler.Symbols.FSharpMemberOrFunctionOrValue as mfv) ->
                        mfv.IsFunction || mfv.IsMember
                    | _ -> false)

            for (_, funcNode) in reachableFunctions do
                psgInfo.AppendLine() |> ignore
                let name = funcNode.Symbol |> Option.map (fun s -> s.FullName) |> Option.defaultValue "(function)"
                psgInfo.AppendLine(sprintf "FUNCTION: %s" name) |> ignore
                printTree funcNode "" true

            psgInfo.AppendLine() |> ignore
            psgInfo.AppendLine("═══════════════════════════════════════════════════════════════════") |> ignore
            psgInfo.AppendLine("Symbol Table") |> ignore
            psgInfo.AppendLine("═══════════════════════════════════════════════════════════════════") |> ignore
            for kvp in markedPsg.SymbolTable do
                psgInfo.AppendLine(sprintf "  %s: %s" kvp.Key kvp.Value.FullName) |> ignore
            File.WriteAllText(psgInfoPath, psgInfo.ToString())
            printfn "[PSG] Wrote debug info: %s" psgInfoPath

            if emitMLIR then
                printfn ""
                printfn "Stopped after MLIR generation (--emit-mlir)"
                0
            else
                // =========================================================================
                // PHASE 5: LLVM IR Generation
                // =========================================================================

                report verbose "LLVM" "Lowering to LLVM IR..."

                let llvmOutput = generateLLVMFromPSG reachabilityResult.MarkedPSG resolved.Name targetTriple

                let llPath = Path.Combine(dir, resolved.Name + ".ll")
                File.WriteAllText(llPath, llvmOutput)
                printfn "[LLVM] Wrote: %s" llPath

                if emitLLVM then
                    printfn ""
                    printfn "Stopped after LLVM IR generation (--emit-llvm)"
                    0
                else
                    // =========================================================================
                    // PHASE 6: Native Code Generation
                    // =========================================================================

                    report verbose "LINK" "Compiling to native executable..."

                    printfn ""
                    printfn "NOTE: Full native code generation not yet implemented."
                    printfn "The MLIR and LLVM IR files have been generated in:"
                    printfn "  %s" dir
                    printfn ""
                    printfn "To manually compile (on Linux):"
                    printfn "  llc -filetype=obj %s -o %s.o" llPath (Path.GetFileNameWithoutExtension(llPath))
                    printfn "  clang %s.o -o %s" (Path.GetFileNameWithoutExtension(llPath)) outputPath
                    0

        | None ->
            // No intermediates requested, but we still did the analysis
            printfn ""
            printfn "Compilation analysis complete."
            printfn "Use -k or --emit-mlir to see generated code."
            0
