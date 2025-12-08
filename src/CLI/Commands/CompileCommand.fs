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
/// Returns the MLIR content and any emission errors
let private generateMLIRFromPSG (psg: Core.PSG.Types.ProgramSemanticGraph) (projectName: string) (targetTriple: string) (_outputKind: Core.Types.MLIRTypes.OutputKind) : Alex.Pipeline.CompilationOrchestrator.MLIRGenerationResult =
    // Use Alex.Pipeline.CompilationOrchestrator.generateMLIRViaAlex
    generateMLIRViaAlex psg projectName targetTriple

/// Lower MLIR to LLVM IR using mlir-opt and mlir-translate
/// This is MLIR's job - PSG is done at this point
let private lowerMLIRToLLVM (mlirPath: string) (llvmPath: string) : Result<unit, string> =
    try
        // Step 1: mlir-opt to convert to LLVM dialect
        let mlirOptArgs = sprintf "%s --convert-func-to-llvm --convert-arith-to-llvm --reconcile-unrealized-casts" mlirPath
        let mlirOptProcess = new System.Diagnostics.Process()
        mlirOptProcess.StartInfo.FileName <- "mlir-opt"
        mlirOptProcess.StartInfo.Arguments <- mlirOptArgs
        mlirOptProcess.StartInfo.UseShellExecute <- false
        mlirOptProcess.StartInfo.RedirectStandardOutput <- true
        mlirOptProcess.StartInfo.RedirectStandardError <- true
        mlirOptProcess.Start() |> ignore
        let mlirOptOutput = mlirOptProcess.StandardOutput.ReadToEnd()
        let mlirOptError = mlirOptProcess.StandardError.ReadToEnd()
        mlirOptProcess.WaitForExit()

        if mlirOptProcess.ExitCode <> 0 then
            Error (sprintf "mlir-opt failed: %s" mlirOptError)
        else
            // Step 2: mlir-translate to convert LLVM dialect to LLVM IR
            let mlirTranslateProcess = new System.Diagnostics.Process()
            mlirTranslateProcess.StartInfo.FileName <- "mlir-translate"
            mlirTranslateProcess.StartInfo.Arguments <- "--mlir-to-llvmir"
            mlirTranslateProcess.StartInfo.UseShellExecute <- false
            mlirTranslateProcess.StartInfo.RedirectStandardInput <- true
            mlirTranslateProcess.StartInfo.RedirectStandardOutput <- true
            mlirTranslateProcess.StartInfo.RedirectStandardError <- true
            mlirTranslateProcess.Start() |> ignore
            mlirTranslateProcess.StandardInput.Write(mlirOptOutput)
            mlirTranslateProcess.StandardInput.Close()
            let llvmOutput = mlirTranslateProcess.StandardOutput.ReadToEnd()
            let translateError = mlirTranslateProcess.StandardError.ReadToEnd()
            mlirTranslateProcess.WaitForExit()

            if mlirTranslateProcess.ExitCode <> 0 then
                Error (sprintf "mlir-translate failed: %s" translateError)
            else
                File.WriteAllText(llvmPath, llvmOutput)
                Ok ()
    with ex ->
        Error (sprintf "MLIR lowering failed: %s" ex.Message)

/// Compile LLVM IR to native binary using llc and clang
/// This is LLVM's job
let private compileLLVMToNative (llvmPath: string) (outputPath: string) (targetTriple: string) : Result<unit, string> =
    try
        let objPath = Path.ChangeExtension(llvmPath, ".o")

        // Step 1: llc to compile LLVM IR to object file
        let llcArgs = sprintf "-filetype=obj %s -o %s" llvmPath objPath
        let llcProcess = new System.Diagnostics.Process()
        llcProcess.StartInfo.FileName <- "llc"
        llcProcess.StartInfo.Arguments <- llcArgs
        llcProcess.StartInfo.UseShellExecute <- false
        llcProcess.StartInfo.RedirectStandardError <- true
        llcProcess.Start() |> ignore
        let llcError = llcProcess.StandardError.ReadToEnd()
        llcProcess.WaitForExit()

        if llcProcess.ExitCode <> 0 then
            Error (sprintf "llc failed: %s" llcError)
        else
            // Step 2: clang to link into executable (freestanding, no stdlib)
            let clangArgs = sprintf "%s -o %s -nostdlib -static -ffreestanding -Wl,-e,main" objPath outputPath
            let clangProcess = new System.Diagnostics.Process()
            clangProcess.StartInfo.FileName <- "clang"
            clangProcess.StartInfo.Arguments <- clangArgs
            clangProcess.StartInfo.UseShellExecute <- false
            clangProcess.StartInfo.RedirectStandardError <- true
            clangProcess.Start() |> ignore
            let clangError = clangProcess.StandardError.ReadToEnd()
            clangProcess.WaitForExit()

            if clangProcess.ExitCode <> 0 then
                Error (sprintf "clang failed: %s" clangError)
            else
                // Clean up object file
                if File.Exists(objPath) then
                    File.Delete(objPath)
                Ok ()
    with ex ->
        Error (sprintf "Native compilation failed: %s" ex.Message)

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

        // Build directory (default "target", configurable via build.build_dir)
        let buildDir = Path.Combine(resolved.ProjectDir, resolved.BuildDir)
        Directory.CreateDirectory(buildDir) |> ignore

        let outputPath =
            args.TryGetResult(Output)
            |> Option.defaultValue (Path.Combine(buildDir, resolved.OutputName))

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
                let dir = Path.Combine(buildDir, "intermediates")
                Directory.CreateDirectory(dir) |> ignore
                Some dir
            else
                None

        // =========================================================================
        // PHASE 2: Build Program Semantic Graph
        // =========================================================================

        report verbose "PSG" "Building Program Semantic Graph..."

        // Enable nanopass intermediate emission when keeping intermediates or in verbose mode
        match intermediatesDir with
        | Some dir ->
            Core.PSG.Construction.Main.emitNanopassIntermediates <- true
            Core.PSG.Construction.Main.nanopassOutputDir <- dir
        | None -> ()

        let psg = buildProgramSemanticGraph checkResults parseResults

        // Reset nanopass emission flags
        Core.PSG.Construction.Main.emitNanopassIntermediates <- false

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

        let mlirResult = generateMLIRFromPSG reachabilityResult.MarkedPSG resolved.Name targetTriple resolved.OutputKind

        // Report emission errors
        if mlirResult.HasErrors then
            printfn ""
            printfn "[MLIR] Emission errors detected:"
            for error in mlirResult.Errors do
                printfn "  ERROR: %s" error.Message
            printfn ""

        match intermediatesDir with
        | Some dir ->
            // Always write MLIR file (even with errors) for debugging
            let mlirPath = Path.Combine(dir, resolved.Name + ".mlir")
            File.WriteAllText(mlirPath, mlirResult.Content)
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

            // Helper to get children (already in source order)
            let getChildren (node: Core.PSG.Types.PSGNode) =
                match node.Children with
                | Core.PSG.Types.ChildrenState.Parent childIds ->
                    childIds
                    |> List.choose (fun id -> Map.tryFind id.Value markedPsg.Nodes)
                | _ -> []

            // Recursive tree printer with cycle detection
            let rec printTree (node: Core.PSG.Types.PSGNode) (indent: string) (isLast: bool) (visited: Set<string>) =
                let nodeId = node.Id.Value
                if Set.contains nodeId visited then
                    // Cycle detected - don't recurse
                    let prefix = if isLast then "└── " else "├── "
                    psgInfo.AppendLine(sprintf "%s%s(CYCLE: %s)" indent prefix node.SyntaxKind) |> ignore
                else
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
                    let newVisited = Set.add nodeId visited
                    children |> List.iteri (fun i child ->
                        let isLastChild = (i = children.Length - 1)
                        printTree child childIndent isLastChild newVisited
                    )

            // Print tree for each entry point and reachable function
            for ep in markedPsg.EntryPoints do
                match Map.tryFind ep.Value markedPsg.Nodes with
                | Some node ->
                    psgInfo.AppendLine() |> ignore
                    let name = node.Symbol |> Option.map (fun s -> s.FullName) |> Option.defaultValue ep.Value
                    psgInfo.AppendLine(sprintf "ENTRY: %s" name) |> ignore
                    printTree node "" true Set.empty
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
                printTree funcNode "" true Set.empty

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
                if mlirResult.HasErrors then
                    printfn "MLIR generation completed with errors (--emit-mlir)"
                    printfn "Check %s for partial output" (Path.Combine(dir, resolved.Name + ".mlir"))
                    1
                else
                    printfn "Stopped after MLIR generation (--emit-mlir)"
                    0
            elif mlirResult.HasErrors then
                // Don't continue to LLVM if MLIR had errors
                printfn ""
                printfn "Compilation failed due to emission errors."
                printfn "MLIR output written to: %s" (Path.Combine(dir, resolved.Name + ".mlir"))
                1
            else
                // =========================================================================
                // PHASE 5: LLVM IR Generation (MLIR's job)
                // =========================================================================

                report verbose "LLVM" "Lowering MLIR to LLVM IR..."

                let mlirPath = Path.Combine(dir, resolved.Name + ".mlir")
                let llPath = Path.Combine(dir, resolved.Name + ".ll")

                match lowerMLIRToLLVM mlirPath llPath with
                | Error msg ->
                    printfn "[LLVM] Error: %s" msg
                    1
                | Ok () ->
                    printfn "[LLVM] Wrote: %s" llPath

                    if emitLLVM then
                        printfn ""
                        printfn "Stopped after LLVM IR generation (--emit-llvm)"
                        0
                    else
                        // =========================================================================
                        // PHASE 6: Native Code Generation (LLVM's job)
                        // =========================================================================

                        report verbose "LINK" "Compiling to native executable..."

                        match compileLLVMToNative llPath outputPath targetTriple with
                        | Error msg ->
                            printfn "[LINK] Error: %s" msg
                            1
                        | Ok () ->
                            printfn "[LINK] Wrote: %s" outputPath
                            printfn ""
                            printfn "Compilation successful!"
                            0

        | None ->
            // No intermediates requested, but we still did the analysis
            printfn ""
            if mlirResult.HasErrors then
                printfn "Compilation analysis found errors:"
                for error in mlirResult.Errors do
                    printfn "  ERROR: %s" error.Message
                printfn ""
                printfn "Use -k to generate intermediate files for debugging."
                1
            else
                printfn "Compilation analysis complete."
                printfn "Use -k or --emit-mlir to see generated code."
                0
