/// CompilationOrchestrator - THE single orchestrator for Firefly compilation
///
/// This is the ONE pipeline. There are no alternate paths.
/// All compilation flows through this orchestrator:
///   1. Load project (FidprojLoader)
///   2. Run IngestionPipeline (PSG + nanopasses + Baker)
///   3. Generate MLIR (Alex/Transfer)
///   4. Lower MLIR → LLVM (Toolchain)
///   5. Link LLVM → Native (Toolchain)
///
/// The CLI is a thin wrapper that parses args and calls this orchestrator.
module Alex.Pipeline.CompilationOrchestrator

open System
open System.IO
open Core.IngestionPipeline
open Core.PSG.Types
open Core.CompilerConfig
open Core.FCS.ProjectContext
open Core.FCS.FidprojLoader
open Core.Toolchain
open Alex.Generation.Transfer

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/// Compilation options passed from CLI
type CompilationOptions = {
    ProjectPath: string
    OutputPath: string option
    TargetTriple: string option
    KeepIntermediates: bool
    EmitMLIROnly: bool
    EmitLLVMOnly: bool
    Verbose: bool
    ShowTiming: bool
}

// ═══════════════════════════════════════════════════════════════════════════
// Pipeline Configuration
// ═══════════════════════════════════════════════════════════════════════════

/// Create pipeline config from compilation options
let private createPipelineConfig (options: CompilationOptions) (intermediatesDir: string option) : PipelineConfig = {
    CacheStrategy = Balanced
    TemplateName = None
    CustomTemplateDir = None
    EnableCouplingAnalysis = true
    EnableMemoryOptimization = true
    OutputIntermediates = options.KeepIntermediates || options.EmitMLIROnly || options.EmitLLVMOnly
    IntermediatesDir = intermediatesDir
}

// ═══════════════════════════════════════════════════════════════════════════
// PSG Debug Output Helper
// ═══════════════════════════════════════════════════════════════════════════

/// Write PSG debug info to file
let private writePSGDebugInfo (psg: ProgramSemanticGraph) (path: string) =
    let sb = System.Text.StringBuilder()
    sb.AppendLine("PSG Summary") |> ignore
    sb.AppendLine(sprintf "Nodes: %d" psg.Nodes.Count) |> ignore
    sb.AppendLine(sprintf "Edges: %d" psg.Edges.Length) |> ignore
    sb.AppendLine(sprintf "Entry Points: %d" psg.EntryPoints.Length) |> ignore
    sb.AppendLine(sprintf "Symbols: %d" psg.SymbolTable.Count) |> ignore

    let reachableCount = psg.Nodes |> Map.filter (fun _ n -> n.IsReachable) |> Map.count
    sb.AppendLine(sprintf "Reachable Nodes: %d (%.1f%%)"
        reachableCount
        (100.0 * float reachableCount / float psg.Nodes.Count)) |> ignore

    sb.AppendLine() |> ignore
    sb.AppendLine("Entry Points:") |> ignore
    for ep in psg.EntryPoints do
        match Map.tryFind ep.Value psg.Nodes with
        | Some node ->
            let name = node.Symbol |> Option.map (fun s -> s.FullName) |> Option.defaultValue "(unknown)"
            sb.AppendLine(sprintf "  - %s (%s)" name (SyntaxKindT.toString node.Kind)) |> ignore
        | None -> ()

    File.WriteAllText(path, sb.ToString())

// ═══════════════════════════════════════════════════════════════════════════
// Main Compilation Entry Point
// ═══════════════════════════════════════════════════════════════════════════

/// Compile a project - THE single entry point for all compilation
let compileProject (options: CompilationOptions) : int =
    // Enable timing if requested
    Core.Timing.setEnabled options.ShowTiming

    // Enable verbose mode if requested
    if options.Verbose then
        enableVerboseMode()

    printfn "Firefly Compiler v0.4.164"
    printfn "========================="
    printfn ""

    // Step 1: Load project via FidprojLoader
    let loadResult =
        Core.Timing.timePhase "LOAD" "Loading project and type-checking" (fun () ->
            loadAndCheckProject options.ProjectPath |> Async.RunSynchronously)

    match loadResult with
    | Error msg ->
        printfn "Error: %s" msg
        1

    | Ok (resolved, checkResults, parseResults, _checker, _projectOptions) ->

        let targetTriple =
            options.TargetTriple
            |> Option.orElse resolved.Target
            |> Option.defaultValue (getDefaultTarget())

        // Build directory
        let buildDir = Path.Combine(resolved.ProjectDir, resolved.BuildDir)
        Directory.CreateDirectory(buildDir) |> ignore

        let outputPath =
            options.OutputPath
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
            if options.KeepIntermediates || options.EmitMLIROnly || options.EmitLLVMOnly then
                let dir = Path.Combine(buildDir, "intermediates")
                Directory.CreateDirectory(dir) |> ignore
                Some dir
            else
                None

        // Step 2: Run IngestionPipeline (THE pipeline)
        // Use runPipelineWithResults since FidprojLoader already loaded the project
        let pipelineConfig = createPipelineConfig options intermediatesDir

        let pipelineResult =
            Core.Timing.timePhase "PIPELINE" "Running ingestion pipeline" (fun () ->
                runPipelineWithResults checkResults parseResults pipelineConfig)

        if not pipelineResult.Success then
            printfn ""
            printfn "Pipeline failed:"
            for diag in pipelineResult.Diagnostics do
                printfn "  [%A] %s" diag.Severity diag.Message
            Core.Timing.printSummary()
            1
        else
            // Extract results
            let enrichedPSG = pipelineResult.ProgramSemanticGraph.Value
            let bakerResult = pipelineResult.BakerResult.Value
            let reachabilityResult = pipelineResult.ReachabilityAnalysis.Value

            // Report reachability stats
            printfn "[REACH] %d/%d symbols reachable (%.1f%% eliminated)"
                reachabilityResult.PruningStatistics.ReachableSymbols
                reachabilityResult.PruningStatistics.TotalSymbols
                (if reachabilityResult.PruningStatistics.TotalSymbols > 0 then
                    (float reachabilityResult.PruningStatistics.EliminatedSymbols /
                     float reachabilityResult.PruningStatistics.TotalSymbols) * 100.0
                 else 0.0)

            printfn "[BAKER] %d member bodies extracted, %d correlated with PSG"
                bakerResult.Statistics.MembersWithBodies
                bakerResult.Statistics.MembersCorrelatedWithPSG

            // Step 3: Generate MLIR
            let mlirResult =
                Core.Timing.timePhase "MLIR" "MLIR Generation" (fun () ->
                    generateMLIR enrichedPSG bakerResult.CorrelationState targetTriple resolved.OutputKind)

            printfn "[MLIR] Collected %d reachable functions:" mlirResult.CollectedFunctions.Length
            for funcInfo in mlirResult.CollectedFunctions do
                printfn "  - %s" funcInfo

            if mlirResult.HasErrors then
                printfn ""
                printfn "[MLIR] Emission errors detected:"
                for error in mlirResult.Errors do
                    printfn "  ERROR: %s" error
                printfn ""

            // Write MLIR if keeping intermediates
            match intermediatesDir with
            | Some dir ->
                let mlirPath = Path.Combine(dir, resolved.Name + ".mlir")
                File.WriteAllText(mlirPath, mlirResult.Content)
                printfn "[MLIR] Wrote: %s" mlirPath

                // Write PSG debug info
                let psgInfoPath = Path.Combine(dir, resolved.Name + ".psg.txt")
                writePSGDebugInfo enrichedPSG psgInfoPath
                printfn "[PSG] Wrote debug info: %s" psgInfoPath

                if options.EmitMLIROnly then
                    printfn ""
                    if mlirResult.HasErrors then
                        printfn "MLIR generation completed with errors (--emit-mlir)"
                    else
                        printfn "Stopped after MLIR generation (--emit-mlir)"
                    Core.Timing.printSummary()
                    if mlirResult.HasErrors then 1 else 0
                elif mlirResult.HasErrors then
                    printfn ""
                    printfn "Compilation failed due to emission errors."
                    Core.Timing.printSummary()
                    1
                else
                    // Step 4: Lower MLIR → LLVM (via Toolchain)
                    let llPath = Path.Combine(dir, resolved.Name + ".ll")

                    let llvmResult =
                        Core.Timing.timePhase "MLIR-LOWER" "Lowering MLIR to LLVM IR" (fun () ->
                            lowerMLIRToLLVM mlirPath llPath)

                    match llvmResult with
                    | Error msg ->
                        printfn "[LLVM] Error: %s" msg
                        Core.Timing.printSummary()
                        1
                    | Ok () ->
                        printfn "[LLVM] Wrote: %s" llPath

                        if options.EmitLLVMOnly then
                            printfn ""
                            printfn "Stopped after LLVM IR generation (--emit-llvm)"
                            Core.Timing.printSummary()
                            0
                        else
                            // Step 5: Link LLVM → Native (via Toolchain)
                            let linkResult =
                                Core.Timing.timePhase "LINK" "Compiling to native binary" (fun () ->
                                    compileLLVMToNative llPath outputPath targetTriple resolved.OutputKind)

                            match linkResult with
                            | Error msg ->
                                printfn "[LINK] Error: %s" msg
                                Core.Timing.printSummary()
                                1
                            | Ok () ->
                                printfn "[LINK] Wrote: %s" outputPath
                                printfn ""
                                printfn "Compilation successful!"
                                Core.Timing.printSummary()
                                0

            | None ->
                // No intermediates - use temp directory
                if mlirResult.HasErrors then
                    printfn ""
                    printfn "Compilation failed due to emission errors."
                    printfn "Use -k to generate intermediate files for debugging."
                    Core.Timing.printSummary()
                    1
                else
                    let tempDir = Path.Combine(Path.GetTempPath(), "firefly-" + Guid.NewGuid().ToString("N").[..7])
                    Directory.CreateDirectory(tempDir) |> ignore

                    let mlirPath = Path.Combine(tempDir, resolved.Name + ".mlir")
                    let llPath = Path.Combine(tempDir, resolved.Name + ".ll")

                    File.WriteAllText(mlirPath, mlirResult.Content)

                    let llvmResult =
                        Core.Timing.timePhase "MLIR-LOWER" "Lowering MLIR to LLVM IR" (fun () ->
                            lowerMLIRToLLVM mlirPath llPath)

                    match llvmResult with
                    | Error msg ->
                        printfn "[LLVM] Error: %s" msg
                        try Directory.Delete(tempDir, true) with _ -> ()
                        Core.Timing.printSummary()
                        1
                    | Ok () ->
                        let linkResult =
                            Core.Timing.timePhase "LINK" "Compiling to native binary" (fun () ->
                                compileLLVMToNative llPath outputPath targetTriple resolved.OutputKind)

                        try Directory.Delete(tempDir, true) with _ -> ()

                        match linkResult with
                        | Error msg ->
                            printfn "[LINK] Error: %s" msg
                            Core.Timing.printSummary()
                            1
                        | Ok () ->
                            printfn "[LINK] Wrote: %s" outputPath
                            printfn ""
                            printfn "Compilation successful!"
                            Core.Timing.printSummary()
                            0
