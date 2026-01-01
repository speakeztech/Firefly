/// CompilationOrchestrator - THE single orchestrator for Firefly compilation
///
/// FNCS-based pipeline (December 2025 rewrite):
///   1. Load project (.fidproj)
///   2. Parse and type-check with FNCS
///   3. Generate MLIR via witness-based emission
///   4. Lower MLIR → LLVM (Toolchain)
///   5. Link LLVM → Native (Toolchain)
module Alex.Pipeline.CompilationOrchestrator

open System
open System.IO
open Core.CompilerConfig
open Core.Toolchain
open Alex.Traversal.MLIRZipper
open Alex.Bindings.BindingTypes
open Core.FNCS.Integration
open Core.FNCS.ProjectLoader
open Alex.Generation.FNCSEmitter
// Import only the specific type we need, avoid shadowing Result.Error
type FNCSDiagnosticSeverity = FSharp.Native.Compiler.Checking.Native.SemanticGraph.DiagnosticSeverity

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

/// Result of MLIR generation
type MLIRGenerationResult = {
    Content: string
    HasErrors: bool
    Errors: string list
    CollectedFunctions: string list
}

// ═══════════════════════════════════════════════════════════════════════════
// Minimal TOML Parser for .fidproj
// ═══════════════════════════════════════════════════════════════════════════

/// Parse a minimal .fidproj file
type FidprojConfig = {
    Name: string
    Sources: string list
    AlloyPath: string option
    OutputKind: Core.Types.MLIRTypes.OutputKind
    OutputName: string
    BuildDir: string
}

/// Parse .fidproj file (minimal TOML parser)
let parseFidproj (path: string) : Result<FidprojConfig, string> =
    try
        let lines = File.ReadAllLines(path)
        let _projectDir = Path.GetDirectoryName(path)

        let mutable name = Path.GetFileNameWithoutExtension(path)
        let mutable sources = []
        let mutable alloyPath: string option = None
        let mutable outputKind = Core.Types.MLIRTypes.OutputKind.Freestanding
        let mutable outputName = name
        let mutable buildDir = "target"
        let mutable inSourcesArray = false
        let mutable sourcesAccum = ""

        for line in lines do
            let line = line.Trim()
            // Skip comments and empty lines
            if line.StartsWith("#") || line = "" then
                ()
            // Handle multi-line arrays for sources
            elif inSourcesArray then
                sourcesAccum <- sourcesAccum + " " + line
                if line.Contains("]") then
                    inSourcesArray <- false
                    // Parse accumulated sources
                    let sourceStr = sourcesAccum.Trim('[', ']', ' ')
                    sources <- sourceStr.Split(',')
                              |> Array.map (fun s -> s.Trim().Trim('"'))
                              |> Array.filter (fun s -> s <> "")
                              |> Array.toList
            elif line.StartsWith("name") then
                let parts = line.Split('=')
                if parts.Length > 1 then
                    name <- parts.[1].Trim().Trim('"')
            elif line.StartsWith("output_kind") then
                let parts = line.Split('=')
                if parts.Length > 1 then
                    outputKind <- Core.Types.MLIRTypes.OutputKind.parse (parts.[1].Trim().Trim('"'))
            elif line.StartsWith("output") && not (line.StartsWith("output_kind")) then
                let parts = line.Split('=')
                if parts.Length > 1 then
                    outputName <- parts.[1].Trim().Trim('"')
            elif line.StartsWith("sources") then
                let parts = line.Split('=')
                if parts.Length > 1 then
                    let sourceStr = parts.[1].Trim()
                    if sourceStr.Contains("[") && sourceStr.Contains("]") then
                        // Single-line array
                        let sourceInner = sourceStr.Trim('[', ']', ' ')
                        sources <- sourceInner.Split(',')
                                  |> Array.map (fun s -> s.Trim().Trim('"'))
                                  |> Array.filter (fun s -> s <> "")
                                  |> Array.toList
                    elif sourceStr.Contains("[") then
                        // Start of multi-line array
                        inSourcesArray <- true
                        sourcesAccum <- sourceStr
            elif line.StartsWith("alloy") then
                let parts = line.Split('=')
                if parts.Length > 1 then
                    let pathStr = parts.[1].Trim()
                    if pathStr.Contains("path") then
                        let innerParts = pathStr.Split('"')
                        if innerParts.Length > 1 then
                            alloyPath <- Some innerParts.[1]
            elif line.StartsWith("build_dir") then
                let parts = line.Split('=')
                if parts.Length > 1 then
                    buildDir <- parts.[1].Trim().Trim('"')

        Ok {
            Name = name
            Sources = sources
            AlloyPath = alloyPath
            OutputKind = outputKind
            OutputName = outputName
            BuildDir = buildDir
        }
    with ex ->
        Error (sprintf "Failed to parse fidproj: %s" ex.Message)

// ═══════════════════════════════════════════════════════════════════════════
// Placeholder MLIR Generation
// ═══════════════════════════════════════════════════════════════════════════

/// Determine target platform from triple
let parsePlatform (triple: string) : TargetPlatform =
    // Use parseTriple for proper parsing, fall back to defaults
    match TargetPlatform.parseTriple triple with
    | Some platform -> platform
    | None ->
        // Fallback heuristics
        if triple.Contains("linux") then
            if triple.Contains("aarch64") || triple.Contains("arm64") then
                { OS = Linux; Arch = ARM64; Triple = triple; Features = Set.empty }
            else
                TargetPlatform.linux_x86_64
        elif triple.Contains("darwin") || triple.Contains("macos") then
            if triple.Contains("x86_64") then
                TargetPlatform.macos_x86_64
            else
                TargetPlatform.macos_arm64
        elif triple.Contains("windows") then
            TargetPlatform.windows_x86_64
        else
            TargetPlatform.linux_x86_64 // Default to Linux

/// Generate MLIR from a project using FNCS SemanticGraph
let generateMLIRFromFNCS (config: FidprojConfig) (projectDir: string) (targetTriple: string) : MLIRGenerationResult =
    // Resolve source paths
    let sourcePaths =
        config.Sources
        |> List.map (fun src ->
            if Path.IsPathRooted(src) then src
            else Path.Combine(projectDir, src))

    printfn "[FNCS] Loading %d source file(s)..." sourcePaths.Length

    // Create project config for FNCS
    let projectConfig : ProjectConfig = {
        Name = config.Name
        Sources = sourcePaths
        AlloyPath = config.AlloyPath
        OutputKind = match config.OutputKind with
                     | Core.Types.MLIRTypes.OutputKind.Freestanding -> "freestanding"
                     | Core.Types.MLIRTypes.OutputKind.Console -> "console"
        TargetTriple = Some targetTriple
    }

    // Parse and check with FNCS
    let projectResult = loadAndCheck projectConfig

    // Check for errors
    let errors =
        projectResult.CheckResult.Diagnostics
        |> List.filter (fun d -> d.Severity = FNCSDiagnosticSeverity.Error)
        |> List.map (fun d -> d.Message)

    if not (List.isEmpty errors) then
        printfn "[FNCS] Type checking found %d error(s)" errors.Length
        for err in errors do
            printfn "[FNCS] ERROR: %s" err
        {
            Content = ""
            HasErrors = true
            Errors = errors
            CollectedFunctions = []
        }
    else
        let graph = projectResult.CheckResult.Graph
        let nodeCount = Map.count graph.Nodes
        let entryCount = List.length graph.EntryPoints

        printfn "[FNCS] SemanticGraph: %d nodes, %d entry points" nodeCount entryCount

        // Debug: Print node details
        if nodeCount <= 30 then
            printfn "[FNCS] Node details:"
            for kvp in graph.Nodes do
                let id = nodeIdToInt kvp.Key
                let node = kvp.Value
                let kindStr =
                    match node.Kind with
                    | FSharp.Native.Compiler.Checking.Native.SemanticGraph.SemanticKind.ModuleDef(name, children) ->
                        sprintf "ModuleDef(%s, children=%d)" name (List.length children)
                    | FSharp.Native.Compiler.Checking.Native.SemanticGraph.SemanticKind.Binding(name, isMut, isRec) ->
                        sprintf "Binding(%s, mutable=%b, rec=%b)" name isMut isRec
                    | FSharp.Native.Compiler.Checking.Native.SemanticGraph.SemanticKind.Literal v ->
                        sprintf "Literal(%A)" v
                    | FSharp.Native.Compiler.Checking.Native.SemanticGraph.SemanticKind.Application(func, args) ->
                        sprintf "Application(func=%d, args=%d)" (nodeIdToInt func) (List.length args)
                    | FSharp.Native.Compiler.Checking.Native.SemanticGraph.SemanticKind.VarRef(name, _) ->
                        sprintf "VarRef(%s)" name
                    | FSharp.Native.Compiler.Checking.Native.SemanticGraph.SemanticKind.Lambda(params', _) ->
                        sprintf "Lambda(params=%d)" (List.length params')
                    | FSharp.Native.Compiler.Checking.Native.SemanticGraph.SemanticKind.Sequential nodes ->
                        sprintf "Sequential(count=%d)" (List.length nodes)
                    | k -> sprintf "%A" k
                let childIds = node.Children |> List.map nodeIdToInt
                printfn "  [%d] %s (children: %A)" id kindStr childIds

        // Determine target platform
        let platform = parsePlatform targetTriple

        // Generate MLIR using FNCSEmitter
        let emissionResult = generateMLIRWithMain graph platform "main"

        printfn "[FNCS] Generated MLIR with %d function(s)" emissionResult.EmittedFunctions.Length

        if not (List.isEmpty emissionResult.Errors) then
            printfn "[FNCS] Emission warnings:"
            for err in emissionResult.Errors do
                printfn "[FNCS]   %s" err

        {
            Content = emissionResult.MLIRContent
            HasErrors = false  // Emission warnings are not fatal for now
            Errors = emissionResult.Errors
            CollectedFunctions = emissionResult.EmittedFunctions
        }

/// Generate MLIR from a project - fallback to placeholder if FNCS fails
let generateMLIR (config: FidprojConfig) (projectDir: string) (targetTriple: string) : MLIRGenerationResult =
    try
        generateMLIRFromFNCS config projectDir targetTriple
    with ex ->
        printfn "[FNCS] Exception during FNCS processing: %s" ex.Message
        printfn "[FNCS] Falling back to placeholder..."

        // Fallback to minimal placeholder
        let mainContent = """module {
  // Fallback - FNCS processing failed
  llvm.func @main() -> i32 attributes {sym_visibility = "public"} {
    %exit_code = arith.constant 0 : i64
    %syscall_num = arith.constant 60 : i64
    %result = llvm.inline_asm has_side_effects "syscall", "={rax},{rax},{rdi},~{rcx},~{r11},~{memory}" %syscall_num, %exit_code : (i64, i64) -> i64
    llvm.unreachable
  }
}
"""
        {
            Content = mainContent
            HasErrors = true
            Errors = [sprintf "FNCS exception: %s" ex.Message]
            CollectedFunctions = ["main"]
        }

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

    printfn "Firefly Compiler v0.5.0 (FNCS)"
    printfn "=============================="
    printfn ""

    // Step 1: Load project
    let configResult = parseFidproj options.ProjectPath

    match configResult with
    | Error msg ->
        printfn "Error: %s" msg
        1

    | Ok config ->
        let projectDir = Path.GetDirectoryName(options.ProjectPath)

        let targetTriple =
            options.TargetTriple
            |> Option.defaultValue (getDefaultTarget())

        // Build directory
        let buildDir = Path.Combine(projectDir, config.BuildDir)
        Directory.CreateDirectory(buildDir) |> ignore

        let outputPath =
            options.OutputPath
            |> Option.defaultValue (Path.Combine(buildDir, config.OutputName))

        printfn "Project: %s" config.Name
        printfn "Sources: %d files" config.Sources.Length
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

        // Step 2: Generate MLIR via FNCS
        let mlirResult =
            Core.Timing.timePhase "MLIR" "MLIR Generation" (fun () ->
                generateMLIR config projectDir targetTriple)

        printfn "[MLIR] Collected %d functions:" mlirResult.CollectedFunctions.Length
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
            let mlirPath = Path.Combine(dir, config.Name + ".mlir")
            File.WriteAllText(mlirPath, mlirResult.Content)
            printfn "[MLIR] Wrote: %s" mlirPath

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
                // Step 3: Lower MLIR → LLVM (via Toolchain)
                let llPath = Path.Combine(dir, config.Name + ".ll")

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
                        // Step 4: Link LLVM → Native (via Toolchain)
                        let linkResult =
                            Core.Timing.timePhase "LINK" "Compiling to native binary" (fun () ->
                                compileLLVMToNative llPath outputPath targetTriple config.OutputKind)

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

                let mlirPath = Path.Combine(tempDir, config.Name + ".mlir")
                let llPath = Path.Combine(tempDir, config.Name + ".ll")

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
                            compileLLVMToNative llPath outputPath targetTriple config.OutputKind)

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
