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
        let projectDir = Path.GetDirectoryName(path)

        let mutable name = Path.GetFileNameWithoutExtension(path)
        let mutable sources = []
        let mutable alloyPath: string option = None
        let mutable outputKind = Core.Types.MLIRTypes.OutputKind.Freestanding
        let mutable outputName = name
        let mutable buildDir = "target"

        for line in lines do
            let line = line.Trim()
            if line.StartsWith("name") then
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
                    let sourceStr = parts.[1].Trim().Trim('[', ']', ' ')
                    sources <- sourceStr.Split(',')
                              |> Array.map (fun s -> s.Trim().Trim('"'))
                              |> Array.filter (fun s -> s <> "")
                              |> Array.toList
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

/// Generate MLIR from a project (placeholder - real implementation TODO)
/// This will be replaced by proper FNCSEmitter using the SemanticGraph
let generateMLIRPlaceholder (config: FidprojConfig) (targetTriple: string) : MLIRGenerationResult =
    // For now, generate a minimal valid MLIR module
    // This is a PLACEHOLDER that just makes the build succeed
    // Real implementation will use FNCS SemanticGraph + MLIRZipper

    let zipper = MLIRZipper.create ()

    // Create a minimal "main" function that exits properly
    // In freestanding mode, we can't just return - we must call exit syscall
    let mainContent = """module {
  // Placeholder - FNCS integration pending
  llvm.func @main() -> i32 attributes {sym_visibility = "public"} {
    // Exit code 0
    %exit_code = arith.constant 0 : i64
    // Syscall 60 = exit on Linux x86-64
    %syscall_num = arith.constant 60 : i64
    %result = llvm.inline_asm has_side_effects "syscall", "={rax},{rax},{rdi},~{rcx},~{r11},~{memory}" %syscall_num, %exit_code : (i64, i64) -> i64
    llvm.unreachable
  }
}
"""

    {
        Content = mainContent
        HasErrors = false
        Errors = []
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

        // Step 2: Generate MLIR
        let mlirResult =
            Core.Timing.timePhase "MLIR" "MLIR Generation" (fun () ->
                generateMLIRPlaceholder config targetTriple)

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
