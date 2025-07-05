// CLI/Program-Enhanced.fs
module CLI.ProgramEnhanced

open System
open System.IO
open Argu
open Core.XParsec.Foundation
open Dabbit.Pipeline.CompilationOrchestrator
open Dabbit.Pipeline.ReachabilityIntegration

/// Command line arguments for Firefly compiler
type FireflyArgs =
    | [<MainCommand; Last>] Project_File of string
    | [<AltCommandLine("-o")>] Output of string
    | [<AltCommandLine("-i")>] Intermediates
    | [<AltCommandLine("-v")>] Verbose
    | [<AltCommandLine("--ast-only")>] AST_Only
    | [<AltCommandLine("--show-stats")>] Show_Stats
    | [<AltCommandLine("--show-elimination")>] Show_Elimination_Stats

    interface IArgParserTemplate with
        member s.Usage =
            match s with
            | Project_File _ -> "F# project file (.fsproj) to compile"
            | Output _ -> "Output directory for compiled binary"
            | Intermediates -> "Generate intermediate files for debugging"
            | Verbose -> "Enable verbose logging"
            | AST_Only -> "Run AST analysis only (no compilation)"
            | Show_Stats -> "Show compilation statistics"
            | Show_Elimination_Stats -> "Show detailed elimination statistics"

/// Enhanced progress reporting
let reportProgress verbose phase message =
    let phaseColor = 
        match phase with
        | ProjectLoading -> ConsoleColor.Cyan
        | FCSProcessing -> ConsoleColor.Green
        | SymbolCollection -> ConsoleColor.Yellow
        | ReachabilityAnalysis -> ConsoleColor.Magenta
        | IntermediateGeneration -> ConsoleColor.Blue
        | ASTTransformation -> ConsoleColor.DarkGreen
        | MLIRGeneration -> ConsoleColor.DarkMagenta
        | LLVMGeneration -> ConsoleColor.DarkRed
        | NativeCompilation -> ConsoleColor.Red

    let timestamp = DateTime.UtcNow.ToString "HH:mm:ss.fff"
    
    Console.ForegroundColor <- ConsoleColor.Gray
    printf "[%s] " timestamp
    
    Console.ForegroundColor <- phaseColor
    printf "[%A] " phase
    
    Console.ForegroundColor <- ConsoleColor.White
    printfn "%s" message
    
    Console.ResetColor()

/// Display compilation statistics
let displayStatistics stats =
    printfn ""
    Console.ForegroundColor <- ConsoleColor.Green
    printfn "=== COMPILATION STATISTICS ==="
    Console.ResetColor()
    
    printfn "Files processed: %d" stats.TotalFiles
    printfn "Total symbols: %d" stats.TotalSymbols
    printfn "Reachable symbols: %d" stats.ReachableSymbols
    printfn "Eliminated symbols: %d" stats.EliminatedSymbols
    
    let eliminationRate = 
        if stats.TotalSymbols > 0 then
            float stats.EliminatedSymbols / float stats.TotalSymbols * 100.0
        else 0.0
    
    Console.ForegroundColor <- ConsoleColor.Yellow
    printfn "Elimination rate: %.1f%%" eliminationRate
    Console.ResetColor()
    
    printfn "Compilation time: %.0fms" stats.CompilationTimeMs

/// Display elimination statistics in detail
let displayEliminationStats projectFile =
    let intermediatesDir = 
        Path.Combine(Path.GetDirectoryName(projectFile: string), "build", "intermediates")
    
    let reportPath = Path.Combine(intermediatesDir, "reachability.report")
    
    if File.Exists reportPath then
        printfn ""
        Console.ForegroundColor <- ConsoleColor.Cyan
        printfn "=== DETAILED ELIMINATION REPORT ==="
        Console.ResetColor()
        
        let reportContent = File.ReadAllText reportPath
        printfn "%s" reportContent
    else
        Console.ForegroundColor <- ConsoleColor.Red
        printfn "No elimination report found. Run with --intermediates first."
        Console.ResetColor()

/// Run AST reachability analysis only
let runASTAnalysisOnly projectFile verbose =
    try
        printfn "Running AST reachability analysis for: %s" (Path.GetFileName(projectFile: string))
        
        let projectDir = Path.GetDirectoryName(projectFile: string)
        let progress = reportProgress verbose
        
        match CompilationOrchestratorIntegration.enhanceCompilationPipeline projectDir progress with
        | Success stats ->
            printfn ""
            Console.ForegroundColor <- ConsoleColor.Green
            printfn "✓ AST Analysis Complete"
            Console.ResetColor()
            // Convert ReachabilityStatistics to CompilationStatistics format for display
            let compilationStats = {
                TotalFiles = 1  // Placeholder
                TotalSymbols = stats.TotalSymbols
                ReachableSymbols = stats.ReachableSymbols  
                EliminatedSymbols = stats.EliminatedSymbols
                CompilationTimeMs = 0.0  // Placeholder
            }
            displayStatistics compilationStats
            0
        | CompilerFailure errors ->
            Console.ForegroundColor <- ConsoleColor.Red
            printfn "✗ AST Analysis Failed"
            Console.ResetColor()
            for error in errors do
                printfn "Error: %A" error
            1
    with
    | ex ->
        Console.ForegroundColor <- ConsoleColor.Red
        printfn "✗ Analysis failed: %s" ex.Message
        Console.ResetColor()
        if verbose then
            printfn "Stack trace: %s" ex.StackTrace
        1

/// Run full compilation pipeline
let runFullCompilation projectFile outputDir generateIntermediates verbose =
    try
        printfn "Compiling: %s" (Path.GetFileName(projectFile: string))
        
        // TODO: This will use the existing CompilationOrchestrator
        // For now, just run AST analysis
        let result = runASTAnalysisOnly projectFile verbose
        
        if result = 0 then
            printfn ""
            Console.ForegroundColor <- ConsoleColor.Green
            printfn "✓ Compilation pipeline ready"
            Console.ForegroundColor <- ConsoleColor.Yellow
            printfn "Note: Full compilation (AST → MLIR → LLVM → Binary) coming in next release"
            Console.ResetColor()
        
        result
    with
    | ex ->
        Console.ForegroundColor <- ConsoleColor.Red
        printfn "✗ Compilation failed: %s" ex.Message
        Console.ResetColor()
        if verbose then
            printfn "Stack trace: %s" ex.StackTrace
        1

/// Main entry point
[<EntryPoint>]
let main argv =
    try
        let parser = ArgumentParser.Create<FireflyArgs>(programName = "firefly")
        
        if Array.isEmpty argv then
            printfn "%s" (parser.PrintUsage())
            printfn ""
            Console.ForegroundColor <- ConsoleColor.Cyan
            printfn "Firefly F# Compiler - Enhanced with AST Reachability Analysis"
            printfn "Version 0.3.0"
            Console.ResetColor()
            printfn ""
            printfn "Examples:"
            printfn "  firefly myproject.fsproj                    # Compile project"
            printfn "  firefly myproject.fsproj --intermediates    # Generate debug files"
            printfn "  firefly myproject.fsproj --ast-only         # AST analysis only"
            printfn "  firefly myproject.fsproj --show-elimination # Show pruning stats"
            0
        else
            let results = parser.Parse argv
            
            let projectFile = results.GetResult Project_File
            let outputDir = results.TryGetResult Output
            let generateIntermediates = results.Contains Intermediates
            let verbose = results.Contains Verbose
            let astOnly = results.Contains AST_Only
            let showStats = results.Contains Show_Stats
            let showElimination = results.Contains Show_Elimination_Stats
            
            // Validate project file exists
            if not <| File.Exists projectFile then
                Console.ForegroundColor <- ConsoleColor.Red
                printfn "✗ Project file not found: %s" projectFile
                Console.ResetColor()
                1
            else
                // Display what we're doing
                if verbose then
                    printfn "Project file: %s" projectFile
                    printfn "Output dir: %s" (outputDir |> Option.defaultValue "default")
                    printfn "Generate intermediates: %b" generateIntermediates
                    printfn "AST only: %b" astOnly
                    printfn ""
                
                // Handle special commands
                if showElimination then
                    displayEliminationStats projectFile
                    0
                elif astOnly then
                    runASTAnalysisOnly projectFile verbose
                else
                    let result = runFullCompilation projectFile outputDir generateIntermediates verbose
                    
                    if showStats && result = 0 then
                        // Stats already displayed in compilation, just add a separator
                        printfn ""
                    
                    result
    with
    | :? ArguParseException as ex ->
        Console.ForegroundColor <- ConsoleColor.Red
        printfn "✗ %s" ex.Message
        Console.ResetColor()
        1
    | ex ->
        Console.ForegroundColor <- ConsoleColor.Red
        printfn "✗ Unexpected error: %s" ex.Message
        Console.ResetColor()
        1