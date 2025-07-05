// CLI/Program-Enhanced.fs
module CLI.ProgramEnhanced

open System
open System.IO
open Argu
open Core.XParsec.Foundation
open Dabbit.Pipeline.CompilationOrchestrator
open Dabbit.Pipeline.ASTReachabilityIntegration

/// Command line arguments for Firefly compiler
type FireflyArgs =
    | [<MainCommand; ExactlyOnce; Last>] Project_File of string
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
let reportProgress (verbose: bool) (phase: CompilationPhase) (message: string) =
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

    let timestamp = DateTime.UtcNow.ToString("HH:mm:ss.fff")
    
    Console.ForegroundColor <- ConsoleColor.Gray
    Console.Write($"[{timestamp}] ")
    
    Console.ForegroundColor <- phaseColor
    Console.Write($"[{phase}] ")
    
    Console.ForegroundColor <- ConsoleColor.White
    Console.WriteLine(message)
    
    Console.ResetColor()

/// Display compilation statistics
let displayStatistics (stats: CompilationStatistics) =
    Console.WriteLine()
    Console.ForegroundColor <- ConsoleColor.Green
    Console.WriteLine("=== COMPILATION STATISTICS ===")
    Console.ResetColor()
    
    Console.WriteLine($"Files processed: {stats.TotalFiles}")
    Console.WriteLine($"Total symbols: {stats.TotalSymbols}")
    Console.WriteLine($"Reachable symbols: {stats.ReachableSymbols}")
    Console.WriteLine($"Eliminated symbols: {stats.EliminatedSymbols}")
    
    let eliminationRate = 
        if stats.TotalSymbols > 0 then
            (float stats.EliminatedSymbols / float stats.TotalSymbols) * 100.0
        else 0.0
    
    Console.ForegroundColor <- ConsoleColor.Yellow
    Console.WriteLine($"Elimination rate: {eliminationRate:F1}%")
    Console.ResetColor()
    
    Console.WriteLine($"Compilation time: {stats.CompilationTimeMs:F0}ms")

/// Display elimination statistics in detail
let displayEliminationStats (projectFile: string) =
    let intermediatesDir = 
        Path.Combine(Path.GetDirectoryName(projectFile), "build", "intermediates")
    
    let reportPath = Path.Combine(intermediatesDir, "reachability.report")
    
    if File.Exists(reportPath) then
        Console.WriteLine()
        Console.ForegroundColor <- ConsoleColor.Cyan
        Console.WriteLine("=== DETAILED ELIMINATION REPORT ===")
        Console.ResetColor()
        
        let reportContent = File.ReadAllText(reportPath)
        Console.WriteLine(reportContent)
    else
        Console.ForegroundColor <- ConsoleColor.Red
        Console.WriteLine("No elimination report found. Run with --intermediates first.")
        Console.ResetColor()

/// Run AST reachability analysis only
let runASTAnalysisOnly (projectFile: string) (verbose: bool) =
    try
        Console.WriteLine($"Running AST reachability analysis for: {Path.GetFileName(projectFile)}")
        
        let projectDir = Path.GetDirectoryName(projectFile)
        let progress = reportProgress verbose
        
        match CompilationOrchestratorIntegration.enhanceCompilationPipeline projectDir progress with
        | Success stats ->
            Console.WriteLine()
            Console.ForegroundColor <- ConsoleColor.Green
            Console.WriteLine("✓ AST Analysis Complete")
            Console.ResetColor()
            displayStatistics stats
            0
        | CompilerFailure errors ->
            Console.ForegroundColor <- ConsoleColor.Red
            Console.WriteLine("✗ AST Analysis Failed")
            Console.ResetColor()
            for error in errors do
                Console.WriteLine($"Error: {error}")
            1
    with
    | ex ->
        Console.ForegroundColor <- ConsoleColor.Red
        Console.WriteLine($"✗ Analysis failed: {ex.Message}")
        Console.ResetColor()
        if verbose then
            Console.WriteLine($"Stack trace: {ex.StackTrace}")
        1

/// Run full compilation pipeline
let runFullCompilation (projectFile: string) (outputDir: string option) (generateIntermediates: bool) (verbose: bool) =
    try
        Console.WriteLine($"Compiling: {Path.GetFileName(projectFile)}")
        
        // TODO: This will use the existing CompilationOrchestrator
        // For now, just run AST analysis
        let result = runASTAnalysisOnly projectFile verbose
        
        if result = 0 then
            Console.WriteLine()
            Console.ForegroundColor <- ConsoleColor.Green
            Console.WriteLine("✓ Compilation pipeline ready")
            Console.ForegroundColor <- ConsoleColor.Yellow
            Console.WriteLine("Note: Full compilation (AST → MLIR → LLVM → Binary) coming in next release")
            Console.ResetColor()
        
        result
    with
    | ex ->
        Console.ForegroundColor <- ConsoleColor.Red
        Console.WriteLine($"✗ Compilation failed: {ex.Message}")
        Console.ResetColor()
        if verbose then
            Console.WriteLine($"Stack trace: {ex.StackTrace}")
        1

/// Main entry point
[<EntryPoint>]
let main argv =
    try
        let parser = ArgumentParser.Create<FireflyArgs>(programName = "firefly")
        
        if Array.isEmpty argv then
            Console.WriteLine(parser.PrintUsage())
            Console.WriteLine()
            Console.ForegroundColor <- ConsoleColor.Cyan
            Console.WriteLine("Firefly F# Compiler - Enhanced with AST Reachability Analysis")
            Console.WriteLine("Version 0.3.0")
            Console.ResetColor()
            Console.WriteLine()
            Console.WriteLine("Examples:")
            Console.WriteLine("  firefly myproject.fsproj                    # Compile project")
            Console.WriteLine("  firefly myproject.fsproj --intermediates    # Generate debug files")
            Console.WriteLine("  firefly myproject.fsproj --ast-only         # AST analysis only")
            Console.WriteLine("  firefly myproject.fsproj --show-elimination # Show pruning stats")
            0
        else
            let results = parser.Parse(argv)
            
            let projectFile = results.GetResult(Project_File)
            let outputDir = results.TryGetResult(Output)
            let generateIntermediates = results.Contains(Intermediates)
            let verbose = results.Contains(Verbose)
            let astOnly = results.Contains(AST_Only)
            let showStats = results.Contains(Show_Stats)
            let showElimination = results.Contains(Show_Elimination_Stats)
            
            // Validate project file exists
            if not (File.Exists(projectFile)) then
                Console.ForegroundColor <- ConsoleColor.Red
                Console.WriteLine($"✗ Project file not found: {projectFile}")
                Console.ResetColor()
                1
            else
                // Display what we're doing
                if verbose then
                    Console.WriteLine($"Project file: {projectFile}")
                    Console.WriteLine($"Output dir: {outputDir |> Option.defaultValue "default"}")
                    Console.WriteLine($"Generate intermediates: {generateIntermediates}")
                    Console.WriteLine($"AST only: {astOnly}")
                    Console.WriteLine()
                
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
                        Console.WriteLine()
                    
                    result
    with
    | :? ArguParseException as ex ->
        Console.ForegroundColor <- ConsoleColor.Red
        Console.WriteLine($"✗ {ex.Message}")
        Console.ResetColor()
        1
    | ex ->
        Console.ForegroundColor <- ConsoleColor.Red
        Console.WriteLine($"✗ Unexpected error: {ex.Message}")
        Console.ResetColor()
        1