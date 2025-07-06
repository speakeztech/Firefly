module CLI.ProgramEnhanced

open System
open System.IO
open Argu
open Core.XParsec.Foundation
open Dabbit.Pipeline.CompilationOrchestrator
open Dabbit.Pipeline.CompilationTypes

// ===================================================================
// Command Line Interface Definition
// ===================================================================

/// Command line arguments for Firefly compiler
type FireflyArgs =
    | Project_File of path: string
    | Output of dir: string
    | Intermediates
    | Verbose
    | AST_Only
    | Show_Stats
    | Show_Elimination_Stats

    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Project_File _ -> "F# project file (.fsproj) or script (.fsx) to compile"
            | Output _ -> "Output directory for compiled binary and intermediates"
            | Intermediates -> "Generate intermediate files for debugging (.fcs, .ra.fcs, .mlir, .ll)"
            | Verbose -> "Enable verbose logging with timestamps"
            | AST_Only -> "Run AST analysis only (no MLIR/LLVM generation)"
            | Show_Stats -> "Show detailed compilation statistics"
            | Show_Elimination_Stats -> "Show detailed dead code elimination statistics"


// ===================================================================
// Progress Reporting
// ===================================================================

/// Progress reporting with colored output
let reportProgress (verbose: bool) (phase: CompilationPhase) (message: string) : unit =
    if verbose then
        let timestamp = DateTime.UtcNow.ToString("HH:mm:ss.fff")
        printfn "[%s] [%A] %s" timestamp phase message
    else
        printf "."

// ===================================================================
// Statistics Display
// ===================================================================

/// Display detailed elimination statistics
let displayEliminationStats (result: CompilationResult) : unit =
    if result.Success then
        Console.ForegroundColor <- ConsoleColor.Yellow
        printfn "=== DEAD CODE ELIMINATION ANALYSIS ==="
        Console.ResetColor()
        
        let stats = result.Statistics
        if stats.EliminatedSymbols > 0 then
            printfn "🗑️  Successfully eliminated %d unused functions" stats.EliminatedSymbols
            printfn "💾 Code size reduction: ~%.1f%%" ((float stats.EliminatedSymbols / float stats.TotalSymbols) * 100.0)
        else
            printfn "ℹ️  No dead code found - all functions are reachable"
        printfn ""

/// Display reachability analysis statistics
let displayStatistics (stats: CompilationStatistics) : unit =
    printfn ""
    Console.ForegroundColor <- ConsoleColor.Green
    printfn "=== FIREFLY COMPILATION STATISTICS ==="
    Console.ResetColor()
    
    printfn "📁 Files processed: %d" stats.TotalFiles
    printfn "🔧 Total symbols: %d" stats.TotalSymbols
    printfn "✅ Reachable symbols: %d" stats.ReachableSymbols
    printfn "❌ Eliminated symbols: %d" stats.EliminatedSymbols
    
    if stats.TotalSymbols > 0 then
        let eliminationRate = (float stats.EliminatedSymbols / float stats.TotalSymbols) * 100.0
        printfn "📊 Elimination rate: %.1f%%" eliminationRate
    
    printfn "⏱️  Compilation time: %.2f ms" stats.CompilationTimeMs
    printfn ""

/// Display errors with formatting
let displayErrors (errors: FireflyError list) : unit =
    Console.ForegroundColor <- ConsoleColor.Red
    printfn "=== COMPILATION ERRORS ==="
    Console.ResetColor()
    
    for error in errors do
        match error with
        | SyntaxError(pos, message, context) ->
            printfn "❌ Syntax Error at %s(%d,%d): %s" pos.File pos.Line pos.Column message
            for ctx in context do
                printfn "   Context: %s" ctx
        
        | TypeCheckError(construct, message, location) ->
            printfn "❌ Type Error in %s at %s(%d,%d): %s" construct location.File location.Line location.Column message
        
        | ConversionError(phase, source, target, message) ->
            printfn "❌ Conversion Error in %s (%s → %s): %s" phase source target message
        
        | InternalError(phase, message, details) ->
            printfn "❌ Internal Error in %s: %s" phase message
            match details with
            | Some detail -> printfn "   Details: %s" detail
            | None -> ()
        
        | ParseError(pos, message) ->
            printfn "❌ Parse Error at %s(%d,%d): %s" pos.File pos.Line pos.Column message
        
        | DependencyResolutionError(symbol, message) ->
            printfn "❌ Dependency Error for %s: %s" symbol message
    
    printfn ""

// ===================================================================
// Main Entry Point
// ===================================================================

[<EntryPoint>]
let main (args: string[]) : int =
    let parser = ArgumentParser.Create<FireflyArgs>(programName = "firefly")
    
    try
        let results = parser.Parse(args)
        
        // Get project file - handle case where it might not be provided
        let projectFileOpt = results.TryGetResult(Project_File)
        match projectFileOpt with
        | None ->
            Console.ForegroundColor <- ConsoleColor.Red
            printfn "❌ Error: Project file is required."
            printfn "Usage: firefly <project-file> [options]"
            Console.ResetColor()
            1
        | Some projectFile ->
            let outputDir = results.GetResult(Output, defaultValue = "./output")
            let generateIntermediates = results.Contains(Intermediates)
            let verbose = results.Contains(Verbose)
            let astOnly = results.Contains(AST_Only)
            let showStats = results.Contains(Show_Stats)
            let showEliminationStats = results.Contains(Show_Elimination_Stats)
            
            // Validate input file
            if not (File.Exists(projectFile)) then
                Console.ForegroundColor <- ConsoleColor.Red
                printfn "❌ Project file not found: %s" projectFile
                Console.ResetColor()
                1
            else
                // Create output directory
                if not (Directory.Exists(outputDir)) then
                    Directory.CreateDirectory(outputDir) |> ignore
                
                // Setup intermediate files directory
                let intermediatesDir = 
                    if generateIntermediates then 
                        let dir = Path.Combine(outputDir, "intermediates")
                        if not (Directory.Exists(dir)) then
                            Directory.CreateDirectory(dir) |> ignore
                        Some dir
                    else 
                        None
                
                // Create progress reporter
                let progress = reportProgress verbose
                
                // Display header
                Console.ForegroundColor <- ConsoleColor.Cyan
                printfn "🔥 Firefly F# Compiler v0.3.022"
                Console.ResetColor()
                printfn "📂 Input: %s" (Path.GetFileName(projectFile))
                printfn "📁 Output: %s" outputDir
                if generateIntermediates then
                    printfn "🔍 Intermediates: Enabled"
                if astOnly then
                    printfn "🎯 Mode: AST Analysis Only"
                printfn ""
                
                if not verbose then
                    printf "Progress: "
                
                // Run compilation pipeline
                let compilationResult = 
                    compile projectFile intermediatesDir progress
                    |> Async.RunSynchronously
                
                if not verbose then
                    printfn ""
                
                // Display results
                match compilationResult.Success with
                | true ->
                    Console.ForegroundColor <- ConsoleColor.Green
                    printfn "✅ Compilation completed successfully!"
                    Console.ResetColor()
                    
                    if showStats then
                        displayStatistics compilationResult.Statistics
                    
                    if showEliminationStats then
                        displayEliminationStats compilationResult
                    
                    if generateIntermediates then
                        printfn "📄 Intermediate files written to: %s" (Option.get intermediatesDir)
                    
                    0
                
                | false ->
                    Console.ForegroundColor <- ConsoleColor.Red
                    printfn "❌ Compilation failed!"
                    Console.ResetColor()
                    
                    displayErrors compilationResult.Diagnostics
                    1
    
    with
    | :? ArguParseException as ex ->
        printfn "%s" ex.Message
        1
    | ex ->
        Console.ForegroundColor <- ConsoleColor.Red
        printfn "❌ Unexpected error: %s" ex.Message
        Console.ResetColor()
        1