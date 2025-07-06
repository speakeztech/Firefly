module CLI.ProgramEnhanced

open System
open System.IO
open Argu
open Core.XParsec.Foundation
open Dabbit.Pipeline.CompilationOrchestrator
open Dabbit.Pipeline.CompilationTypes

/// Command line arguments for Firefly compiler
type FireflyArgs =
    | Project_File of string
    | Output of string
    | Intermediates
    | Verbose
    | AST_Only
    | Show_Stats
    | Show_Elimination_Stats

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
let reportProgress verbose (phase: CompilationPhase) message =
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

/// Display diagnostics with color coding
let displayDiagnostics diagnostics =
    if not (List.isEmpty diagnostics) then
        printfn ""
        Console.ForegroundColor <- ConsoleColor.Red
        printfn "=== COMPILATION DIAGNOSTICS ==="
        Console.ResetColor()
        
        for error in diagnostics do
            match error with
            | SyntaxError (pos, message, context) ->
                Console.ForegroundColor <- ConsoleColor.Red
                printfn "SYNTAX ERROR in %s (%d,%d): %s" pos.File pos.Line pos.Column message
                if not (List.isEmpty context) then
                    printfn "  Context: %s" (String.concat " -> " context)
                Console.ResetColor()
            | TypeCheckError (construct, message, location) ->
                Console.ForegroundColor <- ConsoleColor.Yellow
                printfn "TYPE ERROR in %s (%d,%d): %s" location.File location.Line location.Column message
                printfn "  Construct: %s" construct
                Console.ResetColor()
            | ConversionError (phase, source, target, message) ->
                Console.ForegroundColor <- ConsoleColor.Magenta
                printfn "CONVERSION ERROR in %s: %s" phase message
                printfn "  Converting %s to %s" source target
                Console.ResetColor()
            | InternalError (phase, message, stackTrace) ->
                Console.ForegroundColor <- ConsoleColor.DarkRed
                printfn "INTERNAL ERROR in %s: %s" phase message
                match stackTrace with
                | Some stack -> printfn "Stack: %s" stack
                | None -> ()
                Console.ResetColor()
            | MLIRGenerationError (phase, message, functionName) ->
                Console.ForegroundColor <- ConsoleColor.DarkMagenta
                printfn "MLIR ERROR in %s: %s" phase message
                match functionName with
                | Some func -> printfn "  Function: %s" func
                | None -> ()
                Console.ResetColor()

/// Main entry point
let main argv =
    let parser = ArgumentParser.Create<FireflyArgs>(programName = "firefly")
    
    try
        let results = parser.Parse argv
        
        // Get project file
        match results.TryGetResult Project_File with
        | None ->
            Console.ForegroundColor <- ConsoleColor.Red
            printfn "Error: Project file is required. Use --project-file <path>"
            Console.ResetColor()
            printfn "%s" (parser.PrintUsage())
            1
        | Some projectFile ->
            
            // Validate project file exists
            if not (File.Exists projectFile) then
                Console.ForegroundColor <- ConsoleColor.Red
                printfn "Error: Project file not found: %s" projectFile
                Console.ResetColor()
                1
            else
                
                let verbose = results.Contains Verbose
                let generateIntermediates = results.Contains Intermediates
                let astOnly = results.Contains AST_Only
                let showStats = results.Contains Show_Stats
                let showElimination = results.Contains Show_Elimination_Stats
                
                // Configure progress reporting
                let progress = reportProgress verbose
                
                // Configure intermediate files directory
                let intermediatesDir = 
                    if generateIntermediates then
                        let dir = Path.Combine(Path.GetDirectoryName(projectFile), "build", "intermediates")
                        Directory.CreateDirectory(dir) |> ignore
                        Some dir
                    else None
                
                // Run compilation
                try
                    let result = 
                        if astOnly then
                            analyzeASTOnly projectFile
                        else
                            compileProject projectFile intermediatesDir progress
                        |> Async.RunSynchronously
                    
                    // Display results
                    if result.Success then
                        Console.ForegroundColor <- ConsoleColor.Green
                        if astOnly then
                            printfn "AST analysis completed successfully!"
                        else
                            printfn "Compilation completed successfully!"
                        Console.ResetColor()
                    else
                        Console.ForegroundColor <- ConsoleColor.Red
                        printfn "Compilation failed!"
                        Console.ResetColor()
                    
                    // Show diagnostics if any
                    displayDiagnostics result.Diagnostics
                    
                    // Show statistics if requested
                    if showStats then
                        displayStatistics result.Statistics
                    
                    // Show elimination report if requested
                    if showElimination then
                        displayEliminationStats projectFile
                    
                    // Show reachability report if available
                    match result.ReachabilityReport with
                    | Some report when verbose ->
                        printfn ""
                        Console.ForegroundColor <- ConsoleColor.Cyan
                        printfn "=== REACHABILITY REPORT ==="
                        Console.ResetColor()
                        printfn "%s" report
                    | _ -> ()
                    
                    if result.Success then 0 else 1
                    
                with ex ->
                    Console.ForegroundColor <- ConsoleColor.Red
                    printfn "Fatal error: %s" ex.Message
                    if verbose then
                        printfn "Stack trace: %s" ex.StackTrace
                    Console.ResetColor()
                    1
                    
    with
    | :? ArguParseException as ex ->
        Console.ForegroundColor <- ConsoleColor.Red
        printfn "%s" ex.Message
        Console.ResetColor()
        1
    | ex ->
        Console.ForegroundColor <- ConsoleColor.Red
        printfn "Fatal error: %s" ex.Message
        Console.ResetColor()
        1

/// Entry point for the application
[<EntryPoint>]
let entryPoint argv = main argv