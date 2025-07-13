module CLI.Program

open System
open System.IO
open Argu
open Core.XParsec.Foundation
open Alex.Pipeline.CompilationOrchestrator

/// Command line arguments for minimal testing
type MinimalArgs =
    | Input of path: string
    | Output of dir: string
    | Verbose
    | KeepIntermediates

    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Input _ -> "F# project file (.fsproj) or script (.fsx) to analyze"
            | Output _ -> "Output directory for intermediate files (default: ./intermediates)"
            | Verbose -> "Enable verbose progress reporting"
            | KeepIntermediates -> "Keep intermediate files (default: true in minimal mode)"

/// Progress reporter for console output
let createProgressReporter (verbose: bool) : ProgressCallback =
    fun phase message ->
        let phaseStr = 
            match phase with
            | ProjectLoading -> "📂 LOAD"
            | FCSProcessing -> "🔍 FCS "
            | SymbolCollection -> "🔗 SYMS"
            | ReachabilityAnalysis -> "🎯 REACH"
            | IntermediateGeneration -> "💾 FILES"
            | ASTTransformation -> "🔄 TRANS"
            | MLIRGeneration -> "⚙️  MLIR"
            | LLVMGeneration -> "🏗️  LLVM"
            | NativeCompilation -> "🚀 NATIVE"
        
        if verbose then
            printfn "[%s] %s" phaseStr message
        else
            printf "."

/// Main entry point for minimal testing
[<EntryPoint>]
let main args =
    let parser = ArgumentParser.Create<MinimalArgs>(programName = "firefly-minimal")
    
    try
        let results = parser.Parse(args)
        let inputPath = results.GetResult(Input)
        let outputDir = results.GetResult(Output, defaultValue = "./intermediates")
        let verbose = results.Contains(Verbose)
        let keepIntermediates = results.Contains(KeepIntermediates) || true // Always true in minimal mode
        
        if not verbose then
            printfn "Firefly Minimal - FCS Ingestion & Reachability Analysis"
            printfn "Input: %s" (Path.GetFileName inputPath)
            printf "Progress: "
        
        // Validate input file
        if not (File.Exists inputPath) then
            printfn "\n❌ Input file not found: %s" inputPath
            1
        else
            let intermediatesDir = if keepIntermediates then Some "./intermediates" else None
            let progress = createProgressReporter verbose
            
            // Run the minimal compilation pipeline
            let result = 
                compile inputPath intermediatesDir progress
                |> Async.RunSynchronously
            
            if not verbose then printfn ""  // New line after progress dots
            
            // Report results
            if result.Success then
                printfn "\n✅ Pipeline completed successfully!"
                printfn ""
                printfn "📊 Statistics:"
                printfn "   Files processed: %d" result.Statistics.TotalFiles
                printfn "   Total symbols: %d" result.Statistics.TotalSymbols
                printfn "   Reachable symbols: %d" result.Statistics.ReachableSymbols
                printfn "   Eliminated symbols: %d" result.Statistics.EliminatedSymbols
                printfn "   Processing time: %.2f ms" result.Statistics.CompilationTimeMs
                
                if result.IntermediatesGenerated then
                    printfn ""
                    printfn "💾 Intermediate files written to: %s" outputDir
                    printfn "   📄 fcs/project.initial.ast - Full typed AST from FCS"
                    printfn "   📄 analysis/symbols.analysis - Symbol collection results"
                    printfn "   📄 analysis/reachability.analysis - Reachability analysis data"
                    printfn "   📄 analysis/reachability.report - Human-readable report"
                    printfn "   📄 fcs/project.pruned.ast - Pruned AST (placeholder)"
                
                match result.ReachabilityReport with
                | Some report ->
                    if verbose then
                        printfn ""
                        printfn "📋 Reachability Report:"
                        printfn "%s" report
                | None -> ()
                
                0  // Success
            else
                printfn "\n❌ Pipeline failed with errors:"
                result.Diagnostics
                |> List.iteri (fun i error ->
                    match error with
                    | SyntaxError(pos, msg, _) ->
                        printfn "   [%d] Syntax Error at %s:%d:%d - %s" (i+1) pos.File pos.Line pos.Column msg
                    | TypeCheckError(construct, msg, pos) ->
                        printfn "   [%d] Type Error in %s at %s:%d:%d - %s" (i+1) construct pos.File pos.Line pos.Column msg
                    | ConversionError(phase, source, target, msg) ->
                        printfn "   [%d] Conversion Error in %s (%s → %s) - %s" (i+1) phase source target msg
                    | InternalError(phase, msg, details) ->
                        printfn "   [%d] Internal Error in %s - %s" (i+1) phase msg
                        match details with
                        | Some stack when verbose -> printfn "       %s" stack
                        | _ -> ()
                    | MLIRGenerationError(phase, msg, funcName) ->
                        printfn "   [%d] MLIR Error in %s - %s" (i+1) phase msg
                        match funcName with
                        | Some fn -> printfn "       Function: %s" fn
                        | None -> ())
                
                if result.IntermediatesGenerated then
                    printfn ""
                    printfn "💾 Intermediate files available for debugging: %s" outputDir
                
                1  // Failure
    
    with
    | :? ArguParseException as ex ->
        printfn "%s" ex.Message
        1
    | ex ->
        printfn "❌ Unexpected error: %s" ex.Message
        if args |> Array.contains "--verbose" then
            printfn "Stack trace: %s" ex.StackTrace
        1