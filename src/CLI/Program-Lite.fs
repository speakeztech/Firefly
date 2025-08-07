module CLI.ProgramLite

open System
open System.IO
open Argu
open Core.IngestionPipeline
open Alex.Lite.LitePipeline
open Alex.Lite.SimpleLLVMEmitter

/// Command line arguments for Firefly Lite mode
type FireflyLiteArgs =
    | [<MainCommand; Unique>] Project_File of path: string
    | Output of dir: string
    | Emit_MLIR
    | Emit_Test
    | Run_Optimizer
    | Target of triple: string
    
    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Project_File _ -> "F# project file (.fsproj) to compile"
            | Output _ -> "Output directory for MLIR files"
            | Emit_MLIR -> "Emit LLVM dialect MLIR (simplified)"
            | Emit_Test -> "Emit a test HelloWorld example"
            | Run_Optimizer -> "Run mlir-opt to generate LLVM IR (requires MLIR tools)"
            | Target _ -> "Target triple for LLVM generation"

[<EntryPoint>]
let main (args: string[]) : int =
    let parser = ArgumentParser.Create<FireflyLiteArgs>(programName = "firefly-lite")
    
    try
        let results = parser.Parse(args)
        
        // Check for test mode first
        if results.Contains(Emit_Test) then
            printfn "🔥 Firefly Lite - Test Mode"
            printfn "================================"
            
            let outputDir = results.GetResult(Output, defaultValue = "./lite_output")
            
            // Generate minimal example
            let testPath = Path.Combine(outputDir, "test_hello.mlir")
            if not (Directory.Exists(outputDir)) then
                Directory.CreateDirectory(outputDir) |> ignore
            
            generateMinimalExample testPath
            printfn "✅ Generated test file: %s" testPath
            
            // Also run the pipeline test
            testLitePipeline()
            0
            
        else
            // Regular compilation mode
            match results.TryGetResult(Project_File) with
            | None ->
                printfn "❌ Error: Project file is required (or use --emit-test for test mode)"
                printfn ""
                printfn "%s" (parser.PrintUsage())
                1
                
            | Some projectFile ->
                if not (File.Exists(projectFile)) then
                    printfn "❌ Project file not found: %s" projectFile
                    1
                else
                    printfn "🔥 Firefly Lite Compiler (Simplified LLVM Dialect)"
                    printfn "==================================================="
                    printfn "📂 Input: %s" (Path.GetFileName(projectFile))
                    
                    let outputDir = results.GetResult(Output, defaultValue = "./lite_output")
                    printfn "📁 Output: %s" outputDir
                    
                    // Step 1: Run ingestion pipeline to get PSG
                    printfn ""
                    printfn "Phase 1: Building PSG from F# source..."
                    printfn "----------------------------------------"
                    
                    let pipelineConfig = {
                        PipelineConfig.Default with
                            IntermediatesDir = Some (Path.Combine(outputDir, "intermediates"))
                            OutputIntermediates = true
                    }
                    
                    let ingestionResult = processProject projectFile pipelineConfig |> Async.RunSynchronously
                    
                    match ingestionResult with
                    | Error err ->
                        printfn "❌ PSG generation failed: %s" err
                        1
                        
                    | Ok psgResult ->
                        printfn "✅ PSG generated successfully"
                        printfn "   Nodes: %d (reachable: %d)" 
                            psgResult.PSG.Nodes.Count 
                            (psgResult.PSG.Nodes |> Map.filter (fun _ n -> n.IsReachable) |> Map.count)
                        printfn "   Symbols: %d (reachable: %d)" 
                            psgResult.Metrics.TotalSymbols 
                            psgResult.Metrics.ReachableSymbols
                        
                        // Step 2: Generate LLVM dialect MLIR
                        printfn ""
                        printfn "Phase 2: Generating LLVM dialect MLIR..."
                        printfn "----------------------------------------"
                        
                        let liteConfig = {
                            OutputDir = outputDir
                            GenerateIntermediate = false
                            RunMLIROpt = results.Contains(Run_Optimizer)
                            TargetTriple = results.TryGetResult(Target)
                        }
                        
                        let liteResult = processWithLitePipeline psgResult.PSG liteConfig
                        
                        if liteResult.Success then
                            printfn "✅ LLVM dialect MLIR generated successfully"
                            
                            match liteResult.MLIROutput with
                            | Some mlirPath ->
                                printfn "   MLIR output: %s" mlirPath
                                
                                // Show a preview
                                let content = File.ReadAllText(mlirPath)
                                let preview = 
                                    if content.Length > 500 then
                                        content.Substring(0, 500) + "\n... (truncated)"
                                    else content
                                printfn ""
                                printfn "Preview:"
                                printfn "--------"
                                printfn "%s" preview
                            | None -> ()
                            
                            match liteResult.LLVMOutput with
                            | Some llPath ->
                                printfn "   LLVM IR: %s" llPath
                            | None -> ()
                            
                            printfn ""
                            printfn "✅ Compilation completed successfully!"
                            0
                        else
                            printfn "❌ MLIR generation failed:"
                            for error in liteResult.Errors do
                                printfn "   %s" error
                            1
    
    with
    | :? ArguParseException as ex ->
        printfn "%s" ex.Message
        1
    | ex ->
        printfn "❌ Unexpected error: %s" ex.Message
        printfn "%s" ex.StackTrace
        1