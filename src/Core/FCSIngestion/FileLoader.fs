module Core.FCSIngestion.FileLoader

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Text
open Core.Utilities.IntermediateWriter

/// JSON options configured for F# types
let private jsonOptions =
    let options = JsonSerializerOptions()
    options.Converters.Add(JsonFSharpConverter())
    options

/// Load project files using FCS (works for both .fsx and .fs)
let loadProjectFiles (mainFile: string) (checker: FSharpChecker) (intermediatesDir: string option) =
    async {
        printfn "  Loading project files from: %s" mainFile
        
        // Read the source file
        let! sourceText = File.ReadAllTextAsync(mainFile) |> Async.AwaitTask
        
        // Get project options (works for both script and project mode)
        let! (projectOptions, diagnostics) = 
            if mainFile.EndsWith(".fsx") then
                // Script mode
                checker.GetProjectOptionsFromScript(
                    fileName = mainFile,
                    source = SourceText.ofString sourceText,
                    ?otherFlags = Some [| "--define:ZERO_ALLOCATION"; "--define:FIDELITY" |],
                    ?useFsiAuxLib = Some false,
                    ?useSdkRefs = Some false,
                    ?assumeDotNetFramework = Some false
                )
            else
                // Future: Regular F# project mode
                failwith "TODO: .fs project support"
        
        // Write compilation order if keeping intermediates
        match intermediatesDir with
        | Some dir ->
            let fcsDir = Path.Combine(dir, "fcs", "summary")
            Directory.CreateDirectory(fcsDir) |> ignore
            
            // Write compilation order
            let orderData = 
                projectOptions.SourceFiles 
                |> Array.mapi (fun i f -> {| Order = i; FileName = f; ShortName = Path.GetFileName(f) |})
            
            writeFile fcsDir "compilation_order" ".json" (JsonSerializer.Serialize(orderData, jsonOptions))
        | None -> ()
        
        return (projectOptions, diagnostics)
    }