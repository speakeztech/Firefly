namespace Core.Utilities

open System.IO
open System.Text.Json
open System.Text.Json.Serialization

module IntermediateWriter =

    /// Simple file writer that takes a full path and content
    let writeFileToPath (filePath: string) (content: string) : unit =
        try
            // Ensure directory exists
            let directory = Path.GetDirectoryName(filePath)
            if not (Directory.Exists(directory)) then
                Directory.CreateDirectory(directory) |> ignore

            // Write the file
            File.WriteAllText(filePath, content)
            printfn "  Wrote %s (%d bytes)" (Path.GetFileName(filePath)) content.Length
        with ex ->
            printfn "  Warning: Could not write %s: %s" filePath ex.Message

    /// JSON options for consistent serialization (camelCase, no F# unions)
    let jsonOptions =
        let options = JsonSerializerOptions(WriteIndented = true)
        options.PropertyNamingPolicy <- JsonNamingPolicy.CamelCase
        options

    /// JSON options with F# type support (for F# union types, records, etc.)
    let jsonOptionsWithFSharpSupport =
        let options = JsonSerializerOptions(WriteIndented = true)
        options.Converters.Add(JsonFSharpConverter())
        options

    /// Write data as JSON to specified filename in output directory
    let writeJsonAsset (outputDir: string) (filename: string) (data: obj) : unit =
        let json = JsonSerializer.Serialize(data, jsonOptions)
        let outputPath = Path.Combine(outputDir, filename)
        writeFileToPath outputPath json