namespace Core.Utilities

open System.IO

module IntermediateWriter =
    
    let writeFile (intermediatesDir: string) (baseName: string) (extension: string) (content: string) : unit =
        try
            let filePath = Path.Combine(intermediatesDir, baseName + extension)
            File.WriteAllText(filePath, content, System.Text.UTF8Encoding(false))
            printfn "  Wrote %s (%d bytes)" (Path.GetFileName(filePath)) content.Length
        with ex ->
            printfn "  Warning: Failed to write %s: %s" extension ex.Message