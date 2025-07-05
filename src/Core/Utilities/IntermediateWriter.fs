namespace Core.Utilities

open System.IO

module IntermediateWriter =
    
    /// Simple file writer that takes a full path and content
    let writeIntermediateFile (filePath: string) (content: string) : unit =
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