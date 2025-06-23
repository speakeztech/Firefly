module Core.Utilities.RemoveIntermediates

open System
open System.IO

/// Removes all files from the intermediates directory
let clearIntermediatesDirectory (intermediatesDir: string option) : unit =
    match intermediatesDir with
    | Some dir when Directory.Exists(dir) ->
        try
            // Get all files in the directory
            let files = Directory.GetFiles(dir)
            
            // Delete each file
            files |> Array.iter (fun file ->
                try
                    File.Delete(file)
                with ex ->
                    printfn "  Warning: Could not delete %s: %s" (Path.GetFileName(file)) ex.Message
            )
            
            if files.Length > 0 then
                printfn "  Cleared %d intermediate files from %s" files.Length dir
        with ex ->
            printfn "  Warning: Error clearing intermediates directory: %s" ex.Message
    | Some dir ->
        // Directory doesn't exist yet, nothing to clear
        ()
    | None ->
        // No intermediates directory specified
        ()

/// Ensures the intermediates directory exists and is empty
let prepareIntermediatesDirectory (intermediatesDir: string option) : unit =
    match intermediatesDir with
    | Some dir ->
        // Clear existing files first
        clearIntermediatesDirectory intermediatesDir
        
        // Ensure directory exists
        if not (Directory.Exists(dir)) then
            try
                Directory.CreateDirectory(dir) |> ignore
            with ex ->
                printfn "  Warning: Could not create intermediates directory: %s" ex.Message
    | None ->
        ()