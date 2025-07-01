namespace Core.Utilities

open System
open System.IO

module RemoveIntermediates =

    /// Recursively delete all files and subdirectories
    let rec private deleteDirectoryContents (dir: string) : unit =
        if Directory.Exists(dir) then
            try
                // Delete all files in this directory
                Directory.GetFiles(dir) 
                |> Array.iter (fun file ->
                    try
                        File.Delete(file)
                    with ex ->
                        printfn "  Warning: Could not delete %s: %s" (Path.GetFileName(file)) ex.Message
                )
                
                // Recursively delete all subdirectories
                Directory.GetDirectories(dir)
                |> Array.iter (fun subDir ->
                    deleteDirectoryContents subDir
                    try
                        Directory.Delete(subDir)
                    with ex ->
                        printfn "  Warning: Could not delete directory %s: %s" subDir ex.Message
                )
            with ex ->
                printfn "  Warning: Error processing directory %s: %s" dir ex.Message

    /// Removes all files and subdirectories from the intermediates directory
    let clearIntermediatesDirectory (intermediatesDir: string option) : unit =
        match intermediatesDir with
        | Some dir when Directory.Exists(dir) ->
            printfn "  Clearing intermediates directory..."
            deleteDirectoryContents dir
            printfn "  Intermediates directory cleared"
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
            // Clear existing contents first
            clearIntermediatesDirectory intermediatesDir
            
            // Ensure directory exists
            if not (Directory.Exists(dir)) then
                try
                    Directory.CreateDirectory(dir) |> ignore
                with ex ->
                    printfn "  Warning: Could not create intermediates directory: %s" ex.Message
        | None ->
            ()