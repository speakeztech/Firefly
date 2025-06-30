namespace Core.Utilities

open System
open System.IO
open Core.XParsec.Foundation

module ReadSourceFile = 

    let readSourceFile (inputPath: string) : CompilerResult<string> =
            try
                let content = File.ReadAllText(inputPath)
                if String.IsNullOrEmpty(content) then
                    CompilerFailure [SyntaxError(
                        { Line = 0; Column = 0; File = inputPath; Offset = 0 },
                        "Source file is empty",
                        ["file reading"])]
                else Success content
            with ex ->
                CompilerFailure [SyntaxError(
                    { Line = 0; Column = 0; File = inputPath; Offset = 0 },
                    sprintf "Error reading file: %s" ex.Message,
                    ["file reading"])]