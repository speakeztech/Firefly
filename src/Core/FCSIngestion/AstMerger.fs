module Core.FCSIngestion.AstMerger

open FSharp.Compiler.Syntax
open FSharp.Compiler.CodeAnalysis

/// Merge multiple parsed inputs into a consolidated AST
let mergeParseResults (parseResults: FSharpParseFileResults array) : ParsedInput =
    // Extract all modules from all files
    let allModules = 
        parseResults
        |> Array.collect (fun result ->
            match result.ParseTree with
            | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
                modules |> Array.ofList
            | ParsedInput.SigFile _ ->
                [||]  // Ignore signature files for now
        )
        |> Array.toList
    
    // Take metadata from the first file (main file)
    match parseResults.[0].ParseTree with
    | ParsedInput.ImplFile(ParsedImplFileInput(fileName, isScript, qualName, pragmas, hashDirectives, _, isLastCompiled, isExe, ids)) ->
        // Create consolidated implementation file with all modules
        ParsedInput.ImplFile(
            ParsedImplFileInput(
                fileName, 
                isScript, 
                qualName, 
                pragmas, 
                hashDirectives, 
                allModules,  // All modules from all files
                isLastCompiled, 
                isExe, 
                ids))
    | _ ->
        failwith "Main file must be an implementation file"