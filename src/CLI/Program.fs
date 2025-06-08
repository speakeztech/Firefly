module CLI.Program

open System
open Argu
open CLI.Commands.CompileCommand
open CLI.Commands.VerifyCommand

/// Command line arguments for Firefly CLI
type FireflyArgs =
    | Compile of compile_args: string list
    | Verify of verify_args: string list
    | Version
with
    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Compile _ -> "Compile F# to native code"
            | Verify _ -> "Verify binary meets constraints"
            | Version -> "Display version information"

/// Display version information
let showVersion() =
    let version = "1.0.0"
    printfn "Firefly %s - F# to Native Compiler with Zero-Allocation Guarantees" version
    printfn "Copyright (c) 2025 Speakez LLC"
    0

[<EntryPoint>]
let main argv =
    let errorHandler = ProcessExiter(colorizer = function ErrorCode.HelpText -> None | _ -> Some ConsoleColor.Red)
    let parser = ArgumentParser.Create<FireflyArgs>(programName = "firefly", errorHandler = errorHandler)

    try
        match parser.ParseCommandLine(argv) with
        | results when results.Contains Version -> showVersion()
        | results when results.TryGetSubCommand().IsSome ->
            match results.GetSubCommand() with
            | Compile args ->
                let compileParser = ArgumentParser.Create<CompileArgs>(programName = "firefly compile", errorHandler = errorHandler)
                let compileResults = compileParser.ParseCommandLine(Array.ofList args)
                compile compileResults
            | Verify args ->
                let verifyParser = ArgumentParser.Create<VerifyArgs>(programName = "firefly verify", errorHandler = errorHandler)
                let verifyResults = verifyParser.ParseCommandLine(Array.ofList args)
                verify verifyResults
            | _ -> failwith "Unexpected subcommand"
        | _ -> 
            printfn "%s" (parser.PrintUsage())
            0
    with
    | :? ArguParseException as ex ->
        printfn "%s" ex.Message
        1
    | ex ->
        printfn "Error: %s" ex.Message
        1