module CLI.Program

open System
open Argu
open CLI.Commands

/// Command line arguments for Firefly CLI
type FireflyArgs =
    | [<CliPrefix(CliPrefix.None)>] Compile of ParseResults<CompileArgs>
    | [<CliPrefix(CliPrefix.None)>] Verify of ParseResults<VerifyArgs>
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
    let version = "0.1.0-alpha"
    printfn "Firefly %s - F# to Native Compiler with Zero-Allocation Guarantees" version
    printfn "Copyright (c) 2025 Speakez LLC"
    0

[<EntryPoint>]
let main argv =
    let errorHandler = ProcessExiter(colorizer = function ErrorCode.HelpText -> None | _ -> Some ConsoleColor.Red)
    let parser = ArgumentParser.Create<FireflyArgs>(programName = "firefly", errorHandler = errorHandler)

    try
        match parser.ParseCommandLine(argv) with
        | p when p.Contains Version -> showVersion()
        | p when p.TryGetSubCommand().IsSome ->
            match p.GetSubCommand() with
            | Compile args -> CompileCommand.compile args
            | Verify args -> VerifyCommand.verify args
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