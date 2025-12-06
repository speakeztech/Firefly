module CLI.Program

open System
open Argu
open CLI.Commands.CompileCommand
open CLI.Commands.VerifyCommand
open CLI.Commands.DoctorCommand

/// Command line arguments for Firefly CLI - simplified approach
type ToolArgs =
    | Version
with
    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Version -> "Display version information"

/// Display version information
let showVersion() =
    let version = "0.4.164"
    printfn "Firefly %s - F# to Native Compiler with Deterministic Memory Management" version
    printfn "Copyright (c) 2025 SpeakEZ LLC"
    0

/// Display usage information
let showUsage() =
    printfn "Firefly - F# to Native Compiler"
    printfn ""
    printfn "Usage:"
    printfn "  firefly compile [options]    Compile F# to native code"
    printfn "  firefly verify [options]     Verify binary meets constraints"
    printfn "  firefly doctor [options]     Diagnose toolchain issues"
    printfn "  firefly --version            Display version information"
    printfn ""
    printfn "Use 'firefly <subcommand> --help' for more information about a subcommand."

[<EntryPoint>]
let main argv =
    let errorHandler = ProcessExiter(colorizer = function ErrorCode.HelpText -> None | _ -> Some ConsoleColor.Red)

    try
        if argv.Length = 0 then
            showUsage()
            0
        elif argv.[0] = "--version" then
            showVersion()
        elif argv.[0] = "compile" then
            let compileParser = ArgumentParser.Create<CompileArgs>(programName = "firefly compile", errorHandler = errorHandler)
            let compileArgs = Array.skip 1 argv
            let compileResults = compileParser.ParseCommandLine(compileArgs)
            execute compileResults
        elif argv.[0] = "verify" then
            let verifyParser = ArgumentParser.Create<VerifyArgs>(programName = "firefly verify", errorHandler = errorHandler)
            let verifyArgs = Array.skip 1 argv
            let verifyResults = verifyParser.ParseCommandLine(verifyArgs)
            verify verifyResults
        elif argv.[0] = "doctor" then
            let doctorParser = ArgumentParser.Create<DoctorArgs>(programName = "firefly doctor", errorHandler = errorHandler)
            let doctorArgs = Array.skip 1 argv
            let doctorResults = doctorParser.ParseCommandLine(doctorArgs)
            doctor doctorResults
        elif argv.[0] = "--help" || argv.[0] = "-h" then
            showUsage()
            0
        else
            printfn "Error: Unknown subcommand '%s'" argv.[0]
            printfn ""
            showUsage()
            1
    with
    | :? ArguParseException as ex ->
        printfn "%s" ex.Message
        1
    | ex ->
        printfn "Error: %s" ex.Message
        1