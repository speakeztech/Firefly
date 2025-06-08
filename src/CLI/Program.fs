module CLI.Program

open System
open System.Reflection
open Argu
open CLI.Commands.CompileCommand
open CLI.Commands.VerifyCommand
open CLI.Commands.DoctorCommand

/// Command line arguments for Firefly CLI
type FireflyArgs =
    | Version
with
    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Version -> "Display version information"

/// Gets the version from the assembly metadata
let private getAssemblyVersion() =
    let assembly = Assembly.GetExecutingAssembly()
    let version = assembly.GetName().Version
    let informationalVersion = 
        assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()
        |> Option.ofObj
        |> Option.map (fun attr -> attr.InformationalVersion)
        |> Option.defaultValue (version.ToString())
    informationalVersion

/// Display version information
let showVersion() =
    let version = getAssemblyVersion()
    printfn "Firefly %s - F# to Native Compiler with Zero-Allocation Guarantees" version
    printfn "Copyright (c) 2025 Speakez LLC"
    0

/// Display usage information
let showUsage() =
    printfn "Firefly - F# to Native Compiler"
    printfn ""
    printfn "Usage:"
    printfn "  firefly compile [options]    Compile F# to native code"
    printfn "  firefly verify [options]     Verify binary meets constraints"
    printfn "  firefly doctor [options]     Check system health"
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
            compile compileResults
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
        printfn "Argument parsing error: %s" ex.Message
        1
    | ex ->
        printfn "Unexpected error in CLI: %s" ex.Message
        printfn "Stack trace: %s" ex.StackTrace
        1