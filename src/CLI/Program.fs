module CLI.Program

open System
open Argu
open CLI.Commands.CompileCommand
open CLI.Commands.VerifyCommand
open CLI.Commands.DoctorCommand

/// Command line arguments for Firefly CLI - simplified approach
type FireflyArgs =
    | Version
with
    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Version -> "Display version information"

/// Display version information
let showVersion() =
    let version = "0.1.0-alpha"
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
        printfn "=== FIREFLY CLI ENTRY POINT ==="
        printfn "Command line arguments: %A" argv
        printfn "Number of arguments: %d" argv.Length
        
        if argv.Length = 0 then
            printfn "No arguments provided, showing usage"
            showUsage()
            0
        elif argv.[0] = "--version" then
            printfn "Version command requested"
            showVersion()
        elif argv.[0] = "compile" then
            printfn "=== COMPILE COMMAND DETECTED ==="
            printfn "Routing to CompileCommand.compile"
            let compileParser = ArgumentParser.Create<CompileArgs>(programName = "firefly compile", errorHandler = errorHandler)
            let compileArgs = Array.skip 1 argv
            printfn "Compile arguments: %A" compileArgs
            let compileResults = compileParser.ParseCommandLine(compileArgs)
            printfn "Parsed compile arguments, calling compile function"
            
            // Call the compile function from CompileCommand
            let result = compile compileResults
            printfn "Compile function returned: %d" result
            result
        elif argv.[0] = "verify" then
            printfn "=== VERIFY COMMAND DETECTED ==="
            let verifyParser = ArgumentParser.Create<VerifyArgs>(programName = "firefly verify", errorHandler = errorHandler)
            let verifyArgs = Array.skip 1 argv
            let verifyResults = verifyParser.ParseCommandLine(verifyArgs)
            verify verifyResults
        elif argv.[0] = "doctor" then
            printfn "=== DOCTOR COMMAND DETECTED ==="
            let doctorParser = ArgumentParser.Create<DoctorArgs>(programName = "firefly doctor", errorHandler = errorHandler)
            let doctorArgs = Array.skip 1 argv
            let doctorResults = doctorParser.ParseCommandLine(doctorArgs)
            doctor doctorResults
        elif argv.[0] = "--help" || argv.[0] = "-h" then
            printfn "Help command requested"
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