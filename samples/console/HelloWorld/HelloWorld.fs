/// HelloWorld - Saturated Call Pattern
/// Demonstrates native F# console I/O compiled to machine code
/// Pattern: 02_HelloWorldSaturated - All function calls are saturated (not curried)
module HelloWorld

open Alloy
open Alloy.Console
open Alloy.Memory

/// Entry point for the application
[<EntryPoint>]
let main argv =
    // Stack-allocated buffer for input (no heap allocation)
    use buffer = stackBuffer<byte> 64

    // Write prompt - saturated call (all args provided at once)
    Prompt "What is your name? "

    // Read user input into stack buffer
    // readInto uses SRTP to accept any type with Pointer and Length members
    let name =
        match readInto buffer with
        | Ok length -> "User"  // TODO: Convert buffer to string when Text module available
        | Error _ -> "Unknown"

    // Write greeting - saturated calls
    Write "Hello, "
    Write name
    WriteLine "!"

    0
