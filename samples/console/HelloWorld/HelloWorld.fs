/// HelloWorld - Interactive Firefly Sample
/// Demonstrates native F# console I/O compiled to machine code
module HelloWorld

open FSharp.NativeInterop
open Alloy

/// Entry point for the application
[<EntryPoint>]
let main argv =
    // Write prompt to stdout
    Console.writeLine "What is your name?"

    // Read user input (stack-allocated buffer, no heap)
    let mutable buffer = NativePtr.stackalloc<byte> 64
    let bytesRead = Console.readLine buffer 64

    // Write greeting
    Console.write "Hello, "
    // TODO: Write the actual name from buffer
    Console.writeLine "!"

    0
