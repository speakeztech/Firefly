module Examples.HelloWorldHalfCurried

open FSharp.NativeInterop
open Alloy
open Alloy.NativeTypes
open Alloy.Memory

#nowarn "9"

/// A simple hello world that reads user input and formats a greeting.
/// Uses native types throughout - no BCL string types.
///
/// Tests HALF-CURRIED patterns:
/// - Pipe operator: `x |> ignore`
/// - Partial application in function calls
/// - Curried Console functions
let hello() =
    // Allocate input buffer on stack
    let inputBuffer = NativePtr.stackalloc<byte> 256

    // Write prompt using pipe operator (tests |> ignore pattern)
    Console.writeB "Enter your name: "B

    // Read user input - tests curried function application
    let nameLen = Console.readLine inputBuffer 256
    let name = NativeStr(inputBuffer, nameLen)

    // Allocate output buffer for greeting
    let outputBuffer = NativePtr.stackalloc<byte> 512
    let mutable pos = 0

    // Copy "Hello, " using byte literal helpers
    let helloBytes = "Hello, "B
    for i = 0 to bytesLen helloBytes - 1 do
        NativePtr.set outputBuffer pos helloBytes.[i]
        pos <- pos + 1

    // Copy name
    for i = 0 to name.Length - 1 do
        NativePtr.set outputBuffer pos (NativePtr.get name.Pointer i)
        pos <- pos + 1

    // Add "!"
    NativePtr.set outputBuffer pos (byte '!')
    pos <- pos + 1

    // Write greeting using pipe operator (tests |> pattern)
    let greeting = NativeStr(outputBuffer, pos)
    greeting |> Console.writeln

[<EntryPoint>]
let main _argv =
    hello()
    0
