module Examples.HelloWorldHalfCurried

open FSharp.NativeInterop
open Alloy
open Alloy.NativeTypes
open Alloy.Memory

#nowarn "9"

/// A simple hello world that reads user input and formats a greeting.
/// Uses native types throughout - no BCL string types.
let hello() =
    // Allocate input buffer on stack
    let inputBuffer = NativePtr.stackalloc<byte> 256

    // Write prompt
    let promptBytes = "Enter your name: "B
    Console.writeRaw Console.STDOUT_FILENO (NativePtr.ofNativeInt (NativePtr.toNativeInt &&promptBytes.[0])) (promptBytes.Length - 1) |> ignore

    // Read user input
    let nameLen = Console.readLine inputBuffer 256
    let name = NativeStr(inputBuffer, nameLen)

    // Allocate output buffer
    let outputBuffer = NativePtr.stackalloc<byte> 512

    // Build greeting: "Hello, " + name + "!"
    let helloBytes = "Hello, "B
    let mutable pos = 0

    // Copy "Hello, "
    for i = 0 to helloBytes.Length - 2 do  // -2 to skip null terminator
        NativePtr.set outputBuffer pos helloBytes.[i]
        pos <- pos + 1

    // Copy name
    for i = 0 to name.Length - 1 do
        NativePtr.set outputBuffer pos (NativePtr.get name.Pointer i)
        pos <- pos + 1

    // Add "!"
    NativePtr.set outputBuffer pos (byte '!')
    pos <- pos + 1

    // Write greeting
    let greeting = NativeStr(outputBuffer, pos)
    Console.writeln greeting

[<EntryPoint>]
let main _argv =
    hello()
    0
