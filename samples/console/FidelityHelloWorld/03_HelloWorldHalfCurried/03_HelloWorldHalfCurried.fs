module Examples.HelloWorldHalfCurried

open FSharp.NativeInterop
open Alloy
open Alloy.NativeTypes
open Alloy.NativeTypes.NativeString
open Alloy.Memory

#nowarn "9"

/// A simple hello world that reads user input and formats a greeting.
/// Uses native types throughout - no BCL string types.
///
/// Tests HALF-CURRIED patterns:
/// - Pipe operator: `x |> f`
/// - Partial application in function calls
/// - Curried Console functions
/// - Alloy semantic primitives (concat3)
///
/// Note: String concatenation uses concat3, a semantic primitive
/// that Firefly will lower to target-optimal code (SIMD on x86, tight loops
/// on ARM, parallel lanes on GPU). The loops are hidden in the primitive,
/// keeping user code clean and F#-idiomatic.
let hello() =
    // Allocate input buffer on stack
    let inputBuffer = NativePtr.stackalloc<byte> 256

    // Write prompt - tests curried Console function
    Console.writeB "Enter your name: "B

    // Read user input - tests curried function application
    let nameLen = Console.readLine inputBuffer 256
    let name = NativeStr(inputBuffer, nameLen)

    // Allocate output buffer for greeting
    let outputBuffer = NativePtr.stackalloc<byte> 512

    // Build greeting using semantic primitive
    // concat3 expresses INTENT (concatenate three strings)
    // Firefly/Alex chooses optimal implementation for target hardware
    let prefix = ofBytes "Hello, "B
    let suffix = ofBytes "!"B
    let greeting = concat3 outputBuffer prefix name suffix

    // Write greeting using pipe operator (tests |> pattern)
    greeting |> Console.writeln

[<EntryPoint>]
let main _argv =
    hello()
    0
