/// HelloWorldInteractive - Demonstrates stack-based memory and user input
/// Shows Alloy's zero-allocation patterns with Result handling
module HelloWorldInteractive

open Alloy
open Alloy.Console
open Alloy.Memory

/// Read a name from the user and greet them
/// Demonstrates:
/// - Stack-allocated buffers (no heap allocation)
/// - Result-based error handling
/// - Span operations for zero-copy string handling
let hello() =
    // Allocate 256 bytes on the stack for input buffer
    // This is deallocated automatically when the function returns
    use buffer = stackBuffer<byte> 256

    // Prompt the user
    Write "Enter your name: "

    // Read input into the stack buffer
    // Returns Result<int, Error> where int is bytes read
    let name =
        match readInto buffer with
        | Ok length ->
            // Create a span view over the valid portion of the buffer
            // Convert to string without additional allocation
            buffer.AsSpan(0, length) |> Utf8.toString
        | Error _ ->
            // Default name if read fails
            "World"

    // Output the greeting
    WriteLine $"Hello, {name}!"

/// Entry point
[<EntryPoint>]
let main argv =
    hello()
    0
