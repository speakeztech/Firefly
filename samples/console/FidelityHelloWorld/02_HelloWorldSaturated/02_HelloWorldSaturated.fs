module Examples.HelloWorldSaturated

open Alloy

/// Demonstrates SATURATED function calls - all arguments provided at once.
/// Uses BCL-sympathetic Alloy APIs.
let hello() =
    Console.Write "Enter your name: "
    let name = Console.ReadLine()
    Console.WriteLine $"Hello, {name}!"

[<EntryPoint>]
let main argv =
    hello()
    0
