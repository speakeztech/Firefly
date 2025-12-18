module Examples.HelloWorldHalfCurried

open Alloy

/// Demonstrates HALF-CURRIED patterns:
/// - Pipe operator: `x |> f`
/// - Function composition with pipes
/// Uses a helper function to format the greeting with NativeStr
let greet (name: NativeStr) : unit =
    Console.WriteLine $"Hello, {name}!"

let hello() =
    Console.Write "Enter your name: "

    Console.ReadLine()
    |> greet

[<EntryPoint>]
let main argv =
    hello()
    0
