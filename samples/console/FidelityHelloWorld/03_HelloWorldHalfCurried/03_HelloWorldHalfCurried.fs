module Examples.HelloWorldHalfCurried

open Alloy

/// Demonstrates HALF-CURRIED patterns:
/// - Pipe operator: `x |> f`
/// - Partial application
/// - String interpolation with pipe
let hello() =
    Console.Write "Enter your name: "

    Console.ReadLine()
    |> sprintf "Hello, %s!"
    |> Console.WriteLine

[<EntryPoint>]
let main argv =
    hello()
    0
