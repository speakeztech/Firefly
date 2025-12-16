module Examples.HelloWorldFullCurried

open Alloy
open Alloy.Console

/// Demonstrates FULL-CURRIED patterns:
/// - Curried function with multiple parameters
/// - Pipe operator: `x |> f`
/// - Lambda: `fun name -> ...`
/// - Higher-order function composition
/// - Pattern matching on arrays

/// Curried greeting function - takes prefix then name
let greet prefix name =
    WriteLine $"{prefix}, {name}!"

/// Hello function partially applies greet
let hello prefix =
    Write "Enter your name: "
    ReadLine()
    |> greet prefix

[<EntryPoint>]
let main argv =
    match argv with
    | [|prefix|] -> hello prefix
    | _ -> hello "Hello"
    0
