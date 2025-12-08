/// HelloWorld - Minimal working example
/// Based on FidelityHelloWorld/02_HelloWorldSaturated
module HelloWorld

open Alloy

[<EntryPoint>]
let main argv =
    Console.Write "Enter your name: "
    let name = Console.ReadLine()
    Console.WriteLine $"Hello, {name}!"
    0
