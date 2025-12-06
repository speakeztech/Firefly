module Examples.HelloWorldSaturated

open Alloy

let hello() =
    Console.Write "Enter your name: "
    let name = Console.ReadLine()
    Console.Write "Hello, "
    Console.Write name
    Console.WriteLine "!"

[<EntryPoint>]
let main argv =
    hello()
    0