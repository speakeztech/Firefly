module Examples.HelloWorldHalfCurried

open Alloy
open Alloy.Text
open Alloy.Text.UTF8
open Alloy.Memory


let hello() =
    use buffer = stackBuffer<byte> 256
    Console.Prompt "Enter your name: "
    
    let name = 
        match Console.readInto buffer with
        | Ok length -> spanToString (buffer.AsReadOnlySpan(0, length))
        | Error _ -> "Unknown Person"
    
    name
    |> String.format $"Hello, %s{name}!"
    |> Console.WriteLine

[<EntryPoint>]
let main argv = 
    hello()
    0