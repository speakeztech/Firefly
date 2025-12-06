module Examples.HelloWorldDirect

open Alloy
open Alloy.Text.UTF8
open Alloy.Memory

let hello() =
    use buffer = stackBuffer<byte> 256
    Console.Write "Enter your name: "
    
    let name = 
        match Console.readInto buffer with
        | Ok length -> 
            spanToString (buffer.AsReadOnlySpan(0, length))
        | Error _ -> "Unknown Person"
    
    let message = Console.sprintf "Hello, %s!" name 
    Console.WriteLine message

[<EntryPoint>]
let main argv = 
    hello()
    0