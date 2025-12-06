module Examples.HelloWorldSaturated

open Alloy
open Alloy.Console
open Alloy.Text.UTF8
open Alloy.Memory

let hello() =
    use buffer = stackBuffer<byte> 256
    Prompt "Enter your name: "
    
    let name = 
        match readInto buffer with
        | Ok length -> spanToString (buffer.AsReadOnlySpan(0, length))
        | Error _ -> "Unknown Person"
    
    WriteLine $"Hello, %s{name}!"

[<EntryPoint>]
let main argv = 
    hello()
    0