module Examples.HelloWorldFullCurried

open Alloy
open Alloy.Console
open Alloy.Text
open Alloy.Text.UTF8
open Alloy.Memory

let hello greetingFormat =
    use buffer = stackBuffer<byte> 256
    Prompt "Enter your name: "
    
    readInto buffer
    |> Result.map (fun length -> 
        buffer.AsReadOnlySpan(0, length) 
        |> spanToString)
    |> Result.defaultValue "Unknown Person"
    |> fun name -> sprintf greetingFormat name
    |> WriteLine

[<EntryPoint>]
let main argv = 
    match argv with
    | [|greetingFormat|] -> hello greetingFormat
    | _ -> hello "Hello, %s!"
    
    0