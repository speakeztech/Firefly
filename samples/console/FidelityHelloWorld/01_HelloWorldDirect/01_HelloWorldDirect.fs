/// HelloWorld - Direct Module Calls Pattern (Minimal Version)
/// Tests the basic compilation pipeline with static string output only
module Examples.HelloWorldDirect

open Alloy

[<EntryPoint>]
let main argv =
    // Simple static string output - no input, no variables
    Console.Write "Hello, World!"
    Console.WriteLine ""
    0
