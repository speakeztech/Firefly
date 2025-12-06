/// WhileTest - Minimal test for while loop and mutable variable
module WhileTest

open Alloy
open Alloy.Console

[<EntryPoint>]
let main argv =
    let mutable counter = 0

    WriteLine "Starting loop..."

    while counter < 5 do
        WriteLine "Tick"
        sleep 1000
        counter <- counter + 1

    WriteLine "Done!"
    0
