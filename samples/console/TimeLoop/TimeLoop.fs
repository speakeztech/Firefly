/// TimeLoop - Demonstrates BCL-compatible DateTime and Thread.Sleep
/// Forcing function: Platform-specific time retrieval and sleep via Alex bindings
module TimeLoop

open Alloy
open Alloy.Console

/// Display current datetime in a loop
/// Goal: Print the current datetime 5 times, once per second
let displayTimeLoop (iterations: int) =
    WriteLine "TimeLoop - Current DateTime Demo"
    WriteLine ""

    let mutable counter = 0
    while counter < iterations do
        // BCL-compatible: DateTime.Now
        let now = DateTime.Now

        // BCL-compatible: DateTime.ToString()
        WriteLine (now.ToString())

        // BCL-compatible: Thread.Sleep
        Threading.Thread.Sleep 1000

        counter <- counter + 1

    WriteLine ""
    WriteLine "Done."

/// Entry point
[<EntryPoint>]
let main argv =
    let iterations = 5
    displayTimeLoop iterations
    0
