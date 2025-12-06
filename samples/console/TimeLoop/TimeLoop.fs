/// TimeLoop - Demonstrates getting current time and displaying as datetime
/// The forcing function: Get system time, convert to human-readable datetime, output to console
module TimeLoop

open Alloy
open Alloy.Console

/// Display current datetime in a loop
/// Goal: Print the current datetime 5 times, once per second
let displayTimeLoop (iterations: int) =
    let mutable counter = 0

    WriteLine "TimeLoop - Current DateTime Demo"
    WriteLine ""

    while counter < iterations do
        // Get current datetime as string and print it
        let now = currentDateTimeString()
        WriteLine now

        // Sleep for 1 second
        sleep 1000

        counter <- counter + 1

    WriteLine ""
    WriteLine "Done."

/// Entry point
[<EntryPoint>]
let main argv =
    let iterations = 5
    displayTimeLoop iterations
    0
