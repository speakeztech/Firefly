/// SimpleTimeTest - Basic test of Time operations without complex features
module SimpleTimeTest

open Alloy
open Alloy.Console

/// Simple test: get time, sleep, get time again
[<EntryPoint>]
let main argv =
    // Get start ticks
    let startTicks = highResolutionTicks()
    let freq = tickFrequency()

    WriteLine "Starting simple time test..."

    // Sleep for 1 second
    sleep 1000

    // Get end ticks
    let endTicks = highResolutionTicks()

    // Just return success
    WriteLine "Done!"
    0
