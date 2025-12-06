/// TimeLoopSimple - Demonstrates time operations with cycle timing
/// Shows local time and milliseconds between cycles
module TimeLoopSimple

open Alloy
open Alloy.Console

[<EntryPoint>]
let main argv =
    let mutable counter = 0
    let mutable lastTicks = 0L

    WriteLine "=== TimeLoop Demo ==="
    WriteLine "Showing 5 iterations with timing"
    WriteLine ""

    // Get initial time
    lastTicks <- highResolutionTicks()

    while counter < 5 do
        // Get current high-res ticks
        let currentTicks = highResolutionTicks()

        // Calculate elapsed milliseconds since last iteration
        // tickFrequency is in Hz (ticks per second)
        // elapsed_ms = (current - last) * 1000 / frequency
        let elapsed = currentTicks - lastTicks
        let freq = tickFrequency()
        let elapsedMs = (elapsed * 1000L) / freq

        // Get current Unix timestamp (seconds since 1970)
        let unixTime = currentUnixTimestamp()

        // Output iteration info
        Write "Iteration "
        writeInt (counter + 1)
        WriteLine "/5"

        Write "  Unix timestamp: "
        writeInt64 unixTime
        WriteLine ""

        Write "  Elapsed since last: "
        writeInt64 elapsedMs
        WriteLine " ms"

        WriteLine ""

        // Update for next iteration
        lastTicks <- currentTicks

        // Sleep 1 second
        sleep 1000

        counter <- counter + 1

    WriteLine "=== Done ==="
    0
