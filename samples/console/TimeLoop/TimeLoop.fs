/// TimeLoop - Demonstrates platform-specific time operations
/// Shows Alloy's Time module with high-resolution timing and sleep
module TimeLoop

open Alloy
open Alloy.Console
open Alloy.Time

/// Display elapsed time in a loop
/// Demonstrates:
/// - High-resolution timing using platform-native APIs
/// - Mutable state in a while loop
/// - Platform-agnostic time operations (Windows/Linux/macOS)
let displayTimeLoop (iterations: int) =
    let mutable counter = 0

    // Capture start time using high-resolution timer
    let startTicks = highResolutionTicks()
    let freq = tickFrequency()

    WriteLine $"Starting time loop with {iterations} iterations..."
    WriteLine $"Tick frequency: {freq} ticks/second"
    WriteLine ""

    while counter < iterations do
        // Calculate elapsed time
        let currentTicks = highResolutionTicks()
        let elapsedTicks = currentTicks - startTicks
        let elapsedSeconds = float elapsedTicks / float freq

        // Get current timestamp
        let now = currentUnixTimestamp()

        // Display iteration info
        WriteLine $"Iteration {counter + 1}/{iterations}"
        WriteLine $"  Elapsed: {elapsedSeconds:F3} seconds"
        WriteLine $"  Unix timestamp: {now}"
        WriteLine ""

        // Sleep for 1 second (1000 milliseconds)
        sleep 1000

        counter <- counter + 1

    // Final timing
    let finalTicks = highResolutionTicks()
    let totalSeconds = float (finalTicks - startTicks) / float freq
    WriteLine $"Loop completed in {totalSeconds:F3} seconds"

/// Entry point
[<EntryPoint>]
let main argv =
    // Default to 5 iterations, or parse from command line
    let iterations =
        match argv with
        | [| count |] ->
            match System.Int32.TryParse count with
            | true, n when n > 0 -> n
            | _ -> 5
        | _ -> 5

    displayTimeLoop iterations
    0
