/// Compilation Timing Infrastructure
/// Provides phase-level timing for identifying bottlenecks in the compilation pipeline.
///
/// Usage:
///   Firefly compile project.fidproj --timing
///   Firefly compile project.fidproj -T
///
/// Each phase is timed and reported with millisecond precision.
module Core.Timing

open System
open System.Diagnostics

/// A recorded phase timing
type PhaseTiming = {
    Name: string
    Description: string
    StartTime: DateTime
    ElapsedMs: int64
}

/// Global timing context - tracks all phase timings during a compilation run
type TimingContext = {
    mutable Enabled: bool
    mutable Phases: PhaseTiming list
    mutable CurrentPhase: (string * string * Stopwatch) option
}

/// Global timing context instance
let private context : TimingContext = {
    Enabled = false
    Phases = []
    CurrentPhase = None
}

/// Enable or disable timing
let setEnabled (enabled: bool) =
    context.Enabled <- enabled
    context.Phases <- []
    context.CurrentPhase <- None

/// Check if timing is enabled
let isEnabled () = context.Enabled

/// Start timing a phase
/// Returns a disposable that will end the phase when disposed
let startPhase (name: string) (description: string) : IDisposable =
    if context.Enabled then
        // End any current phase first
        match context.CurrentPhase with
        | Some (prevName, prevDesc, sw) ->
            sw.Stop()
            let timing = {
                Name = prevName
                Description = prevDesc
                StartTime = DateTime.Now.AddMilliseconds(-(float sw.ElapsedMilliseconds))
                ElapsedMs = sw.ElapsedMilliseconds
            }
            context.Phases <- context.Phases @ [timing]
        | None -> ()

        // Start new phase
        let sw = Stopwatch.StartNew()
        context.CurrentPhase <- Some (name, description, sw)

        // Print phase start
        let timestamp = DateTime.Now.ToString("HH:mm:ss.fff")
        printfn "[%s] [%s] %s..." timestamp name description

        // Return disposable that ends the phase
        { new IDisposable with
            member _.Dispose() =
                match context.CurrentPhase with
                | Some (n, d, stopwatch) when n = name ->
                    stopwatch.Stop()
                    let timing = {
                        Name = n
                        Description = d
                        StartTime = DateTime.Now.AddMilliseconds(-(float stopwatch.ElapsedMilliseconds))
                        ElapsedMs = stopwatch.ElapsedMilliseconds
                    }
                    context.Phases <- context.Phases @ [timing]
                    context.CurrentPhase <- None
                    // Print phase end with timing
                    let timestamp = DateTime.Now.ToString("HH:mm:ss.fff")
                    printfn "[%s] [%s] Done (%dms)" timestamp name stopwatch.ElapsedMilliseconds
                | _ -> ()
        }
    else
        // Timing disabled - return no-op disposable
        { new IDisposable with member _.Dispose() = () }

/// Time a phase with a function (functional style)
let timePhase (name: string) (description: string) (f: unit -> 'a) : 'a =
    use _ = startPhase name description
    f()

/// End any current phase (for use when phases don't have clear end points)
let endCurrentPhase () =
    if context.Enabled then
        match context.CurrentPhase with
        | Some (name, desc, sw) ->
            sw.Stop()
            let timing = {
                Name = name
                Description = desc
                StartTime = DateTime.Now.AddMilliseconds(-(float sw.ElapsedMilliseconds))
                ElapsedMs = sw.ElapsedMilliseconds
            }
            context.Phases <- context.Phases @ [timing]
            context.CurrentPhase <- None
            // Print phase end
            let timestamp = DateTime.Now.ToString("HH:mm:ss.fff")
            printfn "[%s] [%s] Done (%dms)" timestamp name sw.ElapsedMilliseconds
        | None -> ()

/// Record a phase timing manually (for phases that don't use startPhase/endPhase)
let recordPhase (name: string) (description: string) (elapsedMs: int64) =
    if context.Enabled then
        let timing = {
            Name = name
            Description = description
            StartTime = DateTime.Now.AddMilliseconds(-(float elapsedMs))
            ElapsedMs = elapsedMs
        }
        context.Phases <- context.Phases @ [timing]
        let timestamp = DateTime.Now.ToString("HH:mm:ss.fff")
        printfn "[%s] [%s] %s (%dms)" timestamp name description elapsedMs

/// Print timing summary
let printSummary () =
    if context.Enabled && not (List.isEmpty context.Phases) then
        printfn ""
        printfn "═══════════════════════════════════════════════════════════════════"
        printfn "                         Timing Summary                            "
        printfn "═══════════════════════════════════════════════════════════════════"

        // Calculate column widths
        let maxNameLen = context.Phases |> List.map (fun p -> p.Name.Length) |> List.max |> max 8
        let maxDescLen = context.Phases |> List.map (fun p -> p.Description.Length) |> List.max |> max 12

        // Header
        printfn "%-*s  %-*s  %10s  %7s" maxNameLen "Phase" maxDescLen "Description" "Time (ms)" "Pct"
        printfn "%s  %s  %s  %s"
            (String.replicate maxNameLen "─")
            (String.replicate maxDescLen "─")
            (String.replicate 10 "─")
            (String.replicate 7 "─")

        let total = context.Phases |> List.sumBy (fun p -> p.ElapsedMs)

        // Phase rows
        for phase in context.Phases do
            let pct = if total > 0L then float phase.ElapsedMs / float total * 100.0 else 0.0
            printfn "%-*s  %-*s  %10d  %6.1f%%"
                maxNameLen phase.Name
                maxDescLen phase.Description
                phase.ElapsedMs
                pct

        // Total
        printfn "%s  %s  %s  %s"
            (String.replicate maxNameLen "─")
            (String.replicate maxDescLen "─")
            (String.replicate 10 "─")
            (String.replicate 7 "─")
        printfn "%-*s  %-*s  %10d  %6.1f%%"
            maxNameLen "TOTAL"
            maxDescLen ""
            total
            100.0

        printfn "═══════════════════════════════════════════════════════════════════"
        printfn ""

/// Get all recorded phases
let getPhases () = context.Phases

/// Get total elapsed time in milliseconds
let getTotalMs () = context.Phases |> List.sumBy (fun p -> p.ElapsedMs)

/// Reset timing context
let reset () =
    context.Phases <- []
    context.CurrentPhase <- None
