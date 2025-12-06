module Alex.Bindings.Time.TimeBindings

open Alex.Bindings.BindingTypes

// ===================================================================
// Unified Time Bindings Registration
// ===================================================================

/// Register all time bindings for the current platform
let registerAllBindings () =
    // Register Linux time binding
    BindingRegistry.register (Linux.createBinding())

    // Register macOS time binding
    BindingRegistry.register (MacOS.createBinding())

    // Register Windows time binding
    BindingRegistry.register (Windows.createBinding())

/// Get all time-related symbol patterns
let allTimePatterns = [
    "Alloy.Time.currentTicks"
    "Alloy.Time.highResolutionTicks"
    "Alloy.Time.tickFrequency"
    "Alloy.Time.sleep"
    "Alloy.Time.currentUnixTimestamp"
    "Alloy.Time.currentTimestamp"
    "Alloy.TimeApi.currentTicks"
    "Alloy.TimeApi.highResolutionTicks"
    "Alloy.TimeApi.tickFrequency"
    "Alloy.TimeApi.sleep"
    "Alloy.TimeApi.currentUnixTimestamp"
]

/// Check if a symbol name is a time operation
let isTimeOperation (symbolName: string) : bool =
    allTimePatterns |> List.exists (fun pattern ->
        symbolName.EndsWith(pattern.Split('.').[pattern.Split('.').Length - 1]) &&
        symbolName.Contains("Time"))
