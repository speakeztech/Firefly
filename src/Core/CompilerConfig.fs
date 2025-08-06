module Core.CompilerConfig

/// Global compiler configuration settings
type CompilerSettings = {
    /// Enable verbose correlation output during PSG building
    VerboseCorrelation: bool
    /// Enable verbose type integration output
    VerboseTypeIntegration: bool
    /// Enable verbose reachability analysis output
    VerboseReachability: bool
    /// Enable all verbose outputs
    VerboseAll: bool
}

/// Default compiler settings (quiet mode)
let defaultSettings = {
    VerboseCorrelation = false
    VerboseTypeIntegration = false
    VerboseReachability = false
    VerboseAll = false
}

/// Mutable global settings
let mutable settings = defaultSettings

/// Enable verbose mode for all components
let enableVerboseMode() =
    settings <- { settings with VerboseAll = true }

/// Disable verbose mode for all components
let disableVerboseMode() =
    settings <- defaultSettings

/// Check if correlation verbosity is enabled
let isCorrelationVerbose() =
    settings.VerboseAll || settings.VerboseCorrelation

/// Check if type integration verbosity is enabled
let isTypeIntegrationVerbose() =
    settings.VerboseAll || settings.VerboseTypeIntegration

/// Check if reachability verbosity is enabled
let isReachabilityVerbose() =
    settings.VerboseAll || settings.VerboseReachability