module Core.CompilerConfig

/// Global compiler configuration settings
type CompilerSettings = {
    /// Enable verbose correlation output during PSG building
    VerboseCorrelation: bool
    /// Enable verbose type integration output
    VerboseTypeIntegration: bool
    /// Enable verbose reachability analysis output
    VerboseReachability: bool
    /// Enable verbose SRTP extraction output
    VerboseSRTP: bool
    /// Enable all verbose outputs
    VerboseAll: bool
    /// Emit nanopass intermediate files
    EmitNanopassIntermediates: bool
    /// Output directory for nanopass intermediates
    NanopassOutputDir: string
}

/// Default compiler settings (quiet mode)
let defaultSettings = {
    VerboseCorrelation = false
    VerboseTypeIntegration = false
    VerboseReachability = false
    VerboseSRTP = false
    VerboseAll = false
    EmitNanopassIntermediates = false
    NanopassOutputDir = ""
}

/// Mutable global settings
let mutable settings = defaultSettings

/// Enable verbose mode for all components
let enableVerboseMode() =
    settings <- { settings with VerboseAll = true }

/// Disable verbose mode for all components
let disableVerboseMode() =
    settings <- defaultSettings

/// Enable nanopass intermediate emission
let enableNanopassIntermediates (outputDir: string) =
    settings <- { settings with EmitNanopassIntermediates = true; NanopassOutputDir = outputDir }

/// Disable nanopass intermediate emission
let disableNanopassIntermediates() =
    settings <- { settings with EmitNanopassIntermediates = false; NanopassOutputDir = "" }

/// Check if correlation verbosity is enabled
let isCorrelationVerbose() =
    settings.VerboseAll || settings.VerboseCorrelation

/// Check if type integration verbosity is enabled
let isTypeIntegrationVerbose() =
    settings.VerboseAll || settings.VerboseTypeIntegration

/// Check if reachability verbosity is enabled
let isReachabilityVerbose() =
    settings.VerboseAll || settings.VerboseReachability

/// Check if SRTP extraction verbosity is enabled
let isSRTPVerbose() =
    settings.VerboseAll || settings.VerboseSRTP

/// Check if nanopass intermediates should be emitted
let shouldEmitNanopassIntermediates() =
    settings.EmitNanopassIntermediates && settings.NanopassOutputDir <> ""

/// Get nanopass output directory
let getNanopassOutputDir() =
    settings.NanopassOutputDir