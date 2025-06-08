module Dabbit.BindingMetadata.BindingAttributes

/// Represents binding options for external libraries
type BindingOption =
    | Static
    | Dynamic
    | Auto

/// Represents metadata for an external binding
type BindingMetadata = {
    SymbolName: string
    LibraryName: string
    BindingType: BindingOption
    SelectiveLinking: bool
}

/// Resolves the binding options for a given symbol
let resolveBindingForSymbol (symbolName: string) : BindingMetadata =
    // In a real implementation, this would use project configuration
    // and sophisticated heuristics to determine optimal binding strategy
    { 
        SymbolName = symbolName
        LibraryName = "libc.so.6"
        BindingType = Static
        SelectiveLinking = true 
    }
/// Represents binding strategies for external libraries
type BindingStrategy =
    | Static
    | Dynamic

/// Provides metadata for native library bindings
type BindingInfo = {
    LibraryName: string
    Strategy: BindingStrategy
    SelectiveLinking: bool
    Symbols: string list
}

/// Extracts binding metadata from program annotations
let extractBindingMetadata (sourceCode: string) : BindingInfo list =
    // This would analyze source code for binding attributes
    // For now, return an empty list as a placeholder
    []
