module Core.XParsec.Foundation

open System
open FSharp.Compiler.Symbols
open FSharp.Compiler.Text

// ===================================================================
// Core Position and Error Types - Single Source of Truth
// ===================================================================

/// Position information for precise error reporting
type SourcePosition = {
    Line: int
    Column: int
    File: string
    Offset: int
}

/// Unified error type for the entire Firefly compiler pipeline
type FireflyError =
    | SyntaxError of position: SourcePosition * message: string * context: string list
    | ConversionError of phase: string * source: string * target: string * message: string
    | TypeCheckError of construct: string * message: string * location: SourcePosition
    | InternalError of phase: string * message: string * details: string option
    | ParseError of position: SourcePosition * message: string
    | DependencyResolutionError of symbol: string * message: string

/// Result type for all compiler operations - no exceptions allowed
type CompilerResult<'T> =
    | Success of 'T
    | CompilerFailure of FireflyError list

// ===================================================================
// AST and Function Types - Core Compiler Data Structures
// ===================================================================

/// Dependency classification for reachability analysis
type DependencyType =
    | DirectCall | AlloyLibraryCall | ConstructorCall | ExternalCall

/// Function dependency edge in the call graph
type Dependency = {
    From: string
    To: string
    CallSite: range
    Type: DependencyType
}

/// Memory allocation site for zero-allocation verification
type AllocationType = 
    | HeapAllocation | ObjectConstruction | CollectionAllocation

type AllocationSite = {
    TypeName: string
    Location: range
    AllocationType: AllocationType
}

// ===================================================================
// Essential Utility Functions
// ===================================================================

/// Convert FCS range to SourcePosition
let rangeToPosition (range: range) : SourcePosition = {
    Line = range.Start.Line
    Column = range.Start.Column
    File = range.FileName
    Offset = 0
}

/// Create a parser error from position and message
let createParseError (pos: SourcePosition) (message: string) : FireflyError =
    ParseError(pos, message)

/// Combine multiple compiler results into a single result
let combineResults (results: CompilerResult<'T> list) : CompilerResult<'T list> =
    let successes = ResizeArray<'T>()
    let failures = ResizeArray<FireflyError>()
    
    for result in results do
        match result with
        | Success value -> successes.Add(value)
        | CompilerFailure errors -> failures.AddRange(errors)
    
    if failures.Count = 0 then
        Success (successes |> List.ofSeq)
    else
        CompilerFailure (failures |> List.ofSeq)