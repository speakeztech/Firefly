/// <summary>
/// Core.XParsec.Foundation provides the foundational parser combinator infrastructure
/// for the Firefly F# to native compiler. Built on top of XParsec, this module provides
/// Firefly-specific error handling, state management, and utility combinators.
/// </summary>
module Core.XParsec.Foundation

open System
open XParsec

// ======================================
// Compiler result types for integration
// ======================================

/// <summary>
/// Represents different types of errors that can occur during compilation phases.
/// </summary>
type CompilerError = 
    /// <summary>Parse error with position information and context stack.</summary>
    /// <param name="position">String representation of the error position</param>
    /// <param name="message">Error message describing what went wrong</param>
    /// <param name="context">List of context strings showing the parsing stack</param>
    | ParseError of position: string * message: string * context: string list
    
    /// <summary>Transformation error between compilation phases.</summary>
    /// <param name="phase">Name of the compilation phase where error occurred</param>
    /// <param name="input">Description of the input that caused the error</param>
    /// <param name="expected">Description of what was expected</param>
    /// <param name="message">Detailed error message</param>
    | TransformError of phase: string * input: string * expected: string * message: string
    
    /// <summary>General compiler error with optional details.</summary>
    /// <param name="phase">Name of the compilation phase where error occurred</param>
    /// <param name="message">Error message</param>
    /// <param name="details">Optional additional details about the error</param>
    | CompilerError of phase: string * message: string * details: string option

    override this.ToString() =
        match this with
        | ParseError(pos, msg, ctx) -> 
            let contextStr = if ctx.IsEmpty then "" else sprintf " (Context: %s)" (String.concat " -> " ctx)
            sprintf "Parse error at %s: %s%s" pos msg contextStr
        | TransformError(phase, input, expected, msg) -> 
            sprintf "Transform error in %s: Expected %s from %s, but %s" phase expected input msg
        | CompilerError(phase, msg, details) -> 
            match details with
            | Some d -> sprintf "Compiler error in %s: %s (%s)" phase msg d
            | None -> sprintf "Compiler error in %s: %s" phase msg

/// <summary>
/// Result type for compiler operations that can either succeed with a value
/// or fail with a list of compilation errors.
/// </summary>
/// <typeparam name="T">The type of value returned on success</typeparam>
type CompilerResult<'T> = 
    /// <summary>Successful result containing the computed value.</summary>
    | Success of 'T
    /// <summary>Failed result containing one or more compilation errors.</summary>
    | CompilerFailure of CompilerError list

/// <summary>
/// Monadic bind operator for CompilerResult, allowing for chaining of
/// operations that may fail with compilation errors.
/// </summary>
/// <param name="result">The input CompilerResult</param>
/// <param name="f">Function to apply if the input is successful</param>
/// <returns>New CompilerResult after applying the function</returns>
let (>>=) (result: CompilerResult<'a>) (f: 'a -> CompilerResult<'b>) : CompilerResult<'b> =
    match result with
    | Success value -> f value
    | CompilerFailure errors -> CompilerFailure errors

/// <summary>
/// Map operator for CompilerResult, applying a function to the success value
/// while preserving any errors.
/// </summary>
/// <param name="result">The input CompilerResult</param>
/// <param name="f">Function to apply to the success value</param>
/// <returns>New CompilerResult with the function applied</returns>
let (|>>) (result: CompilerResult<'a>) (f: 'a -> 'b) : CompilerResult<'b> =
    match result with
    | Success value -> Success (f value)
    | CompilerFailure errors -> CompilerFailure errors

/// <summary>
/// Convenience function to create a parser that fails with a CompilerError.
/// Converts a CompilerError into a parser failure.
/// </summary>
/// <param name="error">The CompilerError to convert to a parser failure</param>
/// <returns>A parser that always fails with the given error message</returns>
let compilerFail (error: CompilerError) : Parser<'T> =
    fail (error.ToString())

// ======================================
// Combinators Module
// ======================================

/// <summary>
/// Extended parser combinators for advanced parsing scenarios in Firefly.
/// These combinators build upon XParsec's base functionality to provide
/// additional capabilities needed for compiler construction.
/// </summary>
module Combinators =
    
    /// <summary>
    /// Parses zero or more occurrences of a pattern separated by a separator.
    /// This is an alias for XParsec's sepBy combinator for consistency.
    /// </summary>
    /// <param name="p">Parser for the main pattern</param>
    /// <param name="sep">Parser for the separator</param>
    /// <returns>Parser that returns a list of parsed values</returns>
    let sepByZeroOrMore (p: Parser<'a>) (sep: Parser<'b>) : Parser<'a list> =
        sepBy p sep
    
    /// <summary>
    /// Runs a parser and saves its result in the parser state metadata
    /// using the specified key. Useful for passing data between parsers.
    /// </summary>
    /// <param name="key">String key to use for storing the result</param>
    /// <param name="p">Parser whose result should be saved</param>
    /// <returns>Parser that returns the original result while saving it to state</returns>
    let saveInState (key: string) (p: Parser<'a>) : Parser<'a> =
        p >>= (fun result ->
            getState >>= (fun state ->
                let newState = { state with Metadata = Map.add key (box result) state.Metadata }
                setState newState >>= (fun _ ->
                    succeed result
                )
            )
        )
    
    /// <summary>
    /// Retrieves a previously saved value from the parser state metadata.
    /// Returns None if no value was saved with the specified key.
    /// </summary>
    /// <param name="key">String key to look up in the state metadata</param>
    /// <returns>Parser that returns Some(value) if found, None otherwise</returns>
    let getFromState<'a> (key: string) : Parser<'a option> =
        getState >>= (fun state ->
            match Map.tryFind key state.Metadata with
            | Some value -> succeed (Some (unbox<'a> value))
            | None -> succeed None
        )
    
    /// <summary>
    /// Applies a parser exactly the specified number of times,
    /// collecting all results in a list.
    /// </summary>
    /// <param name="count">Number of times to apply the parser</param>
    /// <param name="p">Parser to apply repeatedly</param>
    /// <returns>Parser that returns a list of results</returns>
    let repeatParser (count: int) (p: Parser<'a>) : Parser<'a list> =
        let rec repeat n acc =
            if n <= 0 then
                succeed (List.rev acc)
            else
                p >>= (fun result ->
                    repeat (n - 1) (result :: acc)
                )
        repeat count []
    
    /// <summary>
    /// Executes a parser with a temporary modification to the parser state,
    /// then restores the original state after the parser completes.
    /// Useful for scoped state changes.
    /// </summary>
    /// <param name="modifyState">Function to modify the parser state</param>
    /// <param name="p">Parser to run with the modified state</param>
    /// <returns>Parser that runs with temporary state modification</returns>
    let withTemporaryState (modifyState: FireflyParserState -> FireflyParserState) (p: Parser<'a>) : Parser<'a> =
        getState >>= (fun originalState ->
            let modifiedState = modifyState originalState
            setState modifiedState >>= (fun _ ->
                p >>= (fun result ->
                    setState originalState >>= (fun _ ->
                        succeed result
                    )
                )
            )
        )
    
    /// <summary>
    /// Parser that succeeds only at the end of a line (newline) or end of input.
    /// Handles both Unix (\n) and Windows (\r\n) line endings.
    /// </summary>
    let eol : Parser<unit> =
        fun state ->
            if state.Position >= state.Input.Length then
                Success((), state)
            else
                let ch = state.Input.[state.Position]
                if ch = '\n' || ch = '\r' then
                    let newPos = 
                        if ch = '\r' && state.Position + 1 < state.Input.Length && 
                           state.Input.[state.Position + 1] = '\n' then
                            state.Position + 2
                        else
                            state.Position + 1
                    Success((), { state with Position = newPos })
                else
                    Failure("Expected end of line", state)
    
    /// <summary>
    /// Applies a parser exactly n times and returns all results in a list.
    /// This is an alias for repeatParser for convenience.
    /// </summary>
    /// <param name="n">Number of times to apply the parser</param>
    /// <param name="p">Parser to apply</param>
    /// <returns>Parser that returns a list of n results</returns>
    let pnTimes (n: int) (p: Parser<'a>) : Parser<'a list> =
        let rec loop i acc =
            if i = 0 then
                succeed (List.rev acc)
            else
                p >>= (fun x -> loop (i-1) (x::acc))
        loop n []
    
    /// <summary>
    /// Skips zero or more whitespace characters without returning a result.
    /// Preserves the current parser state otherwise.
    /// </summary>
    let skipWhitespace : Parser<unit> =
        many whitespace |>> ignore
    
    /// <summary>
    /// Parser that succeeds only if the current indentation level
    /// in the parser state matches the specified level.
    /// </summary>
    /// <param name="level">Expected indentation level</param>
    /// <returns>Parser that succeeds only at the correct indentation level</returns>
    let atIndentLevel (level: int) : Parser<unit> =
        getState >>= (fun state ->
            if state.IndentLevel = level then
                succeed ()
            else
                fail (sprintf "Expected indentation level %d, got %d" level state.IndentLevel)
        )

/// <summary>
/// Error handling utilities for parser combinators, providing enhanced
/// error reporting and recovery mechanisms for the Firefly compiler.
/// </summary>
module ErrorHandling =
    
    /// <summary>
    /// Adds contextual information to parser errors. The context is added
    /// to the error message stack and helps with debugging parse failures.
    /// </summary>
    /// <param name="context">Descriptive context string</param>
    /// <param name="p">Parser to wrap with context</param>
    /// <returns>Parser with enhanced error reporting</returns>
    let withContext (context: string) (p: Parser<'a>) : Parser<'a> =
        fun state ->
            let errorState = { state with ErrorMessages = context :: state.ErrorMessages }
            match p errorState with
            | Success(value, newState) -> Success(value, { newState with ErrorMessages = state.ErrorMessages })
            | Failure(msg, newState) -> Failure(sprintf "%s: %s" context msg, { newState with ErrorMessages = state.ErrorMessages })
    
    /// <summary>
    /// Attempts to run a parser and returns the result wrapped in an Option.
    /// Returns Some(result) on success, None on failure. Useful for optional parsing.
    /// </summary>
    /// <param name="p">Parser to attempt</param>
    /// <returns>Parser that always succeeds, returning an Option</returns>
    let tryParse (p: Parser<'a>) : Parser<'a option> =
        fun state ->
            match p state with
            | Success(value, newState) -> Success(Some value, newState)
            | Failure(_, _) -> Success(None, state)
    
    /// <summary>
    /// Creates a comprehensive error message with position information,
    /// context stack, and nearby input text for debugging purposes.
    /// </summary>
    /// <param name="message">Base error message</param>
    /// <param name="state">Parser state at the time of error</param>
    /// <returns>Formatted error message with detailed context</returns>
    let createErrorMessage (message: string) (state: FireflyParserState) : string =
        let errorStack = String.concat " -> " (List.rev state.ErrorMessages)
        let pos = state.Position
        let lineStart = state.Input.LastIndexOf('\n', pos - 1) + 1
        let line = state.Input.Substring(lineStart, pos - lineStart)
        let lineNum = state.Input.Substring(0, pos).Split('\n').Length
        let colNum = pos - lineStart + 1
        
        sprintf "Error at line %d, column %d: %s\nContext: %s\nNear: %s" 
                lineNum colNum message errorStack 
                (if pos < state.Input.Length then state.Input.Substring(pos, min 20 (state.Input.Length - pos)) else "<end of input>")

/// <summary>
/// Debugging utilities for parser development and troubleshooting.
/// These tools help with understanding parser execution flow and state changes.
/// </summary>
module DebugTools =
    
    /// <summary>
    /// Logs a debug message with current parser position without affecting
    /// the parser state or result. Useful for tracing parser execution.
    /// </summary>
    /// <param name="message">Debug message to log</param>
    /// <returns>Parser that succeeds with unit while logging the message</returns>
    let logDebug (message: string) : Parser<unit> =
        fun state ->
            printfn "[DEBUG] %s (Position: %d)" message state.Position
            Success((), state)
    
    /// <summary>
    /// Logs detailed information about the current parser state including
    /// position, indentation level, and error message stack.
    /// </summary>
    /// <param name="message">Context message for the state log</param>
    /// <returns>Parser that succeeds with unit while logging state info</returns>
    let logState (message: string) : Parser<unit> =
        getState >>= (fun state ->
            printfn "[STATE] %s: Position=%d, IndentLevel=%d, ErrorMessages=%A" 
                    message state.Position state.IndentLevel state.ErrorMessages
            succeed ()
        )
    
    /// <summary>
    /// Wraps a parser with debug logging that shows entry and exit points.
    /// Helps with understanding the flow of parser execution.
    /// </summary>
    /// <param name="label">Label to identify this parser in debug output</param>
    /// <param name="p">Parser to wrap with debug logging</param>
    /// <returns>Parser with debug entry/exit logging</returns>
    let debug (label: string) (p: Parser<'a>) : Parser<'a> =
        logDebug (sprintf "Entering: %s" label) >>= (fun _ ->
            p >>= (fun result ->
                logDebug (sprintf "Exiting: %s (Success)" label) >>= (fun _ ->
                    succeed result
                )
            )
        )

/// <summary>
/// File handling utilities for parsing files and including external content.
/// Provides error handling and encoding management for file-based parsing operations.
/// </summary>
module FileHandling =
    
    /// <summary>
    /// Parses a complete file using the specified parser with comprehensive
    /// error handling including file I/O errors and parse errors.
    /// </summary>
    /// <param name="parser">Parser to apply to the file contents</param>
    /// <param name="filename">Path to the file to parse</param>
    /// <returns>Result containing either the parsed value or an error message</returns>
    let parseFile (parser: Parser<'a>) (filename: string) : Result<'a, string> =
        try
            let content = System.IO.File.ReadAllText(filename)
            let initialState = createInitialState content
            match parser initialState with
            | Success(result, _) -> Ok result
            | Failure(msg, state) -> Error (ErrorHandling.createErrorMessage msg state)
        with
        | ex -> Error (sprintf "File error: %s" ex.Message)
    
    /// <summary>
    /// Parser combinator for including the contents of another file at parse time.
    /// Supports both quoted and unquoted filenames. Used for implementing
    /// include directives in source files.
    /// </summary>
    let includeFile : Parser<string> =
        let fileNameParser = quotedString <|> (many1 (satisfy (fun c -> c <> ' ' && c <> '\n' && c <> '\r')) |>> (List.map string >> String.concat ""))
        pstring "include" >>. spaces1 >>. fileNameParser >>= (fun filename ->
            try
                let content = System.IO.File.ReadAllText(filename)
                succeed content
            with
            | ex -> fail (sprintf "Failed to include file '%s': %s" filename ex.Message)
        )