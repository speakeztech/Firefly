module Core.XParsec.Foundation

open System
open XParsec
open XParsec.Parsers
open XParsec.Combinators

/// Position information for error reporting
type SourcePosition = {
    Line: int
    Column: int
    File: string
    Offset: int
}

/// Core error types for the Firefly compiler
type FireflyError =
    | SyntaxError of position: SourcePosition * message: string * context: string list
    | ConversionError of phase: string * source: string * target: string * message: string
    | TypeCheckError of construct: string * message: string * location: SourcePosition
    | InternalError of phase: string * message: string * details: string option

/// Result type for compiler operations - no fallbacks allowed
type CompilerResult<'T> =
    | Success of 'T
    | CompilerFailure of FireflyError list

/// Compiler state for tracking translation context
type FireflyState = {
    CurrentFile: string
    ImportedModules: string list
    TypeDefinitions: Map<string, string>
    ScopeStack: string list list
    ErrorStack: string list
}

/// Core parsers specialized for Firefly compilation
module Parsers =
    /// Creates a parser that fails with a compiler error
    let compilerFail (error: FireflyError) : Parser<'T, char, FireflyState, 'Input, 'InputSlice> =
        let errorMsg = Message (error.ToString())
        fun reader -> fail errorMsg reader
    
    /// Gets the current state
    let getState : Parser<FireflyState, char, FireflyState, 'Input, 'InputSlice> =
        fun reader -> preturn reader.State reader
    
    /// Updates the state
    let updateState (f: FireflyState -> FireflyState) : Parser<unit, char, FireflyState, 'Input, 'InputSlice> =
        fun reader ->
            reader.State <- f reader.State
            preturn () reader
    
    /// Adds error context to a parser
    let withErrorContext (context: string) (p: Parser<'T, char, FireflyState, 'Input, 'InputSlice>) : Parser<'T, char, FireflyState, 'Input, 'InputSlice> =
        fun reader ->
            let oldState = reader.State
            let newState = { oldState with ErrorStack = context :: oldState.ErrorStack }
            reader.State <- newState
            
            match p reader with
            | Ok result -> 
                reader.State <- oldState  // Restore original state
                Ok result
            | Error err -> 
                reader.State <- oldState  // Restore original state
                ParseError.createNested (Message context) [err] reader.Position
    
    /// Creates a parser that requires success or fails with a message
    let required (msg: string) (p: Parser<'T, char, FireflyState, 'Input, 'InputSlice>) : Parser<'T, char, FireflyState, 'Input, 'InputSlice> =
        fun reader ->
            let pos = reader.Position
            match p reader with
            | Ok result -> Ok result
            | Error _ -> 
                reader.Position <- pos
                fail (Message msg) reader

/// State management for Firefly compiler
module StateManagement =
    open Parsers
    
    /// Creates a parser that adds a name to the current scope
    let bindInScope (name: string) : Parser<unit, char, FireflyState, 'Input, 'InputSlice> =
        updateState (fun state ->
            match state.ScopeStack with
            | currentScope :: rest ->
                { state with ScopeStack = (name :: currentScope) :: rest }
            | [] ->
                { state with ScopeStack = [[name]] }
        )
    
    /// Creates a parser that pushes a new scope
    let pushScope() : Parser<unit, char, FireflyState, 'Input, 'InputSlice> =
        updateState (fun state ->
            { state with ScopeStack = [] :: state.ScopeStack }
        )
    
    /// Creates a parser that pops the current scope
    let popScope() : Parser<unit, char, FireflyState, 'Input, 'InputSlice> =
        updateState (fun state ->
            match state.ScopeStack with
            | _ :: rest -> { state with ScopeStack = rest }
            | [] -> state
        )
    
    /// Checks if a name exists in any scope
    let isInScope (name: string) : Parser<bool, char, FireflyState, 'Input, 'InputSlice> =
        getState |>> (fun state ->
            state.ScopeStack |> List.exists (List.contains name)
        )

/// String level parsers
module StringParsers =
    /// Parse an identifier
    let identifier() : Parser<string, char, FireflyState, 'Input, 'InputSlice> =
        let isIdFirst c = Char.IsLetter c || c = '_'
        let isIdRest c = Char.IsLetterOrDigit c || c = '_'
        
        satisfy isIdFirst >>= fun first ->
        many (satisfy isIdRest) >>= fun rest ->
        let sb = System.Text.StringBuilder()
        sb.Append(first) |> ignore
        for c in rest do
            sb.Append(c) |> ignore
        preturn (sb.ToString())
    
    /// Parse a qualified identifier (Module.Name)
    let qualifiedIdentifier() : Parser<string list, char, FireflyState, 'Input, 'InputSlice> =
        let dot = satisfy (fun c -> c = '.')
        
        // We need to handle the struct tuple return type carefully
        sepBy1 (identifier()) dot >>= fun result ->
            // Extract the identifiers from the struct tuple
            let values = 
                let struct(ids, _) = result
                // Convert ImmutableArray to list
                Seq.toList ids
            preturn values

/// Result helpers for Firefly compiler
module ResultHelpers =
    /// Combines multiple compiler results
    let combineResults (results: CompilerResult<'T> list) : CompilerResult<'T list> =
        let folder acc result =
            match acc, result with
            | Success accValues, Success value -> Success (value :: accValues)
            | CompilerFailure errors, Success _ -> CompilerFailure errors
            | Success _, CompilerFailure errors -> CompilerFailure errors
            | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
        
        results
        |> List.fold folder (Success [])
        |> function
           | Success values -> Success (List.rev values)
           | CompilerFailure errors -> CompilerFailure errors
    
    /// Monadic bind for CompilerResult
    let bind (f: 'T -> CompilerResult<'U>) (result: CompilerResult<'T>) : CompilerResult<'U> =
        match result with
        | Success value -> f value
        | CompilerFailure errors -> CompilerFailure errors
    
    /// Maps a function over CompilerResult
    let map (f: 'T -> 'U) (result: CompilerResult<'T>) : CompilerResult<'U> =
        match result with
        | Success value -> Success (f value)
        | CompilerFailure errors -> CompilerFailure errors

/// Creates initial compiler state
let initialState (fileName: string) : FireflyState = {
    CurrentFile = fileName
    ImportedModules = []
    TypeDefinitions = Map.empty
    ScopeStack = [[]]
    ErrorStack = []
}

/// Runs a parser on input with state
let runParser 
    (p: Parser<'T, char, FireflyState, ReadableString, ReadableStringSlice>) 
    (input: string) 
    (state: FireflyState) : CompilerResult<'T> =
    
    let reader = Reader.ofString input state
    
    match p reader with
    | Ok result -> Success result.Parsed
    | Error err -> 
        // Convert XParsec error to Firefly error
        let pos = {
            Line = 1  // Would need position tracking to get actual line/column
            Column = int err.Position.Index + 1
            File = state.CurrentFile
            Offset = int err.Position.Index
        }
        
        let errorContext = state.ErrorStack
        let errorMsg = 
            match err.Errors with
            | Message msg -> msg
            | _ -> err.Errors.ToString()
            
        CompilerFailure [SyntaxError(pos, errorMsg, errorContext)]

/// Extension for easier parser composition with computation expressions
type ParserBuilder() =
    member inline _.Bind(p, f) = p >>= f
    member inline _.Return(x) = preturn x
    member inline _.ReturnFrom(p) = p
    member inline _.Zero() = preturn ()
    member inline _.Delay(f) = fun reader -> (f()) reader
    
    member inline _.Combine(p1, p2) = 
        p1 >>. p2
        
    member inline _.TryWith(p, handler) =
        fun reader ->
            try
                p reader
            with e ->
                (handler e) reader
                
    member inline _.TryFinally(p, compensation) =
        fun reader ->
            try
                p reader
            finally
                compensation()
    
    member inline _.Using(resource, f) =
        fun reader ->
            use r = resource
            (f r) reader

/// Parser computation expression
let parser = ParserBuilder()