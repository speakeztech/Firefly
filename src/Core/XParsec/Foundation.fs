module Core.XParsec.Foundation

open System
open System.Text
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
    | MLIRGenerationError of phase: string * message: string * function: string option

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

/// MLIR-specific state for code generation
type MLIRState = {
    Output: StringBuilder
    Indent: int
    SSACounter: int
    LocalVars: Map<string, string * string>  // name -> (ssa, type)
    RequiredExternals: Set<string>
    CurrentFunction: string option
    GeneratedFunctions: Set<string>
    CurrentModule: string list
    HasErrors: bool
}

/// Combined state for MLIR generation
type MLIRBuilderState = {
    Firefly: FireflyState
    MLIR: MLIRState
}

/// MLIR combinator type - transforms MLIR state and produces results
type MLIRCombinator<'T> = MLIRBuilderState -> CompilerResult<'T * MLIRBuilderState>

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
    let updateState (f: FireflyState -> FireflyState) : Parser<unit, char, FireflyState, 'InputSlice> =
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

/// MLIR combinators built on Foundation patterns
module MLIRCombinators =
    
    /// Lift a value into the MLIR context
    let lift (value: 'T): MLIRCombinator<'T> =
        fun state -> Success(value, state)
    
    /// Get current MLIR state
    let getMLIRState: MLIRCombinator<MLIRBuilderState> =
        fun state -> Success(state, state)
    
    /// Update MLIR state
    let updateMLIRState (f: MLIRBuilderState -> MLIRBuilderState): MLIRCombinator<unit> =
        fun state -> Success((), f state)
    
    /// Sequential composition of MLIR combinators
    let (>>=) (combinator: MLIRCombinator<'T>) (f: 'T -> MLIRCombinator<'U>): MLIRCombinator<'U> =
        fun state ->
            match combinator state with
            | Success(value, state') -> f value state'
            | CompilerFailure errors -> CompilerFailure errors
    
    /// Sequential composition, ignoring first result
    let (>>) (first: MLIRCombinator<'T>) (second: MLIRCombinator<'U>): MLIRCombinator<'U> =
        first >>= (fun _ -> second)
    
    /// Apply function to combinator result
    let (|>>) (combinator: MLIRCombinator<'T>) (f: 'T -> 'U): MLIRCombinator<'U> =
        combinator >>= (fun value -> lift (f value))
    
    /// Choice between two combinators
    let (<|>) (first: MLIRCombinator<'T>) (second: MLIRCombinator<'T>): MLIRCombinator<'T> =
        fun state ->
            match first state with
            | Success(result, state') -> Success(result, state')
            | CompilerFailure _ -> second state
    
    /// Optional combinator
    let optional (combinator: MLIRCombinator<'T>): MLIRCombinator<'T option> =
        (combinator |>> Some) <|> (lift None)
    
    /// Many combinator for repeated application
    let rec many (combinator: MLIRCombinator<'T>): MLIRCombinator<'T list> =
        (combinator >>= fun head ->
         many combinator >>= fun tail ->
         lift (head :: tail))
        <|> lift []
    
    /// Fail with MLIR generation error
    let fail (phase: string) (message: string): MLIRCombinator<'T> =
        fun state ->
            let location = state.MLIR.CurrentFunction
            let error = MLIRGenerationError(phase, message, location)
            let state' = { state with MLIR = { state.MLIR with HasErrors = true } }
            // Emit error comment to output
            state'.MLIR.Output.AppendLine(sprintf "    // ERROR: [%s] %s" phase message) |> ignore
            CompilerFailure [error]
    
    /// Try a combinator with fallback on error
    let attempt (combinator: MLIRCombinator<'T>) (fallback: string -> MLIRCombinator<'T>): MLIRCombinator<'T> =
        fun state ->
            match combinator state with
            | Success(result, state') -> Success(result, state')
            | CompilerFailure errors -> 
                let errorMsg = errors |> List.map string |> String.concat "; "
                fallback errorMsg state

/// MLIR emission primitives
module MLIREmission =
    
    /// Generate unique SSA name
    let nextSSA (prefix: string): MLIRCombinator<string> =
        fun state ->
            let name = sprintf "%%%s%d" prefix state.MLIR.SSACounter
            let mlirState' = { state.MLIR with SSACounter = state.MLIR.SSACounter + 1 }
            let state' = { state with MLIR = mlirState' }
            Success(name, state')
    
    /// Emit text with current indentation
    let emit (text: string): MLIRCombinator<unit> =
        fun state ->
            let indent = String.replicate state.MLIR.Indent "  "
            state.MLIR.Output.Append(indent).Append(text).AppendLine() |> ignore
            Success((), state)
    
    /// Emit raw text without indentation
    let emitRaw (text: string): MLIRCombinator<unit> =
        fun state ->
            state.MLIR.Output.Append(text) |> ignore
            Success((), state)
    
    /// Increase indentation for nested scope
    let indented (combinator: MLIRCombinator<'T>): MLIRCombinator<'T> =
        MLIRCombinators.updateMLIRState (fun s -> { s with MLIR = { s.MLIR with Indent = s.MLIR.Indent + 1 } }) >>
        combinator >>= fun result ->
        MLIRCombinators.updateMLIRState (fun s -> { s with MLIR = { s.MLIR with Indent = s.MLIR.Indent - 1 } }) >>
        MLIRCombinators.lift result
    
    /// Add external dependency
    let requireExternal (symbol: string): MLIRCombinator<unit> =
        MLIRCombinators.updateMLIRState (fun s -> 
            { s with MLIR = { s.MLIR with RequiredExternals = Set.add symbol s.MLIR.RequiredExternals } })
    
    /// Bind local variable
    let bindLocal (name: string) (ssa: string) (mlirType: string): MLIRCombinator<unit> =
        MLIRCombinators.updateMLIRState (fun s -> 
            { s with MLIR = { s.MLIR with LocalVars = Map.add name (ssa, mlirType) s.MLIR.LocalVars } })
    
    /// Look up local variable
    let lookupLocal (name: string): MLIRCombinator<(string * string) option> =
        MLIRCombinators.getMLIRState |>> (fun state -> Map.tryFind name state.MLIR.LocalVars)

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

/// Creates initial MLIR state
let initialMLIRState : MLIRState = {
    Output = StringBuilder()
    Indent = 1
    SSACounter = 0
    LocalVars = Map.empty
    RequiredExternals = Set.empty
    CurrentFunction = None
    GeneratedFunctions = Set.empty
    CurrentModule = []
    HasErrors = false
}

/// Creates combined initial state
let initialMLIRBuilderState (fileName: string) : MLIRBuilderState = {
    Firefly = initialState fileName
    MLIR = initialMLIRState
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

/// Runs an MLIR combinator and extracts the result
let runMLIRCombinator (combinator: MLIRCombinator<'T>) (initialState: MLIRBuilderState): CompilerResult<'T * string> =
    match combinator initialState with
    | Success(result, finalState) -> 
        if finalState.MLIR.HasErrors then
            CompilerFailure [InternalError("MLIR Generation", "Compilation completed with errors", None)]
        else
            Success(result, finalState.MLIR.Output.ToString())
    | CompilerFailure errors -> CompilerFailure errors

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

/// MLIR computation expression builder
type MLIRBuilder() =
    member inline _.Bind(c, f) = MLIRCombinators.(>>=) c f
    member inline _.Return(x) = MLIRCombinators.lift x
    member inline _.ReturnFrom(c) = c
    member inline _.Zero() = MLIRCombinators.lift ()
    member inline _.Delay(f) = fun state -> (f()) state
    
    member inline _.Combine(c1, c2) = 
        MLIRCombinators.(>>) c1 c2
        
    member inline _.TryWith(c, handler) =
        fun state ->
            try
                c state
            with e ->
                (handler e) state
                
    member inline _.TryFinally(c, compensation) =
        fun state ->
            try
                c state
            finally
                compensation()

/// Parser computation expression
let parser = ParserBuilder()

/// MLIR computation expression
let mlir = MLIRBuilder()