module Core.XParsec.FoundationHelpers

open XParsec

/// Helper module with additional parser combinators and utilities
/// built on top of the core Foundation.fs functionality
module Combinators =
    /// Parses zero or more occurrences of a pattern separated by a separator
    let sepByZeroOrMore (p: Parser<'a>) (sep: Parser<'b>) : Parser<'a list> =
        sepBy p sep
    
    /// Runs a parser and saves its result in state with the given key
    let saveInState (key: string) (p: Parser<'a>) : Parser<'a> =
        p >>= (fun result ->
            getState >>= (fun state ->
                let newState = { state with Metadata = Map.add key (box result) state.Metadata }
                setState newState >>= (fun _ ->
                    succeed result
                )
            )
        )
    
    /// Retrieves a value from state by key
    let getFromState<'a> (key: string) : Parser<'a option> =
        getState >>= (fun state ->
            match Map.tryFind key state.Metadata with
            | Some value -> succeed (Some (unbox<'a> value))
            | None -> succeed None
        )
    
    /// Parser that runs another parser multiple times and collects the results
    let repeatParser (count: int) (p: Parser<'a>) : Parser<'a list> =
        let rec repeat n acc =
            if n <= 0 then
                succeed (List.rev acc)
            else
                p >>= (fun result ->
                    repeat (n - 1) (result :: acc)
                )
        repeat count []
    
    /// Executes a parser with a temporary state modification, then restores the original state
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
    
    /// Parser that succeeds only at the end of a line or input
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
    
    /// Applies parser p exactly n times and returns the results
    let pnTimes (n: int) (p: Parser<'a>) : Parser<'a list> =
        let rec loop i acc =
            if i = 0 then
                succeed (List.rev acc)
            else
                p >>= (fun x -> loop (i-1) (x::acc))
        loop n []
    
    /// Skips whitespace, preserving the current state otherwise
    let skipWhitespace : Parser<unit> =
        many whitespace |>> ignore
    
    /// Parser that succeeds only if we're at a specific indentation level
    let atIndentLevel (level: int) : Parser<unit> =
        getState >>= (fun state ->
            if state.IndentLevel = level then
                succeed ()
            else
                fail (sprintf "Expected indentation level %d, got %d" level state.IndentLevel)
        )

module ErrorHandling =
    /// Adds context information to errors
    let withContext (context: string) (p: Parser<'a>) : Parser<'a> =
        fun state ->
            let errorState = { state with ErrorMessages = context :: state.ErrorMessages }
            match p errorState with
            | Success(value, newState) -> Success(value, { newState with ErrorMessages = state.ErrorMessages })
            | Failure(msg, newState) -> Failure(sprintf "%s: %s" context msg, { newState with ErrorMessages = state.ErrorMessages })
    
    /// Catches errors from a parser and returns them as an option
    let tryParse (p: Parser<'a>) : Parser<'a option> =
        fun state ->
            match p state with
            | Success(value, newState) -> Success(Some value, newState)
            | Failure(_, _) -> Success(None, state)
    
    /// Creates a comprehensive error message with position information
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

module DebugTools =
    /// Logs debugging information without affecting the parser state
    let logDebug (message: string) : Parser<unit> =
        fun state ->
            printfn "[DEBUG] %s (Position: %d)" message state.Position
            Success((), state)
    
    /// Logs the current parser state for debugging
    let logState (message: string) : Parser<unit> =
        getState >>= (fun state ->
            printfn "[STATE] %s: Position=%d, IndentLevel=%d, ErrorMessages=%A" 
                    message state.Position state.IndentLevel state.ErrorMessages
            succeed ()
        )
    
    /// Attaches a debug checkpoint to a parser
    let debug (label: string) (p: Parser<'a>) : Parser<'a> =
        logDebug (sprintf "Entering: %s" label) >>= (fun _ ->
            p >>= (fun result ->
                logDebug (sprintf "Exiting: %s (Success)" label) >>= (fun _ ->
                    succeed result
                )
            )
        )

module FileHandling =
    /// Parses a complete file with error handling
    let parseFile (parser: Parser<'a>) (filename: string) : Result<'a, string> =
        try
            let content = System.IO.File.ReadAllText(filename)
            let initialState = createInitialState content
            match parser initialState with
            | Success(result, _) -> Ok result
            | Failure(msg, state) -> Error (ErrorHandling.createErrorMessage msg state)
        with
        | ex -> Error (sprintf "File error: %s" ex.Message)
    
    /// Creates a parser for including another file's content
    let includeFile : Parser<string> =
        let fileNameParser = quotedString <|> (many1 (satisfy (fun c -> c <> ' ' && c <> '\n' && c <> '\r')) |>> (List.map string >> String.concat ""))
        pstring "include" >>. spaces1 >>. fileNameParser >>= (fun filename ->
            try
                let content = System.IO.File.ReadAllText(filename)
                succeed content
            with
            | ex -> fail (sprintf "Failed to include file '%s': %s" filename ex.Message)
        )