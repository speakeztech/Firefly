module Firefly.Core.XParsec.Foundation

open System
open XParsec
open Core.XParsec.FoundationHelpers

// ======================================
// Type definitions for Firefly parsers
// ======================================

/// Our custom parser state for Firefly
type FireflyParserState = {
    Position: int
    Input: string
    IndentLevel: int
    ErrorMessages: string list
    Metadata: Map<string, obj>
}

/// Reply type for parser results
type Reply<'T> = 
    | Success of 'T * FireflyParserState
    | Failure of string * FireflyParserState

/// Define our own Parser type that works with FireflyParserState
type Parser<'T> = FireflyParserState -> Reply<'T>

// ======================================
// Basic parser creation functions
// ======================================

/// Creates a parser that always succeeds with the given value
let succeed (value: 'T) : Parser<'T> =
    fun state -> Success(value, state)

/// Creates a parser that always fails with the given message
let fail (message: string) : Parser<'T> =
    fun state -> Failure(message, state)

// ======================================
// Parser Combinators
// ======================================

/// Monadic bind operator
let bind (p: Parser<'a>) (f: 'a -> Parser<'b>) : Parser<'b> =
    fun state ->
        match p state with
        | Success(value, newState) -> f value newState
        | Failure(msg, newState) -> Failure(msg, newState)

/// Infix version of bind
let (>>=) p f = bind p f

/// Maps the result of a parser
let map (f: 'a -> 'b) (p: Parser<'a>) : Parser<'b> =
    p >>= (fun result -> succeed (f result))

/// Infix version of map
let (|>>) p f = map f p

/// Applies a parser and returns its result, then applies another parser and discards its result
let (.>>) (p1: Parser<'a>) (p2: Parser<'b>) : Parser<'a> =
    p1 >>= (fun result1 ->
        p2 >>= (fun _ ->
            succeed result1))

/// Applies a parser and discards its result, then applies another parser and returns its result
let (>>.) (p1: Parser<'a>) (p2: Parser<'b>) : Parser<'b> =
    p1 >>= (fun _ ->
        p2)

/// Applies two parsers in sequence and returns both results as a tuple
let (.>>.) (p1: Parser<'a>) (p2: Parser<'b>) : Parser<'a * 'b> =
    p1 >>= (fun result1 ->
        p2 >>= (fun result2 ->
            succeed (result1, result2)))

/// Tries the first parser, if it fails, tries the second parser
let (<|>) (p1: Parser<'a>) (p2: Parser<'a>) : Parser<'a> =
    fun state ->
        match p1 state with
        | Success(result, newState) -> Success(result, newState)
        | Failure(_, _) -> p2 state

/// Tries multiple parsers in order, returning the result of the first one that succeeds
let choice (parsers: Parser<'a> list) : Parser<'a> =
    List.fold (<|>) (fail "No parsers succeeded") parsers

/// Applies a parser and wraps its result in Some, or returns None if the parser fails
let opt (p: Parser<'a>) : Parser<'a option> =
    fun state ->
        match p state with
        | Success(value, newState) -> Success(Some value, newState)
        | Failure(_, _) -> Success(None, state)

/// Applies a parser that might fail without consuming input
let attempt (p: Parser<'a>) : Parser<'a> =
    fun state ->
        match p state with
        | Success(result, newState) -> Success(result, newState)
        | Failure(msg, _) -> Failure(msg, state)  // Return original state on failure

/// Applies a parser zero or more times, collecting the results in a list
let rec many (p: Parser<'a>) : Parser<'a list> =
    fun state ->
        match p state with
        | Success(first, newState) ->
            match many p newState with
            | Success(rest, finalState) -> Success(first :: rest, finalState)
            | Failure(_, _) -> Success([first], newState)
        | Failure(_, _) -> Success([], state)

/// Applies a parser one or more times, collecting the results in a list
let many1 (p: Parser<'a>) : Parser<'a list> =
    p >>= (fun head ->
        many p >>= (fun tail ->
            succeed (head :: tail)))

/// Applies a parser between two other parsers
let between (pOpen: Parser<'a>) (pClose: Parser<'b>) (p: Parser<'c>) : Parser<'c> =
    pOpen >>. p .>> pClose

/// Applies a parser and a separator parser alternately, collecting the results in a list
let sepBy (p: Parser<'a>) (sep: Parser<'b>) : Parser<'a list> =
    fun state ->
        match p state with
        | Success(first, state1) ->
            let rec sepByRest acc state =
                match sep state with
                | Success(_, state2) ->
                    match p state2 with
                    | Success(item, state3) -> sepByRest (item :: acc) state3
                    | Failure(_, _) -> Success(acc, state) // Separator succeeded but item failed
                | Failure(_, _) -> Success(acc, state) // No separator
            
            match sepByRest [first] state1 with
            | Success(items, finalState) -> Success(List.rev items, finalState)
            | Failure(msg, failState) -> Failure(msg, failState)
        | Failure(_, _) -> Success([], state) // No first item

/// Applies a parser and a separator parser alternately at least once, collecting the results in a list
let sepBy1 (p: Parser<'a>) (sep: Parser<'b>) : Parser<'a list> =
    p >>= (fun head ->
        many (sep >>. p) >>= (fun tail ->
            succeed (head :: tail)))

/// Adds an error message context to a parser
let withErrorContext (context: string) (p: Parser<'a>) : Parser<'a> =
    fun state ->
        match p state with
        | Success(result, newState) -> Success(result, newState)
        | Failure(msg, newState) -> Failure(sprintf "%s: %s" context msg, newState)

// ======================================
// Character and String Parsers
// ======================================

/// Parses a specific character
let pchar (c: char) : Parser<char> =
    fun state ->
        if state.Position >= state.Input.Length then
            Failure(sprintf "End of input, expected '%c'" c, state)
        else
            let currentChar = state.Input.[state.Position]
            if currentChar = c then
                Success(c, { state with Position = state.Position + 1 })
            else
                Failure(sprintf "Expected '%c', found '%c'" c currentChar, state)

/// Parses a specific string
let pstring (s: string) : Parser<string> =
    fun state ->
        let len = s.Length
        let available = state.Input.Length - state.Position
        
        if available < len then
            Failure(sprintf "End of input, expected '%s'" s, state)
        else
            let substring = state.Input.Substring(state.Position, len)
            if substring = s then
                Success(s, { state with Position = state.Position + len })
            else
                Failure(sprintf "Expected '%s', found '%s'" s substring, state)

/// Parses any character that satisfies the predicate
let satisfy (predicate: char -> bool) : Parser<char> =
    fun state ->
        if state.Position >= state.Input.Length then
            Failure("End of input", state)
        else
            let currentChar = state.Input.[state.Position]
            if predicate currentChar then
                Success(currentChar, { state with Position = state.Position + 1 })
            else
                Failure(sprintf "Character '%c' didn't satisfy predicate" currentChar, state)

/// Parses any character
let anyChar : Parser<char> =
    fun state ->
        if state.Position >= state.Input.Length then
            Failure("End of input", state)
        else
            let currentChar = state.Input.[state.Position]
            Success(currentChar, { state with Position = state.Position + 1 })

/// Parses a whitespace character
let whitespace : Parser<char> =
    satisfy Char.IsWhiteSpace

/// Parses zero or more whitespace characters
let spaces : Parser<unit> =
    many whitespace |>> ignore

/// Parses one or more whitespace characters
let spaces1 : Parser<unit> =
    many1 whitespace |>> ignore

/// Parses a digit character
let digit : Parser<char> =
    satisfy Char.IsDigit

/// Parses a letter character
let letter : Parser<char> =
    satisfy Char.IsLetter

/// Parses a letter or digit character
let letterOrDigit : Parser<char> =
    satisfy Char.IsLetterOrDigit

/// Parses an integer
let pint : Parser<int> =
    let isDigit c = c >= '0' && c <= '9'
    let isSign c = c = '-' || c = '+'
    
    let digits = many1 (satisfy isDigit) |>> (fun chars -> String.Concat(chars))
    let optSign = opt (satisfy isSign) |>> (function Some '-' -> "-" | _ -> "")
    
    optSign .>>. digits
    |>> (fun (sign, digits) -> int (sign + digits))

// ======================================
// F# Specific Parsers
// ======================================

/// Parses an F# identifier
let identifier : Parser<string> =
    let isIdentifierFirstChar c = Char.IsLetter c || c = '_'
    let isIdentifierChar c = Char.IsLetterOrDigit c || c = '_'
    
    satisfy isIdentifierFirstChar .>>. many (satisfy isIdentifierChar)
    |>> (fun (first, rest) -> String(first :: rest |> List.toArray))

/// Parses an F# keyword
let keyword (kw: string) : Parser<string> =
    let notFollowedBy (p: Parser<'a>) : Parser<unit> =
        fun state ->
            match p state with
            | Success(_, _) -> Failure("notFollowedBy", state)
            | Failure(_, _) -> Success((), state)
            
    pstring kw .>> notFollowedBy letterOrDigit .>> spaces

// ======================================
// Running Parsers
// ======================================

/// Creates an initial parser state
let createInitialState (input: string) : FireflyParserState =
    {
        Position = 0
        Input = input
        IndentLevel = 0
        ErrorMessages = []
        Metadata = Map.empty
    }

/// Runs a parser on the given input string
let runParser (parser: Parser<'a>) (input: string) : Result<'a, string> =
    let initialState = createInitialState input
    
    match parser initialState with
    | Success(result, _) -> Ok result
    | Failure(msg, _) -> Error msg

/// Runs a parser on the given input string and throws an exception if it fails
let runParserOrThrow (parser: Parser<'a>) (input: string) : 'a =
    match runParser parser input with
    | Ok result -> result
    | Error msg -> failwith msg

// ======================================
// Utility Functions and Combinators
// ======================================

/// Parses a value that might be in parentheses
let parens (p: Parser<'a>) : Parser<'a> =
    between (pchar '(' >>. spaces) (spaces >>. pchar ')') p

/// Parses a value that might be in braces
let braces (p: Parser<'a>) : Parser<'a> =
    between (pchar '{' >>. spaces) (spaces >>. pchar '}') p

/// Parses a value that might be in brackets
let brackets (p: Parser<'a>) : Parser<'a> =
    between (pchar '[' >>. spaces) (spaces >>. pchar ']') p

/// Parses a value that might be in angle brackets
let angleBrackets (p: Parser<'a>) : Parser<'a> =
    between (pchar '<' >>. spaces) (spaces >>. pchar '>') p

/// Parser that succeeds if the next character is not in the given list
let noneOf (chars: char list) : Parser<char> =
    satisfy (fun c -> not (List.contains c chars))

/// Parser that succeeds if the next character is in the given list
let oneOf (chars: char list) : Parser<char> =
    satisfy (fun c -> List.contains c chars)

/// Parser that fails if the next character would be parsed by the given parser
let notFollowedBy (p: Parser<'a>) : Parser<unit> =
    fun state ->
        match p state with
        | Success(_, _) -> Failure("notFollowedBy", state)
        | Failure(_, _) -> Success((), state)

/// Parses the end of input
let eof : Parser<unit> =
    fun state ->
        if state.Position >= state.Input.Length then
            Success((), state)
        else
            Failure(sprintf "Expected end of input, found '%c'" state.Input.[state.Position], state)

/// Parses the given parser followed by spaces
let lexeme (p: Parser<'a>) : Parser<'a> =
    p .>> spaces

/// Applies a parser and returns the specified value
let preturn (x: 'a) (p: Parser<'b>) : Parser<'a> =
    p |>> (fun _ -> x)

/// Applies the parser and returns the specified value if it succeeds
let value (x: 'a) (p: Parser<'b>) : Parser<'a> =
    p |>> (fun _ -> x)

/// Applies a parser repeatedly until the second parser succeeds
let manyTill (p: Parser<'a>) (endp: Parser<'b>) : Parser<'a list> =
    let rec manyTillHelper acc state =
        match endp state with
        | Success(_, newState) -> Success(List.rev acc, newState)
        | Failure(_, _) ->
            match p state with
            | Success(x, newState) -> manyTillHelper (x :: acc) newState
            | Failure(msg, newState) -> Failure(msg, newState)
    fun state -> manyTillHelper [] state

/// Parses a string enclosed in double quotes
let quotedString : Parser<string> =
    let escape = pstring "\\" >>. (pstring "\"" <|> pstring "\\" <|> pstring "n" <|> pstring "r" <|> pstring "t")
                |>> function 
                    | "\"" -> "\""
                    | "\\" -> "\\"
                    | "n" -> "\n"
                    | "r" -> "\r"
                    | "t" -> "\t"
                    | _ -> failwith "Impossible case in escape parser"
    
    let nonEscape = noneOf ['"'; '\\'] |>> string
    
    between (pchar '"') (pchar '"') 
            (many (escape <|> nonEscape) 
             |>> String.concat "")

// ======================================
// State Manipulation
// ======================================

/// Gets the current state
let getState : Parser<FireflyParserState> =
    fun state -> Success(state, state)

/// Sets the current state
let setState (newState: FireflyParserState) : Parser<unit> =
    fun _ -> Success((), newState)

/// Updates the current state
let updateState (f: FireflyParserState -> FireflyParserState) : Parser<unit> =
    fun state -> Success((), f state)

/// Increments the indentation level
let indentMore : Parser<unit> =
    updateState (fun state -> { state with IndentLevel = state.IndentLevel + 1 })

/// Decrements the indentation level
let indentLess : Parser<unit> =
    updateState (fun state -> { state with IndentLevel = state.IndentLevel - 1 })

/// Runs a parser with an increased indentation level
let indented (p: Parser<'a>) : Parser<'a> =
    indentMore >>. p .>> indentLess

/// Adds metadata to the parser state
let addMetadata (key: string) (value: obj) : Parser<unit> =
    updateState (fun state -> 
        { state with Metadata = Map.add key value state.Metadata })

/// Gets metadata from the parser state
let getMetadata (key: string) : Parser<obj option> =
    getState |>> (fun state -> 
        state.Metadata |> Map.tryFind key)

// ======================================
// XParsec Integration - Add as needed
// ======================================

// If you need to integrate with specific XParsec functionality, add wrapper functions here