module Core.XParsec.Foundation

open System
open XParsec

/// Core error types for the Firefly compiler
type FireflyError =
    | ParseError of position: Position * message: string * context: string list
    | TransformError of phase: string * source: string * target: string * message: string
    | SemanticError of construct: string * message: string * location: Position
    | CompilerError of phase: string * message: string * details: string option

/// Position information for error reporting
and Position = {
    Line: int
    Column: int
    File: string
    Offset: int
}

/// Result type for all compiler operations - no fallbacks allowed
type CompilerResult<'T> =
    | Success of 'T
    | CompilerFailure of FireflyError list

/// Parser state for tracking compilation context
type FireflyParserState = {
    CurrentFile: string
    ImportedModules: string list
    TypeDefinitions: Map<string, string>
    ScopeStack: string list list
}

/// Core XParsec combinators specialized for Firefly compilation
module Combinators =
    
    /// Creates a parser that always fails with a compiler error
    let compilerFail<'T> (error: FireflyError) : Parser<'T, FireflyParserState> =
        fun state -> Reply(Error, error.ToString())
    
    /// Creates a parser that succeeds with a value
    let succeed<'T> (value: 'T) : Parser<'T, FireflyParserState> =
        fun state -> Reply(Ok value)
    
    /// Combines two parsers sequentially, requiring both to succeed
    let (.>>.) (p1: Parser<'A, FireflyParserState>) (p2: Parser<'B, FireflyParserState>) : Parser<'A * 'B, FireflyParserState> =
        fun state ->
            match p1 state with
            | Reply(Ok result1, state1) ->
                match p2 state1 with
                | Reply(Ok result2, state2) -> Reply(Ok (result1, result2), state2)
                | Reply(Error, error) -> Reply(Error, error)
            | Reply(Error, error) -> Reply(Error, error)
    
    /// Choice combinator - tries first parser, then second if first fails
    let (<|>) (p1: Parser<'T, FireflyParserState>) (p2: Parser<'T, FireflyParserState>) : Parser<'T, FireflyParserState> =
        fun state ->
            match p1 state with
            | Reply(Ok result, newState) -> Reply(Ok result, newState)
            | Reply(Error, _) -> p2 state
    
    /// Maps a function over a parser result
    let (|>>) (parser: Parser<'A, FireflyParserState>) (f: 'A -> 'B) : Parser<'B, FireflyParserState> =
        fun state ->
            match parser state with
            | Reply(Ok result, newState) -> Reply(Ok (f result), newState)
            | Reply(Error, error) -> Reply(Error, error)
    
    /// Monadic bind for composing parsers
    let (>>=) (parser: Parser<'A, FireflyParserState>) (f: 'A -> Parser<'B, FireflyParserState>) : Parser<'B, FireflyParserState> =
        fun state ->
            match parser state with
            | Reply(Ok result, newState) -> (f result) newState
            | Reply(Error, error) -> Reply(Error, error)
    
    /// Parses many occurrences of a parser (zero or more)
    let many (parser: Parser<'T, FireflyParserState>) : Parser<'T list, FireflyParserState> =
        let rec parseMany acc state =
            match parser state with
            | Reply(Ok result, newState) -> parseMany (result :: acc) newState
            | Reply(Error, _) -> Reply(Ok (List.rev acc), state)
        parseMany []
    
    /// Parses one or more occurrences of a parser
    let many1 (parser: Parser<'T, FireflyParserState>) : Parser<'T list, FireflyParserState> =
        parser >>= fun first ->
        many parser >>= fun rest ->
        succeed (first :: rest)
    
    /// Parses items separated by a separator
    let sepBy (itemParser: Parser<'T, FireflyParserState>) (sepParser: Parser<'Sep, FireflyParserState>) : Parser<'T list, FireflyParserState> =
        let rec parseSeparated acc state =
            match itemParser state with
            | Reply(Ok item, state1) ->
                match sepParser state1 with
                | Reply(Ok _, state2) -> parseSeparated (item :: acc) state2
                | Reply(Error, _) -> Reply(Ok (List.rev (item :: acc)), state1)
            | Reply(Error, _) -> Reply(Ok (List.rev acc), state)
        parseSeparated []
    
    /// Parses items between delimiters
    let between (openParser: Parser<'Open, FireflyParserState>) 
                (closeParser: Parser<'Close, FireflyParserState>) 
                (contentParser: Parser<'T, FireflyParserState>) : Parser<'T, FireflyParserState> =
        openParser >>= fun _ ->
        contentParser >>= fun content ->
        closeParser >>= fun _ ->
        succeed content
    
    /// Optionally parses something
    let opt (parser: Parser<'T, FireflyParserState>) : Parser<'T option, FireflyParserState> =
        (parser |>> Some) <|> succeed None

/// Character-level parsers for F# syntax
module CharParsers =
    
    /// Parses a specific character
    let pchar (c: char) : Parser<char, FireflyParserState> =
        fun state ->
            // Implementation would use XParsec's actual character parsing
            // This is a structural placeholder
            if state.CurrentFile.Length > 0 then
                Reply(Ok c, state)
            else
                Reply(Error, sprintf "Expected '%c'" c)
    
    /// Parses any character in a set
    let anyOf (chars: char list) : Parser<char, FireflyParserState> =
        chars 
        |> List.map pchar 
        |> List.reduce (<|>)
    
    /// Parses a whitespace character
    let whitespace : Parser<char, FireflyParserState> =
        anyOf [' '; '\t'; '\n'; '\r']
    
    /// Parses zero or more whitespace characters
    let ws : Parser<unit, FireflyParserState> =
        many whitespace |>> ignore
    
    /// Parses one or more whitespace characters
    let ws1 : Parser<unit, FireflyParserState> =
        many1 whitespace |>> ignore
    
    /// Parses a letter
    let letter : Parser<char, FireflyParserState> =
        fun state ->
            // Would use XParsec's letter parser
            Reply(Ok 'a', state)  // Placeholder
    
    /// Parses a digit
    let digit : Parser<char, FireflyParserState> =
        anyOf ['0'..'9']
    
    /// Parses an alphanumeric character
    let alphaNum : Parser<char, FireflyParserState> =
        letter <|> digit

/// String-level parsers
module StringParsers =
    
    /// Parses a specific string
    let pstring (s: string) : Parser<string, FireflyParserState> =
        s.ToCharArray()
        |> Array.toList
        |> List.map CharParsers.pchar
        |> List.reduce (Combinators.(.>>.))
        |> Combinators.(|>>) (fun chars -> String(Array.ofList (fst chars :: [])))
    
    /// Parses an identifier
    let identifier : Parser<string, FireflyParserState> =
        CharParsers.letter >>= fun first ->
        Combinators.many (CharParsers.alphaNum <|> CharParsers.pchar '_') >>= fun rest ->
        Combinators.succeed (String(Array.ofList (first :: rest)))
    
    /// Parses a qualified identifier (Module.Name)
    let qualifiedIdentifier : Parser<string list, FireflyParserState> =
        Combinators.sepBy identifier (CharParsers.pchar '.')

/// Error handling utilities
module ErrorHandling =
    
    /// Creates a parse error with position information
    let createParseError (pos: Position) (message: string) (context: string list) : FireflyError =
        ParseError(pos, message, context)
    
    /// Creates a transformation error
    let createTransformError (phase: string) (source: string) (target: string) (message: string) : FireflyError =
        TransformError(phase, source, target, message)
    
    /// Lifts a parser to provide better error context
    let withErrorContext (context: string) (parser: Parser<'T, FireflyParserState>) : Parser<'T, FireflyParserState> =
        fun state ->
            match parser state with
            | Reply(Ok result, newState) -> Reply(Ok result, newState)
            | Reply(Error, originalError) -> 
                let enhancedError = sprintf "%s (in context: %s)" originalError context
                Reply(Error, enhancedError)
    
    /// Requires a parser to succeed or fails with a specific error
    let required (errorMsg: string) (parser: Parser<'T, FireflyParserState>) : Parser<'T, FireflyParserState> =
        fun state ->
            match parser state with
            | Reply(Ok result, newState) -> Reply(Ok result, newState)
            | Reply(Error, _) -> Reply(Error, errorMsg)

/// Initial parser state
let initialState (fileName: string) : FireflyParserState = {
    CurrentFile = fileName
    ImportedModules = []
    TypeDefinitions = Map.empty
    ScopeStack = [[]]
}