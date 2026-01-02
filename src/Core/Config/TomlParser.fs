/// XParsec-based TOML Parser for .fidproj files
/// Supports the subset of TOML needed for Firefly project configuration.
module Core.Config.TomlParser

open System
open XParsec
open XParsec.Parsers
open XParsec.CharParsers

// ═══════════════════════════════════════════════════════════════════════════
// TOML Value Types
// ═══════════════════════════════════════════════════════════════════════════

type TomlValue =
    | TomlString of string
    | TomlInt of int64
    | TomlFloat of float
    | TomlBool of bool
    | TomlArray of TomlValue list
    | TomlTable of Map<string, TomlValue>
    | TomlInlineTable of Map<string, TomlValue>

type TomlDocument = Map<string, TomlValue>

// ═══════════════════════════════════════════════════════════════════════════
// Basic Parsers
// ═══════════════════════════════════════════════════════════════════════════

let inline isWhitespaceNotNewline c = c = ' ' || c = '\t'
let inline isNewline c = c = '\n' || c = '\r'
let inline isBareKeyChar c =
    (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
    (c >= '0' && c <= '9') || c = '_' || c = '-'

// Whitespace (not including newlines)
let ws = skipMany (satisfy isWhitespaceNotNewline)

// Whitespace including newlines
let wsNl = skipMany (satisfy (fun c -> isWhitespaceNotNewline c || isNewline c))

// Comment: # to end of line
let comment = pchar '#' >>. skipMany (satisfy (fun c -> not (isNewline c)))

// Optional whitespace, optional comment, optional newline
let lineEnd = ws >>. optional comment >>. optional (satisfy isNewline)

// ═══════════════════════════════════════════════════════════════════════════
// String Parsers
// ═══════════════════════════════════════════════════════════════════════════

let escapedChar =
    pchar '\\' >>. (
        (pchar 'n' >>% '\n') <|>
        (pchar 't' >>% '\t') <|>
        (pchar 'r' >>% '\r') <|>
        (pchar '\\' >>% '\\') <|>
        (pchar '"' >>% '"')
    )

let stringChar =
    (satisfy (fun c -> c <> '"' && c <> '\\' && not (isNewline c))) <|> escapedChar

let quotedString =
    pchar '"' >>. manyChars stringChar .>> pchar '"'

let bareKey =
    many1Chars (satisfy isBareKeyChar)

let key = quotedString <|> bareKey

// ═══════════════════════════════════════════════════════════════════════════
// Value Parsers
// ═══════════════════════════════════════════════════════════════════════════

// Forward reference for recursive value parsing
let tomlValueRef = RefParser<TomlValue, char, unit, ReadableString, ReadableStringSlice>()

let tomlValue reader = tomlValueRef.Parser reader

let tomlString = quotedString |>> TomlString

let tomlInt = pint64 |>> TomlInt

let tomlFloat = pfloat |>> TomlFloat

let tomlBool =
    (pstring "true" >>% TomlBool true) <|> (pstring "false" >>% TomlBool false)

// Array: [ value, value, ... ]
let arrayElements reader =
    let struct (items, _) = sepBy (ws >>. tomlValue .>> ws) (pchar ',') reader |> function
        | Ok result -> result.Parsed
        | Error _ -> struct (System.Collections.Immutable.ImmutableArray.Empty, System.Collections.Immutable.ImmutableArray.Empty)
    preturn (items |> Seq.toList) reader

let tomlArray =
    pchar '[' >>. wsNl >>. arrayElements .>> wsNl .>> pchar ']'
    |>> TomlArray

// Inline table entry: key = value
let inlineTableEntry =
    ws >>. key .>> ws .>> pchar '=' .>> ws .>>. tomlValue

// Inline table: { key = value, key = value }
let inlineTableEntries reader =
    let struct (items, _) = sepBy inlineTableEntry (ws >>. pchar ',' >>. ws) reader |> function
        | Ok result -> result.Parsed
        | Error _ -> struct (System.Collections.Immutable.ImmutableArray.Empty, System.Collections.Immutable.ImmutableArray.Empty)
    preturn (items |> Seq.toList) reader

let tomlInlineTable =
    pchar '{' >>. ws >>. inlineTableEntries .>> ws .>> pchar '}'
    |>> (fun entries ->
        entries
        |> List.map (fun struct (k, v) -> (k, v))
        |> Map.ofList
        |> TomlInlineTable)

// Set up the value parser - order matters for ambiguous parses
do tomlValueRef.Set(
    choice [
        tomlBool      // Must come before bare identifiers
        tomlFloat     // Must come before int (has decimal point)
        tomlInt
        tomlString
        tomlArray
        tomlInlineTable
    ]
)

// ═══════════════════════════════════════════════════════════════════════════
// Document Parsers
// ═══════════════════════════════════════════════════════════════════════════

// Key-value pair: key = value
let keyValue =
    ws >>. key .>> ws .>> pchar '=' .>> ws .>>. tomlValue .>> lineEnd

// Section header: [section] or [section.subsection]
let sectionNameParts reader =
    let struct (items, _) = sepBy1 bareKey (pchar '.') reader |> function
        | Ok result -> result.Parsed
        | Error e -> raise (Exception(sprintf "Section name parse error"))
    preturn (items |> Seq.toList) reader

let sectionHeader =
    pchar '[' >>. sectionNameParts .>> pchar ']' .>> lineEnd
    |>> (String.concat ".")

// Skip blank lines and comments
let skipBlanks = skipMany (ws >>. (comment <|> (satisfy isNewline >>% ())) )

// Parse entire document
let parseDocument reader =
    let rec loop currentSection (acc: Map<string, TomlValue>) reader =
        match skipBlanks reader with
        | Ok _ ->
            // Try section header
            let pos = reader.Position
            match sectionHeader reader with
            | Ok result ->
                loop result.Parsed acc reader
            | Error _ ->
                reader.Position <- pos
                // Try key-value
                match keyValue reader with
                | Ok result ->
                    let struct (k, v) = result.Parsed
                    let fullKey = if currentSection = "" then k else currentSection + "." + k
                    loop currentSection (Map.add fullKey v acc) reader
                | Error _ ->
                    reader.Position <- pos
                    // Check if at end
                    if reader.AtEnd then
                        preturn acc reader
                    else
                        // Skip one character and try again (for robustness with trailing content)
                        if not reader.AtEnd then
                            preturn acc reader
                        else
                            preturn acc reader
        | Error e -> Error e

    match loop "" Map.empty reader with
    | Ok result ->
        // Now check for EOF
        match eof reader with
        | Ok _ -> preturn result.Parsed reader
        | Error _ -> preturn result.Parsed reader  // Accept even without EOF
    | Error e -> Error e

// ═══════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════

/// Parse a TOML string and return the document
let parse (input: string) : Result<TomlDocument, string> =
    let reader = Reader.ofString input ()
    match parseDocument reader with
    | Ok result -> Ok result.Parsed
    | Error e ->
        let pos = e.Position
        Error (sprintf "TOML parse error at index %d" pos.Index)

/// Get a string value from the document
let getString (key: string) (doc: TomlDocument) : string option =
    match Map.tryFind key doc with
    | Some (TomlString s) -> Some s
    | _ -> None

/// Get a string list from an array value
let getStringList (key: string) (doc: TomlDocument) : string list option =
    match Map.tryFind key doc with
    | Some (TomlArray arr) ->
        arr |> List.choose (function TomlString s -> Some s | _ -> None) |> Some
    | _ -> None

/// Get a nested value from an inline table (e.g., "dependencies.alloy" -> { path = "..." })
let getInlineTable (key: string) (doc: TomlDocument) : Map<string, TomlValue> option =
    match Map.tryFind key doc with
    | Some (TomlInlineTable t) -> Some t
    | _ -> None

/// Get path from an inline table like { path = "/some/path" }
let getPathFromInlineTable (key: string) (doc: TomlDocument) : string option =
    match getInlineTable key doc with
    | Some table ->
        match Map.tryFind "path" table with
        | Some (TomlString s) -> Some s
        | _ -> None
    | None -> None
