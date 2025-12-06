/// PSGXParsec - Bridge between PSGZipper tree traversal and XParsec sequential parsing
///
/// ARCHITECTURAL PRINCIPLE:
/// The zipper navigates the PSG tree (up/down/left/right) - this is tree complexity.
/// XParsec parses the sequential list of children at each zipper position - this is parsing complexity.
/// Good fences make good neighbors: keep these concerns separate.
///
/// XParsec state is MINIMAL: just an ID for context and mutable emission state.
/// The full zipper stays outside XParsec - passed in when running parsers, results extracted after.
///
/// This module provides:
/// - PSGChildSlice: IReadable implementation for PSG node children
/// - PSGParseState: Minimal state for XParsec (supports equality)
/// - PSGChildParser: Type alias for XParsec parser over PSG children
/// - Bridge functions to run XParsec parsers on zipper children
module Alex.Traversal.PSGXParsec

open System
open XParsec
open Core.PSG.Types
open Alex.Traversal.PSGZipper

// ═══════════════════════════════════════════════════════════════════
// PSGChildSlice - IReadable implementation for PSG node children
// ═══════════════════════════════════════════════════════════════════

/// A readable slice of PSG node children for XParsec consumption
[<Struct>]
type PSGChildSlice(nodes: PSGNode array, start: int, length: int) =

    member _.Nodes = nodes
    member _.Start = start
    member _.SliceLength = length

    interface IReadable<PSGNode, PSGChildSlice> with
        member _.Item
            with get index = nodes.[start + int index]

        member _.TryItem(index: int64) =
            let i = start + int index
            if i >= start && i < start + length then
                ValueSome nodes.[i]
            else
                ValueNone

        member _.Length = int64 length

        member _.Slice(newStart: int64, newLength: int64) =
            PSGChildSlice(nodes, start + int newStart, int newLength)

        member _.SpanSlice(index: int64, count: int) =
            ReadOnlySpan(nodes, start + int index, count)

/// Create a PSGChildSlice from an array of nodes
let createSlice (nodes: PSGNode array) : PSGChildSlice =
    PSGChildSlice(nodes, 0, nodes.Length)

/// Create a PSGChildSlice from a list of nodes
let createSliceFromList (nodes: PSGNode list) : PSGChildSlice =
    let arr = Array.ofList nodes
    PSGChildSlice(arr, 0, arr.Length)

// ═══════════════════════════════════════════════════════════════════
// PSGParseState - Minimal state for XParsec (supports equality)
// ═══════════════════════════════════════════════════════════════════

/// Minimal state carried through XParsec parsing.
/// Does NOT contain the full zipper - that stays outside XParsec.
/// This type supports equality for XParsec's backtracking/infinite loop detection.
[<Struct>]
type PSGParseState = {
    /// SSA counter for generating unique names during child parsing
    SSACounter: int
    /// Current focus node ID (for context, not navigation)
    FocusNodeId: string
}

module PSGParseState =
    /// Create initial parse state
    let create (focusNodeId: string) (ssaCounter: int) : PSGParseState =
        { SSACounter = ssaCounter; FocusNodeId = focusNodeId }

    /// Create from a zipper (extracts minimal state)
    let fromZipper (zipper: PSGZipper) : PSGParseState =
        { SSACounter = zipper.State.SSACounter
          FocusNodeId = zipper.Focus.Id.Value }

    /// Increment SSA counter
    let nextSSA (state: PSGParseState) : PSGParseState * string =
        let name = sprintf "%%v%d" state.SSACounter
        { state with SSACounter = state.SSACounter + 1 }, name

// ═══════════════════════════════════════════════════════════════════
// PSGChildParser - XParsec parser type for PSG children
// ═══════════════════════════════════════════════════════════════════

/// XParsec parser specialized for parsing PSG node children
type PSGChildParser<'T> = Parser<'T, PSGNode, PSGParseState, PSGChildSlice, PSGChildSlice>

/// Result of running a child parser, including updated state
type ChildParseResult<'T> = {
    /// The parsed value
    Value: 'T
    /// Updated SSA counter (to merge back into zipper)
    SSACounter: int
}

// ═══════════════════════════════════════════════════════════════════
// Bridge Functions - Running XParsec on Zipper Children
// ═══════════════════════════════════════════════════════════════════

/// Run an XParsec parser on the children of the current zipper focus.
/// Returns the parsed value and the updated SSA counter.
let parseChildren (parser: PSGChildParser<'T>) (zipper: PSGZipper) : Result<ChildParseResult<'T>, string> =
    let children = PSGZipper.childNodes zipper |> Array.ofList
    let input = PSGChildSlice(children, 0, children.Length)
    let state = PSGParseState.fromZipper zipper
    let reader = Reader(input, state, 0L)

    match parser reader with
    | Ok success ->
        Ok { Value = success.Parsed
             SSACounter = reader.State.SSACounter }
    | Error err ->
        let msg =
            match err.Errors with
            | Message m -> m
            | Expected t -> sprintf "Expected: %A" t
            | Unexpected t -> sprintf "Unexpected: %A" t
            | EndOfInput -> "Unexpected end of input"
            | _ -> sprintf "Parse error at position %d" err.Position.Index
        Error msg

/// Run an XParsec parser on the children, returning Option instead of Result
let tryParseChildren (parser: PSGChildParser<'T>) (zipper: PSGZipper) : ChildParseResult<'T> option =
    match parseChildren parser zipper with
    | Ok result -> Some result
    | Error _ -> None

/// Run an XParsec parser on a specific list of nodes
let parseNodes (parser: PSGChildParser<'T>) (nodes: PSGNode list) (zipper: PSGZipper) : Result<ChildParseResult<'T>, string> =
    let input = createSliceFromList nodes
    let state = PSGParseState.fromZipper zipper
    let reader = Reader(input, state, 0L)

    match parser reader with
    | Ok success ->
        Ok { Value = success.Parsed
             SSACounter = reader.State.SSACounter }
    | Error err ->
        Error (sprintf "Parse error at position %d" err.Position.Index)

/// Merge child parse result back into zipper state
let mergeResult (result: ChildParseResult<'T>) (zipper: PSGZipper) : PSGZipper * 'T =
    let newState = { zipper.State with SSACounter = result.SSACounter }
    { zipper with State = newState }, result.Value

// ═══════════════════════════════════════════════════════════════════
// Basic Child Parsers - Building blocks for pattern matching
// ═══════════════════════════════════════════════════════════════════

/// Match if current child satisfies predicate
let satisfyChild (pred: PSGNode -> bool) : PSGChildParser<PSGNode> =
    Parsers.satisfy pred

/// Match any child (always succeeds if there's a child)
let anyChild : PSGChildParser<PSGNode> =
    satisfyChild (fun _ -> true)

/// Match child with specific syntax kind (exact match)
let childSyntaxKind (kind: string) : PSGChildParser<PSGNode> =
    satisfyChild (fun n -> n.SyntaxKind = kind)

/// Match child with syntax kind starting with prefix
let childKindPrefix (prefix: string) : PSGChildParser<PSGNode> =
    satisfyChild (fun n -> n.SyntaxKind.StartsWith(prefix))

/// Match child with specific symbol full name
let childSymbolName (name: string) : PSGChildParser<PSGNode> =
    satisfyChild (fun n ->
        match n.Symbol with
        | Some s -> s.FullName = name
        | None -> false)

/// Match child with symbol display name
let childSymbolDisplayName (name: string) : PSGChildParser<PSGNode> =
    satisfyChild (fun n ->
        match n.Symbol with
        | Some s -> s.DisplayName = name
        | None -> false)

/// Match child with symbol name containing substring
let childSymbolContains (substring: string) : PSGChildParser<PSGNode> =
    satisfyChild (fun n ->
        match n.Symbol with
        | Some s -> s.FullName.Contains(substring) || s.DisplayName.Contains(substring)
        | None -> false)

/// Match child with symbol name ending with suffix
let childSymbolEndsWith (suffix: string) : PSGChildParser<PSGNode> =
    satisfyChild (fun n ->
        match n.Symbol with
        | Some s -> s.FullName.EndsWith(suffix) || s.DisplayName.EndsWith(suffix)
        | None -> false)

// ═══════════════════════════════════════════════════════════════════
// Convenience Parsers - Common patterns
// ═══════════════════════════════════════════════════════════════════

/// Skip current child and return unit
let skipChild : PSGChildParser<unit> =
    fun reader ->
        match anyChild reader with
        | Ok _ -> Parsers.preturn () reader
        | Error e -> Error e

/// Get all remaining children
/// Note: Uses XParsec's many combinator - PSGParseState now supports equality
let remainingChildren : PSGChildParser<PSGNode list> =
    fun reader ->
        match Combinators.many anyChild reader with
        | Ok success -> Parsers.preturn (List.ofSeq success.Parsed) reader
        | Error e -> Error e

/// Match App node child
let appChild : PSGChildParser<PSGNode> =
    childKindPrefix "App"

/// Match Const node child
let constChild : PSGChildParser<PSGNode> =
    childKindPrefix "Const"

/// Match Ident/Value node child
let identChild : PSGChildParser<PSGNode> =
    satisfyChild (fun n ->
        n.SyntaxKind.StartsWith("Ident") ||
        n.SyntaxKind.StartsWith("Value") ||
        n.SyntaxKind.StartsWith("LongIdent"))

/// Match Let binding child
let letChild : PSGChildParser<PSGNode> =
    satisfyChild (fun n ->
        n.SyntaxKind = "LetOrUse" || n.SyntaxKind.StartsWith("Let"))

/// Match Sequential child
let sequentialChild : PSGChildParser<PSGNode> =
    childKindPrefix "Sequential"

/// Match Pattern node child (structural, typically skipped)
let patternChild : PSGChildParser<PSGNode> =
    childKindPrefix "Pattern:"

// ═══════════════════════════════════════════════════════════════════
// State Access Parsers - Minimal state operations within XParsec
// ═══════════════════════════════════════════════════════════════════

/// Get the current focus node ID
let getFocusNodeId : PSGChildParser<string> =
    fun reader -> Parsers.preturn reader.State.FocusNodeId reader

/// Get the current SSA counter value
let getSSACounter : PSGChildParser<int> =
    fun reader -> Parsers.preturn reader.State.SSACounter reader

/// Generate a fresh SSA name (updates state)
let freshSSA : PSGChildParser<string> =
    fun reader ->
        let state, name = PSGParseState.nextSSA reader.State
        reader.State <- state
        Parsers.preturn name reader

/// Generate multiple fresh SSA names
let freshSSAMany (count: int) : PSGChildParser<string list> =
    fun reader ->
        let names =
            [ for _ in 1..count do
                let state, name = PSGParseState.nextSSA reader.State
                reader.State <- state
                yield name ]
        Parsers.preturn names reader
