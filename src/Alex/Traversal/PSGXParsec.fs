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
open FSharp.Compiler.Symbols
open Core.PSG.Types
open Alex.Traversal.PSGZipper

// ═══════════════════════════════════════════════════════════════════
// Safe Symbol Helpers
// ═══════════════════════════════════════════════════════════════════

/// Safely get a symbol's FullName, handling types like 'unit' that throw exceptions
let private tryGetFullName (sym: FSharpSymbol) : string option =
    try Some sym.FullName with _ -> None

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
// Emission Context - Mutable state for MLIR emission (outside XParsec equality)
// ═══════════════════════════════════════════════════════════════════

open System.Text
open Alex.CodeGeneration.MLIRBuilder

/// Mutable emission context - holds state that shouldn't be in XParsec's equality-based state.
/// Access through reference in PSGParseState.
/// This is a CLASS (reference type) so equality is reference equality, which is fine for XParsec.
[<Sealed>]
type EmitContext(graph: ProgramSemanticGraph) =
    /// The full PSG for edge lookups (def-use resolution)
    member _.Graph = graph
    /// MLIR output builder state
    member val Builder = BuilderState.create () with get
    /// Map from NodeId to emitted SSA value (for variable resolution via def-use edges)
    member val NodeSSA : Map<string, string * string> = Map.empty with get, set
    /// Accumulated string literals: content -> global name
    member val StringLiterals : (string * string) list = [] with get, set
    /// Current function name being emitted
    member val CurrentFunction : string option = None with get, set
    /// SSA counter for unique value names
    member val SSACounter : int = 0 with get, set
    /// Label counter for unique block labels
    member val LabelCounter : int = 0 with get, set
    /// External function declarations needed (e.g., "strlen")
    member val ExternalFuncs : Set<string> = Set.empty with get, set
    /// Accumulated errors during emission
    member val Errors : string list = [] with get, set

module EmitContext =
    /// Create emission context from a PSG
    let create (graph: ProgramSemanticGraph) : EmitContext =
        EmitContext(graph)

    /// Record that a node was emitted with a given SSA value
    let recordNodeSSA (ctx: EmitContext) (nodeId: NodeId) (ssa: string) (mlirType: string) : unit =
        ctx.NodeSSA <- Map.add nodeId.Value (ssa, mlirType) ctx.NodeSSA

    /// Look up SSA value for a node by its ID
    let lookupNodeSSA (ctx: EmitContext) (nodeId: NodeId) : (string * string) option =
        Map.tryFind nodeId.Value ctx.NodeSSA

    /// Register a string literal, returning its global name
    let registerStringLiteral (ctx: EmitContext) (content: string) : string =
        match ctx.StringLiterals |> List.tryFind (fun (c, _) -> c = content) with
        | Some (_, name) -> name
        | None ->
            let name = sprintf "@str%d" (List.length ctx.StringLiterals)
            ctx.StringLiterals <- (content, name) :: ctx.StringLiterals
            name

    /// Emit a line of MLIR
    let emitLine (ctx: EmitContext) (text: string) : unit =
        BuilderState.emit ctx.Builder text

    /// Get output string
    let getOutput (ctx: EmitContext) : string =
        ctx.Builder.Output.ToString()

    /// Generate next SSA name and increment counter
    let nextSSA (ctx: EmitContext) : string =
        let name = sprintf "%%v%d" ctx.SSACounter
        ctx.SSACounter <- ctx.SSACounter + 1
        name

    /// Generate next label name and increment counter
    let nextLabel (ctx: EmitContext) : string =
        let name = sprintf "^bb%d" ctx.LabelCounter
        ctx.LabelCounter <- ctx.LabelCounter + 1
        name

    /// Register an external function declaration
    let requireExternalFunc (ctx: EmitContext) (name: string) : unit =
        ctx.ExternalFuncs <- Set.add name ctx.ExternalFuncs

    /// Emit a formatted line of MLIR
    let emitLinef (ctx: EmitContext) fmt =
        Printf.ksprintf (fun s -> emitLine ctx s) fmt

    /// Record an error during emission
    let recordError (ctx: EmitContext) (msg: string) : unit =
        ctx.Errors <- msg :: ctx.Errors

    /// Get all recorded errors
    let getErrors (ctx: EmitContext) : string list =
        ctx.Errors |> List.rev

    /// Check if any errors were recorded
    let hasErrors (ctx: EmitContext) : bool =
        not (List.isEmpty ctx.Errors)

// ═══════════════════════════════════════════════════════════════════
// PSGParseState - Minimal state for XParsec (supports equality)
// ═══════════════════════════════════════════════════════════════════

/// Minimal state carried through XParsec parsing.
/// Does NOT contain the full zipper - that stays outside XParsec.
/// This type supports equality for XParsec's backtracking/infinite loop detection.
///
/// The EmitContext reference is for emission operations - reference equality is fine
/// since we never backtrack to a different context.
[<Struct>]
type PSGParseState = {
    /// SSA counter for generating unique names during child parsing
    SSACounter: int
    /// Label counter for unique block labels
    LabelCounter: int
    /// Current focus node ID (for context, not navigation)
    FocusNodeId: string
    /// Reference to mutable emission context (nullable - not all parsing needs emission)
    EmitCtx: EmitContext option
}

module PSGParseState =
    /// Create initial parse state (no emission context)
    let create (focusNodeId: string) (ssaCounter: int) : PSGParseState =
        { SSACounter = ssaCounter
          LabelCounter = 0
          FocusNodeId = focusNodeId
          EmitCtx = None }

    /// Create from a zipper (extracts minimal state, no emission context)
    let fromZipper (zipper: PSGZipper) : PSGParseState =
        { SSACounter = zipper.State.SSACounter
          LabelCounter = zipper.State.LabelCounter
          FocusNodeId = zipper.Focus.Id.Value
          EmitCtx = None }

    /// Create for emission with full context
    let forEmission (graph: ProgramSemanticGraph) (focusNodeId: string) (ssaCounter: int) : PSGParseState =
        { SSACounter = ssaCounter
          LabelCounter = 0
          FocusNodeId = focusNodeId
          EmitCtx = Some (EmitContext.create graph) }

    /// Create for emission from zipper
    let forEmissionFromZipper (zipper: PSGZipper) : PSGParseState =
        { SSACounter = zipper.State.SSACounter
          LabelCounter = zipper.State.LabelCounter
          FocusNodeId = zipper.Focus.Id.Value
          EmitCtx = Some (EmitContext.create zipper.Graph) }

    /// Increment SSA counter
    let nextSSA (state: PSGParseState) : PSGParseState * string =
        let name = sprintf "%%v%d" state.SSACounter
        { state with SSACounter = state.SSACounter + 1 }, name

    /// Increment label counter
    let nextLabel (state: PSGParseState) : PSGParseState * string =
        let name = sprintf "bb%d" state.LabelCounter
        { state with LabelCounter = state.LabelCounter + 1 }, name

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
        | Some s ->
            match tryGetFullName s with
            | Some fullName -> fullName = name
            | None -> false
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
        | Some s ->
            match tryGetFullName s with
            | Some fullName -> fullName.Contains(substring) || s.DisplayName.Contains(substring)
            | None -> s.DisplayName.Contains(substring)
        | None -> false)

/// Match child with symbol name ending with suffix
let childSymbolEndsWith (suffix: string) : PSGChildParser<PSGNode> =
    satisfyChild (fun n ->
        match n.Symbol with
        | Some s ->
            match tryGetFullName s with
            | Some fullName -> fullName.EndsWith(suffix) || s.DisplayName.EndsWith(suffix)
            | None -> s.DisplayName.EndsWith(suffix)
        | None -> false)

// ═══════════════════════════════════════════════════════════════════
// XParsec Combinator Wrappers - Composable parsing for PSG children
// ═══════════════════════════════════════════════════════════════════
// NOTE: XParsec uses struct tuples and ImmutableArray for performance.
// These wrappers expose XParsec's native types directly to avoid allocation.

open System.Collections.Immutable

/// Map over parser result
let pMap (f: 'a -> 'b) (p: PSGChildParser<'a>) : PSGChildParser<'b> =
    Combinators.(|>>) p f

/// Bind parser - sequence with dependent result
let pBind (f: 'a -> PSGChildParser<'b>) (p: PSGChildParser<'a>) : PSGChildParser<'b> =
    Combinators.(>>=) p f

/// Sequence two parsers, keep left result
let pLeft (p1: PSGChildParser<'a>) (p2: PSGChildParser<'b>) : PSGChildParser<'a> =
    Combinators.(.>>) p1 p2

/// Sequence two parsers, keep right result
let pRight (p1: PSGChildParser<'a>) (p2: PSGChildParser<'b>) : PSGChildParser<'b> =
    Combinators.(>>.) p1 p2

/// Sequence two parsers, keep both results as struct tuple
let pAnd (p1: PSGChildParser<'a>) (p2: PSGChildParser<'b>) : PSGChildParser<struct ('a * 'b)> =
    Combinators.(.>>.) p1 p2

/// Alternative - try first parser, if it fails try second
let pOr (p1: PSGChildParser<'a>) (p2: PSGChildParser<'a>) : PSGChildParser<'a> =
    Combinators.(<|>) p1 p2

/// Parse zero or more occurrences, returns ImmutableArray
let pMany (p: PSGChildParser<'a>) : PSGChildParser<ImmutableArray<'a>> =
    Combinators.many p

/// Parse one or more occurrences, returns ImmutableArray
let pMany1 (p: PSGChildParser<'a>) : PSGChildParser<ImmutableArray<'a>> =
    Combinators.many1 p

/// Try parser choices in order, returning first success
let pChoice (ps: PSGChildParser<'a> seq) : PSGChildParser<'a> =
    Combinators.choice ps

/// Try parser choices with label for better error messages
let pChoiceL (ps: PSGChildParser<'a> seq) (label: string) : PSGChildParser<'a> =
    Combinators.choiceL ps label

/// Optional parser - succeeds with ValueNone if parser fails
let pOpt (p: PSGChildParser<'a>) : PSGChildParser<'a voption> =
    fun reader ->
        let pos = reader.Position
        match p reader with
        | Ok success -> Parsers.preturn (ValueSome success.Parsed) reader
        | Error _ ->
            reader.Position <- pos
            Parsers.preturn ValueNone reader

/// Skip parser - runs parser but discards result
let pSkip (p: PSGChildParser<'a>) : PSGChildParser<unit> =
    Combinators.optional p

/// Parse items separated by separator, returns struct tuple of (items, separators)
let pSepBy (p: PSGChildParser<'a>) (sep: PSGChildParser<'b>) : PSGChildParser<struct (ImmutableArray<'a> * ImmutableArray<'b>)> =
    Combinators.sepBy p sep

/// Parse one or more items separated by separator
let pSepBy1 (p: PSGChildParser<'a>) (sep: PSGChildParser<'b>) : PSGChildParser<struct (ImmutableArray<'a> * ImmutableArray<'b>)> =
    Combinators.sepBy1 p sep

/// Parse content between left and right delimiters
let pBetween (pOpen: PSGChildParser<'l>) (pClose: PSGChildParser<'r>) (p: PSGChildParser<'a>) : PSGChildParser<'a> =
    Combinators.between pOpen pClose p

/// Look ahead without consuming input
let pLookAhead (p: PSGChildParser<'a>) : PSGChildParser<'a> =
    Combinators.lookAhead p

/// Succeed only if parser fails (consumes no input)
let pNotFollowedBy (p: PSGChildParser<'a>) : PSGChildParser<unit> =
    Combinators.notFollowedBy p

/// Skip zero or more occurrences
let pSkipMany (p: PSGChildParser<'a>) : PSGChildParser<unit> =
    Combinators.skipMany p

/// Skip one or more occurrences
let pSkipMany1 (p: PSGChildParser<'a>) : PSGChildParser<unit> =
    Combinators.skipMany1 p

/// Tuple2 - parse two items and combine into struct tuple
let pTuple2 (p1: PSGChildParser<'a>) (p2: PSGChildParser<'b>) : PSGChildParser<struct ('a * 'b)> =
    Combinators.tuple2 p1 p2

/// Tuple3 - parse three items and combine into struct tuple
let pTuple3 (p1: PSGChildParser<'a>) (p2: PSGChildParser<'b>) (p3: PSGChildParser<'c>) : PSGChildParser<struct ('a * 'b * 'c)> =
    Combinators.tuple3 p1 p2 p3

/// Tuple4 - parse four items and combine into struct tuple
let pTuple4 (p1: PSGChildParser<'a>) (p2: PSGChildParser<'b>) (p3: PSGChildParser<'c>) (p4: PSGChildParser<'d>) : PSGChildParser<struct ('a * 'b * 'c * 'd)> =
    Combinators.tuple4 p1 p2 p3 p4

/// Tuple5 - parse five items and combine into struct tuple
let pTuple5 (p1: PSGChildParser<'a>) (p2: PSGChildParser<'b>) (p3: PSGChildParser<'c>) (p4: PSGChildParser<'d>) (p5: PSGChildParser<'e>) : PSGChildParser<struct ('a * 'b * 'c * 'd * 'e)> =
    Combinators.tuple5 p1 p2 p3 p4 p5

/// Pipe2 - parse two items and apply function
let pPipe2 (p1: PSGChildParser<'a>) (p2: PSGChildParser<'b>) (f: 'a -> 'b -> 'c) : PSGChildParser<'c> =
    Combinators.pipe2 p1 p2 f

/// Pipe3 - parse three items and apply function
let pPipe3 (p1: PSGChildParser<'a>) (p2: PSGChildParser<'b>) (p3: PSGChildParser<'c>) (f: 'a -> 'b -> 'c -> 'd) : PSGChildParser<'d> =
    Combinators.pipe3 p1 p2 p3 f

/// Pipe4 - parse four items and apply function
let pPipe4 (p1: PSGChildParser<'a>) (p2: PSGChildParser<'b>) (p3: PSGChildParser<'c>) (p4: PSGChildParser<'d>) (f: 'a -> 'b -> 'c -> 'd -> 'e) : PSGChildParser<'e> =
    Combinators.pipe4 p1 p2 p3 p4 f

/// Pipe5 - parse five items and apply function
let pPipe5 (p1: PSGChildParser<'a>) (p2: PSGChildParser<'b>) (p3: PSGChildParser<'c>) (p4: PSGChildParser<'d>) (p5: PSGChildParser<'e>) (f: 'a -> 'b -> 'c -> 'd -> 'e -> 'f) : PSGChildParser<'f> =
    Combinators.pipe5 p1 p2 p3 p4 p5 f

// NOTE: manyTill, chainl1, chainr1 are not wrapped because they require equality
// on the token type, and PSGNode contains FSharpSymbol which doesn't support equality.
// If needed, implement custom versions that track position instead of token equality.

/// Return a value without consuming input
let pReturn (x: 'a) : PSGChildParser<'a> =
    Parsers.preturn x

/// Fail with message
let pFail (msg: string) : PSGChildParser<'a> =
    Parsers.fail (Message msg)

/// Add label to parser for better error messages
let pLabel (label: string) (p: PSGChildParser<'a>) : PSGChildParser<'a> =
    Combinators.(<?>) p label

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

/// Generate a fresh block label
let freshLabel : PSGChildParser<string> =
    fun reader ->
        let state, name = PSGParseState.nextLabel reader.State
        reader.State <- state
        Parsers.preturn name reader

// ═══════════════════════════════════════════════════════════════════
// Emission Parsers - XParsec parsers that emit MLIR
// ═══════════════════════════════════════════════════════════════════

/// Get the emission context (fails if not in emission mode)
let getEmitCtx : PSGChildParser<EmitContext> =
    fun reader ->
        match reader.State.EmitCtx with
        | Some ctx -> Parsers.preturn ctx reader
        | None -> Error { Position = reader.Position; Errors = Message "Not in emission mode - no EmitContext available" }

/// Get the PSG graph from emission context
let getGraph : PSGChildParser<ProgramSemanticGraph> =
    fun reader ->
        match reader.State.EmitCtx with
        | Some ctx -> Parsers.preturn ctx.Graph reader
        | None -> Error { Position = reader.Position; Errors = Message "Not in emission mode - no graph available" }

/// Emit a line of MLIR
let emitLine (text: string) : PSGChildParser<unit> =
    fun reader ->
        match reader.State.EmitCtx with
        | Some ctx ->
            EmitContext.emitLine ctx text
            Parsers.preturn () reader
        | None -> Error { Position = reader.Position; Errors = Message "Not in emission mode" }

/// Emit MLIR line with format string
let emitLinef fmt =
    Printf.ksprintf emitLine fmt

/// Record SSA value for a node (for later variable resolution via def-use edges)
let recordNodeSSA (nodeId: NodeId) (ssa: string) (mlirType: string) : PSGChildParser<unit> =
    fun reader ->
        match reader.State.EmitCtx with
        | Some ctx ->
            EmitContext.recordNodeSSA ctx nodeId ssa mlirType
            Parsers.preturn () reader
        | None -> Error { Position = reader.Position; Errors = Message "Not in emission mode" }

/// Look up SSA value for a node by following def-use edges
let lookupNodeSSA (nodeId: NodeId) : PSGChildParser<(string * string) option> =
    fun reader ->
        match reader.State.EmitCtx with
        | Some ctx -> Parsers.preturn (EmitContext.lookupNodeSSA ctx nodeId) reader
        | None -> Parsers.preturn None reader

/// Resolve a variable use to its SSA value by following SymbolUse edges
let resolveVariableUse (useNode: PSGNode) : PSGChildParser<(string * string) option> =
    fun reader ->
        match reader.State.EmitCtx with
        | Some ctx ->
            // Find the SymbolUse edge from this node to its definition
            let defEdge =
                ctx.Graph.Edges
                |> List.tryFind (fun edge ->
                    edge.Source = useNode.Id && edge.Kind = SymbolUse)
            match defEdge with
            | Some edge ->
                // Look up the definition node's SSA value
                Parsers.preturn (EmitContext.lookupNodeSSA ctx edge.Target) reader
            | None ->
                Parsers.preturn None reader
        | None -> Parsers.preturn None reader

/// Register a string literal, returning its global name
let registerStringLiteral (content: string) : PSGChildParser<string> =
    fun reader ->
        match reader.State.EmitCtx with
        | Some ctx ->
            let name = EmitContext.registerStringLiteral ctx content
            Parsers.preturn name reader
        | None -> Error { Position = reader.Position; Errors = Message "Not in emission mode" }

/// Push indentation
let pushIndent : PSGChildParser<unit> =
    fun reader ->
        match reader.State.EmitCtx with
        | Some ctx ->
            BuilderState.pushIndent ctx.Builder
            Parsers.preturn () reader
        | None -> Parsers.preturn () reader

/// Pop indentation
let popIndent : PSGChildParser<unit> =
    fun reader ->
        match reader.State.EmitCtx with
        | Some ctx ->
            BuilderState.popIndent ctx.Builder
            Parsers.preturn () reader
        | None -> Parsers.preturn () reader

// ═══════════════════════════════════════════════════════════════════
// Expression Result Type
// ═══════════════════════════════════════════════════════════════════

/// Result of emitting an expression
type ExprResult =
    | Value of ssa: string * mlirType: string
    | Void
    | EmitError of message: string

module ExprResult =
    let isValue = function Value _ -> true | _ -> false
    let isVoid = function Void -> true | _ -> false
    let isError = function EmitError _ -> true | _ -> false

    let getSSA = function
        | Value (ssa, _) -> Some ssa
        | _ -> None

    let getType = function
        | Value (_, t) -> Some t
        | _ -> None

/// Emit type alias for emission parsers that produce ExprResult
type Emission = PSGChildParser<ExprResult>
