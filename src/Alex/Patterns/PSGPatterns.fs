/// PSGPatterns - XParsec-style pattern matching combinators for PSG nodes
///
/// Provides composable pattern matchers for recognizing PSG structures.
/// Inspired by XParsec's parser combinators, adapted for tree matching.
///
/// A Pattern<'T> takes a PSGZipper and returns either:
/// - Some 'T: Successfully matched, extracted value
/// - None: Did not match (no consumption, allows backtracking)
module Alex.Patterns.PSGPatterns

open Core.PSG.Types
open Alex.Traversal.PSGZipper

/// A pattern matcher that tries to extract 'T from a zipper position
type Pattern<'T> = PSGZipper -> 'T option

/// Pattern matching result with context
type MatchResult<'T> =
    | Matched of value: 'T * zipper: PSGZipper
    | NoMatch

// ═══════════════════════════════════════════════════════════════════
// Core Combinators
// ═══════════════════════════════════════════════════════════════════

/// Always succeeds with the given value
let succeed (value: 'T) : Pattern<'T> =
    fun _ -> Some value

/// Always fails
let fail<'T> : Pattern<'T> =
    fun _ -> None

/// Match if predicate is true on focus node
let satisfies (pred: PSGNode -> bool) : Pattern<PSGNode> =
    fun z -> if pred z.Focus then Some z.Focus else None

/// Match a specific syntax kind
let syntaxKind (kind: string) : Pattern<PSGNode> =
    satisfies (fun n -> n.SyntaxKind = kind)

/// Match syntax kind prefix
let syntaxKindPrefix (prefix: string) : Pattern<PSGNode> =
    satisfies (fun n -> n.SyntaxKind.StartsWith(prefix))

/// Match by symbol name
let symbolName (name: string) : Pattern<PSGNode> =
    satisfies (fun n ->
        match n.Symbol with
        | Some s -> s.FullName = name || s.DisplayName = name
        | None -> false)

/// Match by symbol name containing substring
let symbolContains (substring: string) : Pattern<PSGNode> =
    satisfies (fun n ->
        match n.Symbol with
        | Some s -> s.FullName.Contains(substring) || s.DisplayName.Contains(substring)
        | None -> false)

/// Match by symbol name ending with suffix
let symbolEndsWith (suffix: string) : Pattern<PSGNode> =
    satisfies (fun n ->
        match n.Symbol with
        | Some s -> s.FullName.EndsWith(suffix) || s.DisplayName.EndsWith(suffix)
        | None -> false)

/// Get the focus node's symbol
let getSymbol : Pattern<FSharp.Compiler.Symbols.FSharpSymbol> =
    fun z ->
        match z.Focus.Symbol with
        | Some s -> Some s
        | None -> None

/// Get the focus node's type
let getType : Pattern<FSharp.Compiler.Symbols.FSharpType> =
    fun z ->
        match z.Focus.Type with
        | Some t -> Some t
        | None -> None

// ═══════════════════════════════════════════════════════════════════
// Functor and Monad Operations
// ═══════════════════════════════════════════════════════════════════

/// Map a function over the result
let map (f: 'T -> 'U) (p: Pattern<'T>) : Pattern<'U> =
    fun z -> p z |> Option.map f

/// Infix map operator
let (|>>) (p: Pattern<'T>) (f: 'T -> 'U) : Pattern<'U> =
    map f p

/// Monadic bind
let bind (f: 'T -> Pattern<'U>) (p: Pattern<'T>) : Pattern<'U> =
    fun z ->
        match p z with
        | Some x -> f x z
        | None -> None

/// Infix bind operator
let (>>=) (p: Pattern<'T>) (f: 'T -> Pattern<'U>) : Pattern<'U> =
    bind f p

/// Sequence two patterns, keep second result
let andThen (p2: Pattern<'U>) (p1: Pattern<'T>) : Pattern<'U> =
    p1 >>= fun _ -> p2

/// Infix sequence operator
let (>>.) (p1: Pattern<'T>) (p2: Pattern<'U>) : Pattern<'U> =
    andThen p2 p1

/// Sequence, keep first result
let (.>>) (p1: Pattern<'T>) (p2: Pattern<'U>) : Pattern<'T> =
    p1 >>= fun x -> p2 |>> fun _ -> x

/// Sequence and pair results
let (.>>.) (p1: Pattern<'T>) (p2: Pattern<'U>) : Pattern<'T * 'U> =
    p1 >>= fun a -> p2 |>> fun b -> (a, b)

// ═══════════════════════════════════════════════════════════════════
// Choice Combinators
// ═══════════════════════════════════════════════════════════════════

/// Try first pattern, if fails try second
let orElse (p2: Pattern<'T>) (p1: Pattern<'T>) : Pattern<'T> =
    fun z ->
        match p1 z with
        | Some x -> Some x
        | None -> p2 z

/// Infix choice operator
let (<|>) (p1: Pattern<'T>) (p2: Pattern<'T>) : Pattern<'T> =
    orElse p2 p1

/// Try multiple patterns in order
let choice (patterns: Pattern<'T> list) : Pattern<'T> =
    patterns |> List.fold orElse fail

/// Optional pattern (always succeeds, may produce None)
let optional (p: Pattern<'T>) : Pattern<'T option> =
    fun z ->
        match p z with
        | Some x -> Some (Some x)
        | None -> Some None

// ═══════════════════════════════════════════════════════════════════
// Structural Patterns
// ═══════════════════════════════════════════════════════════════════

/// Match if at root
let atRoot : Pattern<unit> =
    fun z ->
        if PSGZipper.isAtRoot z then Some () else None

/// Match if has children
let hasChildren : Pattern<unit> =
    fun z ->
        let kids = PSGZipper.children z
        if not (List.isEmpty kids) then Some () else None

/// Match if has exactly n children
let childCount (n: int) : Pattern<unit> =
    fun z ->
        let kids = PSGZipper.children z
        if List.length kids = n then Some () else None

/// Match on first child
let firstChild (p: Pattern<'T>) : Pattern<'T> =
    fun z ->
        match PSGZipper.down z with
        | NavOk childZ -> p childZ
        | NavFail _ -> None

/// Match on nth child
let nthChild (n: int) (p: Pattern<'T>) : Pattern<'T> =
    fun z ->
        match PSGZipper.downTo n z with
        | NavOk childZ -> p childZ
        | NavFail _ -> None

/// Match on all children
let allChildren (p: Pattern<'T>) : Pattern<'T list> =
    fun z ->
        let kids = PSGZipper.childNodes z
        let results =
            kids |> List.choose (fun node ->
                let childZ = PSGZipper.create z.Graph node
                p { childZ with State = z.State })
        if List.length results = List.length kids then
            Some results
        else
            None

/// Match on any child
let anyChild (p: Pattern<'T>) : Pattern<'T> =
    fun z ->
        let kids = PSGZipper.childNodes z
        kids |> List.tryPick (fun node ->
            let childZ = PSGZipper.create z.Graph node
            p { childZ with State = z.State })

/// Match on parent
let parent (p: Pattern<'T>) : Pattern<'T> =
    fun z ->
        match PSGZipper.parent z with
        | Some parentNode ->
            let parentZ = PSGZipper.create z.Graph parentNode
            p { parentZ with State = z.State }
        | None -> None

/// Check if any ancestor matches
let hasAncestor (p: Pattern<'T>) : Pattern<'T> =
    fun z ->
        let ancestors = PSGZipper.ancestors z
        ancestors |> List.tryPick (fun node ->
            let ancestorZ = PSGZipper.create z.Graph node
            p { ancestorZ with State = z.State })

// ═══════════════════════════════════════════════════════════════════
// Common F# Patterns
// ═══════════════════════════════════════════════════════════════════

/// Match a function application node
let isApplication : Pattern<PSGNode> =
    syntaxKindPrefix "App"

/// Match a let binding
let isLetBinding : Pattern<PSGNode> =
    syntaxKind "LetOrUse" <|> syntaxKindPrefix "Let"

/// Match a sequential expression
let isSequential : Pattern<PSGNode> =
    syntaxKindPrefix "Sequential"

/// Match a constant value
let isConst : Pattern<PSGNode> =
    syntaxKindPrefix "Const"

/// Match a string constant
let isStringConst : Pattern<string> =
    fun z ->
        if z.Focus.SyntaxKind.StartsWith("Const:String") then
            // Extract string value from syntax kind
            let kind = z.Focus.SyntaxKind
            if kind.Contains("\"") then
                // Const:String:"value" format
                let start = kind.IndexOf('"') + 1
                let endIdx = kind.LastIndexOf('"')
                if endIdx > start then
                    Some (kind.Substring(start, endIdx - start))
                else None
            else None
        else None

/// Match an integer constant
let isIntConst : Pattern<int> =
    fun z ->
        if z.Focus.SyntaxKind.StartsWith("Const:Int32") then
            let kind = z.Focus.SyntaxKind
            let colonIdx = kind.LastIndexOf(':')
            if colonIdx > 0 then
                let valueStr = kind.Substring(colonIdx + 1)
                match System.Int32.TryParse(valueStr) with
                | true, v -> Some v
                | false, _ -> None
            else None
        else None

/// Match an int64 constant
let isInt64Const : Pattern<int64> =
    fun z ->
        if z.Focus.SyntaxKind.StartsWith("Const:Int64") then
            let kind = z.Focus.SyntaxKind
            let colonIdx = kind.LastIndexOf(':')
            if colonIdx > 0 then
                let valueStr = kind.Substring(colonIdx + 1)
                match System.Int64.TryParse(valueStr) with
                | true, v -> Some v
                | false, _ -> None
            else None
        else None

/// Match a lambda expression
let isLambda : Pattern<PSGNode> =
    syntaxKindPrefix "Lambda"

/// Match a match expression
let isMatch : Pattern<PSGNode> =
    syntaxKindPrefix "Match"

/// Match a while loop
let isWhile : Pattern<PSGNode> =
    syntaxKindPrefix "While"

/// Match a for loop
let isFor : Pattern<PSGNode> =
    syntaxKindPrefix "For"

/// Match an if expression
let isIf : Pattern<PSGNode> =
    syntaxKindPrefix "If"

// ═══════════════════════════════════════════════════════════════════
// Pattern Extraction Helpers
// ═══════════════════════════════════════════════════════════════════

/// Extract the syntax kind
let extractKind : Pattern<string> =
    fun z -> Some z.Focus.SyntaxKind

/// Extract the node ID
let extractId : Pattern<NodeId> =
    fun z -> Some z.Focus.Id

/// Extract symbol full name
let extractSymbolName : Pattern<string> =
    fun z ->
        match z.Focus.Symbol with
        | Some s -> Some s.FullName
        | None -> None

/// Extract function application target and args
let extractAppParts : Pattern<PSGNode * PSGNode list> =
    fun z ->
        if z.Focus.SyntaxKind.StartsWith("App") then
            let kids = PSGZipper.childNodes z
            match kids with
            | func :: args -> Some (func, args)
            | _ -> None
        else None

/// Extract the focus node
let extractFocus : Pattern<PSGNode> =
    fun z -> Some z.Focus

// ═══════════════════════════════════════════════════════════════════
// Binary Operator Patterns
// ═══════════════════════════════════════════════════════════════════

open FSharp.Compiler.Symbols

/// Extract FSharpMemberOrFunctionOrValue from current node's symbol
let getMemberOrFunction : Pattern<FSharpMemberOrFunctionOrValue> =
    fun z ->
        match z.Focus.Symbol with
        | Some (:? FSharpMemberOrFunctionOrValue as mfv) -> Some mfv
        | _ -> None

/// Information about a binary operator application
type BinaryOpInfo = {
    Operator: FSharpMemberOrFunctionOrValue
    LeftOperand: PSGNode
    RightOperand: PSGNode
}

/// Recognize curried binary operator pattern: App[App[op, arg1], arg2]
/// This is how F# represents infix operators like `a < b`
///
/// Composed from primitives:
/// - isApplication: check we're at an App node
/// - childCount 2: outer App has exactly 2 children
/// - firstChild: navigate to inner App
/// - nthChild: navigate to specific children
/// - extractFocus: get the node at current position
/// - getMemberOrFunction: extract operator symbol
let isCurriedBinaryOp : Pattern<BinaryOpInfo> =
    // Outer App must have exactly 2 children
    isApplication >>. childCount 2 >>.
    // Get the right operand (child 1 of outer App)
    nthChild 1 extractFocus >>= fun rightOperand ->
    // Navigate to inner App (child 0) and check it's also an App with 2 children
    firstChild (
        isApplication >>. childCount 2 >>.
        // Get operator from child 0 of inner App
        firstChild getMemberOrFunction .>>.
        // Get left operand from child 1 of inner App
        nthChild 1 extractFocus
    ) |>> fun (operator, leftOperand) ->
        { Operator = operator; LeftOperand = leftOperand; RightOperand = rightOperand }

/// Check if an operator is an arithmetic operator, return MLIR op name
let arithmeticOp (mfv: FSharpMemberOrFunctionOrValue) : string option =
    match mfv.CompiledName with
    | "op_Addition" -> Some "arith.addi"
    | "op_Subtraction" -> Some "arith.subi"
    | "op_Multiply" -> Some "arith.muli"
    | "op_Division" -> Some "arith.divsi"
    | "op_Modulus" -> Some "arith.remsi"
    | _ -> None

/// Check if an operator is a comparison operator, return MLIR predicate
let comparisonOp (mfv: FSharpMemberOrFunctionOrValue) : string option =
    match mfv.CompiledName with
    | "op_LessThan" -> Some "slt"
    | "op_GreaterThan" -> Some "sgt"
    | "op_LessThanOrEqual" -> Some "sle"
    | "op_GreaterThanOrEqual" -> Some "sge"
    | "op_Equality" -> Some "eq"
    | "op_Inequality" -> Some "ne"
    | _ -> None
