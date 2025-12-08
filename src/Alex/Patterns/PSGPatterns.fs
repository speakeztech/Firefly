/// PSGPatterns - Predicates and extractors for PSG nodes
///
/// DESIGN PRINCIPLE:
/// XParsec (via PSGXParsec) is THE combinator infrastructure.
/// This module provides only:
/// - Predicates: PSGNode -> bool (for use with PSGXParsec.satisfyChild)
/// - Extractors: PSGNode -> 'T option (for extracting data from matched nodes)
///
/// NO custom combinators here. Use XParsec for composition.
module Alex.Patterns.PSGPatterns

open Core.PSG.Types
open FSharp.Compiler.Symbols

// ═══════════════════════════════════════════════════════════════════
// Safe Symbol Helpers
// ═══════════════════════════════════════════════════════════════════

/// Safely get a symbol's FullName, handling types like 'unit' that throw exceptions
let private tryGetFullName (sym: FSharpSymbol) : string option =
    try Some sym.FullName with _ -> None

// ═══════════════════════════════════════════════════════════════════
// Node Predicates - For use with PSGXParsec.satisfyChild
// ═══════════════════════════════════════════════════════════════════

/// Check syntax kind (exact match)
let hasSyntaxKind (kind: string) (node: PSGNode) : bool =
    node.SyntaxKind = kind

/// Check syntax kind prefix
let hasSyntaxKindPrefix (prefix: string) (node: PSGNode) : bool =
    node.SyntaxKind.StartsWith(prefix)

/// Check symbol full name
let hasSymbolName (name: string) (node: PSGNode) : bool =
    match node.Symbol with
    | Some s ->
        match tryGetFullName s with
        | Some fullName -> fullName = name || s.DisplayName = name
        | None -> s.DisplayName = name
    | None -> false

/// Check symbol contains substring
let symbolContains (substring: string) (node: PSGNode) : bool =
    match node.Symbol with
    | Some s ->
        match tryGetFullName s with
        | Some fullName -> fullName.Contains(substring) || s.DisplayName.Contains(substring)
        | None -> s.DisplayName.Contains(substring)
    | None -> false

/// Check symbol ends with suffix
let symbolEndsWith (suffix: string) (node: PSGNode) : bool =
    match node.Symbol with
    | Some s ->
        match tryGetFullName s with
        | Some fullName -> fullName.EndsWith(suffix) || s.DisplayName.EndsWith(suffix)
        | None -> s.DisplayName.EndsWith(suffix)
    | None -> false

/// Is an App node
let isApp (node: PSGNode) : bool =
    node.SyntaxKind.StartsWith("App")

/// Is a Const node
let isConst (node: PSGNode) : bool =
    node.SyntaxKind.StartsWith("Const")

/// Is an Ident/Value node
let isIdent (node: PSGNode) : bool =
    node.SyntaxKind.StartsWith("Ident") ||
    node.SyntaxKind.StartsWith("Value") ||
    node.SyntaxKind.StartsWith("LongIdent")

/// Is a Let binding
/// Matches: LetOrUse, LetOrUse:Let, Let, LetDeclaration, etc.
let isLet (node: PSGNode) : bool =
    node.SyntaxKind.StartsWith("LetOrUse") || node.SyntaxKind.StartsWith("Let")

/// Is a Sequential expression
let isSequential (node: PSGNode) : bool =
    node.SyntaxKind.StartsWith("Sequential")

/// Is a Lambda
let isLambda (node: PSGNode) : bool =
    node.SyntaxKind.StartsWith("Lambda")

/// Is a Match expression
let isMatch (node: PSGNode) : bool =
    node.SyntaxKind.StartsWith("Match")

/// Is a While loop
let isWhile (node: PSGNode) : bool =
    node.SyntaxKind.StartsWith("While")

/// Is a For loop
let isFor (node: PSGNode) : bool =
    node.SyntaxKind.StartsWith("For")

/// Is an If expression
let isIf (node: PSGNode) : bool =
    node.SyntaxKind.StartsWith("If")

/// Is an AddressOf expression (&&var or &var)
let isAddressOf (node: PSGNode) : bool =
    node.SyntaxKind = "AddressOf"

/// Is a Pattern node (structural, typically skipped)
let isPattern (node: PSGNode) : bool =
    node.SyntaxKind.StartsWith("Pattern:")

/// Is a TypeApp node
let isTypeApp (node: PSGNode) : bool =
    node.SyntaxKind.StartsWith("TypeApp")

/// Is a MutableSet node
let isMutableSet (node: PSGNode) : bool =
    node.SyntaxKind.StartsWith("MutableSet:")

/// Is a PropertyAccess node (like receiver.Length)
let isPropertyAccess (node: PSGNode) : bool =
    node.SyntaxKind.StartsWith("PropertyAccess:")

/// Extract property name from a PropertyAccess node
let extractPropertyName (node: PSGNode) : string option =
    if node.SyntaxKind.StartsWith("PropertyAccess:") then
        Some (node.SyntaxKind.Substring("PropertyAccess:".Length))
    else
        None

// ═══════════════════════════════════════════════════════════════════
// Node Extractors - Extract typed data from nodes
// ═══════════════════════════════════════════════════════════════════

/// Extract the symbol from a node
let extractSymbol (node: PSGNode) : FSharpSymbol option =
    node.Symbol

/// Extract the type from a node
let extractType (node: PSGNode) : FSharpType option =
    node.Type

/// Extract MemberOrFunctionOrValue from a node
let extractMemberOrFunction (node: PSGNode) : FSharpMemberOrFunctionOrValue option =
    match node.Symbol with
    | Some (:? FSharpMemberOrFunctionOrValue as mfv) -> Some mfv
    | _ -> None

/// Extract string constant value from a Const:String node using ConstantValue field
let extractStringConst (node: PSGNode) : string option =
    match node.ConstantValue with
    | Some (StringValue s) -> Some s
    | _ -> None

/// Extract int32 constant value using ConstantValue field
let extractIntConst (node: PSGNode) : int option =
    match node.ConstantValue with
    | Some (Int32Value i) -> Some i
    | _ -> None

/// Extract int64 constant value using ConstantValue field
let extractInt64Const (node: PSGNode) : int64 option =
    match node.ConstantValue with
    | Some (Int64Value i) -> Some i
    | _ -> None

/// Extract float constant value using ConstantValue field
let extractFloatConst (node: PSGNode) : float option =
    match node.ConstantValue with
    | Some (FloatValue f) -> Some f
    | _ -> None

/// Extract bool constant value using ConstantValue field
let extractBoolConst (node: PSGNode) : bool option =
    match node.ConstantValue with
    | Some (BoolValue b) -> Some b
    | _ -> None

/// Extract variable name from MutableSet node
let extractMutableSetName (node: PSGNode) : string option =
    if node.SyntaxKind.StartsWith("MutableSet:") then
        Some (node.SyntaxKind.Substring("MutableSet:".Length))
    else None

// ═══════════════════════════════════════════════════════════════════
// Operator Classification
// ═══════════════════════════════════════════════════════════════════

/// Check if an operator is arithmetic, return MLIR op name
let arithmeticOp (mfv: FSharpMemberOrFunctionOrValue) : string option =
    match mfv.CompiledName with
    | "op_Addition" -> Some "arith.addi"
    | "op_Subtraction" -> Some "arith.subi"
    | "op_Multiply" -> Some "arith.muli"
    | "op_Division" -> Some "arith.divsi"
    | "op_Modulus" -> Some "arith.remsi"
    | _ -> None

/// Check if an operator is comparison, return MLIR predicate
let comparisonOp (mfv: FSharpMemberOrFunctionOrValue) : string option =
    match mfv.CompiledName with
    | "op_LessThan" -> Some "slt"
    | "op_GreaterThan" -> Some "sgt"
    | "op_LessThanOrEqual" -> Some "sle"
    | "op_GreaterThanOrEqual" -> Some "sge"
    | "op_Equality" -> Some "eq"
    | "op_Inequality" -> Some "ne"
    | _ -> None

/// Arithmetic operation types (for typed dispatch)
type ArithOp = Add | Sub | Mul | Div | Mod

/// Comparison operation types
type CompareOp = Eq | Neq | Lt | Gt | Lte | Gte

/// Classify an arithmetic operator
let classifyArithOp (mfv: FSharpMemberOrFunctionOrValue) : ArithOp option =
    match mfv.CompiledName with
    | "op_Addition" -> Some Add
    | "op_Subtraction" -> Some Sub
    | "op_Multiply" -> Some Mul
    | "op_Division" -> Some Div
    | "op_Modulus" -> Some Mod
    | _ -> None

/// Classify a comparison operator
let classifyCompareOp (mfv: FSharpMemberOrFunctionOrValue) : CompareOp option =
    match mfv.CompiledName with
    | "op_LessThan" -> Some Lt
    | "op_GreaterThan" -> Some Gt
    | "op_LessThanOrEqual" -> Some Lte
    | "op_GreaterThanOrEqual" -> Some Gte
    | "op_Equality" -> Some Eq
    | "op_Inequality" -> Some Neq
    | _ -> None

// ═══════════════════════════════════════════════════════════════════
// Binary Operator Info (for complex pattern results)
// ═══════════════════════════════════════════════════════════════════

/// Information about a binary operator application
type BinaryOpInfo = {
    Operator: FSharpMemberOrFunctionOrValue
    LeftOperand: PSGNode
    RightOperand: PSGNode
}
