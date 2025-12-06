/// AlloyPatterns - Predicates and classifiers for Alloy library operations
///
/// DESIGN PRINCIPLE:
/// XParsec (via PSGXParsec) is THE combinator infrastructure.
/// This module provides only:
/// - Predicates: PSGNode -> bool (for use with PSGXParsec.satisfyChild)
/// - Classifiers: PSGNode -> 'T option (for extracting operation types)
/// - Helper functions to identify Alloy library calls
///
/// NO custom combinators here. Use XParsec for composition.
///
/// ARCHITECTURE NOTE:
/// Function calls in the PSG have this structure:
///   App:FunctionCall [ModuleName]
///   +-- LongIdent:Module.Function [FunctionName]  <- The actual function symbol
///   +-- Arg1, Arg2, ...                           <- Arguments
///
/// So classifiers must look at the node's children or symbol to identify operations.
module Alex.Patterns.AlloyPatterns

open Core.PSG.Types
open FSharp.Compiler.Symbols

// ===================================================================
// Type Definitions
// ===================================================================

/// Console operation types
type ConsoleOp =
    | Write
    | WriteLine
    | ReadLine
    | Read

/// Time operation types
type TimeOp =
    | CurrentTicks
    | HighResolutionTicks
    | TickFrequency
    | Sleep
    | CurrentUnixTimestamp
    | CurrentTimestamp
    | CurrentDateTimeString

/// Text.Format operation types
type TextFormatOp =
    | IntToString
    | Int64ToString
    | FloatToString
    | BoolToString

/// Memory operation types
type MemoryOp =
    | StackBuffer of elementType: string option  // stackBuffer<T>
    | SpanToString  // Convert Span<byte> to string
    | AsReadOnlySpan  // Get read-only span from buffer

/// All recognized Alloy operations
type AlloyOp =
    | Console of ConsoleOp
    | Time of TimeOp
    | TextFormat of TextFormatOp
    | Memory of MemoryOp

/// Arithmetic operation types
type ArithOp =
    | Add | Sub | Mul | Div | Mod
    | BitwiseAnd | BitwiseOr | BitwiseXor
    | ShiftLeft | ShiftRight
    | Negate | Not

/// Comparison operation types
type CompareOp =
    | Eq | Neq | Lt | Gt | Lte | Gte

/// Information about a function call
type FunctionCallInfo = {
    FunctionSymbol: FSharpSymbol
    FunctionName: string
    Arguments: PSGNode list
}

// ===================================================================
// Helper: Get children of a PSG node
// ===================================================================

/// Get child nodes from a PSGNode using its Children state
let private getChildNodes (graph: ProgramSemanticGraph) (node: PSGNode) : PSGNode list =
    match node.Children with
    | ChildrenState.Parent ids ->
        ids |> List.rev |> List.choose (fun id -> Map.tryFind id.Value graph.Nodes)
    | _ -> []

/// Get the function symbol from an App node's first child
let private getFunctionSymbolFromChildren (children: PSGNode list) : FSharpSymbol option =
    match children with
    | funcChild :: _ -> funcChild.Symbol
    | [] -> None

// ===================================================================
// Symbol Classification Helpers
// ===================================================================

/// Classify a Console operation from a symbol
let classifyConsoleOp (symbol: FSharpSymbol) : ConsoleOp option =
    let fullName = symbol.FullName
    let displayName = symbol.DisplayName

    // Check for WriteLine (must check before Write since Write is substring)
    if displayName = "WriteLine" ||
       fullName.EndsWith(".WriteLine") ||
       fullName.Contains("Console.WriteLine") then
        Some WriteLine
    // Check for Write
    elif displayName = "Write" ||
         fullName.EndsWith(".Write") ||
         fullName.Contains("Console.Write") then
        Some Write
    // Check for Prompt (same as Write semantically)
    elif displayName = "Prompt" ||
         fullName.EndsWith(".Prompt") ||
         fullName.Contains("Console.Prompt") then
        Some Write  // Prompt behaves like Write
    // Check for ReadLine
    elif displayName = "ReadLine" ||
         displayName = "readLine" ||
         fullName.Contains("Console.ReadLine") ||
         fullName.Contains("Console.readLine") then
        Some ReadLine
    // Check for readInto
    elif displayName = "readInto" ||
         fullName.Contains("Console.readInto") then
        Some Read
    else
        None

/// Classify a Time operation from a symbol
let classifyTimeOp (symbol: FSharpSymbol) : TimeOp option =
    let fullName = symbol.FullName
    let displayName = symbol.DisplayName

    if displayName = "currentTicks" || fullName.Contains("currentTicks") then
        Some CurrentTicks
    elif displayName = "highResolutionTicks" || fullName.Contains("highResolutionTicks") then
        Some HighResolutionTicks
    elif displayName = "tickFrequency" || fullName.Contains("tickFrequency") then
        Some TickFrequency
    elif displayName = "sleep" || displayName = "Sleep" || fullName.Contains("sleep") then
        Some Sleep
    elif displayName = "currentUnixTimestamp" || fullName.Contains("currentUnixTimestamp") then
        Some CurrentUnixTimestamp
    elif displayName = "currentTimestamp" || fullName.Contains("currentTimestamp") then
        Some CurrentTimestamp
    elif displayName = "currentDateTimeString" || fullName.Contains("currentDateTimeString") then
        Some CurrentDateTimeString
    else
        None

/// Classify a Text.Format operation from a symbol
let classifyTextFormatOp (symbol: FSharpSymbol) : TextFormatOp option =
    let fullName = symbol.FullName
    let displayName = symbol.DisplayName

    if displayName = "intToString" || fullName.Contains("intToString") then
        Some IntToString
    elif displayName = "int64ToString" || fullName.Contains("int64ToString") then
        Some Int64ToString
    elif displayName = "floatToString" || fullName.Contains("floatToString") then
        Some FloatToString
    elif displayName = "boolToString" || fullName.Contains("boolToString") then
        Some BoolToString
    else
        None

/// Classify a Memory operation from a symbol
let classifyMemoryOp (symbol: FSharpSymbol) : MemoryOp option =
    let fullName = symbol.FullName
    let displayName = symbol.DisplayName

    if displayName = "stackBuffer" || fullName.Contains("Memory.stackBuffer") then
        Some (StackBuffer None)  // Type parameter extracted separately if needed
    elif displayName = "spanToString" || fullName.Contains("spanToString") then
        Some SpanToString
    elif displayName = "AsReadOnlySpan" || fullName.Contains("AsReadOnlySpan") then
        Some AsReadOnlySpan
    else
        None

/// Classify an arithmetic operator from an MFV
let classifyArithOp (mfv: FSharpMemberOrFunctionOrValue) : ArithOp option =
    match mfv.CompiledName with
    | "op_Addition" -> Some Add
    | "op_Subtraction" -> Some Sub
    | "op_Multiply" -> Some Mul
    | "op_Division" -> Some Div
    | "op_Modulus" -> Some Mod
    | "op_BitwiseAnd" -> Some BitwiseAnd
    | "op_BitwiseOr" -> Some BitwiseOr
    | "op_ExclusiveOr" -> Some BitwiseXor
    | "op_LeftShift" -> Some ShiftLeft
    | "op_RightShift" -> Some ShiftRight
    | "op_UnaryNegation" -> Some Negate
    | _ when mfv.DisplayName = "not" -> Some Not
    | _ -> None

/// Classify a comparison operator from an MFV
let classifyCompareOp (mfv: FSharpMemberOrFunctionOrValue) : CompareOp option =
    match mfv.CompiledName with
    | "op_Equality" -> Some Eq
    | "op_Inequality" -> Some Neq
    | "op_LessThan" -> Some Lt
    | "op_GreaterThan" -> Some Gt
    | "op_LessThanOrEqual" -> Some Lte
    | "op_GreaterThanOrEqual" -> Some Gte
    | _ -> None

// ===================================================================
// Node Classifiers - Extract operation type from PSGNode
// ===================================================================

/// Extract Console operation from a node (checks first child for function symbol)
let extractConsoleOp (graph: ProgramSemanticGraph) (node: PSGNode) : ConsoleOp option =
    if not (node.SyntaxKind.StartsWith("App")) then None
    else
        let children = getChildNodes graph node
        match getFunctionSymbolFromChildren children with
        | Some symbol -> classifyConsoleOp symbol
        | None ->
            // Also check the node's own symbol
            match node.Symbol with
            | Some s -> classifyConsoleOp s
            | None -> None

/// Extract Time operation from a node
let extractTimeOp (graph: ProgramSemanticGraph) (node: PSGNode) : TimeOp option =
    if not (node.SyntaxKind.StartsWith("App")) then None
    else
        let children = getChildNodes graph node
        match getFunctionSymbolFromChildren children with
        | Some symbol -> classifyTimeOp symbol
        | None ->
            match node.Symbol with
            | Some s -> classifyTimeOp s
            | None -> None

/// Extract TextFormat operation from a node
let extractTextFormatOp (graph: ProgramSemanticGraph) (node: PSGNode) : TextFormatOp option =
    if not (node.SyntaxKind.StartsWith("App")) then None
    else
        let children = getChildNodes graph node
        match getFunctionSymbolFromChildren children with
        | Some symbol -> classifyTextFormatOp symbol
        | None ->
            match node.Symbol with
            | Some s -> classifyTextFormatOp s
            | None -> None

/// Extract Memory operation from a node
let extractMemoryOp (graph: ProgramSemanticGraph) (node: PSGNode) : MemoryOp option =
    if not (node.SyntaxKind.StartsWith("App")) then None
    else
        let children = getChildNodes graph node
        match getFunctionSymbolFromChildren children with
        | Some symbol -> classifyMemoryOp symbol
        | None ->
            match node.Symbol with
            | Some s -> classifyMemoryOp s
            | None -> None

/// Extract any Alloy operation from a node
let extractAlloyOp (graph: ProgramSemanticGraph) (node: PSGNode) : AlloyOp option =
    match extractConsoleOp graph node with
    | Some op -> Some (Console op)
    | None ->
    match extractTimeOp graph node with
    | Some op -> Some (Time op)
    | None ->
    match extractTextFormatOp graph node with
    | Some op -> Some (TextFormat op)
    | None ->
    match extractMemoryOp graph node with
    | Some op -> Some (Memory op)
    | None -> None

/// Extract arithmetic operation from an App node
let extractArithOp (graph: ProgramSemanticGraph) (node: PSGNode) : ArithOp option =
    if not (node.SyntaxKind.StartsWith("App")) then None
    else
        let children = getChildNodes graph node
        match getFunctionSymbolFromChildren children with
        | Some (:? FSharpMemberOrFunctionOrValue as mfv) -> classifyArithOp mfv
        | _ -> None

/// Extract comparison operation from an App node
let extractCompareOp (graph: ProgramSemanticGraph) (node: PSGNode) : CompareOp option =
    if not (node.SyntaxKind.StartsWith("App")) then None
    else
        let children = getChildNodes graph node
        match getFunctionSymbolFromChildren children with
        | Some (:? FSharpMemberOrFunctionOrValue as mfv) -> classifyCompareOp mfv
        | _ -> None

// ===================================================================
// Node Predicates - Boolean checks for PSGNode
// ===================================================================

/// Check if node is a Console operation
let isConsoleCall (graph: ProgramSemanticGraph) (node: PSGNode) : bool =
    extractConsoleOp graph node |> Option.isSome

/// Check if node is a Time operation
let isTimeCall (graph: ProgramSemanticGraph) (node: PSGNode) : bool =
    extractTimeOp graph node |> Option.isSome

/// Check if node is a TextFormat operation
let isTextFormatCall (graph: ProgramSemanticGraph) (node: PSGNode) : bool =
    extractTextFormatOp graph node |> Option.isSome

/// Check if node is a Memory operation
let isMemoryCall (graph: ProgramSemanticGraph) (node: PSGNode) : bool =
    extractMemoryOp graph node |> Option.isSome

/// Check if node is any Alloy operation
let isAlloyCall (graph: ProgramSemanticGraph) (node: PSGNode) : bool =
    extractAlloyOp graph node |> Option.isSome

/// Check if node is an arithmetic operation
let isArithCall (graph: ProgramSemanticGraph) (node: PSGNode) : bool =
    extractArithOp graph node |> Option.isSome

/// Check if node is a comparison operation
let isCompareCall (graph: ProgramSemanticGraph) (node: PSGNode) : bool =
    extractCompareOp graph node |> Option.isSome

// ===================================================================
// Function Call Info Extraction
// ===================================================================

/// Extract function call information from an App node
let extractFunctionCall (graph: ProgramSemanticGraph) (node: PSGNode) : FunctionCallInfo option =
    if not (node.SyntaxKind.StartsWith("App")) then None
    else
        let children = getChildNodes graph node
        match children with
        | funcNode :: args ->
            match funcNode.Symbol with
            | Some sym ->
                Some {
                    FunctionSymbol = sym
                    FunctionName = sym.FullName
                    Arguments = args
                }
            | None -> None
        | [] -> None

// ===================================================================
// MLIR Operation Names for Operators
// ===================================================================

/// Get MLIR operation name for arithmetic op
let arithOpToMLIR (op: ArithOp) : string =
    match op with
    | Add -> "arith.addi"
    | Sub -> "arith.subi"
    | Mul -> "arith.muli"
    | Div -> "arith.divsi"
    | Mod -> "arith.remsi"
    | BitwiseAnd -> "arith.andi"
    | BitwiseOr -> "arith.ori"
    | BitwiseXor -> "arith.xori"
    | ShiftLeft -> "arith.shli"
    | ShiftRight -> "arith.shrsi"
    | Negate -> "arith.subi" // Will need special handling (0 - x)
    | Not -> "arith.xori"    // Will need special handling (xor with -1)

/// Get MLIR predicate for comparison op
let compareOpToPredicate (op: CompareOp) : string =
    match op with
    | Eq -> "eq"
    | Neq -> "ne"
    | Lt -> "slt"
    | Gt -> "sgt"
    | Lte -> "sle"
    | Gte -> "sge"
