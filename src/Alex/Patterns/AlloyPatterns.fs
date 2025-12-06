/// AlloyPatterns - Pattern matchers for Alloy library operations
///
/// Provides composable patterns for recognizing Alloy library calls
/// (Console, Time, Text.Format, etc.) in the PSG.
module Alex.Patterns.AlloyPatterns

open Core.PSG.Types
open Alex.Traversal.PSGZipper
open Alex.Patterns.PSGPatterns

// ═══════════════════════════════════════════════════════════════════
// Console Module Patterns
// ═══════════════════════════════════════════════════════════════════

/// Console operation types
type ConsoleOp =
    | Write
    | WriteLine
    | ReadLine
    | Read

/// Match any Console.* operation
let isConsoleOp : Pattern<ConsoleOp> =
    fun z ->
        match z.Focus.Symbol with
        | Some symbol ->
            let fullName = symbol.FullName
            let displayName = symbol.DisplayName
            if fullName.Contains("Console.write") || fullName.EndsWith(".write") then
                if fullName.Contains("writeLine") || displayName = "writeLine" then
                    Some WriteLine
                else
                    Some Write
            elif fullName.Contains("Console.readLine") || displayName = "readLine" then
                Some ReadLine
            elif fullName.Contains("Console.read") || displayName = "read" then
                Some Read
            else None
        | None -> None

/// Match Console.write
let isConsoleWrite : Pattern<unit> =
    isConsoleOp >>= fun op ->
        if op = Write then succeed () else fail

/// Match Console.writeLine
let isConsoleWriteLine : Pattern<unit> =
    isConsoleOp >>= fun op ->
        if op = WriteLine then succeed () else fail

/// Match Console.readLine
let isConsoleReadLine : Pattern<unit> =
    isConsoleOp >>= fun op ->
        if op = ReadLine then succeed () else fail

// ═══════════════════════════════════════════════════════════════════
// Time Module Patterns
// ═══════════════════════════════════════════════════════════════════

/// Time operation types
type TimeOp =
    | CurrentTicks
    | HighResolutionTicks
    | TickFrequency
    | Sleep
    | CurrentUnixTimestamp
    | CurrentTimestamp
    | CurrentDateTimeString

/// Match any Time.* operation
let isTimeOp : Pattern<TimeOp> =
    fun z ->
        match z.Focus.Symbol with
        | Some symbol ->
            let fullName = symbol.FullName
            let displayName = symbol.DisplayName
            if fullName.Contains("currentTicks") || displayName = "currentTicks" then
                Some CurrentTicks
            elif fullName.Contains("highResolutionTicks") || displayName = "highResolutionTicks" then
                Some HighResolutionTicks
            elif fullName.Contains("tickFrequency") || displayName = "tickFrequency" then
                Some TickFrequency
            elif fullName.Contains("sleep") || displayName = "sleep" then
                Some Sleep
            elif fullName.Contains("currentUnixTimestamp") || displayName = "currentUnixTimestamp" then
                Some CurrentUnixTimestamp
            elif fullName.Contains("currentTimestamp") || displayName = "currentTimestamp" then
                Some CurrentTimestamp
            elif fullName.Contains("currentDateTimeString") || displayName = "currentDateTimeString" then
                Some CurrentDateTimeString
            else None
        | None -> None

/// Match Time.currentTicks
let isCurrentTicks : Pattern<unit> =
    isTimeOp >>= fun op ->
        if op = CurrentTicks then succeed () else fail

/// Match Time.sleep
let isSleep : Pattern<unit> =
    isTimeOp >>= fun op ->
        if op = Sleep then succeed () else fail

/// Match Time.currentUnixTimestamp
let isCurrentUnixTimestamp : Pattern<unit> =
    isTimeOp >>= fun op ->
        if op = CurrentUnixTimestamp then succeed () else fail

// ═══════════════════════════════════════════════════════════════════
// Text.Format Module Patterns
// ═══════════════════════════════════════════════════════════════════

/// Text.Format operation types
type TextFormatOp =
    | IntToString
    | Int64ToString
    | FloatToString
    | BoolToString

/// Match any Text.Format.* operation
let isTextFormatOp : Pattern<TextFormatOp> =
    fun z ->
        match z.Focus.Symbol with
        | Some symbol ->
            let fullName = symbol.FullName
            let displayName = symbol.DisplayName
            if fullName.Contains("intToString") || displayName = "intToString" then
                Some IntToString
            elif fullName.Contains("int64ToString") || displayName = "int64ToString" then
                Some Int64ToString
            elif fullName.Contains("floatToString") || displayName = "floatToString" then
                Some FloatToString
            elif fullName.Contains("boolToString") || displayName = "boolToString" then
                Some BoolToString
            else None
        | None -> None

// ═══════════════════════════════════════════════════════════════════
// Combined Alloy Operation Pattern
// ═══════════════════════════════════════════════════════════════════

/// All recognized Alloy operations
type AlloyOp =
    | Console of ConsoleOp
    | Time of TimeOp
    | TextFormat of TextFormatOp

/// Match any Alloy operation
let isAlloyOp : Pattern<AlloyOp> =
    choice [
        isConsoleOp |>> Console
        isTimeOp |>> Time
        isTextFormatOp |>> TextFormat
    ]

/// Check if a node is an Alloy operation (quick check)
let isAlloyCall : Pattern<unit> =
    fun z ->
        match z.Focus.Symbol with
        | Some symbol ->
            let fullName = symbol.FullName
            if fullName.StartsWith("Alloy.") ||
               fullName.Contains(".Console.") ||
               fullName.Contains(".Time.") ||
               fullName.Contains(".Text.Format.") then
                Some ()
            else None
        | None -> None

// ═══════════════════════════════════════════════════════════════════
// Application Pattern Extractors
// ═══════════════════════════════════════════════════════════════════

/// Extract Console.writeLine with its string argument
let extractWriteLineArg : Pattern<string option> =
    isConsoleWriteLine >>. fun z ->
        // Look for a string constant child
        let kids = PSGZipper.childNodes z
        kids |> List.tryPick (fun node ->
            let childZ = PSGZipper.create z.Graph node
            isStringConst { childZ with State = z.State })
        |> Some

/// Extract Time.sleep with its milliseconds argument
let extractSleepArg : Pattern<PSGNode option> =
    isSleep >>. fun z ->
        let kids = PSGZipper.childNodes z
        match kids with
        | [_; arg] -> Some (Some arg)  // Function + 1 arg
        | [arg] -> Some (Some arg)     // Just the arg
        | _ -> Some None

/// Pattern that matches a function call and extracts info
type FunctionCallInfo = {
    FunctionName: string
    Arguments: PSGNode list
    ReturnType: string option
}

/// Extract function call information
let extractFunctionCall : Pattern<FunctionCallInfo> =
    fun z ->
        if z.Focus.SyntaxKind.StartsWith("App") ||
           z.Focus.SyntaxKind.StartsWith("Call") then
            let kids = PSGZipper.childNodes z
            match kids with
            | func :: args ->
                let funcName =
                    match func.Symbol with
                    | Some s -> s.FullName
                    | None -> func.SyntaxKind
                let retType =
                    z.Focus.Type |> Option.bind (fun t ->
                        if t.HasTypeDefinition then
                            t.TypeDefinition.TryFullName
                        else None)
                Some {
                    FunctionName = funcName
                    Arguments = args
                    ReturnType = retType
                }
            | _ -> None
        else None

// ═══════════════════════════════════════════════════════════════════
// Arithmetic Pattern Helpers
// ═══════════════════════════════════════════════════════════════════

/// Arithmetic operation types
type ArithOp =
    | Add | Sub | Mul | Div | Mod
    | BitwiseAnd | BitwiseOr | BitwiseXor
    | ShiftLeft | ShiftRight
    | Negate | Not

/// Match arithmetic operators
let isArithOp : Pattern<ArithOp> =
    fun z ->
        match z.Focus.Symbol with
        | Some symbol ->
            match symbol.DisplayName with
            | "op_Addition" | "+" -> Some Add
            | "op_Subtraction" | "-" -> Some Sub
            | "op_Multiply" | "*" -> Some Mul
            | "op_Division" | "/" -> Some Div
            | "op_Modulus" | "%" -> Some Mod
            | "op_BitwiseAnd" | "&&&" -> Some BitwiseAnd
            | "op_BitwiseOr" | "|||" -> Some BitwiseOr
            | "op_ExclusiveOr" | "^^^" -> Some BitwiseXor
            | "op_LeftShift" | "<<<" -> Some ShiftLeft
            | "op_RightShift" | ">>>" -> Some ShiftRight
            | "op_UnaryNegation" | "~-" -> Some Negate
            | "not" -> Some Not
            | _ -> None
        | None -> None

/// Comparison operation types
type CompareOp =
    | Eq | Neq | Lt | Gt | Lte | Gte

/// Match comparison operators
let isCompareOp : Pattern<CompareOp> =
    fun z ->
        match z.Focus.Symbol with
        | Some symbol ->
            match symbol.DisplayName with
            | "op_Equality" | "=" -> Some Eq
            | "op_Inequality" | "<>" -> Some Neq
            | "op_LessThan" | "<" -> Some Lt
            | "op_GreaterThan" | ">" -> Some Gt
            | "op_LessThanOrEqual" | "<=" -> Some Lte
            | "op_GreaterThanOrEqual" | ">=" -> Some Gte
            | _ -> None
        | None -> None
