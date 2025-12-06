/// ExpressionEmitter - Zipper-based expression emission
///
/// Uses PSGZipper for traversal and EmissionMonad for output.
/// The emission operates on the current zipper focus, using navigation
/// operations to traverse children.
module Alex.Emission.ExpressionEmitter

open Core.PSG.Types
open Alex.CodeGeneration.EmissionContext
open Alex.CodeGeneration.TypeMapping
open Alex.CodeGeneration.EmissionMonad
open Alex.Traversal.PSGZipper

// ═══════════════════════════════════════════════════════════════════
// Expression Emission Result
// ═══════════════════════════════════════════════════════════════════

/// Result of emitting an expression
type ExprResult =
    | Value of ssa: string * mlirType: string
    | Void
    | Error of message: string

// ═══════════════════════════════════════════════════════════════════
// Pattern Recognition (operates on current zipper focus)
// ═══════════════════════════════════════════════════════════════════

/// Check if current focus is a Console operation
let recognizeConsoleOp : Emit<string option> =
    getFocus |>> fun node ->
        match node.Symbol with
        | Some symbol ->
            let fullName = symbol.FullName
            let displayName = symbol.DisplayName
            if fullName.Contains("Console.WriteLine") || fullName.EndsWith(".WriteLine") ||
               displayName = "WriteLine" then
                Some "writeLine"
            elif fullName.Contains("Console.Write") || fullName.EndsWith(".Write") ||
                 displayName = "Write" then
                Some "write"
            elif fullName.Contains("Console.Prompt") || fullName.EndsWith(".Prompt") ||
                 displayName = "Prompt" then
                Some "prompt"
            else None
        | None -> None

/// Check if current focus is a Time operation
let recognizeTimeOp : Emit<string option> =
    getFocus |>> fun node ->
        match node.Symbol with
        | Some symbol ->
            let fullName = symbol.FullName
            if fullName.Contains("currentTicks") then Some "currentTicks"
            elif fullName.Contains("currentUnixTimestamp") then Some "currentUnixTimestamp"
            elif fullName.Contains("currentDateTimeString") then Some "currentDateTimeString"
            elif fullName.Contains("currentDateTime") then Some "currentDateTime"
            elif fullName.Contains("sleep") then Some "sleep"
            else None
        | None -> None

/// Check if current focus is an arithmetic operator
let recognizeArithOp : Emit<string option> =
    getFocus |>> fun node ->
        match node.Symbol with
        | Some symbol ->
            match symbol.DisplayName with
            | "op_Addition" | "(+)" | "Add" -> Some "addi"
            | "op_Subtraction" | "(-)" | "Subtract" -> Some "subi"
            | "op_Multiply" | "(*)" | "Multiply" -> Some "muli"
            | "op_Division" | "(/)" | "Divide" -> Some "divsi"
            | "op_Modulus" | "(%)" | "Modulo" -> Some "remsi"
            | _ -> None
        | None -> None

/// Check if current focus is a comparison operator
let recognizeCompareOp : Emit<string option> =
    getFocus |>> fun node ->
        match node.Symbol with
        | Some symbol ->
            match symbol.DisplayName with
            | "op_LessThan" | "(<)" -> Some "slt"
            | "op_GreaterThan" | "(>)" -> Some "sgt"
            | "op_LessThanOrEqual" | "(<=)" -> Some "sle"
            | "op_GreaterThanOrEqual" | "(>=)" -> Some "sge"
            | "op_Equality" | "(=)" -> Some "eq"
            | "op_Inequality" | "(<>)" -> Some "ne"
            | _ -> None
        | None -> None

// ═══════════════════════════════════════════════════════════════════
// Extraction Helpers
// ═══════════════════════════════════════════════════════════════════

/// Extract string constant from syntax kind
let extractStringConst (kind: string) : string option =
    // Format: Const:String ("content", Regular, (line,col--line,col))
    if kind.StartsWith("Const:String") then
        let start = kind.IndexOf("(\"")
        if start >= 0 then
            let contentStart = start + 2
            let endQuote = kind.IndexOf("\",", contentStart)
            if endQuote > contentStart then
                Some (kind.Substring(contentStart, endQuote - contentStart))
            else None
        else None
    else None

/// Extract int32 constant from syntax kind
let extractIntConst (kind: string) : int option =
    // Format: Const:Int32 64 or Const:Int32 64 : type
    if kind.StartsWith("Const:Int32 ") then
        let afterPrefix = kind.Substring("Const:Int32 ".Length)
        let valueStr =
            match afterPrefix.IndexOf(' ') with
            | -1 -> afterPrefix
            | idx -> afterPrefix.Substring(0, idx)
        match System.Int32.TryParse(valueStr) with
        | true, v -> Some v
        | false, _ -> None
    else None

/// Extract int64 constant from syntax kind
let extractInt64Const (kind: string) : int64 option =
    if kind.StartsWith("Const:Int64:") then
        let valueStr = kind.Substring("Const:Int64:".Length).TrimEnd('L')
        match System.Int64.TryParse(valueStr) with
        | true, v -> Some v
        | false, _ -> None
    elif kind.StartsWith("Const:Int64 ") then
        let valueStr = kind.Substring("Const:Int64 ".Length).TrimEnd('L')
        match System.Int64.TryParse(valueStr) with
        | true, v -> Some v
        | false, _ -> None
    else None

// ═══════════════════════════════════════════════════════════════════
// Console Emission (write syscall for Console only)
// TODO: Move to Alex.Bindings.Console.Linux
// ═══════════════════════════════════════════════════════════════════

/// Emit write syscall (for Console)
let emitWriteSyscall (fd: string) (buf: string) (len: string) : Emit<string> =
    emitI64 1L >>= fun sysnumReg ->
    freshSSAWithType "i64" >>= fun result ->
    line (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"=r,{rax},{rdi},{rsi},{rdx}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64"
        result sysnumReg fd buf len) >>.
    emit result

/// Emit Console.writeLine with string content
let emitConsoleWriteLineString (content: string) : Emit<ExprResult> =
    let contentWithNewline = content + "\n"
    registerStringLiteral contentWithNewline >>= fun globalName ->
    emitAddressOf globalName >>= fun ptr ->
    emitI64 (int64 contentWithNewline.Length) >>= fun len ->
    emitI64 1L >>= fun fd ->
    emitWriteSyscall fd ptr len >>= fun _ ->
    emit Void

// ═══════════════════════════════════════════════════════════════════
// Time Emission - Delegates to Alex.Bindings.Time.Linux
// ═══════════════════════════════════════════════════════════════════

open Alex.Bindings.Time.Linux

/// Emit Time.currentTicks via Bindings layer
let emitTimeCurrentTicks : Emit<ExprResult> =
    invokeBinding (fun builder ctx ->
        let ticks = emitClockGettime builder ctx ClockId.REALTIME
        // Add Unix epoch offset
        let epochOffset = SSAContext.nextValue ctx
        let result = SSAContext.nextValue ctx
        MLIRBuilder.line builder (sprintf "%s = arith.constant 621355968000000000 : i64" epochOffset)
        MLIRBuilder.line builder (sprintf "%s = arith.addi %s, %s : i64" result ticks epochOffset)
        Value(result, "i64")
    )

/// Emit Time.currentUnixTimestamp via Bindings layer
let emitTimeCurrentUnixTimestamp : Emit<ExprResult> =
    invokeBinding (fun builder ctx ->
        let ticks = emitClockGettime builder ctx ClockId.REALTIME
        let ticksPerSec = SSAContext.nextValue ctx
        let result = SSAContext.nextValue ctx
        MLIRBuilder.line builder (sprintf "%s = arith.constant 10000000 : i64" ticksPerSec)
        MLIRBuilder.line builder (sprintf "%s = arith.divui %s, %s : i64" result ticks ticksPerSec)
        Value(result, "i64")
    )

/// Emit Time.currentDateTimeString via Bindings layer
let emitTimeCurrentDateTimeString : Emit<ExprResult> =
    invokeBinding (fun builder ctx ->
        let buf = emitCurrentDateTimeString builder ctx
        Value(buf, "!llvm.ptr")
    )

/// Emit Time.sleep via Bindings layer
let emitTimeSleep (milliseconds: string) : Emit<ExprResult> =
    invokeBinding (fun builder ctx ->
        emitNanosleep builder ctx milliseconds
        Void
    )

// ═══════════════════════════════════════════════════════════════════
// Core Expression Emission (Zipper-based)
// ═══════════════════════════════════════════════════════════════════

/// Emit a constant from current zipper focus
let emitConst : Emit<ExprResult> =
    getFocus >>= fun node ->
    let kind = node.SyntaxKind
    if kind.StartsWith("Const:Int32") then
        match extractIntConst kind with
        | Some v -> emitI32 v |>> fun ssa -> Value(ssa, "i32")
        | None -> emit (Error "Invalid i32 constant")
    elif kind.StartsWith("Const:Int64") then
        match extractInt64Const kind with
        | Some v -> emitI64 v |>> fun ssa -> Value(ssa, "i64")
        | None -> emit (Error "Invalid i64 constant")
    elif kind.StartsWith("Const:String:") then
        match extractStringConst kind with
        | Some content ->
            registerStringLiteral content >>= fun globalName ->
            emitAddressOf globalName |>> fun ptr ->
            Value(ptr, "!llvm.ptr")
        | None -> emit (Error "Invalid string constant")
    elif kind = "Const:Unit" || kind.StartsWith("Const:()") then
        emit Void
    else
        emit (Error (sprintf "Unknown constant: %s" kind))

/// Emit a variable reference from current zipper focus
let emitVarRef : Emit<ExprResult> =
    getFocus >>= fun node ->
    match node.Symbol with
    | Some symbol ->
        let name = symbol.DisplayName
        lookupLocal name >>= fun localOpt ->
        match localOpt with
        | Some ssaName ->
            lookupLocalType name >>= fun typeOpt ->
            emit (Value(ssaName, typeOpt |> Option.defaultValue "i32"))
        | None ->
            emit (Error (sprintf "Variable not found: %s" name))
    | None ->
        emit (Error "Variable reference without symbol")

// ═══════════════════════════════════════════════════════════════════
// Operator Recognition
// ═══════════════════════════════════════════════════════════════════

open FSharp.Compiler.Symbols

/// Get the operator name from a symbol - uses CompiledName for operators
/// which gives us the canonical .NET name (op_LessThan vs (<))
let getOperatorName (symbol: FSharpSymbol) : string =
    match symbol with
    | :? FSharpMemberOrFunctionOrValue as mfv ->
        // For operators, CompiledName gives us op_LessThan etc.
        // DisplayName gives us (<) etc.
        mfv.CompiledName
    | _ -> symbol.DisplayName

/// Check if name matches an arithmetic operator
let isArithOp (name: string) : (string -> string -> string -> Emit<string>) option =
    match name with
    | "op_Addition" | "(+)" | "+" | "Add" -> Some emitAddi
    | "op_Subtraction" | "(-)" | "-" | "Subtract" -> Some emitSubi
    | "op_Multiply" | "(*)" | "*" | "Multiply" -> Some emitMuli
    | "op_Division" | "(/)" | "/" | "Divide" -> Some emitDivui
    | "op_Modulus" | "(%)" | "%" | "Modulo" -> Some emitRemui
    | _ -> None

/// Check if name matches a comparison operator
let isCompareOp (name: string) : string option =
    match name with
    | "op_LessThan" | "(<)" | "<" -> Some "slt"
    | "op_GreaterThan" | "(>)" | ">" -> Some "sgt"
    | "op_LessThanOrEqual" | "(<=)" | "<=" -> Some "sle"
    | "op_GreaterThanOrEqual" | "(>=)" | ">=" -> Some "sge"
    | "op_Equality" | "(=)" | "=" -> Some "eq"
    | "op_Inequality" | "(<>)" | "<>" -> Some "ne"
    | _ -> None

// ═══════════════════════════════════════════════════════════════════
// Pattern-based Recognition (using PSGPatterns)
// ═══════════════════════════════════════════════════════════════════

// Note: We use module alias to avoid shadowing EmissionMonad operators
module Patterns = Alex.Patterns.PSGPatterns

// ═══════════════════════════════════════════════════════════════════
// Expression Dispatcher (Zipper-based)
// ═══════════════════════════════════════════════════════════════════

/// Emit the expression at current zipper focus
/// This is the main dispatch function - all expression emission goes through here
let rec emitCurrentExpr : Emit<ExprResult> =
    getFocus >>= fun node ->
    let kind = node.SyntaxKind

    // First check for Alloy operations via pattern recognition
    recognizeConsoleOp >>= fun consoleOpOpt ->
    match consoleOpOpt with
    | Some "writeLine" -> emitConsoleWriteLine
    | Some "write" -> emitConsoleWrite
    | Some "prompt" -> emitConsolePrompt
    | Some _ -> emit (Error "Console operation not yet implemented")
    | None ->

    recognizeTimeOp >>= fun timeOpOpt ->
    match timeOpOpt with
    | Some "currentTicks" -> emitTimeCurrentTicks
    | Some "currentUnixTimestamp" -> emitTimeCurrentUnixTimestamp
    | Some "currentDateTimeString" -> emitTimeCurrentDateTimeString
    | Some "currentDateTime" -> emitTimeCurrentDateTimeString
    | Some "sleep" -> emitSleepFromArgs
    | Some op -> emit (Error (sprintf "Time.%s not yet implemented" op))
    | None ->

    // Regular expression dispatch based on SyntaxKind
    if kind.StartsWith("Const:") then
        emitConst
    elif kind.StartsWith("Value:") || kind.StartsWith("Ident:") then
        emitVarRef
    elif kind.StartsWith("Sequential") then
        emitSequential
    elif kind.StartsWith("Let") || kind = "LetOrUse" || kind.StartsWith("Binding") then
        emitLetBinding
    elif kind.StartsWith("If") then
        emitIfExpr
    elif kind.StartsWith("While") then
        emitWhileLoop
    elif kind.StartsWith("App") then
        emitApplication
    elif kind.StartsWith("TypeApp") then
        // TypeApp is generic instantiation (e.g., stackBuffer<byte>)
        // Look through to the inner expression
        emitTypeApp
    elif kind.StartsWith("LongIdent:") then
        // LongIdent is a qualified identifier reference
        emitVarRef
    elif kind.StartsWith("Pattern:") then
        // Skip pattern nodes - they're structural
        emit Void
    elif kind.StartsWith("MutableSet:") then
        emitMutableSet
    elif kind.StartsWith("Match") then
        emitMatchExpr
    else
        // For unhandled nodes, try to emit children
        emitChildrenSequentially emitCurrentExpr >>= fun resultOpt ->
        match resultOpt with
        | Some result -> emit result
        | None -> emit Void

/// Emit Console.writeLine by navigating to argument child
and emitConsoleWriteLine : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z
    match children with
    | _ :: argNode :: _ ->
        // Navigate to arg and extract string
        match extractStringConst argNode.SyntaxKind with
        | Some content -> emitConsoleWriteLineString content
        | None ->
            // Try to emit the arg and print its result
            atNode argNode emitCurrentExpr >>= fun argResult ->
            match argResult with
            | Value(ssa, _) ->
                // For non-string, emit a placeholder
                line (sprintf "// Console.writeLine with dynamic content: %s" ssa) >>.
                emit Void
            | _ -> emit Void
    | _ ->
        emit (Error "writeLine: missing argument")

/// Emit Console.write (without newline)
and emitConsoleWrite : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z
    match children with
    | _ :: argNode :: _ ->
        match extractStringConst argNode.SyntaxKind with
        | Some content ->
            // Write string without newline
            registerStringLiteral content >>= fun globalName ->
            emitAddressOf globalName >>= fun ptr ->
            emitI64 (int64 content.Length) >>= fun len ->
            emitI64 1L >>= fun fd ->
            emitWriteSyscall fd ptr len >>= fun _ ->
            emit Void
        | None ->
            // Try to emit the arg as a variable reference
            atNode argNode emitCurrentExpr >>= fun argResult ->
            match argResult with
            | Value(ssa, _) ->
                line (sprintf "// Console.write with dynamic content: %s" ssa) >>.
                emit Void
            | _ -> emit Void
    | _ ->
        emit (Error "write: missing argument")

/// Emit Console.Prompt (same as write but for prompts)
and emitConsolePrompt : Emit<ExprResult> =
    // Prompt is just Write (outputs without newline)
    emitConsoleWrite

/// Emit Time.sleep by extracting milliseconds argument
and emitSleepFromArgs : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z
    match children with
    | _ :: argNode :: _ ->
        match extractIntConst argNode.SyntaxKind with
        | Some v ->
            emitI32 v >>= fun msReg ->
            emitTimeSleep msReg
        | None -> emit (Error "sleep requires integer argument")
    | _ ->
        emit (Error "sleep: missing argument")

/// Emit sequential expression - emit all children, return last result
and emitSequential : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z
    match children with
    | [] -> emit Void
    | [single] -> atNode single emitCurrentExpr
    | _ ->
        let allButLast = children |> List.take (List.length children - 1)
        let lastNode = children |> List.last
        // Emit all but last for side effects
        forEach (fun child ->
            atNode child emitCurrentExpr >>= fun _ -> emit ()
        ) allButLast >>.
        // Emit last and return its result
        atNode lastNode emitCurrentExpr

/// Emit let binding using zipper navigation
///
/// LetOrUse:Let nodes have a FLAT list of children:
///   [Binding; Binding; ...; BodyExpr; BodyExpr; ...]
///
/// Each Binding node introduces a variable that must be in scope
/// for all subsequent siblings. We process left-to-right, binding
/// each variable before continuing.
and emitLetBinding : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z

    // Process children sequentially, accumulating bindings into scope
    let rec processChildren (nodes: PSGNode list) (lastResult: ExprResult) : Emit<ExprResult> =
        match nodes with
        | [] -> emit lastResult
        | node :: rest ->
            if node.SyntaxKind.StartsWith("Binding") then
                emitBindingNode node >>= fun result ->
                processChildren rest result
            elif node.SyntaxKind.StartsWith("Pattern:") then
                processChildren rest lastResult
            else
                atNode node emitCurrentExpr >>= fun result ->
                processChildren rest result

    processChildren children Void

/// Emit a single Binding node: evaluate value child and bind to scope
and emitBindingNode (bindingNode: PSGNode) : Emit<ExprResult> =
    // Binding structure: [Pattern, Value]
    // Navigate to binding, get value child (index 1), emit it
    atNode bindingNode (
        getZipper >>= fun bz ->
        let bindingChildren = PSGZipper.childNodes bz
        match bindingChildren with
        | _ :: valueNode :: _ ->
            // Emit the value expression
            atNode valueNode emitCurrentExpr
        | [_patternOnly] ->
            // Binding with no value (e.g., function parameter)
            emit Void
        | [] ->
            emit Void
    ) >>= fun valueResult ->
    // Bind the result to scope using the binding's symbol
    match valueResult with
    | Value(ssa, typ) ->
        match bindingNode.Symbol with
        | Some symbol ->
            bindLocal symbol.DisplayName ssa typ >>.
            emit (Value(ssa, typ))
        | None ->
            emit (Value(ssa, typ))
    | other -> emit other

/// Emit mutable variable assignment
and emitMutableSet : Emit<ExprResult> =
    getFocus >>= fun node ->
    let varName =
        if node.SyntaxKind.StartsWith("MutableSet:") then
            node.SyntaxKind.Substring("MutableSet:".Length)
        else ""

    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z
    match children |> List.tryLast with
    | Some valueNode ->
        atNode valueNode emitCurrentExpr >>= fun valueResult ->
        match valueResult with
        | Value(ssa, typ) ->
            // Update the local binding
            bindLocal varName ssa typ >>.
            emit Void
        | _ -> emit Void
    | None -> emit Void

/// Emit TypeApp (generic instantiation like stackBuffer<byte>)
/// TypeApp wraps an identifier with type arguments - we look through to the inner expr
and emitTypeApp : Emit<ExprResult> =
    getFocus >>= fun node ->
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z
    match children with
    | [innerNode] ->
        // TypeApp has single child - the actual expression
        atNode innerNode emitCurrentExpr
    | [] ->
        // No children - this TypeApp node itself may have the symbol
        // Return a reference to the function (will be applied with args later)
        match node.Symbol with
        | Some symbol ->
            let name = symbol.DisplayName
            lookupLocal name >>= fun localOpt ->
            match localOpt with
            | Some ssaName ->
                lookupLocalType name |>> fun typeOpt ->
                Value(ssaName, typeOpt |> Option.defaultValue "i32")
            | None ->
                // This is a function reference, not a local
                emit (Value(sprintf "@%s" name, "func"))
        | None ->
            emit (Error "TypeApp without symbol or children")
    | _ ->
        // Multiple children - emit first (should be the identifier)
        atNode (List.head children) emitCurrentExpr

/// Emit match expression
/// Structure: Match[scrutinee, MatchClause, MatchClause, ...]
and emitMatchExpr : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z

    match children with
    | scrutineeNode :: clauses when clauses.Length > 0 ->
        // Emit the scrutinee (the value being matched)
        atNode scrutineeNode emitCurrentExpr >>= fun scrutineeResult ->
        match scrutineeResult with
        | Value(scrutineeReg, scrutineeType) ->
            // For now, emit a simplified version that handles Ok/Error Result types
            // This is a placeholder - full pattern matching needs more work
            line (sprintf "// Match on %s : %s" scrutineeReg scrutineeType) >>.
            // For Result<T, E>, we check the discriminator
            // For now, just emit the first clause's body as a placeholder
            match clauses with
            | firstClause :: _ ->
                // Get the clause's body (skip the pattern)
                let clauseChildren =
                    match firstClause.Children with
                    | ChildrenState.Parent ids ->
                        ids |> List.rev |> List.choose (fun id -> Map.tryFind id.Value z.Graph.Nodes)
                    | _ -> []
                match clauseChildren |> List.filter (fun n -> not (n.SyntaxKind.StartsWith("Pattern:"))) with
                | bodyNode :: _ ->
                    atNode bodyNode emitCurrentExpr
                | [] ->
                    emit (Value(scrutineeReg, scrutineeType))
            | [] ->
                emit (Error "Match with no clauses")
        | Error msg -> emit (Error msg)
        | Void ->
            emit (Error "Match scrutinee must produce a value")
    | _ ->
        emit (Error "Match: expected scrutinee and clauses")

/// Emit if expression
and emitIfExpr : Emit<ExprResult> =
    getState >>= fun state ->
    let labelNum = state.SSACounter
    let thenLabel = sprintf "then_%d" labelNum
    let elseLabel = sprintf "else_%d" labelNum
    let mergeLabel = sprintf "merge_%d" labelNum
    modifyState (fun s -> { s with SSACounter = s.SSACounter + 3 }) >>.

    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z
    match children with
    | condNode :: thenNode :: rest ->
        atNode condNode emitCurrentExpr >>= fun condResult ->
        match condResult with
        | Value(condReg, _) ->
            match rest with
            | [elseNode] ->
                emitCondBr condReg thenLabel elseLabel >>.
                emitBlockLabel thenLabel >>.
                atNode thenNode emitCurrentExpr >>= fun thenResult ->
                emitBr mergeLabel >>.
                emitBlockLabel elseLabel >>.
                atNode elseNode emitCurrentExpr >>= fun _ ->
                emitBr mergeLabel >>.
                emitBlockLabel mergeLabel >>.
                emit thenResult
            | [] ->
                emitCondBr condReg thenLabel mergeLabel >>.
                emitBlockLabel thenLabel >>.
                atNode thenNode emitCurrentExpr >>= fun _ ->
                emitBr mergeLabel >>.
                emitBlockLabel mergeLabel >>.
                emit Void
            | _ -> emit (Error "If: unexpected structure")
        | _ -> emit (Error "If condition must produce value")
    | _ -> emit (Error "If: wrong number of children")

/// Emit while loop
and emitWhileLoop : Emit<ExprResult> =
    getState >>= fun state ->
    let labelNum = state.SSACounter
    let condLabel = sprintf "while_cond_%d" labelNum
    let bodyLabel = sprintf "while_body_%d" labelNum
    let exitLabel = sprintf "while_exit_%d" labelNum
    modifyState (fun s -> { s with SSACounter = s.SSACounter + 3 }) >>.

    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z

    // Debug: show what children we found
    let childDesc = children |> List.map (fun c -> c.SyntaxKind) |> String.concat ", "
    line (sprintf "// While loop children: [%s]" childDesc) >>.

    match children with
    | condNode :: bodyNode :: _ when children.Length >= 2 ->
        // More flexible matching - take first two non-pattern children
        emitBr condLabel >>.
        emitBlockLabel condLabel >>.
        line (sprintf "// Condition node: %s" condNode.SyntaxKind) >>.
        atNode condNode emitCurrentExpr >>= fun condResult ->
        line (sprintf "// Condition result: %A" condResult) >>.
        match condResult with
        | Value(condReg, _) ->
            emitCondBr condReg bodyLabel exitLabel >>.
            emitBlockLabel bodyLabel >>.
            atNode bodyNode emitCurrentExpr >>= fun _ ->
            emitBr condLabel >>.
            emitBlockLabel exitLabel >>.
            emit Void
        | Void ->
            // Condition returned void - might be an error in the condition emission
            line "// ERROR: Condition returned Void, expected comparison result" >>.
            emit Void
        | Error msg ->
            line (sprintf "// ERROR in condition: %s" msg) >>.
            emit (Error msg)
    | _ ->
        line (sprintf "// While: expected 2+ children, got %d" children.Length) >>.
        emit (Error "While: wrong number of children")

/// Emit a binary arithmetic operation given the operand nodes
/// Uses atNode to navigate to each operand and emit
and emitBinaryArithWithNodes
    (leftNode: PSGNode)
    (rightNode: PSGNode)
    (emitOp: string -> string -> string -> Emit<string>) : Emit<ExprResult> =
    atNode leftNode emitCurrentExpr >>= fun leftResult ->
    match leftResult with
    | Value(leftReg, leftType) ->
        atNode rightNode emitCurrentExpr >>= fun rightResult ->
        match rightResult with
        | Value(rightReg, _) ->
            emitOp leftReg rightReg leftType |>> fun result ->
            Value(result, leftType)
        | Error msg -> emit (Error msg)
        | Void -> emit (Error "Right operand must produce value")
    | Error msg -> emit (Error msg)
    | Void -> emit (Error "Left operand must produce value")

/// Emit a comparison operation given the operand nodes
and emitCompareWithNodes
    (leftNode: PSGNode)
    (rightNode: PSGNode)
    (pred: string) : Emit<ExprResult> =
    atNode leftNode emitCurrentExpr >>= fun leftResult ->
    match leftResult with
    | Value(leftReg, leftType) ->
        atNode rightNode emitCurrentExpr >>= fun rightResult ->
        match rightResult with
        | Value(rightReg, _) ->
            emitCmpi pred leftReg rightReg leftType |>> fun result ->
            Value(result, "i1")
        | Error msg -> emit (Error msg)
        | Void -> emit (Error "Right operand must produce value")
    | Error msg -> emit (Error msg)
    | Void -> emit (Error "Left operand must produce value")

/// Emit function application - uses pattern-based curried operator recognition
and emitApplication : Emit<ExprResult> =
    // Try to recognize curried binary operator pattern using PSGPatterns
    runPattern Patterns.isCurriedBinaryOp >>= fun curriedOpOpt ->
    match curriedOpOpt with
    | Some opInfo ->
        // Pattern gives us: Operator, LeftOperand, RightOperand
        let mfv = opInfo.Operator
        let compiledName = mfv.CompiledName

        // Check arithmetic operators
        match Patterns.arithmeticOp mfv with
        | Some mlirOp ->
            line (sprintf "// Curried arith: %s" mfv.DisplayName) >>.
            // Use the MLIR op to build the emission function
            let emitOp leftReg rightReg typ =
                freshSSAWithType typ >>= fun result ->
                line (sprintf "%s = %s %s, %s : %s" result mlirOp leftReg rightReg typ) >>.
                emit result
            emitBinaryArithWithNodes opInfo.LeftOperand opInfo.RightOperand emitOp
        | None ->

        // Check comparison operators
        match Patterns.comparisonOp mfv with
        | Some pred ->
            line (sprintf "// Curried comparison: %s" mfv.DisplayName) >>.
            emitCompareWithNodes opInfo.LeftOperand opInfo.RightOperand pred
        | None ->
            emit (Error (sprintf "Curried op not recognized: %s" compiledName))

    | None ->
        // Not a curried binary op - try direct operator application or function call
        getZipper >>= fun z ->
        let children = PSGZipper.childNodes z
        match children with
        | funcNode :: argNodes ->
            match funcNode.Symbol with
            | Some symbol ->
                let compiledName = getOperatorName symbol
                let displayName = symbol.DisplayName

                // Check arithmetic with 2+ args already available
                match isArithOp compiledName |> Option.orElse (isArithOp displayName) with
                | Some op when argNodes.Length >= 2 ->
                    emitBinaryOpViaZipper 1 2 op
                | Some _ ->
                    emit (Error (sprintf "Arith op needs 2 args: %s" displayName))
                | None ->

                // Check comparison with 2+ args available
                match isCompareOp compiledName |> Option.orElse (isCompareOp displayName) with
                | Some pred when argNodes.Length >= 2 ->
                    emitCompareOpViaZipper 1 2 pred
                | Some _ ->
                    emit (Error (sprintf "Compare op needs 2 args: %s" displayName))
                | None ->

                // Check if func is another App (nested application)
                if funcNode.SyntaxKind.StartsWith("App") then
                    atNode funcNode emitCurrentExpr
                else
                    emit (Error (sprintf "Function not supported: %s" displayName))
            | None ->
                // No symbol - might be nested App
                if funcNode.SyntaxKind.StartsWith("App") then
                    atNode funcNode emitCurrentExpr
                else
                    emit (Error "Function without symbol")
        | [] ->
            emit (Error "Empty application")

/// Helper for binary arithmetic operations using zipper navigation by child index
/// leftIdx and rightIdx are child indices in the current node
and emitBinaryOpViaZipper (leftIdx: int) (rightIdx: int)
    (op: string -> string -> string -> Emit<string>) : Emit<ExprResult> =
    atNthChild leftIdx emitCurrentExpr >>= fun leftResult ->
    match leftResult with
    | Error msg -> emit (Error msg)
    | Void -> emit (Error "Left operand must produce value")
    | Value(leftReg, leftType) ->
        atNthChild rightIdx emitCurrentExpr >>= fun rightResult ->
        match rightResult with
        | Error msg -> emit (Error msg)
        | Void -> emit (Error "Right operand must produce value")
        | Value(rightReg, _) ->
            op leftReg rightReg leftType |>> fun result ->
            Value(result, leftType)

/// Helper for comparison operations using zipper navigation by child index
and emitCompareOpViaZipper (leftIdx: int) (rightIdx: int) (pred: string) : Emit<ExprResult> =
    atNthChild leftIdx emitCurrentExpr >>= fun leftResult ->
    match leftResult with
    | Error msg -> emit (Error msg)
    | Void -> emit (Error "Left operand must produce value")
    | Value(leftReg, leftType) ->
        atNthChild rightIdx emitCurrentExpr >>= fun rightResult ->
        match rightResult with
        | Error msg -> emit (Error msg)
        | Void -> emit (Error "Right operand must produce value")
        | Value(rightReg, _) ->
            emitCmpi pred leftReg rightReg leftType |>> fun result ->
            Value(result, "i1")

// ═══════════════════════════════════════════════════════════════════
// Legacy Compatibility (for gradual migration)
// ═══════════════════════════════════════════════════════════════════

/// Legacy emitExpr that takes explicit PSG and node
/// Internally creates a zipper and uses the new zipper-based emission
let emitExpr (psg: ProgramSemanticGraph) (node: PSGNode) : Emit<ExprResult> =
    atNode node emitCurrentExpr
