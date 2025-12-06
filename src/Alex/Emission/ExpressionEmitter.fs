/// ExpressionEmitter - Classifier-driven expression emission
///
/// Uses PSGPatterns predicates/extractors and AlloyPatterns classifiers
/// for pattern matching, with EmissionMonad's tryEmissions for dispatch.
///
/// ARCHITECTURE:
/// 1. Classifiers (PSGPatterns, AlloyPatterns) extract typed data from PSG nodes
/// 2. Each tryEmit* function is self-contained: classify then handle, returning Option
/// 3. tryEmissions dispatches to the first matching handler
/// 4. EmissionMonad threads state and builds MLIR output
module Alex.Emission.ExpressionEmitter

open Core.PSG.Types
open Alex.CodeGeneration.EmissionContext
open Alex.CodeGeneration.TypeMapping
open Alex.CodeGeneration.EmissionMonad
open Alex.Traversal.PSGZipper
open Alex.Patterns.PSGPatterns
open Alex.Patterns.AlloyPatterns

// ===================================================================
// Expression Emission Result
// ===================================================================

/// Result of emitting an expression
type ExprResult =
    | Value of ssa: string * mlirType: string
    | Void
    | Error of message: string

// ===================================================================
// Constant Extraction (from SyntaxKind metadata)
// ===================================================================

/// Extract string content from Const:String syntax kind
let private extractStringFromKind (kind: string) : string option =
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

/// Extract int32 from Const:Int32 syntax kind
let private extractInt32FromKind (kind: string) : int option =
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

/// Extract int64 from Const:Int64 syntax kind
let private extractInt64FromKind (kind: string) : int64 option =
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

// ===================================================================
// Forward declaration for mutual recursion
// ===================================================================

// emitExpr will be defined at the end after all handlers
// Handlers reference it via mutable cell for mutual recursion
let mutable private emitExprImpl : Emit<ExprResult> =
    fun _env state -> (state, Error "emitExpr not initialized")

/// Call emitExpr (used by handlers before emitExpr is defined)
let private callEmitExpr : Emit<ExprResult> =
    fun env state -> emitExprImpl env state

// ===================================================================
// Console Emission Handlers
// ===================================================================

/// Emit write syscall (Linux x86-64: syscall 1 = write)
let private emitWriteSyscall (fd: string) (buf: string) (len: string) : Emit<string> =
    emitI64 1L >>= fun sysnumReg ->
    freshSSAWithType "i64" >>= fun result ->
    line (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"=r,{rax},{rdi},{rsi},{rdx}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64"
        result sysnumReg fd buf len) >>.
    emit result

/// Emit Console.Write with string content
let private emitWriteString (content: string) : Emit<ExprResult> =
    registerStringLiteral content >>= fun globalName ->
    emitAddressOf globalName >>= fun ptr ->
    emitI64 (int64 content.Length) >>= fun len ->
    emitI64 1L >>= fun fd ->
    emitWriteSyscall fd ptr len >>= fun _ ->
    emit Void

/// Emit Console.WriteLine with string content (adds newline)
let private emitWriteLineString (content: string) : Emit<ExprResult> =
    emitWriteString (content + "\n")

/// Handler for Console operations
let private handleConsole (op: ConsoleOp) : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z
    match op with
    | Write ->
        match children with
        | _ :: argNode :: _ ->
            match extractStringFromKind argNode.SyntaxKind with
            | Some content -> emitWriteString content
            | None ->
                atNode argNode callEmitExpr >>= fun argResult ->
                match argResult with
                | Value(ssa, _) ->
                    line (sprintf "// Console.Write with dynamic content: %s" ssa) >>.
                    emit Void
                | _ -> emit Void
        | _ ->
            emit (Error "Console.Write: missing argument")
    | WriteLine ->
        match children with
        | _ :: argNode :: _ ->
            match extractStringFromKind argNode.SyntaxKind with
            | Some content -> emitWriteLineString content
            | None ->
                atNode argNode callEmitExpr >>= fun argResult ->
                match argResult with
                | Value(ssa, _) ->
                    line (sprintf "// Console.WriteLine with dynamic content: %s" ssa) >>.
                    emit Void
                | _ -> emit Void
        | _ ->
            emit (Error "Console.WriteLine: missing argument")
    | ReadLine ->
        line "// Console.ReadLine - not yet implemented" >>.
        emit Void
    | Read ->
        line "// Console.Read - not yet implemented" >>.
        emit Void

/// Try to emit as Console operation
let private tryEmitConsole : Emit<ExprResult option> =
    getZipper >>= fun z ->
    match extractConsoleOp z.Graph z.Focus with
    | Some op -> handleConsole op |>> Some
    | None -> emit None

// ===================================================================
// Time Emission Handlers
// ===================================================================

open Alex.Bindings.Time.Linux

/// Handler for Time operations
let private handleTime (op: TimeOp) : Emit<ExprResult> =
    match op with
    | CurrentTicks ->
        invokeBinding (fun builder ctx ->
            let ticks = emitClockGettime builder ctx ClockId.REALTIME
            let epochOffset = SSAContext.nextValue ctx
            let result = SSAContext.nextValue ctx
            MLIRBuilder.line builder (sprintf "%s = arith.constant 621355968000000000 : i64" epochOffset)
            MLIRBuilder.line builder (sprintf "%s = arith.addi %s, %s : i64" result ticks epochOffset)
            Value(result, "i64")
        )
    | CurrentUnixTimestamp ->
        invokeBinding (fun builder ctx ->
            let ticks = emitClockGettime builder ctx ClockId.REALTIME
            let ticksPerSec = SSAContext.nextValue ctx
            let result = SSAContext.nextValue ctx
            MLIRBuilder.line builder (sprintf "%s = arith.constant 10000000 : i64" ticksPerSec)
            MLIRBuilder.line builder (sprintf "%s = arith.divui %s, %s : i64" result ticks ticksPerSec)
            Value(result, "i64")
        )
    | CurrentDateTimeString ->
        invokeBinding (fun builder ctx ->
            let buf = emitCurrentDateTimeString builder ctx
            Value(buf, "!llvm.ptr")
        )
    | Sleep ->
        getZipper >>= fun z ->
        let children = PSGZipper.childNodes z
        match children with
        | _ :: argNode :: _ ->
            match extractInt32FromKind argNode.SyntaxKind with
            | Some ms ->
                emitI32 ms >>= fun msReg ->
                invokeBinding (fun builder ctx ->
                    emitNanosleep builder ctx msReg
                    Void
                )
            | None ->
                atNode argNode callEmitExpr >>= fun argResult ->
                match argResult with
                | Value(msReg, _) ->
                    invokeBinding (fun builder ctx ->
                        emitNanosleep builder ctx msReg
                        Void
                    )
                | _ -> emit (Error "Time.sleep: argument must produce value")
        | _ ->
            emit (Error "Time.sleep: missing argument")
    | _ ->
        line (sprintf "// Time.%A - not yet implemented" op) >>.
        emit Void

/// Try to emit as Time operation
let private tryEmitTime : Emit<ExprResult option> =
    getZipper >>= fun z ->
    match extractTimeOp z.Graph z.Focus with
    | Some op -> handleTime op |>> Some
    | None -> emit None

// ===================================================================
// Arithmetic and Comparison Handlers
// ===================================================================

/// Handler for arithmetic operations
let private handleArith (op: ArithOp) : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z
    match children with
    | _ :: leftNode :: rightNode :: _ ->
        atNode leftNode callEmitExpr >>= fun leftResult ->
        match leftResult with
        | Value(leftReg, leftType) ->
            atNode rightNode callEmitExpr >>= fun rightResult ->
            match rightResult with
            | Value(rightReg, _) ->
                let mlirOp = arithOpToMLIR op
                freshSSAWithType leftType >>= fun result ->
                line (sprintf "%s = %s %s, %s : %s" result mlirOp leftReg rightReg leftType) >>.
                emit (Value(result, leftType))
            | Error msg -> emit (Error msg)
            | Void -> emit (Error "Right operand must produce value")
        | Error msg -> emit (Error msg)
        | Void -> emit (Error "Left operand must produce value")
    | _ ->
        emit (Error "Arithmetic operation requires 2 operands")

/// Try to emit as arithmetic operation
let private tryEmitArith : Emit<ExprResult option> =
    getZipper >>= fun z ->
    match extractArithOp z.Graph z.Focus with
    | Some op -> handleArith op |>> Some
    | None -> emit None

/// Handler for comparison operations
let private handleCompare (op: CompareOp) : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z
    match children with
    | _ :: leftNode :: rightNode :: _ ->
        atNode leftNode callEmitExpr >>= fun leftResult ->
        match leftResult with
        | Value(leftReg, leftType) ->
            atNode rightNode callEmitExpr >>= fun rightResult ->
            match rightResult with
            | Value(rightReg, _) ->
                let pred = compareOpToPredicate op
                freshSSAWithType "i1" >>= fun result ->
                line (sprintf "%s = arith.cmpi %s, %s, %s : %s" result pred leftReg rightReg leftType) >>.
                emit (Value(result, "i1"))
            | Error msg -> emit (Error msg)
            | Void -> emit (Error "Right operand must produce value")
        | Error msg -> emit (Error msg)
        | Void -> emit (Error "Left operand must produce value")
    | _ ->
        emit (Error "Comparison operation requires 2 operands")

/// Try to emit as comparison operation
let private tryEmitCompare : Emit<ExprResult option> =
    getZipper >>= fun z ->
    match extractCompareOp z.Graph z.Focus with
    | Some op -> handleCompare op |>> Some
    | None -> emit None

// ===================================================================
// Constant Emission Handler
// ===================================================================

/// Handler for constant values
let private handleConst (kind: string) : Emit<ExprResult> =
    if kind.StartsWith("Const:Int32") then
        match extractInt32FromKind kind with
        | Some v -> emitI32 v |>> fun ssa -> Value(ssa, "i32")
        | None -> emit (Error "Invalid i32 constant")
    elif kind.StartsWith("Const:Int64") then
        match extractInt64FromKind kind with
        | Some v -> emitI64 v |>> fun ssa -> Value(ssa, "i64")
        | None -> emit (Error "Invalid i64 constant")
    elif kind.StartsWith("Const:String") then
        match extractStringFromKind kind with
        | Some content ->
            registerStringLiteral content >>= fun globalName ->
            emitAddressOf globalName |>> fun ptr ->
            Value(ptr, "!llvm.ptr")
        | None -> emit (Error "Invalid string constant")
    elif kind = "Const:Unit" || kind.StartsWith("Const:()") then
        emit Void
    else
        emit (Error (sprintf "Unknown constant: %s" kind))

/// Try to emit as constant
let private tryEmitConst : Emit<ExprResult option> =
    getFocus >>= fun node ->
    if isConst node then
        handleConst node.SyntaxKind |>> Some
    else
        emit None

// ===================================================================
// Variable Reference Handler
// ===================================================================

/// Handler for identifier/variable references
let private handleIdent (name: string) : Emit<ExprResult> =
    lookupLocal name >>= fun localOpt ->
    match localOpt with
    | Some ssaName ->
        lookupLocalType name >>= fun typeOpt ->
        emit (Value(ssaName, typeOpt |> Option.defaultValue "i32"))
    | None ->
        emit (Error (sprintf "Variable not found: %s" name))

/// Try to emit as identifier
let private tryEmitIdent : Emit<ExprResult option> =
    getFocus >>= fun node ->
    if isIdent node then
        match node.Symbol with
        | Some sym -> handleIdent sym.DisplayName |>> Some
        | None ->
            let kind = node.SyntaxKind
            let name =
                if kind.StartsWith("Ident:") then kind.Substring("Ident:".Length)
                elif kind.StartsWith("Value:") then kind.Substring("Value:".Length)
                elif kind.StartsWith("LongIdent:") then kind.Substring("LongIdent:".Length)
                else kind
            handleIdent name |>> Some
    else
        emit None

// ===================================================================
// Sequential Expression Handler
// ===================================================================

/// Handler for sequential expressions - emit all children, return last
let private handleSequential () : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z
    match children with
    | [] -> emit Void
    | [single] -> atNode single callEmitExpr
    | _ ->
        let allButLast = children |> List.take (List.length children - 1)
        let lastNode = children |> List.last
        forEach (fun child ->
            atNode child callEmitExpr >>= fun _ -> emit ()
        ) allButLast >>.
        atNode lastNode callEmitExpr

/// Try to emit as sequential
let private tryEmitSequential : Emit<ExprResult option> =
    getFocus >>= fun node ->
    if isSequential node then
        handleSequential () |>> Some
    else
        emit None

// ===================================================================
// Let Binding Handler
// ===================================================================

/// Emit a single Binding node
let private emitBindingNode (bindingNode: PSGNode) : Emit<ExprResult> =
    atNode bindingNode (
        getZipper >>= fun bz ->
        let bindingChildren = PSGZipper.childNodes bz
        match bindingChildren with
        | _ :: valueNode :: _ ->
            atNode valueNode callEmitExpr
        | [_patternOnly] ->
            emit Void
        | [] ->
            emit Void
    ) >>= fun valueResult ->
    match valueResult with
    | Value(ssa, typ) ->
        match bindingNode.Symbol with
        | Some symbol ->
            bindLocal symbol.DisplayName ssa typ >>.
            emit (Value(ssa, typ))
        | None ->
            emit (Value(ssa, typ))
    | other -> emit other

/// Handler for let bindings
let private handleLet () : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z

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
                atNode node callEmitExpr >>= fun result ->
                processChildren rest result

    processChildren children Void

/// Try to emit as let binding
let private tryEmitLet : Emit<ExprResult option> =
    getFocus >>= fun node ->
    if isLet node then
        handleLet () |>> Some
    else
        emit None

// ===================================================================
// If Expression Handler
// ===================================================================

/// Handler for if expressions
let private handleIf () : Emit<ExprResult> =
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
        atNode condNode callEmitExpr >>= fun condResult ->
        match condResult with
        | Value(condReg, _) ->
            match rest with
            | [elseNode] ->
                emitCondBr condReg thenLabel elseLabel >>.
                emitBlockLabel thenLabel >>.
                atNode thenNode callEmitExpr >>= fun thenResult ->
                emitBr mergeLabel >>.
                emitBlockLabel elseLabel >>.
                atNode elseNode callEmitExpr >>= fun _ ->
                emitBr mergeLabel >>.
                emitBlockLabel mergeLabel >>.
                emit thenResult
            | [] ->
                emitCondBr condReg thenLabel mergeLabel >>.
                emitBlockLabel thenLabel >>.
                atNode thenNode callEmitExpr >>= fun _ ->
                emitBr mergeLabel >>.
                emitBlockLabel mergeLabel >>.
                emit Void
            | _ -> emit (Error "If: unexpected structure")
        | _ -> emit (Error "If condition must produce value")
    | _ -> emit (Error "If: wrong number of children")

/// Try to emit as if expression
let private tryEmitIf : Emit<ExprResult option> =
    getFocus >>= fun node ->
    if isIf node then
        handleIf () |>> Some
    else
        emit None

// ===================================================================
// While Loop Handler
// ===================================================================

/// Handler for while loops
let private handleWhile () : Emit<ExprResult> =
    getState >>= fun state ->
    let labelNum = state.SSACounter
    let condLabel = sprintf "while_cond_%d" labelNum
    let bodyLabel = sprintf "while_body_%d" labelNum
    let exitLabel = sprintf "while_exit_%d" labelNum
    modifyState (fun s -> { s with SSACounter = s.SSACounter + 3 }) >>.

    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z

    match children with
    | condNode :: bodyNode :: _ ->
        emitBr condLabel >>.
        emitBlockLabel condLabel >>.
        atNode condNode callEmitExpr >>= fun condResult ->
        match condResult with
        | Value(condReg, _) ->
            emitCondBr condReg bodyLabel exitLabel >>.
            emitBlockLabel bodyLabel >>.
            atNode bodyNode callEmitExpr >>= fun _ ->
            emitBr condLabel >>.
            emitBlockLabel exitLabel >>.
            emit Void
        | Void ->
            line "// ERROR: While condition returned Void" >>.
            emit Void
        | Error msg ->
            emit (Error msg)
    | _ ->
        emit (Error "While: expected condition and body")

/// Try to emit as while loop
let private tryEmitWhile : Emit<ExprResult option> =
    getFocus >>= fun node ->
    if isWhile node then
        handleWhile () |>> Some
    else
        emit None

// ===================================================================
// Match Expression Handler
// ===================================================================

/// Handler for match expressions (simplified - handles first clause only for now)
let private handleMatch () : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z

    match children with
    | scrutineeNode :: clauses when clauses.Length > 0 ->
        atNode scrutineeNode callEmitExpr >>= fun scrutineeResult ->
        match scrutineeResult with
        | Value(scrutineeReg, scrutineeType) ->
            line (sprintf "// Match on %s : %s" scrutineeReg scrutineeType) >>.
            match clauses with
            | firstClause :: _ ->
                let clauseChildren =
                    match firstClause.Children with
                    | ChildrenState.Parent ids ->
                        ids |> List.rev |> List.choose (fun id -> Map.tryFind id.Value z.Graph.Nodes)
                    | _ -> []
                match clauseChildren |> List.filter (fun n -> not (n.SyntaxKind.StartsWith("Pattern:"))) with
                | bodyNode :: _ ->
                    atNode bodyNode callEmitExpr
                | [] ->
                    emit (Value(scrutineeReg, scrutineeType))
            | [] ->
                emit (Error "Match with no clauses")
        | Error msg -> emit (Error msg)
        | Void ->
            emit (Error "Match scrutinee must produce a value")
    | _ ->
        emit (Error "Match: expected scrutinee and clauses")

/// Try to emit as match expression
let private tryEmitMatch : Emit<ExprResult option> =
    getFocus >>= fun node ->
    if isMatch node then
        handleMatch () |>> Some
    else
        emit None

// ===================================================================
// Mutable Set Handler
// ===================================================================

/// Handler for mutable variable assignment
let private handleMutableSet (varName: string) : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z
    match children |> List.tryLast with
    | Some valueNode ->
        atNode valueNode callEmitExpr >>= fun valueResult ->
        match valueResult with
        | Value(ssa, typ) ->
            bindLocal varName ssa typ >>.
            emit Void
        | _ -> emit Void
    | None -> emit Void

/// Try to emit as mutable set
let private tryEmitMutableSet : Emit<ExprResult option> =
    getFocus >>= fun node ->
    match extractMutableSetName node with
    | Some varName -> handleMutableSet varName |>> Some
    | None -> emit None

// ===================================================================
// Type Application Handler
// ===================================================================

/// Handler for TypeApp (generic instantiation like stackBuffer<byte>)
let private handleTypeApp () : Emit<ExprResult> =
    getFocus >>= fun node ->
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z
    match children with
    | [innerNode] ->
        atNode innerNode callEmitExpr
    | [] ->
        match node.Symbol with
        | Some symbol ->
            let name = symbol.DisplayName
            lookupLocal name >>= fun localOpt ->
            match localOpt with
            | Some ssaName ->
                lookupLocalType name |>> fun typeOpt ->
                Value(ssaName, typeOpt |> Option.defaultValue "i32")
            | None ->
                emit (Value(sprintf "@%s" name, "func"))
        | None ->
            emit (Error "TypeApp without symbol or children")
    | _ ->
        atNode (List.head children) callEmitExpr

/// Try to emit as type application
let private tryEmitTypeApp : Emit<ExprResult option> =
    getFocus >>= fun node ->
    if isTypeApp node then
        handleTypeApp () |>> Some
    else
        emit None

// ===================================================================
// Pattern Node Handler
// ===================================================================

/// Try to emit as pattern (structural, returns Void)
let private tryEmitPattern : Emit<ExprResult option> =
    getFocus >>= fun node ->
    if isPattern node then
        emit (Some Void)
    else
        emit None

// ===================================================================
// Main Expression Dispatcher
// ===================================================================

/// Main expression emission using tryEmissions for dispatch
let emitExpr : Emit<ExprResult> =
    tryEmissions [
        // Console operations (highest priority for Alloy calls)
        tryEmitConsole

        // Time operations
        tryEmitTime

        // Arithmetic operations
        tryEmitArith

        // Comparison operations
        tryEmitCompare

        // Structural patterns
        tryEmitConst
        tryEmitIdent
        tryEmitSequential
        tryEmitLet
        tryEmitIf
        tryEmitWhile
        tryEmitMatch
        tryEmitTypeApp
        tryEmitMutableSet
        tryEmitPattern
    ] >>= fun resultOpt ->
    match resultOpt with
    | Some result -> emit result
    | None ->
        // No handler matched - this is a compiler error
        getFocus >>= fun node ->
        emit (Error (sprintf "Unhandled PSG node type: %s" node.SyntaxKind))

// Initialize the mutable reference for recursive calls
do emitExprImpl <- emitExpr

// ===================================================================
// Public API
// ===================================================================

/// Emit expression at current zipper focus (public alias)
let emitCurrentExpr : Emit<ExprResult> = emitExpr

/// Legacy emitExpr that takes explicit PSG and node
let emitExprLegacy (psg: ProgramSemanticGraph) (node: PSGNode) : Emit<ExprResult> =
    atNode node emitExpr
