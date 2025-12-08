/// PSGScribe - Transcribes PSG to MLIR via Zipper traversal and XParsec pattern matching
///
/// ARCHITECTURAL PRINCIPLES:
/// 1. Zipper provides tree traversal with context preservation (up/down/left/right)
/// 2. XParsec provides composable pattern matching on children at each position
/// 3. Bindings provide platform-specific MLIR generation for extern primitives
/// 4. Scribe orchestrates these three - it does NOT have its own traversal logic
///
/// The Scribe does NOT:
/// - Do ad-hoc recursive traversal (that's the Zipper's job)
/// - Have its own state type (uses EmitContext from PSGXParsec)
/// - Pattern match on strings (uses typed SyntaxKind and XParsec combinators)
/// - Know about specific library names (dispatches based on PSG structure)
module Alex.Traversal.PSGScribe

open System.Text
open FSharp.Compiler.Symbols
open Core.PSG.Types
open Alex.Traversal.PSGZipper
open Alex.Traversal.PSGXParsec
open Alex.CodeGeneration.MLIRBuilder
open Alex.Bindings.BindingTypes

// ═══════════════════════════════════════════════════════════════════════════
// Type-Driven Pattern Matchers - Match on typed SyntaxKind, not strings
// ═══════════════════════════════════════════════════════════════════════════

/// Match a node by its typed Kind field
let kindIs (expected: SyntaxKindT) : PSGChildParser<PSGNode> =
    satisfyChild (fun n -> n.Kind = expected)

/// Match any expression node
let anyExpr : PSGChildParser<PSGNode> =
    satisfyChild (fun n -> match n.Kind with SKExpr _ -> true | _ -> false)

/// Match specific expression kind
let exprKind (ek: ExprKind) : PSGChildParser<PSGNode> =
    kindIs (SKExpr ek)

/// Match any pattern node
let anyPattern : PSGChildParser<PSGNode> =
    satisfyChild (fun n -> match n.Kind with SKPattern _ -> true | _ -> false)

/// Match any binding node
let anyBinding : PSGChildParser<PSGNode> =
    satisfyChild (fun n -> match n.Kind with SKBinding _ -> true | _ -> false)

/// Match any declaration node
let anyDecl : PSGChildParser<PSGNode> =
    satisfyChild (fun n -> match n.Kind with SKDecl _ -> true | _ -> false)

// ═══════════════════════════════════════════════════════════════════════════
// Symbol Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Check if a symbol has DllImport("__fidelity") attribute
let private hasFidelityDllImport (sym: FSharpSymbol) : bool =
    match sym with
    | :? FSharpMemberOrFunctionOrValue as mfv ->
        mfv.Attributes
        |> Seq.exists (fun attr ->
            attr.AttributeType.FullName = "System.Runtime.InteropServices.DllImportAttribute" &&
            attr.ConstructorArguments
            |> Seq.exists (fun (_, arg) ->
                match arg with
                | :? string as s -> s = "__fidelity"
                | _ -> false))
    | _ -> false

/// Get the entry point name from a DllImport attribute
let private getExternEntryPoint (sym: FSharpSymbol) : string option =
    match sym with
    | :? FSharpMemberOrFunctionOrValue as mfv ->
        mfv.Attributes
        |> Seq.tryPick (fun attr ->
            if attr.AttributeType.FullName = "System.Runtime.InteropServices.DllImportAttribute" then
                attr.NamedArguments
                |> Seq.tryPick (fun (_, name, _, value) ->
                    if name = "EntryPoint" then
                        match value with
                        | :? string as s -> Some s
                        | _ -> None
                    else None)
                |> Option.orElse (Some mfv.DisplayName)
            else None)
    | _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// Scribe Context - Shared mutable state for MLIR emission
// ═══════════════════════════════════════════════════════════════════════════

/// Shared context for the entire emission pass
/// This is passed through all emission functions to accumulate MLIR output
type ScribeContext = {
    Graph: ProgramSemanticGraph
    mutable Output: StringBuilder
    mutable StringLiterals: (string * string) list
}

module ScribeContext =
    let create (psg: ProgramSemanticGraph) = {
        Graph = psg
        Output = StringBuilder()
        StringLiterals = []
    }

    let emit (ctx: ScribeContext) (line: string) =
        ctx.Output.AppendLine("    " + line) |> ignore

    let emitNoIndent (ctx: ScribeContext) (line: string) =
        ctx.Output.AppendLine(line) |> ignore

    let registerString (ctx: ScribeContext) (content: string) : string =
        match ctx.StringLiterals |> List.tryFind (fun (c, _) -> c = content) with
        | Some (_, name) -> name
        | None ->
            let name = sprintf "@str%d" (List.length ctx.StringLiterals)
            ctx.StringLiterals <- (content, name) :: ctx.StringLiterals
            name

/// Carry context through the traversal
let mutable private currentContext : ScribeContext option = None

let private emit (line: string) =
    match currentContext with
    | Some ctx -> ScribeContext.emit ctx line
    | None -> failwith "No active ScribeContext"

let private registerString (content: string) : string =
    match currentContext with
    | Some ctx -> ScribeContext.registerString ctx content
    | None -> failwith "No active ScribeContext"

// ═══════════════════════════════════════════════════════════════════════════
// Zipper-Based Traversal Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Find a node's body by following ChildOf edges (for bindings)
let private findBody (psg: ProgramSemanticGraph) (node: PSGNode) : PSGNode option =
    let childEdges = psg.Edges |> List.filter (fun e ->
        e.Source = node.Id && e.Kind = EdgeKind.ChildOf)
    childEdges |> List.tryPick (fun e ->
        match Map.tryFind e.Target.Value psg.Nodes with
        | Some n when not (n.SyntaxKind.StartsWith("Pattern:")) -> Some n
        | _ -> None)

/// Follow SymbolUse edge to find definition
let private findDefinition (psg: ProgramSemanticGraph) (useNode: PSGNode) : PSGNode option =
    let edge = psg.Edges |> List.tryFind (fun e -> e.Source = useNode.Id && e.Kind = EdgeKind.SymbolUse)
    eprintfn "[SCRIBE] findDefinition for %s (id=%A): edge=%A" useNode.SyntaxKind useNode.Id (edge |> Option.map (fun e -> e.Target))
    edge |> Option.bind (fun e -> Map.tryFind e.Target.Value psg.Nodes)

/// Check if a node is an extern primitive
let private isExtern (node: PSGNode) : bool =
    node.Symbol |> Option.map hasFidelityDllImport |> Option.defaultValue false

// ═══════════════════════════════════════════════════════════════════════════
// Core Emission Functions - Zipper-based traversal with XParsec patterns
// ═══════════════════════════════════════════════════════════════════════════

/// Emit a node - main dispatch based on typed Kind
let rec private emitNode (zipper: PSGZipper) : PSGZipper * ExprResult =
    let node = zipper.Focus
    eprintfn "[SCRIBE] emitNode: %s (Kind: %A, Operation: %A)" node.SyntaxKind node.Kind node.Operation

    // Dispatch based on typed Kind field
    match node.Kind with
    | SKExpr EConst -> emitConst zipper node
    | SKExpr EApp -> emitApp zipper node
    | SKExpr ETypeApp -> emitTypeApp zipper node
    | SKExpr EIdent -> emitIdent zipper node
    | SKExpr ELongIdent -> emitIdent zipper node
    | SKExpr ELetOrUse -> emitLetOrUse zipper node
    | SKExpr ESequential -> emitSequential zipper node
    | SKExpr EMutableSet -> emitMutableSet zipper node
    | SKExpr EWhileLoop -> emitWhileLoop zipper node
    | SKExpr EForLoop -> emitForLoop zipper node
    | SKExpr EAddressOf -> emitAddressOf zipper node
    | SKBinding _ -> emitBinding zipper node
    | SKPattern _ -> zipper, Void  // Patterns are structural, don't emit
    | SKDecl _ -> zipper, Void     // Declarations handled separately
    | SKUnknown -> emitUnknown zipper node
    | _ -> zipper, EmitError (sprintf "Unhandled Kind: %A for node %s" node.Kind node.Id.Value)

/// Emit a constant
and private emitConst (zipper: PSGZipper) (node: PSGNode) : PSGZipper * ExprResult =
    match node.ConstantValue with
    | Some (StringValue s) ->
        let globalName = registerString s
        let newState, ssa = EmissionState.nextSSA zipper.State
        emit (sprintf "%s = llvm.mlir.addressof %s : !llvm.ptr" ssa globalName)
        { zipper with State = newState }, Value (ssa, "!llvm.ptr")

    | Some (Int32Value v) ->
        let newState, ssa = EmissionState.nextSSA zipper.State
        emit (sprintf "%s = arith.constant %d : i32" ssa v)
        { zipper with State = newState }, Value (ssa, "i32")

    | Some (Int64Value v) ->
        let newState, ssa = EmissionState.nextSSA zipper.State
        emit (sprintf "%s = arith.constant %d : i64" ssa v)
        { zipper with State = newState }, Value (ssa, "i64")

    | Some (ByteValue v) ->
        let newState, ssa = EmissionState.nextSSA zipper.State
        emit (sprintf "%s = arith.constant %d : i8" ssa (int v))
        { zipper with State = newState }, Value (ssa, "i8")

    | Some (BoolValue v) ->
        let newState, ssa = EmissionState.nextSSA zipper.State
        let mlirVal = if v then 1 else 0
        emit (sprintf "%s = arith.constant %d : i1" ssa mlirVal)
        { zipper with State = newState }, Value (ssa, "i1")

    | Some UnitValue -> zipper, Void

    | None ->
        // Fallback: try to parse from SyntaxKind
        zipper, EmitError (sprintf "Const node %s has no ConstantValue" node.Id.Value)

    | _ -> zipper, EmitError (sprintf "Unhandled ConstantValue in node %s" node.Id.Value)

/// Emit an identifier reference - look up via def-use edges
and private emitIdent (zipper: PSGZipper) (node: PSGNode) : PSGZipper * ExprResult =
    // First check if we have this node's SSA already
    match EmissionState.lookupNodeSSA node.Id zipper.State with
    | Some (ssa, ty) -> zipper, Value (ssa, ty)
    | None ->
        // Follow SymbolUse edge to find definition
        let defEdge = zipper.Graph.Edges |> List.tryFind (fun e ->
            e.Source = node.Id && e.Kind = EdgeKind.SymbolUse)
        match defEdge with
        | Some edge ->
            match EmissionState.lookupNodeSSA edge.Target zipper.State with
            | Some (ssa, ty) -> zipper, Value (ssa, ty)
            | None ->
                // Definition not yet emitted - this might be a function reference
                zipper, Void
        | None -> zipper, Void

/// Emit a let binding - traverse children with proper state threading
and private emitLetOrUse (zipper: PSGZipper) (node: PSGNode) : PSGZipper * ExprResult =
    // LetOrUse children: Binding nodes followed by body expression
    // Traverse children sequentially, threading state
    let children = PSGZipper.childNodes zipper

    let rec emitChildren z result children' =
        match children' with
        | [] -> z, result
        | [last] ->
            // Last child is the result
            match PSGZipper.downTo (List.length children - 1) z with
            | NavOk childZ ->
                let z', r = emitNode childZ
                // Go back up
                match PSGZipper.up z' with
                | NavOk parentZ -> parentZ, r
                | NavFail _ -> z', r
            | NavFail msg -> z, EmitError msg
        | child :: rest ->
            // Emit this child, threading state
            let idx = List.length children - List.length children'
            match PSGZipper.downTo idx z with
            | NavOk childZ ->
                let z', _ = emitNode childZ
                match PSGZipper.up z' with
                | NavOk parentZ -> emitChildren parentZ result rest
                | NavFail _ -> emitChildren z' result rest
            | NavFail msg -> z, EmitError msg

    emitChildren zipper Void children

/// Emit a sequential expression - emit children in order, return last result
and private emitSequential (zipper: PSGZipper) (node: PSGNode) : PSGZipper * ExprResult =
    let children = PSGZipper.childNodes zipper

    let rec emitSeq z lastResult idx =
        if idx >= List.length children then
            z, lastResult
        else
            match PSGZipper.downTo idx z with
            | NavOk childZ ->
                let z', result = emitNode childZ
                match PSGZipper.up z' with
                | NavOk parentZ -> emitSeq parentZ result (idx + 1)
                | NavFail _ -> emitSeq z' result (idx + 1)
            | NavFail msg -> z, EmitError msg

    emitSeq zipper Void 0

/// Emit a binding - evaluate body and record SSA for this binding
and private emitBinding (zipper: PSGZipper) (node: PSGNode) : PSGZipper * ExprResult =
    // Find body child (non-pattern)
    let children = PSGZipper.childNodes zipper
    let bodyIdx = children |> List.tryFindIndex (fun c ->
        not (c.SyntaxKind.StartsWith("Pattern:")))

    match bodyIdx with
    | Some idx ->
        match PSGZipper.downTo idx zipper with
        | NavOk childZ ->
            let z', result = emitNode childZ
            // Record this binding's SSA value
            let z'' =
                match result with
                | Value (ssa, ty) ->
                    PSGZipper.recordNodeSSA node.Id ssa ty z'
                | _ -> z'
            match PSGZipper.up z'' with
            | NavOk parentZ -> parentZ, result
            | NavFail _ -> z'', result
        | NavFail msg -> zipper, EmitError msg
    | None -> zipper, Void

/// Emit an application (function call)
and private emitApp (zipper: PSGZipper) (node: PSGNode) : PSGZipper * ExprResult =
    // Check Operation field for classified operations
    match node.Operation with
    | Some op -> emitClassifiedOp zipper node op
    | None -> emitRegularCall zipper node

/// Emit a classified operation (Console, NativePtr, etc.)
and private emitClassifiedOp (zipper: PSGZipper) (node: PSGNode) (op: OperationKind) : PSGZipper * ExprResult =
    match op with
    | Console consoleOp -> emitConsoleOp zipper node consoleOp
    | NativePtr ptrOp -> emitNativePtrOp zipper node ptrOp
    | Arithmetic arithOp -> emitArithOp zipper node arithOp
    | Core coreOp -> emitCoreOp zipper node coreOp
    | Memory memOp -> emitMemoryOp zipper node memOp
    | _ -> zipper, EmitError (sprintf "Unhandled operation kind: %A" op)

/// Emit console operations
and private emitConsoleOp (zipper: PSGZipper) (node: PSGNode) (op: ConsoleOp) : PSGZipper * ExprResult =
    let children = PSGZipper.childNodes zipper
    // Filter out function identifiers to get argument nodes
    let argNodes = children |> List.filter (fun c ->
        not (c.SyntaxKind.StartsWith("LongIdent")) &&
        not (c.SyntaxKind.StartsWith("Ident:")) &&
        not (c.SyntaxKind.StartsWith("TypeApp")))

    match op with
    | ConsoleWrite | ConsoleWriteln ->
        // Find function definition and inline it
        eprintfn "[SCRIBE] ConsoleWrite/Writeln - children: %A" (children |> List.map (fun c -> c.SyntaxKind))
        let funcNode = children |> List.tryFind (fun c ->
            c.SyntaxKind.StartsWith("LongIdent") || c.SyntaxKind.StartsWith("Ident:"))
        eprintfn "[SCRIBE] funcNode: %A" (funcNode |> Option.map (fun n -> n.SyntaxKind))
        match funcNode with
        | Some fn ->
            let defNode = findDefinition zipper.Graph fn
            eprintfn "[SCRIBE] defNode: %A" (defNode |> Option.map (fun n -> n.SyntaxKind))
            match defNode with
            | Some defNode ->
                let bodyNode = findBody zipper.Graph defNode
                eprintfn "[SCRIBE] bodyNode: %A" (bodyNode |> Option.map (fun n -> n.SyntaxKind))
                match bodyNode with
                | Some body ->
                    // Create zipper focused on body and emit
                    let bodyZ = PSGZipper.createWithState zipper.Graph body zipper.State
                    let z', result = emitNode bodyZ
                    eprintfn "[SCRIBE] Console body emit result: %A" result
                    // Merge state back
                    { zipper with State = z'.State }, result
                | None -> zipper, EmitError "No body found for Console function"
            | None -> zipper, EmitError "No definition found for Console function"
        | None -> zipper, EmitError "No function identifier in Console call"

    | ConsoleWriteBytes ->
        // writeBytes(fd, ptr, count) -> syscall
        if argNodes.Length >= 3 then
            // Emit args
            let z1, fdResult = emitArgAt zipper argNodes 0
            let z2, ptrResult = emitArgAt z1 argNodes 1
            let z3, countResult = emitArgAt z2 argNodes 2

            match fdResult, ptrResult, countResult with
            | Value (fdSSA, _), Value (ptrSSA, _), Value (countSSA, _) ->
                // Emit write syscall
                let newState, resultSSA = EmissionState.nextSSA z3.State
                emit (sprintf "// write(%s, %s, %s)" fdSSA ptrSSA countSSA)
                emit (sprintf "%s = llvm.call @fidelity_write_bytes(%s, %s, %s) : (i32, !llvm.ptr, i32) -> i32"
                    resultSSA fdSSA ptrSSA countSSA)
                { z3 with State = newState }, Value (resultSSA, "i32")
            | _ -> z3, EmitError "Failed to emit writeBytes arguments"
        else
            zipper, EmitError (sprintf "writeBytes expects 3 args, got %d" argNodes.Length)

    | ConsoleNewLine ->
        // Inline newLine function body
        let funcNode = children |> List.tryFind (fun c ->
            c.SyntaxKind.StartsWith("LongIdent") || c.SyntaxKind.StartsWith("Ident:"))
        match funcNode with
        | Some fn ->
            match findDefinition zipper.Graph fn with
            | Some defNode ->
                match findBody zipper.Graph defNode with
                | Some body ->
                    let bodyZ = PSGZipper.createWithState zipper.Graph body zipper.State
                    let z', result = emitNode bodyZ
                    { zipper with State = z'.State }, result
                | None -> zipper, EmitError "No body for newLine"
            | None -> zipper, EmitError "No definition for newLine"
        | None -> zipper, EmitError "No function in newLine call"

    | _ -> zipper, EmitError (sprintf "Unhandled console op: %A" op)

/// Emit an argument at index
and private emitArgAt (zipper: PSGZipper) (args: PSGNode list) (idx: int) : PSGZipper * ExprResult =
    if idx >= List.length args then
        zipper, EmitError (sprintf "Arg index %d out of range" idx)
    else
        let argNode = args.[idx]
        let argZ = PSGZipper.createWithState zipper.Graph argNode zipper.State
        let z', result = emitNode argZ
        { zipper with State = z'.State }, result

/// Emit NativePtr operations
and private emitNativePtrOp (zipper: PSGZipper) (node: PSGNode) (op: NativePtrOp) : PSGZipper * ExprResult =
    let children = PSGZipper.childNodes zipper
    let argNodes = children |> List.filter (fun c ->
        not (c.SyntaxKind.StartsWith("LongIdent")) &&
        not (c.SyntaxKind.StartsWith("Ident:")) &&
        not (c.SyntaxKind.StartsWith("TypeApp")))

    match op with
    | PtrToVoidPtr | PtrOfVoidPtr | PtrToNativeInt | PtrOfNativeInt ->
        // These are essentially casts - emit the argument and return it
        if argNodes.Length >= 1 then
            emitArgAt zipper argNodes 0
        else
            zipper, EmitError (sprintf "%A expects at least 1 arg" op)

    | PtrStackAlloc ->
        // Stack allocation
        let newState, ssa = EmissionState.nextSSA zipper.State
        emit (sprintf "%s = llvm.alloca 256 x i8 : (i64) -> !llvm.ptr" ssa)
        { zipper with State = newState }, Value (ssa, "!llvm.ptr")

    | _ -> zipper, EmitError (sprintf "Unhandled NativePtr op: %A" op)

/// Emit type application
and private emitTypeApp (zipper: PSGZipper) (node: PSGNode) : PSGZipper * ExprResult =
    match node.Operation with
    | Some (NativePtr op) -> emitNativePtrOp zipper node op
    | Some op -> emitClassifiedOp zipper node op
    | None ->
        // Unclassified TypeApp - try to emit the inner expression
        let children = PSGZipper.childNodes zipper
        match children with
        | [inner] ->
            let innerZ = PSGZipper.createWithState zipper.Graph inner zipper.State
            let z', result = emitNode innerZ
            { zipper with State = z'.State }, result
        | _ -> zipper, Void

/// Emit a regular (non-classified) function call
and private emitRegularCall (zipper: PSGZipper) (node: PSGNode) : PSGZipper * ExprResult =
    let children = PSGZipper.childNodes zipper
    let funcNode = children |> List.tryFind (fun c ->
        c.SyntaxKind.StartsWith("LongIdent") || c.SyntaxKind.StartsWith("Ident:"))

    match funcNode with
    | Some fn ->
        // Check if extern primitive
        match fn.Symbol with
        | Some sym when hasFidelityDllImport sym ->
            // Emit extern call
            emitExternCall zipper node fn sym
        | _ ->
            // Follow call graph - inline the function
            match findDefinition zipper.Graph fn with
            | Some defNode ->
                match findBody zipper.Graph defNode with
                | Some body ->
                    let bodyZ = PSGZipper.createWithState zipper.Graph body zipper.State
                    let z', result = emitNode bodyZ
                    { zipper with State = z'.State }, result
                | None -> zipper, Void
            | None -> zipper, Void
    | None ->
        // No function identifier - emit children
        emitSequential zipper node

/// Emit extern primitive call
and private emitExternCall (zipper: PSGZipper) (node: PSGNode) (funcNode: PSGNode) (sym: FSharpSymbol) : PSGZipper * ExprResult =
    let entryPoint = getExternEntryPoint sym |> Option.defaultValue "unknown"
    let children = PSGZipper.childNodes zipper
    let argNodes = children |> List.filter (fun c ->
        not (c.SyntaxKind.StartsWith("LongIdent")) &&
        not (c.SyntaxKind.StartsWith("Ident:")) &&
        not (c.SyntaxKind.StartsWith("TypeApp")))

    // Emit all arguments
    let mutable z = zipper
    let mutable argResults = []
    for i in 0 .. argNodes.Length - 1 do
        let z', result = emitArgAt z argNodes i
        z <- { z with State = z'.State }
        argResults <- argResults @ [result]

    // Build extern call
    let argSSAs = argResults |> List.choose (function Value (s, _) -> Some s | _ -> None)
    let argsStr = String.concat ", " argSSAs

    let newState, resultSSA = EmissionState.nextSSA z.State
    emit (sprintf "%s = llvm.call @%s(%s) : (...) -> i32" resultSSA entryPoint argsStr)
    { z with State = newState }, Value (resultSSA, "i32")

/// Emit mutable set
and private emitMutableSet (zipper: PSGZipper) (node: PSGNode) : PSGZipper * ExprResult =
    zipper, EmitError "MutableSet not yet implemented"

/// Emit while loop
and private emitWhileLoop (zipper: PSGZipper) (node: PSGNode) : PSGZipper * ExprResult =
    zipper, EmitError "WhileLoop not yet implemented"

/// Emit for loop
and private emitForLoop (zipper: PSGZipper) (node: PSGNode) : PSGZipper * ExprResult =
    zipper, EmitError "ForLoop not yet implemented"

/// Emit AddressOf (&&expr)
and private emitAddressOf (zipper: PSGZipper) (node: PSGNode) : PSGZipper * ExprResult =
    let children = PSGZipper.childNodes zipper
    match children with
    | [inner] ->
        // Emit inner expression
        let innerZ = PSGZipper.createWithState zipper.Graph inner zipper.State
        let z', result = emitNode innerZ

        match result with
        | Value (valueSSA, valueType) ->
            // Allocate stack space for the value
            let state1, ptrSSA = EmissionState.nextSSA z'.State
            emit (sprintf "%s = llvm.alloca 1 x %s : (i64) -> !llvm.ptr" ptrSSA valueType)
            // Store the value
            emit (sprintf "llvm.store %s, %s : %s, !llvm.ptr" valueSSA ptrSSA valueType)
            { zipper with State = state1 }, Value (ptrSSA, "!llvm.ptr")
        | _ ->
            { zipper with State = z'.State }, EmitError "AddressOf inner returned no value"
    | _ ->
        zipper, EmitError (sprintf "AddressOf expects 1 child, got %d" (List.length children))

/// Emit arithmetic operation
and private emitArithOp (zipper: PSGZipper) (node: PSGNode) (op: ArithmeticOp) : PSGZipper * ExprResult =
    zipper, EmitError (sprintf "ArithOp %A not yet implemented" op)

/// Emit core operation
and private emitCoreOp (zipper: PSGZipper) (node: PSGNode) (op: CoreOp) : PSGZipper * ExprResult =
    match op with
    | Ignore ->
        // Evaluate argument and discard
        let children = PSGZipper.childNodes zipper
        let argNodes = children |> List.filter (fun c ->
            not (c.SyntaxKind.StartsWith("LongIdent")) &&
            not (c.SyntaxKind.StartsWith("Ident:")))
        if argNodes.Length >= 1 then
            let z', _ = emitArgAt zipper argNodes 0
            z', Void
        else
            zipper, Void
    | _ -> zipper, EmitError (sprintf "CoreOp %A not yet implemented" op)

/// Emit memory operation
and private emitMemoryOp (zipper: PSGZipper) (node: PSGNode) (op: MemoryOp) : PSGZipper * ExprResult =
    zipper, EmitError (sprintf "MemoryOp %A not yet implemented" op)

/// Handle unknown node kinds - fallback to string matching for migration
and private emitUnknown (zipper: PSGZipper) (node: PSGNode) : PSGZipper * ExprResult =
    // Fallback for nodes not yet converted to typed Kind
    let sk = node.SyntaxKind

    if sk.StartsWith("Const:") then
        emitConst zipper node
    elif sk.StartsWith("LetOrUse:") then
        emitLetOrUse zipper node
    elif sk.StartsWith("Sequential") then
        emitSequential zipper node
    elif sk.StartsWith("Binding:") then
        emitBinding zipper node
    elif sk.StartsWith("App") then
        emitApp zipper node
    elif sk.StartsWith("TypeApp") then
        emitTypeApp zipper node
    elif sk.StartsWith("LongIdent") || sk.StartsWith("Ident:") then
        emitIdent zipper node
    elif sk.StartsWith("AddressOf") then
        emitAddressOf zipper node
    elif sk.StartsWith("Pattern:") then
        zipper, Void
    else
        // Try traversing children
        let children = PSGZipper.childNodes zipper
        if children.IsEmpty then
            zipper, Void
        else
            emitSequential zipper node

// ═══════════════════════════════════════════════════════════════════════════
// Entry Point Emission
// ═══════════════════════════════════════════════════════════════════════════

/// Transcribe a PSG entry point to MLIR
let private transcribeEntryPoint (psg: ProgramSemanticGraph) (entryNode: PSGNode) : string =
    // Create shared context for the entire emission
    let ctx = ScribeContext.create psg
    currentContext <- Some ctx

    let output = StringBuilder()
    let emitTop (s: string) = output.AppendLine(s) |> ignore

    // Module header
    emitTop "module {"
    emitTop ""

    // Extern declarations
    emitTop "  // Extern declarations for Fidelity primitives"
    emitTop "  llvm.func @fidelity_write_bytes(i32, !llvm.ptr, i32) -> i32"
    emitTop "  llvm.func @fidelity_read_bytes(i32, !llvm.ptr, i32) -> i32"
    emitTop "  llvm.func @fidelity_exit(i32)"
    emitTop "  llvm.func @fidelity_sleep_ms(i32)"
    emitTop ""

    // Main function
    emitTop "  llvm.func @main() -> i32 {"

    // Find body of entry point
    let bodyNode = findBody psg entryNode
    eprintfn "[SCRIBE] Entry node: %s" entryNode.SyntaxKind
    eprintfn "[SCRIBE] Body node: %A" (bodyNode |> Option.map (fun n -> n.SyntaxKind))

    let zipper =
        match bodyNode with
        | Some body ->
            eprintfn "[SCRIBE] Starting emission from body: %s" body.SyntaxKind
            PSGZipper.create psg body
        | None ->
            eprintfn "[SCRIBE] No body found, emitting entry node"
            PSGZipper.create psg entryNode

    let zipper', result = emitNode zipper
    eprintfn "[SCRIBE] Emission result: %A" result

    // Append the body MLIR that was emitted during traversal
    output.Append(ctx.Output.ToString()) |> ignore

    // Exit handling
    match result with
    | Value (ssa, "i32") ->
        emitTop (sprintf "    llvm.call @fidelity_exit(%s) : (i32) -> ()" ssa)
    | _ ->
        emitTop "    %exit_code = arith.constant 0 : i32"
        emitTop "    llvm.call @fidelity_exit(%exit_code) : (i32) -> ()"

    emitTop "    %zero = arith.constant 0 : i32"
    emitTop "    llvm.return %zero : i32"
    emitTop "  }"

    // String literals
    for (content, name) in ctx.StringLiterals do
        let escaped = content.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\0A")
        let len = content.Length + 1
        emitTop (sprintf "  llvm.mlir.global private constant %s(\"%s\\00\") : !llvm.array<%d x i8>" name escaped len)

    emitTop "}"

    // Clean up context
    currentContext <- None

    output.ToString()

/// Find entry point node in PSG and transcribe
let transcribePSG (psg: ProgramSemanticGraph) : string =
    // Find the Binding:EntryPoint node
    let entryNode =
        psg.Nodes
        |> Map.tryPick (fun _ node ->
            if node.SyntaxKind = "Binding:EntryPoint" then Some node
            else None)

    match entryNode with
    | Some entry -> transcribeEntryPoint psg entry
    | None -> "// No entry point found in PSG\nmodule {}\n"
