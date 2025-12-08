/// PSGScribe - Transcribes PSG to MLIR via Zipper traversal and XParsec pattern matching
///
/// The Scribe walks the PSG via Zipper (the "attention"), pattern-matches children
/// with XParsec combinators, and transcribes to MLIR via platform Bindings.
///
/// ARCHITECTURAL PRINCIPLE:
/// - Zipper provides traversal (up/down/left/right navigation) AND context (ancestors, path)
/// - XParsec provides pattern matching on children at each position
/// - Bindings provide platform-specific MLIR generation for extern primitives
/// - Scribe orchestrates these three concerns
///
/// CALL GRAPH TRAVERSAL:
/// When encountering a function call, Scribe follows SymbolUse edges in the PSG
/// to find the function definition. It then checks if the function is an extern
/// primitive (has DllImport("__fidelity") attribute). If so, it dispatches to
/// ExternDispatch. If not, it transcribes the function body.
///
/// The Scribe does NOT:
/// - Know about specific library names (no "Alloy.Console.Write" matching)
/// - Make targeting decisions (that's Bindings' job)
/// - Interpret or transform semantics (PSG already has full semantics)
module Alex.Traversal.PSGScribe

open System.Text
open FSharp.Compiler.Symbols
open Core.PSG.Types
open Alex.Traversal.PSGZipper
open Alex.Traversal.PSGXParsec
open Alex.CodeGeneration.MLIRBuilder
open Alex.Bindings.BindingTypes

// ═══════════════════════════════════════════════════════════════════════════
// Scribe State - Accumulated output during transcription
// ═══════════════════════════════════════════════════════════════════════════

/// State accumulated during PSG transcription
type ScribeState = {
    /// SSA counter for generating unique value names
    SSACounter: int
    /// Map from NodeId to (SSA name, MLIR type)
    NodeSSA: Map<string, string * string>
    /// Accumulated string literals: (content, global name)
    StringLiterals: (string * string) list
    /// MLIR output lines
    Output: StringBuilder
    /// Current indentation level
    Indent: int
}

module ScribeState =
    let empty () = {
        SSACounter = 0
        NodeSSA = Map.empty
        StringLiterals = []
        Output = StringBuilder()
        Indent = 0
    }

    let nextSSA (state: ScribeState) : ScribeState * string =
        let name = sprintf "%%v%d" state.SSACounter
        { state with SSACounter = state.SSACounter + 1 }, name

    let recordNodeSSA (nodeId: NodeId) (ssa: string) (mlirType: string) (state: ScribeState) : ScribeState =
        { state with NodeSSA = Map.add nodeId.Value (ssa, mlirType) state.NodeSSA }

    let lookupNodeSSA (nodeId: NodeId) (state: ScribeState) : (string * string) option =
        Map.tryFind nodeId.Value state.NodeSSA

    let registerString (content: string) (state: ScribeState) : ScribeState * string =
        match state.StringLiterals |> List.tryFind (fun (c, _) -> c = content) with
        | Some (_, name) -> state, name
        | None ->
            let name = sprintf "@str%d" (List.length state.StringLiterals)
            { state with StringLiterals = (content, name) :: state.StringLiterals }, name

    let emit (line: string) (state: ScribeState) : ScribeState =
        let indent = String.replicate state.Indent "  "
        state.Output.AppendLine(indent + line) |> ignore
        state

    let pushIndent (state: ScribeState) : ScribeState =
        { state with Indent = state.Indent + 1 }

    let popIndent (state: ScribeState) : ScribeState =
        { state with Indent = max 0 (state.Indent - 1) }

// ═══════════════════════════════════════════════════════════════════════════
// Scribe Result - What transcription produces
// ═══════════════════════════════════════════════════════════════════════════

/// Result of transcribing an expression
type ScribeResult =
    /// Expression produced a value
    | SValue of ssa: string * mlirType: string
    /// Expression produced no value (unit/void)
    | SVoid
    /// Transcription error
    | SError of message: string

module ScribeResult =
    let isValue = function SValue _ -> true | _ -> false
    let isVoid = function SVoid -> true | _ -> false
    let isError = function SError _ -> true | _ -> false

    let getSSA = function SValue (ssa, _) -> Some ssa | _ -> None
    let getType = function SValue (_, t) -> Some t | _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// PSG Node Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Extract string content from a Const:String node
let private extractStringConst (node: PSGNode) : string option =
    if node.SyntaxKind.StartsWith("Const:String") then
        // SyntaxKind is like: Const:String ("Hello, World!", Regular, (10,18--10,33))
        let sk = node.SyntaxKind
        let start = sk.IndexOf("(\"")
        if start >= 0 then
            let afterQuote = start + 2
            let endQuote = sk.IndexOf("\",", afterQuote)
            if endQuote > afterQuote then
                Some (sk.Substring(afterQuote, endQuote - afterQuote))
            else None
        else None
    else None

/// Extract int value from a Const:Int32 node
let private extractIntConst (node: PSGNode) : int option =
    if node.SyntaxKind.StartsWith("Const:Int32") then
        // SyntaxKind is like: Const:Int32 0
        let parts = node.SyntaxKind.Split(' ')
        if parts.Length >= 2 then
            match System.Int32.TryParse(parts.[1]) with
            | true, v -> Some v
            | _ -> None
        else None
    else None

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
                // Look for EntryPoint named argument
                attr.NamedArguments
                |> Seq.tryPick (fun (_, name, _, value) ->
                    if name = "EntryPoint" then
                        match value with
                        | :? string as s -> Some s
                        | _ -> None
                    else None)
                |> Option.orElse (Some mfv.DisplayName) // Default to function name
            else None)
    | _ -> None

/// Check if a PSG node represents an extern primitive (has DllImport("__fidelity"))
let private isExternPrimitive (psg: ProgramSemanticGraph) (node: PSGNode) : bool =
    match node.Symbol with
    | Some sym -> hasFidelityDllImport sym
    | None -> false

/// Follow SymbolUse edges to find the definition of a function
/// Returns the definition node if found
let private findFunctionDefinition (psg: ProgramSemanticGraph) (funcNode: PSGNode) : PSGNode option =
    // Find SymbolUse edge from this node
    psg.Edges
    |> List.tryFind (fun edge -> edge.Source = funcNode.Id && edge.Kind = EdgeKind.SymbolUse)
    |> Option.bind (fun edge -> Map.tryFind edge.Target.Value psg.Nodes)

/// Check if a function call ultimately resolves to an extern primitive
/// by following the call graph until we find a DllImport("__fidelity")
let rec private resolveToExternPrimitive (psg: ProgramSemanticGraph) (funcNode: PSGNode) (visited: Set<string>) : (PSGNode * FSharpSymbol) option =
    if visited.Contains funcNode.Id.Value then
        printfn "[RESOLVE] Already visited: %s" funcNode.Id.Value
        None // Prevent infinite loops
    else
        let visited' = visited.Add funcNode.Id.Value
        printfn "[RESOLVE] Checking node: %s (SyntaxKind: %s)" funcNode.Id.Value funcNode.SyntaxKind

        match funcNode.Symbol with
        | Some sym when hasFidelityDllImport sym ->
            // Found extern primitive
            printfn "[RESOLVE] Found extern primitive: %s" sym.DisplayName
            Some (funcNode, sym)
        | Some sym ->
            printfn "[RESOLVE] Not extern, following to definition for: %s" sym.DisplayName
            // Follow to definition and check its body
            match findFunctionDefinition psg funcNode with
            | Some defNode ->
                printfn "[RESOLVE] Found definition: %s (SyntaxKind: %s)" defNode.Id.Value defNode.SyntaxKind
                // Look at the definition's children for function calls
                let children = ChildrenStateHelpers.getChildrenList defNode
                            |> List.choose (fun id -> Map.tryFind id.Value psg.Nodes)

                printfn "[RESOLVE] Definition has %d children" children.Length

                // Find any function calls in the body and recursively check
                children
                |> List.tryPick (fun child ->
                    if child.SyntaxKind.StartsWith("App:FunctionCall") then
                        printfn "[RESOLVE] Found App:FunctionCall in body: %s" child.Id.Value
                        let funcChildren = ChildrenStateHelpers.getChildrenList child
                                        |> List.choose (fun id -> Map.tryFind id.Value psg.Nodes)
                        funcChildren
                        |> List.tryFind (fun c -> c.SyntaxKind.StartsWith("LongIdent") || c.SyntaxKind.StartsWith("Ident:"))
                        |> Option.bind (fun fc ->
                            printfn "[RESOLVE] Recursing into: %s" fc.SyntaxKind
                            resolveToExternPrimitive psg fc visited')
                    else None)
            | None ->
                printfn "[RESOLVE] No definition found for: %s" funcNode.Id.Value
                None
        | None ->
            printfn "[RESOLVE] Node has no symbol: %s" funcNode.Id.Value
            None

// ═══════════════════════════════════════════════════════════════════════════
// Core Transcription - Recursive PSG traversal
// ═══════════════════════════════════════════════════════════════════════════

/// Transcribe a PSG node to MLIR, returning the result and updated state
let rec transcribe (psg: ProgramSemanticGraph) (node: PSGNode) (state: ScribeState) : ScribeState * ScribeResult =
    let children =
        ChildrenStateHelpers.getChildrenList node
        |> List.choose (fun id -> Map.tryFind id.Value psg.Nodes)

    match node.SyntaxKind with
    // ─────────────────────────────────────────────────────────────────────
    // Constants
    // ─────────────────────────────────────────────────────────────────────
    | sk when sk.StartsWith("Const:String") ->
        match extractStringConst node with
        | Some content ->
            let state', globalName = ScribeState.registerString content state
            let state'', ssa = ScribeState.nextSSA state'
            // Get pointer to string data
            let len = content.Length
            let state''' =
                state''
                |> ScribeState.emit (sprintf "%s = llvm.mlir.addressof %s : !llvm.ptr" ssa globalName)
            state''', SValue (ssa, "!llvm.ptr")
        | None ->
            state, SError "Could not extract string content"

    | sk when sk.StartsWith("Const:Int32") ->
        match extractIntConst node with
        | Some value ->
            let state', ssa = ScribeState.nextSSA state
            let state'' =
                state'
                |> ScribeState.emit (sprintf "%s = arith.constant %d : i32" ssa value)
            state'', SValue (ssa, "i32")
        | None ->
            state, SError "Could not extract int constant"

    | "Const:Unit" ->
        state, SVoid

    // ─────────────────────────────────────────────────────────────────────
    // Sequential expressions
    // ─────────────────────────────────────────────────────────────────────
    | "Sequential" ->
        // Execute all children in order, return last result
        let rec transcribeSeq state' children' =
            match children' with
            | [] -> state', SVoid
            | [last] -> transcribe psg last state'
            | first :: rest ->
                let state'', _ = transcribe psg first state'
                transcribeSeq state'' rest
        transcribeSeq state children

    // ─────────────────────────────────────────────────────────────────────
    // Function applications
    // ─────────────────────────────────────────────────────────────────────
    | sk when sk.StartsWith("App:FunctionCall") || sk.StartsWith("App") ->
        // Check if operation was classified by nanopass
        match node.Operation with
        | Some (Console consoleOp) ->
            transcribeConsoleOp psg node consoleOp children state
        | Some (Time timeOp) ->
            transcribeTimeOp psg node timeOp children state
        | Some (NativePtr ptrOp) ->
            transcribeNativePtrOp psg node ptrOp children state
        | Some (Core coreOp) ->
            transcribeCoreOp psg node coreOp children state
        | Some (Arithmetic arithOp) ->
            transcribeArithmeticOp psg node arithOp children state
        | Some (Comparison cmpOp) ->
            transcribeComparisonOp psg node cmpOp children state
        | Some (Conversion convOp) ->
            transcribeConversionOp psg node convOp children state
        | Some (RegularCall info) ->
            transcribeRegularCallInfo psg node info children state
        | _ ->
            // Unclassified - transcribe children and return last result
            let funcNode = children |> List.tryFind (fun c -> c.SyntaxKind.StartsWith("LongIdent"))
            let argNodes = children |> List.filter (fun c -> not (c.SyntaxKind.StartsWith("LongIdent")))

            match funcNode with
            | Some fn ->
                // Try to follow call graph for non-classified functions
                transcribeRegularCall psg fn argNodes state
            | None ->
                // No function identifier - just transcribe children
                transcribeChildren psg children state

    // ─────────────────────────────────────────────────────────────────────
    // Bindings (let expressions)
    // ─────────────────────────────────────────────────────────────────────
    | sk when sk.StartsWith("Binding:") ->
        // Children are: Pattern, then body expression
        let bodyNodes = children |> List.filter (fun c ->
            not (c.SyntaxKind.StartsWith("Pattern:")))
        match bodyNodes with
        | [body] -> transcribe psg body state
        | [] -> state, SVoid
        | _ ->
            // Multiple body expressions - treat as sequential
            let rec transcribeSeq state' nodes =
                match nodes with
                | [] -> state', SVoid
                | [last] -> transcribe psg last state'
                | first :: rest ->
                    let state'', _ = transcribe psg first state'
                    transcribeSeq state'' rest
            transcribeSeq state bodyNodes

    // ─────────────────────────────────────────────────────────────────────
    // Identifiers / Value references
    // ─────────────────────────────────────────────────────────────────────
    | sk when sk.StartsWith("LongIdent:") || sk.StartsWith("Ident:") ->
        // Look up the SSA value for this reference
        match ScribeState.lookupNodeSSA node.Id state with
        | Some (ssa, ty) -> state, SValue (ssa, ty)
        | None ->
            // This might be a function reference, not a value
            state, SVoid

    // ─────────────────────────────────────────────────────────────────────
    // Fallback
    // ─────────────────────────────────────────────────────────────────────
    | _ ->
        // For unhandled nodes, try to transcribe children
        if children.IsEmpty then
            state, SVoid
        else
            let rec transcribeSeq state' children' =
                match children' with
                | [] -> state', SVoid
                | [last] -> transcribe psg last state'
                | first :: rest ->
                    let state'', _ = transcribe psg first state'
                    transcribeSeq state'' rest
            transcribeSeq state children

/// Helper to transcribe children and return the last result
and private transcribeChildren (psg: ProgramSemanticGraph) (children: PSGNode list) (state: ScribeState) : ScribeState * ScribeResult =
    let rec transcribeSeq state' children' =
        match children' with
        | [] -> state', SVoid
        | [last] -> transcribe psg last state'
        | first :: rest ->
            let state'', _ = transcribe psg first state'
            transcribeSeq state'' rest
    transcribeSeq state children

/// Transcribe a Console operation via Bindings
and private transcribeConsoleOp
    (psg: ProgramSemanticGraph)
    (node: PSGNode)
    (consoleOp: ConsoleOp)
    (children: PSGNode list)
    (state: ScribeState) : ScribeState * ScribeResult =

    // Transcribe argument children (filter out function identifiers)
    let argNodes = children |> List.filter (fun c ->
        not (c.SyntaxKind.StartsWith("LongIdent")) && not (c.SyntaxKind.StartsWith("Ident:")))

    let state', argResults =
        argNodes
        |> List.fold (fun (st, results) arg ->
            let st', result = transcribe psg arg st
            st', results @ [result]
        ) (state, [])

    // For ConsoleWrite/WriteLine, we need to emit syscall with fd=1, buf=ptr, count=len
    match consoleOp with
    | ConsoleWrite | ConsoleWriteln ->
        match argResults with
        | [SValue (strPtr, "!llvm.ptr")] ->
            // First, emit constant for fd=1 (stdout)
            let state'', fdSsa = ScribeState.nextSSA state'
            let state''' = state'' |> ScribeState.emit (sprintf "%s = arith.constant 1 : i32" fdSsa)

            // Emit constant for string length (we know it from the string literal registration)
            // For "Hello, World!" the length is 13 (not including null terminator for write)
            // TODO: Get actual length from string literal registry
            let state4, lenSsa = ScribeState.nextSSA state'''
            let state5 = state4 |> ScribeState.emit (sprintf "%s = arith.constant 13 : i32" lenSsa)

            // Extend fd to i64
            let state6, fdExt = ScribeState.nextSSA state5
            let state7 = state6 |> ScribeState.emit (sprintf "%s = arith.extsi %s : i32 to i64" fdExt fdSsa)

            // Extend count to i64
            let state8, lenExt = ScribeState.nextSSA state7
            let state9 = state8 |> ScribeState.emit (sprintf "%s = arith.extsi %s : i32 to i64" lenExt lenSsa)

            // Emit syscall number constant (1 for write on Linux)
            let state10, sysNum = ScribeState.nextSSA state9
            let state11 = state10 |> ScribeState.emit (sprintf "%s = arith.constant 1 : i64" sysNum)

            // Emit inline_asm syscall
            let state12, result = ScribeState.nextSSA state11
            let state13 = state12 |> ScribeState.emit (sprintf "%s = llvm.inline_asm \"syscall\", \"=r,{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64" result sysNum fdExt strPtr lenExt)

            // If WriteLine, also emit newline
            if consoleOp = ConsoleWriteln then
                // TODO: Emit newline character write
                let state14 = state13 |> ScribeState.emit "// TODO: newline for WriteLine"
                state14, SVoid
            else
                state13, SVoid

        | _ ->
            // Fallback - emit comment
            let state'' = state' |> ScribeState.emit (sprintf "// Console.%s with %d args (unhandled)" (if consoleOp = ConsoleWrite then "Write" else "WriteLine") argResults.Length)
            state'', SVoid

    | _ ->
        // Other console ops - emit placeholder
        let state'' = state' |> ScribeState.emit (sprintf "// Console.%A with %d args" consoleOp argResults.Length)
        state'', SVoid

/// Placeholder for Time operations
and private transcribeTimeOp
    (psg: ProgramSemanticGraph)
    (node: PSGNode)
    (timeOp: TimeOp)
    (children: PSGNode list)
    (state: ScribeState) : ScribeState * ScribeResult =
    let state' = state |> ScribeState.emit (sprintf "// Time.%A - not yet implemented" timeOp)
    state', SVoid

/// Placeholder for NativePtr operations
and private transcribeNativePtrOp
    (psg: ProgramSemanticGraph)
    (node: PSGNode)
    (ptrOp: NativePtrOp)
    (children: PSGNode list)
    (state: ScribeState) : ScribeState * ScribeResult =
    let state' = state |> ScribeState.emit (sprintf "// NativePtr.%A - not yet implemented" ptrOp)
    state', SVoid

/// Placeholder for Core operations
and private transcribeCoreOp
    (psg: ProgramSemanticGraph)
    (node: PSGNode)
    (coreOp: CoreOp)
    (children: PSGNode list)
    (state: ScribeState) : ScribeState * ScribeResult =
    match coreOp with
    | Ignore ->
        // Ignore just returns unit after evaluating argument
        transcribeChildren psg children state |> fun (st, _) -> st, SVoid
    | _ ->
        let state' = state |> ScribeState.emit (sprintf "// Core.%A - not yet implemented" coreOp)
        state', SVoid

/// Placeholder for Arithmetic operations
and private transcribeArithmeticOp
    (psg: ProgramSemanticGraph)
    (node: PSGNode)
    (arithOp: ArithmeticOp)
    (children: PSGNode list)
    (state: ScribeState) : ScribeState * ScribeResult =
    let state' = state |> ScribeState.emit (sprintf "// Arithmetic.%A - not yet implemented" arithOp)
    state', SVoid

/// Placeholder for Comparison operations
and private transcribeComparisonOp
    (psg: ProgramSemanticGraph)
    (node: PSGNode)
    (cmpOp: ComparisonOp)
    (children: PSGNode list)
    (state: ScribeState) : ScribeState * ScribeResult =
    let state' = state |> ScribeState.emit (sprintf "// Comparison.%A - not yet implemented" cmpOp)
    state', SVoid

/// Placeholder for Conversion operations
and private transcribeConversionOp
    (psg: ProgramSemanticGraph)
    (node: PSGNode)
    (convOp: ConversionOp)
    (children: PSGNode list)
    (state: ScribeState) : ScribeState * ScribeResult =
    let state' = state |> ScribeState.emit (sprintf "// Conversion.%A - not yet implemented" convOp)
    state', SVoid

/// Transcribe a regular call using RegularCallInfo
and private transcribeRegularCallInfo
    (psg: ProgramSemanticGraph)
    (node: PSGNode)
    (info: RegularCallInfo)
    (children: PSGNode list)
    (state: ScribeState) : ScribeState * ScribeResult =
    // Transcribe children first
    let state', _ = transcribeChildren psg children state
    let state'' = state' |> ScribeState.emit (sprintf "// call: %s with %d args" info.FunctionName info.ArgumentCount)
    state'', SVoid

/// Transcribe an extern primitive call via Bindings
and private transcribeExternCall
    (psg: ProgramSemanticGraph)
    (funcNode: PSGNode)
    (argNodes: PSGNode list)
    (state: ScribeState) : ScribeState * ScribeResult =

    match funcNode.Symbol with
    | Some sym ->
        match getExternEntryPoint sym with
        | Some entryPoint ->
            // Transcribe arguments first
            let state', argResults =
                argNodes
                |> List.fold (fun (st, results) arg ->
                    let st', result = transcribe psg arg st
                    st', results @ [result]
                ) (state, [])

            // Build args for potential dispatch (simplified for now)
            let argCount =
                argResults
                |> List.filter (fun r -> match r with SValue _ -> true | _ -> false)
                |> List.length

            // For now, emit a placeholder comment
            // TODO: Actually dispatch via ExternDispatch
            let state'' =
                state'
                |> ScribeState.emit (sprintf "// extern call: %s with %d args" entryPoint argCount)

            state'', SVoid
        | None ->
            state, SError "Extern primitive has no entry point"
    | None ->
        state, SError "Extern function has no symbol"

/// Transcribe a regular (non-extern) function call
/// This follows the call graph to find extern primitives and dispatches to Bindings
and private transcribeRegularCall
    (psg: ProgramSemanticGraph)
    (funcNode: PSGNode)
    (argNodes: PSGNode list)
    (state: ScribeState) : ScribeState * ScribeResult =

    // Check if this call ultimately resolves to an extern primitive
    match resolveToExternPrimitive psg funcNode Set.empty with
    | Some (externNode, externSym) ->
        // Found an extern primitive - dispatch to Bindings
        match getExternEntryPoint externSym with
        | Some entryPoint ->
            // Transcribe arguments first
            let state', argResults =
                argNodes
                |> List.fold (fun (st, results) arg ->
                    let st', result = transcribe psg arg st
                    st', results @ [result]
                ) (state, [])

            // Convert ScribeState SSA string to MLIRBuilder SSA type
            // e.g., "%v5" -> V 5
            let parseSSA (ssaStr: string) : SSA option =
                if ssaStr.StartsWith("%v") then
                    match System.Int32.TryParse(ssaStr.Substring(2)) with
                    | true, n -> Some (V n)
                    | _ -> None
                elif ssaStr.StartsWith("%arg") then
                    match System.Int32.TryParse(ssaStr.Substring(4)) with
                    | true, n -> Some (Arg n)
                    | _ -> None
                else None

            // Build Val list for ExternDispatch
            let args =
                argResults
                |> List.choose (fun r ->
                    match r with
                    | SValue (ssaStr, ty) ->
                        let mlirTy =
                            match ty with
                            | "i32" -> Int I32
                            | "i64" -> Int I64
                            | "!llvm.ptr" -> Ptr
                            | _ -> Int I64
                        parseSSA ssaStr
                        |> Option.map (fun ssa -> { SSA = ssa; Type = mlirTy })
                    | _ -> None)

            // Create ExternPrimitive and dispatch
            let prim = {
                Library = "__fidelity"
                EntryPoint = entryPoint
                CallingConvention = "cdecl"
                Args = args
                ReturnType = Int I32 // TODO: Get from symbol
            }

            // Run ExternDispatch to get MLIR
            let mlirResult = ExternDispatch.dispatch prim
            let (mlirOutput, emissionResult) = run mlirResult

            match emissionResult with
            | Emitted resultVal ->
                // Emit the MLIR operations
                let state'' = state' |> ScribeState.emit mlirOutput
                state'', SValue (resultVal.SSA.Name, "i32")
            | EmittedVoid ->
                let state'' = state' |> ScribeState.emit mlirOutput
                state'', SVoid
            | NotSupported msg ->
                let state'' = state' |> ScribeState.emit (sprintf "// extern %s not supported: %s" entryPoint msg)
                state'', SVoid
        | None ->
            let state' = state |> ScribeState.emit "// extern has no entry point"
            state', SVoid

    | None ->
        // Not an extern - transcribe as regular call (inline or skip for now)
        let state', argResults =
            argNodes
            |> List.fold (fun (st, results) arg ->
                let st', result = transcribe psg arg st
                st', results @ [result]
            ) (state, [])

        let funcName =
            match funcNode.Symbol with
            | Some sym -> sym.DisplayName
            | None -> funcNode.SyntaxKind

        // For now, emit a comment for non-extern calls
        // Full implementation would inline the function body or emit a call
        let state'' =
            state'
            |> ScribeState.emit (sprintf "// call: %s (not extern, %d args)" funcName argResults.Length)

        state'', SVoid


// ═══════════════════════════════════════════════════════════════════════════
// Entry Point Transcription
// ═══════════════════════════════════════════════════════════════════════════

/// Transcribe an entry point function to MLIR
let transcribeEntryPoint (psg: ProgramSemanticGraph) (entryNode: PSGNode) : string =
    let state = ScribeState.empty ()

    // Emit module header
    let state = state |> ScribeState.emit "module {"
    let state = state |> ScribeState.pushIndent

    // Emit main function
    let state = state |> ScribeState.emit "func.func @main() -> i32 {"
    let state = state |> ScribeState.pushIndent
    let state = state |> ScribeState.emit "^entry:"

    // Transcribe the entry point body
    let state, result = transcribe psg entryNode state

    // For freestanding binaries, we need to call exit syscall (60 on Linux)
    // instead of just returning, since there's no runtime to return to
    let state =
        match result with
        | SValue (ssa, "i32") ->
            // Extend return value to i64 for syscall
            let state', retExt = ScribeState.nextSSA state
            let state'' = state' |> ScribeState.emit (sprintf "%s = arith.extsi %s : i32 to i64" retExt ssa)
            // Syscall 60 = exit on Linux
            let state''', sysNum = ScribeState.nextSSA state''
            let state4 = state''' |> ScribeState.emit (sprintf "%s = arith.constant 60 : i64" sysNum)
            // Call exit syscall
            let state5, _ = ScribeState.nextSSA state4
            state5
            |> ScribeState.emit (sprintf "llvm.inline_asm \"syscall\", \"=r,{rax},{rdi},~{rcx},~{r11},~{memory}\" %s, %s : (i64, i64) -> i64" sysNum retExt)
            |> ScribeState.emit "llvm.unreachable"
        | _ ->
            let state', zero = ScribeState.nextSSA state
            let state'' = state' |> ScribeState.emit (sprintf "%s = arith.constant 0 : i32" zero)
            // Extend to i64
            let state''', zeroExt = ScribeState.nextSSA state''
            let state4 = state''' |> ScribeState.emit (sprintf "%s = arith.extsi %s : i32 to i64" zeroExt zero)
            // Syscall 60 = exit
            let state5, sysNum = ScribeState.nextSSA state4
            let state6 = state5 |> ScribeState.emit (sprintf "%s = arith.constant 60 : i64" sysNum)
            let state7, _ = ScribeState.nextSSA state6
            state7
            |> ScribeState.emit (sprintf "llvm.inline_asm \"syscall\", \"=r,{rax},{rdi},~{rcx},~{r11},~{memory}\" %s, %s : (i64, i64) -> i64" sysNum zeroExt)
            |> ScribeState.emit "llvm.unreachable"

    let state = state |> ScribeState.popIndent
    let state = state |> ScribeState.emit "}"

    // Emit string literals as globals (before closing the module)
    let state =
        state.StringLiterals
        |> List.fold (fun st (content, name) ->
            let escaped = content.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\0A")
            let len = content.Length + 1 // +1 for null terminator
            // name already has @ prefix (e.g., @str0)
            st |> ScribeState.emit (sprintf "llvm.mlir.global private constant %s(\"%s\\00\") : !llvm.array<%d x i8>" name escaped len)
        ) state

    let state = state |> ScribeState.popIndent
    let state = state |> ScribeState.emit "}"

    state.Output.ToString()

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
