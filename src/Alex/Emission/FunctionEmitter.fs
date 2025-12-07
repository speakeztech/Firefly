/// FunctionEmitter - Emit complete MLIR functions
///
/// Orchestrates expression emission within function contexts using
/// zipper-based traversal.
module Alex.Emission.FunctionEmitter

open System.Text
open FSharp.Compiler.Symbols
open Core.PSG.Types
open Alex.CodeGeneration.EmissionContext
open Alex.CodeGeneration.TypeMapping
open Alex.CodeGeneration.EmissionMonad
open Alex.Traversal.PSGZipper
open Alex.Emission.ExpressionEmitter

// ═══════════════════════════════════════════════════════════════════
// Safe Symbol Helpers
// ═══════════════════════════════════════════════════════════════════

/// Safely get a symbol's FullName, handling types like 'unit' that throw exceptions
let private tryGetFullName (sym: FSharpSymbol) : string option =
    try Some sym.FullName with _ -> None

/// Get a symbol's FullName or a fallback value
let private getFullNameOrDefault (sym: FSharpSymbol) (fallback: string) : string =
    match tryGetFullName sym with
    | Some name -> name
    | None -> fallback

// ═══════════════════════════════════════════════════════════════════
// Function Discovery
// ═══════════════════════════════════════════════════════════════════

/// Find entry point functions in the PSG
let findEntryPoints (psg: ProgramSemanticGraph) : PSGNode list =
    // First try explicit entry points
    let explicit =
        psg.EntryPoints
        |> List.choose (fun epId -> Map.tryFind epId.Value psg.Nodes)

    if not explicit.IsEmpty then explicit
    else
        // Look for "main" in symbol table
        match Map.tryFind "main" psg.SymbolTable with
        | Some mainSymbol ->
            psg.Nodes
            |> Map.toList
            |> List.map snd
            |> List.filter (fun node ->
                match node.Symbol with
                | Some sym ->
                    match tryGetFullName sym, tryGetFullName mainSymbol with
                    | Some n1, Some n2 -> n1 = n2
                    | _ -> false
                | None -> false)
        | None -> []

/// Find all reachable function definitions (excluding entry points which are handled separately)
/// Uses symbol's IsFunction/IsMember properties (stored at PSG construction, not a new FCS query)
let findReachableFunctions (psg: ProgramSemanticGraph) : PSGNode list =
    let entryPointIds = psg.EntryPoints |> List.map (fun ep -> ep.Value) |> Set.ofList

    psg.Nodes
    |> Map.toList
    |> List.map snd
    |> List.filter (fun node ->
        node.IsReachable &&
        not (Set.contains node.Id.Value entryPointIds) &&
        (node.SyntaxKind = "Binding" ||
         node.SyntaxKind.StartsWith("LetBinding") ||
         node.SyntaxKind.StartsWith("MemberDefn")))
    |> List.filter (fun node ->
        // Use the FCS symbol's IsFunction/IsMember to identify actual function definitions
        // This matches what the PSG debug output uses
        match node.Symbol with
        | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
            mfv.IsFunction || mfv.IsMember
        | _ -> false)
    |> List.filter (fun node ->
        // Skip Alloy functions that have special inline implementations
        // Uses symbol's FullName - just reading stored string, not querying FCS
        match node.Symbol with
        | Some sym ->
            match tryGetFullName sym with
            | Some fullName ->
                // List of Alloy functions that should be inlined at call sites
                let shouldSkip =
                    fullName = "Alloy.Core.string" ||
                    fullName.StartsWith("Alloy.Core.Ok") ||
                    fullName.StartsWith("Alloy.Core.Error") ||
                    fullName.StartsWith("Alloy.Console.Write") ||
                    fullName.StartsWith("Alloy.Console.Prompt") ||
                    fullName.StartsWith("Alloy.Console.ReadLine") ||
                    fullName.StartsWith("Alloy.Time.") ||
                    fullName.StartsWith("Alloy.Math.") ||
                    fullName.StartsWith("Alloy.Memory.stackBuffer") ||
                    fullName.StartsWith("Alloy.Text.UTF8.spanToString")
                not shouldSkip
            | None -> true  // Can't get FullName, include it
        | None -> true)

// ═══════════════════════════════════════════════════════════════════
// Function Body Emission (Zipper-based)
// ═══════════════════════════════════════════════════════════════════

/// Emit the body of a function by traversing its children with the zipper
/// Skips Pattern nodes (which are structural) and emits expression children
let emitFunctionBody : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z

    // Filter out Pattern nodes - they're structural, not expressions
    let exprChildren =
        children
        |> List.filter (fun n -> not (n.SyntaxKind.StartsWith("Pattern:")))

    match exprChildren with
    | [] -> emit Void
    | [single] -> atNode single emitCurrentExpr
    | _ ->
        // Emit all but last for side effects, return last
        let allButLast = exprChildren |> List.take (List.length exprChildren - 1)
        let lastNode = exprChildren |> List.last
        forEach (fun child ->
            atNode child emitCurrentExpr >>= fun _ -> emit ()
        ) allButLast >>.
        atNode lastNode emitCurrentExpr

// ═══════════════════════════════════════════════════════════════════
// Function Emission
// ═══════════════════════════════════════════════════════════════════

/// Emit a function definition using zipper-based traversal
/// Uses PSG node's Type field - NO FCS lookups after PSG construction
let emitFunction (psg: ProgramSemanticGraph) (funcNode: PSGNode) (isEntryPoint: bool) : Emit<unit> =
    // Extract function name from symbol
    // Use FullName to avoid collisions between modules with same-named functions
    // Sanitize for MLIR: remove F# backtick escaping, replace dots with underscores
    let funcName =
        if isEntryPoint then "main"
        else
            match funcNode.Symbol with
            | Some sym ->
                // Use FullName but take only the last two segments (Module.Function)
                // to keep names reasonably short while avoiding collisions
                let fullName = getFullNameOrDefault sym sym.DisplayName
                let segments = fullName.Split('.')
                let shortName =
                    if segments.Length >= 2 then
                        sprintf "%s_%s" segments.[segments.Length - 2] segments.[segments.Length - 1]
                    else
                        segments.[segments.Length - 1]
                shortName
                    .Replace(".", "_")
                    .Replace("``", "")  // Remove F# double-backtick escaping
            | None -> "unknown_func"

    // Get type info from PSG node's Type field - the canonical source
    let (returnType, paramTypes) =
        if isEntryPoint then
            ("i32", [])
        else
            match funcNode.Type with
            | Some ftype ->
                let ret = getReturnTypeFromFSharpType ftype
                let paramList =
                    getParamTypesFromFSharpType ftype
                    |> List.filter (fun typ -> typ <> "()")  // Filter unit params
                (ret, paramList)
            | None ->
                // Fallback if Type not populated - this shouldn't happen
                ("i32", [])

    match funcNode.Symbol with
    | Some sym ->

        // Extract parameter names from FCS symbol
        let paramNames =
            match sym with
            | :? FSharpMemberOrFunctionOrValue as mfv ->
                mfv.CurriedParameterGroups
                |> Seq.collect id
                |> Seq.map (fun p -> p.DisplayName)
                |> Seq.toList
            | _ -> []

        // Clear locals from previous function - each function has its own scope
        clearLocals >>.

        // Build parameter list (using actual names where available, falling back to argN)
        let paramStr =
            paramTypes
            |> List.mapi (fun i typ ->
                let name = if i < List.length paramNames then paramNames.[i] else sprintf "arg%d" i
                sprintf "%%arg%d: %s" i typ)  // MLIR uses argN but we track the source name
            |> String.concat ", "

        // Emit function header
        line "" >>.
        line (sprintf "func.func @%s(%s) -> %s {" funcName paramStr returnType) >>.
        pushIndent >>.

        // Register parameters in local scope with BOTH their source names AND argN names
        forEach (fun (typ, i) ->
            let ssaName = sprintf "%%arg%d" i
            // Bind with source name (e.g., "oldValue" -> %arg1)
            let sourceName = if i < List.length paramNames then paramNames.[i] else sprintf "arg%d" i
            bindLocal sourceName ssaName typ >>= fun _ ->
            // Also bind with argN name for backwards compatibility
            if i < List.length paramNames then
                bindLocal (sprintf "arg%d" i) ssaName typ
            else
                emit ()
        ) (paramTypes |> List.mapi (fun i p -> (p, i))) >>.

        // Navigate to function node and emit its body
        atNode funcNode emitFunctionBody >>= fun result ->

        // For entry points, emit exit syscall
        (if isEntryPoint then
            let exitCodeEmit =
                match result with
                | Value(ssa, typ) when typ = "i32" -> emit ssa
                | Value(ssa, _) ->
                    // Try to truncate to i32
                    emitTrunci ssa "i64" "i32"
                | _ ->
                    emitI32 0

            exitCodeEmit >>= fun exitCode ->
            // Extend to i64 for syscall
            emitExtsi exitCode "i32" "i64" >>= fun exitCode64 ->
            // Emit exit syscall (syscall 60 on Linux x86-64)
            emitI64 60L >>= fun syscallNum ->
            freshSSAWithType "i64" >>= fun result ->
            line (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"={rax},{rax},{rdi},~{rcx},~{r11},~{memory}\" %s, %s : (i64, i64) -> i64"
                result syscallNum exitCode64) >>.
            emitReturn exitCode "i32"
        else
            // Non-entry-point function - match return type
            match result with
            | Value(ssa, typ) ->
                // If function returns unit but body produced a value, ignore the value
                if returnType = "()" then
                    emitReturnVoid
                else
                    emitReturn ssa typ
            | Void ->
                // Function body returned Void - check declared return type
                if returnType = "()" then
                    emitReturnVoid
                else
                    // Need to return a default value of the declared type
                    emitDefaultValue returnType >>= fun defaultVal ->
                    emitReturn defaultVal returnType
            | Error msg ->
                line (sprintf "// ERROR: %s" msg) >>.
                // Still need valid return for MLIR
                if returnType = "()" then
                    emitReturnVoid
                else
                    emitDefaultValue returnType >>= fun defaultVal ->
                    emitReturn defaultVal returnType) >>.

        popIndent >>.
        line "}"

    | None ->
        line (sprintf "// Skipping node without symbol: %s" funcNode.SyntaxKind)

// ═══════════════════════════════════════════════════════════════════
// String Literal Emission
// ═══════════════════════════════════════════════════════════════════

/// Emit all registered string literals as global constants
let emitStringLiterals : Emit<unit> =
    getState >>= fun state ->
    forEach (fun ((content: string), (name: string)) ->
        let escaped = content.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\0A")
        // name is "%strN", globalName is "strN" for the @strN reference
        let globalName = name.Substring(1)
        // MLIR syntax: llvm.mlir.global constant @name("content") : !llvm.array<N x i8>
        line (sprintf "llvm.mlir.global private constant @%s(\"%s\") : !llvm.array<%d x i8>" globalName escaped content.Length)
    ) state.StringLiterals

/// Emit a module with its functions
let emitModule (psg: ProgramSemanticGraph) : Emit<unit> =
    // Emit module header
    line "module {" >>.
    pushIndent >>.

    // Find and emit entry points first
    let entryPoints = findEntryPoints psg
    forEach (fun ep -> emitFunction psg ep true) entryPoints >>.

    // Find and emit other reachable functions
    let otherFunctions = findReachableFunctions psg
    forEach (fun fn -> emitFunction psg fn false) otherFunctions >>.

    // Emit string literals inside the module
    emitStringLiterals >>.

    popIndent >>.
    line "}"

// ═══════════════════════════════════════════════════════════════════
// Complete Program Emission
// ═══════════════════════════════════════════════════════════════════

/// Emit a complete MLIR program from a PSG
let emitProgram (psg: ProgramSemanticGraph) : string =
    let builder = MLIRBuilder.create ()
    let initialState = EmissionState.empty

    // Create a dummy zipper for the initial context
    let dummyNode = {
        Id = NodeId.Create "root"
        SyntaxKind = "Module"
        Symbol = None
        Type = None
        Constraints = None
        Range = FSharp.Compiler.Text.Range.Zero
        SourceFile = ""
        ParentId = None
        Children = ChildrenState.NoChildren
        IsReachable = true
        EliminationPass = None
        EliminationReason = None
        ReachabilityDistance = Some 0
        ContextRequirement = None
        ComputationPattern = None
    }
    let zipper = PSGZipper.create psg dummyNode

    // Run the emission
    let program = emitModule psg

    let _ = run zipper builder initialState program

    MLIRBuilder.toString builder

/// Emit MLIR for a single function from the PSG
let emitSingleFunction (funcName: string) (psg: ProgramSemanticGraph) : string option =
    // Find the function node
    let funcNodeOpt =
        psg.Nodes
        |> Map.toList
        |> List.map snd
        |> List.tryFind (fun n ->
            match n.Symbol with
            | Some s -> s.DisplayName = funcName
            | None -> false)

    match funcNodeOpt with
    | Some funcNode ->
        let builder = MLIRBuilder.create ()
        let initialState = EmissionState.empty
        let zipper = PSGZipper.create psg funcNode

        let _ = run zipper builder initialState (emitFunction psg funcNode false)

        Some (MLIRBuilder.toString builder)
    | None ->
        None
