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
                | Some sym -> sym.FullName = mainSymbol.FullName
                | None -> false)
        | None -> []

/// Find all reachable function definitions (excluding entry points which are handled separately)
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
        match node.Symbol with
        | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
            mfv.IsFunction || mfv.IsMember
        | _ -> false)
    |> List.filter (fun node ->
        // Skip inlined operations (all Alloy.* namespace functions)
        match node.Symbol with
        | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
            // Check if declaring entity is in Alloy namespace
            match mfv.DeclaringEntity with
            | Some entity -> not (entity.FullName.StartsWith("Alloy"))
            | None -> not (mfv.FullName.StartsWith("Alloy"))
        | Some sym ->
            not (sym.FullName.StartsWith("Alloy"))
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
let emitFunction (psg: ProgramSemanticGraph) (funcNode: PSGNode) (isEntryPoint: bool) : Emit<unit> =
    match funcNode.Symbol with
    | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
        let funcName = if isEntryPoint then "main" else mfv.DisplayName.Replace(".", "_")
        let returnType = if isEntryPoint then "i32" else getFunctionReturnType mfv
        // Filter out unit parameters - they don't exist at runtime
        let paramTypes =
            getFunctionParamTypes mfv
            |> List.filter (fun (_, typ) -> typ <> "()")

        // Build parameter list
        let paramStr =
            paramTypes
            |> List.mapi (fun i (name, typ) -> sprintf "%%arg%d: %s" i typ)
            |> String.concat ", "

        // Emit function header
        line "" >>.
        line (sprintf "func.func @%s(%s) -> %s {" funcName paramStr returnType) >>.
        pushIndent >>.

        // Register parameters in local scope
        forEach (fun ((name, typ), i) ->
            let ssaName = sprintf "%%arg%d" i
            bindLocal name ssaName typ
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
            | Value(ssa, typ) -> emitReturn ssa typ
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

    | Some (:? FSharpEntity as entity) when entity.IsAttributeType && isEntryPoint ->
        // Entry point with attribute as symbol - use default main signature
        line "" >>.
        line "func.func @main() -> i32 {" >>.
        pushIndent >>.

        // Navigate and emit body
        atNode funcNode emitFunctionBody >>= fun result ->

        // Emit exit syscall
        let exitCodeEmit =
            match result with
            | Value(ssa, "i32") -> emit ssa
            | _ -> emitI32 0

        exitCodeEmit >>= fun exitCode ->
        emitExtsi exitCode "i32" "i64" >>= fun exitCode64 ->
        emitI64 60L >>= fun syscallNum ->
        freshSSAWithType "i64" >>= fun res ->
        line (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"={rax},{rax},{rdi},~{rcx},~{r11},~{memory}\" %s, %s : (i64, i64) -> i64"
            res syscallNum exitCode64) >>.
        emitReturn exitCode "i32" >>.

        popIndent >>.
        line "}"

    | _ ->
        line (sprintf "// Skipping node without function symbol: %s" funcNode.SyntaxKind)

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
