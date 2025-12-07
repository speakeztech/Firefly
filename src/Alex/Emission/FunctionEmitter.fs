/// FunctionEmitter - Emit complete MLIR functions
///
/// This module uses the EmissionMonad for function-level emission,
/// while delegating expression emission to ExpressionEmitter which
/// uses the MLIR computation expression (no sprintf).
module Alex.Emission.FunctionEmitter

open System.Text
open FSharp.Compiler.Symbols
open Core.PSG.Types
open Alex.CodeGeneration.EmissionContext
open Alex.CodeGeneration.TypeMapping
open Alex.CodeGeneration.EmissionMonad
open Alex.Traversal.PSGZipper

// Import the new MLIR builder types
module MB = Alex.CodeGeneration.MLIRBuilder
module EE = Alex.Emission.ExpressionEmitter

// ═══════════════════════════════════════════════════════════════════
// ExprResult Type - mirrors ExpressionEmitter.EmitResult
// ═══════════════════════════════════════════════════════════════════

/// Result of expression emission - for function-level handling
type FuncEmitResult =
    | Value of ssa: string * typ: string
    | Void
    | EmitError of string

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
// Bridge: MLIR CE to EmissionMonad
// ═══════════════════════════════════════════════════════════════════

/// Convert MLIR builder type to string representation
let private tyToString (ty: MB.Ty) : string =
    match ty with
    | MB.Int MB.I1 -> "i1"
    | MB.Int MB.I8 -> "i8"
    | MB.Int MB.I16 -> "i16"
    | MB.Int MB.I32 -> "i32"
    | MB.Int MB.I64 -> "i64"
    | MB.Float MB.F32 -> "f32"
    | MB.Float MB.F64 -> "f64"
    | MB.Ptr -> "!llvm.ptr"
    | MB.Struct _ -> "!llvm.struct<...>"
    | MB.Array _ -> "!llvm.array<...>"
    | MB.Func _ -> "func"
    | MB.Unit -> "()"
    | MB.Index -> "index"

/// Run an MLIR CE emission within the EmissionMonad context.
/// This bridges the new MLIR CE with the function-level EmissionMonad.
/// Note: String literals are now read directly from PSG.StringLiterals,
/// which is populated during PSG construction (not during emission).
let runMLIREmission (psg: ProgramSemanticGraph) (node: PSGNode) : Emit<FuncEmitResult> =
    fun env state ->
        // Create a fresh BuilderState for expression emission
        let builderState : MB.BuilderState = {
            Output = StringBuilder()
            SSACounter = state.SSACounter
            Indent = 0
            Globals = []
        }

        // Run the MLIR CE emission
        let mlirExpr = EE.emitExpr psg node
        let result = mlirExpr builderState

        // Transfer emitted MLIR lines to the function-level builder
        let emittedText = builderState.Output.ToString()
        for lineText in emittedText.Split('\n') do
            if not (System.String.IsNullOrWhiteSpace(lineText)) then
                MLIRBuilder.line env.Builder lineText

        // Update state with new SSA counter
        // Note: String literals are read from PSG, not tracked in emission state
        let newState = {
            state with
                SSACounter = builderState.SSACounter
        }

        // Convert EE.EmitResult to FuncEmitResult
        let funcResult =
            match result with
            | EE.Emitted v ->
                let ssaName = v.SSA.Name
                let typStr = tyToString v.Type
                Value(ssaName, typStr)
            | EE.Void -> Void
            | EE.Error msg -> EmitError msg

        (newState, funcResult)

// ═══════════════════════════════════════════════════════════════════
// Function Body Emission (Zipper-based)
// ═══════════════════════════════════════════════════════════════════

/// Emit the body of a function by traversing its children with the zipper
/// Skips Pattern nodes (which are structural) and emits expression children
let emitFunctionBody (psg: ProgramSemanticGraph) : Emit<FuncEmitResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z

    // Debug: Print what we're seeing
    printfn "[EMIT DEBUG] emitFunctionBody for node: %s" z.Focus.SyntaxKind
    printfn "[EMIT DEBUG]   Children count: %d" (List.length children)
    for child in children do
        printfn "[EMIT DEBUG]   - %s" child.SyntaxKind

    // Filter out Pattern nodes - they're structural, not expressions
    let exprChildren =
        children
        |> List.filter (fun n -> not (n.SyntaxKind.StartsWith("Pattern:")))

    match exprChildren with
    | [] -> emit Void
    | [single] -> runMLIREmission psg single
    | _ ->
        // Emit all but last for side effects, return last
        let allButLast = exprChildren |> List.take (List.length exprChildren - 1)
        let lastNode = exprChildren |> List.last
        forEach (fun child ->
            runMLIREmission psg child >>= fun _ -> emit ()
        ) allButLast >>.
        runMLIREmission psg lastNode

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

        // Clear state from previous function - each function has its own scope
        clearFunctionScope >>.

        // Build parameter list (using actual names where available, falling back to argN)
        let paramStr =
            paramTypes
            |> List.mapi (fun i typ ->
                let _name = if i < List.length paramNames then paramNames.[i] else sprintf "arg%d" i
                sprintf "%%arg%d: %s" i typ)  // MLIR uses argN but we track the source name
            |> String.concat ", "

        // Emit function header
        line "" >>.
        line (sprintf "func.func @%s(%s) -> %s {" funcName paramStr returnType) >>.
        pushIndent >>.

        // Register function parameters in NodeSSA
        // Use zipper navigation: funcNode -> Pattern:LongIdent -> Pattern:Named children
        getZipper >>= fun z ->
        let parameterNodes =
            // Get children of funcNode (the Binding)
            PSGZipper.childNodes z
            // Find Pattern:LongIdent (function name pattern)
            |> List.filter (fun n -> n.SyntaxKind.StartsWith("Pattern:LongIdent"))
            // Get Pattern:Named children of each (the parameters)
            |> List.collect (fun longIdentNode ->
                match longIdentNode.Children with
                | Parent childIds ->
                    childIds
                    |> List.choose (fun id -> Map.tryFind id.Value z.Graph.Nodes)
                    |> List.filter (fun n -> n.SyntaxKind.StartsWith("Pattern:Named"))
                | _ -> [])

        // Record SSA for each parameter node in order
        forEach (fun (node, idx) ->
            if idx < List.length paramTypes then
                let ssaName = sprintf "%%arg%d" idx
                let typ = paramTypes.[idx]
                recordNodeSSA node.Id ssaName typ
            else
                emit ()
        ) (parameterNodes |> List.mapi (fun i n -> (n, i))) >>.

        // Navigate to function node and emit its body
        atNode funcNode (emitFunctionBody psg) >>= fun result ->

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
            freshSSAWithType "i64" >>= fun resultSSA ->
            line (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"={rax},{rax},{rdi},~{rcx},~{r11},~{memory}\" %s, %s : (i64, i64) -> i64"
                resultSSA syscallNum exitCode64) >>.
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
            | EmitError msg ->
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

/// Escape a string for MLIR string literal syntax
let private escapeForMLIR (s: string) : string =
    s.Replace("\\", "\\\\")
     .Replace("\"", "\\\"")
     .Replace("\n", "\\0A")
     .Replace("\r", "\\0D")
     .Replace("\t", "\\09")

/// Emit all string literals from the PSG as global constants
/// The PSG.StringLiterals map (hash -> content) is populated during PSG construction
let emitStringLiterals (psg: ProgramSemanticGraph) : Emit<unit> =
    let literals = psg.StringLiterals |> Map.toList
    forEach (fun (hash: uint32, content: string) ->
        let escaped = escapeForMLIR content
        // MLIR syntax: llvm.mlir.global constant @str_HASH("content") : !llvm.array<N x i8>
        line (sprintf "llvm.mlir.global private constant @str_%u(\"%s\") : !llvm.array<%d x i8>" hash escaped content.Length)
    ) literals

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

    // Emit string literals from PSG (populated during construction)
    emitStringLiterals psg >>.

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
        Operation = None
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
