/// Transfer.fs - Codata-Driven PSG Observation to MLIR Extraction
///
/// ARCHITECTURAL FOUNDATION (December 2024):
/// This module implements the clean dual-zipper architecture for PSG→MLIR transfer:
/// - PSGZipper navigates the INPUT (PSG tree structure)
/// - MLIRZipper composes the OUTPUT (MLIR compute graph)
///
/// CODATA/COEFFECT VOCABULARY:
/// - **witness** - To observe and record a computation (the fold witnesses each node)
/// - **observe** - To note a context requirement (coeffect - what we need from environment)
/// - **yield** - To produce on demand (codata production of SSA names, labels)
/// - **bind** - To associate an observation with an identity (node → SSA mapping)
/// - **recall** - To retrieve a prior observation (lookup bound SSA)
/// - **extract** - Comonad operation: collapse accumulated context to final value (MLIR text)
///
/// KEY PRINCIPLES:
/// 1. Post-order traversal via foldPostOrder - Children witnessed BEFORE parents
/// 2. NO manual navigation - The fold handles traversal, child SSAs are recalled from map
/// 3. NO recursive witnessNode calls - foldPostOrder visits each node exactly once
/// 4. Dispatch on node.Kind, node.Operation, node.PlatformBinding - NOT symbol names
/// 5. Functional state threading - Via MLIRZipper accumulator
///
/// CRITICAL INSIGHT (from Serena memory):
/// "When the parent node is witnessed, its children's SSA values are already bound"
///
/// This module REPLACES MLIRTransfer.fs
module Alex.Generation.Transfer

open FSharp.Compiler.Symbols
open Core.PSG.Types
open Core.Types.MLIRTypes
open Alex.Traversal.PSGZipper
open Alex.Traversal.MLIRZipper
open Alex.CodeGeneration.MLIRBuilder
open Baker.Types
open Baker.TypedTreeZipper

// ═══════════════════════════════════════════════════════════════════
// Transfer Accumulator - State threaded through foldPostOrder
// ═══════════════════════════════════════════════════════════════════

/// Accumulator for the dual-zipper transfer
/// CRITICAL: This is threaded through foldPostOrder - child SSAs accumulate before parent visits
type TransferAcc = {
    /// The MLIR composition zipper (OUTPUT side)
    MLIR: MLIRZipper
    /// The PSG graph for edge lookups
    Graph: ProgramSemanticGraph
    /// Last transfer result (for sequential composition - the LAST child's result)
    LastResult: TransferResult
}

module TransferAcc =
    /// Create initial transfer accumulator
    let create (graph: ProgramSemanticGraph) : TransferAcc = {
        MLIR = MLIRZipper.create ()
        Graph = graph
        LastResult = TRVoid
    }

    /// Update the MLIR zipper
    let withMLIR (mlirZ: MLIRZipper) (acc: TransferAcc) : TransferAcc =
        { acc with MLIR = mlirZ }

    /// Update the last result
    let withResult (result: TransferResult) (acc: TransferAcc) : TransferAcc =
        { acc with LastResult = result }

    /// Update both MLIR zipper and result
    let withMLIRAndResult (mlirZ: MLIRZipper) (result: TransferResult) (acc: TransferAcc) : TransferAcc =
        { acc with MLIR = mlirZ; LastResult = result }

// ═══════════════════════════════════════════════════════════════════
// Type Mapping - F# types to MLIR types
// ═══════════════════════════════════════════════════════════════════

/// Map F# type to MLIR type string
let mapFSharpTypeToMLIR (fsharpType: FSharp.Compiler.Symbols.FSharpType) : string =
    try
        if fsharpType.HasTypeDefinition then
            match fsharpType.TypeDefinition.TryFullName with
            | Some "System.Int32" | Some "Microsoft.FSharp.Core.int" -> "i32"
            | Some "System.Int64" | Some "Microsoft.FSharp.Core.int64" -> "i64"
            | Some "System.Byte" | Some "Microsoft.FSharp.Core.byte" -> "i8"
            | Some "System.Boolean" | Some "Microsoft.FSharp.Core.bool" -> "i1"
            | Some "System.String" | Some "Microsoft.FSharp.Core.string" -> "!llvm.ptr"
            | Some "System.IntPtr" | Some "Microsoft.FSharp.Core.nativeint" -> "i64"
            | Some "System.UIntPtr" | Some "Microsoft.FSharp.Core.unativeint" -> "i64"
            | Some "Microsoft.FSharp.Core.Unit" | Some "Microsoft.FSharp.Core.unit" -> "()"
            | Some typeName when typeName.Contains("nativeptr") -> "!llvm.ptr"
            | Some typeName when typeName.Contains("NativeStr") -> "!llvm.struct<(!llvm.ptr, i64)>"
            | _ -> "!llvm.ptr"
        else
            "!llvm.ptr"
    with _ -> "!llvm.ptr"

// ═══════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════

/// Get children of a PSG node (for looking up child SSAs)
let getChildren (graph: ProgramSemanticGraph) (node: PSGNode) : PSGNode list =
    match node.Children with
    | Parent childIds ->
        childIds |> List.choose (fun id -> Map.tryFind id.Value graph.Nodes)
    | _ -> []

/// Recall SSA values for all children of a node (they're already witnessed by post-order)
let recallChildSSAs (graph: ProgramSemanticGraph) (node: PSGNode) (mlirZ: MLIRZipper) : (string * string) list =
    getChildren graph node
    |> List.choose (fun child -> MLIRZipper.recallNodeSSA child.Id.Value mlirZ)

/// Recall SSA values for children that are NOT function identifiers
let recallOperandSSAs (graph: ProgramSemanticGraph) (node: PSGNode) (mlirZ: MLIRZipper) : (string * string) list =
    getChildren graph node
    |> List.filter (fun c ->
        not (c.SyntaxKind.StartsWith("LongIdent")) &&
        not (c.SyntaxKind.StartsWith("Ident")))
    |> List.choose (fun child -> MLIRZipper.recallNodeSSA child.Id.Value mlirZ)

/// Extract constant value and MLIR type from a Const node
let extractConstant (node: PSGNode) : (string * string) option =
    match node.ConstantValue with
    | Some (StringValue s) -> Some (s, "string")
    | Some (Int32Value i) -> Some (string i, "i32")
    | Some (Int64Value i) -> Some (sprintf "%dL" i, "i64")
    | Some (BoolValue b) -> Some ((if b then "true" else "false"), "i1")
    | Some (ByteValue b) -> Some (string (int b), "i8")
    | Some (FloatValue f) -> Some (string f, "f64")
    | Some (CharValue c) -> Some (string (int c), "i8")
    | Some UnitValue -> Some ("()", "unit")
    | None ->
        // Fall back to parsing SyntaxKind for legacy nodes
        let kind = node.SyntaxKind
        if kind.StartsWith("Const:Int32 ") then
            Some (kind.Substring(12), "i32")
        elif kind.StartsWith("Const:Int64 ") then
            Some (kind.Substring(12), "i64")
        elif kind.StartsWith("Const:String ") then
            Some (kind.Substring(13), "string")
        elif kind.StartsWith("Const:Boolean ") then
            let v = kind.Substring(14).ToLower()
            Some (v, "i1")
        elif kind = "Const:Unit" || kind = "Const:()" then
            Some ("()", "unit")
        else
            None

// ═══════════════════════════════════════════════════════════════════
// Node Witness Functions - Each witnesses ONE node, children already bound
// ═══════════════════════════════════════════════════════════════════

/// Witness a constant node
/// Children: None (constants are leaves)
let witnessConst (acc: TransferAcc) (node: PSGNode) : TransferAcc =
    match extractConstant node with
    | Some (_, "unit") ->
        TransferAcc.withResult TRVoid acc

    | Some (value, "string") ->
        // Observe string literal requirement, witness addressof
        let globalName, mlirZ1 = MLIRZipper.observeStringLiteral value acc.MLIR
        let ssaName, mlirZ2 = MLIRZipper.witnessAddressOf globalName mlirZ1
        // Bind node SSA for later recall
        let mlirZ3 = MLIRZipper.bindNodeSSA node.Id.Value ssaName "!llvm.ptr" mlirZ2
        TransferAcc.withMLIRAndResult mlirZ3 (TRValue (ssaName, "!llvm.ptr")) acc

    | Some (value, mlirType) ->
        // Numeric/boolean constant
        let ssaName, mlirZ1 = MLIRZipper.yieldSSA acc.MLIR
        let text = sprintf "%s = arith.constant %s : %s" ssaName value mlirType
        let mlirZ2 = MLIRZipper.witnessVoidOp text mlirZ1
        let mlirZ3 = MLIRZipper.bindNodeSSA node.Id.Value ssaName mlirType mlirZ2
        TransferAcc.withMLIRAndResult mlirZ3 (TRValue (ssaName, mlirType)) acc

    | None ->
        TransferAcc.withResult (TRError (sprintf "Unknown constant: %s" node.SyntaxKind)) acc

/// Witness a binding/let node
/// Children: Pattern nodes + value expression (already witnessed by post-order)
/// We bind the observed SSA from the value child to this node
let witnessBinding (acc: TransferAcc) (node: PSGNode) : TransferAcc =
    // Recall the value child SSA (skip Pattern: nodes)
    let valueChildSSA =
        getChildren acc.Graph node
        |> List.filter (fun c ->
            not (c.SyntaxKind.StartsWith("Pattern:")) &&
            not (c.SyntaxKind.StartsWith("Named")))
        |> List.tryHead
        |> Option.bind (fun child -> MLIRZipper.recallNodeSSA child.Id.Value acc.MLIR)

    match valueChildSSA with
    | Some (ssa, mlirType) ->
        // Bind the binding's SSA (same as its value)
        let mlirZ = MLIRZipper.bindNodeSSA node.Id.Value ssa mlirType acc.MLIR
        TransferAcc.withMLIRAndResult mlirZ (TRValue (ssa, mlirType)) acc
    | None ->
        // No value - this is a unit binding
        TransferAcc.withResult TRVoid acc

/// Witness a sequential expression
/// Children: Sequence items (already witnessed by post-order)
/// Result: The LAST child's observed result
let witnessSequential (acc: TransferAcc) (node: PSGNode) : TransferAcc =
    // Recall the last child's SSA (that's the result of a sequence)
    let lastChildSSA =
        getChildren acc.Graph node
        |> List.rev
        |> List.tryHead
        |> Option.bind (fun child -> MLIRZipper.recallNodeSSA child.Id.Value acc.MLIR)

    match lastChildSSA with
    | Some (ssa, mlirType) ->
        let mlirZ = MLIRZipper.bindNodeSSA node.Id.Value ssa mlirType acc.MLIR
        TransferAcc.withMLIRAndResult mlirZ (TRValue (ssa, mlirType)) acc
    | None ->
        // Empty sequence or all unit children
        TransferAcc.withResult TRVoid acc

/// Witness an identifier/value reference
/// Children: None (identifiers are leaves that reference definitions)
let witnessIdent (acc: TransferAcc) (node: PSGNode) : TransferAcc =
    // Try def-use resolution via edges
    let defEdge =
        acc.Graph.Edges |> List.tryFind (fun e ->
            e.Source = node.Id && e.Kind = SymbolUse)

    match defEdge with
    | Some edge ->
        // Recall the definition's SSA value
        match MLIRZipper.recallNodeSSA edge.Target.Value acc.MLIR with
        | Some (ssa, mlirType) ->
            let mlirZ = MLIRZipper.bindNodeSSA node.Id.Value ssa mlirType acc.MLIR
            TransferAcc.withMLIRAndResult mlirZ (TRValue (ssa, mlirType)) acc
        | None ->
            TransferAcc.withResult (TRError (sprintf "Definition node %s has no SSA" edge.Target.Value)) acc
    | None ->
        // No def-use edge - try direct lookup (might be a parameter)
        TransferAcc.withResult (TRError (sprintf "Unresolved identifier: %s" node.SyntaxKind)) acc

/// Witness arithmetic operation
/// Children: Operands (already witnessed by post-order)
let witnessArithmetic (acc: TransferAcc) (node: PSGNode) (op: ArithmeticOp) : TransferAcc =
    let operandSSAs = recallOperandSSAs acc.Graph node acc.MLIR

    if operandSSAs.Length >= 2 then
        let (lhsSSA, lhsType) = operandSSAs.[0]
        let (rhsSSA, _) = operandSSAs.[1]
        let mlirOp = OperationHelpers.arithmeticOpToMLIR op
        let ssaName, mlirZ = MLIRZipper.witnessArith mlirOp lhsSSA rhsSSA (Integer I32) acc.MLIR
        let mlirZ' = MLIRZipper.bindNodeSSA node.Id.Value ssaName lhsType mlirZ
        TransferAcc.withMLIRAndResult mlirZ' (TRValue (ssaName, lhsType)) acc
    else
        TransferAcc.withResult (TRError "Arithmetic requires 2 operands") acc

/// Witness comparison operation
/// Children: Operands (already witnessed by post-order)
let witnessComparison (acc: TransferAcc) (node: PSGNode) (op: ComparisonOp) : TransferAcc =
    let operandSSAs = recallOperandSSAs acc.Graph node acc.MLIR

    if operandSSAs.Length >= 2 then
        let (lhsSSA, _) = operandSSAs.[0]
        let (rhsSSA, _) = operandSSAs.[1]
        let pred = OperationHelpers.comparisonOpToPredicate op
        let ssaName, mlirZ = MLIRZipper.witnessCmpi pred lhsSSA rhsSSA (Integer I32) acc.MLIR
        let mlirZ' = MLIRZipper.bindNodeSSA node.Id.Value ssaName "i1" mlirZ
        TransferAcc.withMLIRAndResult mlirZ' (TRValue (ssaName, "i1")) acc
    else
        TransferAcc.withResult (TRError "Comparison requires 2 operands") acc

/// Witness console Write/WriteLine operation
/// Children: Argument (already witnessed by post-order)
let witnessConsoleWrite (acc: TransferAcc) (node: PSGNode) (addNewline: bool) : TransferAcc =
    // Get argument children (already processed)
    let argChildren =
        getChildren acc.Graph node
        |> List.filter (fun c ->
            not (c.SyntaxKind.StartsWith("LongIdent")) &&
            not (c.SyntaxKind.StartsWith("Ident")))

    match argChildren with
    | [argNode] ->
        // Check if it's a string constant (special case: known content and length)
        if argNode.SyntaxKind.StartsWith("Const:String") then
            let stringContent =
                match argNode.ConstantValue with
                | Some (StringValue s) -> s
                | _ ->
                    if argNode.SyntaxKind.StartsWith("Const:String ") then
                        argNode.SyntaxKind.Substring(13)
                    else ""

            if stringContent <> "" then
                let strLen = stringContent.Length
                let globalName, mlirZ1 = MLIRZipper.observeStringLiteral stringContent acc.MLIR

                // Observe write syscall extern requirement (coeffect)
                let mlirZ2 = MLIRZipper.observeExternFunc "write" "(i32, !llvm.ptr, i64) -> i64" mlirZ1

                // Witness addressof for string pointer
                let ptrSSA, mlirZ3 = MLIRZipper.witnessAddressOf globalName mlirZ2

                // Witness write syscall via inline asm
                // Args: fd (i64), ptr (!llvm.ptr), len (i64)
                let sysNumSSA, mlirZ4 = MLIRZipper.witnessConstant 1L I64 mlirZ3
                let fdSSA, mlirZ5 = MLIRZipper.witnessConstant 1L I64 mlirZ4
                let lenSSA, mlirZ6 = MLIRZipper.witnessConstant (int64 strLen) I64 mlirZ5
                let _, mlirZ7 = MLIRZipper.witnessSyscall sysNumSSA [(fdSSA, "i64"); (ptrSSA, "!llvm.ptr"); (lenSSA, "i64")] (Integer I64) mlirZ6

                // Add newline for WriteLine
                let mlirZFinal =
                    if addNewline then
                        let nlGlobalName, mlirZ8 = MLIRZipper.observeStringLiteral "\n" mlirZ7
                        let nlPtrSSA, mlirZ9 = MLIRZipper.witnessAddressOf nlGlobalName mlirZ8
                        let nlSysNumSSA, mlirZ10 = MLIRZipper.witnessConstant 1L I64 mlirZ9
                        let nlFdSSA, mlirZ11 = MLIRZipper.witnessConstant 1L I64 mlirZ10
                        let nlLenSSA, mlirZ12 = MLIRZipper.witnessConstant 1L I64 mlirZ11
                        let _, mlirZ13 = MLIRZipper.witnessSyscall nlSysNumSSA [(nlFdSSA, "i64"); (nlPtrSSA, "!llvm.ptr"); (nlLenSSA, "i64")] (Integer I64) mlirZ12
                        mlirZ13
                    else
                        mlirZ7

                TransferAcc.withMLIRAndResult mlirZFinal TRVoid acc
            else
                TransferAcc.withResult TRVoid acc
        else
            // Dynamic string - recall the child's SSA (pointer to string)
            match MLIRZipper.recallNodeSSA argNode.Id.Value acc.MLIR with
            | Some (ptrSSA, _) ->
                // For dynamic strings, we need to know the length
                // This should come from StringInfo on the node (nanopass enrichment)
                // For now, record an error to indicate this needs PSG enrichment
                TransferAcc.withResult (TRError "Dynamic string write requires StringInfo enrichment") acc
            | None ->
                TransferAcc.withResult (TRError "Console.Write argument has no SSA") acc
    | [] ->
        TransferAcc.withResult (TRError "Console.Write requires one argument") acc
    | _ ->
        // Multiple arguments - not yet supported
        TransferAcc.withResult (TRError "Console.Write with multiple arguments not yet supported") acc

/// Witness console ReadLine operation
let witnessConsoleReadLine (acc: TransferAcc) (node: PSGNode) : TransferAcc =
    // Allocate stack buffer and read from stdin
    let countSSA, mlirZ1 = MLIRZipper.witnessConstant 256L I64 acc.MLIR
    let allocText = sprintf "%%buf = llvm.alloca %s x i8 : (i64) -> !llvm.ptr" countSSA
    let mlirZ2 = MLIRZipper.witnessVoidOp allocText mlirZ1
    // This is simplified - full implementation would witness read syscall
    TransferAcc.withMLIRAndResult mlirZ2 (TRValue ("%buf", "!llvm.ptr")) acc

/// Witness console operations
let witnessConsole (acc: TransferAcc) (node: PSGNode) (op: ConsoleOp) : TransferAcc =
    match op with
    | ConsoleWrite -> witnessConsoleWrite acc node false
    | ConsoleWriteln -> witnessConsoleWrite acc node true
    | ConsoleReadLine -> witnessConsoleReadLine acc node
    | _ ->
        TransferAcc.withResult (TRError (sprintf "Console operation %A not yet implemented" op)) acc

/// Witness an application node
/// Dispatch by Operation field (set by ClassifyOperations nanopass)
let witnessApp (acc: TransferAcc) (node: PSGNode) : TransferAcc =
    match node.Operation with
    | Some (OperationKind.Arithmetic op) -> witnessArithmetic acc node op
    | Some (OperationKind.Comparison op) -> witnessComparison acc node op
    | Some (OperationKind.Console op) -> witnessConsole acc node op
    | Some (OperationKind.Core Ignore) ->
        // ignore: argument was evaluated (by post-order), result discarded
        TransferAcc.withResult TRVoid acc
    | Some op ->
        TransferAcc.withResult (TRError (sprintf "Operation %A not yet implemented in Transfer.fs" op)) acc
    | None ->
        // Unclassified app - children already processed
        // Try to get the last operand's result as this node's result
        let lastChildSSA =
            recallOperandSSAs acc.Graph node acc.MLIR
            |> List.tryLast

        match lastChildSSA with
        | Some (ssa, mlirType) ->
            let mlirZ = MLIRZipper.bindNodeSSA node.Id.Value ssa mlirType acc.MLIR
            TransferAcc.withMLIRAndResult mlirZ (TRValue (ssa, mlirType)) acc
        | None ->
            TransferAcc.withResult TRVoid acc

// ═══════════════════════════════════════════════════════════════════
// Main Witness Function - Single dispatch point for foldPostOrder
// ═══════════════════════════════════════════════════════════════════

/// Main node witness function - Called by foldPostOrder for each node
/// CRITICAL: Children are ALREADY witnessed when this is called (post-order)
/// DO NOT manually navigate to children or recursively call witnessNode
let witnessNode (acc: TransferAcc) (zipper: PSGZipper) : TransferAcc =
    let node = zipper.Focus
    let kind = node.SyntaxKind

    // Skip unreachable nodes
    if not node.IsReachable then
        acc
    else

    // Dispatch by SyntaxKind (using typed Kind where available)
    match node.Kind with
    | SKExpr EConst -> witnessConst acc node
    | SKExpr EIdent | SKExpr ELongIdent -> witnessIdent acc node
    | SKExpr ESequential -> witnessSequential acc node
    | SKExpr EApp | SKExpr ETypeApp -> witnessApp acc node
    | SKBinding _ -> witnessBinding acc node
    | SKPattern _ -> acc  // Patterns are structural, no emission
    | _ ->
        // Fall back to string matching for migration period
        if kind.StartsWith("Const:") then
            witnessConst acc node
        elif kind.StartsWith("Ident:") || kind.StartsWith("LongIdent:") || kind.StartsWith("Value:") then
            witnessIdent acc node
        elif kind.StartsWith("Sequential") then
            witnessSequential acc node
        elif kind.StartsWith("App") then
            witnessApp acc node
        elif kind.StartsWith("Binding") || kind.StartsWith("Let") || kind.StartsWith("LetOrUse") then
            witnessBinding acc node
        elif kind.StartsWith("Pattern:") || kind.StartsWith("Named") then
            acc  // Skip patterns
        else
            // Unknown node - just pass through (children already processed)
            TransferAcc.withResult (TRError (sprintf "Unknown node kind: %s" kind)) acc

// ═══════════════════════════════════════════════════════════════════
// Entry Point - Codata Extraction from PSG Observation
// ═══════════════════════════════════════════════════════════════════

/// Observe a complete function and extract the witnessed MLIR
/// Uses foldPostOrder - the ONLY traversal call, handles all navigation
/// The fold WITNESSES each node; extract COLLAPSES the accumulator to MLIR text
let observeFunction (graph: ProgramSemanticGraph) (entryNode: PSGNode) : string =
    // Create PSG zipper at entry point
    let psgZipper = PSGZipper.create graph entryNode

    // Create transfer accumulator (observation context)
    let acc = TransferAcc.create graph

    // Use post-order fold to witness children before parents
    // THIS IS THE ONLY TRAVERSAL - no manual navigation anywhere else
    let finalAcc = PSGZipper.foldPostOrder witnessNode acc psgZipper

    // Extract: comonad operation - collapse accumulated observations to MLIR text
    MLIRZipper.extract finalAcc.MLIR

/// Observe and extract entry point using existing infrastructure
/// This is a bridge function that can be called from CompilationOrchestrator
let observe (graph: ProgramSemanticGraph) (entryNodeId: NodeId) : string option =
    match Map.tryFind entryNodeId.Value graph.Nodes with
    | Some entryNode ->
        Some (observeFunction graph entryNode)
    | None ->
        None

// ═══════════════════════════════════════════════════════════════════
// Generation Result - Compatible with MLIRTransfer API
// ═══════════════════════════════════════════════════════════════════

/// Result of MLIR generation (same shape as MLIRTransfer.GenerationResult)
type GenerationResult = {
    Content: string
    Errors: string list
    HasErrors: bool
}

// ═══════════════════════════════════════════════════════════════════
// Module-Level MLIR Generation - Entry Point for Compilation
// ═══════════════════════════════════════════════════════════════════

/// Get function name from entry point node
let private getFunctionName (node: PSGNode) : string =
    match node.Symbol with
    | Some sym ->
        let name =
            match sym with
            | :? FSharpMemberOrFunctionOrValue as mfv ->
                try mfv.CompiledName with _ -> mfv.DisplayName
            | _ -> sym.DisplayName
        if name = "main" || name.EndsWith(".main") then "main"
        else name.Replace(".", "_")
    | None -> "main"

/// Witness return/exit handling based on output kind
/// This is the final observation before function close
let private witnessReturn (acc: TransferAcc) (outputKind: OutputKind) : MLIRZipper =
    match outputKind with
    | Freestanding | Embedded ->
        // Freestanding: emit exit syscall
        let exitCode, mlirZ0 =
            match acc.LastResult with
            | TRValue (ssa, "i32") -> ssa, acc.MLIR
            | _ ->
                let zeroSSA, mlirZ = MLIRZipper.yieldSSA acc.MLIR
                let mlirZ' = MLIRZipper.witnessVoidOp (sprintf "%s = arith.constant 0 : i32" zeroSSA) mlirZ
                zeroSSA, mlirZ'

        // Extend to i64
        let exitCode64, mlirZ1 = MLIRZipper.yieldSSA mlirZ0
        let mlirZ2 = MLIRZipper.witnessVoidOp (sprintf "%s = arith.extsi %s : i32 to i64" exitCode64 exitCode) mlirZ1

        // Exit syscall (60 on Linux x86-64) - single arg: exit code (i64)
        let sysNumSSA, mlirZ3 = MLIRZipper.witnessConstant 60L I64 mlirZ2
        let _, mlirZ4 = MLIRZipper.witnessSyscall sysNumSSA [(exitCode64, "i64")] (Integer I64) mlirZ3
        MLIRZipper.witnessVoidOp "llvm.unreachable" mlirZ4

    | Console | Library ->
        // Console/Library: standard return
        match acc.LastResult with
        | TRValue (ssa, "i32") ->
            MLIRZipper.witnessVoidOp (sprintf "llvm.return %s : i32" ssa) acc.MLIR
        | _ ->
            let zeroSSA, mlirZ = MLIRZipper.yieldSSA acc.MLIR
            let mlirZ2 = MLIRZipper.witnessVoidOp (sprintf "%s = arith.constant 0 : i32" zeroSSA) mlirZ
            MLIRZipper.witnessVoidOp (sprintf "llvm.return %s : i32" zeroSSA) mlirZ2

/// Generate complete MLIR module from PSG
/// This is the API-compatible entry point that replaces MLIRTransfer.generateMLIR
let generateMLIR (psg: ProgramSemanticGraph) (_correlations: CorrelationState) (_targetTriple: string) (outputKind: OutputKind) : GenerationResult =
    let errors = ResizeArray<string>()

    // Create a shared zipper for the module
    let mutable mlirZ = MLIRZipper.create ()

    // Process each entry point
    for entryId in psg.EntryPoints do
        match Map.tryFind entryId.Value psg.Nodes with
        | Some entryNode ->
            let funcName = getFunctionName entryNode

            // Witness function header
            mlirZ <- MLIRZipper.witnessVoidOp (sprintf "llvm.func @%s() -> i32 {" funcName) mlirZ

            // Create zipper at entry point and witness all nodes
            let psgZipper = PSGZipper.create psg entryNode
            let acc = { TransferAcc.create psg with MLIR = mlirZ }
            let finalAcc = PSGZipper.foldPostOrder witnessNode acc psgZipper

            // Check for errors in final result
            match finalAcc.LastResult with
            | TRError err -> errors.Add(sprintf "[%s] %s" funcName err)
            | _ -> ()

            // Witness return/exit handling
            let mlirZWithReturn = witnessReturn finalAcc outputKind

            // Witness function footer
            mlirZ <- MLIRZipper.witnessVoidOp "}" mlirZWithReturn

        | None ->
            errors.Add(sprintf "Entry point not found: %s" entryId.Value)

    // Extract: collapse all witnessed observations to MLIR text
    {
        Content = MLIRZipper.extract mlirZ
        Errors = errors |> Seq.toList
        HasErrors = errors.Count > 0
    }
