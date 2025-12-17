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
/// This is the ONLY MLIR generation module (MLIRTransfer.fs has been removed)
module Alex.Generation.Transfer

open FSharp.Compiler.Symbols
open Core.PSG.Types
open Core.PSG.NavigationUtils
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
            let typeDef = fsharpType.TypeDefinition
            let displayName = typeDef.DisplayName

            // First check DisplayName for common type abbreviations
            match displayName with
            | "unit" -> "()"
            | "int" | "int32" -> "i32"
            | "int64" -> "i64"
            | "byte" -> "i8"
            | "bool" -> "i1"
            | "string" -> "!llvm.ptr"
            | "nativeint" -> "i64"
            | "unativeint" -> "i64"
            | name when name.Contains("nativeptr") -> "!llvm.ptr"
            | "NativeStr" -> "!llvm.struct<(!llvm.ptr, i64)>"
            | _ ->
                // Try full name for more specific matching
                match typeDef.TryFullName with
                | Some "System.Int32" -> "i32"
                | Some "System.Int64" -> "i64"
                | Some "System.Byte" -> "i8"
                | Some "System.Boolean" -> "i1"
                | Some "System.String" -> "!llvm.ptr"
                | Some "System.IntPtr" -> "i64"
                | Some "System.UIntPtr" -> "i64"
                | Some "Microsoft.FSharp.Core.Unit" -> "()"
                | Some typeName when typeName.Contains("nativeptr") -> "!llvm.ptr"
                | Some typeName when typeName.Contains("NativeStr") -> "!llvm.struct<(!llvm.ptr, i64)>"
                | Some typeName ->
                    // Log unknown types for debugging (but don't fail)
                    // printfn "[TRANSFER] Unknown type: %s (display: %s)" typeName displayName
                    "!llvm.ptr"  // Default for unknown reference types
                | None ->
                    // No full name available - use display name heuristics
                    "()"  // Default to unit for safety
        else
            "()"  // No type definition - default to unit for safety
    with _ ->
        "()"  // On exception, default to unit

// ═══════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════

/// Recall SSA values for all children of a node (they're already witnessed by post-order)
let recallChildSSAs (graph: ProgramSemanticGraph) (node: PSGNode) (mlirZ: MLIRZipper) : (string * string) list =
    getChildNodes graph node
    |> List.choose (fun child -> MLIRZipper.recallNodeSSA child.Id.Value mlirZ)

/// Recall SSA values for children that are NOT function identifiers
let recallOperandSSAs (graph: ProgramSemanticGraph) (node: PSGNode) (mlirZ: MLIRZipper) : (string * string) list =
    getChildNodes graph node
    |> List.filter (fun c ->
        not (SyntaxKindT.isLongIdent c.Kind) &&
        not (SyntaxKindT.isIdent c.Kind))
    |> List.choose (fun child -> MLIRZipper.recallNodeSSA child.Id.Value mlirZ)

/// Extract constant value and MLIR type from a Const node
let extractConstant (node: PSGNode) : (string * string) option =
    match node.ConstantValue with
    | Some (StringValue s) -> Some (s, "string")
    | Some (Int32Value i) -> Some (string i, "i32")
    | Some (Int64Value i) -> Some (sprintf "%dL" i, "i64")
    | Some (BoolValue b) -> Some ((if b then "1" else "0"), "i1")  // MLIR uses 1/0 for i1, not true/false
    | Some (ByteValue b) -> Some (string (int b), "i8")
    | Some (FloatValue f) -> Some (string f, "f64")
    | Some (CharValue c) -> Some (string (int c), "i8")
    | Some UnitValue -> Some ("()", "unit")
    | None -> None

// ═══════════════════════════════════════════════════════════════════
// Function Inlining Helpers - For witnessing user-defined function calls
// ═══════════════════════════════════════════════════════════════════

/// Find a function's Binding node in the PSG by name
let findFunctionBinding (graph: ProgramSemanticGraph) (funcName: string) : PSGNode option =
    graph.Nodes
    |> Map.toSeq
    |> Seq.tryFind (fun (_, node) ->
        let isBinding = SyntaxKindT.isBinding node.Kind
        let symbolMatch =
            match node.Symbol with
            | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
                mfv.DisplayName = funcName ||
                (try mfv.CompiledName = funcName with _ -> false)
            | _ -> false
        isBinding && symbolMatch && node.IsReachable)
    |> Option.map snd

/// Extract parameter nodes and body node from a function binding
let getFunctionParamsAndBody (graph: ProgramSemanticGraph) (bindingNode: PSGNode) : (PSGNode list * PSGNode option) =
    let children = getChildNodes graph bindingNode
    let patternNodes =
        children
        |> List.filter (fun c -> SyntaxKindT.isPattern c.Kind)
    let bodyNode =
        children
        |> List.tryFind (fun c -> not (SyntaxKindT.isPattern c.Kind))
    // Extract named parameters from pattern nodes
    let rec extractNamedParams (node: PSGNode) : PSGNode list =
        if SyntaxKindT.isNamedPattern node.Kind then
            [node]
        else
            getChildNodes graph node
            |> List.collect extractNamedParams
    let paramNodes = patternNodes |> List.collect extractNamedParams
    (paramNodes, bodyNode)

/// Extract function name from a node (ident or longident)
let extractFunctionName (node: PSGNode) : string option =
    match node.Symbol with
    | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
        Some (try mfv.CompiledName with _ -> mfv.DisplayName)
    | _ -> None

/// Check if a binding node represents a function (has parameters, not a value binding)
let isFunction (graph: ProgramSemanticGraph) (node: PSGNode) : bool =
    let children = getChildNodes graph node
    // A function has Pattern:LongIdent or Pattern:Named children that contain parameters
    // Look for patterns with nested patterns (parameters)
    let patternChildren = children |> List.filter (fun c -> SyntaxKindT.isPattern c.Kind)
    if List.isEmpty patternChildren then false
    else
        // Check if any pattern has children (indicating parameters)
        patternChildren |> List.exists (fun pat ->
            let patKids = getChildNodes graph pat
            // Pattern:Const indicates unit parameter (), Pattern:Named indicates real param
            patKids |> List.exists (fun pk ->
                match pk.Kind with
                | SKPattern PNamed | SKPattern PConst | SKPattern PWild -> true
                | _ -> false))

/// Get mangled MLIR function name from a binding node
let getMangledFunctionName (node: PSGNode) : string =
    match node.Symbol with
    | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
        // Use full name with module path, replace . with _
        let fullName =
            try
                match mfv.DeclaringEntity with
                | Some entity -> entity.FullName + "." + mfv.CompiledName
                | None -> mfv.CompiledName
            with _ -> mfv.DisplayName
        fullName.Replace(".", "_").Replace("`", "_")
    | Some sym -> sym.DisplayName.Replace(".", "_")
    | None -> "unknown_func"

/// Determine return type for a function (unit functions return void/unit)
let getFunctionReturnType (graph: ProgramSemanticGraph) (node: PSGNode) : string =
    // For now, default to unit return - we'd need type info for accurate returns
    // Most Alloy functions return unit or simple types
    match node.Symbol with
    | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
        try
            let retType = mfv.ReturnParameter.Type
            mapFSharpTypeToMLIR retType
        with _ -> "()"  // Default to unit
    | _ -> "()"

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
        TransferAcc.withResult (TRError (sprintf "Unknown constant: %s" (SyntaxKindT.toString node.Kind))) acc

/// Witness a binding/let node
/// Children: Pattern nodes + value expression (already witnessed by post-order)
/// We bind the observed SSA from the value child to this node
let witnessBinding (acc: TransferAcc) (node: PSGNode) : TransferAcc =
    // Recall the value child SSA (skip Pattern nodes)
    let valueChildSSA =
        getChildNodes acc.Graph node
        |> List.filter (fun c -> not (SyntaxKindT.isPattern c.Kind))
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

/// Witness an if-then-else expression
/// NOTE: In post-order fold, children are already witnessed. This means the
/// then/else branch operations are already in the linear MLIR stream.
/// For proper control flow, we would need a different traversal strategy.
/// For now, we recall child SSAs and propagate the result.
let witnessIfThenElse (acc: TransferAcc) (node: PSGNode) : TransferAcc =
    let children = getChildNodes acc.Graph node
    // IfThenElse has: condition, thenBranch, [elseBranch]
    let conditionChild = children |> List.tryHead
    let thenChild = children |> List.tryItem 1
    let elseChild = children |> List.tryItem 2

    // Recall condition SSA (should be i1)
    let condSSA = conditionChild |> Option.bind (fun c -> MLIRZipper.recallNodeSSA c.Id.Value acc.MLIR)

    // For now, just recall the then-branch result as the if result
    // (proper implementation would use phi nodes after control flow blocks)
    let resultSSA =
        match thenChild |> Option.bind (fun c -> MLIRZipper.recallNodeSSA c.Id.Value acc.MLIR) with
        | Some r -> Some r
        | None -> elseChild |> Option.bind (fun c -> MLIRZipper.recallNodeSSA c.Id.Value acc.MLIR)

    match resultSSA with
    | Some (ssa, mlirType) ->
        let mlirZ = MLIRZipper.bindNodeSSA node.Id.Value ssa mlirType acc.MLIR
        TransferAcc.withMLIRAndResult mlirZ (TRValue (ssa, mlirType)) acc
    | None ->
        // Both branches are unit - the if expression is unit
        TransferAcc.withResult TRVoid acc

/// Witness a while loop
/// NOTE: In post-order fold, children (condition, body) are already witnessed.
/// The loop semantics require re-evaluation which post-order doesn't support.
/// For now, we recognize the construct and return unit.
let witnessWhileLoop (acc: TransferAcc) (node: PSGNode) : TransferAcc =
    // While loops always return unit
    // The body has already been witnessed (linear emission) during post-order
    TransferAcc.withResult TRVoid acc

/// Witness a match expression
/// NOTE: Similar to IfThenElse - branches are already witnessed in post-order.
/// We recall the result from the first match clause for now.
let witnessMatch (acc: TransferAcc) (node: PSGNode) : TransferAcc =
    let children = getChildNodes acc.Graph node
    // Match has: expression, clauses...
    // Skip the first child (matched expression), look for clause results
    let clauseChildren =
        children
        |> List.filter (fun c -> match c.Kind with SKExpr EMatchClause -> true | _ -> false)

    // Recall the first clause's result
    let resultSSA =
        clauseChildren
        |> List.tryHead
        |> Option.bind (fun clause ->
            // Get the result expression of the clause (last child that's not a pattern)
            getChildNodes acc.Graph clause
            |> List.filter (fun c -> not (SyntaxKindT.isPattern c.Kind))
            |> List.tryLast
            |> Option.bind (fun resultExpr -> MLIRZipper.recallNodeSSA resultExpr.Id.Value acc.MLIR))

    match resultSSA with
    | Some (ssa, mlirType) ->
        let mlirZ = MLIRZipper.bindNodeSSA node.Id.Value ssa mlirType acc.MLIR
        TransferAcc.withMLIRAndResult mlirZ (TRValue (ssa, mlirType)) acc
    | None ->
        TransferAcc.withResult TRVoid acc

/// Witness a mutable variable set (assignment)
/// Children: target identifier, new value expression
let witnessMutableSet (acc: TransferAcc) (node: PSGNode) : TransferAcc =
    // Mutable set is a statement that returns unit
    // The actual assignment would require memory operations which we defer
    TransferAcc.withResult TRVoid acc

/// Witness a tuple expression
/// Children: Elements of the tuple (already witnessed by post-order)
/// Fidelity Memory Model: Tuples are pure structural assembly - values in positions.
/// FSharpType determines the layout; we compute the MLIR struct type on-demand.
let witnessTuple (acc: TransferAcc) (node: PSGNode) : TransferAcc =
    let elementSSAs = recallOperandSSAs acc.Graph node acc.MLIR

    match elementSSAs with
    | [] -> TransferAcc.withResult TRVoid acc
    | [(ssa, mlirType)] ->
        // Single element tuple - pass through the element
        let mlirZ = MLIRZipper.bindNodeSSA node.Id.Value ssa mlirType acc.MLIR
        TransferAcc.withMLIRAndResult mlirZ (TRValue (ssa, mlirType)) acc
    | elements ->
        // Multi-element tuple: compute struct type from element types
        // Type layout: !llvm.struct<(type1, type2, ...)>
        let types = elements |> List.map snd
        let tupleType = sprintf "!llvm.struct<(%s)>" (String.concat ", " types)

        // Emit: undef + insertvalue sequence (structural assembly)
        let ssaName, mlirZ1 = MLIRZipper.yieldSSA acc.MLIR
        let mlirZ2 = MLIRZipper.witnessVoidOp (sprintf "%s = llvm.mlir.undef : %s" ssaName tupleType) mlirZ1

        // Insert each element value into its position
        let (finalSSA, finalZ) =
            elements |> List.indexed
            |> List.fold (fun (currentSSA, z) (idx, (elemSSA, _)) ->
                let newSSA, z1 = MLIRZipper.yieldSSA z
                let op = sprintf "%s = llvm.insertvalue %s, %s[%d] : %s" newSSA elemSSA currentSSA idx tupleType
                (newSSA, MLIRZipper.witnessVoidOp op z1)
            ) (ssaName, mlirZ2)

        let mlirZ3 = MLIRZipper.bindNodeSSA node.Id.Value finalSSA tupleType finalZ
        TransferAcc.withMLIRAndResult mlirZ3 (TRValue (finalSSA, tupleType)) acc

/// Witness a record construction (Record:New)
/// Handles TWO syntactic forms that both map to structural construction:
/// 1. Record expression: { Field1 = v1; Field2 = v2 } → children are RecordField:* nodes
/// 2. Struct constructor: TypeName(arg1, arg2) → children are Ident + Tuple
///
/// Fidelity Memory Model: Records/structs are pure structural assembly - values in slots.
/// FSharpType determines the layout; we compute the MLIR struct type on-demand.
let witnessRecord (acc: TransferAcc) (node: PSGNode) : TransferAcc =
    let children = getChildNodes acc.Graph node

    // Try both syntactic forms to extract field SSAs
    let fieldSSAs =
        // Form 1: Record expression syntax - look for RecordField children
        let recordFieldSSAs =
            children
            |> List.filter (fun c -> match c.Kind with SKExpr ERecordField -> true | _ -> false)
            |> List.choose (fun fieldNode ->
                // RecordField has one child: the value expression
                getChildNodes acc.Graph fieldNode
                |> List.tryHead
                |> Option.bind (fun valueNode -> MLIRZipper.recallNodeSSA valueNode.Id.Value acc.MLIR))

        if not (List.isEmpty recordFieldSSAs) then
            recordFieldSSAs
        else
            // Form 2: Struct constructor syntax - look for Tuple child (skipping the Ident)
            // Structure: Record:New → [Ident:TypeName, Tuple → [arg1, arg2, ...]]
            let tupleChild =
                children
                |> List.tryFind (fun c -> match c.Kind with SKExpr ETuple -> true | _ -> false)

            match tupleChild with
            | Some tuple ->
                // Get the tuple's children (the constructor arguments)
                getChildNodes acc.Graph tuple
                |> List.choose (fun argNode -> MLIRZipper.recallNodeSSA argNode.Id.Value acc.MLIR)
            | None ->
                // Maybe single argument (not wrapped in tuple)
                // Look for any non-Ident child that might be the single argument
                children
                |> List.filter (fun c -> not (SyntaxKindT.isIdent c.Kind))
                |> List.choose (fun argNode -> MLIRZipper.recallNodeSSA argNode.Id.Value acc.MLIR)

    match fieldSSAs with
    | [] -> TransferAcc.withResult TRVoid acc
    | [(ssa, mlirType)] ->
        // Single-field record - just bind the SSA
        let mlirZ = MLIRZipper.bindNodeSSA node.Id.Value ssa mlirType acc.MLIR
        TransferAcc.withMLIRAndResult mlirZ (TRValue (ssa, mlirType)) acc
    | fields ->
        // Multi-field record: compute struct type from field types
        // Type layout: !llvm.struct<(type1, type2, ...)>
        let types = fields |> List.map snd
        let recordType = sprintf "!llvm.struct<(%s)>" (String.concat ", " types)

        // Emit: undef + insertvalue sequence (structural assembly)
        let ssaName, mlirZ1 = MLIRZipper.yieldSSA acc.MLIR
        let mlirZ2 = MLIRZipper.witnessVoidOp (sprintf "%s = llvm.mlir.undef : %s" ssaName recordType) mlirZ1

        // Insert each field value into its slot
        let (finalSSA, finalZ) =
            fields |> List.indexed
            |> List.fold (fun (currentSSA, z) (idx, (fieldSSA, _)) ->
                let newSSA, z1 = MLIRZipper.yieldSSA z
                let op = sprintf "%s = llvm.insertvalue %s, %s[%d] : %s" newSSA fieldSSA currentSSA idx recordType
                (newSSA, MLIRZipper.witnessVoidOp op z1)
            ) (ssaName, mlirZ2)

        let mlirZ3 = MLIRZipper.bindNodeSSA node.Id.Value finalSSA recordType finalZ
        TransferAcc.withMLIRAndResult mlirZ3 (TRValue (finalSSA, recordType)) acc

/// Witness a sequential expression
/// Children: Sequence items (already witnessed by post-order)
/// Result: The LAST child's observed result
let witnessSequential (acc: TransferAcc) (node: PSGNode) : TransferAcc =
    // Recall the last child's SSA (that's the result of a sequence)
    let lastChildSSA =
        getChildNodes acc.Graph node
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
        TransferAcc.withResult (TRError (sprintf "Unresolved identifier: %s" (SyntaxKindT.toString node.Kind))) acc

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
        getChildNodes acc.Graph node
        |> List.filter (fun c ->
            not (SyntaxKindT.isLongIdent c.Kind) &&
            not (SyntaxKindT.isIdent c.Kind))

    match argChildren with
    | [argNode] ->
        // Check if it's a string constant (special case: known content and length)
        if SyntaxKindT.isConst argNode.Kind then
            let stringContent =
                match argNode.ConstantValue with
                | Some (StringValue s) -> s
                | _ -> ""

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

/// Try to find a function binding by name and get its mangled MLIR name
let tryGetFunctionMLIRName (graph: ProgramSemanticGraph) (funcName: string) : string option =
    findFunctionBinding graph funcName
    |> Option.map getMangledFunctionName

/// Witness a function call - emit llvm.call to a user-defined function
let witnessFunctionCall (acc: TransferAcc) (node: PSGNode) (funcName: string) : TransferAcc =
    match tryGetFunctionMLIRName acc.Graph funcName with
    | Some mlirFuncName ->
        // Get return type of the function
        let returnType =
            findFunctionBinding acc.Graph funcName
            |> Option.map (fun fn -> getFunctionReturnType acc.Graph fn)
            |> Option.defaultValue "()"

        // For now, call with no arguments (unit parameter functions)
        // TODO: Handle functions with real parameters
        if returnType = "()" then
            let callOp = sprintf "llvm.call @%s() : () -> ()" mlirFuncName
            let mlirZ = MLIRZipper.witnessVoidOp callOp acc.MLIR
            TransferAcc.withMLIRAndResult mlirZ TRVoid acc
        else
            let ssaName, mlirZ1 = MLIRZipper.yieldSSA acc.MLIR
            let callOp = sprintf "%s = llvm.call @%s() : () -> %s" ssaName mlirFuncName returnType
            let mlirZ2 = MLIRZipper.witnessVoidOp callOp mlirZ1
            let mlirZ3 = MLIRZipper.bindNodeSSA node.Id.Value ssaName returnType mlirZ2
            TransferAcc.withMLIRAndResult mlirZ3 (TRValue (ssaName, returnType)) acc
    | None ->
        // Function not found - likely an extern/intrinsic we haven't handled
        TransferAcc.withResult (TRError (sprintf "Unknown function: %s" funcName)) acc

/// Witness an application node
/// Dispatch by Operation field (set by ClassifyOperations nanopass)
let witnessApp (acc: TransferAcc) (node: PSGNode) : TransferAcc =
    // DIAGNOSTIC: Log operation
    eprintfn "  [witnessApp] %s: Operation=%A" node.Id.Value node.Operation

    match node.Operation with
    | Some (OperationKind.Arithmetic op) -> witnessArithmetic acc node op
    | Some (OperationKind.Comparison op) -> witnessComparison acc node op
    | Some (OperationKind.Console op) -> witnessConsole acc node op
    | Some (OperationKind.Core Ignore) ->
        // ignore: argument was evaluated (by post-order), result discarded
        TransferAcc.withResult TRVoid acc
    | Some (OperationKind.RegularCall callInfo) ->
        // User-defined function call - emit llvm.call
        witnessFunctionCall acc node callInfo.FunctionName
    | Some op ->
        eprintfn "  [witnessApp] UNHANDLED: %A" op
        TransferAcc.withResult (TRError (sprintf "Operation %A not yet implemented in Transfer.fs" op)) acc
    | None ->
        // Unclassified app - check if it's a function call to a user-defined function
        let children = getChildNodes acc.Graph node
        let funcIdent = children |> List.tryFind (fun c ->
            SyntaxKindT.isIdent c.Kind || SyntaxKindT.isLongIdent c.Kind)

        match funcIdent with
        | Some identNode ->
            // Extract function name and check if it's a known user-defined function
            match extractFunctionName identNode with
            | Some funcName when findFunctionBinding acc.Graph funcName |> Option.isSome ->
                // It's a call to a user-defined function - emit llvm.call
                witnessFunctionCall acc node funcName
            | Some ".ctor" ->
                // HARD ERROR: .ctor is a BCL/CLR artifact that should NEVER appear in PSG.
                // F# construction is structural (Record:New, Tuple) - not method calls.
                // If we see .ctor here, something upstream is broken:
                // - PSG Builder produced wrong structure, OR
                // - Typed tree correlation failed, OR
                // - Alloy has a stub instead of real implementation
                // This MUST be surfaced as an error, not silently handled.
                TransferAcc.withResult (TRError "PIPELINE BUG: .ctor in PSG - construction should be structural (Record:New, Tuple), not method calls") acc
            | _ ->
                // Not a user-defined function - fall through to default behavior
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
        | None ->
            // No function identifier - just pass through last child result
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

    // Skip unreachable nodes
    if not node.IsReachable then
        acc
    else

    // Dispatch on typed Kind
    let result =
        match node.Kind with
        | SKExpr EConst -> witnessConst acc node
        | SKExpr EIdent | SKExpr ELongIdent -> witnessIdent acc node
        | SKExpr ESequential -> witnessSequential acc node
        | SKExpr EApp | SKExpr ETypeApp -> witnessApp acc node
        | SKExpr EIfThenElse -> witnessIfThenElse acc node
        | SKExpr EWhileLoop -> witnessWhileLoop acc node
        | SKExpr EForLoop -> acc  // For loops - handled structurally
        | SKExpr EMatch -> witnessMatch acc node
        | SKExpr EMatchClause -> acc  // Match clauses are handled by witnessMatch
        | SKExpr EMatchLambda -> acc  // Match lambdas are handled like functions
        | SKExpr EMutableSet -> witnessMutableSet acc node
        | SKExpr ETuple -> witnessTuple acc node
        | SKExpr ERecord -> witnessRecord acc node  // Structural record construction
        | SKExpr ERecordField -> acc  // RecordField nodes are structural, witnessed via parent Record
        | SKExpr ELetOrUse -> acc  // Let bindings - handled via children
        | SKExpr ELambda -> acc  // Lambdas - handled via witnessBinding
        | SKExpr ETryWith | SKExpr ETryFinally -> acc  // Exception handling - structural
        | SKExpr EMethodCall | SKExpr EPropertyAccess -> acc  // Method/property - handled via App
        | SKExpr EAddressOf -> acc  // Address-of - structural
        | SKExpr EArrayOrList -> acc  // Array/list construction - structural
        | SKExpr EIndexGet | SKExpr EIndexSet -> acc  // Indexer operations - structural
        | SKExpr ETraitCall -> acc  // SRTP - resolved via typed tree
        | SKExpr EInterpolatedString | SKExpr EInterpolatedPart -> acc  // Interpolated strings - lowered in nanopass
        | SKExpr EObjExpr -> acc  // Object expressions - structural
        | SKExpr EUpcast | SKExpr EDowncast | SKExpr ETypeTest -> acc  // Type casts - structural
        | SKExpr EDo | SKExpr EAssert | SKExpr ELazy -> acc  // Effects - structural
        | SKExpr ENew -> acc  // Object construction - structural
        | SKBinding _ -> witnessBinding acc node
        | SKPattern _ -> acc  // Patterns are structural, no emission
        | SKDecl _ -> acc  // Declarations are structural, no emission
        | SKUnknown ->
            // Unknown nodes - pass through silently
            TransferAcc.withResult (TRError (sprintf "Unknown node kind: %s" (SyntaxKindT.toString node.Kind))) acc

    // DIAGNOSTIC: Log errors
    match result.LastResult with
    | TRError err -> eprintfn "  [ERR] %s: %s" node.Id.Value err
    | _ -> ()

    result

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
// Generation Result
// ═══════════════════════════════════════════════════════════════════

/// Result of MLIR generation
type GenerationResult = {
    Content: string
    Errors: string list
    HasErrors: bool
    /// Diagnostic: functions collected by collectReachableFunctions
    CollectedFunctions: string list
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

// ═══════════════════════════════════════════════════════════════════
// Function Collection - Find all reachable function bindings
// ═══════════════════════════════════════════════════════════════════

/// Diagnostic: count how many nodes pass each filter stage
type FilterDiagnostics = {
    TotalNodes: int
    AfterReachable: int
    AfterIsBinding: int
    AfterNotEntryPoint: int
    AfterNotMutable: int
    AfterNotInEntrySet: int
    AfterNoPlatformBinding: int
    AfterNotInline: int
    AfterIsFunction: int
}

/// Collect all reachable function bindings (excluding entry points which are handled separately)
let collectReachableFunctions (psg: ProgramSemanticGraph) : PSGNode list =
    let entryPointIds = psg.EntryPoints |> List.map (fun id -> id.Value) |> Set.ofList
    let allNodes = psg.Nodes |> Map.toList |> List.map snd

    // Diagnostic: track filter stages
    let afterReachable = allNodes |> List.filter (fun n -> n.IsReachable)
    let afterIsBinding = afterReachable |> List.filter (fun n -> SyntaxKindT.isBinding n.Kind)
    let afterNotEntry = afterIsBinding |> List.filter (fun n -> not (match n.Kind with SKBinding BEntryPoint -> true | _ -> false))
    let afterNotMutable = afterNotEntry |> List.filter (fun n -> not (match n.Kind with SKBinding BMutable -> true | _ -> false))
    let afterNotInEntrySet = afterNotMutable |> List.filter (fun n -> not (Set.contains n.Id.Value entryPointIds))
    let afterNoPlatform = afterNotInEntrySet |> List.filter (fun n -> n.PlatformBinding.IsNone)
    let afterNotInline = afterNoPlatform |> List.filter (fun n -> not n.IsInlineFunction)
    let afterIsFunc = afterNotInline |> List.filter (fun n -> isFunction psg n)

    // Print diagnostic
    eprintfn "[DIAG] collectReachableFunctions filter stages:"
    eprintfn "  Total nodes: %d" allNodes.Length
    eprintfn "  After IsReachable: %d" afterReachable.Length
    eprintfn "  After isBinding: %d" afterIsBinding.Length
    eprintfn "  After not EntryPoint: %d" afterNotEntry.Length
    eprintfn "  After not Mutable: %d" afterNotMutable.Length
    eprintfn "  After not in EntrySet: %d" afterNotInEntrySet.Length
    eprintfn "  After no PlatformBinding: %d" afterNoPlatform.Length
    eprintfn "  After not Inline: %d" afterNotInline.Length
    eprintfn "  After isFunction: %d" afterIsFunc.Length

    // Show some candidates that fail isFunction
    if afterNotInline.Length > 0 && afterIsFunc.Length = 0 then
        eprintfn "  [DIAG] First 5 candidates failing isFunction:"
        for node in afterNotInline |> List.truncate 5 do
            let children = getChildNodes psg node
            let patternChildren = children |> List.filter (fun c -> SyntaxKindT.isPattern c.Kind)
            eprintfn "    %s (kind=%A, %d children, %d pattern children)"
                node.Id.Value node.Kind children.Length patternChildren.Length
            for pat in patternChildren do
                let patKids = getChildNodes psg pat
                eprintfn "      Pattern %s has %d children: %A"
                    pat.Id.Value patKids.Length (patKids |> List.map (fun k -> k.Kind))

    afterIsFunc

// ═══════════════════════════════════════════════════════════════════
// Function MLIR Generation - Generate function definitions
// ═══════════════════════════════════════════════════════════════════

/// Generate MLIR for a single non-entry-point function
let witnessNonEntryFunction (psg: ProgramSemanticGraph) (funcNode: PSGNode) (mlirZ: MLIRZipper) : MLIRZipper * string list =
    let errors = ResizeArray<string>()
    let funcName = getMangledFunctionName funcNode
    let returnType = getFunctionReturnType psg funcNode

    // Witness function header - for now, all non-entry functions take no args and return unit
    // Full parameter handling would require more sophisticated type analysis
    let mlirZ1 =
        if returnType = "()" then
            MLIRZipper.witnessVoidOp (sprintf "llvm.func @%s() {" funcName) mlirZ
        else
            MLIRZipper.witnessVoidOp (sprintf "llvm.func @%s() -> %s {" funcName returnType) mlirZ

    // Create zipper and witness all nodes in function body
    let psgZipper = PSGZipper.create psg funcNode
    let acc = { TransferAcc.create psg with MLIR = mlirZ1 }
    let finalAcc = PSGZipper.foldPostOrder witnessNode acc psgZipper

    // Check for errors
    match finalAcc.LastResult with
    | TRError err -> errors.Add(sprintf "[%s] %s" funcName err)
    | _ -> ()

    // Witness return - for non-entry functions, generate proper return
    let mlirZ2 =
        if returnType = "()" then
            MLIRZipper.witnessVoidOp "llvm.return" finalAcc.MLIR
        else
            match finalAcc.LastResult with
            | TRValue (ssa, mlirType) when mlirType = returnType ->
                MLIRZipper.witnessVoidOp (sprintf "llvm.return %s : %s" ssa returnType) finalAcc.MLIR
            | TRValue (ssa, _) ->
                // Type mismatch - try to return anyway (may need cast)
                MLIRZipper.witnessVoidOp (sprintf "llvm.return %s : %s" ssa returnType) finalAcc.MLIR
            | _ ->
                // No value available - generate a default zero value for the return type
                // This handles cases where function bodies have unimplemented constructs
                let defaultSSA, mlirZ' = MLIRZipper.yieldSSA finalAcc.MLIR
                let (mlirZ'', returnVal) =
                    match returnType with
                    | "i32" ->
                        let m = MLIRZipper.witnessVoidOp (sprintf "%s = arith.constant 0 : i32" defaultSSA) mlirZ'
                        (m, defaultSSA)
                    | "i64" ->
                        let m = MLIRZipper.witnessVoidOp (sprintf "%s = arith.constant 0 : i64" defaultSSA) mlirZ'
                        (m, defaultSSA)
                    | "i1" ->
                        let m = MLIRZipper.witnessVoidOp (sprintf "%s = arith.constant 0 : i1" defaultSSA) mlirZ'
                        (m, defaultSSA)
                    | "!llvm.ptr" ->
                        let m = MLIRZipper.witnessVoidOp (sprintf "%s = llvm.mlir.zero : !llvm.ptr" defaultSSA) mlirZ'
                        (m, defaultSSA)
                    | _ when returnType.StartsWith("!llvm.struct") ->
                        // For struct types, generate llvm.mlir.undef
                        let m = MLIRZipper.witnessVoidOp (sprintf "%s = llvm.mlir.undef : %s" defaultSSA returnType) mlirZ'
                        (m, defaultSSA)
                    | _ ->
                        // Unknown type - use undef
                        let m = MLIRZipper.witnessVoidOp (sprintf "%s = llvm.mlir.undef : %s" defaultSSA returnType) mlirZ'
                        (m, defaultSSA)
                errors.Add(sprintf "[%s] Generated default return value - function body incomplete" funcName)
                MLIRZipper.witnessVoidOp (sprintf "llvm.return %s : %s" returnVal returnType) mlirZ''

    // Close function
    let mlirZ3 = MLIRZipper.witnessVoidOp "}" mlirZ2

    (mlirZ3, errors |> Seq.toList)

/// Generate complete MLIR module from PSG
/// Main entry point for MLIR generation
let generateMLIR (psg: ProgramSemanticGraph) (_correlations: CorrelationState) (_targetTriple: string) (outputKind: OutputKind) : GenerationResult =
    let errors = ResizeArray<string>()

    // Create a shared zipper for the module
    let mutable mlirZ = MLIRZipper.create ()

    // STEP 1: Generate MLIR function definitions for all non-entry reachable functions
    // This ensures they exist before any call sites reference them
    let reachableFunctions = collectReachableFunctions psg
    for funcNode in reachableFunctions do
        let (mlirZ', funcErrors) = witnessNonEntryFunction psg funcNode mlirZ
        mlirZ <- mlirZ'
        errors.AddRange(funcErrors)

    // STEP 2: Process entry points (main functions with exit handling)
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

    // Diagnostic: collect function names for debugging
    let collectedFunctionNames =
        reachableFunctions
        |> List.map (fun node ->
            let name = getMangledFunctionName node
            sprintf "%s (id=%s, isFunc=%b)" name node.Id.Value (isFunction psg node))

    // Extract: collapse all witnessed observations to MLIR text
    {
        Content = MLIRZipper.extract mlirZ
        Errors = errors |> Seq.toList
        HasErrors = errors.Count > 0
        CollectedFunctions = collectedFunctionNames
    }
