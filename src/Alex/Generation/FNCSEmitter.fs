/// FNCSEmitter - Witness-based MLIR generation from FNCS SemanticGraph
///
/// ARCHITECTURAL FOUNDATION (December 2025):
/// - Receives FNCS SemanticGraph (types attached, SRTP resolved, hard-pruned)
/// - Folds post-order (children before parents - required for SSA)
/// - Witnesses each node according to its SemanticKind
/// - Uses PlatformDispatch for platform bindings
/// - Extracts final MLIR text via MLIRZipper.extract
///
/// The flow:
///   FNCS SemanticGraph → foldPostOrder → witness nodes → MLIRZipper → extract → MLIR text
module Alex.Generation.FNCSEmitter

open Core.FNCS.Integration
open Alex.CodeGeneration.MLIRTypes
open Alex.Traversal.MLIRZipper
open Alex.Bindings.BindingTypes

// ═══════════════════════════════════════════════════════════════════════════
// Emission State - Tracks node→SSA mappings during traversal
// ═══════════════════════════════════════════════════════════════════════════

/// State carried through FNCS graph traversal
type EmissionContext = {
    /// The MLIRZipper accumulating MLIR output
    Zipper: MLIRZipper
    /// Map from FNCS NodeId (as int) to (SSA name, MLIR type)
    NodeSSA: Map<int, string * MLIRType>
    /// Current target platform
    Platform: TargetPlatform
    /// Errors accumulated during emission
    Errors: string list
}

module EmissionContext =
    /// Create initial emission context
    let create (platform: TargetPlatform) : EmissionContext = {
        Zipper = MLIRZipper.create ()
        NodeSSA = Map.empty
        Platform = platform
        Errors = []
    }

    /// Record SSA for a node (using NodeId converted to int)
    let bindSSA (nid: FNCSNodeId) (ssa: string) (ty: MLIRType) (ctx: EmissionContext) : EmissionContext =
        let intId = nodeIdToInt nid
        { ctx with NodeSSA = Map.add intId (ssa, ty) ctx.NodeSSA }

    /// Recall SSA for a node (using NodeId converted to int)
    let recallSSA (nid: FNCSNodeId) (ctx: EmissionContext) : (string * MLIRType) option =
        let intId = nodeIdToInt nid
        Map.tryFind intId ctx.NodeSSA

    /// Add an error
    let addError (msg: string) (ctx: EmissionContext) : EmissionContext =
        { ctx with Errors = msg :: ctx.Errors }

    /// Update zipper
    let withZipper (zipper: MLIRZipper) (ctx: EmissionContext) : EmissionContext =
        { ctx with Zipper = zipper }

// ═══════════════════════════════════════════════════════════════════════════
// Type Mapping - FNCS NativeType → MLIR Type
// ═══════════════════════════════════════════════════════════════════════════

/// Map FNCS NativeType to MLIRType
/// TODO: Complete mapping once FNCS exposes full NativeType structure
let mapFNCSType (ty: FNCSType) : MLIRType =
    // For now, use string representation to determine type
    let typeStr = formatTypeStr ty
    match typeStr with
    | "int32" | "int" -> Integer I32
    | "int64" -> Integer I64
    | "int16" -> Integer I16
    | "int8" | "byte" -> Integer I8
    | "bool" -> Integer I1
    | "unit" -> Unit
    | "string" | "NativeStr" -> Pointer  // Native strings are fat pointers
    | s when s.StartsWith("ptr<") -> Pointer
    | _ ->
        // Default to i32 for unknown types
        // TODO: Proper type mapping from FNCS NativeType variants
        Integer I32

// ═══════════════════════════════════════════════════════════════════════════
// Literal Emission
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a literal value
let witnessLiteral (ctx: EmissionContext) (value: FNCSLiteralValue option) (nid: FNCSNodeId) : EmissionContext =
    match value with
    | None ->
        EmissionContext.addError (sprintf "Node %d: Missing literal value" (nodeIdToInt nid)) ctx
    | Some litValue ->
        // TODO: Pattern match on LiteralValue variants from FNCS
        // For now, handle the string representation
        let litStr = sprintf "%A" litValue
        match litStr with
        | s when s.StartsWith("Int32") || s.StartsWith("int") ->
            // Parse integer value
            let numStr = s.Replace("Int32", "").Replace("int", "").Trim(' ', '(', ')')
            match System.Int64.TryParse(numStr) with
            | true, n ->
                let ssaName, zipper' = MLIRZipper.witnessConstant n I32 ctx.Zipper
                ctx |> EmissionContext.withZipper zipper'
                    |> EmissionContext.bindSSA nid ssaName (Integer I32)
            | false, _ ->
                EmissionContext.addError (sprintf "Node %d: Cannot parse int literal '%s'" (nodeIdToInt nid) s) ctx
        | s when s.StartsWith("Int64") ->
            let numStr = s.Replace("Int64", "").Trim(' ', '(', ')')
            match System.Int64.TryParse(numStr) with
            | true, n ->
                let ssaName, zipper' = MLIRZipper.witnessConstant n I64 ctx.Zipper
                ctx |> EmissionContext.withZipper zipper'
                    |> EmissionContext.bindSSA nid ssaName (Integer I64)
            | false, _ ->
                EmissionContext.addError (sprintf "Node %d: Cannot parse int64 literal '%s'" (nodeIdToInt nid) s) ctx
        | s when s.StartsWith("String") || s.StartsWith("\"") ->
            // String literal - observe and get pointer
            let content = s.Trim('"').Replace("String(", "").TrimEnd(')')
            let globalName, zipper1 = MLIRZipper.observeStringLiteral content ctx.Zipper
            let ptrSSA, zipper2 = MLIRZipper.witnessAddressOf globalName zipper1
            ctx |> EmissionContext.withZipper zipper2
                |> EmissionContext.bindSSA nid ptrSSA Pointer
        | s when s.StartsWith("Bool") || s = "true" || s = "false" ->
            let boolVal = if s.Contains("true") then 1L else 0L
            let ssaName, zipper' = MLIRZipper.witnessConstant boolVal I1 ctx.Zipper
            ctx |> EmissionContext.withZipper zipper'
                |> EmissionContext.bindSSA nid ssaName (Integer I1)
        | s when s = "Unit" || s = "()" ->
            // Unit literal - no SSA value produced
            ctx
        | _ ->
            EmissionContext.addError (sprintf "Node %d: Unknown literal kind '%s'" (nodeIdToInt nid) litStr) ctx

// ═══════════════════════════════════════════════════════════════════════════
// Platform Binding Emission
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a platform binding call
let witnessPlatformBinding (ctx: EmissionContext) (bindingName: string) (nid: FNCSNodeId) (node: FNCSNode) : EmissionContext =
    // Get child nodes as arguments
    let childIds = nodeChildren node
    let args =
        childIds
        |> List.choose (fun childId ->
            match EmissionContext.recallSSA childId ctx with
            | Some (ssa, ty) -> Some (ssa, ty)
            | None -> None)

    // Create PlatformPrimitive
    let prim : PlatformPrimitive = {
        EntryPoint = bindingName
        Library = "platform"
        CallingConvention = "ccc"
        Args = args
        ReturnType = mapFNCSType (nodeType node)
        BindingStrategy = Static
    }

    // Set target platform in dispatcher
    PlatformDispatch.setTargetPlatform ctx.Platform

    // Dispatch to platform binding
    let zipper', result = PlatformDispatch.dispatch prim ctx.Zipper

    let ctx' = EmissionContext.withZipper zipper' ctx

    match result with
    | WitnessedValue (ssa, ty) ->
        EmissionContext.bindSSA nid ssa ty ctx'
    | WitnessedVoid ->
        ctx'
    | NotSupported reason ->
        EmissionContext.addError (sprintf "Node %d: Platform binding '%s' not supported: %s" (nodeIdToInt nid) bindingName reason) ctx'

// ═══════════════════════════════════════════════════════════════════════════
// Variable Reference Emission
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a variable reference (looks up definition's SSA)
let witnessVarRef (ctx: EmissionContext) (defId: FNCSNodeId) (nid: FNCSNodeId) : EmissionContext =
    match EmissionContext.recallSSA defId ctx with
    | Some (ssa, ty) ->
        // Variable reference just uses the same SSA as the definition
        EmissionContext.bindSSA nid ssa ty ctx
    | None ->
        EmissionContext.addError (sprintf "Node %d: Variable reference to undefined node %d" (nodeIdToInt nid) (nodeIdToInt defId)) ctx

// ═══════════════════════════════════════════════════════════════════════════
// Binding (let) Emission
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a let binding
/// The binding's value should already be processed (post-order traversal)
let witnessBinding (ctx: EmissionContext) (node: FNCSNode) (nid: FNCSNodeId) : EmissionContext =
    // In post-order, children are already processed
    // The binding's value is the last child
    let childIds = nodeChildren node
    match List.tryLast childIds with
    | Some valueNodeId ->
        match EmissionContext.recallSSA valueNodeId ctx with
        | Some (ssa, ty) ->
            // The binding node maps to the same SSA as its value
            EmissionContext.bindSSA nid ssa ty ctx
        | None ->
            EmissionContext.addError (sprintf "Node %d: Binding value node %d has no SSA" (nodeIdToInt nid) (nodeIdToInt valueNodeId)) ctx
    | None ->
        EmissionContext.addError (sprintf "Node %d: Binding has no value child" (nodeIdToInt nid)) ctx

// ═══════════════════════════════════════════════════════════════════════════
// Application (Function Call) Emission
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a function application
let witnessApplication (ctx: EmissionContext) (node: FNCSNode) (nid: FNCSNodeId) : EmissionContext =
    // TODO: Handle different application kinds from FNCS
    // For now, assume it's a direct call where first child is function, rest are args
    let childIds = nodeChildren node
    match childIds with
    | [] ->
        EmissionContext.addError (sprintf "Node %d: Application has no children" (nodeIdToInt nid)) ctx
    | funcNodeId :: argNodeIds ->
        // Collect argument SSAs
        let args =
            argNodeIds
            |> List.choose (fun argId -> EmissionContext.recallSSA argId ctx)

        let argSSAs = args |> List.map fst
        let argTypes = args |> List.map snd

        // TODO: Get function name from FNCS node
        // For now, we'll emit a placeholder
        let funcName = sprintf "func_%d" (nodeIdToInt funcNodeId)
        let returnType = mapFNCSType (nodeType node)

        match returnType with
        | Unit ->
            let zipper' = MLIRZipper.witnessCallVoid funcName argSSAs argTypes ctx.Zipper
            EmissionContext.withZipper zipper' ctx
        | _ ->
            let resultSSA, zipper' = MLIRZipper.witnessCall funcName argSSAs argTypes returnType ctx.Zipper
            ctx |> EmissionContext.withZipper zipper'
                |> EmissionContext.bindSSA nid resultSSA returnType

// ═══════════════════════════════════════════════════════════════════════════
// Main Node Witness Function
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a single FNCS node - the main dispatch based on SemanticKind
/// Called by foldPostOrder, so children are already processed
let witnessNode (ctx: EmissionContext) (node: FNCSNode) : EmissionContext =
    // Get node ID for tracking
    let nid = nodeId node

    // Dispatch based on SemanticKind
    if isLiteral node then
        witnessLiteral ctx (literalValue node) nid
    elif isPlatformBinding node then
        match platformBindingName node with
        | Some name -> witnessPlatformBinding ctx name nid node
        | None -> EmissionContext.addError (sprintf "Node %d: Platform binding missing name" (nodeIdToInt nid)) ctx
    elif isBinding node then
        witnessBinding ctx node nid
    elif isApplication node then
        witnessApplication ctx node nid
    elif isLambda node then
        // TODO: Implement lambda emission
        EmissionContext.addError (sprintf "Node %d: Lambda emission not yet implemented" (nodeIdToInt nid)) ctx
    else
        // Other node kinds - pass through for now
        // TODO: Implement If, While, Tuple, Record, etc.
        ctx

// ═══════════════════════════════════════════════════════════════════════════
// Entry Point - Generate MLIR from FNCS Graph
// ═══════════════════════════════════════════════════════════════════════════

/// Result of MLIR generation
type EmissionResult = {
    /// Generated MLIR text
    MLIRContent: string
    /// Errors encountered during emission
    Errors: string list
    /// Functions that were emitted
    EmittedFunctions: string list
}

/// Generate MLIR from FNCS SemanticGraph
/// This is the main entry point for FNCS-based code generation
let generateMLIR (graph: FNCSGraph) (platform: TargetPlatform) : EmissionResult =
    // Create initial context
    let initialCtx = EmissionContext.create platform

    // Fold over graph in post-order (children before parents)
    // This ensures SSA values are available when needed
    let finalCtx = foldPostOrder witnessNode initialCtx graph

    // Extract MLIR text
    let mlirContent = MLIRZipper.extract finalCtx.Zipper

    {
        MLIRContent = mlirContent
        Errors = List.rev finalCtx.Errors
        EmittedFunctions = []  // TODO: Track emitted functions
    }

/// Generate MLIR with a minimal main function wrapper
/// For use when FNCS graph represents module-level code
let generateMLIRWithMain (graph: FNCSGraph) (platform: TargetPlatform) (mainName: string) : EmissionResult =
    // Create initial context
    let initialCtx = EmissionContext.create platform

    // TODO: Wrap in function definition
    // For now, emit as raw operations within a main function

    // Fold over graph in post-order
    let finalCtx = foldPostOrder witnessNode initialCtx graph

    // Get operations text
    let opsText = MLIRZipper.getOperationsText finalCtx.Zipper

    // Build MLIR module with main function
    let sb = System.Text.StringBuilder()
    sb.AppendLine("module {") |> ignore

    // Emit globals first
    for glb in List.rev finalCtx.Zipper.Globals do
        match glb with
        | StringLiteral (name, content, len) ->
            let escaped = Serialize.escape content
            sb.AppendLine(sprintf "  llvm.mlir.global internal constant @%s(\"%s\\00\") : !llvm.array<%d x i8>"
                name escaped len) |> ignore
        | ExternFunc (name, signature) ->
            sb.AppendLine(sprintf "  llvm.func @%s%s attributes {sym_visibility = \"private\"}"
                name signature) |> ignore

    // Emit main function
    sb.AppendLine(sprintf "  llvm.func @%s() -> i32 attributes {sym_visibility = \"public\"} {" mainName) |> ignore
    if not (System.String.IsNullOrWhiteSpace opsText) then
        for line in opsText.Split('\n') do
            if not (System.String.IsNullOrWhiteSpace line) then
                sb.AppendLine(sprintf "    %s" line) |> ignore

    // Default return 0 if no explicit return
    sb.AppendLine("    %retval = arith.constant 0 : i32") |> ignore
    sb.AppendLine("    llvm.return %retval : i32") |> ignore
    sb.AppendLine("  }") |> ignore
    sb.AppendLine("}") |> ignore

    {
        MLIRContent = sb.ToString()
        Errors = List.rev finalCtx.Errors
        EmittedFunctions = [mainName]
    }
