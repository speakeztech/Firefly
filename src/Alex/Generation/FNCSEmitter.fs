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
open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open Alex.CodeGeneration.MLIRTypes
open Alex.Traversal.MLIRZipper
open Alex.Bindings.BindingTypes
open Alex.Bindings.Console.ConsoleBindings

// ═══════════════════════════════════════════════════════════════════════════
// Emission State - Tracks node→SSA mappings during traversal
// ═══════════════════════════════════════════════════════════════════════════

/// State carried through FNCS graph traversal
type EmissionContext = {
    /// The MLIRZipper accumulating MLIR output
    Zipper: MLIRZipper
    /// Map from FNCS NodeId (as int) to (SSA name, MLIR type)
    NodeSSA: Map<int, string * MLIRType>
    /// Map from FNCS NodeId (as int) to string length (for write syscalls)
    StringLengths: Map<int, int>
    /// Map from FNCS NodeId (as int) to SSA name for dynamic string length
    DynamicStringLengths: Map<int, string>
    /// Current target platform
    Platform: TargetPlatform
    /// The full semantic graph (for looking up nodes by ID)
    Graph: FNCSGraph
    /// Errors accumulated during emission
    Errors: string list
    /// User-defined function bodies: Map from function name to MLIR body text
    UserFunctions: Map<string, string>
    /// Set of node IDs that are part of function definitions (to skip in main)
    FunctionNodes: Set<int>
}

module EmissionContext =
    /// Create initial emission context
    let create (platform: TargetPlatform) (graph: FNCSGraph) : EmissionContext = {
        Zipper = MLIRZipper.create ()
        NodeSSA = Map.empty
        StringLengths = Map.empty
        DynamicStringLengths = Map.empty
        Platform = platform
        Graph = graph
        Errors = []
        UserFunctions = Map.empty
        FunctionNodes = Set.empty
    }

    /// Look up a node in the graph by ID
    let lookupNode (nid: FNCSNodeId) (ctx: EmissionContext) : FNCSNode option =
        ctx.Graph.Nodes |> Map.tryFind nid

    /// Record SSA for a node (using NodeId converted to int)
    let bindSSA (nid: FNCSNodeId) (ssa: string) (ty: MLIRType) (ctx: EmissionContext) : EmissionContext =
        let intId = nodeIdToInt nid
        { ctx with NodeSSA = Map.add intId (ssa, ty) ctx.NodeSSA }

    /// Recall SSA for a node (using NodeId converted to int)
    let recallSSA (nid: FNCSNodeId) (ctx: EmissionContext) : (string * MLIRType) option =
        let intId = nodeIdToInt nid
        Map.tryFind intId ctx.NodeSSA

    /// Record string length for a literal node
    let bindStringLength (nid: FNCSNodeId) (len: int) (ctx: EmissionContext) : EmissionContext =
        let intId = nodeIdToInt nid
        { ctx with StringLengths = Map.add intId len ctx.StringLengths }

    /// Recall string length for a literal node
    let recallStringLength (nid: FNCSNodeId) (ctx: EmissionContext) : int option =
        let intId = nodeIdToInt nid
        Map.tryFind intId ctx.StringLengths

    /// Bind dynamic string length SSA for a node (for ReadLine etc.)
    let bindDynamicStringLength (nid: FNCSNodeId) (ssaName: string) (ctx: EmissionContext) : EmissionContext =
        let intId = nodeIdToInt nid
        { ctx with DynamicStringLengths = Map.add intId ssaName ctx.DynamicStringLengths }

    /// Recall dynamic string length SSA for a node
    let recallDynamicStringLength (nid: FNCSNodeId) (ctx: EmissionContext) : string option =
        let intId = nodeIdToInt nid
        Map.tryFind intId ctx.DynamicStringLengths

    /// Add an error
    let addError (msg: string) (ctx: EmissionContext) : EmissionContext =
        { ctx with Errors = msg :: ctx.Errors }

    /// Update zipper
    let withZipper (zipper: MLIRZipper) (ctx: EmissionContext) : EmissionContext =
        { ctx with Zipper = zipper }

    /// Register a user-defined function body
    let registerUserFunction (name: string) (body: string) (ctx: EmissionContext) : EmissionContext =
        { ctx with UserFunctions = Map.add name body ctx.UserFunctions }

    /// Mark a node as part of a function definition
    let markAsFunctionNode (nid: FNCSNodeId) (ctx: EmissionContext) : EmissionContext =
        { ctx with FunctionNodes = Set.add (nodeIdToInt nid) ctx.FunctionNodes }

    /// Check if a node is part of a function definition
    let isFunctionNode (nid: FNCSNodeId) (ctx: EmissionContext) : bool =
        Set.contains (nodeIdToInt nid) ctx.FunctionNodes

    /// Check if a name is a user-defined function
    let isUserFunction (name: string) (ctx: EmissionContext) : bool =
        Map.containsKey name ctx.UserFunctions

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
            // Extract the string content
            let content = 
                if s.StartsWith("String(\"") && s.EndsWith("\")") then
                    s.Substring(8, s.Length - 10)  // Remove String(" and ")
                elif s.StartsWith("String \"") && s.EndsWith("\"") then
                    s.Substring(8, s.Length - 9)   // Remove String " and "
                elif s.StartsWith("\"") && s.EndsWith("\"") then
                    s.Substring(1, s.Length - 2)   // Remove surrounding quotes
                else
                    s.Replace("String(", "").TrimEnd(')')
            let globalName, zipper1 = MLIRZipper.observeStringLiteral content ctx.Zipper
            let ptrSSA, zipper2 = MLIRZipper.witnessAddressOf globalName zipper1
            // Store both the pointer SSA AND the string length
            ctx |> EmissionContext.withZipper zipper2
                |> EmissionContext.bindSSA nid ptrSSA Pointer
                |> EmissionContext.bindStringLength nid content.Length
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
            let ctx' = EmissionContext.bindSSA nid ssa ty ctx
            // Propagate string length if present (static)
            let ctx'' =
                match EmissionContext.recallStringLength valueNodeId ctx with
                | Some len -> EmissionContext.bindStringLength nid len ctx'
                | None -> ctx'
            // Propagate dynamic string length if present
            match EmissionContext.recallDynamicStringLength valueNodeId ctx with
            | Some dynLenSSA -> EmissionContext.bindDynamicStringLength nid dynLenSSA ctx''
            | None -> ctx''
        | None ->
            EmissionContext.addError (sprintf "Node %d: Binding value node %d has no SSA" (nodeIdToInt nid) (nodeIdToInt valueNodeId)) ctx
    | None ->
        EmissionContext.addError (sprintf "Node %d: Binding has no value child" (nodeIdToInt nid)) ctx

// ═══════════════════════════════════════════════════════════════════════════
// Interpolated String Emission
// ═══════════════════════════════════════════════════════════════════════════

/// Witness an interpolated string by emitting sequential writes for each part
/// For string parts: emit a write syscall with the literal
/// For expression parts: emit the expression value (if string, write it; otherwise placeholder)
let witnessInterpolatedString (ctx: EmissionContext) (parts: InterpolatedPart list) (nid: FNCSNodeId) : EmissionContext =
    // For now, we emit a series of writes to stdout for each part
    let fd = 1 // stdout

    let rec emitParts (ctx: EmissionContext) (parts: InterpolatedPart list) : EmissionContext =
        match parts with
        | [] -> ctx
        | part :: rest ->
            let ctx' =
                match part with
                | InterpolatedPart.StringPart content ->
                    if System.String.IsNullOrEmpty content then
                        ctx  // Skip empty strings
                    else
                        // Emit write syscall for this string literal
                        // 1. Observe the string literal
                        let globalName, zipper1 = MLIRZipper.observeStringLiteral content ctx.Zipper
                        // 2. Get pointer to it
                        let ptrSSA, zipper2 = MLIRZipper.witnessAddressOf globalName zipper1
                        // 3. Emit fd constant
                        let fdSSA, zipper3 = MLIRZipper.witnessConstant (int64 fd) I32 zipper2
                        // 4. Emit length constant
                        let lenSSA, zipper4 = MLIRZipper.witnessConstant (int64 content.Length) I64 zipper3
                        // 5. Emit write syscall
                        let prim : PlatformPrimitive = {
                            EntryPoint = "writeBytes"
                            Library = "platform"
                            CallingConvention = "ccc"
                            Args = [(fdSSA, Integer I32); (ptrSSA, Pointer); (lenSSA, Integer I64)]
                            ReturnType = Integer I32
                            BindingStrategy = Static
                        }
                        PlatformDispatch.setTargetPlatform ctx.Platform
                        let zipper5, _result = PlatformDispatch.dispatch prim zipper4
                        EmissionContext.withZipper zipper5 ctx
                | InterpolatedPart.ExprPart exprNodeId ->
                    // The expression has already been processed (post-order)
                    // If it's a string, we can write it
                    // For now, check if we have SSA and if it's a pointer (string)
                    match EmissionContext.recallSSA exprNodeId ctx with
                    | Some (ptrSSA, Pointer) ->
                        // It's a string - check for dynamic length first, then static
                        match EmissionContext.recallDynamicStringLength exprNodeId ctx with
                        | Some dynLenSSA ->
                            // Use the dynamic length SSA directly
                            let fdSSA, zipper1 = MLIRZipper.witnessConstant (int64 fd) I32 ctx.Zipper
                            let prim : PlatformPrimitive = {
                                EntryPoint = "writeBytes"
                                Library = "platform"
                                CallingConvention = "ccc"
                                Args = [(fdSSA, Integer I32); (ptrSSA, Pointer); (dynLenSSA, Integer I64)]
                                ReturnType = Integer I32
                                BindingStrategy = Static
                            }
                            PlatformDispatch.setTargetPlatform ctx.Platform
                            let zipper2, _result = PlatformDispatch.dispatch prim zipper1
                            EmissionContext.withZipper zipper2 ctx
                        | None ->
                            // Fall back to static length
                            match EmissionContext.recallStringLength exprNodeId ctx with
                            | Some len ->
                                let fdSSA, zipper1 = MLIRZipper.witnessConstant (int64 fd) I32 ctx.Zipper
                                let lenSSA, zipper2 = MLIRZipper.witnessConstant (int64 len) I64 zipper1
                                let prim : PlatformPrimitive = {
                                    EntryPoint = "writeBytes"
                                    Library = "platform"
                                    CallingConvention = "ccc"
                                    Args = [(fdSSA, Integer I32); (ptrSSA, Pointer); (lenSSA, Integer I64)]
                                    ReturnType = Integer I32
                                    BindingStrategy = Static
                                }
                                PlatformDispatch.setTargetPlatform ctx.Platform
                                let zipper3, _result = PlatformDispatch.dispatch prim zipper2
                                EmissionContext.withZipper zipper3 ctx
                            | None ->
                                // No length info - emit placeholder
                                let zipper' = MLIRZipper.witnessVoidOp "// TODO: string length unknown for interpolated expression" ctx.Zipper
                                EmissionContext.withZipper zipper' ctx
                    | Some (ssaVal, ty) ->
                        // Non-string type - need to_string conversion
                        // For now, just emit placeholder
                        let zipper' = MLIRZipper.witnessVoidOp (sprintf "// TODO: convert %s to string for interpolation" (sprintf "%A" ty)) ctx.Zipper
                        EmissionContext.withZipper zipper' ctx
                    | None ->
                        let zipper' = MLIRZipper.witnessVoidOp (sprintf "// ERROR: no SSA for interpolated expr node %d" (nodeIdToInt exprNodeId)) ctx.Zipper
                        EmissionContext.withZipper zipper' ctx
            emitParts ctx' rest

    let finalCtx = emitParts ctx parts

    // The interpolated string itself produces a string pointer, but since we've
    // already emitted the writes, we don't need to bind an SSA value
    // (In a more complete implementation, we'd concatenate into a buffer)
    finalCtx

// ═══════════════════════════════════════════════════════════════════════════
// Library Binding Resolution
// ═══════════════════════════════════════════════════════════════════════════

/// Resolved binding target for known library functions
type ResolvedBinding =
    | PlatformWrite of fd: int    // Console.Write/WriteLine → write syscall to fd
    | PlatformRead of fd: int     // Console.ReadLine → read syscall from fd
    | PlatformExit                // Process.Exit → exit syscall
    | UnknownBinding of name: string

/// Resolve known Alloy library functions to their underlying platform bindings.
/// This bridges the gap between FNCS's external VarRefs and platform primitives.
/// NOTE: This is NOT pattern-matching in MLIR generation. This is semantic resolution
/// that happens before emission, mapping known library semantics to platform bindings.
let resolveLibraryBinding (funcName: string) : ResolvedBinding =
    match funcName with
    // Console output (all go to stdout=1)
    | "Console.Write" | "Console.write" | "write" -> PlatformWrite 1
    | "Console.WriteLine" | "Console.writeln" | "writeln" -> PlatformWrite 1
    | "Console.WriteErr" | "Console.writeErr" | "writeErr" -> PlatformWrite 2
    | "Console.WritelnErr" | "Console.writelnErr" | "writelnErr" -> PlatformWrite 2
    // Console input (from stdin=0)
    | "Console.ReadLine" | "Console.readln" | "readln" -> PlatformRead 0
    // Process control
    | "Bindings.abort" | "abort" -> PlatformExit
    // Platform bindings are already resolved
    | s when s.StartsWith("Bindings.") || s.StartsWith("Platform.Bindings.") ->
        match s with
        | "Bindings.writeBytes" | "Platform.Bindings.writeBytes" -> PlatformWrite 1
        | "Bindings.readBytes" | "Platform.Bindings.readBytes" -> PlatformRead 0
        | "Bindings.abort" | "Platform.Bindings.abort" -> PlatformExit
        | _ -> UnknownBinding s
    | _ -> UnknownBinding funcName

// ═══════════════════════════════════════════════════════════════════════════
// Application (Function Call) Emission
// ═══════════════════════════════════════════════════════════════════════════

/// Emit a platform write syscall
let emitPlatformWrite (ctx: EmissionContext) (fd: int) (argSSAs: string list) (argTypes: MLIRType list) (argNodeIds: FNCSNodeId list) (nid: FNCSNodeId) : EmissionContext =
    // For string arguments, we emit a write syscall
    // string -> ptr, length
    match argSSAs, argTypes, argNodeIds with
    | [ptrSSA], [Pointer], [argNid] ->
        // We have a string pointer - look up the length from the literal node
        match EmissionContext.recallStringLength argNid ctx with
        | Some len ->
            // Emit the file descriptor as a constant
            let fdSSA, zipper1 = MLIRZipper.witnessConstant (int64 fd) I32 ctx.Zipper
            // Emit the length as a constant
            let lenSSA, zipper2 = MLIRZipper.witnessConstant (int64 len) I64 zipper1
            // Now emit write syscall with all 3 arguments: fd, buf, count
            let prim : PlatformPrimitive = {
                EntryPoint = "writeBytes"
                Library = "platform"
                CallingConvention = "ccc"
                Args = [(fdSSA, Integer I32); (ptrSSA, Pointer); (lenSSA, Integer I64)]
                ReturnType = Integer I32
                BindingStrategy = Static
            }
            PlatformDispatch.setTargetPlatform ctx.Platform
            let zipper', result = PlatformDispatch.dispatch prim zipper2
            let ctx' = EmissionContext.withZipper zipper' ctx
            match result with
            | WitnessedValue (ssa, ty) -> EmissionContext.bindSSA nid ssa ty ctx'
            | WitnessedVoid -> ctx'
            | NotSupported reason ->
                EmissionContext.addError (sprintf "Node %d: Platform write not supported: %s" (nodeIdToInt nid) reason) ctx'
        | None ->
            // No string length recorded - try to emit without length (fallback)
            EmissionContext.addError (sprintf "Node %d: String length not available for write syscall" (nodeIdToInt nid)) ctx
    | _ ->
        EmissionContext.addError (sprintf "Node %d: Unexpected args for platform write (expected 1 pointer arg)" (nodeIdToInt nid)) ctx

/// Witness a function application
let witnessApplication (ctx: EmissionContext) (node: FNCSNode) (nid: FNCSNodeId) : EmissionContext =
    let childIds = nodeChildren node
    match childIds with
    | [] ->
        EmissionContext.addError (sprintf "Node %d: Application has no children" (nodeIdToInt nid)) ctx
    | funcNodeId :: argNodeIds ->
        // Collect argument SSAs (already processed in post-order)
        let args =
            argNodeIds
            |> List.choose (fun argId -> EmissionContext.recallSSA argId ctx)

        let argSSAs = args |> List.map fst
        let argTypes = args |> List.map snd

        // Look up the function node to get its name
        match EmissionContext.lookupNode funcNodeId ctx with
        | Some funcNode ->
            // Try to get the VarRef name
            match varRefName funcNode with
            | Some name ->
                // Resolve known library bindings
                match resolveLibraryBinding name with
                | PlatformWrite fd ->
                    // Pass argNodeIds so we can look up string lengths
                    emitPlatformWrite ctx fd argSSAs argTypes argNodeIds nid
                | PlatformRead fd ->
                    // Emit read syscall with a static buffer
                    // 1. Observe a static buffer for reading (256 bytes)
                    let bufferName = "read_buffer"
                    let bufferSize = 256
                    let zipper0 = MLIRZipper.observeStaticBuffer bufferName bufferSize ctx.Zipper
                    // 2. Get the buffer address
                    let bufPtrSSA, zipper1 = MLIRZipper.witnessAddressOf bufferName zipper0
                    // 3. Emit fd constant
                    let fdSSA, zipper2 = MLIRZipper.witnessConstant (int64 fd) I32 zipper1
                    // 4. Emit max count constant
                    let countSSA, zipper3 = MLIRZipper.witnessConstant (int64 (bufferSize - 1)) I64 zipper2
                    // 5. Sign extend fd to i64 for syscall
                    let fdExtSSA, zipper4 = MLIRZipper.yieldSSA zipper3
                    let extText = sprintf "%s = arith.extsi %s : i32 to i64" fdExtSSA fdSSA
                    let zipper5 = MLIRZipper.witnessOpWithResult extText fdExtSSA (Integer I64) zipper4
                    // 6. Emit syscall 0 (read): read(fd, buf, count) -> bytes_read
                    let syscallNumSSA, zipper6 = MLIRZipper.witnessConstant 0L I64 zipper5
                    let readResultSSA, zipper7 =
                        MLIRZipper.witnessSyscall syscallNumSSA
                            [(fdExtSSA, "i64"); (bufPtrSSA, "!llvm.ptr"); (countSSA, "i64")]
                            (Integer I64) zipper6
                    // 7. Subtract 1 from bytes_read to exclude the newline character
                    let oneSSA, zipper8 = MLIRZipper.witnessConstant 1L I64 zipper7
                    let adjustedLenSSA, zipper9 = MLIRZipper.witnessArith "arith.subi" readResultSSA oneSSA (Integer I64) zipper8
                    // 8. Store the buffer pointer as SSA for this node, and record dynamic length
                    ctx |> EmissionContext.withZipper zipper9
                        |> EmissionContext.bindSSA nid bufPtrSSA Pointer
                        // Track the dynamic length SSA (bytes read minus newline)
                        |> EmissionContext.bindDynamicStringLength nid adjustedLenSSA
                | PlatformExit ->
                    // TODO: Implement exit syscall
                    EmissionContext.addError (sprintf "Node %d: Exit syscall not yet implemented" (nodeIdToInt nid)) ctx
                | UnknownBinding unknownName ->
                    // Check if it's a user-defined function
                    if EmissionContext.isUserFunction unknownName ctx then
                        // Call user-defined function - always void for now
                        // (User functions return void, main body handles return value separately)
                        let zipper' = MLIRZipper.witnessCallVoid unknownName argSSAs argTypes ctx.Zipper
                        EmissionContext.withZipper zipper' ctx
                    else
                        // Unknown function - emit placeholder call
                        let returnType = mapFNCSType (nodeType node)
                        match returnType with
                        | Unit ->
                            let zipper' = MLIRZipper.witnessCallVoid unknownName argSSAs argTypes ctx.Zipper
                            EmissionContext.withZipper zipper' ctx
                        | _ ->
                            let resultSSA, zipper' = MLIRZipper.witnessCall unknownName argSSAs argTypes returnType ctx.Zipper
                            ctx |> EmissionContext.withZipper zipper'
                                |> EmissionContext.bindSSA nid resultSSA returnType
            | None ->
                // Function node is not a VarRef - might be a lambda or direct application
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
        | None ->
            EmissionContext.addError (sprintf "Node %d: Function node %d not found in graph" (nodeIdToInt nid) (nodeIdToInt funcNodeId)) ctx

// ═══════════════════════════════════════════════════════════════════════════
// User Function Identification and Emission
// ═══════════════════════════════════════════════════════════════════════════

/// Get binding name from a Binding node
let getBindingName (node: FNCSNode) : string option =
    bindingInfo node |> Option.map (fun (name, _, _) -> name)

/// Check if a node is a function definition (Binding with Lambda child)
let isFunctionBinding (node: FNCSNode) (graph: FNCSGraph) : bool =
    if isBinding node then
        let children = nodeChildren node
        match children with
        | [childId] ->
            match graph.Nodes |> Map.tryFind childId with
            | Some childNode -> isLambda childNode
            | None -> false
        | _ -> false
    else
        false

/// Get the Lambda node from a function binding
let getFunctionLambda (node: FNCSNode) (graph: FNCSGraph) : FNCSNode option =
    if isBinding node then
        let children = nodeChildren node
        match children with
        | [childId] ->
            match graph.Nodes |> Map.tryFind childId with
            | Some childNode when isLambda childNode -> Some childNode
            | _ -> None
        | _ -> None
    else
        None

/// Collect all descendant node IDs from a node (for marking function body nodes)
let rec collectDescendants (graph: FNCSGraph) (nid: FNCSNodeId) : FNCSNodeId list =
    match graph.Nodes |> Map.tryFind nid with
    | Some node ->
        let children = nodeChildren node
        nid :: (children |> List.collect (collectDescendants graph))
    | None -> [nid]

/// Identify all function bindings in the graph
let identifyFunctionBindings (graph: FNCSGraph) : (string * FNCSNode * FNCSNode) list =
    graph.Nodes
    |> Map.values
    |> Seq.choose (fun node ->
        if isFunctionBinding node graph then
            match getBindingName node, getFunctionLambda node graph with
            | Some name, Some lambdaNode -> Some (name, node, lambdaNode)
            | _ -> None
        else
            None)
    |> Seq.toList

// Note: emitUserFunction is defined after witnessNode to avoid forward reference

// ═══════════════════════════════════════════════════════════════════════════
// Main Node Witness Function
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a single FNCS node - the main dispatch based on SemanticKind
/// Called by foldPostOrder, so children are already processed
let witnessNode (ctx: EmissionContext) (node: FNCSNode) : EmissionContext =
    // Get node ID for tracking
    let nid = nodeId node

    // Skip nodes that are part of function definitions (they're emitted separately)
    if EmissionContext.isFunctionNode nid ctx then
        ctx
    // Dispatch based on SemanticKind
    elif isLiteral node then
        witnessLiteral ctx (literalValue node) nid
    elif isPlatformBinding node then
        match platformBindingName node with
        | Some name -> witnessPlatformBinding ctx name nid node
        | None -> EmissionContext.addError (sprintf "Node %d: Platform binding missing name" (nodeIdToInt nid)) ctx
    elif isBinding node then
        // Check if this is a function binding - if so, skip it (function is emitted separately)
        if isFunctionBinding node ctx.Graph then
            ctx
        else
            witnessBinding ctx node nid
    elif isApplication node then
        witnessApplication ctx node nid
    elif isVarRef node then
        // Variable reference - look up the definition's SSA
        match varRefDefinition node with
        | Some defId ->
            // Look up the binding that contains this definition
            // The definition is the binding node, so look up its SSA
            match EmissionContext.recallSSA defId ctx with
            | Some (ssa, ty) ->
                // Copy SSA binding
                let ctx' = EmissionContext.bindSSA nid ssa ty ctx
                // Copy static string length if present
                let ctx'' =
                    match EmissionContext.recallStringLength defId ctx with
                    | Some len -> EmissionContext.bindStringLength nid len ctx'
                    | None -> ctx'
                // Copy dynamic string length if present
                match EmissionContext.recallDynamicStringLength defId ctx with
                | Some dynLenSSA -> EmissionContext.bindDynamicStringLength nid dynLenSSA ctx''
                | None -> ctx''
            | None ->
                // Definition might not have been processed yet, or is external
                // For external refs (like Console.Write), we don't need SSA
                ctx
        | None ->
            // No definition tracked - might be external or unresolved
            ctx
    elif isLambda node then
        // Lambda nodes are handled when emitting function definitions
        ctx
    elif isInterpolatedString node then
        match interpolatedStringParts node with
        | Some parts -> witnessInterpolatedString ctx parts nid
        | None -> ctx
    else
        // Check for Error nodes
        match nodeKind node with
        | FSharp.Native.Compiler.Checking.Native.SemanticGraph.SemanticKind.Error msg ->
            // Emit a comment for the error
            let zipper' = MLIRZipper.witnessVoidOp (sprintf "// ERROR: %s" msg) ctx.Zipper
            EmissionContext.withZipper zipper' ctx
        | FSharp.Native.Compiler.Checking.Native.SemanticGraph.SemanticKind.Sequential _ ->
            // Sequential nodes just represent control flow - children already processed
            // If there are children, bind the last child's SSA to this node
            let childIds = nodeChildren node
            match List.tryLast childIds with
            | Some lastChildId ->
                match EmissionContext.recallSSA lastChildId ctx with
                | Some (ssa, ty) -> EmissionContext.bindSSA nid ssa ty ctx
                | None -> ctx
            | None -> ctx
        | _ ->
            // Other node kinds - pass through for now
            // TODO: Implement If, While, Tuple, Record, etc.
            ctx

// ═══════════════════════════════════════════════════════════════════════════
// User Function Emission
// ═══════════════════════════════════════════════════════════════════════════

/// Emit a single user function and return (updated context, function body MLIR, globals)
let emitUserFunction (graph: FNCSGraph) (platform: TargetPlatform) (userFuncNames: Set<string>) (name: string) (lambdaNode: FNCSNode) : string * MLIRGlobal list =
    // Create a fresh emission context for this function with user function names registered
    let funcCtx =
        { EmissionContext.create platform graph with
            UserFunctions = userFuncNames |> Set.toList |> List.map (fun n -> (n, "")) |> Map.ofList }

    // Get the Lambda's body - children of the Lambda minus the params
    let lambdaChildren = nodeChildren lambdaNode

    // Post-order traverse each body node's subtree
    let rec traverseSubtree (ctx: EmissionContext) (nid: FNCSNodeId) : EmissionContext =
        match EmissionContext.lookupNode nid ctx with
        | Some node ->
            // First process children
            let childIds = nodeChildren node
            let ctxAfterChildren = childIds |> List.fold traverseSubtree ctx
            // Then process this node
            witnessNode ctxAfterChildren node
        | None -> ctx

    let bodyCtx = lambdaChildren |> List.fold traverseSubtree funcCtx

    // Get the operations text for this function body
    let bodyText = MLIRZipper.getOperationsText bodyCtx.Zipper

    // Return body text and globals (for string literals defined in this function)
    (bodyText, bodyCtx.Zipper.Globals)

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
    // Ensure platform bindings are registered
    registerBindings()
    // Create initial context with the graph for node lookups
    let initialCtx = EmissionContext.create platform graph

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
    // Ensure platform bindings are registered
    registerBindings()

    // Step 1: Identify function bindings
    let functionBindings = identifyFunctionBindings graph

    // Step 2: Check if user defined a main function
    let hasUserMain = functionBindings |> List.exists (fun (name, _, _) -> name = mainName)

    // Step 3: Separate main from other user functions
    let (mainFuncOpt, otherFunctions) =
        if hasUserMain then
            let mainFunc = functionBindings |> List.find (fun (name, _, _) -> name = mainName)
            (Some mainFunc, functionBindings |> List.filter (fun (name, _, _) -> name <> mainName))
        else
            (None, functionBindings)

    // Collect all user function names for context
    let allUserFuncNames = functionBindings |> List.map (fun (name, _, _) -> name) |> Set.ofList

    // Step 4: Emit non-main user functions and collect their bodies + globals
    let userFunctionResults =
        otherFunctions
        |> List.map (fun (name, _bindingNode, lambdaNode) ->
            let bodyText, globals = emitUserFunction graph platform allUserFuncNames name lambdaNode
            (name, bodyText, globals))

    // Step 5: Emit main function body if user-defined
    let (mainBodyText, mainGlobals) =
        match mainFuncOpt with
        | Some (_, _, lambdaNode) ->
            emitUserFunction graph platform allUserFuncNames mainName lambdaNode
        | None ->
            ("", [])

    // Collect all function names and their body nodes
    let functionNodeSets =
        functionBindings
        |> List.collect (fun (_name, bindingNode, lambdaNode) ->
            let bindingId = nodeId bindingNode
            let lambdaId = nodeId lambdaNode
            let lambdaDescendants = collectDescendants graph lambdaId
            bindingId :: lambdaDescendants)
        |> List.map nodeIdToInt
        |> Set.ofList

    // Step 6: Create initial context with function nodes marked and user functions registered
    let initialCtx =
        { EmissionContext.create platform graph with
            FunctionNodes = functionNodeSets
            UserFunctions = functionBindings |> List.map (fun (name, _, _) -> (name, "")) |> Map.ofList }

    // Step 7: Fold over graph in post-order (skipping function nodes)
    let finalCtx = foldPostOrder witnessNode initialCtx graph

    // Step 8: Get operations text for main function body (non-function statements)
    let moduleOpsText = MLIRZipper.getOperationsText finalCtx.Zipper

    // Step 9: Collect all globals (from user functions and main)
    let allGlobals =
        let userFuncGlobals = userFunctionResults |> List.collect (fun (_, _, globals) -> globals)
        let allFuncGlobals = userFuncGlobals @ mainGlobals
        allFuncGlobals @ (List.rev finalCtx.Zipper.Globals)
        |> List.distinctBy (function
            | StringLiteral (name, _, _) -> name
            | ExternFunc (name, _) -> name
            | StaticBuffer (name, _) -> name)

    // Step 10: Build MLIR module
    let sb = System.Text.StringBuilder()
    sb.AppendLine("module {") |> ignore

    // Emit globals first
    for glb in allGlobals do
        match glb with
        | StringLiteral (name, content, len) ->
            let escaped = Serialize.escape content
            sb.AppendLine(sprintf "  llvm.mlir.global internal constant @%s(\"%s\\00\") : !llvm.array<%d x i8>"
                name escaped len) |> ignore
        | ExternFunc (name, signature) ->
            sb.AppendLine(sprintf "  llvm.func @%s%s attributes {sym_visibility = \"private\"}"
                name signature) |> ignore
        | StaticBuffer (name, size) ->
            // Use an empty string initializer for zero-init (null bytes)
            sb.AppendLine(sprintf "  llvm.mlir.global internal @%s() : !llvm.array<%d x i8>"
                name size) |> ignore

    // Emit non-main user-defined functions
    for (funcName, bodyText, _) in userFunctionResults do
        sb.AppendLine(sprintf "  llvm.func @%s() attributes {sym_visibility = \"private\"} {" funcName) |> ignore
        if not (System.String.IsNullOrWhiteSpace bodyText) then
            for line in bodyText.Split('\n') do
                if not (System.String.IsNullOrWhiteSpace line) then
                    sb.AppendLine(sprintf "    %s" line) |> ignore
        sb.AppendLine("    llvm.return") |> ignore
        sb.AppendLine("  }") |> ignore

    // Emit main function (user-defined body + any module-level statements + exit)
    sb.AppendLine(sprintf "  llvm.func @%s() -> i32 attributes {sym_visibility = \"public\"} {" mainName) |> ignore

    // First, any module-level operations
    if not (System.String.IsNullOrWhiteSpace moduleOpsText) then
        for line in moduleOpsText.Split('\n') do
            if not (System.String.IsNullOrWhiteSpace line) then
                sb.AppendLine(sprintf "    %s" line) |> ignore

    // Then, user-defined main body
    if not (System.String.IsNullOrWhiteSpace mainBodyText) then
        for line in mainBodyText.Split('\n') do
            if not (System.String.IsNullOrWhiteSpace line) then
                sb.AppendLine(sprintf "    %s" line) |> ignore

    // For freestanding: emit exit syscall (syscall 60 on Linux x86-64)
    // The exit code is 0
    sb.AppendLine("    %exit_code = arith.constant 0 : i64") |> ignore
    sb.AppendLine("    %syscall_num = arith.constant 60 : i64") |> ignore
    sb.AppendLine("    %_exit_result = llvm.inline_asm has_side_effects \"syscall\", \"={rax},{rax},{rdi},~{rcx},~{r11},~{memory}\" %syscall_num, %exit_code : (i64, i64) -> i64") |> ignore
    sb.AppendLine("    llvm.unreachable") |> ignore
    sb.AppendLine("  }") |> ignore
    sb.AppendLine("}") |> ignore

    {
        MLIRContent = sb.ToString()
        Errors = List.rev finalCtx.Errors
        EmittedFunctions = mainName :: (otherFunctions |> List.map (fun (name, _, _) -> name))
    }
