/// MLIRGeneration - MLIR generation using Zipper + XParsec architecture
///
/// This module implements code generation following the designed architecture:
/// - PSGZipper provides tree traversal (the "attention")
/// - XParsec provides local pattern matching on children
/// - ExternDispatch handles platform-specific extern primitives
/// - EmitContext accumulates MLIR output
///
/// NO central dispatch. NO pattern matching on symbol names.
/// The PSG structure drives generation.
module Alex.Generation.MLIRGeneration

open FSharp.Compiler.Symbols
open FSharp.Compiler.Symbols.FSharpExprPatterns
open Core.PSG.Types
open Alex.Traversal.PSGZipper
open Alex.Traversal.PSGXParsec
open Alex.Patterns.PSGPatterns
open Alex.CodeGeneration.MLIRBuilder
open Alex.Bindings.BindingTypes
open Core.PSG.Nanopass.DefUseEdges
open Baker.Types

// ═══════════════════════════════════════════════════════════════════
// Emission Helpers
// ═══════════════════════════════════════════════════════════════════

/// Extract constant value from node
let extractConstant (node: PSGNode) : (string * string) option =
    match node.ConstantValue with
    | Some (StringValue s) -> Some (s, "string")
    | Some (Int32Value i) -> Some (string i, "i32")
    | Some (Int64Value i) -> Some (string i, "i64")
    | Some (ByteValue b) -> Some (string (int b), "i8")
    | Some (BoolValue b) -> Some ((if b then "1" else "0"), "i1")
    | Some (CharValue c) -> Some (string (int c), "i8")
    | Some (FloatValue f) -> Some (string f, "f64")
    | Some UnitValue -> Some ("", "unit")
    | None ->
        let kind = node.SyntaxKind
        if kind.StartsWith("Const:") then
            let constKind = kind.Substring(6)
            match constKind with
            | "Int32 0" -> Some ("0", "i32")
            | s when s.StartsWith("Int32 ") -> Some (s.Substring(6), "i32")
            | s when s.StartsWith("Int64 ") -> Some (s.Substring(6), "i64")
            | s when s.StartsWith("Byte ") -> Some (s.Substring(5).TrimEnd([|'u'; 'y'|]), "i8")
            | "Unit" -> Some ("", "unit")
            | s when s.StartsWith("String ") -> Some (s.Substring(7), "string")
            | "String" -> Some ("", "string")
            | "Null" -> Some ("null", "!llvm.ptr")
            | _ -> None
        else None

// ═══════════════════════════════════════════════════════════════════
// Node Emitters - Local functions using XParsec at each zipper position
// ═══════════════════════════════════════════════════════════════════

/// Emit a constant node
let emitConstant (ctx: EmitContext) (node: PSGNode) : ExprResult =
    match extractConstant node with
    | Some (_, "unit") -> Void
    | Some (value, "string") ->
        let globalName = EmitContext.registerStringLiteral ctx value
        let ssa = EmitContext.nextSSA ctx
        EmitContext.emitLine ctx (sprintf "%s = llvm.mlir.addressof %s : !llvm.ptr" ssa globalName)
        Value (ssa, "!llvm.ptr")
    | Some ("null", "!llvm.ptr") ->
        let ssa = EmitContext.nextSSA ctx
        EmitContext.emitLine ctx (sprintf "%s = llvm.mlir.zero : !llvm.ptr" ssa)
        Value (ssa, "!llvm.ptr")
    | Some (value, mlirType) ->
        let ssa = EmitContext.nextSSA ctx
        EmitContext.emitLine ctx (sprintf "%s = arith.constant %s : %s" ssa value mlirType)
        Value (ssa, mlirType)
    | None -> EmitError (sprintf "Unknown constant: %s" node.SyntaxKind)

/// Emit an identifier (variable reference via def-use edges)
/// For mutable variables, emits a load from the pointer
let emitIdent (ctx: EmitContext) (node: PSGNode) : ExprResult =
    // Special handling for Unchecked.defaultof - returns zero/null for the type
    if node.SyntaxKind.Contains("Unchecked.defaultof") || node.SyntaxKind.Contains("defaultof") then
        // Default value - for pointer types, emit null; for numeric, emit 0
        match node.Type with
        | Some t when t.HasTypeDefinition ->
            let typeName = try t.TypeDefinition.TryFullName with _ -> None
            match typeName with
            | Some n when n.Contains("nativeptr") || n.Contains("voidptr") || n.Contains("IntPtr") ->
                let ssa = EmitContext.nextSSA ctx
                EmitContext.emitLine ctx (sprintf "%s = llvm.mlir.zero : !llvm.ptr" ssa)
                Value (ssa, "!llvm.ptr")
            | _ ->
                // Default to i64 zero for unknown types
                let ssa = EmitContext.nextSSA ctx
                EmitContext.emitLine ctx (sprintf "%s = arith.constant 0 : i64" ssa)
                Value (ssa, "i64")
        | _ ->
            // Without type info, assume pointer (most common use in our context)
            let ssa = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = llvm.mlir.zero : !llvm.ptr" ssa)
            Value (ssa, "!llvm.ptr")
    else
        match findDefiningNode ctx.Graph node with
        | Some defNode ->
            match EmitContext.lookupNodeSSA ctx defNode.Id with
            | Some (ptrSsa, "!llvm.ptr") ->
                // Check if this is a mutable binding (need to emit load)
                match EmitContext.lookupMutableBinding ctx defNode.Id with
                | Some valueType ->
                    // Emit a load from the pointer to get the actual value
                    let loadedSsa = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = llvm.load %s : !llvm.ptr -> %s" loadedSsa ptrSsa valueType)
                    Value (loadedSsa, valueType)
                | None ->
                    // Not a mutable binding, return pointer as-is (e.g., for nativeptr)
                    Value (ptrSsa, "!llvm.ptr")
            | Some (ssa, ty) -> Value (ssa, ty)
            | None ->
                // Definition not yet emitted - this is a forward reference issue
                // The definition should have been emitted by the normal tree walk
                // For now, report the error - proper fix would require dependency ordering
                EmitError (sprintf "Definition %s not yet emitted (forward reference)" defNode.Id.Value)
        | None ->
            EmitError "No defining node found"

/// Map FSharpType to MLIR type string (for struct field types)
let private mapFSharpTypeToMLIR (ftype: FSharpType) : string =
    try
        if ftype.HasTypeDefinition then
            match ftype.TypeDefinition.TryFullName with
            | Some "System.Int32" -> "i32"
            | Some "System.Int64" -> "i64"
            | Some "System.Int16" -> "i16"
            | Some "System.Byte" | Some "System.SByte" -> "i8"
            | Some "System.Boolean" -> "i1"
            | Some "System.IntPtr" | Some "System.UIntPtr" -> "!llvm.ptr"
            | Some n when n.Contains("nativeptr") -> "!llvm.ptr"
            | Some n when n.Contains("NativeStr") -> "!llvm.struct<(ptr, i64)>"
            | _ -> "i64"  // Default for struct fields
        else "i64"
    with _ -> "i64"

/// Map containing type to MLIR struct type
let private mapContainingTypeToMLIR (ftype: FSharpType) : string =
    try
        if ftype.HasTypeDefinition then
            match ftype.TypeDefinition.TryFullName with
            | Some n when n.Contains("NativeStr") -> "!llvm.struct<(ptr, i32)>"
            | _ -> "!llvm.struct<()>"  // Fallback
        else "!llvm.struct<()>"
    with _ -> "!llvm.struct<()>"

/// Check if a function call is a struct constructor (.ctor)
let private isStructConstructor (funcNode: PSGNode) : bool =
    funcNode.SyntaxKind.Contains(".ctor") ||
    (match funcNode.Symbol with
     | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
         try mfv.CompiledName = ".ctor" || mfv.IsConstructor with _ -> false
     | _ -> false)

/// Check if type is NativeStr
let private isNativeStrType (ftype: FSharpType option) : bool =
    match ftype with
    | Some t ->
        try
            if t.HasTypeDefinition then
                match t.TypeDefinition.TryFullName with
                | Some name -> name.Contains("NativeStr")
                | None -> false
            else false
        with _ -> false
    | None -> false

/// Get struct type info for construction
/// Returns (structType, fieldTypes) where fieldTypes are in order
let private getStructTypeForConstruction (funcNode: PSGNode) : (string * string list) option =
    // Check the result type of the constructor call
    if isNativeStrType funcNode.Type then
        Some ("!llvm.struct<(ptr, i32)>", ["!llvm.ptr"; "i32"])
    else
        None

/// Emit a sequential expression - process children in order, return last result
let rec emitSequential (ctx: EmitContext) (zipper: PSGZipper) : ExprResult =
    let children = PSGZipper.childNodes zipper
    children |> List.fold (fun _ child -> emitNode ctx child) Void

/// Emit a let binding
and emitLetBinding (ctx: EmitContext) (zipper: PSGZipper) : ExprResult =
    let node = zipper.Focus
    let children = PSGZipper.childNodes zipper

    match children with
    | [bindingNode; continuation] when bindingNode.SyntaxKind.StartsWith("Binding") ->
        let isMutable = bindingNode.SyntaxKind.Contains("Mutable")
        let bindingChildren = ChildrenStateHelpers.getChildrenList bindingNode
                              |> List.choose (fun id -> Map.tryFind id.Value ctx.Graph.Nodes)

        match bindingChildren with
        | [_pattern; value] ->
            if isMutable then
                let valueResult = emitNode ctx value
                match valueResult with
                | Value (valueSsa, valueTy) ->
                    let sizeSsa = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = arith.constant 1 : i64" sizeSsa)
                    let allocSsa = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = llvm.alloca %s x %s : (i64) -> !llvm.ptr" allocSsa sizeSsa valueTy)
                    EmitContext.emitLine ctx (sprintf "llvm.store %s, %s : %s, !llvm.ptr" valueSsa allocSsa valueTy)
                    EmitContext.recordNodeSSA ctx bindingNode.Id allocSsa "!llvm.ptr"
                    // Track mutable binding's value type for loads when reading
                    EmitContext.recordMutableBinding ctx bindingNode.Id valueTy
                    emitNode ctx continuation
                | EmitError msg ->
                    let fullMsg = sprintf "Mutable binding %s failed: %s" bindingNode.Id.Value msg
                    EmitContext.recordError ctx fullMsg
                    EmitError fullMsg
                | Void ->
                    let msg = sprintf "Mutable binding %s produced Void - cannot bind" bindingNode.Id.Value
                    EmitContext.recordError ctx msg
                    EmitError msg
            else
                let valueResult = emitNode ctx value
                match valueResult with
                | Value (ssa, ty) ->
                    EmitContext.recordNodeSSA ctx bindingNode.Id ssa ty
                    emitNode ctx continuation
                | EmitError msg ->
                    let fullMsg = sprintf "Binding %s failed: %s" bindingNode.Id.Value msg
                    EmitContext.recordError ctx fullMsg
                    EmitError fullMsg
                | Void ->
                    let msg = sprintf "Binding %s produced Void - cannot bind" bindingNode.Id.Value
                    EmitContext.recordError ctx msg
                    EmitError msg
        | _ ->
            let msg = sprintf "Binding %s has unexpected structure" bindingNode.Id.Value
            EmitContext.recordError ctx msg
            EmitError msg

    | [_pattern; value] when node.SyntaxKind.StartsWith("Binding") ->
        emitNode ctx value

    | [_pattern; value; continuation] ->
        let valueResult = emitNode ctx value
        match valueResult with
        | Value (ssa, ty) ->
            EmitContext.recordNodeSSA ctx node.Id ssa ty
            emitNode ctx continuation
        | EmitError msg ->
            let fullMsg = sprintf "Binding %s failed: %s" node.Id.Value msg
            EmitContext.recordError ctx fullMsg
            EmitError fullMsg
        | Void ->
            let msg = sprintf "Binding %s produced Void - cannot bind" node.Id.Value
            EmitContext.recordError ctx msg
            EmitError msg

    | child :: _ -> emitNode ctx child
    | [] -> Void

/// Emit if-then-else with basic blocks
and emitIfThenElse (ctx: EmitContext) (zipper: PSGZipper) : ExprResult =
    let node = zipper.Focus
    let children = PSGZipper.childNodes zipper
    // Get the result type from the IfThenElse node itself - more reliable than checking branches
    let resultType = getNodeMLIRType node

    match children with
    | [condition; thenBranch; elseBranch] ->
        emitIfThenElseFull ctx resultType condition thenBranch (Some elseBranch)
    | [condition; thenBranch] ->
        emitIfThenElseFull ctx resultType condition thenBranch None
    | _ -> EmitError "Malformed IfThenElse node"

/// Helper to get MLIR type from PSG node's F# type
/// Returns None for Unit/Void types (no value), Some type for value types
and private getNodeMLIRType (node: PSGNode) : string option =
    match node.Type with
    | Some t ->
        try
            if t.HasTypeDefinition then
                let entity = t.TypeDefinition
                // Check for Unit type first (various ways FCS might represent it)
                let isUnit =
                    match entity.TryFullName with
                    | Some n when n.Contains("Unit") || n.Contains("unit") -> true
                    | Some n when n.Contains("Void") || n = "System.Void" -> true
                    | _ -> entity.DisplayName = "unit"
                if isUnit then None
                else
                    match entity.TryFullName with
                    | Some "System.Int32" -> Some "i32"
                    | Some "System.Int64" -> Some "i64"
                    | Some "System.Byte" -> Some "i8"
                    | Some "System.Boolean" -> Some "i1"
                    | Some n when n.Contains("NativeStr") -> Some "!llvm.struct<(ptr, i32)>"
                    | Some n when n.Contains("nativeptr") -> Some "!llvm.ptr"
                    | Some n when n.Contains("Char") -> Some "i32"  // char as i32
                    | _ -> None  // Don't assume i64 for unknown types
            else None
        with _ -> None
    | None -> None

and emitIfThenElseFull (ctx: EmitContext) (resultType: string option) (condition: PSGNode) (thenBranch: PSGNode) (elseBranch: PSGNode option) : ExprResult =
    let condResult = emitNode ctx condition

    match condResult with
    | Value (condSsa, _) ->
        let thenLabel = EmitContext.nextLabel ctx
        let elseLabel = EmitContext.nextLabel ctx
        let mergeLabel = EmitContext.nextLabel ctx

        // Use the IfThenElse node's result type to determine if we need block arguments
        // This is more reliable than checking individual branch node types
        let useBlockArgs = resultType.IsSome && elseBranch.IsSome

        let branchLine =
            match elseBranch with
            | Some _ -> sprintf "llvm.cond_br %s, %s, %s" condSsa thenLabel elseLabel
            | None -> sprintf "llvm.cond_br %s, %s, %s" condSsa thenLabel mergeLabel
        EmitContext.emitLine ctx branchLine

        // Emit then block with terminator
        EmitContext.emitLine ctx (sprintf "%s:" thenLabel)
        let thenResult = emitNode ctx thenBranch

        // Emit then terminator - use the known result type if available
        match thenResult, useBlockArgs, resultType with
        | Value (thenSsa, thenTy), true, _ ->
            EmitContext.emitLine ctx (sprintf "llvm.br %s(%s : %s)" mergeLabel thenSsa thenTy)
        | _ ->
            EmitContext.emitLine ctx (sprintf "llvm.br %s" mergeLabel)

        // Emit else block with terminator (if present)
        let elseResult =
            match elseBranch with
            | Some eb ->
                EmitContext.emitLine ctx (sprintf "%s:" elseLabel)
                let r = emitNode ctx eb
                match r, useBlockArgs, resultType with
                | Value (elseSsa, elseTy), true, _ ->
                    EmitContext.emitLine ctx (sprintf "llvm.br %s(%s : %s)" mergeLabel elseSsa elseTy)
                | _ ->
                    EmitContext.emitLine ctx (sprintf "llvm.br %s" mergeLabel)
                r
            | None -> Void

        // Emit merge block
        match thenResult, elseResult, useBlockArgs, resultType with
        | Value (_, thenTy), Value (_, _), true, _ ->
            let mergeArg = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s(%s: %s):" mergeLabel mergeArg thenTy)
            Value (mergeArg, thenTy)
        | _, _, true, Some ty ->
            // We expected block args but branches didn't produce values - emit diagnostic
            let mergeArg = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s(%s: %s):" mergeLabel mergeArg ty)
            EmitContext.recordError ctx (sprintf "IfThenElse: expected value of type %s but branches produced Void" ty)
            Value (mergeArg, ty)
        | _ ->
            EmitContext.emitLine ctx (sprintf "%s:" mergeLabel)
            Void

    | _ -> EmitError "IfThenElse condition did not produce a value"

/// Emit function application
and emitApp (ctx: EmitContext) (zipper: PSGZipper) : ExprResult =
    let node = zipper.Focus
    let children = PSGZipper.childNodes zipper

    match children with
    | funcNode :: argNodes ->
        // Check SRTP resolution on BOTH the App node AND the funcNode
        // The resolution may be on either depending on the expression structure
        let srtpRes = node.SRTPResolution |> Option.orElse funcNode.SRTPResolution
        match srtpRes with
        | Some (FSMethodByName methodFullName) ->
            emitInlinedCallByName ctx methodFullName argNodes
        | Some (FSMethod (_, mfv, _)) ->
            try
                emitInlinedCallByName ctx mfv.FullName argNodes
            with _ ->
                emitRegularApp ctx node funcNode argNodes
        | Some (MultipleOverloads (traitName, candidates)) ->
            // Match based on argument types for proper overload resolution
            // Get the type of the second argument (first is the trait witness like WritableString)
            let argType =
                match argNodes with
                | _ :: actualArg :: _ -> actualArg.Type
                | [singleArg] -> singleArg.Type
                | [] -> None

            // Find matching candidate - look for one whose parameter type matches our argument
            let findMatchingCandidate () =
                candidates |> List.tryFind (fun c ->
                    // Check if the candidate's method name suggests the right type
                    // For op_Dollar overloads: one is for NativeStr, one for string
                    let methodName = c.TargetMethodFullName
                    match argType with
                    | Some t when t.HasTypeDefinition ->
                        let typeName = try Some (t.TypeDefinition.FullName) with _ -> None
                        match typeName with
                        | Some n when n = "System.String" || n = "string" ->
                            // For string args, prefer the overload that calls writeSystemString
                            // This is a bit of a heuristic - look for overload [1] which is typically the string one
                            methodName.EndsWith("[1]") || methodName.Contains("string")
                        | Some n when n.Contains("NativeStr") ->
                            methodName.EndsWith("[0]") || methodName.Contains("NativeStr")
                        | _ -> true  // Unknown type, any candidate
                    | _ -> true  // No type info, any candidate
                )

            match findMatchingCandidate () with
            | Some candidate ->
                emitInlinedCallByName ctx candidate.TargetMethodFullName argNodes
            | None ->
                // Fall back to first candidate
                match candidates with
                | candidate :: _ ->
                    emitInlinedCallByName ctx candidate.TargetMethodFullName argNodes
                | [] ->
                    let msg = sprintf "SRTP MultipleOverloads has no candidates for trait '%s'" traitName
                    EmitContext.recordError ctx msg
                    EmitError msg
        | Some BuiltIn ->
            emitBuiltInOperator ctx node funcNode argNodes
        | Some (Unresolved reason) ->
            emitRegularApp ctx node funcNode argNodes
        | _ -> emitRegularApp ctx node funcNode argNodes
    | [] -> Void

/// Emit regular (non-SRTP) function application
/// appNode is the App node with the result type; funcNode is the function being called
and emitRegularApp (ctx: EmitContext) (appNode: PSGNode) (funcNode: PSGNode) (argNodes: PSGNode list) : ExprResult =
    // Check for struct constructors first (.ctor for NativeStr, etc.)
    // Must be BOTH a constructor AND return a struct type we handle
    if isStructConstructor funcNode && isNativeStrType appNode.Type then
        emitStructConstruction ctx appNode funcNode argNodes
    // Check for NativeInterop intrinsics
    elif isNativeInteropIntrinsic funcNode then
        emitNativeInteropIntrinsic ctx funcNode argNodes
    // Check for extern primitives - use ExternDispatch
    elif isExternPrimitive funcNode then
        emitExternPrimitive ctx funcNode argNodes
    // Check for FSharp.Core operators
    else
        match isFSharpCoreOperator funcNode with
        | Some (mfv, opKind) ->
            emitFSharpCoreOperator ctx mfv opKind argNodes
        | None ->
            emitInlinedCall ctx funcNode argNodes

/// Emit struct construction (e.g., NativeStr(ptr, len))
/// Builds struct using llvm.mlir.undef and llvm.insertvalue
and emitStructConstruction (ctx: EmitContext) (appNode: PSGNode) (funcNode: PSGNode) (argNodes: PSGNode list) : ExprResult =
    // Get the struct type from the App node's result type
    let structTypeOpt =
        if isNativeStrType appNode.Type then
            Some ("!llvm.struct<(ptr, i32)>", ["!llvm.ptr"; "i32"])
        else
            getStructTypeForConstruction funcNode

    match structTypeOpt with
    | Some (structType, fieldTypes) ->
        // Flatten tuple arguments - Tuple nodes contain the actual struct field values
        let flatArgs = argNodes |> List.collect (fun n ->
            if n.SyntaxKind = "Tuple" || n.SyntaxKind.StartsWith("Tuple") then
                ChildrenStateHelpers.getChildrenList n
                |> List.choose (fun id -> Map.tryFind id.Value ctx.Graph.Nodes)
            else [n])

        // Emit arguments
        let argResults = flatArgs |> List.map (emitNode ctx)
        let argSsas = argResults |> List.choose (function Value (s, t) -> Some (s, t) | _ -> None)

        if argSsas.Length <> fieldTypes.Length then
            let msg = sprintf "Struct construction: expected %d args, got %d" fieldTypes.Length argSsas.Length
            EmitContext.recordError ctx msg
            EmitError msg
        else
            // Build struct: undef -> insertvalue[0] -> insertvalue[1] -> ...
            let undefSsa = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = llvm.mlir.undef : %s" undefSsa structType)

            let finalSsa =
                argSsas
                |> List.mapi (fun i (argSsa, argTy) -> (i, argSsa, argTy))
                |> List.fold (fun currentSsa (i, argSsa, _argTy) ->
                    let nextSsa = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = llvm.insertvalue %s, %s[%d] : %s" nextSsa argSsa currentSsa i structType)
                    nextSsa
                ) undefSsa

            Value (finalSsa, structType)

    | None ->
        // Unknown struct type - fall back to inlined call
        emitInlinedCall ctx funcNode argNodes

/// Emit extern primitive using platform bindings
and emitExternPrimitive (ctx: EmitContext) (funcNode: PSGNode) (argNodes: PSGNode list) : ExprResult =
    match tryExtractExternPrimitiveInfo funcNode with
    | Some info when info.Library = "__fidelity" || info.Library = "platform" ->
        emitPlatformBinding ctx info argNodes
    | Some info ->
        let msg = sprintf "Non-platform extern not supported: %s/%s" info.Library info.EntryPoint
        EmitContext.recordError ctx msg
        EmitError msg
    | None ->
        let msg = "Extern primitive call with no entry point information"
        EmitContext.recordError ctx msg
        EmitError msg

/// Emit platform binding call (syscalls for Platform.Bindings functions)
and emitPlatformBinding (ctx: EmitContext) (info: ExternPrimitiveInfo) (argNodes: PSGNode list) : ExprResult =
    let flatArgs = argNodes |> List.collect (fun n ->
        if n.SyntaxKind = "Tuple" then
            ChildrenStateHelpers.getChildrenList n
            |> List.choose (fun id -> Map.tryFind id.Value ctx.Graph.Nodes)
        else [n])
    let argResults = flatArgs |> List.map (emitNode ctx)
    let argSsas = argResults |> List.choose (function Value (s, t) -> Some (s, t) | _ -> None)

    match info.EntryPoint with
    | "writeBytes" ->
        match argSsas with
        | [(fdSsa, _); (bufSsa, _); (countSsa, _)] ->
            let fdExt = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = arith.extsi %s : i32 to i64" fdExt fdSsa)
            let countExt = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = arith.extsi %s : i32 to i64" countExt countSsa)
            let sysNum = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = arith.constant 1 : i64" sysNum)
            let result = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64" result sysNum fdExt bufSsa countExt)
            let truncResult = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = arith.trunci %s : i64 to i32" truncResult result)
            Value (truncResult, "i32")
        | _ ->
            let msg = sprintf "writeBytes: expected 3 args, got %d" (List.length argSsas)
            EmitContext.recordError ctx msg
            EmitError msg

    | "readBytes" ->
        match argSsas with
        | [(fdSsa, _); (bufSsa, _); (countSsa, _)] ->
            let fdExt = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = arith.extsi %s : i32 to i64" fdExt fdSsa)
            let countExt = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = arith.extsi %s : i32 to i64" countExt countSsa)
            let sysNum = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = arith.constant 0 : i64" sysNum)
            let result = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64" result sysNum fdExt bufSsa countExt)
            let truncResult = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = arith.trunci %s : i64 to i32" truncResult result)
            Value (truncResult, "i32")
        | _ ->
            let msg = sprintf "readBytes: expected 3 args, got %d" (List.length argSsas)
            EmitContext.recordError ctx msg
            EmitError msg

    | "strlen" ->
        match argSsas with
        | [(ptrSsa, _)] ->
            let result = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = llvm.inline_asm \"xor %%rcx, %%rcx\\0A1: cmpb $$0, (%%rdi,%%rcx)\\0Aje 2f\\0Ainc %%rcx\\0Ajmp 1b\\0A2:\", \"={rcx},{rdi}\" %s : (!llvm.ptr) -> i64" result ptrSsa)
            let truncResult = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = arith.trunci %s : i64 to i32" truncResult result)
            Value (truncResult, "i32")
        | _ ->
            let msg = sprintf "strlen: expected 1 arg, got %d" (List.length argSsas)
            EmitContext.recordError ctx msg
            EmitError msg

    | _ ->
        let msg = sprintf "Unknown platform binding: %s" info.EntryPoint
        EmitContext.recordError ctx msg
        EmitError msg

/// Emit NativeInterop intrinsic
and emitNativeInteropIntrinsic (ctx: EmitContext) (funcNode: PSGNode) (argNodes: PSGNode list) : ExprResult =
    let intrinsicName = getNativeInteropIntrinsicName funcNode |> Option.defaultValue "unknown"
    let argResults = argNodes |> List.map (emitNode ctx)
    let argSsas = argResults |> List.choose (function Value (s, t) -> Some (s, t) | _ -> None)

    match intrinsicName with
    | "toNativeInt" | "toVoidPtr" | "ofVoidPtr" | "ofNativeInt" ->
        match argSsas with
        | [(ssa, _)] -> Value (ssa, "!llvm.ptr")
        | _ ->
            let ssa = EmitContext.nextSSA ctx
            Value (ssa, "!llvm.ptr")
    | "stackalloc" ->
        match argSsas with
        | [(sizeSsa, sizeTy)] ->
            // Extend size to i64 if needed (llvm.alloca expects i64)
            let size64 =
                if sizeTy = "i64" then sizeSsa
                else
                    let extSsa = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = arith.extsi %s : %s to i64" extSsa sizeSsa sizeTy)
                    extSsa
            let allocSsa = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = llvm.alloca %s x i8 : (i64) -> !llvm.ptr" allocSsa size64)
            Value (allocSsa, "!llvm.ptr")
        | _ -> Void
    | "get" ->
        match argSsas with
        | [(ptrSsa, _); (idxSsa, _)] ->
            let gepSsa = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i32) -> !llvm.ptr, i8" gepSsa ptrSsa idxSsa)
            let loadSsa = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = llvm.load %s : !llvm.ptr -> i8" loadSsa gepSsa)
            Value (loadSsa, "i8")
        | _ ->
            let msg = sprintf "NativePtr.get: expected 2 args, got %d" (List.length argSsas)
            EmitContext.recordError ctx msg
            EmitError msg
    | "set" ->
        match argSsas with
        | [(ptrSsa, _); (idxSsa, _); (valSsa, valTy)] ->
            let gepSsa = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i32) -> !llvm.ptr, i8" gepSsa ptrSsa idxSsa)
            EmitContext.emitLine ctx (sprintf "llvm.store %s, %s : %s, !llvm.ptr" valSsa gepSsa valTy)
            Void
        | _ ->
            let msg = sprintf "NativePtr.set: expected 3 args, got %d" (List.length argSsas)
            EmitContext.recordError ctx msg
            EmitError msg
    | _ ->
        let msg = sprintf "Unknown NativeInterop intrinsic: %s" intrinsicName
        EmitContext.recordError ctx msg
        EmitError msg

/// Emit FSharp.Core operator
and emitFSharpCoreOperator (ctx: EmitContext) (mfv: FSharpMemberOrFunctionOrValue) (opKind: string) (argNodes: PSGNode list) : ExprResult =
    let flatArgs = argNodes |> List.collect (fun n ->
        if n.SyntaxKind = "Tuple" then
            ChildrenStateHelpers.getChildrenList n
            |> List.choose (fun id -> Map.tryFind id.Value ctx.Graph.Nodes)
        else [n])
    let argResults = flatArgs |> List.map (emitNode ctx)
    let argSsas = argResults |> List.choose (function Value (s, t) -> Some (s, t) | _ -> None)

    // Check for failed argument emissions (for debugging arity issues)
    if List.length argSsas < List.length flatArgs then
        let failedArgs =
            List.zip flatArgs argResults
            |> List.choose (fun (node, result) ->
                match result with
                | Value _ -> None
                | EmitError msg -> Some (sprintf "%s: EmitError(%s)" node.SyntaxKind msg)
                | Void -> Some (sprintf "%s: Void" node.SyntaxKind))
        if not (List.isEmpty failedArgs) then
            let failInfo = String.concat "; " failedArgs
            EmitContext.recordError ctx (sprintf "Argument emission failures for %s: %s" mfv.CompiledName failInfo)


    match opKind, argSsas with
    | "arith", [(left, leftTy); (right, _)] ->
        match arithmeticOp mfv with
        | Some mlirOp ->
            let ssa = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = %s %s, %s : %s" ssa mlirOp left right leftTy)
            Value (ssa, leftTy)
        | None ->
            let msg = sprintf "Unknown arithmetic operator: %s" mfv.CompiledName
            EmitContext.recordError ctx msg
            EmitError msg

    | "bitwise", [(left, leftTy); (right, _)] ->
        match bitwiseOp mfv with
        | Some mlirOp ->
            let ssa = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = %s %s, %s : %s" ssa mlirOp left right leftTy)
            Value (ssa, leftTy)
        | None ->
            let msg = sprintf "Unknown bitwise operator: %s" mfv.CompiledName
            EmitContext.recordError ctx msg
            EmitError msg

    | "compare", [(left, leftTy); (right, _)] ->
        match comparisonOp mfv with
        | Some predicate ->
            let ssa = EmitContext.nextSSA ctx
            let line =
                if leftTy = "!llvm.ptr" then
                    sprintf "%s = llvm.icmp \"%s\" %s, %s : !llvm.ptr" ssa predicate left right
                else
                    sprintf "%s = arith.cmpi \"%s\", %s, %s : %s" ssa predicate left right leftTy
            EmitContext.emitLine ctx line
            Value (ssa, "i1")
        | None ->
            let msg = sprintf "Unknown comparison operator: %s" mfv.CompiledName
            EmitContext.recordError ctx msg
            EmitError msg

    | "conversion", [(input, inputTy)] ->
        match conversionOp mfv with
        | Some targetTy ->
            // Determine conversion direction
            let inputBits = match inputTy with "i8" -> 8 | "i16" -> 16 | "i32" -> 32 | "i64" -> 64 | _ -> 32
            let targetBits = match targetTy with "i8" -> 8 | "i16" -> 16 | "i32" -> 32 | "i64" -> 64 | _ -> 32
            let ssa = EmitContext.nextSSA ctx
            if inputBits < targetBits then
                // Sign extend or zero extend (using sign extend for F# semantics)
                EmitContext.emitLine ctx (sprintf "%s = arith.extsi %s : %s to %s" ssa input inputTy targetTy)
            elif inputBits > targetBits then
                // Truncate
                EmitContext.emitLine ctx (sprintf "%s = arith.trunci %s : %s to %s" ssa input inputTy targetTy)
            else
                // Same size, just use the value (might be reinterpreting signed/unsigned)
                EmitContext.emitLine ctx (sprintf "%s = arith.bitcast %s : %s to %s" ssa input inputTy targetTy)
            Value (ssa, targetTy)
        | None ->
            let msg = sprintf "Unknown conversion operator: %s" mfv.CompiledName
            EmitContext.recordError ctx msg
            EmitError msg

    | _ ->
        let msg = sprintf "FSharp.Core operator with wrong arity: %s has %d args" mfv.CompiledName argSsas.Length
        EmitContext.recordError ctx msg
        EmitError msg

/// Emit built-in operator
and emitBuiltInOperator (ctx: EmitContext) (appNode: PSGNode) (funcNode: PSGNode) (argNodes: PSGNode list) : ExprResult =
    emitRegularApp ctx appNode funcNode argNodes

/// Emit FSharpExpr body from Baker's MemberBodies
/// Used when SRTP-resolved functions aren't in PSG as Binding nodes (e.g., static type members)
/// NOTE: For inline operators, FCS may have already inlined the body, so we may encounter complex expressions.
/// In that case, we try to find the first meaningful Call and emit from the PSG target.
and emitFSharpExprBody (ctx: EmitContext) (expr: FSharpExpr) (argSsas: (string * string) list) : ExprResult =
    // Try to find a Call expression anywhere in the tree that targets a PSG function
    let rec tryFindCallTarget (e: FSharpExpr) : string option =
        match e with
        | Call(_, memberOrFunc, _, _, _) ->
            try Some memberOrFunc.FullName with _ -> None
        | Application(funcExpr, _, _) ->
            tryFindCallTarget funcExpr
        | Let((_, _, _), bodyExpr) ->
            tryFindCallTarget bodyExpr
        | IfThenElse(_, thenExpr, elseExpr) ->
            match tryFindCallTarget thenExpr with
            | Some t -> Some t
            | None -> tryFindCallTarget elseExpr
        | Sequential(e1, e2) ->
            match tryFindCallTarget e1 with
            | Some t -> Some t
            | None -> tryFindCallTarget e2
        | _ -> None

    match expr with
    | Application(funcExpr, _typeArgs, _argExprs) ->
        // Application: func arg - common for operators
        // The funcExpr is the function being applied, argExprs are the arguments
        // Try to get the function's full name and recursively inline
        match funcExpr with
        | Call(_, memberOrFunc, _, _, _) ->
            try
                let targetFullName = memberOrFunc.FullName
                // Recursively inline the target function
                match EmitContext.lookupMemberBody ctx targetFullName with
                | Some (_mfv, innerBody) ->
                    emitFSharpExprBody ctx innerBody argSsas
                | None ->
                    match EmitContext.lookupBindingNode ctx targetFullName with
                    | Some binding ->
                        let bindingChildren = ChildrenStateHelpers.getChildrenList binding
                                              |> List.choose (fun id -> Map.tryFind id.Value ctx.Graph.Nodes)
                        match bindingChildren with
                        | [pattern; body] ->
                            let paramNodes = findParameterNodes ctx.Graph pattern
                            if paramNodes.Length = argSsas.Length then
                                List.zip paramNodes argSsas
                                |> List.iter (fun (paramNode, (ssa, ty)) ->
                                    EmitContext.recordNodeSSA ctx paramNode.Id ssa ty)
                            emitNode ctx body
                        | _ -> EmitError (sprintf "Malformed Application target: %s" targetFullName)
                    | None -> EmitError (sprintf "Application target not found: %s" targetFullName)
            with ex -> EmitError (sprintf "Error processing Application: %s" ex.Message)
        | FSharpExprPatterns.Value(v) ->
            // Application of a value (might be a parameter or local function)
            try
                let targetFullName = v.FullName
                match EmitContext.lookupBindingNode ctx targetFullName with
                | Some binding ->
                    let bindingChildren = ChildrenStateHelpers.getChildrenList binding
                                          |> List.choose (fun id -> Map.tryFind id.Value ctx.Graph.Nodes)
                    match bindingChildren with
                    | [pattern; body] ->
                        let paramNodes = findParameterNodes ctx.Graph pattern
                        if paramNodes.Length = argSsas.Length then
                            List.zip paramNodes argSsas
                            |> List.iter (fun (paramNode, (ssa, ty)) ->
                                EmitContext.recordNodeSSA ctx paramNode.Id ssa ty)
                        emitNode ctx body
                    | _ -> EmitError (sprintf "Malformed Application value target: %s" targetFullName)
                | None -> EmitError (sprintf "Application value target not found: %s" targetFullName)
            with ex -> EmitError (sprintf "Error processing Application value: %s" ex.Message)
        | _ ->
            EmitError "Application with unsupported function expression"
    | Call(_objExprOpt, memberOrFunc, _typeArgs1, _typeArgs2, _argExprs) ->
        // The body is a call to another function - recursively inline that
        try
            let targetFullName = memberOrFunc.FullName
            // IMPORTANT: Check PSG FIRST - prefer emitting from PSG over FSharpExpr
            // PSG has proper structure for parameter binding; FSharpExpr may have complex inlined bodies
            match EmitContext.lookupBindingNode ctx targetFullName with
            | Some binding ->
                let bindingChildren = ChildrenStateHelpers.getChildrenList binding
                                      |> List.choose (fun id -> Map.tryFind id.Value ctx.Graph.Nodes)
                match bindingChildren with
                | [pattern; body] ->
                    let paramNodes = findParameterNodes ctx.Graph pattern
                    if paramNodes.Length = argSsas.Length then
                        List.zip paramNodes argSsas
                        |> List.iter (fun (paramNode, (ssa, ty)) ->
                            EmitContext.recordNodeSSA ctx paramNode.Id ssa ty)
                    emitNode ctx body
                | _ ->
                    let msg = sprintf "Malformed binding for FSharpExpr target: %s" targetFullName
                    EmitContext.recordError ctx msg
                    EmitError msg
            | None ->
                // PSG binding not found - try Baker's MemberBodies as fallback
                match EmitContext.lookupMemberBody ctx targetFullName with
                | Some (_mfv, innerBody) ->
                    // Recursively emit the inner function's body
                    emitFSharpExprBody ctx innerBody argSsas
                | None ->
                    let msg = sprintf "FSharpExpr target not found in PSG or Baker: %s" targetFullName
                    EmitContext.recordError ctx msg
                    EmitError msg
        with ex ->
            let msg = sprintf "Error processing FSharpExpr Call: %s" ex.Message
            EmitContext.recordError ctx msg
            EmitError msg
    | Let((_bindingVar, _bindingExpr, _debugPoint), bodyExpr) ->
        // Let expression - for inline operators, the body might be inlined here
        // Try to find the first Call target and emit from PSG
        match tryFindCallTarget bodyExpr with
        | Some targetFullName ->
            // Found a call - try to emit from PSG
            match EmitContext.lookupBindingNode ctx targetFullName with
            | Some binding ->
                let bindingChildren = ChildrenStateHelpers.getChildrenList binding
                                      |> List.choose (fun id -> Map.tryFind id.Value ctx.Graph.Nodes)
                match bindingChildren with
                | [pattern; body] ->
                    let paramNodes = findParameterNodes ctx.Graph pattern
                    if paramNodes.Length = argSsas.Length then
                        List.zip paramNodes argSsas
                        |> List.iter (fun (paramNode, (ssa, ty)) ->
                            EmitContext.recordNodeSSA ctx paramNode.Id ssa ty)
                    emitNode ctx body
                | _ -> EmitError (sprintf "Malformed Let call target: %s" targetFullName)
            | None ->
                // Try MemberBodies
                match EmitContext.lookupMemberBody ctx targetFullName with
                | Some (_mfv, innerBody) ->
                    emitFSharpExprBody ctx innerBody argSsas
                | None -> EmitError (sprintf "Let call target not found: %s" targetFullName)
        | None ->
            // No call target found - try emitting the body directly
            emitFSharpExprBody ctx bodyExpr argSsas
    | IfThenElse(condExpr, thenExpr, elseExpr) ->
        // For conditional expressions, try to find the first call in either branch
        match tryFindCallTarget thenExpr with
        | Some targetFullName ->
            match EmitContext.lookupBindingNode ctx targetFullName with
            | Some binding ->
                let bindingChildren = ChildrenStateHelpers.getChildrenList binding
                                      |> List.choose (fun id -> Map.tryFind id.Value ctx.Graph.Nodes)
                match bindingChildren with
                | [pattern; body] ->
                    let paramNodes = findParameterNodes ctx.Graph pattern
                    if paramNodes.Length = argSsas.Length then
                        List.zip paramNodes argSsas
                        |> List.iter (fun (paramNode, (ssa, ty)) ->
                            EmitContext.recordNodeSSA ctx paramNode.Id ssa ty)
                    emitNode ctx body
                | _ -> EmitError (sprintf "Malformed IfThenElse target: %s" targetFullName)
            | None -> EmitError (sprintf "IfThenElse target not found: %s" targetFullName)
        | None -> EmitError "IfThenElse with no call target found"
    | Sequential(_e1, e2) ->
        // Sequential expressions - try the second expression
        emitFSharpExprBody ctx e2 argSsas
    | FSharpExprPatterns.Value(v) when v.IsMemberThisValue || v.IsConstructorThisValue ->
        // 'this' reference - skip it for static member calls
        Void
    | FSharpExprPatterns.Value(_v) ->
        // A parameter reference - look up from argSsas by position
        // For static member ops, we pass args in order
        if argSsas.Length > 0 then
            let (ssa, ty) = argSsas.[0]
            Value (ssa, ty)
        else Void
    | other ->
        // Try to get more info about the expression
        let exprString =
            try sprintf "%A" other
            with _ -> "unable to stringify"
        let subExprCount =
            try other.ImmediateSubExpressions.Length
            with _ -> -1
        let msg = sprintf "Unsupported FSharpExpr pattern in SRTP body. SubExprs=%d, Expr=%s" subExprCount (exprString.Substring(0, min 200 exprString.Length))
        EmitContext.recordError ctx msg
        EmitError msg

/// Emit inlined function call by name
and emitInlinedCallByName (ctx: EmitContext) (methodFullName: string) (argNodes: PSGNode list) : ExprResult =
    let flatArgs = argNodes |> List.collect (fun n ->
        if n.SyntaxKind = "Tuple" then
            ChildrenStateHelpers.getChildrenList n
            |> List.choose (fun id -> Map.tryFind id.Value ctx.Graph.Nodes)
        else [n])
    let argResults = flatArgs |> List.map (emitNode ctx)
    let argSsas = argResults |> List.choose (function Value (s, t) -> Some (s, t) | _ -> None)

    // O(1) lookup via lazy-built binding index (replaces O(N) linear search)
    let funcBinding = EmitContext.lookupBindingNode ctx methodFullName

    match funcBinding with
    | Some binding ->
        let bindingChildren = ChildrenStateHelpers.getChildrenList binding
                              |> List.choose (fun id -> Map.tryFind id.Value ctx.Graph.Nodes)
        match bindingChildren with
        | [pattern; body] ->
            let paramNodes = findParameterNodes ctx.Graph pattern
            if paramNodes.Length = argSsas.Length then
                List.zip paramNodes argSsas
                |> List.iter (fun (paramNode, (ssa, ty)) ->
                    EmitContext.recordNodeSSA ctx paramNode.Id ssa ty)
            // Emit the function body
            emitNode ctx body
        | _ ->
            let msg = sprintf "Malformed SRTP function binding for: %s" methodFullName
            EmitContext.recordError ctx msg
            EmitError msg
    | None ->
        // PSG binding not found - try Baker's MemberBodies (for static type members like op_Dollar)
        match EmitContext.lookupMemberBody ctx methodFullName with
        | Some (_mfv, body) ->
            // Found in Baker - emit from FSharpExpr
            emitFSharpExprBody ctx body argSsas
        | None ->
            let msg = sprintf "SRTP function not found in PSG or Baker: %s - cannot inline" methodFullName
            EmitContext.recordError ctx msg
            EmitError msg

/// Emit inlined function call
and emitInlinedCall (ctx: EmitContext) (funcNode: PSGNode) (argNodes: PSGNode list) : ExprResult =
    let flatArgs = argNodes |> List.collect (fun n ->
        if n.SyntaxKind = "Tuple" then
            ChildrenStateHelpers.getChildrenList n
            |> List.choose (fun id -> Map.tryFind id.Value ctx.Graph.Nodes)
        else [n])
    let argResults = flatArgs |> List.map (emitNode ctx)
    let argSsas = argResults |> List.choose (function Value (s, t) -> Some (s, t) | _ -> None)

    let funcName =
        match funcNode.Symbol with
        | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
            try Some mfv.FullName with _ -> None
        | _ -> None

    match funcName with
    | Some name ->
        // O(1) lookup via lazy-built binding index (replaces O(N) linear search)
        let funcBinding = EmitContext.lookupBindingNode ctx name

        match funcBinding with
        | Some binding ->
            let bindingChildren = ChildrenStateHelpers.getChildrenList binding
                                  |> List.choose (fun id -> Map.tryFind id.Value ctx.Graph.Nodes)
            match bindingChildren with
            | [pattern; body] ->
                let paramNodes = findParameterNodes ctx.Graph pattern
                if paramNodes.Length = argSsas.Length then
                    List.zip paramNodes argSsas
                    |> List.iter (fun (paramNode, (ssa, ty)) ->
                        EmitContext.recordNodeSSA ctx paramNode.Id ssa ty)
                // Emit the function body
                emitNode ctx body
            | _ ->
                let msg = sprintf "Malformed function binding for: %s" name
                EmitContext.recordError ctx msg
                EmitError msg
        | None ->
            let msg = sprintf "Function not found in PSG: %s - cannot generate code" name
            EmitContext.recordError ctx msg
            EmitError msg
    | None ->
        // Debug: show what node we're trying to emit so we can diagnose SRTP issues
        let debugInfo = sprintf "SyntaxKind=%s, SRTPResolution=%A" funcNode.SyntaxKind funcNode.SRTPResolution
        let msg = sprintf "Function call with no symbol - cannot determine target (%s)" debugInfo
        EmitContext.recordError ctx msg
        EmitError msg

/// Find parameter nodes from a pattern
and findParameterNodes (graph: ProgramSemanticGraph) (pattern: PSGNode) : PSGNode list =
    if pattern.SyntaxKind.StartsWith("Pattern:Named:") then
        [pattern]
    elif pattern.SyntaxKind.StartsWith("Pattern:LongIdent:") then
        ChildrenStateHelpers.getChildrenList pattern
        |> List.choose (fun id -> Map.tryFind id.Value graph.Nodes)
        |> List.collect (findParameterNodes graph)
    elif pattern.SyntaxKind = "Pattern:Paren" || pattern.SyntaxKind = "Pattern:Tuple" then
        ChildrenStateHelpers.getChildrenList pattern
        |> List.choose (fun id -> Map.tryFind id.Value graph.Nodes)
        |> List.collect (findParameterNodes graph)
    else
        ChildrenStateHelpers.getChildrenList pattern
        |> List.choose (fun id -> Map.tryFind id.Value graph.Nodes)
        |> List.collect (findParameterNodes graph)

/// Emit short-circuit boolean operator (&&) as a value-producing expression
/// Returns the resulting boolean SSA value
and emitShortCircuitAnd (ctx: EmitContext) (condition: PSGNode) : ExprResult =
    // For short-circuit &&, we need to evaluate:
    // if firstCond then secondCond else false
    let children = ChildrenStateHelpers.getChildrenList condition
                   |> List.choose (fun id -> Map.tryFind id.Value ctx.Graph.Nodes)
    match children with
    | [firstCond; secondCond; _falseVal] ->
        // Allocate a result variable on the stack
        let resultPtr = EmitContext.nextSSA ctx
        EmitContext.emitLine ctx (sprintf "%s = arith.constant 1 : i64" (EmitContext.nextSSA ctx))
        let sizeSsa = sprintf "%%v%d" (ctx.SSACounter - 1)
        EmitContext.emitLine ctx (sprintf "%s = llvm.alloca %s x i1 : (i64) -> !llvm.ptr" resultPtr sizeSsa)

        // Default to false
        let falseSsa = EmitContext.nextSSA ctx
        EmitContext.emitLine ctx (sprintf "%s = arith.constant 0 : i1" falseSsa)
        EmitContext.emitLine ctx (sprintf "llvm.store %s, %s : i1, !llvm.ptr" falseSsa resultPtr)

        // Evaluate first condition
        let firstResult = emitNode ctx firstCond
        match firstResult with
        | Value (firstSsa, _) ->
            let checkSecondBlock = EmitContext.nextLabel ctx
            let doneBlock = EmitContext.nextLabel ctx

            // If first is false, skip to done (result is already false)
            EmitContext.emitLine ctx (sprintf "llvm.cond_br %s, %s, %s" firstSsa checkSecondBlock doneBlock)

            // Check second block
            EmitContext.emitLine ctx (sprintf "%s:" checkSecondBlock)
            let secondResult = emitNode ctx secondCond
            match secondResult with
            | Value (secondSsa, _) ->
                // Store second result (which is the final result)
                EmitContext.emitLine ctx (sprintf "llvm.store %s, %s : i1, !llvm.ptr" secondSsa resultPtr)
            | _ -> ()
            EmitContext.emitLine ctx (sprintf "llvm.br %s" doneBlock)

            // Done block - load and return result
            EmitContext.emitLine ctx (sprintf "%s:" doneBlock)
            let loadedResult = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = llvm.load %s : !llvm.ptr -> i1" loadedResult resultPtr)
            Value (loadedResult, "i1")
        | _ ->
            EmitError "First condition of && did not produce a value"
    | _ ->
        EmitError "Malformed BoolOp(&&) structure"

/// Emit while loop
/// WhileLoop has 2 children: condition and body
and emitWhileLoop (ctx: EmitContext) (zipper: PSGZipper) : ExprResult =
    let children = PSGZipper.childNodes zipper

    match children with
    | [condition; body] ->
        // Generate unique block labels
        let condBlock = EmitContext.nextLabel ctx
        let bodyBlock = EmitContext.nextLabel ctx
        let exitBlock = EmitContext.nextLabel ctx

        // Jump to condition block
        EmitContext.emitLine ctx (sprintf "llvm.br %s" condBlock)

        // Condition block
        EmitContext.emitLine ctx (sprintf "%s:" condBlock)

        // Handle short-circuit && specially
        let condResult =
            if condition.SyntaxKind.StartsWith("IfThenElse:BoolOp") then
                emitShortCircuitAnd ctx condition
            else
                emitNode ctx condition

        let condSsa =
            match condResult with
            | Value (ssa, _) -> ssa
            | _ -> "%cond_error"

        // Conditional branch
        EmitContext.emitLine ctx (sprintf "llvm.cond_br %s, %s, %s" condSsa bodyBlock exitBlock)

        // Body block
        EmitContext.emitLine ctx (sprintf "%s:" bodyBlock)
        emitNode ctx body |> ignore
        EmitContext.emitLine ctx (sprintf "llvm.br %s" condBlock)

        // Exit block
        EmitContext.emitLine ctx (sprintf "%s:" exitBlock)
        Void

    | _ ->
        let msg = sprintf "WhileLoop expected 2 children (condition, body), got %d" (List.length children)
        EmitContext.recordError ctx msg
        EmitError msg

/// Emit mutable assignment (MutableSet:varname)
/// Finds the mutable binding pointer and stores the new value
and emitMutableSet (ctx: EmitContext) (zipper: PSGZipper) : ExprResult =
    let node = zipper.Focus
    let children = PSGZipper.childNodes zipper

    match children with
    | [newValueNode] ->
        // Find the mutable variable's definition via def-use edges
        match findDefiningNode ctx.Graph node with
        | Some defNode ->
            // Get the pointer from the mutable binding
            match EmitContext.lookupMutableBinding ctx defNode.Id with
            | Some valueTy ->
                match EmitContext.lookupNodeSSA ctx defNode.Id with
                | Some (ptrSsa, _) ->
                    // Emit the new value
                    let valueResult = emitNode ctx newValueNode
                    match valueResult with
                    | Value (valueSsa, _) ->
                        // Store the new value
                        EmitContext.emitLine ctx (sprintf "llvm.store %s, %s : %s, !llvm.ptr" valueSsa ptrSsa valueTy)
                        Void  // Assignment doesn't return a value
                    | EmitError msg -> EmitError (sprintf "MutableSet value failed: %s" msg)
                    | Void ->
                        EmitContext.recordError ctx (sprintf "MutableSet for %s: new value produced Void" node.SyntaxKind)
                        Void
                | None ->
                    let msg = sprintf "MutableSet: no SSA for mutable binding %s" defNode.Id.Value
                    EmitContext.recordError ctx msg
                    EmitError msg
            | None ->
                // Not a mutable binding - this might be an error
                let msg = sprintf "MutableSet: target %s is not a mutable binding" defNode.Id.Value
                EmitContext.recordError ctx msg
                EmitError msg
        | None ->
            // Try to find via SymbolUse edges directly
            let symbolUseEdge =
                ctx.Graph.Edges
                |> List.tryFind (fun e -> e.Source = node.Id && e.Kind = SymbolUse)

            match symbolUseEdge with
            | Some edge ->
                match Map.tryFind edge.Target.Value ctx.Graph.Nodes with
                | Some defNode ->
                    match EmitContext.lookupMutableBinding ctx defNode.Id with
                    | Some valueTy ->
                        match EmitContext.lookupNodeSSA ctx defNode.Id with
                        | Some (ptrSsa, _) ->
                            let valueResult = emitNode ctx newValueNode
                            match valueResult with
                            | Value (valueSsa, _) ->
                                EmitContext.emitLine ctx (sprintf "llvm.store %s, %s : %s, !llvm.ptr" valueSsa ptrSsa valueTy)
                                Void
                            | EmitError msg -> EmitError (sprintf "MutableSet value failed: %s" msg)
                            | Void -> Void
                        | None ->
                            let msg = sprintf "MutableSet: no SSA for mutable binding %s" defNode.Id.Value
                            EmitContext.recordError ctx msg
                            EmitError msg
                    | None ->
                        let msg = sprintf "MutableSet: target via edge %s is not mutable" defNode.Id.Value
                        EmitContext.recordError ctx msg
                        EmitError msg
                | None ->
                    let msg = "MutableSet: edge target node not found"
                    EmitContext.recordError ctx msg
                    EmitError msg
            | None ->
                let msg = sprintf "MutableSet: no def-use edge from %s" node.Id.Value
                EmitContext.recordError ctx msg
                EmitError msg
    | _ ->
        let msg = sprintf "MutableSet expected 1 child (new value), got %d" (List.length children)
        EmitContext.recordError ctx msg
        EmitError msg

/// Emit property access
and emitPropertyAccess (ctx: EmitContext) (zipper: PSGZipper) : ExprResult =
    let node = zipper.Focus
    let kind = node.SyntaxKind
    let propName = if kind.StartsWith("PropertyAccess:") then kind.Substring(15) else ""
    let children = PSGZipper.childNodes zipper

    // First check if we have field access info from Baker correlations (O(1) range-indexed lookup)
    match EmitContext.lookupFieldAccess ctx node with
    | Some fieldInfo ->
        // We have typed tree correlation - use it to emit extractvalue
        match children with
        | [target] ->
            let targetResult = emitNode ctx target
            match targetResult with
            | Value (targetSsa, _) ->
                let containerType = mapContainingTypeToMLIR fieldInfo.ContainingType
                let resultType = mapFSharpTypeToMLIR fieldInfo.Field.FieldType
                let ssa = EmitContext.nextSSA ctx
                EmitContext.emitLine ctx (sprintf "%s = llvm.extractvalue %s[%d] : %s" ssa targetSsa fieldInfo.FieldIndex containerType)
                Value (ssa, resultType)
            | EmitError msg -> EmitError msg
            | Void ->
                let msg = sprintf "PropertyAccess:%s target emitted Void" propName
                EmitContext.recordError ctx msg
                EmitError msg
        | _ ->
            let msg = sprintf "PropertyAccess:%s with field info but wrong number of children (%d)" propName (List.length children)
            EmitContext.recordError ctx msg
            EmitError msg

    | None ->
        // No correlation info - fall back to type-based inference
        match node.ConstantValue, propName with
        | Some (Int32Value length), "Length" ->
            let ssa = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = arith.constant %d : i32" ssa length)
            Value (ssa, "i32")
        | _ ->
            match children, propName with
            | [target], "Length" ->
                // Check if target is a NativeStr (fat pointer struct)
                if isNativeStrType target.Type then
                    // NativeStr is struct { Pointer: ptr, Length: i32 }
                    // Length is field index 1
                    let targetResult = emitNode ctx target
                    match targetResult with
                    | Value (targetSsa, _) ->
                        let ssa = EmitContext.nextSSA ctx
                        EmitContext.emitLine ctx (sprintf "%s = llvm.extractvalue %s[1] : !llvm.struct<(ptr, i32)>" ssa targetSsa)
                        Value (ssa, "i32")
                    | EmitError msg -> EmitError msg
                    | Void -> EmitError "PropertyAccess:Length on NativeStr - target emitted Void"
                else
                    // Check for constant string
                    match target.ConstantValue with
                    | Some (StringValue s) ->
                        let ssa = EmitContext.nextSSA ctx
                        EmitContext.emitLine ctx (sprintf "%s = arith.constant %d : i32" ssa s.Length)
                        Value (ssa, "i32")
                    | _ ->
                        let msg = "PropertyAccess:Length - could not determine target type for extraction"
                        EmitContext.recordError ctx msg
                        EmitError msg
            | [target], "Pointer" ->
                // Check if target is a NativeStr
                if isNativeStrType target.Type then
                    // NativeStr Pointer is field index 0
                    let targetResult = emitNode ctx target
                    match targetResult with
                    | Value (targetSsa, _) ->
                        let ssa = EmitContext.nextSSA ctx
                        EmitContext.emitLine ctx (sprintf "%s = llvm.extractvalue %s[0] : !llvm.struct<(ptr, i32)>" ssa targetSsa)
                        Value (ssa, "!llvm.ptr")
                    | EmitError msg -> EmitError msg
                    | Void -> EmitError "PropertyAccess:Pointer on NativeStr - target emitted Void"
                else
                    let msg = sprintf "PropertyAccess:Pointer - unknown target type"
                    EmitContext.recordError ctx msg
                    EmitError msg
            | _ ->
                let msg = sprintf "Unknown property access: %s - cannot generate code" propName
                EmitContext.recordError ctx msg
                EmitError msg

/// Emit semantic primitive
and emitSemanticPrimitive (ctx: EmitContext) (zipper: PSGZipper) : ExprResult =
    let node = zipper.Focus
    let kind = node.SyntaxKind
    let primName = if kind.StartsWith("SemanticPrimitive:") then kind.Substring(18) else ""
    let children = PSGZipper.childNodes zipper

    match primName, children with
    | "fidelity_strlen", [target] ->
        let targetResult = emitNode ctx target
        match targetResult with
        | Value (targetSsa, "!llvm.ptr") ->
            let result = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = llvm.inline_asm \"xor %%rcx, %%rcx\\0A1: cmpb $$0, (%%rdi,%%rcx)\\0Aje 2f\\0Ainc %%rcx\\0Ajmp 1b\\0A2:\", \"={rcx},{rdi}\" %s : (!llvm.ptr) -> i64" result targetSsa)
            let truncResult = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = arith.trunci %s : i64 to i32" truncResult result)
            Value (truncResult, "i32")
        | _ ->
            let msg = "SemanticPrimitive:fidelity_strlen - target is not !llvm.ptr"
            EmitContext.recordError ctx msg
            EmitError msg
    | _ ->
        let msg = sprintf "Unknown semantic primitive: %s - cannot generate code" primName
        EmitContext.recordError ctx msg
        EmitError msg

/// Emit string concatenation (lowered interpolated string)
/// For now, this emits each part sequentially - a tactical solution.
/// A proper implementation would concat into a buffer and return NativeStr.
and emitStringConcat (ctx: EmitContext) (zipper: PSGZipper) : ExprResult =
    let children = PSGZipper.childNodes zipper

    // Emit each child part
    // String constants become static string references
    // Fill expressions emit their values (expected to be NativeStr or string)
    let mutable lastResult = Void
    for child in children do
        let result = emitNode ctx child
        match result with
        | Value (ssa, ty) ->
            // For string constants, we get a pointer to static data
            // For NativeStr expressions, we get the struct
            // We need to write each part - for now, just emit the parts
            // and the caller (Console.WriteLine) will handle them
            lastResult <- result
        | Void -> ()
        | EmitError msg ->
            EmitContext.recordError ctx msg

    // Return the last result
    // This is a simplification - proper concat would return a combined NativeStr
    lastResult

/// Emit tuple
and emitTuple (ctx: EmitContext) (zipper: PSGZipper) : ExprResult =
    let children = PSGZipper.childNodes zipper
    match children with
    | [] -> Void
    | _ -> children |> List.fold (fun _ child -> emitNode ctx child) Void

/// Emit address-of
/// For mutable bindings, returns the pointer directly (without loading)
/// For other values, returns the SSA value as a pointer type
and emitAddressOf (ctx: EmitContext) (zipper: PSGZipper) : ExprResult =
    let children = PSGZipper.childNodes zipper
    match children with
    | [child] ->
        // Check if child is a reference to a mutable binding
        // If so, return the pointer to the mutable's storage directly (don't load)
        match findDefiningNode ctx.Graph child with
        | Some defNode ->
            match EmitContext.lookupNodeSSA ctx defNode.Id with
            | Some (ptrSsa, "!llvm.ptr") ->
                // Check if this is a mutable binding
                match EmitContext.lookupMutableBinding ctx defNode.Id with
                | Some _ ->
                    // Mutable binding: return the pointer directly (address of the mutable)
                    Value (ptrSsa, "!llvm.ptr")
                | None ->
                    // Non-mutable: evaluate normally and cast to pointer type
                    let result = emitNode ctx child
                    match result with
                    | Value (ssa, _) -> Value (ssa, "!llvm.ptr")
                    | other -> other
            | Some (ssa, _) ->
                // Non-pointer SSA - just re-type it
                Value (ssa, "!llvm.ptr")
            | None ->
                // Not yet emitted - evaluate and get result
                let result = emitNode ctx child
                match result with
                | Value (ssa, _) -> Value (ssa, "!llvm.ptr")
                | EmitError msg -> EmitError msg
                | Void ->
                    let msg = "AddressOf child produced Void - cannot take address"
                    EmitContext.recordError ctx msg
                    EmitError msg
        | None ->
            // No defining node - evaluate normally
            let result = emitNode ctx child
            match result with
            | Value (ssa, _) -> Value (ssa, "!llvm.ptr")
            | EmitError msg -> EmitError msg
            | Void ->
                let msg = "AddressOf child produced Void - cannot take address"
                EmitContext.recordError ctx msg
                EmitError msg
    | _ ->
        let msg = sprintf "AddressOf expected 1 child, got %d" (List.length children)
        EmitContext.recordError ctx msg
        EmitError msg

/// Emit type application (passthrough)
and emitTypeApp (ctx: EmitContext) (zipper: PSGZipper) : ExprResult =
    let children = PSGZipper.childNodes zipper
    match children with
    | [child] -> emitNode ctx child
    | _ ->
        let msg = sprintf "TypeApp expected 1 child, got %d" (List.length children)
        EmitContext.recordError ctx msg
        EmitError msg

/// Emit dot-indexed get (e.g., s[i] for string/array indexing)
and emitDotIndexedGet (ctx: EmitContext) (zipper: PSGZipper) : ExprResult =
    let children = PSGZipper.childNodes zipper
    match children with
    | [target; index] ->
        let targetResult = emitNode ctx target
        let indexResult = emitNode ctx index
        match targetResult, indexResult with
        | Value (targetSsa, "!llvm.ptr"), Value (indexSsa, indexTy) ->
            // String indexing: compute pointer to character at offset
            // First, extend index to i64 if needed
            let indexSsa64 =
                if indexTy = "i64" then indexSsa
                else
                    let extSsa = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = arith.extsi %s : %s to i64" extSsa indexSsa indexTy)
                    extSsa
            // GEP to get pointer to character at index
            let gepSsa = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i64) -> !llvm.ptr, i8" gepSsa targetSsa indexSsa64)
            // Load the character (i8)
            let charSsa = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = llvm.load %s : !llvm.ptr -> i8" charSsa gepSsa)
            // Return as i32 (char in F# is 16-bit, but we're treating it as byte for now)
            let resultSsa = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = arith.extui %s : i8 to i32" resultSsa charSsa)
            Value (resultSsa, "i32")
        | Value (targetSsa, targetTy), Value (indexSsa, _) ->
            let msg = sprintf "DotIndexedGet: expected !llvm.ptr target, got %s" targetTy
            EmitContext.recordError ctx msg
            EmitError msg
        | EmitError msg, _ -> EmitError msg
        | _, EmitError msg -> EmitError msg
        | _ ->
            let msg = "DotIndexedGet: target or index did not produce a value"
            EmitContext.recordError ctx msg
            EmitError msg
    | _ ->
        let msg = sprintf "DotIndexedGet expected 2 children (target, index), got %d" (List.length children)
        EmitContext.recordError ctx msg
        EmitError msg

// ═══════════════════════════════════════════════════════════════════
// Main Emission - Uses Zipper for context, dispatches by SyntaxKind
// ═══════════════════════════════════════════════════════════════════

/// Emit MLIR for a single node
and emitNode (ctx: EmitContext) (node: PSGNode) : ExprResult =
    // Create a zipper focused on this node for context-aware emission
    let zipper = PSGZipper.create ctx.Graph node
    let kind = node.SyntaxKind

    if kind.StartsWith("Const:") then
        emitConstant ctx node
    elif kind.StartsWith("Ident:") || kind.StartsWith("LongIdent:") then
        emitIdent ctx node
    elif kind = "Sequential" then
        emitSequential ctx zipper
    elif kind.StartsWith("LetOrUse:") || kind.StartsWith("Binding") then
        emitLetBinding ctx zipper
    elif kind = "IfThenElse" || kind.StartsWith("IfThenElse:") then
        emitIfThenElse ctx zipper
    elif kind = "App:StringConcat" then
        emitStringConcat ctx zipper
    elif kind.StartsWith("App:") then
        emitApp ctx zipper
    elif kind = "Tuple" then
        emitTuple ctx zipper
    elif kind = "AddressOf" then
        emitAddressOf ctx zipper
    elif kind.StartsWith("TypeApp:") then
        emitTypeApp ctx zipper
    elif kind.StartsWith("PropertyAccess:") then
        emitPropertyAccess ctx zipper
    elif kind.StartsWith("SemanticPrimitive:") then
        emitSemanticPrimitive ctx zipper
    elif kind = "WhileLoop" then
        emitWhileLoop ctx zipper
    elif kind = "DotIndexedGet" then
        emitDotIndexedGet ctx zipper
    elif kind.StartsWith("MutableSet:") then
        emitMutableSet ctx zipper
    elif kind.StartsWith("Pattern:") then
        let children = PSGZipper.childNodes zipper
        match children with
        | [child] -> emitNode ctx child
        | [] -> Void  // Empty pattern is valid (e.g., unit pattern)
        | _ ->
            // Multi-child patterns should be handled in specific cases
            Void
    else
        let children = PSGZipper.childNodes zipper
        match children with
        | [child] -> emitNode ctx child
        | [] ->
            // Unknown node with no children - might be valid for some constructs
            Void
        | _ ->
            let msg = sprintf "Unknown node kind: %s with %d children - cannot generate code" kind (List.length children)
            EmitContext.recordError ctx msg
            EmitError msg

// ═══════════════════════════════════════════════════════════════════
// Entry Point Generation
// ═══════════════════════════════════════════════════════════════════

/// Generate MLIR for an entry point function
/// In freestanding mode, main IS the entry point (no libc _start),
/// so we must emit exit syscall before returning
let emitEntryPoint (ctx: EmitContext) (entryNode: PSGNode) : unit =
    EmitContext.emitLine ctx "llvm.func @main() -> i32 {"
    EmitContext.emitLine ctx "  ^entry:"

    let children = ChildrenStateHelpers.getChildrenList entryNode
                   |> List.choose (fun id -> Map.tryFind id.Value ctx.Graph.Nodes)
    let body =
        match children with
        | [_pattern; body] -> Some body
        | _ -> None

    match body with
    | Some b -> emitNode ctx b |> ignore
    | None -> ()

    // In freestanding mode, we must call exit syscall (60 on Linux x86_64)
    // before returning, since there's no libc _start to handle the exit
    let exitCodeSsa = EmitContext.nextSSA ctx
    EmitContext.emitLine ctx (sprintf "%s = arith.constant 0 : i64" exitCodeSsa)
    let syscallNumSsa = EmitContext.nextSSA ctx
    EmitContext.emitLine ctx (sprintf "%s = arith.constant 60 : i64" syscallNumSsa)  // exit syscall
    let resultSsa = EmitContext.nextSSA ctx
    EmitContext.emitLine ctx (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"={rax},{rax},{rdi},~{rcx},~{r11},~{memory}\" %s, %s : (i64, i64) -> i64"
        resultSsa syscallNumSsa exitCodeSsa)

    // The exit syscall won't return, but LLVM requires a terminator
    let retVal = EmitContext.nextSSA ctx
    EmitContext.emitLine ctx (sprintf "%s = arith.constant 0 : i32" retVal)
    EmitContext.emitLine ctx (sprintf "llvm.return %s : i32" retVal)
    EmitContext.emitLine ctx "}"

// ═══════════════════════════════════════════════════════════════════
// Module Entry Point
// ═══════════════════════════════════════════════════════════════════

/// Result type for MLIR generation
type MLIRGenerationResult = {
    Content: string
    Errors: string list
    HasErrors: bool
}

/// Generate complete MLIR module from PSG
///
/// This is the single entry point for MLIR generation. It:
/// 1. Registers platform bindings (Console, Time, Process)
/// 2. Sets target platform from triple
/// 3. Traverses PSG and emits MLIR
/// 4. Returns the complete MLIR module with any errors
let generateMLIR (psg: ProgramSemanticGraph) (correlationState: Baker.TypedTreeZipper.CorrelationState) (targetTriple: string) : MLIRGenerationResult =
    // Register all platform bindings
    Alex.Bindings.Time.TimeBindings.registerBindings ()
    Alex.Bindings.Console.ConsoleBindings.registerBindings ()
    Alex.Bindings.Process.ProcessBindings.registerBindings ()

    // Set target platform from triple
    match TargetPlatform.parseTriple targetTriple with
    | Some platform -> ExternDispatch.setTargetPlatform platform
    | None -> ()  // Use default (auto-detect)

    let ctx = EmitContext.create psg correlationState
    let header = sprintf "// Firefly-generated MLIR\n// Target: %s\n\nmodule {\n" targetTriple

    // Find and emit entry points
    let entryPoints = psg.EntryPoints |> List.choose (fun id -> Map.tryFind id.Value psg.Nodes)
    entryPoints |> List.iter (emitEntryPoint ctx)

    let body = EmitContext.getOutput ctx

    // Check for errors BEFORE proceeding
    let errors = EmitContext.getErrors ctx

    // External function declarations
    let externalFuncDecls =
        ctx.ExternalFuncs
        |> Set.toList
        |> List.map (fun name ->
            match name with
            | "strlen" -> "llvm.func @strlen(!llvm.ptr) -> i64"
            | _ -> sprintf "// Unknown external: %s" name)
        |> String.concat "\n"

    // String literals as globals
    let stringLiterals =
        ctx.StringLiterals
        |> List.map (fun (content, name) ->
            let escaped = content.Replace("\\", "\\\\").Replace("\"", "\\\"")
            let len = content.Length + 1
            sprintf "llvm.mlir.global internal constant %s(\"%s\\00\") : !llvm.array<%d x i8>" name escaped len)
        |> String.concat "\n"

    let footer = "\n}\n"
    let content = header + body + "\n" + externalFuncDecls + "\n" + stringLiterals + footer

    {
        Content = content
        Errors = errors
        HasErrors = EmitContext.hasErrors ctx
    }
