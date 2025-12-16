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
open Core.PSG.Types
open Alex.Traversal.PSGZipper
open Alex.Traversal.PSGXParsec
open Alex.Patterns.PSGPatterns
open Alex.CodeGeneration.MLIRBuilder
open Alex.Bindings.BindingTypes
open Core.PSG.Nanopass.DefUseEdges
open Baker.Types

// ═══════════════════════════════════════════════════════════════════
// Type Mapping
// ═══════════════════════════════════════════════════════════════════

/// Map F# type to MLIR type string
let mapType (ftype: FSharpType option) : string =
    match ftype with
    | None -> "i32"
    | Some t ->
        try
            if t.HasTypeDefinition then
                match t.TypeDefinition.TryFullName with
                | Some "System.Int32" -> "i32"
                | Some "System.Int64" -> "i64"
                | Some "System.Int16" -> "i16"
                | Some "System.Byte" | Some "System.SByte" -> "i8"
                | Some "System.UInt32" -> "i32"
                | Some "System.UInt64" -> "i64"
                | Some "System.Boolean" -> "i1"
                | Some "System.Single" -> "f32"
                | Some "System.Double" -> "f64"
                | Some "System.Void" | Some "Microsoft.FSharp.Core.unit" -> "unit"
                | Some "System.IntPtr" | Some "System.UIntPtr" -> "!llvm.ptr"
                | Some n when n.Contains("nativeptr") -> "!llvm.ptr"
                | Some "Microsoft.FSharp.Core.string" -> "!llvm.ptr"
                | _ -> "i32"
            elif t.IsGenericParameter then
                "!llvm.ptr"
            else
                "i32"
        with _ -> "i32"

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
            EmitError (sprintf "Definition %s not yet emitted" defNode.Id.Value)
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
                | _ -> emitNode ctx continuation
            else
                let valueResult = emitNode ctx value
                match valueResult with
                | Value (ssa, ty) ->
                    EmitContext.recordNodeSSA ctx bindingNode.Id ssa ty
                    emitNode ctx continuation
                | _ -> emitNode ctx continuation
        | _ -> emitNode ctx continuation

    | [_pattern; value] when node.SyntaxKind.StartsWith("Binding") ->
        emitNode ctx value

    | [_pattern; value; continuation] ->
        let valueResult = emitNode ctx value
        match valueResult with
        | Value (ssa, ty) ->
            EmitContext.recordNodeSSA ctx node.Id ssa ty
            emitNode ctx continuation
        | _ -> emitNode ctx continuation

    | child :: _ -> emitNode ctx child
    | [] -> Void

/// Emit if-then-else with basic blocks
and emitIfThenElse (ctx: EmitContext) (zipper: PSGZipper) : ExprResult =
    let children = PSGZipper.childNodes zipper

    match children with
    | [condition; thenBranch; elseBranch] ->
        emitIfThenElseFull ctx condition thenBranch (Some elseBranch)
    | [condition; thenBranch] ->
        emitIfThenElseFull ctx condition thenBranch None
    | _ -> EmitError "Malformed IfThenElse node"

and emitIfThenElseFull (ctx: EmitContext) (condition: PSGNode) (thenBranch: PSGNode) (elseBranch: PSGNode option) : ExprResult =
    let condResult = emitNode ctx condition

    match condResult with
    | Value (condSsa, _) ->
        let thenLabel = EmitContext.nextLabel ctx
        let elseLabel = EmitContext.nextLabel ctx
        let mergeLabel = EmitContext.nextLabel ctx

        let branchLine =
            match elseBranch with
            | Some _ -> sprintf "llvm.cond_br %s, %s, %s" condSsa thenLabel elseLabel
            | None -> sprintf "llvm.cond_br %s, %s, %s" condSsa thenLabel mergeLabel
        EmitContext.emitLine ctx branchLine

        EmitContext.emitLine ctx (sprintf "%s:" thenLabel)
        let thenResult = emitNode ctx thenBranch
        EmitContext.emitLine ctx (sprintf "llvm.br %s" mergeLabel)

        let elseResult =
            match elseBranch with
            | Some eb ->
                EmitContext.emitLine ctx (sprintf "%s:" elseLabel)
                let r = emitNode ctx eb
                EmitContext.emitLine ctx (sprintf "llvm.br %s" mergeLabel)
                r
            | None -> Void

        EmitContext.emitLine ctx (sprintf "%s:" mergeLabel)

        match thenResult, elseResult with
        | Value (thenSsa, thenTy), Value (_, elseTy) when thenTy = elseTy ->
            Value (thenSsa, thenTy)
        | _ -> Void
    | _ -> EmitError "IfThenElse condition did not produce a value"

/// Emit function application
and emitApp (ctx: EmitContext) (zipper: PSGZipper) : ExprResult =
    let node = zipper.Focus
    let children = PSGZipper.childNodes zipper

    match children with
    | funcNode :: argNodes ->
        // Check SRTP resolution first
        match node.SRTPResolution with
        | Some (FSMethodByName methodFullName) ->
            emitRegularApp ctx node funcNode argNodes
        | Some (FSMethod (_, mfv, _)) ->
            emitRegularApp ctx node funcNode argNodes
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

/// Emit extern primitive using Bindings dispatch
and emitExternPrimitive (ctx: EmitContext) (funcNode: PSGNode) (argNodes: PSGNode list) : ExprResult =
    match tryExtractExternPrimitiveInfo funcNode with
    | Some info when info.Library = "__fidelity" ->
        emitFidelityExtern ctx info argNodes
    | Some info ->
        let msg = sprintf "Non-fidelity extern not supported: %s/%s" info.Library info.EntryPoint
        EmitContext.recordError ctx msg
        EmitError msg
    | None ->
        let msg = "Extern primitive call with no entry point information"
        EmitContext.recordError ctx msg
        EmitError msg

/// Emit Fidelity extern call (syscalls via Bindings)
and emitFidelityExtern (ctx: EmitContext) (info: ExternPrimitiveInfo) (argNodes: PSGNode list) : ExprResult =
    let flatArgs = argNodes |> List.collect (fun n ->
        if n.SyntaxKind = "Tuple" then
            ChildrenStateHelpers.getChildrenList n
            |> List.choose (fun id -> Map.tryFind id.Value ctx.Graph.Nodes)
        else [n])
    let argResults = flatArgs |> List.map (emitNode ctx)
    let argSsas = argResults |> List.choose (function Value (s, t) -> Some (s, t) | _ -> None)

    match info.EntryPoint with
    | "fidelity_write_bytes" ->
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
            let msg = sprintf "fidelity_write_bytes: expected 3 args, got %d" (List.length argSsas)
            EmitContext.recordError ctx msg
            EmitError msg

    | "fidelity_read_bytes" ->
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
            let msg = sprintf "fidelity_read_bytes: expected 3 args, got %d" (List.length argSsas)
            EmitContext.recordError ctx msg
            EmitError msg

    | "fidelity_strlen" ->
        match argSsas with
        | [(ptrSsa, _)] ->
            let result = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = llvm.inline_asm \"xor %%rcx, %%rcx\\0A1: cmpb $$0, (%%rdi,%%rcx)\\0Aje 2f\\0Ainc %%rcx\\0Ajmp 1b\\0A2:\", \"={rcx},{rdi}\" %s : (!llvm.ptr) -> i64" result ptrSsa)
            let truncResult = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = arith.trunci %s : i64 to i32" truncResult result)
            Value (truncResult, "i32")
        | _ ->
            let msg = sprintf "fidelity_strlen: expected 1 arg, got %d" (List.length argSsas)
            EmitContext.recordError ctx msg
            EmitError msg

    | _ ->
        let msg = sprintf "Unknown fidelity extern: %s" info.EntryPoint
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
        | [(sizeSsa, _)] ->
            let allocSsa = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = llvm.alloca %s x i8 : (i64) -> !llvm.ptr" allocSsa sizeSsa)
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

/// Emit inlined function call by name
and emitInlinedCallByName (ctx: EmitContext) (methodFullName: string) (argNodes: PSGNode list) : ExprResult =
    let flatArgs = argNodes |> List.collect (fun n ->
        if n.SyntaxKind = "Tuple" then
            ChildrenStateHelpers.getChildrenList n
            |> List.choose (fun id -> Map.tryFind id.Value ctx.Graph.Nodes)
        else [n])
    let argResults = flatArgs |> List.map (emitNode ctx)
    let argSsas = argResults |> List.choose (function Value (s, t) -> Some (s, t) | _ -> None)

    let funcBinding =
        ctx.Graph.Nodes
        |> Map.toSeq
        |> Seq.tryFind (fun (_, n) ->
            n.SyntaxKind.StartsWith("Binding") &&
            match n.Symbol with
            | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
                try mfv.FullName = methodFullName with _ -> false
            | _ -> false)
        |> Option.map snd

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
        let msg = sprintf "SRTP function not found in PSG: %s - cannot inline" methodFullName
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
        let funcBinding =
            ctx.Graph.Nodes
            |> Map.toSeq
            |> Seq.tryFind (fun (_, n) ->
                n.SyntaxKind.StartsWith("Binding") &&
                match n.Symbol with
                | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
                    try mfv.FullName = name with _ -> false
                | _ -> false)
            |> Option.map snd

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
        EmitContext.recordError ctx "Function call with no symbol - cannot determine target"
        EmitError "Function call with no symbol"

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

/// Emit tuple
and emitTuple (ctx: EmitContext) (zipper: PSGZipper) : ExprResult =
    let children = PSGZipper.childNodes zipper
    match children with
    | [] -> Void
    | _ -> children |> List.fold (fun _ child -> emitNode ctx child) Void

/// Emit address-of
and emitAddressOf (ctx: EmitContext) (zipper: PSGZipper) : ExprResult =
    let children = PSGZipper.childNodes zipper
    match children with
    | [child] ->
        let result = emitNode ctx child
        match result with
        | Value (ssa, _) -> Value (ssa, "!llvm.ptr")
        | EmitError msg ->
            // Propagate child error
            EmitError msg
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

    let retVal = EmitContext.nextSSA ctx
    EmitContext.emitLine ctx (sprintf "  %s = arith.constant 0 : i32" retVal)
    EmitContext.emitLine ctx (sprintf "  llvm.return %s : i32" retVal)
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
