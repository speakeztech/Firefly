/// FNCSTransfer - Witness-based transfer from FNCS SemanticGraph to MLIR
///
/// ARCHITECTURAL FOUNDATION (Coeffects & Codata):
/// This module witnesses the FNCS SemanticGraph structure and produces MLIR
/// via the MLIRZipper codata accumulator.
///
/// KEY DESIGN PRINCIPLES:
/// 1. Post-order traversal: Children before parents (SSAs available when parent visited)
/// 2. Dispatch on SemanticKind ONLY - no pattern matching on symbol names
/// 3. Platform bindings via SemanticKind.PlatformBinding marker → PlatformDispatch
/// 4. Codata vocabulary: witness, observe, yield, bind, recall, extract
///
/// This replaces the deleted FNCSEmitter.fs antipattern.
module Alex.Traversal.FNCSTransfer

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open Alex.CodeGeneration.MLIRTypes
open Alex.Traversal.MLIRZipper
open Alex.Bindings.BindingTypes

// ═══════════════════════════════════════════════════════════════════
// Type Mapping: NativeType → MLIRType
// ═══════════════════════════════════════════════════════════════════

/// Map FNCS NativeType to MLIR type
/// NativeType uses TApp(tycon, args) for applied types
let rec mapType (ty: NativeType) : MLIRType =
    match ty with
    // Applied types - match on type constructor name
    | NativeType.TApp(tycon, _args) ->
        match tycon.Name with
        | "unit" -> Unit
        | "bool" -> Integer I1
        | "int8" | "sbyte" -> Integer I8
        | "uint8" | "byte" -> Integer I8
        | "int16" -> Integer I16
        | "uint16" -> Integer I16
        | "int" | "int32" -> Integer I32
        | "uint" | "uint32" -> Integer I32
        | "int64" -> Integer I64
        | "uint64" -> Integer I64
        | "nativeint" -> Integer I64  // Platform dependent
        | "unativeint" -> Integer I64
        | "float32" | "single" -> Float F32
        | "float" | "double" -> Float F64
        | "char" -> Integer I32  // Unicode codepoint
        | "string" -> NativeStrType  // Fat pointer {ptr: *u8, len: i64}
        | "Ptr" | "nativeptr" -> Pointer
        | "array" -> Pointer
        | "list" -> Pointer
        | "option" | "voption" -> Pointer  // TODO: Value type layout
        | _ -> Pointer  // Default to pointer for unknown applied types

    // Function types
    | NativeType.TFun _ -> Pointer  // Function pointer + closure

    // Tuple types
    | NativeType.TTuple _ -> Pointer  // TODO: Proper struct layout

    // Type variables (erased)
    | NativeType.TVar _ -> Pointer

    // Byref types
    | NativeType.TByref _ -> Pointer

    // Native pointers
    | NativeType.TNativePtr _ -> Pointer

    // Forall types - look at body
    | NativeType.TForall(_, body) -> mapType body

    // Other types
    | NativeType.TRecord _ -> Pointer
    | NativeType.TUnion _ -> Pointer
    | NativeType.TAnon _ -> Pointer
    | NativeType.TMeasure _ -> Unit  // Phantom type, no runtime representation
    | NativeType.TError msg -> failwithf "Type error in semantic graph: %s" msg

// ═══════════════════════════════════════════════════════════════════
// Witness Functions for SemanticKind
// ═══════════════════════════════════════════════════════════════════

/// Witness a literal value
let witnessLiteral (lit: LiteralValue) (zipper: MLIRZipper) : MLIRZipper * TransferResult =
    match lit with
    | LiteralValue.Unit ->
        // Unit needs a concrete representation for SSA references
        // Generate a zero constant (LLVM doesn't have true unit type)
        let ssaName, zipper' = MLIRZipper.witnessConstant 0L I32 zipper
        zipper', TRValue (ssaName, "i32")

    | LiteralValue.Bool b ->
        let value = if b then 1L else 0L
        let ssaName, zipper' = MLIRZipper.witnessConstant value I1 zipper
        zipper', TRValue (ssaName, "i1")

    | LiteralValue.Int32 n ->
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I32 zipper
        zipper', TRValue (ssaName, "i32")

    | LiteralValue.Int64 n ->
        let ssaName, zipper' = MLIRZipper.witnessConstant n I64 zipper
        zipper', TRValue (ssaName, "i64")

    | LiteralValue.String s ->
        // NATIVE STRING: Fat pointer struct {ptr: *u8, len: i64}
        // See Serena memories: string_length_handling, fidelity_memory_model
        //
        // Step 1: Observe string literal (creates global constant)
        let globalName, zipper1 = MLIRZipper.observeStringLiteral s zipper
        
        // Step 2: Get address of character data
        let ptrSSA, zipper2 = MLIRZipper.witnessAddressOf globalName zipper1
        
        // Step 3: Create length constant (string length without null terminator)
        let len = s.Length
        let lenSSA, zipper3 = MLIRZipper.witnessConstant (int64 len) I64 zipper2
        
        // Step 4: Build fat pointer struct using llvm.mlir.undef + insertvalue
        // This follows the Fidelity memory model: F# types dictate MLIR layout
        let undefSSA, zipper4 = MLIRZipper.yieldSSA zipper3
        let undefText = sprintf "%s = llvm.mlir.undef : %s" undefSSA NativeStrTypeStr
        let zipper5 = MLIRZipper.witnessOpWithResult undefText undefSSA NativeStrType zipper4
        
        // Step 5: Insert pointer at index 0
        let withPtrSSA, zipper6 = MLIRZipper.yieldSSA zipper5
        let insertPtrText = sprintf "%s = llvm.insertvalue %s, %s[0] : %s" withPtrSSA ptrSSA undefSSA NativeStrTypeStr
        let zipper7 = MLIRZipper.witnessOpWithResult insertPtrText withPtrSSA NativeStrType zipper6
        
        // Step 6: Insert length at index 1
        let fatPtrSSA, zipper8 = MLIRZipper.yieldSSA zipper7
        let insertLenText = sprintf "%s = llvm.insertvalue %s, %s[1] : %s" fatPtrSSA lenSSA withPtrSSA NativeStrTypeStr
        let zipper9 = MLIRZipper.witnessOpWithResult insertLenText fatPtrSSA NativeStrType zipper8
        
        zipper9, TRValue (fatPtrSSA, NativeStrTypeStr)

    | LiteralValue.Float32 f ->
        let ssaName, zipper' = MLIRZipper.yieldSSA zipper
        let text = sprintf "%s = arith.constant %e : f32" ssaName (float f)
        let zipper'' = MLIRZipper.witnessOpWithResult text ssaName (Float F32) zipper'
        zipper'', TRValue (ssaName, "f32")

    | LiteralValue.Float64 f ->
        let ssaName, zipper' = MLIRZipper.yieldSSA zipper
        let text = sprintf "%s = arith.constant %e : f64" ssaName f
        let zipper'' = MLIRZipper.witnessOpWithResult text ssaName (Float F64) zipper'
        zipper'', TRValue (ssaName, "f64")

    | LiteralValue.Char c ->
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 c) I32 zipper
        zipper', TRValue (ssaName, "i32")

    | LiteralValue.Int8 n ->
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I8 zipper
        zipper', TRValue (ssaName, "i8")

    | LiteralValue.UInt8 n ->
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I8 zipper
        zipper', TRValue (ssaName, "i8")

    | LiteralValue.Int16 n ->
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I16 zipper
        zipper', TRValue (ssaName, "i16")

    | LiteralValue.UInt16 n ->
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I16 zipper
        zipper', TRValue (ssaName, "i16")

    | LiteralValue.UInt32 n ->
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I32 zipper
        zipper', TRValue (ssaName, "i32")

    | LiteralValue.UInt64 n ->
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I64 zipper
        zipper', TRValue (ssaName, "i64")

    | _ ->
        // Unsupported literal type
        zipper, TRError (sprintf "Unsupported literal: %A" lit)

/// Check if a name is a built-in operator or function
let isBuiltInOperator (name: string) =
    name.StartsWith("op_") ||
    name.StartsWith("NativePtr.") ||
    name.StartsWith("Sys.") ||
    name.StartsWith("Unchecked.") ||
    List.contains name ["not"; "int"; "int8"; "int16"; "int32"; "int64";
                        "byte"; "uint8"; "uint16"; "uint32"; "uint64";
                        "float"; "float32"; "double"; "single"; "decimal";
                        "nativeint"; "unativeint"; "char"; "string";
                        "box"; "unbox"; "Some"; "None"; "ValueSome"; "ValueNone";
                        "printf"; "printfn"; "sprintf"; "failwith"; "failwithf";
                        "Array.zeroCreate"; "Array.length"; "Array.get"; "Array.set";
                        "ignore"; "raise"; "reraise"; "typeof"; "sizeof"; "nameof"]

/// Witness a variable reference (recall its SSA from prior observation)
/// INVARIANT: FNCS guarantees definitions are traversed before uses
/// If this fails, it's an FNCS graph construction bug - hard stop with diagnostic
let witnessVarRef (name: string) (defId: NodeId option) (graph: SemanticGraph) (zipper: MLIRZipper) : MLIRZipper * TransferResult =
    // First try to look up by variable name (for Lambda parameters)
    match MLIRZipper.recallVar name zipper with
    | Some (ssaName, mlirType) ->
        zipper, TRValue (ssaName, mlirType)
    | None ->
        // Not a bound variable - try definition node
        match defId with
        | Some nodeId ->
            // Recall the SSA value bound to this definition
            match MLIRZipper.recallNodeSSA (string (NodeId.value nodeId)) zipper with
            | Some (ssaName, mlirType) ->
                zipper, TRValue (ssaName, mlirType)
            | None ->
                // Definition node wasn't traversed - check if it's a module-level binding
                // Module-level bindings may be siblings of entry points, not children
                match SemanticGraph.tryGetNode nodeId graph with
                | None ->
                    failwithf "FNCS GRAPH ERROR: Variable '%s' references node %d which is NOT in graph. Tree-shaking must include all referenced definitions." name (NodeId.value nodeId)
                | Some defNode ->
                    // If it's a Binding, try to find its value (first child) SSA
                    match defNode.Kind, defNode.Children with
                    | SemanticKind.Binding (_, _, _, _), valueNodeId :: _ ->
                        // Check if the binding's value (Lambda) was traversed
                        match MLIRZipper.recallNodeSSA (string (NodeId.value valueNodeId)) zipper with
                        | Some (lambdaSSA, lambdaType) ->
                            // Found the value's SSA - use it
                            zipper, TRValue (lambdaSSA, lambdaType)
                        | None ->
                            // Value wasn't traversed - try looking for its function name marker
                            match MLIRZipper.recallNodeSSA (string (NodeId.value valueNodeId) + "_lambdaName") zipper with
                            | Some (lambdaName, _) ->
                                // Found the lambda name marker - construct function reference
                                zipper, TRValue ("@" + lambdaName, "!llvm.ptr")
                            | None ->
                                // DEFERRED RESOLUTION: The binding's value wasn't traversed yet
                                // Check if the value is a Literal (constant) or Lambda (function)
                                match SemanticGraph.tryGetNode valueNodeId graph with
                                | Some valueNode ->
                                    match valueNode.Kind with
                                    | SemanticKind.Literal lit ->
                                        // It's a constant - emit the literal value directly
                                        witnessLiteral lit zipper
                                    | SemanticKind.Lambda _ ->
                                        // It's a function - use its actual type from the graph
                                        let funcName = name
                                        let signature =
                                            match valueNode.Type with
                                            | NativeType.TFun(paramTy, retTy) ->
                                                sprintf "(%s) -> %s"
                                                    (Serialize.mlirType (mapType paramTy))
                                                    (Serialize.mlirType (mapType retTy))
                                            | _ ->
                                                sprintf "(%s) -> %s"
                                                    (Serialize.mlirType (mapType valueNode.Type))
                                                    "!llvm.ptr"
                                        let retType =
                                            match valueNode.Type with
                                            | NativeType.TFun(_, retTy) -> Serialize.mlirType (mapType retTy)
                                            | _ -> "!llvm.ptr"
                                        let zipper' = MLIRZipper.observeExternFunc funcName signature zipper
                                        zipper', TRValue ("@" + funcName, retType)
                                    | _ ->
                                        // Unknown value type - use its actual type
                                        let funcName = name
                                        let signature =
                                            match valueNode.Type with
                                            | NativeType.TFun(paramTy, retTy) ->
                                                sprintf "(%s) -> %s"
                                                    (Serialize.mlirType (mapType paramTy))
                                                    (Serialize.mlirType (mapType retTy))
                                            | _ -> sprintf "(%s) -> %s"
                                                    (Serialize.mlirType (mapType valueNode.Type))
                                                    "!llvm.ptr"
                                        let retType =
                                            match valueNode.Type with
                                            | NativeType.TFun(_, retTy) -> Serialize.mlirType (mapType retTy)
                                            | _ -> "!llvm.ptr"
                                        let zipper' = MLIRZipper.observeExternFunc funcName signature zipper
                                        zipper', TRValue ("@" + funcName, retType)
                                | None ->
                                    // Value node doesn't exist - use binding's type
                                    let funcName = name
                                    let signature =
                                        match defNode.Type with
                                        | NativeType.TFun(paramTy, retTy) ->
                                            sprintf "(%s) -> %s"
                                                (Serialize.mlirType (mapType paramTy))
                                                (Serialize.mlirType (mapType retTy))
                                        | _ -> sprintf "(%s) -> %s"
                                                (Serialize.mlirType (mapType defNode.Type))
                                                "!llvm.ptr"
                                    let retType =
                                        match defNode.Type with
                                        | NativeType.TFun(_, retTy) -> Serialize.mlirType (mapType retTy)
                                        | _ -> "!llvm.ptr"
                                    let zipper' = MLIRZipper.observeExternFunc funcName signature zipper
                                    zipper', TRValue ("@" + funcName, retType)
                    | _ ->
                        // Non-binding reference - use its actual type
                        let funcName = name
                        let signature =
                            match defNode.Type with
                            | NativeType.TFun(paramTy, retTy) ->
                                sprintf "(%s) -> %s"
                                    (Serialize.mlirType (mapType paramTy))
                                    (Serialize.mlirType (mapType retTy))
                            | _ -> sprintf "(%s) -> %s"
                                    (Serialize.mlirType (mapType defNode.Type))
                                    "!llvm.ptr"
                        let retType =
                            match defNode.Type with
                            | NativeType.TFun(_, retTy) -> Serialize.mlirType (mapType retTy)
                            | _ -> "!llvm.ptr"
                        let zipper' = MLIRZipper.observeExternFunc funcName signature zipper
                        zipper', TRValue ("@" + funcName, retType)
        | None ->
            // No definition node - check if it's a built-in
            if isBuiltInOperator name then
                zipper, TRBuiltin name
            else
                zipper, TRError (sprintf "Variable '%s' has no definition node" name)

/// Witness a platform binding call
let witnessPlatformBinding (entryPoint: string) (argSSAs: (string * MLIRType) list) (returnType: NativeType) (zipper: MLIRZipper) : MLIRZipper * TransferResult =
    // Create PlatformPrimitive for dispatch
    let prim: PlatformPrimitive = {
        EntryPoint = entryPoint
        Library = "platform"
        CallingConvention = "ccc"
        Args = argSSAs
        ReturnType = mapType returnType
        BindingStrategy = Static
    }

    // Dispatch to platform-specific implementation
    let zipper', result = PlatformDispatch.dispatch prim zipper

    match result with
    | WitnessedValue (ssa, ty) ->
        zipper', TRValue (ssa, Serialize.mlirType ty)
    | WitnessedVoid ->
        zipper', TRVoid
    | NotSupported reason ->
        zipper', TRError (sprintf "Platform binding '%s' not supported: %s" entryPoint reason)

/// Witness a function application
let witnessApplication (funcNodeId: NodeId) (argNodeIds: NodeId list) (returnType: NativeType) (graph: SemanticGraph) (zipper: MLIRZipper) : MLIRZipper * TransferResult =
    // Get the function node to understand what kind of call this is
    match SemanticGraph.tryGetNode funcNodeId graph with
    | Some funcNode ->
        match funcNode.Kind with
        | SemanticKind.PlatformBinding entryPoint ->
            // This is a platform binding call - get argument SSAs
            let argSSAs =
                argNodeIds
                |> List.choose (fun nodeId ->
                    match SemanticGraph.tryGetNode nodeId graph with
                    | Some argNode ->
                        match MLIRZipper.recallNodeSSA (string (NodeId.value nodeId)) zipper with
                        | Some (ssa, _) -> Some (ssa, mapType argNode.Type)
                        | None -> None
                    | None -> None)
            
            // Platform bindings have TVar types from FNCS (types aren't constrained).
            // Use known parameter counts for platform bindings.
            let expectedParamCount =
                match entryPoint with
                | "writeBytes" | "readBytes" -> 3   // fd, buffer, count
                | "getCurrentTicks" -> 0
                | "sleep" -> 1
                | _ -> 0  // Unknown - assume fully applied
            // Debug: %s has %d/%d args - use intermediates for inspection
            
            if List.length argSSAs < expectedParamCount then
                // Partial application - return a marker for later
                // Format: $platform:entryPoint:arg0:type0:arg1:type1:...
                let argsEncoded = 
                    argSSAs 
                    |> List.collect (fun (ssa, ty) -> [ssa; Serialize.mlirType ty]) 
                    |> String.concat ":"
                let marker = 
                    if argsEncoded.Length > 0 then sprintf "$platform:%s:%s" entryPoint argsEncoded
                    else sprintf "$platform:%s" entryPoint
                ()
                zipper, TRValue (marker, "func")
            else
                // All arguments present - call the platform binding
                witnessPlatformBinding entryPoint argSSAs returnType zipper

        | SemanticKind.Intrinsic intrinsicName ->
            // Handle compiler intrinsics like NativePtr.toNativeInt
            let argSSAs =
                argNodeIds
                |> List.choose (fun nodeId ->
                    MLIRZipper.recallNodeSSA (string (NodeId.value nodeId)) zipper)

            match intrinsicName, argSSAs with
            | "NativePtr.toNativeInt", [(argSSA, _)] ->
                // nativeptr<'T> -> nativeint (pointer to integer conversion)
                let ssaName, zipper' = MLIRZipper.yieldSSA zipper
                // Use ptrtoint to convert pointer to integer
                let text = sprintf "%s = llvm.ptrtoint %s : !llvm.ptr to i64" ssaName argSSA
                let zipper'' = MLIRZipper.witnessOpWithResult text ssaName (Integer I64) zipper'
                zipper'', TRValue (ssaName, "i64")
            | "NativePtr.ofNativeInt", [(argSSA, _)] ->
                // nativeint -> nativeptr<'T> (integer to pointer conversion)
                let ssaName, zipper' = MLIRZipper.yieldSSA zipper
                let text = sprintf "%s = llvm.inttoptr %s : i64 to !llvm.ptr" ssaName argSSA
                let zipper'' = MLIRZipper.witnessOpWithResult text ssaName Pointer zipper'
                zipper'', TRValue (ssaName, "!llvm.ptr")
            | "NativePtr.toVoidPtr", [(argSSA, _)] ->
                // nativeptr<'T> -> voidptr (just pass through, same type at MLIR level)
                zipper, TRValue (argSSA, "!llvm.ptr")
            | "NativePtr.ofVoidPtr", [(argSSA, _)] ->
                // voidptr -> nativeptr<'T> (just pass through, same type at MLIR level)
                zipper, TRValue (argSSA, "!llvm.ptr")
            | "NativePtr.get", [(ptrSSA, _); (idxSSA, _)] ->
                // nativeptr<'T> -> int -> 'T (indexed load)
                let elemType = Serialize.mlirType (mapType returnType)
                let gepSSA, zipper' = MLIRZipper.yieldSSA zipper
                let gepText = sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i32) -> !llvm.ptr, i8" gepSSA ptrSSA idxSSA
                let zipper'' = MLIRZipper.witnessOpWithResult gepText gepSSA Pointer zipper'
                let loadSSA, zipper''' = MLIRZipper.yieldSSA zipper''
                let loadText = sprintf "%s = llvm.load %s : !llvm.ptr -> %s" loadSSA gepSSA elemType
                let zipper4 = MLIRZipper.witnessOpWithResult loadText loadSSA (mapType returnType) zipper'''
                zipper4, TRValue (loadSSA, elemType)
            | "NativePtr.set", [(ptrSSA, _); (idxSSA, _); (valSSA, _)] ->
                // nativeptr<'T> -> int -> 'T -> unit (indexed store)
                let gepSSA, zipper' = MLIRZipper.yieldSSA zipper
                let gepText = sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i32) -> !llvm.ptr, i8" gepSSA ptrSSA idxSSA
                let zipper'' = MLIRZipper.witnessOpWithResult gepText gepSSA Pointer zipper'
                let storeText = sprintf "llvm.store %s, %s : i8, !llvm.ptr" valSSA gepSSA
                let zipper''' = MLIRZipper.witnessVoidOp storeText zipper''
                zipper''', TRVoid
            | "NativePtr.stackalloc", [(countSSA, _)] ->
                // int -> nativeptr<'T> (stack allocation)
                // Allocate 'count' elements of the element type on the stack
                // The return type tells us the element type: nativeptr<'T> -> T
                let elemType =
                    match returnType with
                    | NativeType.TNativePtr elemTy -> Serialize.mlirType (mapType elemTy)
                    | _ -> "i8"  // Default to byte if type unknown
                let ssaName, zipper' = MLIRZipper.yieldSSA zipper
                // Convert count from i32 to i64 for llvm.alloca
                let countSSA64, zipper'' = MLIRZipper.yieldSSA zipper'
                let extText = sprintf "%s = arith.extsi %s : i32 to i64" countSSA64 countSSA
                let zipper''' = MLIRZipper.witnessOpWithResult extText countSSA64 (Integer I64) zipper''
                // Emit llvm.alloca
                let allocaText = sprintf "%s = llvm.alloca %s x %s : (i64) -> !llvm.ptr" ssaName countSSA64 elemType
                let zipper4 = MLIRZipper.witnessOpWithResult allocaText ssaName Pointer zipper'''
                zipper4, TRValue (ssaName, "!llvm.ptr")
            | "NativePtr.copy", [(destSSA, _); (srcSSA, _); (countSSA, _)] ->
                // dest:nativeptr<'T> -> src:nativeptr<'T> -> count:int -> unit
                // Maps to llvm.memcpy intrinsic
                // Convert count from i32 to i64 for llvm.intr.memcpy
                let countSSA64, zipper' = MLIRZipper.yieldSSA zipper
                let extText = sprintf "%s = arith.extsi %s : i32 to i64" countSSA64 countSSA
                let zipper'' = MLIRZipper.witnessOpWithResult extText countSSA64 (Integer I64) zipper'
                // Emit llvm.intr.memcpy: dest, src, len, isVolatile
                let memcpyText = sprintf "\"llvm.intr.memcpy\"(%s, %s, %s) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()" destSSA srcSSA countSSA64
                let zipper''' = MLIRZipper.witnessVoidOp memcpyText zipper''
                zipper''', TRVoid
            | "NativePtr.add", [(ptrSSA, _); (offsetSSA, _)] ->
                // nativeptr<'T> -> int -> nativeptr<'T> (pointer arithmetic)
                let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
                let gepText = sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i32) -> !llvm.ptr, i8" resultSSA ptrSSA offsetSSA
                let zipper'' = MLIRZipper.witnessOpWithResult gepText resultSSA Pointer zipper'
                zipper'', TRValue (resultSSA, "!llvm.ptr")

            // ═══════════════════════════════════════════════════════════════════════════
            // Sys intrinsics - Dispatch to platform bindings layer
            // Platform-specific MLIR (syscall numbers, ABIs) is DATA in Bindings
            // ═══════════════════════════════════════════════════════════════════════════
            | intrinsicName, argSSAs when intrinsicName.StartsWith("Sys.") ->
                // Extract the entry point (e.g., "Sys.write" -> "Sys.write")
                // Map argument SSAs to (ssa, type) pairs for PlatformPrimitive
                let argSSAsWithTypes =
                    argSSAs |> List.map (fun (ssa, tyStr) -> (ssa, Serialize.deserializeType tyStr))
                // Dispatch to platform bindings - all platform-specific logic is there
                witnessPlatformBinding intrinsicName argSSAsWithTypes returnType zipper

            // ═══════════════════════════════════════════════════════════════════════════
            // NativeStr intrinsics - Native string (fat pointer) construction
            // ═══════════════════════════════════════════════════════════════════════════
            | "NativeStr.fromPointer", [(ptrSSA, _); (lenSSA, _)] ->
                // Construct fat pointer struct from pointer and length
                // Same pattern as string literal emission: undef + insertvalue
                let undefSSA, zipper1 = MLIRZipper.yieldSSA zipper
                let undefText = sprintf "%s = llvm.mlir.undef : %s" undefSSA NativeStrTypeStr
                let zipper2 = MLIRZipper.witnessOpWithResult undefText undefSSA NativeStrType zipper1

                // Insert pointer at index 0
                let withPtrSSA, zipper3 = MLIRZipper.yieldSSA zipper2
                let insertPtrText = sprintf "%s = llvm.insertvalue %s, %s[0] : %s" withPtrSSA ptrSSA undefSSA NativeStrTypeStr
                let zipper4 = MLIRZipper.witnessOpWithResult insertPtrText withPtrSSA NativeStrType zipper3

                // Insert length at index 1 (need to extend to i64 if it's i32)
                let lenSSA64, zipper5 = MLIRZipper.yieldSSA zipper4
                let extText = sprintf "%s = arith.extsi %s : i32 to i64" lenSSA64 lenSSA
                let zipper6 = MLIRZipper.witnessOpWithResult extText lenSSA64 (Integer I64) zipper5

                let fatPtrSSA, zipper7 = MLIRZipper.yieldSSA zipper6
                let insertLenText = sprintf "%s = llvm.insertvalue %s, %s[1] : %s" fatPtrSSA lenSSA64 withPtrSSA NativeStrTypeStr
                let zipper8 = MLIRZipper.witnessOpWithResult insertLenText fatPtrSSA NativeStrType zipper7

                zipper8, TRValue (fatPtrSSA, NativeStrTypeStr)

            // ═══════════════════════════════════════════════════════════════════════════
            // NativeDefault intrinsics - Zero-initialized values
            // ═══════════════════════════════════════════════════════════════════════════
            | "NativeDefault.zeroed", [] ->
                // Zero value for the return type
                // The return type tells us what kind of zero to emit
                let zeroSSA, zipper' = MLIRZipper.yieldSSA zipper
                let mlirRetType = mapType returnType
                let mlirTypeStr = Serialize.mlirType mlirRetType
                let zeroText =
                    match mlirRetType with
                    | Integer _ -> sprintf "%s = arith.constant 0 : %s" zeroSSA mlirTypeStr
                    | Float F32 -> sprintf "%s = arith.constant 0.0 : f32" zeroSSA
                    | Float F64 -> sprintf "%s = arith.constant 0.0 : f64" zeroSSA
                    | Pointer -> sprintf "%s = llvm.mlir.zero : !llvm.ptr" zeroSSA
                    | Struct _ when mlirTypeStr = NativeStrTypeStr ->
                        // Zero string: undef (ptr=null, len=0 effectively)
                        sprintf "%s = llvm.mlir.undef : %s" zeroSSA NativeStrTypeStr
                    | Struct _ ->
                        // For other structs, use undef (zero-initialized)
                        sprintf "%s = llvm.mlir.undef : %s" zeroSSA mlirTypeStr
                    | _ ->
                        // Default: use undef
                        sprintf "%s = llvm.mlir.undef : %s" zeroSSA mlirTypeStr
                let zipper'' = MLIRZipper.witnessOpWithResult zeroText zeroSSA mlirRetType zipper'
                zipper'', TRValue (zeroSSA, mlirTypeStr)

            | intrinsicName, [] ->
                // TypeApp or partial application - return marker for subsequent application
                // This handles polymorphic intrinsics (e.g., stackalloc<byte>) before value args are applied
                zipper, TRValue ("$intrinsic:" + intrinsicName, "func")
            | _ ->
                // Unknown intrinsic with args - report error
                zipper, TRError (sprintf "Unknown intrinsic: %s with %d args" intrinsicName (List.length argSSAs))

        | SemanticKind.VarRef (name, defId) ->
            // Function call to a named function - follow the definition to find the Lambda's SSA
            // Get both SSA values AND their actual types (not PSG declared types!)
            let argSSAsAndTypes =
                argNodeIds
                |> List.choose (fun nodeId ->
                    MLIRZipper.recallNodeSSA (string (NodeId.value nodeId)) zipper)
            let argSSAs = argSSAsAndTypes |> List.map fst
            // Use actual SSA types for the call signature
            let argTypes = argSSAsAndTypes |> List.map (fun (_, ty) -> Serialize.deserializeType ty)

            // Special handling for built-in operators like pipe (|>)
            // op_PipeRight(x) returns a partial application marker; op_PipeRight(x)(f) becomes f(x)
            if name = "op_PipeRight" || name = "op_PipeLeft" then
                match argSSAs with
                | [argSSA] ->
                    // Partial application of pipe: |> x
                    // Return a marker that includes the argument so we can inline when called
                    let (argType: string) = argSSAsAndTypes |> List.head |> snd
                    let marker = sprintf "$pipe:%s:%s" argSSA argType
                    zipper, TRValue (marker, "func")
                | _ ->
                    zipper, TRError (sprintf "Pipe operator '%s' expects 1 argument, got %d" name (List.length argSSAs))
            else

            // Try to recall the Lambda's SSA from its definition node
            // This is critical: VarRef points to a Binding, whose child is the Lambda
            // The Lambda's SSA is bound to its node ID (e.g., "@lambda_6")
            match defId with
            | Some defNodeId ->
                // Try to find the Lambda's SSA via the definition binding
                match MLIRZipper.recallNodeSSA (string (NodeId.value defNodeId)) zipper with
                | Some (funcSSA, _funcType) ->
                    // Got the Lambda's SSA (like "@lambda_6") - use it for the call
                    // Check if this is a function reference (starts with @)
                    if funcSSA.StartsWith("@") then
                        // Direct function call
                        let funcName = funcSSA.Substring(1) // Remove leading @
                        
                        // Check for partial application: fewer arguments than function parameters
                        let expectedParams = MLIRZipper.lookupFuncParamCount funcName zipper
                        match expectedParams with
                        | Some paramCount when paramCount > List.length argSSAs ->
                            // Partial application - return a curried function marker
                            // Format: $partial:funcName:arg0:type0:arg1:type1:...
                            let argPairs = List.zip argSSAs (argTypes |> List.map Serialize.mlirType)
                            let argsEncoded = argPairs |> List.collect (fun (a, t) -> [a; t]) |> String.concat ":"
                            let marker = sprintf "$partial:%s:%s" funcName argsEncoded
                            zipper, TRValue (marker, "func")
                        | _ ->
                            // Full application - call the function
                            let ssaName, zipper' = MLIRZipper.witnessCall funcName argSSAs argTypes (mapType returnType) zipper
                            // Use tracked return type if available (for correct lowered types)
                            // This is critical: PSG may have TVar which maps to !llvm.ptr, but the
                            // actual function may return i32 - we need the ACTUAL type for correct SSA binding
                            let actualRetType =
                                match MLIRZipper.lookupFuncReturnType funcName zipper' with
                                | Some t -> t
                                | None -> Serialize.mlirType (mapType returnType)
                            zipper', TRValue (ssaName, actualRetType)
                    else
                        // Function value in SSA register - indirect call
                        let ssaName, zipper' = MLIRZipper.witnessIndirectCall funcSSA argSSAs argTypes (mapType returnType) zipper
                        zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))
                | None ->
                    // Definition wasn't traversed - fall back to name (may be undefined)
                    let ssaName, zipper' = MLIRZipper.witnessCall name argSSAs argTypes (mapType returnType) zipper
                    zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))
            | None ->
                // No definition ID - use the name directly (may be external)
                let ssaName, zipper' = MLIRZipper.witnessCall name argSSAs argTypes (mapType returnType) zipper
                zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))

        | SemanticKind.Lambda _ ->
            // Lambda application - for now, treat as error (needs closure support)
            zipper, TRError "Lambda application not yet supported"

        | SemanticKind.Application (innerFuncId, innerArgIds) ->
            // Curried application: the function is itself a partial application
            // The inner application was already processed (post-order), so recall its result
            match MLIRZipper.recallNodeSSA (string (NodeId.value funcNodeId)) zipper with
            | Some (funcSSA, _funcType) ->
                // Get argument SSAs AND their actual types (not PSG declared types!)
                let argSSAsAndTypes =
                    argNodeIds
                    |> List.choose (fun nodeId ->
                        MLIRZipper.recallNodeSSA (string (NodeId.value nodeId)) zipper)
                let argSSAs = argSSAsAndTypes |> List.map fst
                // Use actual SSA types for the call signature
                let argTypes = argSSAsAndTypes |> List.map (fun (_, ty) -> Serialize.deserializeType ty)

                // Check if this is a pipe operator application: $pipe:argSSA:argType
                // If so, inline it: (|> x)(f) becomes f(x)
                if funcSSA.StartsWith("$pipe:") then
                    let parts = funcSSA.Split(':')
                    if parts.Length >= 3 then
                        let pipedArgSSA = parts.[1]
                        let pipedArgType = parts.[2]
                        // The function f is the first argument to this application
                        match argSSAs with
                        | [fSSA] ->
                            // Check return type - if it's unit (i32) and piped arg is also unit,
                            // this is likely `unit |> ignore` pattern - just return unit
                            let retTypeStr = Serialize.mlirType (mapType returnType)
                            if retTypeStr = "i32" && pipedArgType = "i32" then
                                // Unit piped to unit-returning function (like ignore)
                                // Just return unit directly to avoid type mismatch
                                let unitSSA, zipper' = MLIRZipper.yieldSSA zipper
                                let unitText = sprintf "%s = arith.constant 0 : i32" unitSSA
                                let zipper'' = MLIRZipper.witnessOpWithResult unitText unitSSA (Integer I32) zipper'
                                zipper'', TRValue (unitSSA, "i32")
                            else
                                // Apply f to the piped argument: f(x)
                                let pipedTypes = [Serialize.deserializeType pipedArgType]
                                // Check if f is a direct function reference (starts with @)
                                if fSSA.StartsWith("@") then
                                    let funcName = fSSA.Substring(1) // Remove leading @
                                    // Special handling for ignore - just discard the argument and return unit
                                    if funcName = "ignore" then
                                        let unitSSA, zipper' = MLIRZipper.yieldSSA zipper
                                        let unitText = sprintf "%s = arith.constant 0 : i32" unitSSA
                                        let zipper'' = MLIRZipper.witnessOpWithResult unitText unitSSA (Integer I32) zipper'
                                        zipper'', TRValue (unitSSA, "i32")
                                    else
                                        let ssaName, zipper' = MLIRZipper.witnessCall funcName [pipedArgSSA] pipedTypes (mapType returnType) zipper
                                        zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))
                                else
                                    let ssaName, zipper' = MLIRZipper.witnessIndirectCall fSSA [pipedArgSSA] pipedTypes (mapType returnType) zipper
                                    zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))
                        | _ ->
                            zipper, TRError (sprintf "Pipe application expected 1 function arg, got %d" (List.length argSSAs))
                    else
                        zipper, TRError (sprintf "Invalid pipe marker: %s" funcSSA)
                // Handle partial application: $partial:funcName:arg0:type0:arg1:type1:...
                elif funcSSA.StartsWith("$partial:") then
                    let parts = funcSSA.Split(':')
                    if parts.Length >= 2 then
                        let funcName = parts.[1]
                        // Collect already-applied arguments from the marker
                        let appliedArgs = 
                            parts 
                            |> Array.skip 2 
                            |> Array.chunkBySize 2 
                            |> Array.choose (function
                                | [| arg; ty |] -> Some (arg, ty)
                                | _ -> None)
                            |> Array.toList
                        // Combine with new arguments
                        let allArgSSAs = (appliedArgs |> List.map fst) @ argSSAs
                        let allArgTypes = (appliedArgs |> List.map (snd >> Serialize.deserializeType)) @ argTypes
                        
                        // Check if we now have enough arguments
                        match MLIRZipper.lookupFuncParamCount funcName zipper with
                        | Some paramCount when paramCount > List.length allArgSSAs ->
                            // Still partial - return updated marker
                            let argPairs = List.zip allArgSSAs (allArgTypes |> List.map Serialize.mlirType)
                            let argsEncoded = argPairs |> List.collect (fun (a, t) -> [a; t]) |> String.concat ":"
                            let marker = sprintf "$partial:%s:%s" funcName argsEncoded
                            zipper, TRValue (marker, "func")
                        | _ ->
                            // Full application - call the function
                            let ssaName, zipper' = MLIRZipper.witnessCall funcName allArgSSAs allArgTypes (mapType returnType) zipper
                            let actualRetType =
                                match MLIRZipper.lookupFuncReturnType funcName zipper' with
                                | Some t -> t
                                | None -> Serialize.mlirType (mapType returnType)
                            zipper', TRValue (ssaName, actualRetType)
                    else
                        zipper, TRError (sprintf "Invalid partial marker: %s" funcSSA)
                // Handle partial platform binding: $platform:entryPoint:arg0:type0:arg1:type1:...
                elif funcSSA.StartsWith("$platform:") then
                    ()
                    let parts = funcSSA.Split(':')
                    if parts.Length >= 2 then
                        let entryPoint = parts.[1]
                        // Collect already-applied arguments from the marker
                        let appliedArgs = 
                            parts 
                            |> Array.skip 2 
                            |> Array.chunkBySize 2 
                            |> Array.choose (function
                                | [| arg; ty |] -> Some (arg, Serialize.deserializeType ty)
                                | _ -> None)
                            |> Array.toList
                        // Combine with new arguments
                        let allArgSSAs = appliedArgs @ (List.zip argSSAs argTypes)
                        
                        // Check expected param count from platform bindings (3 for writeBytes, etc.)
                        // For now, use simple heuristic: writeBytes/readBytes have 3 params
                        let expectedParamCount = 
                            match entryPoint with
                            | "writeBytes" | "readBytes" -> 3
                            | "getCurrentTicks" -> 0
                            | "sleep" -> 1
                            | _ -> List.length allArgSSAs  // Assume we have all
                        
                        if List.length allArgSSAs < expectedParamCount then
                            // Still partial - return updated marker
                            let argsEncoded = allArgSSAs |> List.collect (fun (a, t) -> [a; Serialize.mlirType t]) |> String.concat ":"
                            let marker = sprintf "$platform:%s:%s" entryPoint argsEncoded
                            zipper, TRValue (marker, "func")
                        else
                            // All arguments present - call the platform binding
                            witnessPlatformBinding entryPoint allArgSSAs returnType zipper
                    else
                        zipper, TRError (sprintf "Invalid platform marker: %s" funcSSA)
                else

                let ssaName, zipper' = MLIRZipper.witnessIndirectCall funcSSA argSSAs argTypes (mapType returnType) zipper
                zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))
            | None ->
                // Inner application wasn't computed - this shouldn't happen in post-order
                zipper, TRError (sprintf "Curried function application not computed: %A" (innerFuncId, innerArgIds))

        | _ ->
            zipper, TRError (sprintf "Unexpected function node kind: %A" funcNode.Kind)

    | None ->
        zipper, TRError (sprintf "Function node not found: %d" (NodeId.value funcNodeId))

/// Witness a sequential expression (children already processed in post-order)
let witnessSequential (nodeIds: NodeId list) (zipper: MLIRZipper) : MLIRZipper * TransferResult =
    // In post-order, children are already witnessed
    // Return the result of the last expression
    match List.tryLast nodeIds with
    | Some lastId ->
        match MLIRZipper.recallNodeSSA (string (NodeId.value lastId)) zipper with
        | Some (ssa, ty) -> zipper, TRValue (ssa, ty)
        | None -> zipper, TRVoid
    | None ->
        zipper, TRVoid

/// Witness a binding (let x = value)
let witnessBinding (name: string) (isMutable: bool) (valueNodeId: NodeId) (node: SemanticNode) (zipper: MLIRZipper) : MLIRZipper * TransferResult =
    // The value has already been witnessed (post-order)
    match MLIRZipper.recallNodeSSA (string (NodeId.value valueNodeId)) zipper with
    | Some (ssa, ty) ->
        if isMutable then
            // Mutable binding: allocate storage with alloca, store value, bind pointer
            // First emit a constant for the array size (MLIR requires SSA operand, not literal)
            let sizeSSA, z1 = MLIRZipper.yieldSSA zipper
            let sizeText = sprintf "%s = arith.constant 1 : i64" sizeSSA
            let z2 = MLIRZipper.witnessOpWithResult sizeText sizeSSA (Integer I64) z1
            let ptrSSA, z3 = MLIRZipper.yieldSSA z2
            let allocaText = sprintf "%s = llvm.alloca %s x i8 : (i64) -> !llvm.ptr" ptrSSA sizeSSA
            let z4 = MLIRZipper.witnessOpWithResult allocaText ptrSSA Pointer z3
            let storeText = sprintf "llvm.store %s, %s : %s, !llvm.ptr" ssa ptrSSA ty
            let z5 = MLIRZipper.witnessVoidOp storeText z4
            // Bind the pointer to both the node and the variable name
            let z6 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ptrSSA "!llvm.ptr" z5
            let z7 = MLIRZipper.bindVar name ptrSSA "!llvm.ptr" z6
            z7, TRValue (ptrSSA, "!llvm.ptr")
        else
            // Immutable binding: just bind the SSA value
            let zipper' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ssa ty zipper
            let zipper'' = MLIRZipper.bindVar name ssa ty zipper'
            zipper'', TRValue (ssa, ty)
    | None ->
        // HARD STOP: Binding's value expression didn't produce an SSA
        // This means either:
        // 1. The value is genuinely unit (OK)
        // 2. The value expression isn't implemented (BUG - should fail at source)
        // Check the node's type to distinguish
        match node.Type with
        | NativeType.TApp (tycon, _) when tycon.Name = "unit" ->
            // Unit binding - no SSA needed, but still bind the node for consistency
            zipper, TRVoid
        | _ ->
            // Non-unit value didn't produce SSA - this is an implementation gap
            failwithf "CODEGEN ERROR: Binding '%s' (node %d) has type %A but value expression produced no SSA. The value expression is not yet implemented for native compilation." name (NodeId.value node.Id) node.Type

// ═══════════════════════════════════════════════════════════════════
// Main Transfer Fold
// ═══════════════════════════════════════════════════════════════════

/// Witness a single node based on its SemanticKind
/// Dispatch is ONLY on SemanticKind - no symbol name pattern matching
let witnessNode (graph: SemanticGraph) (node: SemanticNode) (zipper: MLIRZipper) : MLIRZipper * TransferResult =
    match node.Kind with
    // Literals
    | SemanticKind.Literal lit ->
        let zipper', result = witnessLiteral lit zipper
        // Bind result to this node for future reference
        match result with
        | TRValue (ssa, ty) ->
            let zipper'' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ssa ty zipper'
            zipper'', result
        | _ -> zipper', result

    // Variable references
    | SemanticKind.VarRef (name, defId) ->
        let zipper', result = witnessVarRef name defId graph zipper
        // CRITICAL: Bind result to this node for future reference
        // This is needed when VarRef is used as value in a Binding or TypeAnnotation
        match result with
        | TRValue (ssa, ty) ->
            let zipper'' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ssa ty zipper'
            zipper'', result
        | TRBuiltin opName ->
            // Built-in operator - bind a marker so TypeAnnotation can forward it
            let marker = "$builtin:" + opName
            let zipper'' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) marker "func" zipper'
            zipper'', result
        | _ -> zipper', result

    // Platform bindings (the ONLY place platform calls are recognized)
    | SemanticKind.PlatformBinding entryPoint ->
        // Platform binding node itself doesn't produce a value
        // The Application that uses it will call witnessPlatformBinding
        zipper, TRVoid

    // Compiler intrinsics (e.g., NativePtr.toNativeInt)
    | SemanticKind.Intrinsic intrinsicName ->
        // Intrinsic node itself produces a function value
        // The Application will handle generating actual MLIR code
        // Bind a marker so Application can recognize and handle it
        let zipper' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ("$intrinsic:" + intrinsicName) "func" zipper
        zipper', TRValue ("$intrinsic:" + intrinsicName, "func")

    // Function applications
    | SemanticKind.Application (funcId, argIds) ->
        // DEBUG: Trace Application processing  
        ()
        let zipper', result = witnessApplication funcId argIds node.Type graph zipper
        match result with
        | TRValue (ssa, ty) ->
            let zipper'' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ssa ty zipper'
            zipper'', result
        | _ -> zipper', result

    // Sequential expressions
    | SemanticKind.Sequential nodeIds ->
        // DEBUG: Trace Sequential processing
        let nodeIdStrs = nodeIds |> List.map (fun nid -> string (NodeId.value nid)) |> String.concat ", "
        ()
        let zipper', result = witnessSequential nodeIds zipper
        // Bind result to this node for TypeAnnotation to recall
        match result with
        | TRValue (ssa, ty) ->
            let zipper'' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ssa ty zipper'
            zipper'', result
        | _ -> zipper', result

    // Bindings
    | SemanticKind.Binding (name, isMutable, _isRecursive, _isEntryPoint) ->
        // Get the value node (first child)
        match node.Children with
        | valueId :: _ ->
            witnessBinding name isMutable valueId node zipper
        | [] ->
            zipper, TRError "Binding has no value child"

    // Module definitions (container - children already processed)
    | SemanticKind.ModuleDef (name, _members) ->
        zipper, TRVoid

    // Type definitions (don't generate runtime code)
    | SemanticKind.TypeDef (_name, _kind, _members) ->
        zipper, TRVoid

    // Record expressions - construct a record value
    | SemanticKind.RecordExpr (fields, _copyFrom) ->
        // For records, we need to allocate space and set fields
        // For now, treat struct records as tuples
        let fieldSSAs =
            fields
            |> List.choose (fun (_fieldName, valueId) ->
                MLIRZipper.recallNodeSSA (string (NodeId.value valueId)) zipper)

        if List.length fieldSSAs <> List.length fields then
            // Some fields weren't computed
            zipper, TRError "Record fields not all computed"
        else
            // For a single-field record, just return that field's value
            match fieldSSAs with
            | [(ssa, ty)] ->
                let zipper1 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ssa ty zipper
                zipper1, TRValue (ssa, ty)
            | _ ->
                // Multi-field record - for now, return unit and handle properly later
                zipper, TRVoid

    // Lambdas - non-capturing lambdas become named functions
    // Body operations were captured inside function scope by preBindLambdaParams
    | SemanticKind.Lambda (params', bodyId) ->
        // Check if we're in function scope (preBindLambdaParams should have entered it)
        match zipper.Focus with
        | InFunction funcName ->
            // Get the lambda name that was stored during pre-bind
            let lambdaName = 
                match MLIRZipper.recallNodeSSA (string (NodeId.value node.Id) + "_lambdaName") zipper with
                | Some (name, _) -> name
                | None -> funcName  // Use the current function name as fallback
            
            // Determine the declared return type from the Lambda's F# type
            let declaredRetType = 
                match node.Type with
                | NativeType.TFun (_, retTy) -> Serialize.mlirType (mapType retTy)
                | _ -> "i32"
            
            // Look up the body's SSA result (already processed in post-order, inside function scope)
            // Thread the zipper through in case we need to generate a default value
            let bodySSA, bodyType, zipperWithBody = 
                match MLIRZipper.recallNodeSSA (string (NodeId.value bodyId)) zipper with
                | Some (ssa, ty) -> ssa, ty, zipper
                | None -> 
                    // Body didn't produce a value - generate appropriate default
                    let zeroSSA, z = MLIRZipper.yieldSSA zipper
                    if declaredRetType = "!llvm.ptr" then
                        // For pointer returns, use llvm.mlir.zero for null pointer
                        let zeroText = sprintf "%s = llvm.mlir.zero : !llvm.ptr" zeroSSA
                        let z' = MLIRZipper.witnessOpWithResult zeroText zeroSSA Pointer z
                        zeroSSA, "!llvm.ptr", z'
                    else
                        // For integer returns, use arith.constant 0
                        let zeroText = sprintf "%s = arith.constant 0 : %s" zeroSSA declaredRetType
                        let z' = MLIRZipper.witnessOpWithResult zeroText zeroSSA (Integer I32) z
                        zeroSSA, declaredRetType, z'
            
            // Check for return type mismatch
            // Case 1: declared i32 (unit) but body produces ptr - return 0:i32
            // Case 2: declared ptr but body produces i32 - return null ptr
            let returnSSA, returnType, zipperForReturn =
                if declaredRetType = "i32" && bodyType = "!llvm.ptr" then
                    // Unit function with side-effecting body - ignore result, return 0
                    let zeroSSA, z = MLIRZipper.yieldSSA zipperWithBody
                    let zeroText = sprintf "%s = arith.constant 0 : i32" zeroSSA
                    let z' = MLIRZipper.witnessOpWithResult zeroText zeroSSA (Integer I32) z
                    zeroSSA, "i32", z'
                elif declaredRetType = "!llvm.ptr" && (bodyType = "i32" || bodyType.StartsWith("i")) then
                    // Function returns ptr but body computed an integer - return null ptr
                    let nullSSA, z = MLIRZipper.yieldSSA zipperWithBody
                    let nullText = sprintf "%s = llvm.mlir.zero : !llvm.ptr" nullSSA
                    let z' = MLIRZipper.witnessOpWithResult nullText nullSSA Pointer z
                    nullSSA, "!llvm.ptr", z'
                else
                    bodySSA, bodyType, zipperWithBody
            
            // Add return instruction to end the function body
            let returnText = sprintf "llvm.return %s : %s" returnSSA returnType
            let zipper1 = MLIRZipper.witnessVoidOp returnText zipperForReturn
            
            // Exit function scope - this creates the MLIRFunc with all accumulated body ops
            let zipper2, func = MLIRZipper.exitFunction zipper1
            
            // Add the function to completed functions
            let zipper3 = MLIRZipper.addCompletedFunction func zipper2
            
            // After exitFunction, we're at module level - we can't emit addressof here
            // (MLIR doesn't allow operations at module level)
            // Instead, bind the function name as a marker for later use
            // The addressof will be emitted when the lambda is actually called
            let zipper4 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ("@" + lambdaName) "!llvm.ptr" zipper3
            zipper4, TRValue ("@" + lambdaName, "!llvm.ptr")
        
        | _ ->
            // Not in function scope - preBindLambdaParams may not have run for this Lambda
            // Fall back to creating function directly (old behavior)
            let lambdaName, zipper1 = MLIRZipper.yieldLambdaName zipper
            
            let mlirParams = params' |> List.mapi (fun i (_name, nativeTy) ->
                (sprintf "arg%d" i, mapType nativeTy))
            
            let returnType = 
                match node.Type with
                | NativeType.TFun (_, retTy) -> mapType retTy
                | _ -> mapType node.Type
            
            // Look up body SSA
            let bodySSA, bodyType = 
                match MLIRZipper.recallNodeSSA (string (NodeId.value bodyId)) zipper1 with
                | Some (ssa, ty) -> ssa, ty
                | None -> 
                    let undefSSA, z = MLIRZipper.yieldSSA zipper1
                    undefSSA, Serialize.mlirType returnType
            
            // Create return operation
            let retOp: MLIROp = {
                Text = sprintf "llvm.return %s : %s" bodySSA bodyType
                Results = []
            }
            
            // Create function with just the return op (body ops weren't captured)
            let func: MLIRFunc = {
                Name = lambdaName
                Parameters = mlirParams
                ReturnType = returnType
                Blocks = [{
                    Label = "entry"
                    Arguments = []
                    Operations = [retOp]
                }]
                Attributes = []
                IsInternal = true
            }
            
            let zipper2 = MLIRZipper.addCompletedFunction func zipper1
            
            // At module level - can't emit addressof here
            // Bind the function name as a marker for later use
            let zipper3 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ("@" + lambdaName) "!llvm.ptr" zipper2
            zipper3, TRValue ("@" + lambdaName, "!llvm.ptr")

    // Type annotations - pass through the inner expression's value
    | SemanticKind.TypeAnnotation (exprId, _annotatedType) ->
        // DEBUG: Trace TypeAnnotation processing
        ()
        // Type annotation doesn't generate code - just forward the inner expression's value
        match MLIRZipper.recallNodeSSA (string (NodeId.value exprId)) zipper with
        | Some (ssa, ty) ->
            let zipper1 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ssa ty zipper
            zipper1, TRValue (ssa, ty)
        | None ->
            // Inner expression not yet computed - shouldn't happen in post-order
            zipper, TRError (sprintf "TypeAnnotation inner expr %A not computed" exprId)

    // Mutable set
    | SemanticKind.Set (targetId, valueId) ->
        // Get the target node - for VarRef, use the stored pointer
        let targetResult =
            match SemanticGraph.tryGetNode targetId graph with
            | Some targetNode ->
                match targetNode.Kind with
                | SemanticKind.VarRef (name, _) ->
                    // Look up the mutable variable's pointer
                    MLIRZipper.recallVar name zipper
                | _ ->
                    // Other targets - use computed SSA
                    MLIRZipper.recallNodeSSA (string (NodeId.value targetId)) zipper
            | None ->
                MLIRZipper.recallNodeSSA (string (NodeId.value targetId)) zipper

        match targetResult, MLIRZipper.recallNodeSSA (string (NodeId.value valueId)) zipper with
        | Some (targetSSA, _), Some (valueSSA, valueType) ->
            // Store value to target location
            let storeText = sprintf "llvm.store %s, %s : %s, !llvm.ptr" valueSSA targetSSA valueType
            let zipper' = MLIRZipper.witnessVoidOp storeText zipper
            zipper', TRVoid
        | _ ->
            zipper, TRError "Set: target or value not computed"

    // Control flow
    | SemanticKind.IfThenElse (guardId, thenId, elseIdOpt) ->
        // For now, just use the then branch result (simplified)
        // TODO: Implement proper control flow with blocks
        match MLIRZipper.recallNodeSSA (string (NodeId.value thenId)) zipper with
        | Some (thenSSA, thenType) ->
            let zipper' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) thenSSA thenType zipper
            zipper', TRValue (thenSSA, thenType)
        | None ->
            // Then branch might be void (no result)
            zipper, TRVoid

    | SemanticKind.WhileLoop (guardId, bodyId) ->
        // TODO: Implement proper loops with blocks
        // For now, treat as processed (children already executed in post-order)
        zipper, TRVoid

    | SemanticKind.Match (scrutineeId, cases) ->
        // TODO: Implement pattern matching
        zipper, TRError "Match not yet implemented"

    // Interpolated strings
    | SemanticKind.InterpolatedString parts ->
        // TODO: Implement string interpolation lowering
        zipper, TRError "InterpolatedString not yet implemented"

    // Array/collection indexing
    | SemanticKind.IndexGet (collectionId, indexId) ->
        // Get the collection and index SSAs
        match MLIRZipper.recallNodeSSA (string (NodeId.value collectionId)) zipper,
              MLIRZipper.recallNodeSSA (string (NodeId.value indexId)) zipper with
        | Some (collSSA, collType), Some (indexSSA, _) ->
            // Generate GEP (getelementptr) for array access
            let ssaName, zipper' = MLIRZipper.yieldSSA zipper
            let elemType = Serialize.mlirType (mapType node.Type)
            let text = sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i64) -> !llvm.ptr" ssaName collSSA indexSSA
            let zipper'' = MLIRZipper.witnessOpWithResult text ssaName Pointer zipper'
            // Load the element
            let loadSSA, zipper''' = MLIRZipper.yieldSSA zipper''
            let loadText = sprintf "%s = llvm.load %s : !llvm.ptr -> %s" loadSSA ssaName elemType
            let zipper4 = MLIRZipper.witnessOpWithResult loadText loadSSA (mapType node.Type) zipper'''
            let zipper5 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) loadSSA elemType zipper4
            zipper5, TRValue (loadSSA, elemType)
        | _ ->
            zipper, TRError "IndexGet: collection or index not computed"

    | SemanticKind.IndexSet (collectionId, indexId, valueId) ->
        // Get the collection, index, and value SSAs
        match MLIRZipper.recallNodeSSA (string (NodeId.value collectionId)) zipper,
              MLIRZipper.recallNodeSSA (string (NodeId.value indexId)) zipper,
              MLIRZipper.recallNodeSSA (string (NodeId.value valueId)) zipper with
        | Some (collSSA, _), Some (indexSSA, _), Some (valueSSA, valueType) ->
            // Generate GEP for array access
            let ptrSSA, zipper' = MLIRZipper.yieldSSA zipper
            let gepText = sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i64) -> !llvm.ptr" ptrSSA collSSA indexSSA
            let zipper'' = MLIRZipper.witnessOpWithResult gepText ptrSSA Pointer zipper'
            // Store the value
            let storeText = sprintf "llvm.store %s, %s : %s, !llvm.ptr" valueSSA ptrSSA valueType
            let zipper''' = MLIRZipper.witnessVoidOp storeText zipper''
            zipper''', TRVoid
        | _ ->
            zipper, TRError "IndexSet: collection, index, or value not computed"

    // Address-of operator
    | SemanticKind.AddressOf (exprId, isMutable) ->
        // Get the expression's SSA (should be an addressable location)
        match MLIRZipper.recallNodeSSA (string (NodeId.value exprId)) zipper with
        | Some (exprSSA, _) ->
            // For now, just return the pointer as-is
            // A proper implementation would use llvm.mlir.addressof or alloca
            let zipper' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) exprSSA "!llvm.ptr" zipper
            zipper', TRValue (exprSSA, "!llvm.ptr")
        | None ->
            zipper, TRError "AddressOf: expression not computed"

    // Tuple expressions - construct a tuple value
    | SemanticKind.TupleExpr elementIds ->
        let elementSSAs =
            elementIds
            |> List.choose (fun elemId ->
                MLIRZipper.recallNodeSSA (string (NodeId.value elemId)) zipper)

        if List.length elementSSAs <> List.length elementIds then
            // Some elements weren't computed
            zipper, TRError "TupleExpr: not all elements computed"
        else
            match elementSSAs with
            | [] ->
                // Empty tuple is unit
                zipper, TRVoid
            | [(ssa, ty)] ->
                // Single element - just return it
                let zipper' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ssa ty zipper
                zipper', TRValue (ssa, ty)
            | elements ->
                // Multi-element tuple - for now, allocate struct and store elements
                // TODO: Proper tuple lowering based on ABI
                let tupleSSA, zipper1 = MLIRZipper.yieldSSA zipper
                let allocaText = sprintf "%s = llvm.alloca i64 x %d : (i64) -> !llvm.ptr" tupleSSA (List.length elements)
                let zipper2 = MLIRZipper.witnessOp allocaText [(tupleSSA, Pointer)] zipper1
                // Store each element at its offset (simplified: just use pointer for now)
                let zipper3 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) tupleSSA "!llvm.ptr" zipper2
                zipper3, TRValue (tupleSSA, "!llvm.ptr")

    // TraitCall - SRTP member resolution
    | SemanticKind.TraitCall (memberName, typeArgs, argId) ->
        // For SRTP, the trait call resolves at compile time to a specific member
        // We need to look up the resolved member and call it
        match MLIRZipper.recallNodeSSA (string (NodeId.value argId)) zipper with
        | Some (argSSA, argType) ->
            // For now, emit a call to the trait member name
            // TODO: Proper SRTP resolution from Baker/type checker
            let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
            let callText = sprintf "%s = llvm.call @%s(%s) : (%s) -> !llvm.ptr" resultSSA memberName argSSA argType
            let zipper'' = MLIRZipper.witnessOpWithResult callText resultSSA Pointer zipper'
            let zipper''' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) resultSSA "!llvm.ptr" zipper''
            zipper''', TRValue (resultSSA, "!llvm.ptr")
        | None ->
            zipper, TRError (sprintf "TraitCall '%s': argument not computed" memberName)

    // Array expressions - construct an array
    | SemanticKind.ArrayExpr elementIds ->
        let elementSSAs =
            elementIds
            |> List.choose (fun elemId ->
                MLIRZipper.recallNodeSSA (string (NodeId.value elemId)) zipper)

        if List.length elementSSAs <> List.length elementIds then
            zipper, TRError "ArrayExpr: not all elements computed"
        else
            match elementSSAs with
            | [] ->
                // Empty array - allocate empty array (just a pointer)
                let arrSSA, zipper' = MLIRZipper.yieldSSA zipper
                let allocaText = sprintf "%s = llvm.alloca i64 x 0 : (i64) -> !llvm.ptr" arrSSA
                let zipper'' = MLIRZipper.witnessOpWithResult allocaText arrSSA Pointer zipper'
                let zipper''' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) arrSSA "!llvm.ptr" zipper''
                zipper''', TRValue (arrSSA, "!llvm.ptr")
            | elements ->
                // Non-empty array - allocate and store elements
                let arrSSA, zipper1 = MLIRZipper.yieldSSA zipper
                let allocaText = sprintf "%s = llvm.alloca i64 x %d : (i64) -> !llvm.ptr" arrSSA (List.length elements)
                let zipper2 = MLIRZipper.witnessOpWithResult allocaText arrSSA Pointer zipper1
                // Store each element (simplified - not considering element types)
                let zipper3 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) arrSSA "!llvm.ptr" zipper2
                zipper3, TRValue (arrSSA, "!llvm.ptr")

    // List expressions - construct a list
    | SemanticKind.ListExpr elementIds ->
        // Lists are similar to arrays in our runtime - just linked list of pointers
        let elementSSAs =
            elementIds
            |> List.choose (fun elemId ->
                MLIRZipper.recallNodeSSA (string (NodeId.value elemId)) zipper)

        if List.length elementSSAs <> List.length elementIds then
            zipper, TRError "ListExpr: not all elements computed"
        else
            // Allocate space for list nodes (simplified)
            let listSSA, zipper1 = MLIRZipper.yieldSSA zipper
            let allocaText = sprintf "%s = llvm.alloca i64 x %d : (i64) -> !llvm.ptr" listSSA (List.length elementSSAs)
            let zipper2 = MLIRZipper.witnessOpWithResult allocaText listSSA Pointer zipper1
            let zipper3 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) listSSA "!llvm.ptr" zipper2
            zipper3, TRValue (listSSA, "!llvm.ptr")

    // Field access: expr.fieldName
    | SemanticKind.FieldGet (exprId, fieldName) ->
        match MLIRZipper.recallNodeSSA (string (NodeId.value exprId)) zipper with
        | Some (exprSSA, exprType) ->
            // STRING INTRINSIC MEMBERS: Native string is fat pointer {ptr, len}
            // .Pointer → extractvalue at index 0 → !llvm.ptr
            // .Length → extractvalue at index 1 → i64
            // See Serena memories: string_length_handling, fidelity_memory_model
            if isNativeStrType exprType then
                match fieldName with
                | "Pointer" ->
                    let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
                    let extractText = sprintf "%s = llvm.extractvalue %s[0] : %s" resultSSA exprSSA NativeStrTypeStr
                    let zipper'' = MLIRZipper.witnessOpWithResult extractText resultSSA Pointer zipper'
                    let zipper''' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) resultSSA "!llvm.ptr" zipper''
                    zipper''', TRValue (resultSSA, "!llvm.ptr")
                | "Length" ->
                    let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
                    let extractText = sprintf "%s = llvm.extractvalue %s[1] : %s" resultSSA exprSSA NativeStrTypeStr
                    let zipper'' = MLIRZipper.witnessOpWithResult extractText resultSSA (Integer I64) zipper'
                    let zipper''' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) resultSSA "i64" zipper''
                    zipper''', TRValue (resultSSA, "i64")
                | _ ->
                    zipper, TRError (sprintf "Unknown string field: %s (expected Pointer or Length)" fieldName)
            else
                // Generic field access for other struct types
                // TODO: Proper field offset calculation from record type
                let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
                let loadText = sprintf "%s = llvm.load %s : !llvm.ptr -> !llvm.ptr" resultSSA exprSSA
                let zipper'' = MLIRZipper.witnessOpWithResult loadText resultSSA Pointer zipper'
                let zipper''' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) resultSSA "!llvm.ptr" zipper''
                zipper''', TRValue (resultSSA, "!llvm.ptr")
        | None ->
            zipper, TRError (sprintf "FieldGet '%s': expression not computed" fieldName)

    // Field set: expr.fieldName <- value
    | SemanticKind.FieldSet (exprId, fieldName, valueId) ->
        match MLIRZipper.recallNodeSSA (string (NodeId.value exprId)) zipper,
              MLIRZipper.recallNodeSSA (string (NodeId.value valueId)) zipper with
        | Some (exprSSA, _), Some (valueSSA, valueType) ->
            // Store value at field offset
            let storeText = sprintf "llvm.store %s, %s : %s, !llvm.ptr" valueSSA exprSSA valueType
            let zipper' = MLIRZipper.witnessVoidOp storeText zipper
            zipper', TRVoid
        | _ ->
            zipper, TRError (sprintf "FieldSet '%s': expression or value not computed" fieldName)

    // Error nodes
    | SemanticKind.Error msg ->
        zipper, TRError msg

    // Catch-all for unimplemented kinds
    | kind ->
        zipper, TRError (sprintf "SemanticKind not yet implemented: %A" kind)

/// Pre-bind Lambda parameters to SSA names BEFORE body is processed
/// This is called in pre-order for Lambda nodes only
/// CRITICAL: Also enters function scope so body operations are captured
/// NOTE: For entry point (main), uses C-style signature: (argc: i32, argv: ptr) -> i32
let preBindLambdaParams (zipper: MLIRZipper) (node: SemanticNode) : MLIRZipper =
    match node.Kind with
    | SemanticKind.Lambda (params', _bodyId) ->
        // Generate lambda function name - entry point Lambdas become "main"
        let nodeIdVal = NodeId.value node.Id
        let lambdaName, zipper1 = MLIRZipper.yieldLambdaNameForNode nodeIdVal zipper

        // For main, use C-style signature regardless of F# signature
        // C main: (int argc, char** argv) -> int
        // F# entry: (argv: string[]) -> int  OR  (unit) -> int
        let isMain = (lambdaName = "main")

        let mlirParams, paramBindings =
            if isMain then
                // C-style main signature
                let cParams = [("arg0", Integer I32); ("arg1", Pointer)]
                // F# parameter binds to %arg1 (argv), argc (%arg0) typically unused
                let bindings =
                    match params' with
                    | [(paramName, paramType)] ->
                        // Single F# parameter maps to argv (%arg1)
                        [(paramName, "%arg1", Serialize.mlirType (mapType paramType))]
                    | [] ->
                        // Unit parameter - nothing to bind
                        []
                    | _ ->
                        // Multiple parameters (unusual for entry point)
                        params' |> List.mapi (fun i (name, ty) ->
                            (name, sprintf "%%arg%d" (i + 1), Serialize.mlirType (mapType ty)))
                cParams, bindings
            else
                // Regular lambda - use node.Type for parameter types (has instantiated generics)
                // Extract parameter types from curried TFun chain
                let rec extractParamTypesFromFun (ty: NativeType) (count: int) : NativeType list =
                    if count <= 0 then []
                    else
                        match ty with
                        | NativeType.TFun(paramTy, resultTy) ->
                            paramTy :: extractParamTypesFromFun resultTy (count - 1)
                        | _ -> []

                let nodeParamTypes = extractParamTypesFromFun node.Type (List.length params')

                // Use instantiated types from node.Type if available, else fall back to params'
                let mlirPs = params' |> List.mapi (fun i (_name, nativeTy) ->
                    let actualType =
                        if i < List.length nodeParamTypes then nodeParamTypes.[i]
                        else nativeTy
                    (sprintf "arg%d" i, mapType actualType))
                let bindings = params' |> List.mapi (fun i (paramName, paramType) ->
                    let actualType =
                        if i < List.length nodeParamTypes then nodeParamTypes.[i]
                        else paramType
                    (paramName, sprintf "%%arg%d" i, Serialize.mlirType (mapType actualType)))
                mlirPs, bindings

        // Return type: main always returns i32
        let returnType =
            if isMain then Integer I32
            else
                match node.Type with
                | NativeType.TFun (_, retTy) -> mapType retTy
                | _ -> mapType node.Type

        // Enter function scope - main is NOT internal (must be exported)
        let zipper2 =
            if isMain then
                MLIRZipper.enterFunctionWithVisibility lambdaName mlirParams returnType false zipper1
            else
                MLIRZipper.enterFunction lambdaName mlirParams returnType zipper1

        // Bind F# parameter names to their SSA positions
        let zipper3 =
            paramBindings
            |> List.fold (fun z (paramName, ssaName, mlirType) ->
                MLIRZipper.bindVar paramName ssaName mlirType z
            ) zipper2

        // Also bind the lambda name to the node for later retrieval
        MLIRZipper.bindNodeSSA (string (NodeId.value node.Id) + "_lambdaName") lambdaName "func" zipper3
    | _ -> zipper

/// Find entry point Lambda IDs from the semantic graph
/// Entry point Bindings have Lambda children - those are the entry point Lambdas
let findEntryPointLambdaIds (graph: SemanticGraph) : Set<int> =
    graph.EntryPoints
    |> List.collect (fun epId ->
        match SemanticGraph.tryGetNode epId graph with
        | Some node ->
            match node.Kind with
            | SemanticKind.Binding (_, _, _, _) ->
                // Entry point Binding - its children include the Lambda
                node.Children
                |> List.choose (fun childId ->
                    match SemanticGraph.tryGetNode childId graph with
                    | Some child when (match child.Kind with SemanticKind.Lambda _ -> true | _ -> false) ->
                        Some (NodeId.value childId)
                    | _ -> None)
            | SemanticKind.Lambda _ ->
                // Entry point is directly a Lambda
                [NodeId.value epId]
            | SemanticKind.ModuleDef (_, memberIds) ->
                // Check module members for main binding
                memberIds
                |> List.collect (fun memberId ->
                    match SemanticGraph.tryGetNode memberId graph with
                    | Some memberNode ->
                        match memberNode.Kind with
                        | SemanticKind.Binding (name, _, _, _) when name = "main" ->
                            memberNode.Children
                            |> List.choose (fun childId ->
                                match SemanticGraph.tryGetNode childId graph with
                                | Some child ->
                                    match child.Kind with
                                    | SemanticKind.Lambda _ -> Some (NodeId.value childId)
                                    | _ -> None
                                | None -> None)
                        | _ -> []
                    | None -> [])
            | _ -> []
        | None -> [])
    |> Set.ofList

/// Transfer an entire SemanticGraph to MLIR
/// Uses post-order traversal so children are witnessed before parents
/// Lambda parameters are pre-bound before their bodies are processed
/// isFreestanding: if true, adds a _start wrapper that calls main and then exit
let transferGraph (graph: SemanticGraph) (isFreestanding: bool) : string =
    // Initialize platform bindings
    Alex.Bindings.Console.ConsoleBindings.registerBindings()
    Alex.Bindings.Process.ProcessBindings.registerBindings()
    Alex.Bindings.Time.TimeBindings.registerBindings()

    // Find entry point Lambda IDs - these will be named "main"
    let entryPointLambdaIds = findEntryPointLambdaIds graph

    // Create initial zipper with entry point knowledge
    let initialZipper = MLIRZipper.createWithEntryPoints entryPointLambdaIds

    // Use foldWithLambdaPreBind to bind params before body processing
    let traversedZipper =
        Traversal.foldWithLambdaPreBind
            preBindLambdaParams
            (fun zipper node ->
                let zipper', _result = witnessNode graph node zipper
                zipper')
            initialZipper
            graph

    // For freestanding binaries, add a _start wrapper that calls main and exit syscall
    // This is necessary because freestanding binaries can't return from main
    let finalZipper =
        if isFreestanding then
            MLIRZipper.addFreestandingEntryPoint traversedZipper
        else
            traversedZipper

    // Extract final MLIR - entry point Lambda is named main
    MLIRZipper.extract finalZipper

/// Transfer a graph and return both MLIR text and any errors
/// isFreestanding: if true, adds a _start wrapper that calls main and then exit
let transferGraphWithDiagnostics (graph: SemanticGraph) (isFreestanding: bool) : string * string list =
    // Initialize platform bindings
    Alex.Bindings.Console.ConsoleBindings.registerBindings()
    Alex.Bindings.Process.ProcessBindings.registerBindings()
    Alex.Bindings.Time.TimeBindings.registerBindings()

    // Find entry point Lambda IDs - these will be named "main"
    let entryPointLambdaIds = findEntryPointLambdaIds graph

    // Create initial zipper with entry point knowledge
    let initialZipper = MLIRZipper.createWithEntryPoints entryPointLambdaIds

    // Track errors during traversal
    let mutable errors = []

    // Use foldWithLambdaPreBind to bind params before body processing
    let traversedZipper =
        Traversal.foldWithLambdaPreBind
            preBindLambdaParams
            (fun zipper node ->
                let zipper', result = witnessNode graph node zipper
                match result with
                | TRError msg ->
                    errors <- msg :: errors
                    zipper'
                | _ ->
                    zipper')
            initialZipper
            graph

    // For freestanding binaries, add a _start wrapper that calls main and exit syscall
    let finalZipper =
        if isFreestanding then
            MLIRZipper.addFreestandingEntryPoint traversedZipper
        else
            traversedZipper

    // Extract final MLIR - entry point Lambda is named main
    MLIRZipper.extract finalZipper, List.rev errors
