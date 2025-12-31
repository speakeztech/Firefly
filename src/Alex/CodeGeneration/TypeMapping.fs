/// TypeMapping - FNCS NativeType to MLIR type conversion
///
/// Maps FNCS native types to their MLIR representations.
/// Handles primitives, functions, tuples, options, lists, and arrays.
///
/// FNCS-native: Uses NativeType from FSharp.Native.Compiler.Checking.Native
module Alex.CodeGeneration.TypeMapping

open FSharp.Native.Compiler.Checking.Native.NativeTypes

/// Convert a FNCS NativeType to its MLIR representation
let rec nativeTypeToMLIR (ty: NativeType) : string =
    match ty with
    // Function types: domain -> range
    | NativeType.TFun(domain, range) ->
        let paramType = nativeTypeToMLIR domain
        let retType = nativeTypeToMLIR range
        sprintf "(%s) -> %s" paramType retType

    // Tuple types
    | NativeType.TTuple(elements, _isStruct) ->
        let elemTypes =
            elements
            |> List.map nativeTypeToMLIR
            |> String.concat ", "
        sprintf "tuple<%s>" elemTypes

    // Byref types (pointer-like)
    | NativeType.TByref(_, _) ->
        "!llvm.ptr"

    // Native pointers
    | NativeType.TNativePtr _ ->
        "!llvm.ptr"

    // Type applications (e.g., int, string, option<'T>, list<'T>)
    | NativeType.TApp(conRef, args) ->
        mapTypeApp conRef args

    // Type variables (generic parameters)
    | NativeType.TVar _ ->
        // For now, assume generic types are pointers (common in Alloy)
        "!llvm.ptr"

    // Forall types (polymorphic)
    | NativeType.TForall(_, body) ->
        // Instantiate body (generic erased at runtime)
        nativeTypeToMLIR body

    // Measure types (used for UMX phantom types)
    | NativeType.TMeasure _ ->
        // Measures are erased at runtime, use the underlying type
        "i32"

    // Anonymous record types
    | NativeType.TAnon fields ->
        let fieldTypes =
            fields
            |> List.map (fun (_, ty) -> nativeTypeToMLIR ty)
            |> String.concat ", "
        sprintf "!llvm.struct<(%s)>" fieldTypes

    // Record types
    | NativeType.TRecord(_, fields) ->
        let fieldTypes =
            fields
            |> List.map (fun (_, ty) -> nativeTypeToMLIR ty)
            |> String.concat ", "
        sprintf "!llvm.struct<(%s)>" fieldTypes

    // Union types (discriminated unions)
    | NativeType.TUnion(_, _cases) ->
        // For now, represent as tagged union (tag + max payload)
        "!llvm.struct<(i32, i64)>"

    // Error types (shouldn't reach code generation)
    | NativeType.TError _ ->
        "i32"

/// Map a type constructor application to MLIR
and mapTypeApp (conRef: TypeConRef) (args: NativeType list) : string =
    match conRef.Name with
    // Integral types
    | "int" | "int32" | "Int32" -> "i32"
    | "int64" | "Int64" -> "i64"
    | "int16" | "Int16" -> "i16"
    | "byte" | "Byte" | "uint8" -> "i8"
    | "sbyte" | "SByte" | "int8" -> "i8"
    | "uint32" | "UInt32" -> "i32"
    | "uint64" | "UInt64" -> "i64"
    | "uint16" | "UInt16" -> "i16"

    // Boolean
    | "bool" | "Boolean" -> "i1"

    // Floating point
    | "float32" | "Single" -> "f32"
    | "float" | "float64" | "Double" -> "f64"

    // Strings - native string is fat pointer (ptr + length)
    | "string" | "String" -> "!llvm.struct<(ptr, i64)>"

    // Native pointers
    | "nativeint" | "IntPtr" -> "!llvm.ptr"
    | "unativeint" | "UIntPtr" -> "!llvm.ptr"
    | "nativeptr" | "Ptr" -> "!llvm.ptr"

    // Char (Unicode codepoint)
    | "char" | "Char" -> "i32"

    // Unit / Void
    | "unit" | "Unit" | "Void" -> "()"

    // Option type - value type in native (tag + payload)
    | "option" | "Option" | "voption" | "ValueOption" ->
        let innerType =
            match args with
            | [arg] -> nativeTypeToMLIR arg
            | _ -> "i32"
        sprintf "!llvm.struct<(i1, %s)>" innerType  // tag + payload

    // Result type
    | "Result" ->
        "!llvm.struct<(i32, i64, i64)>"  // tag + ok_payload + error_payload

    // List type
    | "list" | "List" ->
        "!llvm.ptr"  // Lists are pointers to cons cells

    // Array type
    | "array" | "Array" ->
        let elemType =
            match args with
            | [arg] -> nativeTypeToMLIR arg
            | _ -> "i32"
        sprintf "memref<?x%s>" elemType

    // Memory types from Alloy
    | "StackBuffer" | "Span" | "ReadOnlySpan" ->
        "!llvm.ptr"

    // Default fallback
    | _ ->
        // Check if it's a pointer-like type by name
        if conRef.Name.Contains("ptr") || conRef.Name.Contains("Ptr") ||
           conRef.Name.Contains("Buffer") || conRef.Name.Contains("Span") then
            "!llvm.ptr"
        else
            "i32"

/// Extract return type from a function type
/// Walks through curried function type to get the final return type
let getReturnType (ty: NativeType) : string =
    let rec getReturn t =
        match t with
        | NativeType.TFun(_, range) -> getReturn range
        | _ -> t
    nativeTypeToMLIR (getReturn ty)

/// Extract parameter types from a function type
/// Returns list of MLIR types for each curried parameter
let getParamTypes (ty: NativeType) : string list =
    let rec extractParams funcType acc =
        match funcType with
        | NativeType.TFun(domain, range) ->
            let paramType = nativeTypeToMLIR domain
            extractParams range (paramType :: acc)
        | _ ->
            List.rev acc
    extractParams ty []

/// Check if a type is a primitive MLIR type (i32, i64, f32, etc.)
let isPrimitive (mlirType: string) : bool =
    match mlirType with
    | "i1" | "i8" | "i16" | "i32" | "i64" -> true
    | "f32" | "f64" -> true
    | "()" -> true
    | _ -> false

/// Check if a type is an integer type
let isInteger (mlirType: string) : bool =
    match mlirType with
    | "i1" | "i8" | "i16" | "i32" | "i64" -> true
    | _ -> false

/// Check if a type is a floating-point type
let isFloat (mlirType: string) : bool =
    match mlirType with
    | "f32" | "f64" -> true
    | _ -> false

/// Check if a type is a pointer type
let isPointer (mlirType: string) : bool =
    mlirType = "!llvm.ptr" || mlirType.StartsWith("!llvm.ptr")

/// Get the bit width of an integer type
let integerBitWidth (mlirType: string) : int option =
    match mlirType with
    | "i1" -> Some 1
    | "i8" -> Some 8
    | "i16" -> Some 16
    | "i32" -> Some 32
    | "i64" -> Some 64
    | _ -> None

/// Get the appropriate zero constant for a type
let zeroConstant (mlirType: string) : string =
    match mlirType with
    | "i1" -> "0"
    | "i8" | "i16" | "i32" | "i64" -> "0"
    | "f32" -> "0.0"
    | "f64" -> "0.0"
    | _ -> "0"

/// Get the appropriate one constant for a type
let oneConstant (mlirType: string) : string =
    match mlirType with
    | "i1" -> "1"
    | "i8" | "i16" | "i32" | "i64" -> "1"
    | "f32" -> "1.0"
    | "f64" -> "1.0"
    | _ -> "1"
