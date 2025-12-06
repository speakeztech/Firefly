/// TypeMapping - F# to MLIR type conversion
///
/// Maps F# types (from FCS) to their MLIR representations.
/// Handles primitives, functions, tuples, options, lists, and arrays.
///
/// Extracted from Core.MLIR.Emitter to support the Alex architecture.
module Alex.CodeGeneration.TypeMapping

open FSharp.Compiler.Symbols

/// Convert an F# type to its MLIR representation
let rec fsharpTypeToMLIR (ftype: FSharpType) : string =
    try
        if ftype.IsAbbreviation then
            fsharpTypeToMLIR ftype.AbbreviatedType
        elif ftype.IsFunctionType then
            let args = ftype.GenericArguments
            if args.Count >= 2 then
                let paramType = fsharpTypeToMLIR args.[0]
                let retType = fsharpTypeToMLIR args.[1]
                sprintf "(%s) -> %s" paramType retType
            else "() -> i32"
        elif ftype.IsTupleType then
            let elemTypes =
                ftype.GenericArguments
                |> Seq.map fsharpTypeToMLIR
                |> String.concat ", "
            sprintf "tuple<%s>" elemTypes
        elif ftype.HasTypeDefinition then
            match ftype.TypeDefinition.TryFullName with
            // Integral types
            | Some "System.Int32" -> "i32"
            | Some "System.Int64" -> "i64"
            | Some "System.Int16" -> "i16"
            | Some "System.Byte" -> "i8"
            | Some "System.SByte" -> "i8"
            | Some "System.UInt32" -> "i32"  // Unsigned same size
            | Some "System.UInt64" -> "i64"
            | Some "System.UInt16" -> "i16"

            // Boolean
            | Some "System.Boolean" -> "i1"

            // Floating point
            | Some "System.Single" -> "f32"
            | Some "System.Double" -> "f64"

            // Strings and pointers
            | Some "System.String" -> "!llvm.ptr"
            | Some "System.IntPtr" | Some "System.UIntPtr" -> "!llvm.ptr"
            | Some "System.Char" -> "i32"  // Unicode codepoint

            // Unit / Void
            | Some "System.Void" | Some "Microsoft.FSharp.Core.unit" -> "()"

            // Native pointer
            | Some name when name.StartsWith("Microsoft.FSharp.Core.nativeptr") -> "!llvm.ptr"

            // Option type
            | Some name when name.StartsWith("Microsoft.FSharp.Core.FSharpOption") ->
                let innerType =
                    if ftype.GenericArguments.Count > 0 then
                        fsharpTypeToMLIR ftype.GenericArguments.[0]
                    else "i32"
                sprintf "!variant<Some: %s, None: ()>" innerType

            // List type
            | Some name when name.StartsWith("Microsoft.FSharp.Collections.FSharpList") ->
                "!llvm.ptr"  // Lists are pointers to cons cells

            // Array type
            | Some _ when ftype.TypeDefinition.IsArrayType ->
                let elemType =
                    if ftype.GenericArguments.Count > 0 then
                        fsharpTypeToMLIR ftype.GenericArguments.[0]
                    else "i32"
                sprintf "memref<?x%s>" elemType

            // Fallback
            | Some _ -> "i32"
            | None -> "i32"
        else "i32"
    with _ -> "i32"

/// Get the MLIR return type for a function/method
let getFunctionReturnType (mfv: FSharpMemberOrFunctionOrValue) : string =
    try
        fsharpTypeToMLIR mfv.ReturnParameter.Type
    with _ -> "i32"

/// Get MLIR parameter types for a function/method
/// Returns list of (paramName, mlirType) pairs
let getFunctionParamTypes (mfv: FSharpMemberOrFunctionOrValue) : (string * string) list =
    try
        mfv.CurriedParameterGroups
        |> Seq.collect id
        |> Seq.map (fun p ->
            let name = p.DisplayName
            let mlirType = fsharpTypeToMLIR p.Type
            (name, mlirType))
        |> Seq.toList
    with _ -> []

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
