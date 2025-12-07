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
        // Check for nativeptr early - it may not have HasTypeDefinition = true
        // but its string representation contains "nativeptr"
        let typeStr = ftype.ToString()
        if typeStr.Contains("nativeptr") then
            "!llvm.ptr"
        // Check for unit first, before other patterns
        elif ftype.HasTypeDefinition && ftype.TypeDefinition.DisplayName = "unit" then
            "()"
        elif ftype.IsAbbreviation then
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

            // Alloy StackBuffer - stack-allocated byte buffer, represented as pointer
            | Some name when name.Contains("Alloy.Memory.StackBuffer") -> "!llvm.ptr"

            // Alloy Span/ReadOnlySpan - memory spans, represented as pointer
            | Some name when name.Contains("Alloy.Memory.Span") -> "!llvm.ptr"
            | Some name when name.Contains("Alloy.Memory.ReadOnlySpan") -> "!llvm.ptr"

            // Alloy NativeStr - fat pointer struct (ptr + length)
            // Emitted as !llvm.struct<(ptr, i64)> per NativeString.fs docs
            | Some name when name.Contains("Alloy.NativeTypes.NativeStr") ||
                            name.Contains("NativeStr") ->
                "!llvm.struct<(ptr, i64)>"

            // Option type
            | Some name when name.StartsWith("Microsoft.FSharp.Core.FSharpOption") ->
                let innerType =
                    if ftype.GenericArguments.Count > 0 then
                        fsharpTypeToMLIR ftype.GenericArguments.[0]
                    else "i32"
                sprintf "!variant<Some: %s, None: ()>" innerType

            // Result type (from Alloy.Core or Microsoft.FSharp.Core)
            // Represented as struct<(tag: i32, payload1: i64, payload2: i64)>
            // Tag 0 = Ok, Tag 1 = Error
            | Some name when name.Contains("Result`2") || name.Contains("Result<") ->
                "!llvm.struct<(i32, i64, i64)>"

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

            // Fallback - print type info for debugging (comment out in production)
            | Some name ->
                // Check if it looks like a pointer/buffer type by name
                if name.Contains("nativeptr") || name.Contains("Ptr") ||
                   name.Contains("Buffer") || name.Contains("Span") then
                    "!llvm.ptr"
                else
                    "i32"
            | None -> "i32"
        // Generic parameters (SRTP type variables) - check if pointer-like
        elif ftype.IsGenericParameter then
            // For now, assume SRTP buffer types are pointers
            // This covers common patterns in Alloy library
            "!llvm.ptr"
        else "i32"
    with _ -> "i32"

/// Extract return type from a function type (stored in PSG node.Type)
/// Walks through curried function type to get the final return type
let getReturnTypeFromFSharpType (ftype: FSharpType) : string =
    try
        let rec getReturnType (t: FSharpType) =
            if t.IsFunctionType && t.GenericArguments.Count >= 2 then
                getReturnType t.GenericArguments.[1]
            else
                t
        fsharpTypeToMLIR (getReturnType ftype)
    with _ -> "i32"

/// Extract parameter types from a function type (stored in PSG node.Type)
/// Returns list of MLIR types for each curried parameter
let getParamTypesFromFSharpType (ftype: FSharpType) : string list =
    try
        let rec extractParamTypes (funcType: FSharpType) (acc: string list) =
            if funcType.IsFunctionType && funcType.GenericArguments.Count >= 2 then
                let argType = funcType.GenericArguments.[0]
                let paramType = fsharpTypeToMLIR argType
                extractParamTypes funcType.GenericArguments.[1] (paramType :: acc)
            else
                List.rev acc
        extractParamTypes ftype []
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
