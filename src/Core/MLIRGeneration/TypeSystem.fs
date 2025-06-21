module Core.MLIRGeneration.TypeSystem

open Dabbit.Parsing.OakAst

/// Represents the cost of converting between types
type ConversionCost =
    | NoCost        // Same types
    | Widening      // Safe widening conversion (i32 -> i64)
    | Narrowing     // Potentially unsafe narrowing (i64 -> i32)
    | IntToFloat    // Integer to floating point
    | FloatToInt    // Floating point to integer (potentially lossy)
    | Coercion      // Complex coercion (int -> string)
    | Impossible    // Cannot convert

/// Core type categories for MLIR
type MLIRTypeCategory =
    | IntegerCategory
    | FloatCategory
    | VoidCategory
    | MemoryRefCategory
    | FunctionCategory
    | StructCategory

/// Represents MLIR types with explicit structure
type MLIRType = {
    Category: MLIRTypeCategory
    Width: int option
    ElementType: MLIRType option
    Shape: int list
    Parameters: MLIRType list
    ReturnType: MLIRType option
    Fields: MLIRType list
}

/// Type creation functions that replace discriminated union constructors
module MLIRTypes =
    
    /// Creates an integer type with specified bit width
    let createInteger (width: int) : MLIRType = {
        Category = IntegerCategory
        Width = Some width
        ElementType = None
        Shape = []
        Parameters = []
        ReturnType = None
        Fields = []
    }
    
    /// Creates a floating point type with specified bit width
    let createFloat (width: int) : MLIRType = {
        Category = FloatCategory
        Width = Some width
        ElementType = None
        Shape = []
        Parameters = []
        ReturnType = None
        Fields = []
    }
    
    /// Creates a void type
    let createVoid () : MLIRType = {
        Category = VoidCategory
        Width = None
        ElementType = None
        Shape = []
        Parameters = []
        ReturnType = None
        Fields = []
    }
    
    /// Creates a memory reference type
    let createMemRef (elementType: MLIRType) (shape: int list) : MLIRType = {
        Category = MemoryRefCategory
        Width = None
        ElementType = Some elementType
        Shape = shape
        Parameters = []
        ReturnType = None
        Fields = []
    }
    
    /// Creates a function type
    let createFunction (parameters: MLIRType list) (returnType: MLIRType) : MLIRType = {
        Category = FunctionCategory
        Width = None
        ElementType = None
        Shape = []
        Parameters = parameters
        ReturnType = Some returnType
        Fields = []
    }
    
    /// Creates a struct type
    let createStruct (fields: MLIRType list) : MLIRType = {
        Category = StructCategory
        Width = None
        ElementType = None
        Shape = []
        Parameters = []
        ReturnType = None
        Fields = fields
    }

/// Core type conversion and analysis functions - declared as mutually recursive
let rec mapOakTypeToMLIR (oakType: OakType) : MLIRType =
    match oakType with
    | IntType -> MLIRTypes.createInteger 32
    | FloatType -> MLIRTypes.createFloat 32
    | BoolType -> MLIRTypes.createInteger 1
    | UnitType -> MLIRTypes.createVoid ()
    | StringType -> MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
    | ArrayType elemType -> MLIRTypes.createMemRef (mapOakTypeToMLIR elemType) []
    | FunctionType(paramTypes, returnType) ->
        let mappedParams = paramTypes |> List.map mapOakTypeToMLIR
        let mappedReturn = mapOakTypeToMLIR returnType
        MLIRTypes.createFunction mappedParams mappedReturn
    | StructType fields ->
        let fieldTypes = fields |> List.map (snd >> mapOakTypeToMLIR)
        MLIRTypes.createStruct fieldTypes
    | UnionType cases ->
        // For unions, use a tag (i8) and the largest possible payload
        MLIRTypes.createStruct [MLIRTypes.createInteger 8; MLIRTypes.createInteger 64]

/// Converts MLIR type to string representation
and mlirTypeToString (mlirType: MLIRType) : string =
    match mlirType.Category with
    | IntegerCategory ->
        match mlirType.Width with
        | Some width -> sprintf "i%d" width
        | None -> "i32"
    | FloatCategory ->
        match mlirType.Width with
        | Some width -> sprintf "f%d" width
        | None -> "f32"
    | VoidCategory -> "()"
    | MemoryRefCategory ->
        match mlirType.ElementType with
        | Some elementType ->
            let elementStr = mlirTypeToString elementType
            if mlirType.Shape.IsEmpty then
                sprintf "memref<?x%s>" elementStr
            else
                let shapeStr = mlirType.Shape |> List.map string |> String.concat "x"
                sprintf "memref<%sx%s>" shapeStr elementStr
        | None -> "memref<?xi8>"
    | FunctionCategory ->
        let inputStrs = mlirType.Parameters |> List.map mlirTypeToString |> String.concat ", "
        let outputStr = 
            match mlirType.ReturnType with
            | Some returnType -> mlirTypeToString returnType
            | None -> "()"
        sprintf "(%s) -> %s" inputStrs outputStr
    | StructCategory ->
        let fieldStrs = mlirType.Fields |> List.map mlirTypeToString |> String.concat ", "
        sprintf "!llvm.struct<(%s)>" fieldStrs

/// Gets the size in bytes of an MLIR type (for layout calculations)
and getTypeSize (mlirType: MLIRType) : int =
    match mlirType.Category with
    | IntegerCategory ->
        match mlirType.Width with
        | Some 1 -> 1
        | Some 8 -> 1
        | Some 16 -> 2
        | Some 32 -> 4
        | Some 64 -> 8
        | Some width -> (width + 7) / 8
        | None -> 4
    | FloatCategory ->
        match mlirType.Width with
        | Some 32 -> 4
        | Some 64 -> 8
        | Some width -> (width + 7) / 8
        | None -> 4
    | VoidCategory -> 0
    | MemoryRefCategory -> 8
    | FunctionCategory -> 8
    | StructCategory -> mlirType.Fields |> List.sumBy getTypeSize

/// Checks if a type requires heap allocation
and requiresHeapAllocation (mlirType: MLIRType) : bool =
    match mlirType.Category with
    | IntegerCategory | FloatCategory | VoidCategory -> false
    | MemoryRefCategory -> mlirType.Shape.IsEmpty
    | FunctionCategory -> false
    | StructCategory -> mlirType.Fields |> List.exists requiresHeapAllocation

/// Gets the alignment requirement for a type
and getTypeAlignment (mlirType: MLIRType) : int =
    match mlirType.Category with
    | IntegerCategory ->
        match mlirType.Width with
        | Some 1 -> 1
        | Some width when width <= 8 -> 1
        | Some width when width <= 16 -> 2
        | Some width when width <= 32 -> 4
        | Some width -> 8
        | None -> 4
    | FloatCategory ->
        match mlirType.Width with
        | Some 32 -> 4
        | Some 64 -> 8
        | None -> 4
        | Some _ -> 8
    | VoidCategory -> 1
    | MemoryRefCategory -> 8
    | FunctionCategory -> 8
    | StructCategory -> 
        if mlirType.Fields.IsEmpty then 1
        else mlirType.Fields |> List.map getTypeAlignment |> List.max

/// Enhanced type compatibility checking with detailed analysis
and areTypesCompatible (type1: MLIRType) (type2: MLIRType) : bool =
    if type1.Category <> type2.Category then false
    else
        match type1.Category with
        | IntegerCategory | FloatCategory -> type1.Width = type2.Width
        | VoidCategory -> true
        | MemoryRefCategory ->
            match type1.ElementType, type2.ElementType with
            | Some elem1, Some elem2 -> areTypesCompatible elem1 elem2 && type1.Shape = type2.Shape
            | None, None -> type1.Shape = type2.Shape
            | _ -> false
        | FunctionCategory ->
            type1.Parameters.Length = type2.Parameters.Length &&
            List.forall2 areTypesCompatible type1.Parameters type2.Parameters &&
            match type1.ReturnType, type2.ReturnType with
            | Some ret1, Some ret2 -> areTypesCompatible ret1 ret2
            | None, None -> true
            | _ -> false
        | StructCategory ->
            type1.Fields.Length = type2.Fields.Length &&
            List.forall2 areTypesCompatible type1.Fields type2.Fields

/// Type analysis and conversion functions
module TypeAnalysis =
    
    /// Gets compatible type conversions with their costs
    let getCompatibleConversions (mlirType: MLIRType) : (MLIRType * ConversionCost) list =
        match mlirType.Category with
        | IntegerCategory ->
            match mlirType.Width with
            | Some width ->
                [
                    // Widening conversions to larger integers
                    if width < 64 then yield (MLIRTypes.createInteger 64, Widening)
                    if width < 32 then yield (MLIRTypes.createInteger 32, Widening)
                    if width < 16 then yield (MLIRTypes.createInteger 16, Widening)
                    
                    // Float conversions
                    yield (MLIRTypes.createFloat 32, IntToFloat)
                    yield (MLIRTypes.createFloat 64, IntToFloat)
                    
                    // String coercion for display
                    yield (MLIRTypes.createMemRef (MLIRTypes.createInteger 8) [], Coercion)
                ]
            | None -> []
        | FloatCategory ->
            match mlirType.Width with
            | Some width ->
                [
                    // Widening to larger floats
                    if width < 64 then yield (MLIRTypes.createFloat 64, Widening)
                    
                    // Integer conversions (potentially lossy)
                    yield (MLIRTypes.createInteger 32, FloatToInt)
                    yield (MLIRTypes.createInteger 64, FloatToInt)
                    
                    // String coercion
                    yield (MLIRTypes.createMemRef (MLIRTypes.createInteger 8) [], Coercion)
                ]
            | None -> []
        | VoidCategory -> []
        | MemoryRefCategory ->
            [
                // Generic pointer conversions
                yield (MLIRTypes.createMemRef (MLIRTypes.createInteger 8) [], Coercion)
            ]
        | FunctionCategory -> []
        | StructCategory -> []
    
    /// Checks if this type can be converted to another
    let canConvertTo (fromType: MLIRType) (toType: MLIRType) : bool =
        fromType = toType ||
        getCompatibleConversions fromType
        |> List.exists (fun (compatibleType, _) -> compatibleType = toType)
    
    /// Gets the cost of conversion to another type
    let getConversionCost (fromType: MLIRType) (toType: MLIRType) : ConversionCost =
        if fromType = toType then NoCost
        else
            getCompatibleConversions fromType
            |> List.tryFind (fun (compatibleType, _) -> compatibleType = toType)
            |> Option.map snd
            |> Option.defaultValue Impossible

/// Checks if a conversion between types is safe (no data loss)
let isSafeConversion (fromType: MLIRType) (toType: MLIRType) : bool =
    match TypeAnalysis.getConversionCost fromType toType with
    | NoCost | Widening -> true
    | _ -> false

/// Checks if a conversion between types requires explicit coercion
let requiresCoercion (fromType: MLIRType) (toType: MLIRType) : bool =
    match TypeAnalysis.getConversionCost fromType toType with
    | NoCost -> false
    | Widening | IntToFloat -> false
    | _ -> true

/// Gets the most specific common type between two types
let getCommonType (type1: MLIRType) (type2: MLIRType) : MLIRType option =
    if areTypesCompatible type1 type2 then
        Some type1
    else
        match type1.Category, type2.Category with
        | IntegerCategory, IntegerCategory ->
            match type1.Width, type2.Width with
            | Some w1, Some w2 -> Some (MLIRTypes.createInteger (max w1 w2))
            | _ -> None
        | FloatCategory, FloatCategory ->
            match type1.Width, type2.Width with
            | Some w1, Some w2 -> Some (MLIRTypes.createFloat (max w1 w2))
            | _ -> None
        | IntegerCategory, FloatCategory ->
            type2.Width |> Option.map MLIRTypes.createFloat
        | FloatCategory, IntegerCategory ->
            type1.Width |> Option.map MLIRTypes.createFloat
        | _ -> None

/// Determines the best conversion path between types
let findConversionPath (fromType: MLIRType) (toType: MLIRType) : ConversionCost =
    TypeAnalysis.getConversionCost fromType toType

/// Checks if a type is a memory reference type
let isMemRefType (mlirType: MLIRType) : bool =
    mlirType.Category = MemoryRefCategory

/// Checks if a type is a numeric type
let isNumericType (mlirType: MLIRType) : bool =
    mlirType.Category = IntegerCategory || mlirType.Category = FloatCategory

/// Gets the element type of a memory reference
let getMemRefElementType (mlirType: MLIRType) : MLIRType option =
    if mlirType.Category = MemoryRefCategory then
        mlirType.ElementType
    else
        None

/// Determines if a type can be used in a specific context based on expected type
let canUseTypeInContext (actualType: MLIRType) (expectedType: MLIRType option) : bool =
    match expectedType with
    | None -> true  // No expectation, any type is acceptable
    | Some expected -> 
        actualType = expected || TypeAnalysis.canConvertTo actualType expected

/// Creates an appropriate default value for a given type
let defaultValueForType (mlirType: MLIRType) : string =
    match mlirType.Category with
    | IntegerCategory -> "0"
    | FloatCategory -> "0.0"
    | VoidCategory -> ""
    | MemoryRefCategory -> "null" // Placeholder, would require proper MLIR null pointer
    | FunctionCategory -> "null" // Placeholder
    | StructCategory -> "null" // Placeholder