module Core.MLIRGeneration.TypeSystem

open Dabbit.Parsing.OakAst

/// Represents MLIR types used in code generation
type MLIRType =
    | Integer of width: int
    | Float of width: int
    | Void
    | MemRef of elementType: MLIRType * shape: int list
    | Function of inputs: MLIRType list * output: MLIRType
    | Struct of elements: MLIRType list

/// Maps Oak AST types to corresponding MLIR types
let mapOakTypeToMLIR (oakType: OakType) : MLIRType =
    match oakType with
    | IntType -> Integer 32
    | FloatType -> Float 32
    | BoolType -> Integer 1
    | UnitType -> Void
    | StringType -> MemRef(Integer 8, []) // Treated as byte array
    | ArrayType elemType -> MemRef(mapOakTypeToMLIR elemType, [])
    | FunctionType(paramTypes, returnType) ->
        let mappedParams = paramTypes |> List.map mapOakTypeToMLIR
        let mappedReturn = mapOakTypeToMLIR returnType
        Function(mappedParams, mappedReturn)
    | StructType fields ->
        let fieldTypes = fields |> List.map (snd >> mapOakTypeToMLIR)
        Struct fieldTypes
    | UnionType cases ->
        // For unions, we need a tag (i8) and the largest possible payload
        // This is simplified; real implementation would calculate proper layout
        Struct [Integer 8; Integer 64] // Placeholder size

/// Converts MLIR type to string representation
let mlirTypeToString (mlirType: MLIRType) : string =
    match mlirType with
    | Integer width -> sprintf "i%d" width
    | Float width -> sprintf "f%d" width
    | Void -> "()"
    | MemRef(elementType, shape) ->
        let elementStr = mlirTypeToString elementType
        if shape.IsEmpty then
            sprintf "memref<?x%s>" elementStr
        else
            let shapeStr = shape |> List.map string |> String.concat "x"
            sprintf "memref<%sx%s>" shapeStr elementStr
    | Function(inputs, output) ->
        let inputStrs = inputs |> List.map mlirTypeToString |> String.concat ", "
        let outputStr = mlirTypeToString output
        sprintf "(%s) -> %s" inputStrs outputStr
    | Struct elements ->
        let elementStrs = elements |> List.map mlirTypeToString |> String.concat ", "
        sprintf "!llvm.struct<(%s)>" elementStrs

/// Gets the size in bytes of an MLIR type (for layout calculations)
let getTypeSize (mlirType: MLIRType) : int =
    match mlirType with
    | Integer 1 -> 1  // i1 (bool)
    | Integer 8 -> 1  // i8
    | Integer 16 -> 2 // i16
    | Integer 32 -> 4 // i32
    | Integer 64 -> 8 // i64
    | Integer width -> (width + 7) / 8 // Round up to nearest byte
    | Float 32 -> 4   // f32
    | Float 64 -> 8   // f64
    | Float width -> (width + 7) / 8
    | Void -> 0
    | MemRef(_, _) -> 8 // Pointer size on 64-bit systems
    | Function(_, _) -> 8 // Function pointer
    | Struct elements -> elements |> List.sumBy getTypeSize

/// Checks if a type requires heap allocation
let requiresHeapAllocation (mlirType: MLIRType) : bool =
    match mlirType with
    | Integer _ | Float _ | Void -> false
    | MemRef(_, shape) -> shape.IsEmpty // Dynamic arrays need heap
    | Function(_, _) -> false // Function pointers are stack-allocated
    | Struct elements -> elements |> List.exists requiresHeapAllocation

/// Gets the alignment requirement for a type
let getTypeAlignment (mlirType: MLIRType) : int =
    match mlirType with
    | Integer 1 -> 1
    | Integer width when width <= 8 -> 1
    | Integer width when width <= 16 -> 2
    | Integer width when width <= 32 -> 4
    | Integer width -> 8
    | Float 32 -> 4
    | Float 64 -> 8
    | Float _ -> 8
    | Void -> 1
    | MemRef(_, _) -> 8
    | Function(_, _) -> 8
    | Struct elements -> 
        if elements.IsEmpty then 1
        else elements |> List.map getTypeAlignment |> List.max

/// Determines if two MLIR types are compatible for assignment/comparison
let areTypesCompatible (type1: MLIRType) (type2: MLIRType) : bool =
    match type1, type2 with
    | Integer w1, Integer w2 -> w1 = w2
    | Float w1, Float w2 -> w1 = w2
    | Void, Void -> true
    | MemRef(elem1, shape1), MemRef(elem2, shape2) -> 
        areTypesCompatible elem1 elem2 && shape1 = shape2
    | Function(in1, out1), Function(in2, out2) ->
        List.length in1 = List.length in2 &&
        List.forall2 areTypesCompatible in1 in2 &&
        areTypesCompatible out1 out2
    | Struct elems1, Struct elems2 ->
        List.length elems1 = List.length elems2 &&
        List.forall2 areTypesCompatible elems1 elems2
    | _ -> false