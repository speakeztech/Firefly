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
let rec mapOakTypeToMLIR (oakType: OakType) : MLIRType =
    match oakType with
    | IntType -> Integer 32
    | FloatType -> Float 32
    | BoolType -> Integer 1
    | UnitType -> Void
    | StringType -> MemRef(Integer 8, []) // String as byte array with null terminator
    | ArrayType elemType -> MemRef(mapOakTypeToMLIR elemType, [])
    | FunctionType(paramTypes, returnType) ->
        let mappedParams = paramTypes |> List.map mapOakTypeToMLIR
        let mappedReturn = mapOakTypeToMLIR returnType
        Function(mappedParams, mappedReturn)
    | StructType fields ->
        let fieldTypes = fields |> List.map (snd >> mapOakTypeToMLIR)
        Struct fieldTypes
    | UnionType cases ->
        // For unions, we represent them as structs with a tag field
        // and a payload field that is sized to fit the largest case
        Struct [Integer 8; Integer 64] // tag: i8, payload: i64 (placeholder)

/// Converts MLIR type to string representation
let rec mlirTypeToString (mlirType: MLIRType) : string =
    match mlirType with
    | Integer width -> sprintf "i%d" width
    | Float 32 -> "f32"
    | Float 64 -> "f64"
    | Float width -> sprintf "f%d" width
    | Void -> "void"
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

/// Gets the size in bytes of an MLIR type
let rec getTypeSize (mlirType: MLIRType) : int =
    match mlirType with
    | Integer 1 -> 1  // Boolean uses 1 byte
    | Integer n when n <= 8 -> 1
    | Integer n when n <= 16 -> 2
    | Integer n when n <= 32 -> 4
    | Integer n when n <= 64 -> 8
    | Integer n -> (n + 7) / 8 // Round up to nearest byte
    | Float 32 -> 4
    | Float 64 -> 8
    | Float n -> (n + 7) / 8
    | Void -> 0
    | MemRef(_, _) -> 8 // Pointer size on 64-bit systems
    | Function(_, _) -> 8 // Function pointer
    | Struct elements -> 
        // Calculate size with alignment
        let sizes = elements |> List.map getTypeSize
        let alignments = elements |> List.map getTypeAlignment
        
        // Simple struct layout algorithm with alignment
        let rec calculateSize (sizes: int list) (alignments: int list) offset =
            match sizes, alignments with
            | [], [] -> offset
            | size :: restSizes, alignment :: restAlignments ->
                // Align current field
                let alignedOffset = ((offset + alignment - 1) / alignment) * alignment
                calculateSize restSizes restAlignments (alignedOffset + size)
            | _ -> failwith "Mismatched sizes and alignments"
        
        calculateSize sizes alignments 0

/// Gets the alignment requirement for a type
and getTypeAlignment (mlirType: MLIRType) : int =
    match mlirType with
    | Integer 1 -> 1
    | Integer n when n <= 8 -> 1
    | Integer n when n <= 16 -> 2
    | Integer n when n <= 32 -> 4
    | Integer n -> 8
    | Float 32 -> 4
    | Float 64 -> 8
    | Float _ -> 8
    | Void -> 1
    | MemRef(_, _) -> 8
    | Function(_, _) -> 8
    | Struct elements -> 
        if elements.IsEmpty then 1
        else elements |> List.map getTypeAlignment |> List.max

/// Determines if two MLIR types are compatible for assignment
let rec areTypesCompatible (type1: MLIRType) (type2: MLIRType) : bool =
    match type1, type2 with
    | Integer w1, Integer w2 -> w1 = w2
    | Float w1, Float w2 -> w1 = w2
    | Void, Void -> true
    | MemRef(elem1, shape1), MemRef(elem2, shape2) -> 
        areTypesCompatible elem1 elem2 && 
        (shape1 = shape2 || shape1.IsEmpty || shape2.IsEmpty)
    | Function(in1, out1), Function(in2, out2) ->
        List.length in1 = List.length in2 &&
        List.forall2 areTypesCompatible in1 in2 &&
        areTypesCompatible out1 out2
    | Struct elems1, Struct elems2 ->
        List.length elems1 = List.length elems2 &&
        List.forall2 areTypesCompatible elems1 elems2
    | _ -> false

/// Checks if a type requires heap allocation
let rec requiresHeapAllocation (mlirType: MLIRType) : bool =
    match mlirType with
    | Integer _ | Float _ | Void -> false
    | MemRef(_, shape) -> shape.IsEmpty // Dynamic arrays need heap
    | Function(_, _) -> false // Function pointers are stack-allocated
    | Struct elements -> elements |> List.exists requiresHeapAllocation

/// Generates a fixed-size type equivalent for zero-allocation
let rec makeFixedSizeType (mlirType: MLIRType) : MLIRType =
    match mlirType with
    | MemRef(elementType, []) -> 
        // Replace dynamic array with fixed-size buffer
        MemRef(elementType, [256]) // Use a 256-element buffer
    | Struct elements ->
        // Recursively fix struct elements
        Struct (elements |> List.map makeFixedSizeType)
    | _ -> mlirType // Other types are already fixed-size

/// Creates a stack-allocated version of a type
let stackAllocatedType (mlirType: MLIRType) : MLIRType =
    match mlirType with
    | MemRef(elementType, []) -> 
        // For dynamic arrays, provide a stack buffer
        MemRef(elementType, [256])
    | _ -> mlirType