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