module Core.MLIRGeneration.TypeSystem

open Dabbit.Parsing.OakAst

/// MLIR type categories
type MLIRTypeCategory =
    | Integer = 0
    | Float = 1
    | Void = 2
    | MemRef = 3
    | Function = 4
    | Struct = 5

/// Simplified MLIR type representation
type MLIRType = {
    Category: MLIRTypeCategory
    Width: int option
    ElementType: MLIRType option
    Parameters: MLIRType list
    ReturnType: MLIRType option
}

/// Type utility functions to simplify working with MLIR types
module MLIRTypeUtils = 
    /// Creates an integer type with specified width
    let createInteger (width: int) = {
        Category = MLIRTypeCategory.Integer
        Width = Some width
        ElementType = None
        Parameters = []
        ReturnType = None
    }

    /// Creates a float type with specified width
    let createFloat (width: int) = {
        Category = MLIRTypeCategory.Float
        Width = Some width
        ElementType = None
        Parameters = []
        ReturnType = None
    }

    /// Creates a void type
    let createVoid() = {
        Category = MLIRTypeCategory.Void
        Width = None
        ElementType = None
        Parameters = []
        ReturnType = None
    }

    /// Creates a memory reference type with element type
    let createMemRef (elementType: MLIRType) = {
        Category = MLIRTypeCategory.MemRef
        Width = None
        ElementType = Some elementType
        Parameters = []
        ReturnType = None
    }

    /// Creates a function type with parameters and return type
    let createFunction (inputTypes: MLIRType list) (returnType: MLIRType) = {
        Category = MLIRTypeCategory.Function
        Width = None
        ElementType = None
        Parameters = inputTypes
        ReturnType = Some returnType
    }

    /// Checks if types are compatible for conversion
    let canConvertTo (fromType: MLIRType) (toType: MLIRType) : bool =
        fromType = toType || 
        (fromType.Category = MLIRTypeCategory.Integer && toType.Category = MLIRTypeCategory.Integer) ||
        (fromType.Category = MLIRTypeCategory.Float && toType.Category = MLIRTypeCategory.Float) ||
        (fromType.Category = MLIRTypeCategory.Integer && toType.Category = MLIRTypeCategory.Float) ||
        (fromType.Category = MLIRTypeCategory.MemRef && toType.Category = MLIRTypeCategory.MemRef)

/// Converts Oak type to MLIR type
let rec mapOakTypeToMLIR (oakType: OakType) : MLIRType =
    match oakType with
    | IntType -> MLIRTypeUtils.createInteger 32
    | FloatType -> MLIRTypeUtils.createFloat 32  
    | BoolType -> MLIRTypeUtils.createInteger 1
    | StringType -> MLIRTypeUtils.createMemRef (MLIRTypeUtils.createInteger 8)
    | UnitType -> MLIRTypeUtils.createVoid()
    | ArrayType elemType -> MLIRTypeUtils.createMemRef (mapOakTypeToMLIR elemType)
    | FunctionType(paramTypes, returnType) ->
        MLIRTypeUtils.createFunction 
            (paramTypes |> List.map mapOakTypeToMLIR) 
            (mapOakTypeToMLIR returnType)
    | StructType _ | UnionType _ -> 
        // Simplified struct handling - could be expanded if needed
        MLIRTypeUtils.createMemRef (MLIRTypeUtils.createInteger 8)

/// Converts MLIR type to string representation
let rec mlirTypeToString (mlirType: MLIRType) : string =
    match mlirType.Category with
    | MLIRTypeCategory.Integer -> 
        match mlirType.Width with
        | Some width -> sprintf "i%d" width
        | None -> "i32"
    | MLIRTypeCategory.Float -> 
        match mlirType.Width with
        | Some 32 -> "f32"
        | Some 64 -> "f64"
        | _ -> "f32"
    | MLIRTypeCategory.Void -> "()"
    | MLIRTypeCategory.MemRef -> 
        match mlirType.ElementType with
        | Some elemType -> sprintf "memref<?x%s>" (mlirTypeToString elemType)
        | None -> "memref<?xi8>"
    | MLIRTypeCategory.Function ->
        let paramStr = 
            mlirType.Parameters 
            |> List.map mlirTypeToString 
            |> String.concat ", "
        let retStr = 
            match mlirType.ReturnType with
            | Some ret -> mlirTypeToString ret
            | None -> "()"
        sprintf "(%s) -> %s" paramStr retStr
    | _ -> "!llvm.struct<()>" // Default for structs and other types