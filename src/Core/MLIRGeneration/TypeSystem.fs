// In TypeSystem.fs, we need these definitions:
module Core.MLIRGeneration.TypeSystem

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

/// Type utility functions - THIS IS WHAT'S MISSING
module MLIRTypes = 
    /// Creates an integer type with specified width
    let i32 = {
        Category = MLIRTypeCategory.Integer
        Width = Some 32
        ElementType = None
        Parameters = []
        ReturnType = None
    }

    let i64 = {
        Category = MLIRTypeCategory.Integer
        Width = Some 64
        ElementType = None
        Parameters = []
        ReturnType = None
    }

    let i8 = {
        Category = MLIRTypeCategory.Integer
        Width = Some 8
        ElementType = None
        Parameters = []
        ReturnType = None
    }

    let i1 = {
        Category = MLIRTypeCategory.Integer
        Width = Some 1
        ElementType = None
        Parameters = []
        ReturnType = None
    }

    let f32 = {
        Category = MLIRTypeCategory.Float
        Width = Some 32
        ElementType = None
        Parameters = []
        ReturnType = None
    }

    let f64 = {
        Category = MLIRTypeCategory.Float
        Width = Some 64
        ElementType = None
        Parameters = []
        ReturnType = None
    }

    let void_ = {
        Category = MLIRTypeCategory.Void
        Width = None
        ElementType = None
        Parameters = []
        ReturnType = None
    }

    /// Creates a memory reference type with element type
    let memref elementType = {
        Category = MLIRTypeCategory.MemRef
        Width = None
        ElementType = Some elementType
        Parameters = []
        ReturnType = None
    }

    /// Creates a function type with parameters and return type
    let func inputTypes returnType = {
        Category = MLIRTypeCategory.Function
        Width = None
        ElementType = None
        Parameters = inputTypes
        ReturnType = Some returnType
    }

    /// Creates a struct type
    let struct_ fields = {
        Category = MLIRTypeCategory.Struct
        Width = None
        ElementType = None
        Parameters = fields
        ReturnType = None
    }

    /// Creates an array type
    let array elementType size = {
        Category = MLIRTypeCategory.MemRef
        Width = Some size
        ElementType = Some elementType
        Parameters = []
        ReturnType = None
    }

    /// Nullable type helper
    let nullable baseType = memref baseType