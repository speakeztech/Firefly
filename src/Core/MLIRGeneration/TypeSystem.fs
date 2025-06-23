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

/// Type utility functions
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

/// Type analysis and compatibility checking
module TypeAnalysis =
    /// Check if two types are exactly equal
    let rec areEqual (t1: MLIRType) (t2: MLIRType) : bool =
        t1.Category = t2.Category &&
        t1.Width = t2.Width &&
        match t1.ElementType, t2.ElementType with
        | Some e1, Some e2 -> areEqual e1 e2
        | None, None -> true
        | _ -> false
        &&
        t1.Parameters.Length = t2.Parameters.Length &&
        List.forall2 areEqual t1.Parameters t2.Parameters &&
        match t1.ReturnType, t2.ReturnType with
        | Some r1, Some r2 -> areEqual r1 r2
        | None, None -> true
        | _ -> false

    /// Check if source type can be converted to target type
    let rec canConvertTo (source: MLIRType) (target: MLIRType) : bool =
        // Exact match
        if areEqual source target then true
        // Check conversions by category
        else
            match source.Category, target.Category with
            // Integer conversions
            | MLIRTypeCategory.Integer, MLIRTypeCategory.Integer ->
                match source.Width, target.Width with
                | Some sw, Some tw -> sw <= tw  // Can widen integers
                | _ -> false
            
            // Float conversions
            | MLIRTypeCategory.Float, MLIRTypeCategory.Float ->
                match source.Width, target.Width with
                | Some 32, Some 64 -> true  // f32 -> f64
                | Some w1, Some w2 -> w1 = w2
                | _ -> false
            
            // Integer to float
            | MLIRTypeCategory.Integer, MLIRTypeCategory.Float ->
                true  // All integers can convert to floats
            
            // Memory reference conversions
            | MLIRTypeCategory.MemRef, MLIRTypeCategory.MemRef ->
                match source.ElementType, target.ElementType with
                | Some se, Some te -> 
                    // Allow memref covariance for compatible element types
                    canConvertTo se te
                | _ -> false
            
            // Function conversions (contravariant parameters, covariant return)
            | MLIRTypeCategory.Function, MLIRTypeCategory.Function ->
                match source.ReturnType, target.ReturnType with
                | Some sr, Some tr ->
                    // Return type must be convertible (covariant)
                    canConvertTo sr tr &&
                    // Parameters must be convertible in reverse (contravariant)
                    source.Parameters.Length = target.Parameters.Length &&
                    List.forall2 (fun tp sp -> canConvertTo tp sp) target.Parameters source.Parameters
                | _ -> false
            
            // Struct conversions (only if exact match for now)
            | MLIRTypeCategory.Struct, MLIRTypeCategory.Struct ->
                source.Parameters.Length = target.Parameters.Length &&
                List.forall2 canConvertTo source.Parameters target.Parameters
            
            // Void can only convert to void
            | MLIRTypeCategory.Void, MLIRTypeCategory.Void -> true
            
            // All other conversions are disallowed
            | MLIRTypeCategory.Void, _
            | _, MLIRTypeCategory.Void
            | _, _ -> false

    /// Get the common type that both types can convert to
    let commonType (t1: MLIRType) (t2: MLIRType) : MLIRType option =
        if areEqual t1 t2 then Some t1
        else if canConvertTo t1 t2 then Some t2
        else if canConvertTo t2 t1 then Some t1
        else
            match t1.Category, t2.Category with
            | MLIRTypeCategory.Integer, MLIRTypeCategory.Integer ->
                match t1.Width, t2.Width with
                | Some w1, Some w2 -> 
                    if w1 > w2 then Some t1 else Some t2
                | _ -> None
            | MLIRTypeCategory.Float, MLIRTypeCategory.Float ->
                match t1.Width, t2.Width with
                | Some w1, Some w2 -> 
                    if w1 > w2 then Some t1 else Some t2
                | _ -> None
            | MLIRTypeCategory.Integer, MLIRTypeCategory.Float -> Some t2
            | MLIRTypeCategory.Float, MLIRTypeCategory.Integer -> Some t1
            | MLIRTypeCategory.MemRef, MLIRTypeCategory.MemRef
            | MLIRTypeCategory.Function, MLIRTypeCategory.Function
            | MLIRTypeCategory.Struct, MLIRTypeCategory.Struct
            | MLIRTypeCategory.Void, MLIRTypeCategory.Void
            | _, _ -> None

    /// Check if a type is a primitive type (integer, float, or void)
    let isPrimitive (t: MLIRType) : bool =
        match t.Category with
        | MLIRTypeCategory.Integer
        | MLIRTypeCategory.Float
        | MLIRTypeCategory.Void -> true
        | MLIRTypeCategory.MemRef
        | MLIRTypeCategory.Function
        | MLIRTypeCategory.Struct -> false
        | _ -> false  // Handle any future enum cases

    /// Check if a type requires heap allocation
    let rec requiresHeapAllocation (t: MLIRType) : bool =
        match t.Category with
        | MLIRTypeCategory.MemRef -> 
            // Dynamic arrays require heap, fixed-size can be stack allocated
            t.Width.IsNone
        | MLIRTypeCategory.Struct ->
            // Structs with dynamic members require heap
            t.Parameters |> List.exists requiresHeapAllocation
        | MLIRTypeCategory.Integer
        | MLIRTypeCategory.Float
        | MLIRTypeCategory.Void
        | MLIRTypeCategory.Function -> false
        | _ -> false  // Handle any future enum cases

    /// Get the size of a type in bytes (if statically known)
    let rec sizeInBytes (t: MLIRType) : int option =
        match t.Category with
        | MLIRTypeCategory.Integer ->
            t.Width |> Option.map (fun w -> (w + 7) / 8)  // Round up to bytes
        | MLIRTypeCategory.Float ->
            t.Width |> Option.map (fun w -> w / 8)
        | MLIRTypeCategory.Void -> Some 0
        | MLIRTypeCategory.MemRef -> Some 8  // Pointer size
        | MLIRTypeCategory.Function -> Some 8  // Function pointer
        | MLIRTypeCategory.Struct ->
            // Sum of all field sizes
            t.Parameters 
            |> List.map sizeInBytes
            |> List.fold (fun acc size ->
                match acc, size with
                | Some a, Some s -> Some (a + s)
                | _ -> None) (Some 0)
        | _ -> None  // Handle any future enum cases

/// Type conversion helpers for string representation
let rec mlirTypeToString (t: MLIRType) : string =
    match t.Category with
    | MLIRTypeCategory.Integer ->
        match t.Width with
        | Some w -> sprintf "i%d" w
        | None -> "i32"  // Default
    | MLIRTypeCategory.Float ->
        match t.Width with
        | Some 32 -> "f32"
        | Some 64 -> "f64"
        | _ -> "f32"  // Default
    | MLIRTypeCategory.Void -> "void"
    | MLIRTypeCategory.MemRef ->
        match t.ElementType, t.Width with
        | Some elem, Some size -> sprintf "memref<%dx%s>" size (mlirTypeToString elem)
        | Some elem, None -> sprintf "memref<?x%s>" (mlirTypeToString elem)
        | None, _ -> "memref<?xi8>"  // Default
    | MLIRTypeCategory.Function ->
        match t.Parameters, t.ReturnType with
        | parameters, Some ret ->
            let paramStr = 
                if parameters.IsEmpty then "void"
                else parameters |> List.map mlirTypeToString |> String.concat ", "
            sprintf "(%s) -> %s" paramStr (mlirTypeToString ret)
        | _ -> "(void) -> void"  // Default
    | MLIRTypeCategory.Struct ->
        if t.Parameters.IsEmpty then "struct<>"
        else
            let fields = t.Parameters |> List.map mlirTypeToString |> String.concat ", "
            sprintf "struct<%s>" fields
    | _ -> "unknown"  // Handle any future enum cases