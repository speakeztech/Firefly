module Core.Types.TypeSystem

open Dialects

/// Dialect-specific operation registry
type DialectOperation = {
    Dialect: MLIRDialect
    OpName: string
    Description: string
}

type MLIRTypeCategory =
    | Builtin
    | Integer
    | Float
    | Index
    | MemRef
    | Tensor
    | Vector
    | Function
    | Struct
    | Void
    | Tuple      
    | Variant    
    | Opaque     
    
/// Core MLIR type representation
type MLIRType = {
    Category: MLIRTypeCategory
    BitWidth: int option
    ElementType: MLIRType option
    Shape: int list option
    ParameterTypes: MLIRType list option
    ReturnType: MLIRType option
    Fields: (string * MLIRType) list option
}

/// MLIR value representation
type MLIRValue = {
    SSA: string
    Type: string
    IsConstant: bool
}

/// Common MLIR type constructors
module MLIRTypes =
    /// Void type
    let void_ = {
        Category = Void
        BitWidth = None
        ElementType = None
        Shape = None
        ParameterTypes = None
        ReturnType = None
        Fields = None
    }
    
    /// Integer types
    let i1 = { void_ with Category = Integer; BitWidth = Some 1 }
    let i8 = { void_ with Category = Integer; BitWidth = Some 8 }
    let i16 = { void_ with Category = Integer; BitWidth = Some 16 }
    let i32 = { void_ with Category = Integer; BitWidth = Some 32 }
    let i64 = { void_ with Category = Integer; BitWidth = Some 64 }
    
    /// Float types
    let f16 = { void_ with Category = Float; BitWidth = Some 16 }
    let f32 = { void_ with Category = Float; BitWidth = Some 32 }
    let f64 = { void_ with Category = Float; BitWidth = Some 64 }
    
    /// Index type (architecture-dependent integer)
    let index = { void_ with Category = MLIRTypeCategory.Index }
    
    /// Create integer type with specified width
    let int (width: int) = { void_ with Category = Integer; BitWidth = Some width }
    
    /// Create float type with specified width
    let float (width: int) = { void_ with Category = Float; BitWidth = Some width }
    
    /// Create memref type
    let memref (elementType: MLIRType) = {
        void_ with 
            Category = MLIRTypeCategory.MemRef
            ElementType = Some elementType
    }
    
    /// Create shaped memref type
    let memrefWithShape (shape: int list) (elementType: MLIRType) = {
        void_ with 
            Category = MLIRTypeCategory.MemRef
            ElementType = Some elementType
            Shape = Some shape
    }
    
    /// Create tensor type
    let tensor (elementType: MLIRType) = {
        void_ with 
            Category = Tensor
            ElementType = Some elementType
    }
    
    /// Create shaped tensor type
    let tensorWithShape (shape: int list) (elementType: MLIRType) = {
        void_ with 
            Category = Tensor
            ElementType = Some elementType
            Shape = Some shape
    }
    
    /// Create vector type
    let vector (size: int) (elementType: MLIRType) = {
        void_ with 
            Category = Vector
            ElementType = Some elementType
            Shape = Some [size]
    }
    
    /// Create function type
    let func (paramTypes: MLIRType list) (returnType: MLIRType) = {
        void_ with 
            Category = Function
            ParameterTypes = Some paramTypes
            ReturnType = Some returnType
    }
    
    /// Create struct type
    let struct_ (fields: (string * MLIRType) list) = {
        void_ with 
            Category = Struct
            Fields = Some fields
    }
    
    /// Create struct type without field names
    let structNoNames (fieldTypes: MLIRType list) = {
        void_ with 
            Category = Struct
            Fields = Some (fieldTypes |> List.mapi (fun i t -> sprintf "field%d" i, t))
    }

    
    
    /// String type (alias for memref of i8)
    let string_ = memref i8

    /// Create tuple type (ordered, unnamed fields)
    let tuple (elementTypes: MLIRType list) = {
        Category = Tuple
        BitWidth = None
        ElementType = None
        Shape = None
        ParameterTypes = Some elementTypes  // Reuse for tuple elements
        ReturnType = None
        Fields = None
    }
    
    /// Create variant type (discriminated union)
    let variant (cases: (string * MLIRType) list) = {
        Category = Variant
        BitWidth = None
        ElementType = None
        Shape = None
        ParameterTypes = None
        ReturnType = None
        Fields = Some cases  // Reuse for variant cases
    }
    
    /// Create opaque type (for unresolved or generic types)
    let opaque (name: string) = {
        Category = Opaque
        BitWidth = None
        ElementType = None
        Shape = None
        ParameterTypes = None
        ReturnType = None
        Fields = Some [(name, void_)]  // Store type name as single field
    }

    /// Polymorphic "any" type for generic operations
    let any = opaque "any"

// Helper function to check if a type is the polymorphic any type
let isAnyType (t: MLIRType) =
    match t.Category with
    | Opaque -> 
        match t.Fields with
        | Some [("any", _)] -> true
        | _ -> false
    | _ -> false

/// Helper function to get type from MLIRValue
let parseTypeFromMLIRValue (value: MLIRValue): MLIRType =
    match value.Type with
    | "i1" -> MLIRTypes.i1
    | "i8" -> MLIRTypes.i8
    | "i16" -> MLIRTypes.i16
    | "i32" -> MLIRTypes.i32
    | "i64" -> MLIRTypes.i64
    | "f32" -> MLIRTypes.f32
    | "f64" -> MLIRTypes.f64
    | "void" -> MLIRTypes.void_
    | _ -> MLIRTypes.i32  // Default fallback

/// Helper to parse type string back to MLIRType (temporary until better type system integration)
let parseTypeFromString (typeStr: string): MLIRType =
    match typeStr with
    | "i1" -> MLIRTypes.i1
    | "i8" -> MLIRTypes.i8
    | "i16" -> MLIRTypes.i16
    | "i32" -> MLIRTypes.i32
    | "i64" -> MLIRTypes.i64
    | "f32" -> MLIRTypes.f32
    | "f64" -> MLIRTypes.f64
    | "void" -> MLIRTypes.void_
    | _ -> MLIRTypes.i32  

/// Standard MLIR dialect operations
module StandardOps =
    let operations = [
        { Dialect = Standard; OpName = "std.constant"; Description = "Constant value operation" }
        { Dialect = Standard; OpName = "std.return"; Description = "Return from function" }
        { Dialect = Standard; OpName = "std.call"; Description = "Direct function call" }
        { Dialect = Standard; OpName = "std.br"; Description = "Unconditional branch" }
        { Dialect = Standard; OpName = "std.cond_br"; Description = "Conditional branch" }
    ]

/// Arithmetic dialect operations
module ArithOps =
    let operations = [
        { Dialect = Arith; OpName = "arith.constant"; Description = "Arithmetic constant" }
        { Dialect = Arith; OpName = "arith.addi"; Description = "Integer addition" }
        { Dialect = Arith; OpName = "arith.subi"; Description = "Integer subtraction" }
        { Dialect = Arith; OpName = "arith.muli"; Description = "Integer multiplication" }
        { Dialect = Arith; OpName = "arith.divsi"; Description = "Signed integer division" }
        { Dialect = Arith; OpName = "arith.remsi"; Description = "Signed integer remainder" }
        { Dialect = Arith; OpName = "arith.cmpi"; Description = "Integer comparison" }
        { Dialect = Arith; OpName = "arith.addf"; Description = "Float addition" }
        { Dialect = Arith; OpName = "arith.subf"; Description = "Float subtraction" }
        { Dialect = Arith; OpName = "arith.mulf"; Description = "Float multiplication" }
        { Dialect = Arith; OpName = "arith.divf"; Description = "Float division" }
        { Dialect = Arith; OpName = "arith.cmpf"; Description = "Float comparison" }
    ]

/// Function dialect operations
module FuncOps =
    let operations = [
        { Dialect = Func; OpName = "func.func"; Description = "Function definition" }
        { Dialect = Func; OpName = "func.return"; Description = "Return from function" }
        { Dialect = Func; OpName = "func.call"; Description = "Function call" }
        { Dialect = Func; OpName = "func.call_indirect"; Description = "Indirect function call" }
    ]

/// LLVM dialect operations
module LLVMOps =
    let operations = [
        { Dialect = LLVM; OpName = "llvm.func"; Description = "LLVM function" }
        { Dialect = LLVM; OpName = "llvm.return"; Description = "LLVM return" }
        { Dialect = LLVM; OpName = "llvm.call"; Description = "LLVM call" }
        { Dialect = LLVM; OpName = "llvm.alloca"; Description = "Stack allocation" }
        { Dialect = LLVM; OpName = "llvm.load"; Description = "Load from memory" }
        { Dialect = LLVM; OpName = "llvm.store"; Description = "Store to memory" }
        { Dialect = LLVM; OpName = "llvm.getelementptr"; Description = "Get element pointer" }
        { Dialect = LLVM; OpName = "llvm.bitcast"; Description = "Bitcast operation" }
        { Dialect = LLVM; OpName = "llvm.mlir.constant"; Description = "LLVM constant" }
        { Dialect = LLVM; OpName = "llvm.mlir.null"; Description = "Null pointer" }
        { Dialect = LLVM; OpName = "llvm.mlir.undef"; Description = "Undefined value" }
        { Dialect = LLVM; OpName = "llvm.mlir.addressof"; Description = "Address of global" }
        { Dialect = LLVM; OpName = "llvm.ptr"; Description = "LLVM pointer type" }
        { Dialect = LLVM; OpName = "llvm.void"; Description = "LLVM void type" }
    ]

/// MemRef dialect operations
module MemRefOps =
    let operations = [
        { Dialect = MLIRDialect.MemRef; OpName = "memref.alloc"; Description = "Allocate memory" }
        { Dialect = MLIRDialect.MemRef; OpName = "memref.alloca"; Description = "Stack allocate memory" }
        { Dialect = MLIRDialect.MemRef; OpName = "memref.dealloc"; Description = "Deallocate memory" }
        { Dialect = MLIRDialect.MemRef; OpName = "memref.load"; Description = "Load from memref" }
        { Dialect = MLIRDialect.MemRef; OpName = "memref.store"; Description = "Store to memref" }
        { Dialect = MLIRDialect.MemRef; OpName = "memref.cast"; Description = "Cast memref type" }
        { Dialect = MLIRDialect.MemRef; OpName = "memref.view"; Description = "Create view of memref" }
        { Dialect = MLIRDialect.MemRef; OpName = "memref.subview"; Description = "Create subview" }
    ]

/// SCF (Structured Control Flow) dialect operations
module SCFOps =
    let operations = [
        { Dialect = SCF; OpName = "scf.if"; Description = "If-then-else" }
        { Dialect = SCF; OpName = "scf.for"; Description = "For loop" }
        { Dialect = SCF; OpName = "scf.while"; Description = "While loop" }
        { Dialect = SCF; OpName = "scf.condition"; Description = "While condition" }
        { Dialect = SCF; OpName = "scf.yield"; Description = "Yield value" }
    ]

/// Get all operations for a dialect
let getDialectOperations (dialect: MLIRDialect) : DialectOperation list =
    match dialect with
    | Standard -> StandardOps.operations
    | Arith -> ArithOps.operations
    | Func -> FuncOps.operations
    | LLVM -> LLVMOps.operations
    | MLIRDialect.MemRef -> MemRefOps.operations
    | SCF -> SCFOps.operations
    | MLIRDialect.Index -> []
    | Affine -> []
    | MLIRDialect.Builtin -> []



let rec mlirTypeToString (t: MLIRType) : string =
    match t.Category with
    | Void -> "void"
    | Integer ->
        match t.BitWidth with
        | Some width -> sprintf "i%d" width
        | None -> "i32"
    | Float ->
        match t.BitWidth with
        | Some 16 -> "f16"
        | Some 32 -> "f32"
        | Some 64 -> "f64"
        | _ -> "f32"
    | MLIRTypeCategory.Index -> "index"
    | MLIRTypeCategory.MemRef ->
        match t.ElementType, t.Shape with
        | Some elem, Some shape ->
            let shapeStr = shape |> List.map string |> String.concat "x"
            sprintf "memref<%sx%s>" shapeStr (mlirTypeToString elem)
        | Some elem, None ->
            sprintf "memref<?x%s>" (mlirTypeToString elem)
        | None, _ -> "memref<?x?>"
    | Tensor ->
        match t.ElementType, t.Shape with
        | Some elem, Some shape ->
            let shapeStr = shape |> List.map string |> String.concat "x"
            sprintf "tensor<%sx%s>" shapeStr (mlirTypeToString elem)
        | Some elem, None ->
            sprintf "tensor<?x%s>" (mlirTypeToString elem)
        | None, _ -> "tensor<?x?>"
    | Vector ->
        match t.ElementType, t.Shape with
        | Some elem, Some [size] ->
            sprintf "vector<%dx%s>" size (mlirTypeToString elem)
        | _ -> "vector<?x?>"
    | Function ->
        match t.ParameterTypes, t.ReturnType with
        | Some parameters, Some ret ->
            let paramStr = parameters |> List.map mlirTypeToString |> String.concat ", "
            sprintf "(%s) -> %s" paramStr (mlirTypeToString ret)
        | _ -> "(?) -> ?"
    | Struct ->
        match t.Fields with
        | Some fields ->
            let fieldStr = 
                fields 
                |> List.map (fun (name, typ) -> sprintf "%s: %s" name (mlirTypeToString typ))
                |> String.concat ", "
            sprintf "!llvm.struct<(%s)>" fieldStr
        | None -> "!llvm.struct<()>"
    | Tuple ->
        match t.ParameterTypes with
        | Some types ->
            let typeStr = types |> List.map mlirTypeToString |> String.concat ", "
            sprintf "tuple<%s>" typeStr
        | None -> "tuple<>"
    | Variant ->
        match t.Fields with
        | Some cases ->
            let caseStr = 
                cases 
                |> List.map (fun (name, typ) -> sprintf "%s: %s" name (mlirTypeToString typ))
                |> String.concat " | "
            sprintf "!variant<%s>" caseStr
        | None -> "!variant<>"
    | Opaque ->
        match t.Fields with
        | Some [(name, _)] -> sprintf "!opaque<%s>" name
        | _ -> "!opaque<unknown>"
    | MLIRTypeCategory.Builtin -> "builtin"


/// Dialect-specific type conversions
module DialectTypes =
    /// Convert to LLVM dialect type representation
    let toLLVMType (t: MLIRType) : string =
        match t.Category with
        | MLIRTypeCategory.MemRef when t.ElementType = Some MLIRTypes.i8 -> "!llvm.ptr"
        | MLIRTypeCategory.MemRef -> "!llvm.ptr"
        | Void -> "!llvm.void"
        | Struct -> mlirTypeToString t  // Already in LLVM format
        | _ -> mlirTypeToString t
    
    /// Check if type requires LLVM dialect
    let requiresLLVMDialect (t: MLIRType) : bool =
        match t.Category with
        | MLIRTypeCategory.MemRef -> true
        | Struct -> true
        | Void -> true
        | _ -> false

/// Type validation helpers
module TypeValidation =
    /// Check if type is valid for arithmetic operations
    let isArithmeticType (t: MLIRType) : bool =
        match t.Category with
        | Integer | Float | MLIRTypeCategory.Index -> true
        | _ -> false
    
    /// Check if type is valid for comparison
    let isComparableType (t: MLIRType) : bool =
        match t.Category with
        | Integer | Float | MLIRTypeCategory.Index -> true
        | _ -> false
    
    /// Check if types are compatible for operations
    let areCompatible (t1: MLIRType) (t2: MLIRType) : bool =
        t1.Category = t2.Category && t1.BitWidth = t2.BitWidth
    
    /// Get the result type for binary operations
    let getBinaryOpResultType (op: string) (t1: MLIRType) (t2: MLIRType) : MLIRType option =
        match op with
        | "arith.addi" | "arith.subi" | "arith.muli" | "arith.divsi" | "arith.remsi" 
            when t1.Category = Integer && areCompatible t1 t2 -> Some t1
        | "arith.addf" | "arith.subf" | "arith.mulf" | "arith.divf"
            when t1.Category = Float && areCompatible t1 t2 -> Some t1
        | "arith.cmpi" when t1.Category = Integer && areCompatible t1 t2 -> Some MLIRTypes.i1
        | "arith.cmpf" when t1.Category = Float && areCompatible t1 t2 -> Some MLIRTypes.i1
        | _ -> None
    
    /// Check if type is a composite type
    let isCompositeType (t: MLIRType) : bool =
        match t.Category with
        | Struct | Tuple | Variant -> true
        | _ -> false
    
    /// Get element types from composite type
    let getElementTypes (t: MLIRType) : MLIRType list option =
        match t.Category with
        | Tuple -> t.ParameterTypes
        | Struct | Variant -> 
            t.Fields |> Option.map (List.map snd)
        | _ -> None

/// Type size and alignment calculations
module TypeMetrics =
    
    /// Get alignment requirement in bytes
    let rec alignmentOf (t: MLIRType) : int =
        match t.Category with
        | Void -> 1
        | Integer ->
            match t.BitWidth with
            | Some bits when bits <= 8 -> 1
            | Some bits when bits <= 16 -> 2
            | Some bits when bits <= 32 -> 4
            | _ -> 8
        | Float ->
            match t.BitWidth with
            | Some 16 -> 2
            | Some 32 -> 4
            | Some 64 -> 8
            | _ -> 4
        | MLIRTypeCategory.Index -> 8
        | MLIRTypeCategory.MemRef -> 8
        | Vector -> alignmentOf (Option.defaultValue MLIRTypes.i32 t.ElementType)
        | Struct ->
            match t.Fields with
            | Some fields when fields.Length > 0 ->
                fields |> List.map (fun (_, ft) -> alignmentOf ft) |> List.max
            | _ -> 1
        | _ -> 8

    /// Get size of type in bytes
    let rec sizeOf (t: MLIRType) : int =
        match t.Category with
        | Void -> 0
        | Integer ->
            match t.BitWidth with
            | Some bits -> (bits + 7) / 8  // Round up to nearest byte
            | None -> 4  // Default to 32-bit
        | Float ->
            match t.BitWidth with
            | Some 16 -> 2
            | Some 32 -> 4
            | Some 64 -> 8
            | _ -> 4
        | MLIRTypeCategory.Index -> 8  // Typically pointer-sized
        | MLIRTypeCategory.MemRef -> 8  // Pointer size
        | Tuple ->
            match t.ParameterTypes with
            | Some types -> types |> List.sumBy sizeOf
            | None -> 0
        | Struct ->
            match t.Fields with
            | Some fields -> 
                fields |> List.sumBy (fun (_, typ) -> sizeOf typ)
            | None -> 0
        | Variant ->
            // Tag + largest case
            match t.Fields with
            | Some cases ->
                let tagSize = 4  // 32-bit discriminator
                let maxCaseSize = 
                    cases 
                    |> List.map (fun (_, typ) -> sizeOf typ) 
                    |> List.fold max 0
                tagSize + maxCaseSize
            | None -> 4  // Just the tag
        | Opaque -> 0  // Unknown size
        | _ -> 0