module Core.Types.TypeSystem

/// MLIR dialect definitions
type MLIRDialect =
    | Standard
    | LLVM
    | Func
    | Arith
    | SCF
    | MemRef
    | Index
    | Affine
    | Builtin
    
/// Dialect-specific operation registry
type DialectOperation = {
    Dialect: MLIRDialect
    OpName: string
    Description: string
}

/// Core MLIR type categories
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
        { Dialect = MemRef; OpName = "memref.alloc"; Description = "Allocate memory" }
        { Dialect = MemRef; OpName = "memref.alloca"; Description = "Stack allocate memory" }
        { Dialect = MemRef; OpName = "memref.dealloc"; Description = "Deallocate memory" }
        { Dialect = MemRef; OpName = "memref.load"; Description = "Load from memref" }
        { Dialect = MemRef; OpName = "memref.store"; Description = "Store to memref" }
        { Dialect = MemRef; OpName = "memref.cast"; Description = "Cast memref type" }
        { Dialect = MemRef; OpName = "memref.view"; Description = "Create view of memref" }
        { Dialect = MemRef; OpName = "memref.subview"; Description = "Create subview" }
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
    | MemRef -> MemRefOps.operations
    | SCF -> SCFOps.operations
    | _ -> []

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
    let index = { void_ with Category = Index }
    
    /// Create integer type with specified width
    let int (width: int) = { void_ with Category = Integer; BitWidth = Some width }
    
    /// Create float type with specified width
    let float (width: int) = { void_ with Category = Float; BitWidth = Some width }
    
    /// Create memref type
    let memref (elementType: MLIRType) = {
        void_ with 
            Category = MemRef
            ElementType = Some elementType
    }
    
    /// Create shaped memref type
    let memrefWithShape (shape: int list) (elementType: MLIRType) = {
        void_ with 
            Category = MemRef
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
    
    /// String type (alias for memref of i8)
    let string_ = memref i8

/// Convert MLIR type to string representation
let rec mlirTypeToString (t: MLIRType) : string =
    match t.Category with
    | Void -> "void"
    | Integer ->
        match t.BitWidth with
        | Some width -> sprintf "i%d" width
        | None -> "i32"  // Default integer
    | Float ->
        match t.BitWidth with
        | Some width -> sprintf "f%d" width
        | None -> "f32"  // Default float
    | Index -> "index"
    | MemRef ->
        match t.ElementType, t.Shape with
        | Some elemType, Some shape ->
            let shapeStr = shape |> List.map string |> String.concat "x"
            sprintf "memref<%sx%s>" shapeStr (mlirTypeToString elemType)
        | Some elemType, None ->
            sprintf "memref<?x%s>" (mlirTypeToString elemType)
        | _ -> "memref<?x?>"
    | Tensor ->
        match t.ElementType, t.Shape with
        | Some elemType, Some shape ->
            let shapeStr = shape |> List.map string |> String.concat "x"
            sprintf "tensor<%sx%s>" shapeStr (mlirTypeToString elemType)
        | Some elemType, None ->
            sprintf "tensor<?x%s>" (mlirTypeToString elemType)
        | _ -> "tensor<?x?>"
    | Vector ->
        match t.ElementType, t.Shape with
        | Some elemType, Some [size] ->
            sprintf "vector<%dx%s>" size (mlirTypeToString elemType)
        | _ -> "vector<?x?>"
    | Function ->
        match t.ParameterTypes, t.ReturnType with
        | Some params, Some ret ->
            let paramStr = params |> List.map mlirTypeToString |> String.concat ", "
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

/// Dialect-specific type conversions
module DialectTypes =
    /// Convert to LLVM dialect type representation
    let toLLVMType (t: MLIRType) : string =
        match t.Category with
        | MemRef when t.ElementType = Some MLIRTypes.i8 -> "!llvm.ptr"
        | MemRef -> "!llvm.ptr"
        | Void -> "!llvm.void"
        | Struct -> mlirTypeToString t  // Already in LLVM format
        | _ -> mlirTypeToString t
    
    /// Check if type requires LLVM dialect
    let requiresLLVMDialect (t: MLIRType) : bool =
        match t.Category with
        | MemRef -> true
        | Struct -> true
        | Void -> true
        | _ -> false

/// Type validation helpers
module TypeValidation =
    /// Check if type is valid for arithmetic operations
    let isArithmeticType (t: MLIRType) : bool =
        match t.Category with
        | Integer | Float | Index -> true
        | _ -> false
    
    /// Check if type is valid for comparison
    let isComparableType (t: MLIRType) : bool =
        match t.Category with
        | Integer | Float | Index -> true
        | _ -> false
    
    /// Check if types are compatible for operations
    let areCompatible (t1: MLIRType) (t2: MLIRType) : bool =
        t1.Category = t2.Category && t1.BitWidth = t2.BitWidth