/// MLIRTypes - Core MLIR type representations for Alex code generation
///
/// This module contains the essential type definitions used throughout
/// the MLIR generation pipeline. It is imported by MLIRZipper and bindings.
///
/// Extracted from legacy MLIRBuilder.fs during FNCS integration (December 2025).
/// The monad-based MLIR<'T> computation expression has been removed in favor
/// of the codata accumulator pattern in MLIRZipper.fs.
module Alex.CodeGeneration.MLIRTypes

// ═══════════════════════════════════════════════════════════════════
// Core Types
// ═══════════════════════════════════════════════════════════════════

/// MLIR integer bit widths
type IntegerBitWidth = I1 | I8 | I16 | I32 | I64

/// MLIR floating point precision
type FloatPrecision = F32 | F64

/// MLIR types - the structured representation used throughout Alex
/// String serialization happens only in Serialize module during MLIR text emission
type MLIRType =
    | Integer of IntegerBitWidth
    | Float of FloatPrecision
    | Pointer
    | Struct of MLIRType list
    | Array of int * MLIRType
    | Function of MLIRType list * MLIRType
    | Unit
    | Index

/// An SSA value - the primary currency of MLIR operations
[<Struct>]
type SSA =
    | V of int
    | Arg of int
    member this.Name =
        match this with
        | V n -> sprintf "%%v%d" n
        | Arg n -> sprintf "%%arg%d" n

/// A typed SSA value
type Val = { SSA: SSA; Type: MLIRType }

/// Global symbol reference
type Global =
    | GFunc of string
    | GStr of uint32
    | GBytes of int

/// Integer comparison predicates
type ICmp = Eq | Ne | Slt | Sle | Sgt | Sge | Ult | Ule | Ugt | Uge

// ═══════════════════════════════════════════════════════════════════
// Type Serialization
// ═══════════════════════════════════════════════════════════════════

module Serialize =
    let integerBitWidth = function
        | I1 -> "i1" | I8 -> "i8" | I16 -> "i16" | I32 -> "i32" | I64 -> "i64"

    let floatPrecision = function F32 -> "f32" | F64 -> "f64"

    let rec mlirType = function
        | Integer bitWidth -> integerBitWidth bitWidth
        | Float precision -> floatPrecision precision
        | Pointer -> "!llvm.ptr"
        | Struct fields ->
            let fieldTypes = fields |> List.map mlirType |> String.concat ", "
            sprintf "!llvm.struct<(%s)>" fieldTypes
        | Array (count, elementType) -> sprintf "!llvm.array<%d x %s>" count (mlirType elementType)
        | Function (parameterTypes, returnType) ->
            let parameterString = parameterTypes |> List.map mlirType |> String.concat ", "
            sprintf "(%s) -> %s" parameterString (mlirType returnType)
        | Unit -> "i32"  // Unit maps to i32 for compatibility (returns 0)
        | Index -> "index"

    /// Deserialize an MLIR type string back to MLIRType
    /// This is the inverse of mlirType
    let deserializeType (s: string) : MLIRType =
        match s.Trim() with
        | "i1" -> Integer I1
        | "i8" -> Integer I8
        | "i16" -> Integer I16
        | "i32" -> Integer I32
        | "i64" -> Integer I64
        | "f32" -> Float F32
        | "f64" -> Float F64
        | "!llvm.ptr" -> Pointer
        | "index" -> Index
        | _ when s.StartsWith("!llvm.ptr") -> Pointer  // Handle opaque pointer variations
        | _ -> Pointer  // Default to pointer for unknown types (conservative)

    let ssa (s: SSA) = s.Name

    let global_ = function
        | GFunc name -> sprintf "@%s" name
        | GStr hash -> sprintf "@str_%u" hash
        | GBytes idx -> sprintf "@bytes_%d" idx

    let icmp = function
        | Eq -> "eq" | Ne -> "ne"
        | Slt -> "slt" | Sle -> "sle" | Sgt -> "sgt" | Sge -> "sge"
        | Ult -> "ult" | Ule -> "ule" | Ugt -> "ugt" | Uge -> "uge"

    /// Escape string content for MLIR literals
    let escape (s: string) =
        s.Replace("\\", "\\\\")
         .Replace("\"", "\\\"")
         .Replace("\n", "\\0A")
         .Replace("\r", "\\0D")
         .Replace("\t", "\\09")
