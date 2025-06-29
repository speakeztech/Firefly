module Dabbit.CodeGeneration.MLIRTypeOperations

open FSharp.Compiler.Syntax
open Core.Types.TypeSystem
open MLIREmitter

/// Type mapping from F# SynType to MLIR using Foundation patterns
module Mapping =
    
    /// Convert F# SynType to MLIR type using combinators
    let rec synTypeToMLIR (synType: SynType): MLIRCombinator<MLIRType> =
        mlir {
            match synType with
            | SynType.LongIdent(SynLongIdent(ids, _, _)) ->
                let typeName = ids |> List.map (fun id -> id.idText) |> String.concat "."
                return! mapPrimitiveType typeName
                
            | SynType.Array(rank, elementType, _) ->
                let! elemType = synTypeToMLIR elementType
                if rank = 1 then
                    return MLIRTypes.memref elemType
                else
                    return! fail "type_mapping" "Multi-dimensional arrays not yet supported"
                    
            | SynType.App(typeName, _, typeArgs, _, _, _, _) ->
                return! mapGenericType typeName typeArgs
                
            | SynType.Fun(argType, returnType, _, _) ->
                let! argMLIR = synTypeToMLIR argType
                let! retMLIR = synTypeToMLIR returnType
                return MLIRTypes.func [argMLIR] retMLIR
                
            | SynType.Paren(innerType, _) ->
                return! synTypeToMLIR innerType
                
            | SynType.Tuple(types, _) ->
                let! mlirTypes = mapM synTypeToMLIR types
                return MLIRTypes.struct_ mlirTypes
                
            | _ ->
                return! fail "type_mapping" (sprintf "Unsupported type pattern: %A" synType)
        }
    
    /// Map primitive F# types to MLIR types
    and mapPrimitiveType (typeName: string): MLIRCombinator<MLIRType> =
        mlir {
            match typeName with
            | "int" | "int32" -> return MLIRTypes.i32
            | "int64" -> return MLIRTypes.i64
            | "int16" -> return MLIRTypes.i16
            | "int8" | "sbyte" -> return MLIRTypes.i8
            | "uint32" -> return MLIRTypes.i32  // Treat as signed for now
            | "uint64" -> return MLIRTypes.i64  // Treat as signed for now
            | "uint16" -> return MLIRTypes.i16
            | "byte" | "uint8" -> return MLIRTypes.i8
            | "bool" -> return MLIRTypes.i1
            | "float" | "float32" | "single" -> return MLIRTypes.f32
            | "double" | "float64" -> return MLIRTypes.f64
            | "string" -> return MLIRTypes.string_
            | "unit" -> return MLIRTypes.void_
            | "char" -> return MLIRTypes.i8  // UTF-8 character
            | "nativeint" -> return MLIRTypes.i64  // Assume 64-bit target
            | "unativeint" -> return MLIRTypes.i64 // Assume 64-bit target
            | _ ->
                return! fail "primitive_type" (sprintf "Unknown primitive type: %s" typeName)
        }
    
    /// Map generic F# types to MLIR types
    and mapGenericType (typeName: SynType) (typeArgs: SynType list): MLIRCombinator<MLIRType> =
        mlir {
            match typeName with
            | SynType.LongIdent(SynLongIdent([id], _, _)) when id.idText = "Span" ->
                match typeArgs with
                | [elementType] ->
                    let! elemType = synTypeToMLIR elementType
                    return MLIRTypes.memref elemType
                | _ ->
                    return! fail "span_type" "Span must have exactly one type argument"
                    
            | SynType.LongIdent(SynLongIdent([id], _, _)) when id.idText = "ReadOnlySpan" ->
                match typeArgs with
                | [elementType] ->
                    let! elemType = synTypeToMLIR elementType
                    return MLIRTypes.memref elemType  // Read-only spans still use memref
                | _ ->
                    return! fail "readonly_span_type" "ReadOnlySpan must have exactly one type argument"
                    
            | SynType.LongIdent(SynLongIdent([id], _, _)) when id.idText = "option" ->
                match typeArgs with
                | [innerType] ->
                    let! innerMLIR = synTypeToMLIR innerType
                    // Option types represented as struct with tag and value
                    return MLIRTypes.struct_ [MLIRTypes.i1; innerMLIR]
                | _ ->
                    return! fail "option_type" "Option must have exactly one type argument"
                    
            | SynType.LongIdent(SynLongIdent([id], _, _)) when id.idText = "Result" ->
                match typeArgs with
                | [okType; errType] ->
                    let! okMLIR = synTypeToMLIR okType
                    let! errMLIR = synTypeToMLIR errType
                    // Result types represented as struct with tag, ok value, and error value
                    return MLIRTypes.struct_ [MLIRTypes.i1; okMLIR; errMLIR]
                | _ ->
                    return! fail "result_type" "Result must have exactly two type arguments"
                    
            | _ ->
                return! fail "generic_type" (sprintf "Unsupported generic type: %A" typeName)
        }

/// Type conversion operations using Foundation combinators
module Conversions =
    
    /// Implicit type conversions (widening)
    let implicitConvert (sourceType: MLIRType) (targetType: MLIRType) (value: MLIRValue): MLIRCombinator<MLIRValue> =
        mlir {
            if TypeAnalysis.areEqual sourceType targetType then
                return value
            elif TypeAnalysis.canConvertTo sourceType targetType then
                return! explicitConvert sourceType targetType value
            else
                return! fail "implicit_convert" 
                    (sprintf "Cannot implicitly convert %s to %s" 
                     (Core.formatType sourceType) (Core.formatType targetType))
        }
    
    /// Explicit type conversions
    and explicitConvert (sourceType: MLIRType) (targetType: MLIRType) (value: MLIRValue): MLIRCombinator<MLIRValue> =
        mlir {
            let! result = nextSSA "convert"
            let sourceTypeStr = Core.formatType sourceType
            let targetTypeStr = Core.formatType targetType
            
            match sourceType.Category, targetType.Category with
            // Integer conversions
            | MLIRTypeCategory.Integer, MLIRTypeCategory.Integer ->
                return! convertInteger sourceType targetType value result
                
            // Float conversions
            | MLIRTypeCategory.Float, MLIRTypeCategory.Float ->
                return! convertFloat sourceType targetType value result
                
            // Integer to float
            | MLIRTypeCategory.Integer, MLIRTypeCategory.Float ->
                do! emitLine (sprintf "%s = arith.sitofp %s : %s to %s" result value.SSA sourceTypeStr targetTypeStr)
                return Core.createValue result targetType
                
            // Float to integer
            | MLIRTypeCategory.Float, MLIRTypeCategory.Integer ->
                do! emitLine (sprintf "%s = arith.fptosi %s : %s to %s" result value.SSA sourceTypeStr targetTypeStr)
                return Core.createValue result targetType
                
            // Pointer/reference conversions
            | MLIRTypeCategory.MemRef, MLIRTypeCategory.MemRef ->
                return! convertMemref sourceType targetType value result
                
            | _ ->
                return! fail "explicit_convert" 
                    (sprintf "Unsupported conversion from %s to %s" sourceTypeStr targetTypeStr)
        }
    
    /// Convert between integer types
    and convertInteger (sourceType: MLIRType) (targetType: MLIRType) (value: MLIRValue) (result: string): MLIRCombinator<MLIRValue> =
        mlir {
            let sourceWidth = sourceType.Width |> Option.defaultValue 32
            let targetWidth = targetType.Width |> Option.defaultValue 32
            let sourceTypeStr = Core.formatType sourceType
            let targetTypeStr = Core.formatType targetType
            
            if sourceWidth = targetWidth then
                return value  // No conversion needed
            elif sourceWidth < targetWidth then
                // Sign extend
                do! emitLine (sprintf "%s = arith.extsi %s : %s to %s" result value.SSA sourceTypeStr targetTypeStr)
                return Core.createValue result targetType
            else
                // Truncate
                do! emitLine (sprintf "%s = arith.trunci %s : %s to %s" result value.SSA sourceTypeStr targetTypeStr)
                return Core.createValue result targetType
        }
    
    /// Convert between float types
    and convertFloat (sourceType: MLIRType) (targetType: MLIRType) (value: MLIRValue) (result: string): MLIRCombinator<MLIRValue> =
        mlir {
            let sourceWidth = sourceType.Width |> Option.defaultValue 32
            let targetWidth = targetType.Width |> Option.defaultValue 32
            let sourceTypeStr = Core.formatType sourceType
            let targetTypeStr = Core.formatType targetType
            
            if sourceWidth = targetWidth then
                return value  // No conversion needed
            elif sourceWidth < targetWidth then
                // Extend precision
                do! emitLine (sprintf "%s = arith.extf %s : %s to %s" result value.SSA sourceTypeStr targetTypeStr)
                return Core.createValue result targetType
            else
                // Truncate precision
                do! emitLine (sprintf "%s = arith.truncf %s : %s to %s" result value.SSA sourceTypeStr targetTypeStr)
                return Core.createValue result targetType
        }
    
    /// Convert between memory reference types
    and convertMemref (sourceType: MLIRType) (targetType: MLIRType) (value: MLIRValue) (result: string): MLIRCombinator<MLIRValue> =
        mlir {
            let sourceTypeStr = Core.formatType sourceType
            let targetTypeStr = Core.formatType targetType
            
            // For now, emit a cast operation
            do! emitLine (sprintf "%s = memref.cast %s : %s to %s" result value.SSA sourceTypeStr targetTypeStr)
            return Core.createValue result targetType
        }

/// Type checking and validation using Foundation patterns
module Validation =
    
    /// Check if two types are compatible for operations
    let areCompatible (type1: MLIRType) (type2: MLIRType): bool =
        TypeAnalysis.areEqual type1 type2 || 
        TypeAnalysis.canConvertTo type1 type2 || 
        TypeAnalysis.canConvertTo type2 type1
    
    /// Validate function signature compatibility
    let validateFunctionCall (expectedParams: MLIRType list) (actualArgs: MLIRValue list): MLIRCombinator<MLIRValue list> =
        mlir {
            if expectedParams.Length <> actualArgs.Length then
                return! fail "function_call" 
                    (sprintf "Expected %d arguments, got %d" expectedParams.Length actualArgs.Length)
            else
                let! convertedArgs = Utilities.mapM (fun (arg, expectedType) ->
                    if TypeAnalysis.areEqual arg.Type expectedType then
                        lift arg
                    else
                        Conversions.implicitConvert arg.Type expectedType arg
                ) (List.zip actualArgs expectedParams)
                return convertedArgs
        }
    
    /// Validate assignment compatibility
    let validateAssignment (targetType: MLIRType) (sourceValue: MLIRValue): MLIRCombinator<MLIRValue> =
        mlir {
            if TypeAnalysis.areEqual targetType sourceValue.Type then
                return sourceValue
            elif TypeAnalysis.canConvertTo sourceValue.Type targetType then
                return! Conversions.implicitConvert sourceValue.Type targetType sourceValue
            else
                return! fail "assignment" 
                    (sprintf "Cannot assign %s to %s" 
                     (Core.formatType sourceValue.Type) (Core.formatType targetType))
        }

/// Discriminated union and pattern matching support
module UnionTypes =
    
    /// Generate discriminated union type
    let createUnionType (cases: (string * MLIRType list) list): MLIRCombinator<MLIRType> =
        mlir {
            // DU represented as struct with tag + largest case data
            let maxCaseSize = cases 
                            |> List.map (fun (_, types) -> types.Length)
                            |> List.max
            
            // Create struct with tag (i32) + data slots
            let tagType = MLIRTypes.i32
            let dataTypes = List.replicate maxCaseSize MLIRTypes.i64  // Use i64 for general storage
            
            return MLIRTypes.struct_ (tagType :: dataTypes)
        }
    
    /// Generate union case constructor
    let createUnionCase (unionType: MLIRType) (caseIndex: int) (caseData: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            let! result = nextSSA "union_case"
            let! unionValue = Memory.alloca unionType None
            
            // Set tag
            let! tagValue = Constants.intConstant caseIndex 32
            let! tagIndex = Constants.intConstant 0 32
            do! Memory.store tagValue unionValue [tagIndex]
            
            // Set case data
            for i, data in List.indexed caseData do
                let! dataIndex = Constants.intConstant (i + 1) 32
                do! Memory.store data unionValue [dataIndex]
            
            return! Memory.load unionValue []
        }
    
    /// Generate union case pattern match
    let matchUnionCase (unionValue: MLIRValue) (caseIndex: int): MLIRCombinator<MLIRValue * MLIRValue list> =
        mlir {
            // Load tag
            let! tagIndex = Constants.intConstant 0 32
            let! tag = Memory.load unionValue [tagIndex]
            
            // Compare with expected case
            let! expectedTag = Constants.intConstant caseIndex 32
            let! isMatch = BinaryOps.compare "eq" tag expectedTag
            
            // Load case data (assuming match for now)
            let! data1Index = Constants.intConstant 1 32
            let! data1 = Memory.load unionValue [data1Index]
            let! data2Index = Constants.intConstant 2 32
            let! data2 = Memory.load unionValue [data2Index]
            
            return (isMatch, [data1; data2])
        }

/// Record and struct type operations
module StructTypes =
    
    /// Generate struct/record type from field definitions
    let createStructType (fields: (string * MLIRType) list): MLIRCombinator<MLIRType> =
        mlir {
            let fieldTypes = fields |> List.map snd
            return MLIRTypes.struct_ fieldTypes
        }
    
    /// Generate record constructor
    let createRecord (structType: MLIRType) (fieldValues: MLIRValue list): MLIRCombinator<MLIRValue> =
        mlir {
            let! result = nextSSA "record"
            let! recordPtr = Memory.alloca structType None
            
            // Initialize each field
            for i, value in List.indexed fieldValues do
                let! fieldIndex = Constants.intConstant i 32
                do! Memory.store value recordPtr [fieldIndex]
            
            return! Memory.load recordPtr []
        }
    
    /// Generate field access
    let accessField (recordValue: MLIRValue) (fieldIndex: int): MLIRCombinator<MLIRValue> =
        mlir {
            // For now, assume record is in memory
            let! result = nextSSA "field"
            let! recordPtr = Memory.alloca recordValue.Type None
            do! Memory.store recordValue recordPtr []
            let! index = Constants.intConstant fieldIndex 32
            return! Memory.load recordPtr [index]
        }
    
    /// Generate field update (functional update)
    let updateField (recordValue: MLIRValue) (fieldIndex: int) (newValue: MLIRValue): MLIRCombinator<MLIRValue> =
        mlir {
            let! result = nextSSA "updated_record"
            let! recordPtr = Memory.alloca recordValue.Type None
            
            // Copy original record
            do! Memory.store recordValue recordPtr []
            
            // Update specific field
            let! index = Constants.intConstant fieldIndex 32
            do! Memory.store newValue recordPtr [index]
            
            return! Memory.load recordPtr []
        }

/// Type inference support using Foundation patterns
module Inference =
    
    /// Infer type from literal constant
    let inferLiteralType (constant: SynConst): MLIRCombinator<MLIRType> =
        mlir {
            match constant with
            | SynConst.Int32 _ -> return MLIRTypes.i32
            | SynConst.Int64 _ -> return MLIRTypes.i64
            | SynConst.Int16 _ -> return MLIRTypes.i16
            | SynConst.SByte _ -> return MLIRTypes.i8
            | SynConst.Byte _ -> return MLIRTypes.i8
            | SynConst.UInt32 _ -> return MLIRTypes.i32
            | SynConst.UInt64 _ -> return MLIRTypes.i64
            | SynConst.UInt16 _ -> return MLIRTypes.i16
            | SynConst.Single _ -> return MLIRTypes.f32
            | SynConst.Double _ -> return MLIRTypes.f64
            | SynConst.Bool _ -> return MLIRTypes.i1
            | SynConst.String _ -> return MLIRTypes.string_
            | SynConst.Char _ -> return MLIRTypes.i8
            | SynConst.Unit -> return MLIRTypes.void_
            | _ -> return! fail "literal_inference" "Unsupported literal type"
        }
    
    /// Infer result type from binary operation
    let inferBinaryOpType (leftType: MLIRType) (rightType: MLIRType) (operator: string): MLIRCombinator<MLIRType> =
        mlir {
            match operator with
            | "+" | "-" | "*" | "/" | "%" ->
                // Arithmetic - promote to larger type
                if TypeAnalysis.areEqual leftType rightType then
                    return leftType
                elif TypeAnalysis.canConvertTo leftType rightType then
                    return rightType
                elif TypeAnalysis.canConvertTo rightType leftType then
                    return leftType
                else
                    return! fail "binary_inference" "Incompatible types for arithmetic operation"
                    
            | "=" | "<>" | "<" | "<=" | ">" | ">=" ->
                // Comparison always returns bool
                return MLIRTypes.i1
                
            | "&&" | "||" ->
                // Logical operations expect and return bool
                return MLIRTypes.i1
                
            | "&&&" | "|||" | "^^^" | "<<<" | ">>>" ->
                // Bitwise operations preserve input type
                return leftType
                
            | _ ->
                return! fail "binary_inference" (sprintf "Unknown binary operator: %s" operator)
        }