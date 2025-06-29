module Dabbit.CodeGeneration.MLIRTypeOperations

open FSharp.Compiler.Syntax
open Core.Types.TypeSystem
open Core.XParsec.Foundation
open Dabbit.CodeGeneration.MLIREmitter
open Dabbit.CodeGeneration.MLIRSyntax
open Dabbit.CodeGeneration.MLIRBuiltins

/// Utility function for mapping operations
let rec mapM (f: 'a -> MLIRBuilder<'b>) (list: 'a list): MLIRBuilder<'b list> =
    mlir {
        match list with
        | [] -> return []
        | head :: tail ->
            let! mappedHead = f head
            let! mappedTail = mapM f tail
            return mappedHead :: mappedTail
    }

/// Helper functions for type analysis
let areTypesEqual (type1: MLIRType) (type2: MLIRType): bool =
    type1.Category = type2.Category && type1.BitWidth = type2.BitWidth

let canConvertImplicitly (sourceType: MLIRType) (targetType: MLIRType): bool =
    match sourceType.Category, targetType.Category with
    | Integer, Integer -> true
    | Float, Float -> true
    | Integer, Float -> true  // Widening conversion
    | _ -> false

/// Convert between integer types
let convertInteger (sourceType: MLIRType) (targetType: MLIRType) (value: MLIRValue) (result: string): MLIRBuilder<MLIRValue> =
    mlir {
        let sourceWidth = sourceType.BitWidth |> Option.defaultValue 32
        let targetWidth = targetType.BitWidth |> Option.defaultValue 32
        let sourceTypeStr = mlirTypeToString sourceType
        let targetTypeStr = mlirTypeToString targetType
        
        if sourceWidth = targetWidth then
            return value  // No conversion needed
        elif sourceWidth < targetWidth then
            // Sign extend
            do! emitLine (sprintf "%s = arith.extsi %s : %s to %s" result value.SSA sourceTypeStr targetTypeStr)
            return createValue result targetType
        else
            // Truncate
            do! emitLine (sprintf "%s = arith.trunci %s : %s to %s" result value.SSA sourceTypeStr targetTypeStr)
            return createValue result targetType
    }

/// Convert between float types
let convertFloat (sourceType: MLIRType) (targetType: MLIRType) (value: MLIRValue) (result: string): MLIRBuilder<MLIRValue> =
    mlir {
        let sourceWidth = sourceType.BitWidth |> Option.defaultValue 32
        let targetWidth = targetType.BitWidth |> Option.defaultValue 32
        let sourceTypeStr = mlirTypeToString sourceType
        let targetTypeStr = mlirTypeToString targetType
        
        if sourceWidth = targetWidth then
            return value  // No conversion needed
        elif sourceWidth < targetWidth then
            do! emitLine (sprintf "%s = arith.extf %s : %s to %s" result value.SSA sourceTypeStr targetTypeStr)
            return createValue result targetType
        else
            do! emitLine (sprintf "%s = arith.truncf %s : %s to %s" result value.SSA sourceTypeStr targetTypeStr)
            return createValue result targetType
    }

/// Convert between memref types
let convertMemref (sourceType: MLIRType) (targetType: MLIRType) (value: MLIRValue) (result: string): MLIRBuilder<MLIRValue> =
    mlir {
        let sourceTypeStr = mlirTypeToString sourceType
        let targetTypeStr = mlirTypeToString targetType
        do! emitLine (sprintf "%s = memref.cast %s : %s to %s" result value.SSA sourceTypeStr targetTypeStr)
        return createValue result targetType
    }

/// Explicit type conversions
let explicitConvert (sourceType: MLIRType) (targetType: MLIRType) (value: MLIRValue): MLIRBuilder<MLIRValue> =
    mlir {
        let! result = nextSSA "convert"
        let sourceTypeStr = mlirTypeToString sourceType
        let targetTypeStr = mlirTypeToString targetType
        
        match sourceType.Category, targetType.Category with
        // Integer conversions
        | Integer, Integer ->
            return! convertInteger sourceType targetType value result
            
        // Float conversions
        | Float, Float ->
            return! convertFloat sourceType targetType value result
            
        // Integer to float
        | Integer, Float ->
            do! emitLine (sprintf "%s = arith.sitofp %s : %s to %s" result value.SSA sourceTypeStr targetTypeStr)
            return createValue result targetType
            
        // Float to integer
        | Float, Integer ->
            do! emitLine (sprintf "%s = arith.fptosi %s : %s to %s" result value.SSA sourceTypeStr targetTypeStr)
            return createValue result targetType
            
        // Pointer/reference conversions
        | MemRef, MemRef ->
            return! convertMemref sourceType targetType value result
            
        | _ ->
            return! failHard "explicit_convert" 
                (sprintf "Unsupported conversion from %s to %s" sourceTypeStr targetTypeStr)
    }

/// Check if two types are compatible for operations
let areCompatible (type1: MLIRType) (type2: MLIRType): bool =
    areTypesEqual type1 type2 || 
    canConvertImplicitly type1 type2 || 
    canConvertImplicitly type2 type1

/// Implicit type conversions between compatible types
let implicitConvert (sourceType: MLIRType) (targetType: MLIRType) (value: MLIRValue): MLIRBuilder<MLIRValue> =
    mlir {
        // Check if the types are equal first
        let! result = 
            if areTypesEqual sourceType targetType then
                // Just return the same value
                mlir.Return value
            elif canConvertImplicitly sourceType targetType then
                // Apply the conversion
                explicitConvert sourceType targetType value
            else
                // Format error message with pre-computed strings
                let sourceTypeStr = mlirTypeToString sourceType
                let targetTypeStr = mlirTypeToString targetType
                let errorMsg = sprintf "Cannot implicitly convert %s to %s" sourceTypeStr targetTypeStr
                failHard "implicit_convert" errorMsg
                
        return result
    }

/// Validate assignment compatibility
let validateAssignment (targetType: MLIRType) (sourceValue: MLIRValue): MLIRBuilder<MLIRValue> =
    mlir {
        let sourceType = parseTypeFromString sourceValue.Type
        
        // Use let! to ensure all branches return the same type
        let! result = 
            if areTypesEqual targetType sourceType then
                // Use mlir.Return instead of return
                mlir.Return sourceValue
            elif canConvertImplicitly sourceType targetType then
                // This already returns MLIRBuilder<MLIRValue>
                implicitConvert sourceType targetType sourceValue
            else
                // Pre-compute formatted strings
                let sourceTypeStr = mlirTypeToString sourceType
                let targetTypeStr = mlirTypeToString targetType
                let errorMsg = sprintf "Cannot assign %s to %s" sourceTypeStr targetTypeStr
                // This returns MLIRBuilder<MLIRValue>
                failHard "assignment" errorMsg
                
        return result
    }

/// Map primitive F# types to MLIR types
let mapPrimitiveType (typeName: string): MLIRBuilder<MLIRType> =
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
            return! failHard "primitive_type" (sprintf "Unknown primitive type: %s" typeName)
    }

/// Convert F# SynType to MLIRType using proper Foundation patterns
let rec synTypeToMLIR (synType: SynType): MLIRBuilder<MLIRType> =
    mlir {
        match synType with
        | SynType.LongIdent(SynLongIdent([id], _, _)) ->
            return! mapPrimitiveType id.idText
            
        | SynType.App(typeName, _, typeArgs, _, _, _, _) ->
            return! mapGenericType typeName typeArgs
            
        | SynType.Fun(argType, returnType, _, _) ->
            let! argMLIR = synTypeToMLIR argType
            let! retMLIR = synTypeToMLIR returnType
            return MLIRTypes.func [argMLIR] retMLIR
            
        | SynType.Paren(innerType, _) ->
            return! synTypeToMLIR innerType
            
        | SynType.Tuple(isStruct, typeSegments, _) ->
            let componentTypes = 
                typeSegments 
                |> List.choose (function 
                    | SynTupleTypeSegment.Type(synType) -> Some synType
                    | _ -> None)
            let! mlirTypes = mapM synTypeToMLIR componentTypes
            return MLIRTypes.struct_ (mlirTypes |> List.mapi (fun i t -> sprintf "field%d" i, t))
            
        | _ ->
            return! failHard "type_mapping" (sprintf "Unsupported type pattern: %A" synType)
    }

/// Map generic F# types to MLIR types
and mapGenericType (typeName: SynType) (typeArgs: SynType list): MLIRBuilder<MLIRType> =
    mlir {
        match typeName with
        | SynType.LongIdent(SynLongIdent([id], _, _)) when id.idText = "Span" ->
            match typeArgs with
            | [elementType] ->
                let! elemType = synTypeToMLIR elementType
                return MLIRTypes.memref elemType
            | _ ->
                return! failHard "span_type" "Span must have exactly one type argument"
                
        | SynType.LongIdent(SynLongIdent([id], _, _)) when id.idText = "ReadOnlySpan" ->
            match typeArgs with
            | [elementType] ->
                let! elemType = synTypeToMLIR elementType
                return MLIRTypes.memref elemType  // Read-only spans still use memref
            | _ ->
                return! failHard "readonly_span_type" "ReadOnlySpan must have exactly one type argument"
                
        | SynType.LongIdent(SynLongIdent([id], _, _)) when id.idText = "option" ->
            match typeArgs with
            | [valueType] ->
                let! valType = synTypeToMLIR valueType
                // Option represented as struct with tag + value
                return MLIRTypes.struct_ [("tag", MLIRTypes.i1); ("value", valType)]
            | _ ->
                return! failHard "option_type" "Option must have exactly one type argument"
                
        | _ ->
            return! failHard "generic_type" (sprintf "Unsupported generic type: %A" typeName)
    }

/// Generate discriminated union type
let createUnionType (cases: (string * MLIRType list) list): MLIRBuilder<MLIRType> =
    mlir {
        // DU represented as struct with tag + largest case data
        let maxCaseSize = cases 
                        |> List.map (fun (_, types) -> List.length types)
                        |> List.max
        
        // Create struct with tag (i32) + data slots
        let tagType = MLIRTypes.i32
        let dataTypes = List.replicate maxCaseSize MLIRTypes.i64  // Use i64 for general storage
        
        let allFields = ("tag", tagType) :: (dataTypes |> List.mapi (fun i t -> sprintf "data%d" i, t))
        return MLIRTypes.struct_ allFields
    }

/// Generate union case constructor
let createUnionCase (unionType: MLIRType) (caseIndex: int) (caseData: MLIRValue list): MLIRBuilder<MLIRValue> =
    mlir {
        let! result = nextSSA "union_case"
        let! unionPtr = nextSSA "union_ptr"
        
        // Allocate union instance
        do! emitLine (sprintf "%s = memref.alloca() : memref<%s>" unionPtr (mlirTypeToString unionType))
        
        // Set tag
        let! tagIndex = Constants.intConstant 0 32  // Tag is at index 0
        let! tag = Constants.intConstant caseIndex 32
        do! emitLine (sprintf "memref.store %s, %s[%s] : memref<%s>" tag.SSA unionPtr tagIndex.SSA (mlirTypeToString unionType))
        
        // Store case data using List operations
        let! _ = 
            caseData 
            |> List.mapi (fun i d -> i, d)
            |> mapM (fun (index, data) -> 
                mlir {
                    let! dataIndex = Constants.intConstant (index + 1) 32  // Data starts at index 1
                    do! emitLine (sprintf "memref.store %s, %s[%s] : memref<%s>" data.SSA unionPtr dataIndex.SSA (mlirTypeToString unionType))
                    return ()
                })
        
        // Load the complete union
        do! emitLine (sprintf "%s = memref.load %s : memref<%s>" result unionPtr (mlirTypeToString unionType))
        return createValue result unionType
    }

/// Generate pattern match against union case
let matchUnionCase (unionValue: MLIRValue) (caseIndex: int): MLIRBuilder<MLIRValue> =
    mlir {
        let! result = nextSSA "case_match"
        let! unionPtr = nextSSA "union_ptr"
        let unionType = parseTypeFromString unionValue.Type
        
        // Store union to memory for access
        do! emitLine (sprintf "%s = memref.alloca() : memref<%s>" unionPtr (mlirTypeToString unionType))
        do! emitLine (sprintf "memref.store %s, %s : memref<%s>" unionValue.SSA unionPtr (mlirTypeToString unionType))
        
        // Load tag
        let! tagIndex = Constants.intConstant 0 32
        do! emitLine (sprintf "%s = memref.load %s[%s] : memref<%s>" result unionPtr tagIndex.SSA (mlirTypeToString unionType))
        
        // Compare with expected case index
        let! expectedTag = Constants.intConstant caseIndex 32
        let! compareResult = nextSSA "tag_cmp"
        do! emitLine (sprintf "%s = arith.cmpi eq, %s, %s : i32" compareResult result expectedTag.SSA)
        
        return createValue compareResult MLIRTypes.i1
    }

/// Infer type from literal constant
let inferLiteralType (constant: SynConst): MLIRBuilder<MLIRType> =
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
        | _ -> return! failHard "literal_inference" (sprintf "Unsupported literal type: %A" constant)
    }

/// Infer result type from binary operation
let inferBinaryOpType (leftType: MLIRType) (rightType: MLIRType) (operator: string): MLIRBuilder<MLIRType> =
    mlir {
        match operator with
        | "+" | "-" | "*" | "/" | "%" when areTypesEqual leftType rightType ->
            return leftType
        | "+." | "-." | "*." | "/." when areTypesEqual leftType rightType ->
            return leftType
        | "=" | "<>" | "<" | "<=" | ">" | ">=" ->
            return MLIRTypes.i1
        | "&&" | "||" ->
            return MLIRTypes.i1
        | _ ->
            return! failHard "binary_op_inference" (sprintf "Cannot infer type for operator %s with types %s and %s" 
                                                        operator (mlirTypeToString leftType) (mlirTypeToString rightType))
    }

/// Infer type from expression context
let inferFromContext (expectedType: MLIRType option) (actualType: MLIRType): MLIRBuilder<MLIRType> =
    mlir {
        match expectedType with
        | Some expected when areTypesEqual expected actualType -> return expected
        | Some expected when canConvertImplicitly actualType expected -> return expected
        | _ -> return actualType
    }

/// Allocate memory for a type
let alloca (typ: MLIRType) (count: MLIRValue option): MLIRBuilder<MLIRValue> =
    mlir {
        let! ptr = nextSSA "alloca"
        let typeStr = mlirTypeToString typ
        
        // Create allocation command string outside the conditional
        let allocCmd = 
            match count with
            | Some countVal -> 
                sprintf "%s = memref.alloca(%s) : memref<%s>" ptr countVal.SSA typeStr
            | None -> 
                sprintf "%s = memref.alloca() : memref<%s>" ptr typeStr
                
        // Execute the command string - no conditional needed here
        do! emitLine allocCmd
        
        return createValue ptr (MLIRTypes.memref typ)
    }

/// Store value to memory
let store (value: MLIRValue) (ptr: MLIRValue) (indices: MLIRValue list): MLIRBuilder<unit> =
    mlir {
        let indexStr = 
            if List.isEmpty indices then ""
            else "[" + (indices |> List.map (fun idx -> idx.SSA) |> String.concat ", ") + "]"
        
        do! emitLine (sprintf "memref.store %s, %s%s : %s" value.SSA ptr.SSA indexStr ptr.Type)
    }

/// Load value from memory
let load (ptr: MLIRValue) (indices: MLIRValue list): MLIRBuilder<MLIRValue> =
    mlir {
        let! result = nextSSA "load"
        let indexStr = 
            if List.isEmpty indices then ""
            else "[" + (indices |> List.map (fun idx -> idx.SSA) |> String.concat ", ") + "]"
        
        do! emitLine (sprintf "%s = memref.load %s%s : %s" result ptr.SSA indexStr ptr.Type)
        
        // Extract element type from memref type
        let elementType = 
            match ptr.Type with
            | typeStr when typeStr.StartsWith("memref<") && typeStr.EndsWith(">") ->
                let inner = typeStr.Substring(7, typeStr.Length - 8)
                parseTypeFromString inner
            | _ -> MLIRTypes.i32
        
        return createValue result elementType
    }