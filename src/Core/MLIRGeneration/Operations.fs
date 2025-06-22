module Core.MLIRGeneration.Operations

open Dabbit.Parsing.OakAst
open Core.MLIRGeneration.TypeSystem

/// Represents an MLIR operation with its attributes and results
type MLIROperation = {
    Name: string
    Operands: string list
    Results: string list
    Attributes: Map<string, string>
    ResultTypes: MLIRType list  // Enhanced to track result types
}

/// Helper to create operation with common defaults
let private createBaseOperation name operands results attributes resultTypes = {
    Name = name
    Operands = operands
    Results = results 
    Attributes = attributes
    ResultTypes = resultTypes
}

/// Helper to create operation with a single result
let private createSingleResultOp name operands resultId attributes resultType =
    createBaseOperation name operands [resultId] attributes [resultType]

/// Builds MLIR operations for different Oak AST expressions
let buildArithmeticOp (op: string) (lhs: string) (rhs: string) (resultType: MLIRType) =
    createSingleResultOp (sprintf "arith.%s" op) [lhs; rhs] "%result" Map.empty resultType

/// Creates function definition operation
let buildFuncOp (name: string) (args: (string * MLIRType) list) (returnType: MLIRType) =
    let argTypeStrs = args |> List.map (fun (_, argType) -> mlirTypeToString argType)
    let returnTypeStr = mlirTypeToString returnType
    let functionTypeStr = sprintf "(%s) -> %s" (String.concat ", " argTypeStrs) returnTypeStr
    
    createBaseOperation "func.func" [] [] 
        (Map.ofList [
            "sym_name", sprintf "\"%s\"" name
            "function_type", functionTypeStr
        ]) []

/// Creates call operation
let buildCallOp (funcName: string) (args: string list) (argTypes: MLIRType list) (resultType: MLIRType option) =
    let (results, resultTypes) = 
        match resultType with
        | Some typ when typ.Category <> VoidCategory -> (["%result"], [typ])
        | _ -> ([], [])
    
    createBaseOperation "func.call" args results 
        (Map.ofList ["callee", sprintf "\"%s\"" funcName]) 
        resultTypes

/// Creates return operation with type information
let buildReturnOp (values: string list) (valueTypes: MLIRType list) =
    createBaseOperation "func.return" values [] Map.empty []

/// Creates constant operation
let buildConstantOp (value: string) (mlirType: MLIRType) =
    createSingleResultOp "arith.constant" [] "%const" 
        (Map.ofList ["value", value]) 
        mlirType

/// Creates load operation
let buildLoadOp (memref: string) (memrefType: MLIRType) (resultType: MLIRType) =
    createSingleResultOp "memref.load" [memref] "%loaded" Map.empty resultType

/// Creates store operation
let buildStoreOp (value: string) (valueType: MLIRType) (memref: string) (memrefType: MLIRType) =
    createBaseOperation "memref.store" [value; memref] [] Map.empty []

/// Creates alloca operation for stack allocation
let buildAllocaOp (elementType: MLIRType) (size: int option) =
    let sizeStr = match size with Some s -> sprintf "%d x " s | None -> ""
    let typeStr = mlirTypeToString elementType
    let resultType = MLIRTypes.createMemRef elementType (match size with Some s -> [s] | None -> [])
    
    createSingleResultOp "memref.alloca" [] "%alloca" 
        (Map.ofList ["element_type", sprintf "%s%s" sizeStr typeStr]) 
        resultType

/// Creates a cast operation when types don't match exactly
let buildCastOp (sourceValue: string) (sourceType: MLIRType) (targetType: MLIRType) =
    // Determine operation name and attributes based on types
    let (opName, attributes) = 
        match sourceType.Category, targetType.Category with
        | IntegerCategory, IntegerCategory when sourceType.Width <> targetType.Width ->
            if (sourceType.Width |> Option.defaultValue 32) < (targetType.Width |> Option.defaultValue 32) then
                ("arith.extsi", Map.empty)  // Sign extension
            else
                ("arith.trunci", Map.empty) // Truncation
                
        | IntegerCategory, FloatCategory ->
            ("arith.sitofp", Map.empty) // Int to float
            
        | FloatCategory, IntegerCategory ->
            ("arith.fptosi", Map.empty) // Float to int
            
        | FloatCategory, FloatCategory when sourceType.Width <> targetType.Width ->
            if (sourceType.Width |> Option.defaultValue 32) < (targetType.Width |> Option.defaultValue 64) then
                ("arith.extf", Map.empty)   // Float extension
            else
                ("arith.truncf", Map.empty) // Float truncation
                
        | IntegerCategory, MemoryRefCategory ->
            ("memref.cast", Map.empty) // Int to pointer (unusual but handled)
            
        | MemoryRefCategory, MemoryRefCategory ->
            ("memref.cast", Map.empty) // Pointer type conversion
            
        | _ ->
            // Default to bitcast for other conversions
            ("llvm.bitcast", Map.empty)
    
    createSingleResultOp opName [sourceValue] "%cast" attributes targetType

/// Creates a comparison operation
let buildCompareOp (lhs: string) (rhs: string) (lhsType: MLIRType) (predicate: string) =
    let opName = 
        match lhsType.Category with
        | IntegerCategory -> "arith.cmpi"
        | FloatCategory -> "arith.cmpf"
        | _ -> "arith.cmpi"
    
    createSingleResultOp opName [lhs; rhs] "%cmp" 
        (Map.ofList [("predicate", predicate)]) 
        (MLIRTypes.createInteger 1)  // Always returns i1

/// Creates a global string constant
let buildGlobalString (name: string) (value: string) =
    let escapedValue = value.Replace("\\", "\\\\").Replace("\"", "\\\"")
    let constSize = escapedValue.Length + 1  // +1 for null terminator
    
    createBaseOperation "memref.global" [] [] 
        (Map.ofList [
            ("constant", "true")
            ("sym_name", sprintf "\"%s\"" name)
            ("type", sprintf "memref<%dxi8>" constSize)
            ("value", sprintf "dense<\"%s\\00\">" escapedValue)
        ]) []

/// Creates an address-of operation for globals
let buildAddressOfOp (globalName: string) =
    let resultType = MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
    createSingleResultOp "memref.get_global" [] "%addr" Map.empty resultType

/// Creates a conditional branch operation
let buildCondBranchOp (condition: string) (trueDest: string) (falseDest: string) =
    createBaseOperation "cond_br" [condition] [] 
        (Map.ofList [
            ("trueDest", sprintf "^%s" trueDest)
            ("falseDest", sprintf "^%s" falseDest)
        ]) []

/// Creates an unconditional branch operation
let buildBranchOp (dest: string) =
    createBaseOperation "br" [] [] 
        (Map.ofList [("dest", sprintf "^%s" dest)]) 
        []

/// Converts MLIROperation to string representation
let operationToString (op: MLIROperation) : string =
    let resultStr = if op.Results.IsEmpty then "" else sprintf "%s = " (String.concat ", " op.Results)
    let operandStr = if op.Operands.IsEmpty then "" else sprintf "(%s)" (String.concat ", " op.Operands)
    
    let attrStr = 
        if op.Attributes.IsEmpty then ""
        else 
            op.Attributes 
            |> Map.toList 
            |> List.map (fun (k, v) -> sprintf "%s = %s" k v)
            |> String.concat ", "
            |> sprintf " {%s}"
    
    // Add type information for result types
    let typeStr =
        if op.ResultTypes.IsEmpty then ""
        else
            let typeStrs = op.ResultTypes |> List.map mlirTypeToString
            sprintf " : %s" (String.concat ", " typeStrs)
    
    sprintf "%s%s%s%s%s" resultStr op.Name operandStr attrStr typeStr