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

/// Builds MLIR operations for different Oak AST expressions
let buildArithmeticOp (op: string) (lhs: string) (rhs: string) (resultType: MLIRType) =
    let resultId = "%result"
    {
        Name = sprintf "arith.%s" op
        Operands = [lhs; rhs]
        Results = [resultId]
        Attributes = Map.empty
        ResultTypes = [resultType]
    }

/// Creates function definition operation
let buildFuncOp (name: string) (args: (string * MLIRType) list) (returnType: MLIRType) =
    let argTypeStrs = args |> List.map (fun (_, argType) -> mlirTypeToString argType)
    let returnTypeStr = mlirTypeToString returnType
    let functionTypeStr = sprintf "(%s) -> %s" (String.concat ", " argTypeStrs) returnTypeStr
    
    {
        Name = "func.func"
        Operands = []
        Results = []
        Attributes = 
            Map.ofList [
                "sym_name", sprintf "\"%s\"" name
                "function_type", functionTypeStr
            ]
        ResultTypes = []
    }

/// Creates call operation
let buildCallOp (funcName: string) (args: string list) (argTypes: MLIRType list) (resultType: MLIRType option) =
    let results = 
        match resultType with
        | Some typ when typ.Category <> VoidCategory -> ["%result"]
        | _ -> []
    
    let resultTypes = 
        match resultType with
        | Some typ when typ.Category <> VoidCategory -> [typ]
        | _ -> []
    
    {
        Name = "func.call"
        Operands = args
        Results = results
        Attributes = Map.ofList ["callee", sprintf "\"%s\"" funcName]
        ResultTypes = resultTypes
    }

/// Creates return operation with type information
let buildReturnOp (values: string list) (valueTypes: MLIRType list) =
    {
        Name = "func.return"
        Operands = values
        Results = []
        Attributes = Map.empty
        ResultTypes = []
    }

/// Creates constant operation
let buildConstantOp (value: string) (mlirType: MLIRType) =
    let resultId = "%const"
    let typeStr = mlirTypeToString mlirType
    
    {
        Name = "arith.constant"
        Operands = []
        Results = [resultId]
        Attributes = Map.ofList ["value", value]
        ResultTypes = [mlirType]
    }

/// Creates load operation
let buildLoadOp (memref: string) (memrefType: MLIRType) (resultType: MLIRType) =
    let resultId = "%loaded"
    {
        Name = "memref.load"
        Operands = [memref]
        Results = [resultId]
        Attributes = Map.empty
        ResultTypes = [resultType]
    }

/// Creates store operation
let buildStoreOp (value: string) (valueType: MLIRType) (memref: string) (memrefType: MLIRType) =
    {
        Name = "memref.store"
        Operands = [value; memref]
        Results = []
        Attributes = Map.empty
        ResultTypes = []
    }

/// Creates alloca operation for stack allocation
let buildAllocaOp (elementType: MLIRType) (size: int option) =
    let resultId = "%alloca"
    let sizeStr = match size with Some s -> sprintf "%d x " s | None -> ""
    let typeStr = mlirTypeToString elementType
    
    let resultType = MLIRTypes.createMemRef elementType (match size with Some s -> [s] | None -> [])
    
    {
        Name = "memref.alloca"
        Operands = []
        Results = [resultId]
        Attributes = Map.ofList ["element_type", sprintf "%s%s" sizeStr typeStr]
        ResultTypes = [resultType]
    }

/// Creates a cast operation when types don't match exactly
let buildCastOp (sourceValue: string) (sourceType: MLIRType) (targetType: MLIRType) =
    let resultId = "%cast"
    
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
    
    {
        Name = opName
        Operands = [sourceValue]
        Results = [resultId]
        Attributes = attributes
        ResultTypes = [targetType]
    }

/// Creates a comparison operation
let buildCompareOp (lhs: string) (rhs: string) (lhsType: MLIRType) (predicate: string) =
    let resultId = "%cmp"
    
    let (opName, attributes) =
        match lhsType.Category with
        | IntegerCategory -> 
            ("arith.cmpi", Map.ofList [("predicate", predicate)])
        | FloatCategory -> 
            ("arith.cmpf", Map.ofList [("predicate", predicate)])
        | _ -> 
            ("arith.cmpi", Map.ofList [("predicate", predicate)])
    
    {
        Name = opName
        Operands = [lhs; rhs]
        Results = [resultId]
        Attributes = attributes
        ResultTypes = [MLIRTypes.createInteger 1]  // Always returns i1
    }

/// Creates a global string constant
let buildGlobalString (name: string) (value: string) =
    let escapedValue = value.Replace("\\", "\\\\").Replace("\"", "\\\"")
    let constSize = escapedValue.Length + 1  // +1 for null terminator
    
    {
        Name = "memref.global"
        Operands = []
        Results = []
        Attributes = Map.ofList [
            ("constant", "true")
            ("sym_name", sprintf "\"%s\"" name)
            ("type", sprintf "memref<%dxi8>" constSize)
            ("value", sprintf "dense<\"%s\\00\">" escapedValue)
        ]
        ResultTypes = []
    }

/// Creates an address-of operation for globals
let buildAddressOfOp (globalName: string) =
    let resultId = "%addr"
    let resultType = MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
    
    {
        Name = "memref.get_global"
        Operands = []
        Results = [resultId]
        Attributes = Map.empty
        ResultTypes = [resultType]
    }

/// Creates a conditional branch operation
let buildCondBranchOp (condition: string) (trueDest: string) (falseDest: string) =
    {
        Name = "cond_br"
        Operands = [condition]
        Results = []
        Attributes = Map.ofList [
            ("trueDest", sprintf "^%s" trueDest)
            ("falseDest", sprintf "^%s" falseDest)
        ]
        ResultTypes = []
    }

/// Creates an unconditional branch operation
let buildBranchOp (dest: string) =
    {
        Name = "br"
        Operands = []
        Results = []
        Attributes = Map.ofList [("dest", sprintf "^%s" dest)]
        ResultTypes = []
    }

/// Converts MLIROperation to string representation
let operationToString (op: MLIROperation) : string =
    let resultStr = if op.Results.IsEmpty then "" else sprintf "%s = " (String.concat ", " op.Results)
    let operandStr = if op.Operands.IsEmpty then "" else sprintf "(%s)" (String.concat ", " op.Operands)
    
    let attrPairs = op.Attributes |> Map.toList |> List.map (fun (k, v) -> sprintf "%s = %s" k v)
    let attrStr = if attrPairs.IsEmpty then "" else sprintf " {%s}" (String.concat ", " attrPairs)
    
    // Add type information for result types
    let typeStr =
        if op.ResultTypes.IsEmpty then 
            ""
        else
            let typeStrs = op.ResultTypes |> List.map mlirTypeToString
            sprintf " : %s" (String.concat ", " typeStrs)
    
    sprintf "%s%s%s%s%s" resultStr op.Name operandStr attrStr typeStr