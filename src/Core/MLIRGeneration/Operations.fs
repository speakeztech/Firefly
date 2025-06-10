module Core.MLIRGeneration.Operations

open Dabbit.Parsing.OakAst
open Core.MLIRGeneration.TypeSystem

/// Represents an MLIR operation with its attributes and results
type MLIROperation = {
    Name: string
    Operands: string list
    Results: string list
    Attributes: Map<string, string>
    ResultTypes: MLIRType list
}

/// Creates an arithmetic operation with two operands
let buildArithmeticOp (op: string) (lhs: string) (rhs: string) (resultType: MLIRType) =
    let resultId = "%result"
    {
        Name = sprintf "arith.%s" op
        Operands = [lhs; rhs]
        Results = [resultId]
        Attributes = Map.empty
        ResultTypes = [resultType]
    }

/// Creates a function definition operation
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

/// Creates a function call operation
let buildCallOp (funcName: string) (args: string list) (resultTypes: MLIRType list) =
    let results = if resultTypes.IsEmpty then [] else ["%result"]
    {
        Name = "func.call"
        Operands = args
        Results = results
        Attributes = Map.ofList ["callee", sprintf "\"%s\"" funcName]
        ResultTypes = resultTypes
    }

/// Creates a return operation
let buildReturnOp (values: string list) =
    {
        Name = "func.return"
        Operands = values
        Results = []
        Attributes = Map.empty
        ResultTypes = []
    }

/// Creates a constant operation
let buildConstantOp (value: string) (mlirType: MLIRType) =
    let resultId = "%const"
    {
        Name = "arith.constant"
        Operands = []
        Results = [resultId]
        Attributes = Map.ofList ["value", value]
        ResultTypes = [mlirType]
    }

/// Creates a memory load operation
let buildLoadOp (memref: string) (resultType: MLIRType) =
    let resultId = "%loaded"
    {
        Name = "memref.load"
        Operands = [memref]
        Results = [resultId]
        Attributes = Map.empty
        ResultTypes = [resultType]
    }

/// Creates a memory store operation
let buildStoreOp (value: string) (memref: string) =
    {
        Name = "memref.store"
        Operands = [value; memref]
        Results = []
        Attributes = Map.empty
        ResultTypes = []
    }

/// Creates a stack memory allocation operation
let buildAllocaOp (elementType: MLIRType) (size: int option) =
    let resultId = "%alloca"
    let sizeStr = match size with Some s -> sprintf "%d x " s | None -> ""
    let typeStr = mlirTypeToString elementType
    
    {
        Name = "memref.alloca"
        Operands = []
        Results = [resultId]
        Attributes = Map.ofList ["element_type", sprintf "%s%s" sizeStr typeStr]
        ResultTypes = [MemRef(elementType, match size with Some s -> [s] | None -> [])]
    }

/// Creates a global memory allocation operation
let buildGlobalOp (name: string) (value: string) (elementType: MLIRType) (isConstant: bool) =
    {
        Name = "memref.global"
        Operands = []
        Results = []
        Attributes = Map.ofList [
            "sym_name", sprintf "\"%s\"" name
            "type", mlirTypeToString elementType
            "initial_value", value
            "constant", if isConstant then "true" else "false"
        ]
        ResultTypes = []
    }

/// Creates a global memory reference operation
let buildGetGlobalOp (name: string) (elementType: MLIRType) =
    let resultId = "%global_ref"
    {
        Name = "memref.get_global"
        Operands = []
        Results = [resultId]
        Attributes = Map.ofList ["name", sprintf "\"%s\"" name]
        ResultTypes = [elementType]
    }

/// Creates a conditional branch operation
let buildCondBranchOp (condition: string) (trueLabel: string) (falseLabel: string) =
    {
        Name = "cf.cond_br"
        Operands = [condition]
        Results = []
        Attributes = Map.ofList [
            "true_dest", sprintf "^%s" trueLabel
            "false_dest", sprintf "^%s" falseLabel
        ]
        ResultTypes = []
    }

/// Creates an unconditional branch operation
let buildBranchOp (label: string) =
    {
        Name = "cf.br"
        Operands = []
        Results = []
        Attributes = Map.ofList ["dest", sprintf "^%s" label]
        ResultTypes = []
    }

/// Creates a branch operation with arguments
let buildBranchWithArgsOp (label: string) (args: string list) (argTypes: MLIRType list) =
    let argsWithTypes = 
        List.zip args argTypes
        |> List.map (fun (arg, ty) -> sprintf "%s : %s" arg (mlirTypeToString ty))
        |> String.concat ", "
    
    {
        Name = "cf.br"
        Operands = args
        Results = []
        Attributes = Map.ofList ["dest", sprintf "^%s(%s)" label argsWithTypes]
        ResultTypes = []
    }

/// Creates a block definition with arguments
let buildBlockOp (name: string) (args: (string * MLIRType) list) =
    let argsStr = 
        args
        |> List.map (fun (arg, ty) -> sprintf "%s: %s" arg (mlirTypeToString ty))
        |> String.concat ", "
    
    {
        Name = sprintf "^%s%s:" name (if args.IsEmpty then "" else sprintf "(%s)" argsStr)
        Operands = []
        Results = []
        Attributes = Map.empty
        ResultTypes = []
    }

/// Converts MLIROperation to string representation
let operationToString (op: MLIROperation) : string =
    let resultStr = if op.Results.IsEmpty then "" else sprintf "%s = " (String.concat ", " op.Results)
    let operandStr = if op.Operands.IsEmpty then "" else sprintf "(%s)" (String.concat ", " op.Operands)
    let attrPairs = op.Attributes |> Map.toList |> List.map (fun (k, v) -> sprintf "%s = %s" k v)
    let attrStr = if attrPairs.IsEmpty then "" else sprintf " {%s}" (String.concat ", " attrPairs)
    
    sprintf "%s%s%s%s" resultStr op.Name operandStr attrStr

/// Converts multiple operations to string with proper indentation
let operationsToString (operations: MLIROperation list) : string =
    operations
    |> List.map (fun op -> 
        if op.Name.StartsWith("^") then
            operationToString op // Don't indent block labels
        else
            "  " + operationToString op)
    |> String.concat "\n"

/// Creates an I/O operation for printf
let buildPrintfOp (formatPtr: string) (args: string list) =
    let resultId = "%printf_result"
    let allArgs = formatPtr :: args
    let argTypes = "memref<?xi8>" :: List.replicate args.Length "i32"
    
    {
        Name = "func.call"
        Operands = allArgs
        Results = [resultId]
        Attributes = Map.ofList ["callee", "\"printf\""]
        ResultTypes = [Integer 32]
    }

/// Creates an external function declaration
let buildExternalFuncOp (name: string) (paramTypes: MLIRType list) (returnType: MLIRType) =
    let paramTypeStrs = paramTypes |> List.map mlirTypeToString
    let returnTypeStr = mlirTypeToString returnType
    
    {
        Name = "func.func"
        Operands = []
        Results = []
        Attributes = 
            Map.ofList [
                "sym_name", sprintf "\"%s\"" name
                "function_type", sprintf "(%s) -> %s" (String.concat ", " paramTypeStrs) returnTypeStr
                "private", "unit"
            ]
        ResultTypes = []
    }