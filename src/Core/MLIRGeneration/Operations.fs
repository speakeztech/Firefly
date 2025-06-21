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
    let argTypeStrs = args |> List.map (fun (_, argType) -> 
        match argType.Category with
        | IntegerCategory -> 
            match argType.Width with 
            | Some width -> sprintf "i%d" width
            | None -> "i32"
        | FloatCategory -> 
            match argType.Width with 
            | Some width -> sprintf "f%d" width
            | None -> "f32"
        | VoidCategory -> "()"
        | _ -> "i32")
    
    let returnTypeStr = 
        match returnType.Category with
        | IntegerCategory -> 
            match returnType.Width with 
            | Some width -> sprintf "i%d" width
            | None -> "i32"
        | FloatCategory -> 
            match returnType.Width with 
            | Some width -> sprintf "f%d" width
            | None -> "f32"
        | VoidCategory -> "()"
        | _ -> "i32"
    
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
let buildCallOp (funcName: string) (args: string list) (resultTypes: MLIRType list) =
    let results = if resultTypes.IsEmpty then [] else ["%result"]
    {
        Name = "func.call"
        Operands = args
        Results = results
        Attributes = Map.ofList ["callee", sprintf "\"%s\"" funcName]
        ResultTypes = resultTypes
    }

/// Creates return operation
let buildReturnOp (values: string list) =
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
    let typeStr = 
        match mlirType.Category with
        | IntegerCategory -> 
            match mlirType.Width with 
            | Some width -> sprintf "i%d" width
            | None -> "i32"
        | FloatCategory -> 
            match mlirType.Width with 
            | Some width -> sprintf "f%d" width
            | None -> "f32"
        | _ -> "i32"
    
    {
        Name = "arith.constant"
        Operands = []
        Results = [resultId]
        Attributes = Map.ofList ["value", value]
        ResultTypes = [mlirType]
    }

/// Creates load operation
let buildLoadOp (memref: string) (resultType: MLIRType) =
    let resultId = "%loaded"
    {
        Name = "memref.load"
        Operands = [memref]
        Results = [resultId]
        Attributes = Map.empty
        ResultTypes = [resultType]
    }

/// Creates store operation
let buildStoreOp (value: string) (memref: string) =
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
    let typeStr = match elementType.Category with
                  | IntegerCategory -> 
                      match elementType.Width with 
                      | Some width -> sprintf "i%d" width
                      | None -> "i32"
                  | FloatCategory -> 
                      match elementType.Width with 
                      | Some width -> sprintf "f%d" width
                      | None -> "f32"
                  | _ -> "i32"
    
    {
        Name = "memref.alloca"
        Operands = []
        Results = [resultId]
        Attributes = Map.ofList ["element_type", sprintf "%s%s" sizeStr typeStr]
        ResultTypes = [MLIRTypes.createMemRef elementType []]
    }

/// Converts MLIROperation to string representation
let operationToString (op: MLIROperation) : string =
    let resultStr = if op.Results.IsEmpty then "" else sprintf "%s = " (String.concat ", " op.Results)
    let operandStr = if op.Operands.IsEmpty then "" else sprintf "(%s)" (String.concat ", " op.Operands)
    let attrPairs = op.Attributes |> Map.toList |> List.map (fun (k, v) -> sprintf "%s = %s" k v)
    let attrStr = if attrPairs.IsEmpty then "" else sprintf " {%s}" (String.concat ", " attrPairs)
    
    sprintf "%s%s%s%s" resultStr op.Name operandStr attrStr