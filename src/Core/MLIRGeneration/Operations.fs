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
    {
        Name = "func"
        Operands = []
        Results = []
        Attributes = 
            Map.ofList [
                "sym_name", sprintf "\"%s\"" name
                "function_type", "()"
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
open Dabbit.Parsing.OakAst
open Firefly.Core.MLIRGeneration.TypeSystem

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
    {
        Name = "func"
        Operands = []
        Results = []
        Attributes = 
            Map.ofList [
                "sym_name", sprintf "\"%s\"" name
                "function_type", "()"
            ]
        ResultTypes = []
    }
module Core.MLIRGeneration.Operations
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
    {
        Name = "func"
        Operands = []
        Results = []
        Attributes = 
            Map.ofList [
                "sym_name", sprintf "\"%s\"" name
                "function_type", "()"
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
    {
        Name = "func"
        Operands = []
        Results = []
        Attributes = 
            Map.ofList [
                "sym_name", sprintf "\"%s\"" name
                "function_type", "()"
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
