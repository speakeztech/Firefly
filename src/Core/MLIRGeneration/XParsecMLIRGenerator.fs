module Core.MLIRGeneration.XParsecMLIRGenerator

open System
open Dabbit.Parsing.OakAst
open Dabbit.Parsing.Translator
open Core.MLIRGeneration.TypeSystem
open Core.MLIRGeneration.Operations

/// Variable counter for generating unique SSA values
let mutable private varCounter = 0

/// Generates a unique SSA variable name
let private nextVar() =
    varCounter <- varCounter + 1
    sprintf "%%v%d" varCounter

/// Resets the variable counter for a new function
let private resetVarCounter() =
    varCounter <- 0

/// Converts Oak type to MLIR type string
let rec private oakTypeToMLIRString (oakType: OakType) : string =
    match oakType with
    | IntType -> "i32"
    | FloatType -> "f32"
    | BoolType -> "i1"
    | StringType -> "!llvm.ptr<i8>"
    | UnitType -> "()"
    | ArrayType elemType -> sprintf "memref<?x%s>" (oakTypeToMLIRString elemType)
    | FunctionType(paramTypes, returnType) ->
        let paramStrs = paramTypes |> List.map oakTypeToMLIRString |> String.concat ", "
        let returnStr = oakTypeToMLIRString returnType
        sprintf "(%s) -> %s" paramStrs returnStr
    | StructType fields ->
        let fieldStrs = fields |> List.map (snd >> oakTypeToMLIRString) |> String.concat ", "
        sprintf "!llvm.struct<(%s)>" fieldStrs
    | UnionType _ -> "!llvm.struct<(i8, i64)>"

/// Generates MLIR for Oak expressions
let rec private generateExpression (expr: OakExpression) : string * string =
    match expr with
    | Literal lit ->
        match lit with
        | IntLiteral value ->
            let var = nextVar()
            let instr = sprintf "  %s = arith.constant %d : i32" var value
            (var, instr)
        | StringLiteral value ->
            // For now, just create a placeholder for string literals
            let var = nextVar()
            let instr = sprintf "  %s = arith.constant 0 : i32  // String: \"%s\"" var value
            (var, instr)
        | UnitLiteral ->
            ("", "")
        | _ ->
            let var = nextVar()
            let instr = sprintf "  %s = arith.constant 0 : i32" var
            (var, instr)
    
    | Variable name -> 
        // For now, treat variables as constants
        let var = nextVar()
        let instr = sprintf "  %s = arith.constant 0 : i32  // Variable: %s" var name
        (var, instr)
    
    | Application(func, args) ->
        match func with
        | Variable "printf" ->
            // Generate call to external printf function
            let var = nextVar()
            let instr = sprintf "  %s = func.call @printf() : () -> i32" var
            (var, instr)
        | Variable "printfn" ->
            // Generate call to external printfn function  
            let var = nextVar()
            let instr = sprintf "  %s = func.call @printfn() : () -> i32" var
            (var, instr)
        | Variable "readLine" ->
            // Generate call to external readLine function
            let var = nextVar()
            let instr = sprintf "  %s = func.call @readLine() : () -> i32" var
            (var, instr)
        | Variable funcName ->
            // Generate call to user-defined function
            let var = nextVar()
            let instr = sprintf "  %s = func.call @%s() : () -> i32" var funcName
            (var, instr)
        | _ ->
            let var = nextVar()
            let instr = sprintf "  %s = arith.constant 0 : i32" var
            (var, instr)
    
    | Sequential(first, second) ->
        let (_, firstCode) = generateExpression first
        let (secondVar, secondCode) = generateExpression second
        let combinedCode = 
            if String.IsNullOrEmpty(firstCode) then secondCode
            else sprintf "%s\n%s" firstCode secondCode
        (secondVar, combinedCode)
    
    | _ ->
        let var = nextVar()
        let instr = sprintf "  %s = arith.constant 0 : i32" var
        (var, instr)

/// Generates MLIR for Oak declarations
let private generateDeclaration (decl: OakDeclaration) : string =
    match decl with
    | FunctionDecl(name, params', returnType, body) ->
        resetVarCounter()
        let (bodyVar, bodyCode) = generateExpression body
        let returnStmt = 
            if String.IsNullOrEmpty(bodyVar) then
                "  %ret = arith.constant 0 : i32\n  func.return %ret : i32"
            else
                sprintf "  func.return %s : i32" bodyVar
        
        sprintf "func.func @%s() -> i32 {\n%s\n%s\n}" name bodyCode returnStmt
    
    | EntryPoint(expr) ->
        resetVarCounter()
        let (exprVar, exprCode) = generateExpression expr
        let returnStmt = 
            if String.IsNullOrEmpty(exprVar) then
                "  %ret = arith.constant 0 : i32\n  func.return %ret : i32"
            else
                sprintf "  func.return %s : i32" exprVar
        
        sprintf "func.func @main() -> i32 {\n%s\n%s\n}" exprCode returnStmt
    
    | TypeDecl(name, oakType) ->
        sprintf "// Type declaration: %s = %s" name (oakTypeToMLIRString oakType)

/// Uses Oak AST to generate proper MLIR
let generateMLIR (program: OakProgram) : MLIROutput =
    let firstModule = 
        if program.Modules.IsEmpty then
            { Name = "Main"; Declarations = [] }
        else
            program.Modules.[0]

    let operations = 
        firstModule.Declarations
        |> List.map generateDeclaration

    { ModuleName = firstModule.Name; Operations = operations }