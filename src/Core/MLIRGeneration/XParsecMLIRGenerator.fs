module Core.MLIRGeneration.XParsecMLIRGenerator

open System
open Dabbit.Parsing.OakAst
open Dabbit.Parsing.Translator
open Core.MLIRGeneration.TypeSystem
open Core.MLIRGeneration.Operations
open XParsec

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
        | FloatLiteral value ->
            let var = nextVar()
            let instr = sprintf "  %s = arith.constant %f : f32" var value
            (var, instr)
        | BoolLiteral value ->
            let var = nextVar()
            let boolVal = if value then "1" else "0"
            let instr = sprintf "  %s = arith.constant %s : i1" var boolVal
            (var, instr)
        | StringLiteral value ->
            let var = nextVar()
            let instr = sprintf "  %s = llvm.mlir.addressof @str_%d : !llvm.ptr<i8>" var (abs(value.GetHashCode()))
            (var, instr)
        | UnitLiteral ->
            ("", "")
        | ArrayLiteral elements ->
            let var = nextVar()
            let size = elements.Length
            let instr = sprintf "  %s = memref.alloc() : memref<%dx i32>" var size
            (var, instr)
    
    | Variable name -> 
        (name, "")
    
    | Application(func, args) ->
        let (funcVar, funcCode) = generateExpression func
        let argResults = args |> List.map generateExpression
        let argVars = argResults |> List.map fst
        let argCodes = argResults |> List.map snd |> List.filter (not << String.IsNullOrEmpty)
        let resultVar = nextVar()
        let argStr = String.concat ", " argVars
        let callInstr = sprintf "  %s = func.call %s(%s) : () -> i32" resultVar funcVar argStr
        let allCode = String.concat "\n" (funcCode :: argCodes @ [callInstr])
        (resultVar, allCode)
    
    | Lambda(params', body) ->
        let paramStrs = params' |> List.map (fun (name, typ) -> sprintf "%s: %s" name (oakTypeToMLIRString typ))
        let (bodyVar, bodyCode) = generateExpression body
        let funcName = sprintf "@lambda_%d" (abs(params'.GetHashCode()))
        let lambdaCode = sprintf "func.func %s(%s) -> i32 {\n%s\n  func.return %s : i32\n}" funcName (String.concat ", " paramStrs) bodyCode bodyVar
        (funcName, lambdaCode)
    
    | Let(name, value, body) ->
        let (valueVar, valueCode) = generateExpression value
        let (bodyVar, bodyCode) = generateExpression body
        let letCode = if String.IsNullOrEmpty(valueCode) then bodyCode else sprintf "%s\n%s" valueCode bodyCode
        (bodyVar, letCode)
    
    | IfThenElse(cond, thenExpr, elseExpr) ->
        let (condVar, condCode) = generateExpression cond
        let (thenVar, thenCode) = generateExpression thenExpr
        let (elseVar, elseCode) = generateExpression elseExpr
        let resultVar = nextVar()
        let ifCode = sprintf "%s\n  cf.cond_br %s, ^then, ^else\n^then:\n%s\n  cf.br ^merge(%s : i32)\n^else:\n%s\n  cf.br ^merge(%s : i32)\n^merge(%s: i32):" condCode condVar thenCode thenVar elseCode elseVar resultVar
        (resultVar, ifCode)
    
    | Sequential(first, second) ->
        let (_, firstCode) = generateExpression first
        let (secondVar, secondCode) = generateExpression second
        let seqCode = if String.IsNullOrEmpty(firstCode) then secondCode else sprintf "%s\n%s" firstCode secondCode
        (secondVar, seqCode)
    
    | FieldAccess(target, fieldName) ->
        let (targetVar, targetCode) = generateExpression target
        let resultVar = nextVar()
        let accessCode = sprintf "  %s = llvm.extractvalue %s[0] : !llvm.struct<(i32)>" resultVar targetVar
        let fullCode = sprintf "%s\n%s" targetCode accessCode
        (resultVar, fullCode)
    
    | MethodCall(target, methodName, args) ->
        generateExpression (Application(target, args))

/// Generates MLIR for Oak declarations
let private generateDeclaration (decl: OakDeclaration) : string =
    match decl with
    | FunctionDecl(name, params', returnType, body) ->
        resetVarCounter()
        let paramStrs = params' |> List.map (fun (pName, pType) -> sprintf "%%%s: %s" pName (oakTypeToMLIRString pType))
        let (bodyVar, bodyCode) = generateExpression body
        let returnTypeStr = oakTypeToMLIRString returnType
        sprintf "func.func @%s(%s) -> %s {\n%s\n  func.return %s : %s\n}" name (String.concat ", " paramStrs) returnTypeStr bodyCode bodyVar returnTypeStr
    
    | EntryPoint(expr) ->
        resetVarCounter()
        let (exprVar, exprCode) = generateExpression expr
        let returnVar = if String.IsNullOrEmpty(exprVar) then "  %c0_i32 = arith.constant 0 : i32\n  %c0_i32" else exprVar
        sprintf "func.func @main() -> i32 {\n%s\n  func.return %s : i32\n}" exprCode returnVar
    
    | TypeDecl(name, oakType) ->
        sprintf "// Type declaration: %s = %s" name (oakTypeToMLIRString oakType)

/// Uses XParsec to generate MLIR from Oak AST
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