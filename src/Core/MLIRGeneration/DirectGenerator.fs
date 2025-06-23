module Core.MLIRGeneration.DirectGenerator

open System.Text
open FSharp.Compiler.Syntax
open FSharp.Compiler.Symbols
open Core.MLIRGeneration.TypeMapping
open Core.MLIRGeneration.TypeSystem
open Core.MLIRGeneration.Dialect
open Core.XParsec.Foundation
open Dabbit.Bindings.SymbolRegistry

/// MLIR generation state with full context
type GenState = {
    SSACounter: int
    LocalVariables: Map<string, string>  // variable name -> SSA value
    Output: StringBuilder
    TypeContext: TypeContext
    SymbolRegistry: SymbolRegistry
    IndentLevel: int
    CurrentModulePath: string
    RequiredExternals: Set<string>  // Track external functions needed
    CurrentFunction: string option   // Track current function for error reporting
}

/// Generate unique SSA value name
let generateSSAName prefix state =
    let ssaName = sprintf "%%%s%d" prefix state.SSACounter
    let updatedState = { state with SSACounter = state.SSACounter + 1 }
    (ssaName, updatedState)

/// Emit MLIR operation with proper indentation
let emit mlirCode state =
    let indentation = String.replicate state.IndentLevel "  "
    state.Output.AppendLine(indentation + mlirCode) |> ignore
    state

/// Emit without newline
let emitInline mlirCode state =
    let indentation = String.replicate state.IndentLevel "  "
    state.Output.Append(indentation + mlirCode) |> ignore
    state

/// Forward declarations for mutual recursion
let rec generateExpression state expr =
    match expr with
    | SynExpr.Const(constant, range) ->
        generateConstant state constant
    
    | SynExpr.Ident ident ->
        generateIdentifier state ident
    
    | SynExpr.App(_, _, funcExpr, argExpr, _) ->
        generateApplication state funcExpr argExpr
    
    | SynExpr.TypeApp(baseExpr, _, typeArgs, _, _, _, _) ->
        // For now, ignore type arguments and process base expression
        generateExpression state baseExpr
    
    | SynExpr.LetOrUse(isRec, isUse, bindings, body, range, trivia) ->
        generateLet state bindings body
    
    | SynExpr.Sequential(_, _, expr1, expr2, _, _) ->
        let (_, _, state1) = generateExpression state expr1
        generateExpression state1 expr2
    
    | SynExpr.Match(_, matchExpr, clauses, _, _) ->
        generateMatch state matchExpr clauses
    
    | SynExpr.Paren(inner, _, _, _) ->
        generateExpression state inner
    
    | SynExpr.LongIdent(_, SynLongIdent(ids, _, _), _, _) ->
        let name = ids |> List.map (fun id -> id.idText) |> String.concat "."
        generateQualifiedIdentifier state name
    
    | SynExpr.DotGet(targetExpr, _, SynLongIdent(ids, _, _), _) ->
        generateFieldAccess state targetExpr ids
    
    | SynExpr.Tuple(_, exprs, _, _) ->
        generateTuple state exprs
    
    | _ ->
        let (ssa, s) = generateSSAName "unsupported" state
        let s1 = emit (sprintf "%s = arith.constant 0 : i32  // TODO: %A" ssa (expr.GetType().Name)) s
        (ssa, MLIRTypes.i32, s1)

/// Generate constants
and generateConstant state = function
    | SynConst.Int32 n ->
        let (ssa, s) = generateSSAName "c" state
        let s1 = emit (sprintf "%s = arith.constant %d : i32" ssa n) s
        (ssa, MLIRTypes.i32, s1)
    
    | SynConst.String(text, _, _) ->
        let (ssa, s) = generateSSAName "str" state
        // TODO: Proper string constant generation
        let s1 = emit (sprintf "%s = llvm.mlir.undef : !llvm.ptr<i8>  // \"%s\"" ssa (text.Replace("\"", "\\\""))) s
        (ssa, MLIRTypes.memref MLIRTypes.i8, s1)
    
    | SynConst.Unit ->
        let (ssa, s) = generateSSAName "unit" state
        let s1 = emit (sprintf "%s = llvm.mlir.undef : i32  // unit" ssa) s
        (ssa, MLIRTypes.void_, s1)
    
    | _ ->
        let (ssa, s) = generateSSAName "const" state
        let s1 = emit (sprintf "%s = arith.constant 0 : i32  // unsupported constant" ssa) s
        (ssa, MLIRTypes.i32, s1)

/// Generate identifier reference
and generateIdentifier state ident =
    match Map.tryFind ident.idText state.LocalVariables with
    | Some ssa -> (ssa, MLIRTypes.i32, state)  // TODO: Track types properly
    | None ->
        // Might be a function reference
        let (ssa, s) = generateSSAName "undef" state
        let s1 = emit (sprintf "%s = llvm.mlir.undef : i32  // undefined: %s" ssa ident.idText) s
        (ssa, MLIRTypes.i32, s1)

/// Generate qualified identifier
and generateQualifiedIdentifier state name =
    // For now, treat as function reference
    let (ssa, s) = generateSSAName "qid" state
    let s1 = emit (sprintf "%s = llvm.mlir.undef : i32  // %s" ssa name) s
    (ssa, MLIRTypes.i32, s1)

/// Generate application
and generateApplication state funcExpr argExpr =
    match funcExpr with
    | SynExpr.Ident ident when ident.idText = "op_PipeRight" || ident.idText = "|>" ->
        // Pipe operator - just evaluate the argument
        generateExpression state argExpr
    
    | SynExpr.Ident ident ->
        generateKnownCall state ident.idText argExpr
    
    | SynExpr.LongIdent(_, SynLongIdent(ids, _, _), _, _) ->
        let funcName = ids |> List.map (fun id -> id.idText) |> String.concat "."
        generateKnownCall state funcName argExpr
    
    | SynExpr.TypeApp(SynExpr.Ident ident, _, _, _, _, _, _) ->
        generateKnownCall state ident.idText argExpr
    
    | _ ->
        // General application
        let (funcSSA, _, s1) = generateExpression state funcExpr
        let (argSSA, argType, s2) = generateExpression s1 argExpr
        let (resultSSA, s3) = generateSSAName "app" s2
        let s4 = emit (sprintf "%s = func.call_indirect %s(%s) : (%s) -> i32" 
                        resultSSA funcSSA argSSA (mlirTypeToString argType)) s3
        (resultSSA, MLIRTypes.i32, s4)

/// Generate known function calls
and generateKnownCall state funcName argExpr =
    match funcName with
    | "stackBuffer" ->
        let (sizeSSA, _, s1) = generateExpression state argExpr
        let (bufferSSA, s2) = generateSSAName "buf" s1
        let s3 = emit (sprintf "%s = memref.alloca() : memref<256xi8>" bufferSSA) s2
        (bufferSSA, MLIRTypes.memref MLIRTypes.i8, s3)
    
    | "prompt" | "writeLine" ->
        let (argSSA, _, s1) = generateExpression state argExpr
        let s2 = { s1 with RequiredExternals = Set.add "printf" s1.RequiredExternals }
        let (resultSSA, s3) = generateSSAName "io" s2
        let s4 = emit (sprintf "%s = func.call @printf(%s) : (!llvm.ptr<i8>) -> i32" resultSSA argSSA) s3
        (resultSSA, MLIRTypes.i32, s4)
    
    | "readInto" ->
        let (bufferSSA, _, s1) = generateExpression state argExpr
        let s2 = { s1 with RequiredExternals = Set.add "fgets" s1.RequiredExternals }
        let (resultSSA, s3) = generateSSAName "read" s2
        // Simplified - would need stdin handle
        let s4 = emit (sprintf "%s = llvm.mlir.undef : i32  // readInto" resultSSA) s3
        (resultSSA, MLIRTypes.i32, s4)
    
    | "String.format" | "format" ->
        match argExpr with
        | SynExpr.Const(SynConst.String(fmt, _, _), _) ->
            let (fmtSSA, s1) = generateSSAName "fmt" state
            let s2 = emit (sprintf "%s = llvm.mlir.undef : !llvm.ptr<i8>  // \"%s\"" fmtSSA fmt) s1
            (fmtSSA, MLIRTypes.memref MLIRTypes.i8, s2)
        | _ ->
            generateExpression state argExpr
    
    | "spanToString" | "AsSpan" ->
        generateExpression state argExpr
    
    | _ ->
        // Unknown function
        let (argSSA, argType, s1) = generateExpression state argExpr
        let (resultSSA, s2) = generateSSAName "call" s1
        let s3 = emit (sprintf "%s = llvm.mlir.undef : i32  // call %s" resultSSA funcName) s2
        (resultSSA, MLIRTypes.i32, s3)

/// Generate let binding
and generateLet state bindings body =
    let s1 = bindings |> List.fold processLetBinding state
    generateExpression s1 body

and processLetBinding state (SynBinding(_, _, _, _, _, _, _, pat, _, expr, _, _, _)) =
    match pat with
    | SynPat.Named(SynIdent(ident, _), _, _, _) ->
        let (ssa, _, s1) = generateExpression state expr
        { s1 with LocalVariables = Map.add ident.idText ssa s1.LocalVariables }
    | _ -> state

/// Generate match expression
and generateMatch state matchExpr clauses =
    let (matchSSA, _, s1) = generateExpression state matchExpr
    
    // Simplified match - just take first clause
    match clauses with
    | SynMatchClause(pat, _, resultExpr, _, _, _) :: _ ->
        // Bind pattern variables if needed
        let s2 = 
            match pat with
            | SynPat.Named(SynIdent(ident, _), _, _, _) ->
                { s1 with LocalVariables = Map.add ident.idText matchSSA s1.LocalVariables }
            | _ -> s1
        generateExpression s2 resultExpr
    | [] ->
        let (ssa, s2) = generateSSAName "match" s1
        let s3 = emit (sprintf "%s = arith.constant 0 : i32  // empty match" ssa) s2
        (ssa, MLIRTypes.i32, s3)

/// Generate field access
and generateFieldAccess state targetExpr fieldIds =
    let (targetSSA, _, s1) = generateExpression state targetExpr
    let fieldName = fieldIds |> List.map (fun id -> id.idText) |> String.concat "."
    let (ssa, s2) = generateSSAName "field" s1
    let s3 = emit (sprintf "%s = llvm.mlir.undef : i32  // field %s of %s" ssa fieldName targetSSA) s2
    (ssa, MLIRTypes.i32, s3)

/// Generate tuple
and generateTuple state exprs =
    let (ssas, s) = 
        exprs |> List.fold (fun (ssaList, st) expr ->
            let (ssa, _, st1) = generateExpression st expr
            (ssa :: ssaList, st1)
        ) ([], state)
    
    match List.rev ssas with
    | [ssa] -> (ssa, MLIRTypes.i32, s)
    | _ ->
        let (tupleSSA, s1) = generateSSAName "tuple" s
        let s2 = emit (sprintf "%s = llvm.mlir.undef : i32  // tuple" tupleSSA) s1
        (tupleSSA, MLIRTypes.i32, s2)

/// Generate MLIR function
let rec generateFunction state functionName (attributes: SynAttributes) expression =
    let qualifiedName = 
        if state.CurrentModulePath = "" then functionName 
        else state.CurrentModulePath + "." + functionName
    
    // Check if entry point
    let isMain = attributes |> List.exists (fun attrList ->
        attrList.Attributes |> List.exists (fun attr ->
            match attr.TypeName with
            | SynLongIdent([ident], [], [None]) -> ident.idText = "EntryPoint"
            | _ -> false))
    
    let mlirName = if isMain then "@main" else "@" + functionName
    
    // Function header
    let stateWithFunc = 
        state
        |> emit (sprintf "func.func %s() -> i32 {" mlirName)
        |> fun s -> { s with 
                        IndentLevel = state.IndentLevel + 1
                        CurrentFunction = Some functionName
                        LocalVariables = Map.empty
                        SSACounter = 0 }
    
    // Generate body
    let (resultSSA, resultType, stateAfterBody) = generateExpression stateWithFunc expression
    
    // Return statement
    let stateWithReturn = 
        match resultType with
        | t when t = MLIRTypes.void_ || isMain ->
            let (zeroSSA, s1) = generateSSAName "ret" stateAfterBody
            s1
            |> emit (sprintf "%s = arith.constant 0 : i32" zeroSSA)
            |> emit (sprintf "func.return %s : i32" zeroSSA)
        | _ ->
            stateAfterBody
            |> emit (sprintf "func.return %s : %s" resultSSA (mlirTypeToString resultType))
    
    // Close function
    { stateWithReturn with 
        IndentLevel = state.IndentLevel
        CurrentFunction = None }
    |> emit "}"

/// Process function binding
let processBinding state (SynBinding(access, kind, isInline, isMutable, attributes, xmlDoc, valData, pattern, returnInfo, expression, range, sp, trivia)) =
    match pattern with
    | SynPat.Named(SynIdent(ident, _), _, _, _) ->
        generateFunction state ident.idText attributes expression
    | SynPat.LongIdent(SynLongIdent(ids, _, _), _, _, _, _, _) ->
        let name = ids |> List.map (fun id -> id.idText) |> List.last
        generateFunction state name attributes expression
    | _ -> state

/// Process declarations
let rec processDeclaration state = function
    | SynModuleDecl.Let(isRec, bindings, range) ->
        bindings |> List.fold processBinding state
    | SynModuleDecl.NestedModule(componentInfo, isRec, nestedDecls, range, trivia, moduleKeyword) ->
        let (SynComponentInfo(_, _, _, longId, _, _, _, _)) = componentInfo
        let nestedName = longId |> List.map (fun ident -> ident.idText) |> String.concat "."
        let fullPath = 
            if state.CurrentModulePath = "" then nestedName 
            else state.CurrentModulePath + "." + nestedName
        let stateWithPath = { state with CurrentModulePath = fullPath }
        nestedDecls |> List.fold processDeclaration stateWithPath
    | _ -> state

/// Process module or namespace
let processModuleOrNamespace state (SynModuleOrNamespace(longId, isRec, kind, declarations, xmlDoc, attrs, access, range, trivia)) =
    let modulePath = longId |> List.map (fun ident -> ident.idText) |> String.concat "."
    let stateWithPath = { state with CurrentModulePath = modulePath }
    declarations |> List.fold processDeclaration stateWithPath

/// Process a parsed input file
let processInputFile state filePath parsedInput =
    match parsedInput with
    | ParsedInput.ImplFile(ParsedImplFileInput(fileName, isScript, qualName, pragmas, directives, modules, isLast, trivia, ids)) ->
        modules |> List.fold processModuleOrNamespace state
    | ParsedInput.SigFile(_) -> state

/// Generate MLIR for a complete program
let generateProgram (programName: string) (typeCtx: TypeContext) (symbolRegistry: SymbolRegistry) 
                   (reachableInputs: (string * ParsedInput) list) : string =
    
    let initialState = {
        SSACounter = 0
        LocalVariables = Map.empty
        Output = StringBuilder()
        TypeContext = typeCtx
        SymbolRegistry = symbolRegistry
        IndentLevel = 0
        CurrentModulePath = ""
        RequiredExternals = Set.empty
        CurrentFunction = None
    }
    
    // Start MLIR module
    let state = 
        initialState
        |> emit (sprintf "module @%s {" programName)
        |> fun s -> { s with IndentLevel = 1 }
    
    // Process all reachable inputs
    let stateAfterInputs = 
        reachableInputs 
        |> List.fold (fun currentState (filePath, parsedInput) ->
            processInputFile currentState filePath parsedInput
        ) state
    
    // Emit external function declarations
    let finalState = 
        stateAfterInputs.RequiredExternals
        |> Set.fold (fun s ext ->
            match ext with
            | "printf" -> emit "func.func private @printf(!llvm.ptr<i8>, ...) -> i32" s
            | "sprintf" -> emit "func.func private @sprintf(!llvm.ptr<i8>, !llvm.ptr<i8>, ...) -> !llvm.ptr<i8>" s
            | "fgets" -> emit "func.func private @fgets(!llvm.ptr<i8>, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>" s
            | "strlen" -> emit "func.func private @strlen(!llvm.ptr<i8>) -> i32" s
            | _ -> s
        ) stateAfterInputs
        |> fun s -> { s with IndentLevel = 0 }
        |> emit "}"
    
    finalState.Output.ToString()