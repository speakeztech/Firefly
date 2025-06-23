module Core.MLIRGeneration.DirectGenerator

open System.Text
open FSharp.Compiler.Syntax
open FSharp.Compiler.Symbols
open Dabbit.Bindings.PatternLibrary
open Core.MLIRGeneration.TypeMapping
open Core.MLIRGeneration.TypeSystem
open Core.MLIRGeneration.Dialect

/// MLIR generation state
type GenState = {
    SSACounter: int
    LocalScope: Map<string, string>  // variable -> SSA value
    Output: StringBuilder
    TypeCtx: TypeContext
    Indent: int
}

/// Generate SSA value name
let nextSSA prefix state =
    let name = sprintf "%%%s%d" prefix state.SSACounter
    (name, { state with SSACounter = state.SSACounter + 1 })

/// Emit MLIR operation
let emit str state =
    let indent = String.replicate state.Indent "  "
    state.Output.AppendLine(indent + str) |> ignore
    state

/// Generate MLIR for expression
let rec genExpr (state: GenState) expr =
    match expr with
    | SynExpr.Const(SynConst.Int32 n, _) ->
        let (ssa, state') = nextSSA "const" state
        let state'' = emit (sprintf "%s = arith.constant %d : i32" ssa n) state'
        (ssa, MLIRTypes.i32, state'')
    
    | SynExpr.Ident ident ->
        match Map.tryFind ident.idText state.LocalScope with
        | Some ssa -> (ssa, MLIRTypes.i32, state)  // TODO: track types
        | None ->
            // Check pattern library
            match findByName ident.idText with
            | Some pattern ->
                let (ssa, state') = nextSSA "func" state
                let funcType = MLIRTypes.func (fst pattern.TypeSig) (snd pattern.TypeSig)
                let typeStr = Core.MLIRGeneration.TypeSystem.mlirTypeToString funcType
                let formatted = sprintf "%s = func.constant @%s : %s" ssa pattern.QualifiedName typeStr
                let state'' = emit formatted state'
                (ssa, snd pattern.TypeSig, state'')
            | None ->
                let (ssa, state') = nextSSA "undef" state
                let state'' = emit (sprintf "%s = arith.constant 0 : i32 // undefined: %s" ssa ident.idText) state'
                (ssa, MLIRTypes.i32, state'')
    
    | SynExpr.App(_, _, func, arg, _) ->
        // Check if this matches a pattern
        match findByExpression expr with
        | Some pattern ->
            genPatternApp state pattern expr
        | None ->
            // Generic application
            let (funcSSA, funcType, state1) = genExpr state func
            let (argSSA, argType, state2) = genExpr state1 arg
            let (resultSSA, state3) = nextSSA "call" state2
            let argTypeStr = Core.MLIRGeneration.TypeSystem.mlirTypeToString argType
            let formatted = sprintf "%s = func.call_indirect %s(%s) : (%s) -> i32" resultSSA funcSSA argSSA argTypeStr
            let state4 = emit formatted state3
            (resultSSA, MLIRTypes.i32, state4)
    
    | SynExpr.LetOrUse(_, _, bindings, body, _, _) ->
        let state' = bindings |> List.fold genBinding state
        genExpr state' body
    
    | SynExpr.IfThenElse(cond, thenExpr, Some elseExpr, _, _, _, _) ->
        let (condSSA, _, state1) = genExpr state cond
        let (resultSSA, state2) = nextSSA "if.result" state1
        
        // Generate blocks
        let state3 = emit (sprintf "cf.cond_br %s, ^then, ^else" condSSA) state2
        let state4 = emit "^then:" state3
        let (thenSSA, thenType, state5) = genExpr { state4 with Indent = state4.Indent + 1 } thenExpr
        let thenTypeStr = Core.MLIRGeneration.TypeSystem.mlirTypeToString thenType
        let state6 = emit (sprintf "cf.br ^merge(%s : %s)" thenSSA thenTypeStr) state5
        
        let state7 = emit "^else:" { state6 with Indent = state.Indent }
        let (elseSSA, elseType, state8) = genExpr { state7 with Indent = state7.Indent + 1 } elseExpr
        let elseTypeStr = Core.MLIRGeneration.TypeSystem.mlirTypeToString elseType
        let state9 = emit (sprintf "cf.br ^merge(%s : %s)" elseSSA elseTypeStr) state8
        
        let mergeTypeStr = Core.MLIRGeneration.TypeSystem.mlirTypeToString thenType
        let state10 = emit (sprintf "^merge(%s : %s):" resultSSA mergeTypeStr) { state9 with Indent = state.Indent }
        (resultSSA, thenType, state10)
    
    | _ ->
        let (ssa, state') = nextSSA "todo" state
        let state'' = emit (sprintf "%s = arith.constant 0 : i32 // TODO: %A" ssa expr) state'
        (ssa, MLIRTypes.i32, state'')

/// Generate pattern-based operation
and genPatternApp state pattern expr =
    match pattern.OpPattern with
    | DialectOp(dialect, op, attrs) ->
        let (resultSSA, state') = nextSSA "op" state
        let dialectStr = dialectToString dialect
        let attrStr = 
            if Map.isEmpty attrs then ""
            else attrs |> Map.toList |> List.map (fun (k, v) -> sprintf "%s = %s" k v) |> String.concat ", " |> sprintf " {%s}"
        let returnType = snd pattern.TypeSig
        let returnTypeStr = Core.MLIRGeneration.TypeSystem.mlirTypeToString returnType
        let formatted = sprintf "%s = %s.%s()%s : %s" resultSSA dialectStr op attrStr returnTypeStr
        let state'' = emit formatted state'
        (resultSSA, returnType, state'')
    
    | ExternalCall(func, _) ->
        let (resultSSA, state') = nextSSA "call" state
        let returnType = snd pattern.TypeSig
        let returnTypeStr = Core.MLIRGeneration.TypeSystem.mlirTypeToString returnType
        let formatted = sprintf "%s = func.call @%s() : () -> %s" resultSSA func returnTypeStr
        let state'' = emit formatted state'
        (resultSSA, returnType, state'')
    
    | _ ->
        let (ssa, state') = nextSSA "pattern" state
        let state'' = emit (sprintf "%s = arith.constant 0 : i32 // pattern: %s" ssa pattern.Name) state'
        (ssa, MLIRTypes.i32, state'')

/// Generate binding
and genBinding state binding =
    let (SynBinding(_, _, _, _, _, _, _, pat, _, expr, _, _, _)) = binding
    match pat with
    | SynPat.Named(SynIdent(ident, _), _, _, _) ->
        let (ssa, _, state') = genExpr state expr
        { state' with LocalScope = Map.add ident.idText ssa state'.LocalScope }
    | _ -> state

/// Generate MLIR module
and generateModule (moduleName: string) (typeCtx: TypeContext) (input: ParsedInput) =
    let state = {
        SSACounter = 0
        LocalScope = Map.empty
        Output = StringBuilder()
        TypeCtx = typeCtx
        Indent = 0
    }
    
    let state' = emit (sprintf "module @%s {" moduleName) state
    
    let finalState = 
        match input with
        | ParsedInput.ImplFile(ParsedImplFileInput(fileName, isScript, qualifiedNameOfFile, scopedPragmas, hashDirectives, modules, isLastCompiland, trivia, identifiers)) ->
            modules |> List.fold (fun s (SynModuleOrNamespace(_, _, _, decls, _, _, _, _, _)) ->
                decls |> List.fold genModuleDecl s) { state' with Indent = 1 }
        | _ -> state'
    
    let state'' = emit "}" { finalState with Indent = 0 }
    state''.Output.ToString()

/// Generate module declaration
and genModuleDecl state = function
    | SynModuleDecl.Let(_, bindings, _) ->
        bindings |> List.fold genTopLevelBinding state
    | _ -> state

/// Generate top-level binding as function
and genTopLevelBinding state binding =
    let (SynBinding(_, _, _, _, _, _, _, pat, _, expr, _, _, _)) = binding
    match pat with
    | SynPat.Named(SynIdent(ident, _), _, _, _) ->
        let state' = emit (sprintf "func.func @%s() -> i32 {" ident.idText) state
        let (resultSSA, _, state'') = genExpr { state' with Indent = state'.Indent + 1 } expr
        let state''' = emit (sprintf "func.return %s : i32" resultSSA) state''
        emit "}" { state''' with Indent = state.Indent }
    | _ -> state