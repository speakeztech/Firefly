module Core.MLIRGeneration.DirectGenerator

open System.Text
open FSharp.Compiler.Syntax
open FSharp.Compiler.Symbols
open Core.MLIRGeneration.TypeMapping
open Core.MLIRGeneration.TypeSystem
open Core.MLIRGeneration.Dialect
open Core.XParsec.Foundation
open Dabbit.Bindings.SymbolRegistry

/// Convert MLIRType to MLIR string with proper LLVM dialect syntax
let mlirTypeToStringWithLLVM (t: MLIRType) : string =
    match t.Category with
    | MLIRTypeCategory.MemRef ->
        match t.ElementType with
        | Some elem when elem = MLIRTypes.i8 -> "!llvm.ptr" 
        | _ -> mlirTypeToString t
    | _ -> mlirTypeToString t

/// MLIR builder type - transforms a builder state
type MLIRBuilder<'T> = MLIRBuilderState -> ('T * MLIRBuilderState)

/// Builder state with output and context
and MLIRBuilderState = {
    Output: StringBuilder
    Indent: int
    SSACounter: int
    LocalVars: Map<string, string>
    TypeContext: TypeContext
    SymbolRegistry: SymbolRegistry
    RequiredExternals: Set<string>
    CurrentFunction: string option
}

/// MLIR builder computation expression with all required methods
type MLIRBuilderCE() =
    member _.Return(x) : MLIRBuilder<'T> = 
        fun state -> (x, state)
    
    member _.ReturnFrom(m: MLIRBuilder<'T>) : MLIRBuilder<'T> = m
    
    member _.Bind(m: MLIRBuilder<'T>, f: 'T -> MLIRBuilder<'U>) : MLIRBuilder<'U> =
        fun state ->
            let (x, state') = m state
            (f x) state'
    
    member _.Zero() : MLIRBuilder<unit> = 
        fun state -> ((), state)
        
    member _.Combine(m1: MLIRBuilder<unit>, m2: MLIRBuilder<'T>) : MLIRBuilder<'T> =
        fun state ->
            let (_, state') = m1 state
            m2 state'
            
    member _.Delay(f: unit -> MLIRBuilder<'T>) : MLIRBuilder<'T> =
        fun state -> (f()) state
        
    member _.For(sequence: seq<'T>, body: 'T -> MLIRBuilder<unit>) : MLIRBuilder<unit> =
        fun state ->
            let mutable currentState = state
            for item in sequence do
                let ((), newState) = (body item) currentState
                currentState <- newState
            ((), currentState)

let mlir = MLIRBuilderCE()

/// Get current state
let getState : MLIRBuilder<MLIRBuilderState> =
    fun state -> (state, state)

/// Update state
let updateState (f: MLIRBuilderState -> MLIRBuilderState) : MLIRBuilder<unit> =
    fun state -> ((), f state)

/// Generate unique SSA name
let nextSSA (prefix: string) : MLIRBuilder<string> =
    fun state ->
        let name = sprintf "%%%s%d" prefix state.SSACounter
        let state' = { state with SSACounter = state.SSACounter + 1 }
        (name, state')

/// Emit raw text
let emit (text: string) : MLIRBuilder<unit> =
    fun state ->
        state.Output.Append(text) |> ignore
        ((), state)

/// Emit with current indentation
let emitIndented (text: string) : MLIRBuilder<unit> =
    fun state ->
        let indent = String.replicate state.Indent "  "
        state.Output.Append(indent).Append(text) |> ignore
        ((), state)

/// Emit line with indentation
let emitLine (text: string) : MLIRBuilder<unit> =
    mlir {
        do! emitIndented text
        do! emit "\n"
    }

/// Emit newline only
let newline : MLIRBuilder<unit> = emit "\n"

/// Increase indentation for nested scope
let indent (builder: MLIRBuilder<'T>) : MLIRBuilder<'T> =
    mlir {
        do! updateState (fun s -> { s with Indent = s.Indent + 1 })
        let! result = builder
        do! updateState (fun s -> { s with Indent = s.Indent - 1 })
        return result
    }

/// Add external dependency
let requireExternal (name: string) : MLIRBuilder<unit> =
    updateState (fun s -> { s with RequiredExternals = Set.add name s.RequiredExternals })

/// Bind local variable
let bindLocal (name: string) (ssa: string) : MLIRBuilder<unit> =
    updateState (fun s -> { s with LocalVars = Map.add name ssa s.LocalVars })

/// Lookup local variable
let lookupLocal (name: string) : MLIRBuilder<string option> =
    mlir {
        let! state = getState
        return Map.tryFind name state.LocalVars
    }

/// Emit MLIR type
let emitType (t: MLIRType) : MLIRBuilder<unit> =
    emit (mlirTypeToStringWithLLVM t)

/// Emit SSA value
let emitSSA (name: string) : MLIRBuilder<unit> =
    emit name

/// Emit comma-separated list
let emitList (sep: string) (emitItem: 'T -> MLIRBuilder<unit>) (items: 'T list) : MLIRBuilder<unit> =
    match items with
    | [] -> mlir.Zero()
    | [x] -> emitItem x
    | x::xs ->
        mlir {
            do! emitItem x
            for item in xs do
                do! emit sep
                do! emitItem item
        }

/// Emit operation result assignment
let emitResultAssign (ssa: string) : MLIRBuilder<unit> =
    mlir {
        do! emitSSA ssa
        do! emit " = "
    }

/// Emit constant operation
let emitConstant (ssa: string) (value: int) (typ: MLIRType) : MLIRBuilder<unit> =
    mlir {
        do! emitResultAssign ssa
        do! emit "arith.constant "
        do! emit (string value)
        do! emit " : "
        do! emitType typ
    }

/// Emit undefined value
let emitUndef (ssa: string) (typ: MLIRType) (comment: string option) : MLIRBuilder<unit> =
    mlir {
        do! emitResultAssign ssa
        do! emit "llvm.mlir.undef : "
        do! emitType typ
        match comment with
        | Some c -> do! emit (sprintf "  // %s" c)
        | None -> ()
    }

/// Emit memory allocation
let emitAlloca (ssa: string) (size: int) (elemType: MLIRType) : MLIRBuilder<unit> =
    mlir {
        do! emitResultAssign ssa
        do! emit (sprintf "memref.alloca() : memref<%dx" size)
        do! emitType elemType
        do! emit ">"
    }

/// Emit function call
let emitCall (ssa: string) (func: string) (args: string list) (argTypes: MLIRType list) (retType: MLIRType) : MLIRBuilder<unit> =
    mlir {
        do! emitResultAssign ssa
        do! emit "func.call @"
        do! emit func
        do! emit "("
        do! emitList ", " emitSSA args
        do! emit ") : ("
        // Convert memref types to opaque pointers for LLVM calls
        let llvmArgTypes = argTypes |> List.map (fun t ->
            match t.Category with
            | MLIRTypeCategory.MemRef -> { t with Category = MLIRTypeCategory.MemRef } // Will become !llvm.ptr
            | _ -> t
        )
        do! emitList ", " emitType llvmArgTypes
        do! emit ") -> "
        do! emitType retType
    }

/// Emit indirect call
let emitIndirectCall (ssa: string) (func: string) (arg: string) (argType: MLIRType) (retType: MLIRType) : MLIRBuilder<unit> =
    mlir {
        do! emitResultAssign ssa
        do! emit "func.call_indirect "
        do! emitSSA func
        do! emit "("
        do! emitSSA arg
        do! emit ") : ("
        // Ensure we use opaque pointers for LLVM
        let llvmArgType = 
            match argType.Category with
            | MLIRTypeCategory.MemRef -> argType  // Will render as !llvm.ptr
            | _ -> argType
        do! emitType llvmArgType
        do! emit ") -> "
        do! emitType retType
    }

/// Emit return statement
let emitReturn (value: string) (typ: MLIRType) : MLIRBuilder<unit> =
    mlir {
        do! emit "func.return "
        do! emitSSA value
        do! emit " : "
        do! emitType typ
    }

/// Emit function declaration
let emitFuncDecl (name: string) (parameters: (string * MLIRType) list) (retType: MLIRType) (varargs: bool) : MLIRBuilder<unit> =
    mlir {
        do! emit "func.func private @"
        do! emit name
        do! emit "("
        do! emitList ", " emitType (parameters |> List.map snd)
        if varargs then do! emit ", ..."
        do! emit ") -> "
        do! emitType retType
    }

/// Forward declaration
let rec generateExpression (expr: SynExpr) : MLIRBuilder<string * MLIRType> =
    mlir {
        match expr with
        | SynExpr.Const(constant, _) ->
            return! generateConstant constant
            
        | SynExpr.Ident ident ->
            return! generateIdentifier ident
            
        | SynExpr.App(_, _, funcExpr, argExpr, _) ->
            return! generateApplication funcExpr argExpr
            
        | SynExpr.TypeApp(baseExpr, _, _, _, _, _, _) ->
            return! generateExpression baseExpr
            
        | SynExpr.LetOrUse(_, _, bindings, body, _, _) ->
            return! generateLet bindings body
            
        | SynExpr.Sequential(_, _, expr1, expr2, _, _) ->
            let! _ = generateExpression expr1
            return! generateExpression expr2
            
        | SynExpr.Match(_, matchExpr, clauses, _, _) ->
            return! generateMatch matchExpr clauses
            
        | SynExpr.Paren(inner, _, _, _) ->
            return! generateExpression inner
            
        | SynExpr.LongIdent(_, SynLongIdent(ids, _, _), _, _) ->
            let name = ids |> List.map (fun id -> id.idText) |> String.concat "."
            return! generateQualifiedIdentifier name
            
        | SynExpr.DotGet(targetExpr, _, SynLongIdent(ids, _, _), _) ->
            return! generateFieldAccess targetExpr ids
            
        | SynExpr.Tuple(_, exprs, _, _) ->
            return! generateTuple exprs
            
        | _ ->
            let! ssa = nextSSA "unsupported"
            do! emitLine (sprintf "%s = arith.constant 0 : i32  // TODO: %A" ssa (expr.GetType().Name))
            return (ssa, MLIRTypes.i32)
    }

and generateConstant (constant: SynConst) : MLIRBuilder<string * MLIRType> =
    mlir {
        match constant with
        | SynConst.Int32 n ->
            let! ssa = nextSSA "c"
            do! emitLine (sprintf "%s = arith.constant %d : i32" ssa n)
            return (ssa, MLIRTypes.i32)
            
        | SynConst.String(text, _, _) ->
            let! ssa = nextSSA "str"
            let escaped = text.Replace("\"", "\\\"")
            do! emitUndef ssa (MLIRTypes.memref MLIRTypes.i8) (Some (sprintf "\"%s\"" escaped))
            do! newline
            return (ssa, MLIRTypes.memref MLIRTypes.i8)
            
        | SynConst.Unit ->
            let! ssa = nextSSA "unit"
            do! emitUndef ssa MLIRTypes.i32 (Some "unit")
            do! newline
            return (ssa, MLIRTypes.void_)
            
        | _ ->
            let! ssa = nextSSA "const"
            do! emitLine (sprintf "%s = arith.constant 0 : i32  // unsupported constant" ssa)
            return (ssa, MLIRTypes.i32)
    }

and generateIdentifier (ident: Ident) : MLIRBuilder<string * MLIRType> =
    mlir {
        let! maybeLocal = lookupLocal ident.idText
        match maybeLocal with
        | Some ssa -> return (ssa, MLIRTypes.i32)
        | None ->
            let! ssa = nextSSA "undef"
            // Check if this is a function that will be called
            let typ = 
                match ident.idText with
                | "op_PipeRight" | "op_PipeLeft" | "op_ComposeRight" | "op_ComposeLeft" ->
                    // These are function operators - need function pointer type
                    MLIRTypes.memref MLIRTypes.i8  // Will render as !llvm.ptr
                | _ -> MLIRTypes.i32
            do! emitUndef ssa typ (Some (sprintf "undefined: %s" ident.idText))
            do! newline
            return (ssa, typ)
    }

and generateQualifiedIdentifier (name: string) : MLIRBuilder<string * MLIRType> =
    mlir {
        let! ssa = nextSSA "qid"
        do! emitUndef ssa MLIRTypes.i32 (Some name)
        do! newline
        return (ssa, MLIRTypes.i32)
    }

and generateApplication (funcExpr: SynExpr) (argExpr: SynExpr) : MLIRBuilder<string * MLIRType> =
    mlir {
        match funcExpr with
        | SynExpr.Ident ident when ident.idText = "op_PipeRight" || ident.idText = "|>" ->
            return! generateExpression argExpr
            
        | SynExpr.Ident ident ->
            return! generateKnownCall ident.idText argExpr
            
        | SynExpr.LongIdent(_, SynLongIdent(ids, _, _), _, _) ->
            let funcName = ids |> List.map (fun id -> id.idText) |> String.concat "."
            return! generateKnownCall funcName argExpr
            
        | SynExpr.TypeApp(SynExpr.Ident ident, _, _, _, _, _, _) ->
            return! generateKnownCall ident.idText argExpr
            
        | _ ->
            let! (funcSSA, funcType) = generateExpression funcExpr
            let! (argSSA, argType) = generateExpression argExpr
            let! resultSSA = nextSSA "app"
            
            // Check if we're calling a function pointer
            match funcType.Category with
            | MLIRTypeCategory.MemRef ->
                // Function pointer call
                do! emitIndirectCall resultSSA funcSSA argSSA argType MLIRTypes.i32
            | _ ->
                // Regular call (shouldn't happen with proper typing)
                do! emitUndef resultSSA MLIRTypes.i32 (Some "invalid function call")
            do! newline
            return (resultSSA, MLIRTypes.i32)
    }

and generateKnownCall (funcName: string) (argExpr: SynExpr) : MLIRBuilder<string * MLIRType> =
    mlir {
        match funcName with
        | "op_PipeRight" | "|>" ->
            // Pipe operator - evaluate argument
            return! generateExpression argExpr
            
        | "call" when isCallPlaceholder argExpr ->
            // This is a placeholder for a function call
            let! callSSA = nextSSA "call"
            do! emitUndef callSSA (MLIRTypes.memref MLIRTypes.i8) (Some (sprintf "call %s" funcName))
            do! newline
            return (callSSA, MLIRTypes.memref MLIRTypes.i8)
            
        // ... rest of the cases remain the same
        | _ ->
            let! (argSSA, argType) = generateExpression argExpr
            let! resultSSA = nextSSA "call"
            do! emitUndef resultSSA MLIRTypes.i32 (Some (sprintf "call %s" funcName))
            do! newline
            return (resultSSA, MLIRTypes.i32)
    }

and isCallPlaceholder expr =
    match expr with
    | SynExpr.Ident ident -> 
        match ident.idText with
        | "op_PipeRight" | "op_PipeLeft" | "printf" | "sprintf" | "printfn" -> true
        | _ -> false
    | _ -> false

and generateLet (bindings: SynBinding list) (body: SynExpr) : MLIRBuilder<string * MLIRType> =
    mlir {
        for binding in bindings do
            do! processLetBinding binding
        return! generateExpression body
    }

and processLetBinding (SynBinding(_, _, _, _, _, _, _, pat, _, expr, _, _, _)) : MLIRBuilder<unit> =
    mlir {
        match pat with
        | SynPat.Named(SynIdent(ident, _), _, _, _) ->
            let! (ssa, _) = generateExpression expr
            do! bindLocal ident.idText ssa
        | _ -> ()
    }

and generateMatch (matchExpr: SynExpr) (clauses: SynMatchClause list) : MLIRBuilder<string * MLIRType> =
    mlir {
        let! (matchSSA, _) = generateExpression matchExpr
        
        match clauses with
        | SynMatchClause(pat, _, resultExpr, _, _, _) :: _ ->
            match pat with
            | SynPat.Named(SynIdent(ident, _), _, _, _) ->
                do! bindLocal ident.idText matchSSA
            | _ -> ()
            return! generateExpression resultExpr
        | [] ->
            let! ssa = nextSSA "match"
            do! emitLine (sprintf "%s = arith.constant 0 : i32  // empty match" ssa)
            return (ssa, MLIRTypes.i32)
    }

and generateFieldAccess (targetExpr: SynExpr) (fieldIds: Ident list) : MLIRBuilder<string * MLIRType> =
    mlir {
        let! (targetSSA, _) = generateExpression targetExpr
        let fieldName = fieldIds |> List.map (fun id -> id.idText) |> String.concat "."
        let! ssa = nextSSA "field"
        do! emitUndef ssa MLIRTypes.i32 (Some (sprintf "field %s of %s" fieldName targetSSA))
        do! newline
        return (ssa, MLIRTypes.i32)
    }

and generateTuple (exprs: SynExpr list) : MLIRBuilder<string * MLIRType> =
    mlir {
        let! ssas = 
            exprs |> List.fold (fun accM expr ->
                mlir {
                    let! acc = accM
                    let! (ssa, _) = generateExpression expr
                    return ssa :: acc
                }
            ) (mlir.Return [])
        
        match List.rev ssas with
        | [ssa] -> return (ssa, MLIRTypes.i32)
        | _ ->
            let! tupleSSA = nextSSA "tuple"
            do! emitUndef tupleSSA MLIRTypes.i32 (Some "tuple")
            do! newline
            return (tupleSSA, MLIRTypes.i32)
    }

/// Generate MLIR function
let generateFunction (functionName: string) (attributes: SynAttributes) (expression: SynExpr) : MLIRBuilder<unit> =
    mlir {
        let! state = getState
        let qualifiedName = 
            match state.CurrentFunction with
            | Some module' -> module' + "." + functionName
            | None -> functionName
        
        let isMain = attributes |> List.exists (fun attrList ->
            attrList.Attributes |> List.exists (fun attr ->
                match attr.TypeName with
                | SynLongIdent([ident], [], [None]) -> ident.idText = "EntryPoint"
                | _ -> false))
        
        let mlirName = if isMain then "@main" else "@" + functionName
        
        do! emitLine (sprintf "func.func %s() -> i32 {" mlirName)
        do! indent (mlir {
            do! updateState (fun s -> { s with 
                                         CurrentFunction = Some functionName
                                         LocalVars = Map.empty
                                         SSACounter = 0 })
            
            let! (resultSSA, resultType) = generateExpression expression
            
            match resultType with
            | t when t = MLIRTypes.void_ || isMain ->
                let! zeroSSA = nextSSA "ret"
                do! emitConstant zeroSSA 0 MLIRTypes.i32
                do! newline
                do! emitReturn zeroSSA MLIRTypes.i32
            | _ ->
                do! emitReturn resultSSA resultType
            do! newline
        })
        do! emitLine "}"
    }

/// Process binding
let processBinding (SynBinding(_, _, _, _, attributes, _, _, pattern, _, expression, _, _, _)) : MLIRBuilder<unit> =
    mlir {
        match pattern with
        | SynPat.Named(SynIdent(ident, _), _, _, _) ->
            do! generateFunction ident.idText attributes expression
        | SynPat.LongIdent(SynLongIdent(ids, _, _), _, _, _, _, _) ->
            let name = ids |> List.map (fun id -> id.idText) |> List.last
            do! generateFunction name attributes expression
        | _ -> ()
    }

/// Process declaration
let rec processDeclaration (decl: SynModuleDecl) : MLIRBuilder<unit> =
    mlir {
        match decl with
        | SynModuleDecl.Let(_, bindings, _) ->
            for binding in bindings do
                do! processBinding binding
        | SynModuleDecl.NestedModule(componentInfo, _, nestedDecls, _, _, _) ->
            let (SynComponentInfo(_, _, _, longId, _, _, _, _)) = componentInfo
            let nestedName = longId |> List.map (fun ident -> ident.idText) |> String.concat "."
            do! updateState (fun s -> 
                let fullPath = 
                    match s.CurrentFunction with
                    | Some curr -> curr + "." + nestedName
                    | None -> nestedName
                { s with CurrentFunction = Some fullPath })
            for decl in nestedDecls do
                do! processDeclaration decl
        | _ -> ()
    }

/// Process module
let processModuleOrNamespace (SynModuleOrNamespace(longId, _, _, declarations, _, _, _, _, _)) : MLIRBuilder<unit> =
    mlir {
        let modulePath = longId |> List.map (fun ident -> ident.idText) |> String.concat "."
        do! updateState (fun s -> { s with CurrentFunction = Some modulePath })
        for decl in declarations do
            do! processDeclaration decl
    }

/// Process input file
let processInputFile (parsedInput: ParsedInput) : MLIRBuilder<unit> =
    mlir {
        match parsedInput with
        | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
            for module' in modules do
                do! processModuleOrNamespace module'
        | ParsedInput.SigFile(_) -> ()
    }

/// Emit external declarations
let emitExternalDeclarations : MLIRBuilder<unit> =
    mlir {
        let! state = getState
        for ext in state.RequiredExternals do
            match ext with
            | "printf" ->
                // All pointer parameters are opaque
                do! emit "func.func private @printf(!llvm.ptr, ...) -> i32"
                do! newline
            | "sprintf" ->
                do! emit "func.func private @sprintf(!llvm.ptr, !llvm.ptr, ...) -> !llvm.ptr"
                do! newline
            | "fgets" ->
                do! emit "func.func private @fgets(!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr"
                do! newline
            | "strlen" ->
                do! emit "func.func private @strlen(!llvm.ptr) -> i32"
                do! newline
            | _ -> ()
    }

/// Generate MLIR for a complete program
let generateProgram (programName: string) (typeCtx: TypeContext) (symbolRegistry: SymbolRegistry) 
                   (reachableInputs: (string * ParsedInput) list) : string =
    
    let initialState = {
        Output = StringBuilder()
        Indent = 1
        SSACounter = 0
        LocalVars = Map.empty
        TypeContext = typeCtx
        SymbolRegistry = symbolRegistry
        RequiredExternals = Set.empty
        CurrentFunction = None
    }
    
    let builder = mlir {
        do! emitLine (sprintf "module @%s {" programName)
        
        for (_, parsedInput) in reachableInputs do
            do! processInputFile parsedInput
        
        do! emitExternalDeclarations
        
        do! updateState (fun s -> { s with Indent = 0 })
        do! emitLine "}"
    }
    
    let (_, finalState) = builder initialState
    finalState.Output.ToString()