module Core.MLIRGeneration.DirectGenerator

open System.Text
open FSharp.Compiler.Syntax
open FSharp.Compiler.Symbols
open Core.MLIRGeneration.TypeMapping
open Core.MLIRGeneration.TypeSystem
open Core.MLIRGeneration.Dialect
open Core.XParsec.Foundation
open Dabbit.Bindings.SymbolRegistry
open Dabbit.Bindings.PatternLibrary

/// Critical error for undefined MLIR generation
exception MLIRGenerationException of string * string option

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
    LocalVars: Map<string, (string * MLIRType)>  // SSA name and type
    TypeContext: TypeContext
    SymbolRegistry: SymbolRegistry
    RequiredExternals: Set<string>
    CurrentFunction: string option
    GeneratedFunctions: Set<string>
    CurrentModule: string list  // Module path for symbol resolution
    HasErrors: bool  // Track if we've encountered errors
}

/// MLIR builder computation expression
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
            
    member _.TryWith(body: MLIRBuilder<'T>, handler: exn -> MLIRBuilder<'T>) : MLIRBuilder<'T> =
        fun state ->
            try
                body state
            with e ->
                (handler e) state
                
    member _.TryFinally(body: MLIRBuilder<'T>, compensation: unit -> unit) : MLIRBuilder<'T> =
        fun state ->
            try
                body state
            finally
                compensation()

let mlir = MLIRBuilderCE()

/// Fail with hard error - no soft landings
let failHard (phase: string) (message: string) : MLIRBuilder<'T> =
    fun state ->
        let location = 
            match state.CurrentFunction with
            | Some f -> sprintf " in function '%s'" f
            | None -> ""
        let state' = { state with HasErrors = true }
        raise (MLIRGenerationException(sprintf "[%s]%s: %s" phase location message, None))

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

/// Bind local variable with type
let bindLocal (name: string) (ssa: string) (typ: MLIRType) : MLIRBuilder<unit> =
    updateState (fun s -> { s with LocalVars = Map.add name (ssa, typ) s.LocalVars })

/// Lookup local variable
let lookupLocal (name: string) : MLIRBuilder<(string * MLIRType) option> =
    mlir {
        let! state = getState
        return Map.tryFind name state.LocalVars
    }

/// Resolve symbol in registry
let resolveSymbol (name: string) : MLIRBuilder<ResolvedSymbol> =
    mlir {
        let! state = getState
        // Add current module context to registry for resolution
        let registryWithContext = 
            RegistryConstruction.withNamespaceContext state.CurrentModule state.SymbolRegistry
        
        match RegistryConstruction.resolveSymbolInRegistry name registryWithContext with
        | Success (symbol, _) -> return symbol
        | CompilerFailure _ ->
            return! failHard "Symbol Resolution" 
                (sprintf "Cannot resolve symbol '%s'. Available symbols: %s" 
                    name 
                    (state.SymbolRegistry.State.SymbolsByShort |> Map.toList |> List.map fst |> String.concat ", "))
    }

/// Emit MLIR type
let emitType (t: MLIRType) : MLIRBuilder<unit> =
    emit (mlirTypeToStringWithLLVM t)

/// Emit SSA value
let emitSSA (name: string) : MLIRBuilder<unit> =
    emit name

/// Emit operation result assignment
let emitResultAssign (ssa: string) : MLIRBuilder<unit> =
    mlir {
        do! emitSSA ssa
        do! emit " = "
    }

/// Emit memory allocation
let emitAlloca (ssa: string) (size: int) : MLIRBuilder<unit> =
    mlir {
        do! emitResultAssign ssa
        do! emit (sprintf "memref.alloca() : memref<%dxi8>" size)
    }

/// Emit function call with proper types
let emitCall (ssa: string) (func: string) (args: string list) (argTypes: MLIRType list) (retType: MLIRType) : MLIRBuilder<unit> =
    mlir {
        if retType = MLIRTypes.void_ then
            do! emitIndented (sprintf "call @%s(" func)
        else
            do! emitResultAssign ssa
            do! emit (sprintf "func.call @%s(" func)
        
        do! emit (String.concat ", " args)
        do! emit ") : ("
        do! emit (argTypes |> List.map mlirTypeToStringWithLLVM |> String.concat ", ")
        do! emit ") -> "
        do! emitType retType
    }

/// Generate call based on symbol registry
let rec generateSymbolCall (symbol: ResolvedSymbol) (args: (string * MLIRType) list) : MLIRBuilder<string * MLIRType> =
    mlir {
        match symbol.Operation with
        | MLIROperationPattern.DialectOp(dialect, operation, attrs) ->
            let! resultSSA = nextSSA "op"
            do! emitIndented (sprintf "%s = %s.%s" resultSSA (dialectToString dialect) operation)
            
            if not args.IsEmpty then
                do! emit "("
                do! emit (args |> List.map fst |> String.concat ", ")
                do! emit ")"
            
            // Emit attributes
            if not (Map.isEmpty attrs) then
                do! emit " {"
                let attrStrs = attrs |> Map.toList |> List.map (fun (k, v) -> sprintf "%s = %s" k v)
                do! emit (String.concat ", " attrStrs)
                do! emit "}"
            
            do! emit " : "
            if not args.IsEmpty then
                do! emit "("
                do! emit (args |> List.map (snd >> mlirTypeToStringWithLLVM) |> String.concat ", ")
                do! emit ") -> "
            do! emitType symbol.ReturnType
            do! newline
            
            return (resultSSA, symbol.ReturnType)
            
        | MLIROperationPattern.ExternalCall(funcName, lib) ->
            let! resultSSA = nextSSA "call"
            do! requireExternal funcName
            do! emitCall resultSSA funcName (args |> List.map fst) (args |> List.map snd) symbol.ReturnType
            do! newline
            return (resultSSA, symbol.ReturnType)
            
        | MLIROperationPattern.Composite operations ->
            // Handle composite operations - chain them together
            let! finalResult = 
                operations |> List.fold (fun accM (i, op) ->
                    mlir {
                        let! (prevSSA, prevType) = accM
                        let stepArgs = 
                            if i = 0 then args  // First operation gets original args
                            else [(prevSSA, prevType)]  // Subsequent ops get previous result
                        
                        match op with
                        | MLIROperationPattern.ExternalCall(funcName, lib) ->
                            let! stepSSA = nextSSA (sprintf "step%d" i)
                            do! requireExternal funcName
                            
                            // Determine return type for this step
                            let stepRetType = 
                                match funcName with
                                | "fgets" -> MLIRTypes.memref MLIRTypes.i8
                                | "strlen" -> MLIRTypes.i32
                                | _ -> MLIRTypes.i32
                            
                            do! emitCall stepSSA funcName (stepArgs |> List.map fst) (stepArgs |> List.map snd) stepRetType
                            do! newline
                            return (stepSSA, stepRetType)
                            
                        | MLIROperationPattern.Transform(transformName, params') ->
                            let! stepSSA = nextSSA (sprintf "transform%d" i)
                            
                            match transformName with
                            | "result_wrapper" ->
                                // Wrap the result in a Result type (simplified for now)
                                do! emitLine (sprintf "%s = arith.constant 1 : i32  // Ok tag" stepSSA)
                                return (stepSSA, MLIRTypes.i32)
                            | _ ->
                                do! emitLine (sprintf "%s = arith.constant 0 : i32  // Transform: %s" stepSSA transformName)
                                return (stepSSA, MLIRTypes.i32)
                                
                        | _ ->
                            let! stepSSA = nextSSA "composite_step"
                            do! emitLine (sprintf "%s = arith.constant 0 : i32  // Unsupported composite step" stepSSA)
                            return (stepSSA, MLIRTypes.i32)
                    }
                ) (mlir.Return (args |> List.head))
                 (operations |> List.indexed)
            
            return finalResult
            
        | MLIROperationPattern.Transform(transformName, params') ->
            let! resultSSA = nextSSA "transform"
            
            match transformName with
            | "span_to_string" ->
                // Just pass through the buffer for now
                match args with
                | [(argSSA, argType)] -> return (argSSA, argType)
                | _ -> 
                    do! emitLine (sprintf "%s = arith.constant 0 : i32  // Transform error" resultSSA)
                    return (resultSSA, MLIRTypes.i32)
            | _ ->
                do! emitLine (sprintf "%s = arith.constant 0 : i32  // Transform: %s" resultSSA transformName)
                return (resultSSA, MLIRTypes.i32)
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
            // Handle generic type application
            return! generateExpression baseExpr
            
        | SynExpr.LetOrUse(_, isUse, bindings, body, _, _) ->
            return! generateLet isUse bindings body
            
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
            return! failHard "Expression Generation" 
                (sprintf "Unsupported expression type: %A" (expr.GetType().Name))
    }

and generateConstant (constant: SynConst) : MLIRBuilder<string * MLIRType> =
    mlir {
        match constant with
        | SynConst.Int32 n ->
            let! ssa = nextSSA "c"
            do! emitLine (sprintf "%s = arith.constant %d : i32" ssa n)
            return (ssa, MLIRTypes.i32)
            
        | SynConst.String(text, _, _) ->
            // For now, create a global string constant
            let! ssa = nextSSA "str"
            do! requireExternal "string_literal"
            do! emitLine (sprintf "%s = llvm.mlir.addressof @.str_%d : !llvm.ptr" ssa (text.GetHashCode()))
            return (ssa, MLIRTypes.memref MLIRTypes.i8)
            
        | SynConst.Unit ->
            // Unit is void - no value needed
            let! ssa = nextSSA "unit"
            return (ssa, MLIRTypes.void_)
            
        | _ ->
            return! failHard "Constant Generation" 
                (sprintf "Unsupported constant type: %A" constant)
    }

and generateIdentifier (ident: Ident) : MLIRBuilder<string * MLIRType> =
    mlir {
        let! maybeLocal = lookupLocal ident.idText
        match maybeLocal with
        | Some (ssa, typ) -> return (ssa, typ)
        | None ->
            // Not a local - might be a function reference
            return! failHard "Identifier Resolution" 
                (sprintf "Unbound identifier '%s'" ident.idText)
    }

and generateQualifiedIdentifier (name: string) : MLIRBuilder<string * MLIRType> =
    mlir {
        // Try to resolve as a symbol
        let! symbol = resolveSymbol name
        // Return a function reference
        let! ssa = nextSSA "fref"
        return (ssa, MLIRTypes.func symbol.ParameterTypes symbol.ReturnType)
    }

and generateApplication (funcExpr: SynExpr) (argExpr: SynExpr) : MLIRBuilder<string * MLIRType> =
    mlir {
        match funcExpr with
        | SynExpr.Ident ident when ident.idText = "op_PipeRight" || ident.idText = "|>" ->
            // Pipe operator - just evaluate the argument
            return! generateExpression argExpr
            
        | SynExpr.Ident ident ->
            // Direct function call
            let! symbol = resolveSymbol ident.idText
            let! (argSSA, argType) = generateExpression argExpr
            return! generateSymbolCall symbol [(argSSA, argType)]
            
        | SynExpr.LongIdent(_, SynLongIdent(ids, _, _), _, _) ->
            let funcName = ids |> List.map (fun id -> id.idText) |> String.concat "."
            let! symbol = resolveSymbol funcName
            let! (argSSA, argType) = generateExpression argExpr
            return! generateSymbolCall symbol [(argSSA, argType)]
            
        | SynExpr.TypeApp(SynExpr.Ident ident, _, _, _, _, _, _) ->
            // Generic function call (e.g., stackBuffer<byte>)
            let! symbol = resolveSymbol ident.idText
            let! (argSSA, argType) = generateExpression argExpr
            
            // Special handling for stackBuffer
            if ident.idText = "stackBuffer" then
                match argExpr with
                | SynExpr.Const(SynConst.Int32 size, _) ->
                    let! bufferSSA = nextSSA "buffer"
                    do! emitAlloca bufferSSA size
                    do! newline
                    return (bufferSSA, MLIRTypes.memref MLIRTypes.i8)
                | _ ->
                    return! failHard "stackBuffer" "Size must be a constant"
            else
                return! generateSymbolCall symbol [(argSSA, argType)]
            
        | _ ->
            return! failHard "Application" 
                "Complex function expressions not yet supported"
    }

and generateLet (isUse: bool) (bindings: SynBinding list) (body: SynExpr) : MLIRBuilder<string * MLIRType> =
    mlir {
        for binding in bindings do
            do! processLetBinding binding
        return! generateExpression body
    }

and processLetBinding (SynBinding(_, _, _, _, _, _, _, pat, _, expr, _, _, _)) : MLIRBuilder<unit> =
    mlir {
        match pat with
        | SynPat.Named(SynIdent(ident, _), _, _, _) ->
            let! (ssa, typ) = generateExpression expr
            do! bindLocal ident.idText ssa typ
        | _ ->
            return! failHard "Let Binding" "Complex patterns not yet supported"
    }

and generateMatch (matchExpr: SynExpr) (clauses: SynMatchClause list) : MLIRBuilder<string * MLIRType> =
    mlir {
        let! (matchSSA, matchType) = generateExpression matchExpr
        
        // For now, simplified match handling
        match clauses with
        | SynMatchClause(SynPat.LongIdent(SynLongIdent([okIdent], _, _), _, _, _, _, _), _, okExpr, _, _, _) :: 
          SynMatchClause(SynPat.LongIdent(SynLongIdent([errorIdent], _, _), _, _, _, _, _), _, errorExpr, _, _, _) :: _ 
            when okIdent.idText = "Ok" && errorIdent.idText = "Error" ->
            // Result pattern match
            let! resultSSA = nextSSA "match_result"
            
            // For now, assume Ok case (proper implementation would check the tag)
            let! (okSSA, okType) = generateExpression okExpr
            return (okSSA, okType)
            
        | _ ->
            return! failHard "Match Expression" "Complex match patterns not yet supported"
    }

and generateFieldAccess (targetExpr: SynExpr) (fieldIds: Ident list) : MLIRBuilder<string * MLIRType> =
    mlir {
        let! (targetSSA, targetType) = generateExpression targetExpr
        let fieldName = fieldIds |> List.map (fun id -> id.idText) |> String.concat "."
        
        // Special case for buffer.AsSpan
        if fieldName = "AsSpan" then
            // Just return the buffer itself for now
            return (targetSSA, targetType)
        else
            return! failHard "Field Access" 
                (sprintf "Field access '%s' not implemented" fieldName)
    }

and generateTuple (exprs: SynExpr list) : MLIRBuilder<string * MLIRType> =
    mlir {
        let! elements = 
            exprs |> List.fold (fun accM expr ->
                mlir {
                    let! acc = accM
                    let! (ssa, typ) = generateExpression expr
                    return (ssa, typ) :: acc
                }
            ) (mlir.Return [])
        
        let elements' = List.rev elements
        
        // For now, return the first element of a tuple
        match elements' with
        | (ssa, typ) :: _ -> return (ssa, typ)
        | [] -> 
            return! failHard "Tuple" "Empty tuple not supported"
    }

/// Generate MLIR function
let generateFunction (functionName: string) (attributes: SynAttributes) (expression: SynExpr) : MLIRBuilder<unit> =
    mlir {
        let! state = getState
        
        let qualifiedName = 
            match state.CurrentModule with
            | [] -> functionName
            | modules -> (modules @ [functionName]) |> String.concat "."
        
        // Check if already generated
        if Set.contains qualifiedName state.GeneratedFunctions then
            return ()
        else
            do! updateState (fun s -> { s with GeneratedFunctions = Set.add qualifiedName s.GeneratedFunctions })
            
            let isMain = attributes |> List.exists (fun attrList ->
                attrList.Attributes |> List.exists (fun attr ->
                    match attr.TypeName with
                    | SynLongIdent([ident], [], [None]) -> ident.idText = "EntryPoint"
                    | _ -> false))
            
            let mlirName = 
                if isMain then "@main" 
                else "@" + qualifiedName.Replace(".", "_")
            
            do! emitLine (sprintf "func.func %s() -> i32 {" mlirName)
            do! indent (mlir {
                do! updateState (fun s -> { s with 
                                             CurrentFunction = Some qualifiedName
                                             LocalVars = Map.empty
                                             SSACounter = 0 })
                
                try
                    let! (resultSSA, resultType) = generateExpression expression
                    
                    match resultType with
                    | t when t = MLIRTypes.void_ || isMain ->
                        let! zeroSSA = nextSSA "ret"
                        do! emitLine (sprintf "%s = arith.constant 0 : i32" zeroSSA)
                        do! emitLine (sprintf "func.return %s : i32" zeroSSA)
                    | _ ->
                        do! emitLine (sprintf "func.return %s : %s" resultSSA (mlirTypeToStringWithLLVM resultType))
                with
                | MLIRGenerationException(msg, _) ->
                    // Re-raise with function context
                    raise (MLIRGenerationException(msg, Some qualifiedName))
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
        | _ ->
            return! failHard "Binding" "Complex binding patterns not supported"
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
            let nestedName = longId |> List.map (fun ident -> ident.idText)
            let! state = getState
            do! updateState (fun s -> { s with CurrentModule = state.CurrentModule @ nestedName })
            for decl in nestedDecls do
                do! processDeclaration decl
            do! updateState (fun s -> { s with CurrentModule = state.CurrentModule })
        | _ -> ()
    }

/// Process module
let processModuleOrNamespace (SynModuleOrNamespace(longId, _, _, declarations, _, _, _, _, _)) : MLIRBuilder<unit> =
    mlir {
        let modulePath = longId |> List.map (fun ident -> ident.idText)
        do! updateState (fun s -> { s with CurrentModule = modulePath })
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
                do! emitLine "func.func private @printf(!llvm.ptr, ...) -> i32"
            | "sprintf" ->
                do! emitLine "func.func private @sprintf(!llvm.ptr, !llvm.ptr, ...) -> i32"
            | "fgets" ->
                do! emitLine "func.func private @fgets(!llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr"
            | "strlen" ->
                do! emitLine "func.func private @strlen(!llvm.ptr) -> i32"
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
        GeneratedFunctions = Set.empty
        CurrentModule = []
        HasErrors = false
    }
    
    let builder = mlir {
        do! emitLine (sprintf "module @%s {" programName)
        
        try
            for (_, parsedInput) in reachableInputs do
                do! processInputFile parsedInput
            
            do! emitExternalDeclarations
            
        with
        | MLIRGenerationException(msg, funcOpt) ->
            // Add context about where we failed
            let context = 
                match funcOpt with
                | Some func -> sprintf " (in function %s)" func
                | None -> ""
            return! failHard "MLIR Generation" (msg + context)
        
        do! updateState (fun s -> { s with Indent = 0 })
        do! emitLine "}"
    }
    
    try
        let (_, finalState) = builder initialState
        let mlir = finalState.Output.ToString()
        
        // Final validation - check for any undefined values
        if mlir.Contains("llvm.mlir.undef") then
            let lines = mlir.Split('\n')
            let undefLines = 
                lines 
                |> Array.indexed 
                |> Array.filter (fun (_, line) -> line.Contains("llvm.mlir.undef"))
                |> Array.map (fun (i, line) -> sprintf "Line %d: %s" (i + 1) (line.Trim()))
            
            raise (MLIRGenerationException(
                sprintf "Generated MLIR contains undefined values:\n%s" (String.concat "\n" undefLines),
                None))
        
        mlir
    with
    | MLIRGenerationException(msg, _) as ex ->
        // IMPORTANT: Return partial MLIR for debugging
        let (_, finalState) = builder initialState
        let partialMLIR = finalState.Output.ToString()
        
        // Add error marker to the MLIR
        let errorMLIR = 
            partialMLIR + 
            sprintf "\n// ERROR: %s\n" msg +
            "// PARTIAL MLIR GENERATED UP TO ERROR POINT\n"
        
        // Still throw the exception but the partial MLIR is available
        printfn "Partial MLIR generated (see .mlir file)"
        failwith (sprintf "MLIR Generation Failed: %s\n\nPartial MLIR:\n%s" msg errorMLIR)