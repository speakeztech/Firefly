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
    LocalFunctions: Map<string, MLIRType>  
    CurrentModule: string list  // Module path for symbol resolution
    OpenedNamespaces: string list list  // List of opened namespaces
    HasErrors: bool
    ExpectedType: MLIRType option  // Add this for type context
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

let builtInFunctions = 
    Map.ofList [
        // String operations
        "concat", ([MLIRTypes.string_; MLIRTypes.string_], MLIRTypes.string_)
        "replace", ([MLIRTypes.string_; MLIRTypes.string_; MLIRTypes.string_], MLIRTypes.string_)
        "intToString", ([MLIRTypes.i32], MLIRTypes.string_)
        "toString", ([MLIRTypes.i32], MLIRTypes.string_)
        "length", ([MLIRTypes.string_], MLIRTypes.i32)
        
        // Array operations  
        "Array.zeroCreate", ([MLIRTypes.i32], MLIRTypes.memref MLIRTypes.i8)
        "Array.create", ([MLIRTypes.i32; MLIRTypes.i32], MLIRTypes.memref MLIRTypes.i32)
        "Array.length", ([MLIRTypes.memref MLIRTypes.i8], MLIRTypes.i32)
        
        // Math operations
        "min", ([MLIRTypes.i32; MLIRTypes.i32], MLIRTypes.i32)
        "max", ([MLIRTypes.i32; MLIRTypes.i32], MLIRTypes.i32)
        "abs", ([MLIRTypes.i32], MLIRTypes.i32)
    ]

/// Get current state
let getState : MLIRBuilder<MLIRBuilderState> =
    fun state -> (state, state)

/// Set state
let setState (newState: MLIRBuilderState) : MLIRBuilder<unit> =
    fun _ -> ((), newState)

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

/// Fail with hard error - no soft landings
let failHard (phase: string) (message: string) : MLIRBuilder<'T> =
    fun state ->
        // Emit error marker to the output
        state.Output.AppendLine() |> ignore
        state.Output.AppendLine(sprintf "    // ERROR: %s - %s" phase message) |> ignore
        
        let location = 
            match state.CurrentFunction with
            | Some f -> sprintf " in function '%s'" f
            | None -> ""
        let state' = { state with HasErrors = true }
        raise (MLIRGenerationException(sprintf "[%s]%s: %s" phase location message, None))

/// Generate built-in function call
let generateBuiltInCall (funcName: string) (args: (string * MLIRType) list) : MLIRBuilder<string * MLIRType> =
    mlir {
        let! resultSSA = nextSSA funcName
        
        match funcName with
        | "concat" ->
            match args with
            | [(left, _); (right, _)] ->
                // For now, just return the first argument
                return (left, MLIRTypes.string_)
            | _ -> return! failHard "concat" "Expected 2 arguments"
            
        | "replace" ->
            match args with
            | [(str, _); (pattern, _); (replacement, _)] ->
                // For now, just return the string argument
                return (str, MLIRTypes.string_)
            | _ -> return! failHard "replace" "Expected 3 arguments"
            
        | "intToString" | "toString" ->
            match args with
            | [(value, _)] ->
                // Simplified: just return a dummy string
                do! emitLine (sprintf "%s = llvm.mlir.addressof @.str.num : !llvm.ptr" resultSSA)
                return (resultSSA, MLIRTypes.string_)
            | _ -> return! failHard funcName "Expected 1 argument"
            
        | _ ->
            // Generic built-in handling
            do! emitLine (sprintf "%s = arith.constant 0 : i32  // Built-in: %s" resultSSA funcName)
            return (resultSSA, MLIRTypes.i32)
    }

/// Convert F# type syntax to MLIR type
let rec synTypeToMLIRType (synType: SynType) : MLIRType =
    match synType with
    | SynType.LongIdent(SynLongIdent(ids, _, _)) ->
        let typeName = ids |> List.map (fun id -> id.idText) |> String.concat "."
        match typeName with
        | "int" | "int32" -> MLIRTypes.i32
        | "int64" -> MLIRTypes.i64
        | "byte" | "uint8" -> MLIRTypes.i8
        | "bool" -> MLIRTypes.i1
        | "float" | "float32" -> MLIRTypes.f32
        | "double" | "float64" -> MLIRTypes.f64
        | "string" -> MLIRTypes.memref MLIRTypes.i8
        | "unit" -> MLIRTypes.void_
        | _ -> MLIRTypes.i32  // Default fallback
            
    | SynType.Array(rank, elementType, _) ->
        let elemType = synTypeToMLIRType elementType
        if rank = 1 then
            MLIRTypes.memref elemType
        else
            MLIRTypes.memref elemType  // Simplified for multi-dim
            
    | SynType.App(typeName, _, typeArgs, _, _, _, _) ->
        match typeName with
        | SynType.LongIdent(SynLongIdent([id], _, _)) when id.idText = "Span" ->
            match typeArgs with
            | [elementType] -> MLIRTypes.memref (synTypeToMLIRType elementType)
            | _ -> MLIRTypes.memref MLIRTypes.i8
        | _ -> MLIRTypes.i32  // Default fallback
        
    | SynType.Fun(argType, returnType, _, _) ->
        let argMLIR = synTypeToMLIRType argType
        let retMLIR = synTypeToMLIRType returnType
        MLIRTypes.func [argMLIR] retMLIR
        
    | SynType.Paren(innerType, _) ->
        synTypeToMLIRType innerType
        
    | _ -> MLIRTypes.i32  // Default fallback

/// Generate type conversion between MLIR types
let generateTypeConversion (fromValue: string, fromType: MLIRType) (toType: MLIRType) : MLIRBuilder<string * MLIRType> =
    mlir {
        if fromType = toType then
            return (fromValue, fromType)
        else
            match fromType.Category, toType.Category with
            | MLIRTypeCategory.Integer, MLIRTypeCategory.Integer ->
                let! resultName = nextSSA "cast"
                if fromType.Width < toType.Width then
                    do! emitLine (sprintf "%s = arith.extsi %s : %s to %s" 
                                         resultName fromValue 
                                         (mlirTypeToString fromType) 
                                         (mlirTypeToString toType))
                else
                    do! emitLine (sprintf "%s = arith.trunci %s : %s to %s" 
                                         resultName fromValue 
                                         (mlirTypeToString fromType) 
                                         (mlirTypeToString toType))
                return (resultName, toType)
            | _ ->
                // For now, just return the value as-is
                return (fromValue, toType)
    }

/// Resolve symbol in registry with namespace handling
let resolveSymbol (name: string) : MLIRBuilder<ResolvedSymbol> =
    mlir {
        let! state = getState
        
        // Check if it's a local function first
        match Map.tryFind name state.LocalFunctions with
        | Some funcType ->
            // Create a pseudo-symbol for the local function
            let qualifiedName = 
                match state.CurrentModule with
                | [] -> name
                | modules -> (modules @ [name]) |> String.concat "."
                
            let symbol = {
                QualifiedName = qualifiedName
                ShortName = name
                ParameterTypes = []  // Simplified
                ReturnType = MLIRTypes.i32  // Simplified
                Operation = ExternalCall(qualifiedName.Replace(".", "_"), None)  // Use qualified name
                Namespace = ""
                SourceLibrary = "Local"
                RequiresExternal = false
            }
            return symbol
        | None ->
            // Continue with existing resolution logic...
            // Debug: emit opened namespaces
            if name = "String.format" then
                do! emit (sprintf "\n    // DEBUG: Resolving '%s', opened namespaces: %s\n" 
                            name 
                            (state.OpenedNamespaces |> List.map (String.concat ".") |> String.concat ", "))
            
            // Try direct resolution first
            let registryWithContext = 
                RegistryConstruction.withNamespaceContext state.CurrentModule state.SymbolRegistry
            
            match RegistryConstruction.resolveSymbolInRegistry name registryWithContext with
            | Success (symbol, _) -> return symbol
            | CompilerFailure _ ->
                // Try with opened namespaces
                let mutable found = None
                for ns in state.OpenedNamespaces do
                    if found.IsNone then
                        let qualifiedName = (ns @ [name]) |> String.concat "."
                        if name = "String.format" then
                            do! emit (sprintf "    // DEBUG: Trying '%s'\n" qualifiedName)
                        match RegistryConstruction.resolveSymbolInRegistry qualifiedName registryWithContext with
                        | Success (symbol, _) -> found <- Some symbol
                        | CompilerFailure _ -> ()
                
                match found with
                | Some symbol -> return symbol
                | None ->
                    // Also try partial namespace resolution (e.g., String.format -> Alloy.IO.String.format)
                    if name.Contains(".") then
                        let parts = name.Split('.')
                        let mutable partialFound = None
                        
                        for ns in state.OpenedNamespaces do
                            if partialFound.IsNone then
                                let qualifiedName = (ns @ Array.toList parts) |> String.concat "."
                                if name = "String.format" then
                                    do! emit (sprintf "    // DEBUG: Trying partial '%s'\n" qualifiedName)
                                match RegistryConstruction.resolveSymbolInRegistry qualifiedName registryWithContext with
                                | Success (symbol, _) -> partialFound <- Some symbol
                                | CompilerFailure _ -> ()
                        
                        match partialFound with
                        | Some symbol -> return symbol
                        | None ->
                            // List all available symbols in all namespaces
                            let allSymbols = 
                                state.SymbolRegistry.State.SymbolsByQualified 
                                |> Map.toList 
                                |> List.map fst
                                |> List.filter (fun s -> s.Contains("format"))
                                |> String.concat ", "
                            
                            return! failHard "Symbol Resolution" 
                                (sprintf "Cannot resolve symbol '%s'. Format-related symbols: %s" 
                                    name allSymbols)
                    else
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
        if symbol.SourceLibrary = "Local" then
            let! resultSSA = nextSSA "call"
            let funcName = "@" + symbol.Operation.ToString().Replace("ExternalCall(", "").Replace(", None)", "")
            do! emitCall resultSSA funcName (args |> List.map fst) (args |> List.map snd) symbol.ReturnType
            do! newline
            return (resultSSA, symbol.ReturnType)
        else
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
            let rec processOperations (ops: (int * MLIROperationPattern) list) (currentArgs: (string * MLIRType) list) : MLIRBuilder<string * MLIRType> =
                mlir {
                    match ops with
                    | [] -> 
                        // No operations, return the first argument
                        match currentArgs with
                        | (ssa, typ) :: _ -> return (ssa, typ)
                        | [] -> return! failHard "Composite" "No arguments for composite operation"
                        
                    | (i, op) :: rest ->
                        // Process current operation
                        let! (stepSSA, stepType) = 
                            match op with
                            | MLIROperationPattern.ExternalCall(funcName, lib) ->
                                mlir {
                                    let! stepSSA = nextSSA (sprintf "step%d" i)
                                    do! requireExternal funcName
                                    
                                    // Determine return type for this step
                                    let stepRetType = 
                                        match funcName with
                                        | "fgets" -> MLIRTypes.memref MLIRTypes.i8
                                        | "strlen" -> MLIRTypes.i32
                                        | _ -> MLIRTypes.i32
                                    
                                    do! emitCall stepSSA funcName (currentArgs |> List.map fst) (currentArgs |> List.map snd) stepRetType
                                    do! newline
                                    return (stepSSA, stepRetType)
                                }
                                
                            | MLIROperationPattern.Transform(transformName, params') ->
                                mlir {
                                    let! stepSSA = nextSSA (sprintf "transform%d" i)
                                    
                                    match transformName with
                                    | "result_wrapper" ->
                                        // Wrap the result in a Result type (simplified for now)
                                        do! emitLine (sprintf "%s = arith.constant 1 : i32  // Ok tag" stepSSA)
                                        return (stepSSA, MLIRTypes.i32)
                                    | _ ->
                                        do! emitLine (sprintf "%s = arith.constant 0 : i32  // Transform: %s" stepSSA transformName)
                                        return (stepSSA, MLIRTypes.i32)
                                }
                                
                            | _ ->
                                mlir {
                                    let! stepSSA = nextSSA "composite_step"
                                    do! emitLine (sprintf "%s = arith.constant 0 : i32  // Unsupported composite step" stepSSA)
                                    return (stepSSA, MLIRTypes.i32)
                                }
                        
                        // Continue with remaining operations using the result of this step
                        return! processOperations rest [(stepSSA, stepType)]
                }
            
            return! processOperations (operations |> List.indexed) args
            
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

and generateExpression (expr: SynExpr) : MLIRBuilder<string * MLIRType> =
    mlir {
        match expr with
        | SynExpr.Const(constant, _) ->
            return! generateConstant constant
            
        | SynExpr.Ident ident ->
            return! generateIdentifier ident
            
        | SynExpr.New(_, targetType, argExpr, _) ->
            // Handle constructor calls
            let! (argSSA, argType) = generateExpression argExpr
            // For StackBuffer, this is essentially an alloca
            match targetType with
            | SynType.App(SynType.LongIdent(SynLongIdent(ids, _, _)), _, _, _, _, _, _) ->
                let typeName = ids |> List.map (fun id -> id.idText) |> String.concat "."
                if typeName.Contains("StackBuffer") then
                    // For StackBuffer, generate an alloca
                    let! bufferSSA = nextSSA "buffer"
                    // The size should come from the argument
                    match argExpr with
                    | SynExpr.Const(SynConst.Int32 size, _) ->
                        do! emitAlloca bufferSSA size
                    | _ ->
                        // Dynamic size - use the computed value
                        do! emitLine (sprintf "%s = memref.alloca(%s) : memref<?xi8>" bufferSSA argSSA)
                    do! newline
                    return (bufferSSA, MLIRTypes.memref MLIRTypes.i8)
                else
                    return! failHard "Constructor" (sprintf "Constructor for type '%s' not implemented" typeName)
            | _ ->
                return! failHard "Constructor" "Complex type constructors not supported"
            
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

        | SynExpr.For(spFor, spTo, ident, equalsRange, startExpr, isToNotDownto, endExpr, doBody, range) ->
            return! generateFor ident startExpr endExpr isToNotDownto doBody
            
        | SynExpr.Typed(innerExpr, synType, _) ->
            // Handle typed expressions with type preservation
            let targetType = synTypeToMLIRType synType
            
            // Save current expected type
            let! state = getState
            let previousExpectedType = state.ExpectedType
            
            // Set expected type for inner expression
            do! setState { state with ExpectedType = Some targetType }
            
            // Generate inner expression
            let! (value, actualType) = generateExpression innerExpr
            
            // Restore previous expected type
            do! updateState (fun s -> { s with ExpectedType = previousExpectedType })
            
            // Generate type conversion if needed
            if actualType = targetType then
                return (value, targetType)
            else
                return! generateTypeConversion (value, actualType) targetType
            
        | SynExpr.LongIdent(_, SynLongIdent(ids, _, _), _, _) ->
            let parts = ids |> List.map (fun id -> id.idText)
            match parts with
            | [varName; methodName] ->
                // This is a method-like call (e.g., buffer.AsSpan)
                // Look up the variable
                let! maybeVar = lookupLocal varName
                match maybeVar with
                | Some (varSSA, varType) ->
                    // Handle known methods
                    match methodName with
                    | "AsSpan" ->
                        // For now, just return the buffer
                        return (varSSA, varType)
                    | "Pointer" ->
                        // Return the buffer pointer
                        return (varSSA, varType)
                    | "Length" ->
                        // Return a constant for now
                        let! lengthSSA = nextSSA "len"
                        do! emitLine (sprintf "%s = arith.constant 256 : i32" lengthSSA)
                        return (lengthSSA, MLIRTypes.i32)
                    | _ ->
                        return! failHard "Method Access" 
                            (sprintf "Unknown method '%s' on '%s'" methodName varName)
                | None ->
                    // Not a local variable, try as qualified name
                    let name = String.concat "." parts
                    return! generateQualifiedIdentifier name
            | _ ->
                // Regular qualified identifier
                let name = String.concat "." parts
                return! generateQualifiedIdentifier name
            
        | SynExpr.DotGet(targetExpr, _, SynLongIdent(ids, _, _), _) ->
            return! generateFieldAccess targetExpr ids
            
        | SynExpr.Tuple(_, exprs, _, _) ->
            return! generateTuple exprs

        // ADD: While loops
        | SynExpr.While(_, whileExpr, doExpr, _) ->
            return! generateWhile whileExpr doExpr
            
        // ADD: Array/List literals
        | SynExpr.ArrayOrList(isArray, elements, _) ->
            return! generateArrayOrList isArray elements
            
        // ADD: Array indexing
        | SynExpr.DotIndexedGet(expr, indexExpr, _, _) ->
            return! generateArrayIndex expr indexExpr
            
        // ADD: Array set
        | SynExpr.DotIndexedSet(expr, indexExpr, valueExpr, _, _, _) ->
            return! generateArraySet expr indexExpr valueExpr
            
        // ADD: Assignment
        | SynExpr.LongIdentSet(SynLongIdent(ids, _, _), expr, _) ->
            return! generateAssignment ids expr
            
        // ADD: Mutable binding
        | SynExpr.LetOrUseBang(_, _, _, pat, rhsExpr, andBangs, body, range, _) ->
            return! generateLetBang pat rhsExpr body
            
        // ADD: Do expressions
        | SynExpr.Do(expr, _) ->
            return! generateExpression expr
            
        // ADD: Computation expressions
        | SynExpr.YieldOrReturn(_, expr, _, _) ->
            return! generateExpression expr
            
        // ADD: Record construction
        | SynExpr.Record(_, _, fields, _) ->
            return! generateRecord fields
            
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
        // First check if it's a built-in function
        if Map.containsKey ident.idText builtInFunctions then
            // Return a function reference placeholder
            let (paramTypes, retType) = builtInFunctions.[ident.idText]
            let! funcRef = nextSSA (sprintf "%s_ref" ident.idText)
            return (funcRef, MLIRTypes.func paramTypes retType)
        else
            // Check local variables
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

and generateFor (loopVar: Ident) (startExpr: SynExpr) (endExpr: SynExpr) (isUp: bool) (body: SynExpr) : MLIRBuilder<string * MLIRType> =
    mlir {
        let! (startSSA, _) = generateExpression startExpr
        let! (endSSA, _) = generateExpression endExpr
        
        let stepValue = if isUp then "1" else "-1"
        let! stepSSA = nextSSA "step"
        do! emitLine (sprintf "%s = arith.constant %s : i32" stepSSA stepValue)
        
        do! emitLine (sprintf "scf.for %%iv = %s to %s step %s {" startSSA endSSA stepSSA)
        do! indent (mlir {
            do! bindLocal loopVar.idText "%iv" MLIRTypes.i32
            let! _ = generateExpression body
            do! emitLine "scf.yield"
        })
        do! emitLine "}"
        
        let! unitSSA = nextSSA "unit"
        do! emitLine (sprintf "%s = llvm.mlir.constant() : !llvm.void" unitSSA)
        return (unitSSA, MLIRTypes.void_)
    }

and generateWhile (condExpr: SynExpr) (bodyExpr: SynExpr) : MLIRBuilder<string * MLIRType> =
    mlir {
        do! emitLine "scf.while : () -> () {"
        do! indent (mlir {
            let! (condSSA, _) = generateExpression condExpr
            do! emitLine (sprintf "scf.condition(%s)" condSSA)
        })
        do! emitLine "} do {"
        do! indent (mlir {
            let! _ = generateExpression bodyExpr
            do! emitLine "scf.yield"
        })
        do! emitLine "}"
        
        let! unitSSA = nextSSA "unit"
        return (unitSSA, MLIRTypes.void_)
    }

and generateArrayOrList (isArray: bool) (elements: SynExpr list) : MLIRBuilder<string * MLIRType> =
    mlir {
        let! arraySSA = nextSSA "array"
        let size = List.length elements
        
        // Allocate array
        do! emitLine (sprintf "%s = memref.alloc() : memref<%dxi32>" arraySSA size)
        
        // Initialize elements
        for i, elem in List.indexed elements do
            let! (valueSSA, _) = generateExpression elem
            let! indexSSA = nextSSA "idx"
            do! emitLine (sprintf "%s = arith.constant %d : index" indexSSA i)
            do! emitLine (sprintf "memref.store %s, %s[%s] : memref<%dxi32>" valueSSA arraySSA indexSSA size)
        
        return (arraySSA, MLIRTypes.memref MLIRTypes.i32)
    }

and generateArrayIndex (arrayExpr: SynExpr) (indexExpr: SynExpr) : MLIRBuilder<string * MLIRType> =
    mlir {
        let! (arraySSA, arrayType) = generateExpression arrayExpr
        let! (indexSSA, _) = generateExpression indexExpr
        let! resultSSA = nextSSA "elem"
        do! emitLine (sprintf "%s = memref.load %s[%s] : %s" resultSSA arraySSA indexSSA (mlirTypeToString arrayType))
        return (resultSSA, MLIRTypes.i32)
    }

and generateArraySet (arrayExpr: SynExpr) (indexExpr: SynExpr) (valueExpr: SynExpr) : MLIRBuilder<string * MLIRType> =
    mlir {
        let! (arraySSA, arrayType) = generateExpression arrayExpr
        let! (valueSSA, _) = generateExpression valueExpr
        let! (indexSSA, _) = generateExpression indexExpr
        do! emitLine (sprintf "memref.store %s, %s[%s] : %s" valueSSA arraySSA indexSSA (mlirTypeToString arrayType))
        return (valueSSA, MLIRTypes.i32)
    }

and generateAssignment (ids: Ident list) (expr: SynExpr) : MLIRBuilder<string * MLIRType> =
    mlir {
        let varName = ids |> List.map (fun id -> id.idText) |> String.concat "."
        let! (valueSSA, valueType) = generateExpression expr
        
        // For mutable variables, we need to update the binding
        let! maybeVar = lookupLocal varName
        match maybeVar with
        | Some (varSSA, varType) ->
            // In MLIR, we'd typically use memref.store for mutable refs
            // For now, just update the binding
            do! bindLocal varName valueSSA valueType
            return (valueSSA, valueType)
        | None ->
            return! failHard "Assignment" (sprintf "Unbound mutable variable '%s'" varName)
    }

and generateLetBang (pat: SynPat) (rhsExpr: SynExpr) (body: SynExpr) : MLIRBuilder<string * MLIRType> =
    mlir {
        // Simplified computation expression handling
        let! (rhsSSA, rhsType) = generateExpression rhsExpr
        match pat with
        | SynPat.Named(SynIdent(ident, _), _, _, _) ->
            do! bindLocal ident.idText rhsSSA rhsType
        | _ ->
            return! failHard "Let Bang" "Complex patterns in let! not supported"
        return! generateExpression body
    }

and generateRecord (fields: SynExprRecordField list) : MLIRBuilder<string * MLIRType> =
    mlir {
        // Simplified record handling - extract field info from SynExprRecordField
        let! recordSSA = nextSSA "record"
        do! emitLine (sprintf "%s = llvm.mlir.undef : !llvm.struct<()>  // Record placeholder" recordSSA)
        return (recordSSA, MLIRTypes.struct_ [])
    }

and generateApplication (funcExpr: SynExpr) (argExpr: SynExpr) : MLIRBuilder<string * MLIRType> =
    mlir {
        match funcExpr with
        | SynExpr.Ident ident when ident.idText = "op_PipeRight" || ident.idText = "|>" ->
            // Pipe operator - just evaluate the argument
            return! generateExpression argExpr
            
        // Handle arithmetic operators as Ident
        | SynExpr.Ident ident when ident.idText.StartsWith("op_") ->
            return! generateBinaryOperator ident.idText argExpr
            
        | SynExpr.LongIdent(_, SynLongIdent(ids, _, _), _, _) ->
            let parts = ids |> List.map (fun id -> id.idText)
            match parts with
            | ["op_PipeRight"] | ["|>"] ->
                // Pipe operator as LongIdent - just evaluate the argument
                return! generateExpression argExpr

            // Handle arithmetic operators as LongIdent
            | [opName] when opName.StartsWith("op_") ->
                return! generateBinaryOperator opName argExpr
                
            | [varName; methodName] ->
                // This is a method call pattern (e.g., buffer.AsSpan(...))
                let! maybeVar = lookupLocal varName
                match maybeVar with
                | Some (varSSA, varType) ->
                    // Handle known methods
                    match methodName with
                    | "AsSpan" ->
                        // buffer.AsSpan(start, length) - for now just return the buffer
                        match argExpr with
                        | SynExpr.Tuple(_, [startExpr; lengthExpr], _, _) ->
                            let! (startSSA, _) = generateExpression startExpr
                            let! (lengthSSA, _) = generateExpression lengthExpr
                            // For now, just return the buffer itself
                            return (varSSA, varType)
                        | _ ->
                            // AsSpan() with no arguments
                            return (varSSA, varType)
                    | _ ->
                        return! failHard "Method Call" 
                            (sprintf "Unknown method '%s' on variable '%s'" methodName varName)
                | None ->
                    // Not a local variable, try as a regular function
                    let funcName = String.concat "." parts
                    let! symbol = resolveSymbol funcName
                    let! (argSSA, argType) = generateExpression argExpr
                    return! generateSymbolCall symbol [(argSSA, argType)]
            | _ ->
                // Regular qualified function name
                let funcName = String.concat "." parts
                let! symbol = resolveSymbol funcName
                let! (argSSA, argType) = generateExpression argExpr
                return! generateSymbolCall symbol [(argSSA, argType)]
            
        | SynExpr.App(_, _, innerFunc, innerArg, _) ->
            // Handle nested applications (like piped expressions)
            match innerFunc with
            | SynExpr.LongIdent(_, SynLongIdent([pipeId], _, _), _, _) when pipeId.idText = "op_PipeRight" ->
                // This is a pipe application
                match argExpr with
                | SynExpr.App(_, _, func2, arg2, _) ->
                    // Pipeline: innerArg |> func2 arg2
                    let! (innerResult, innerType) = generateExpression innerArg
                    
                    // Now apply func2 to both innerResult and arg2
                    match func2 with
                    | SynExpr.LongIdent(_, SynLongIdent(ids, _, _), _, _) ->
                        let funcName = ids |> List.map (fun id -> id.idText) |> String.concat "."
                        let! symbol = resolveSymbol funcName
                        let! (arg2SSA, arg2Type) = generateExpression arg2
                        return! generateSymbolCall symbol [(arg2SSA, arg2Type); (innerResult, innerType)]
                    | SynExpr.Ident ident ->
                        let! symbol = resolveSymbol ident.idText
                        let! (arg2SSA, arg2Type) = generateExpression arg2
                        return! generateSymbolCall symbol [(arg2SSA, arg2Type); (innerResult, innerType)]
                    | _ ->
                        return! failHard "Pipeline" "Complex function in pipeline"
                | _ ->
                    // Simple pipeline: innerArg |> func
                    let! (innerResult, innerType) = generateExpression innerArg
                    match argExpr with
                    | SynExpr.Ident ident ->
                        let! symbol = resolveSymbol ident.idText
                        return! generateSymbolCall symbol [(innerResult, innerType)]
                    | _ ->
                        return! failHard "Pipeline" "Complex expression in pipeline"
                        
            // Handle arithmetic operators in nested applications
            | SynExpr.LongIdent(_, SynLongIdent([opId], _, _), _, _) when opId.idText.StartsWith("op_") ->
                // This is a binary operator: App(App(op, left), right)
                // innerArg is the left operand, argExpr is the right operand
                let! (leftSSA, leftType) = generateExpression innerArg
                let! (rightSSA, rightType) = generateExpression argExpr
                let! resultSSA = nextSSA "binop"
                
                // Generate the appropriate MLIR operation
                match opId.idText with
                | "op_Addition" | "op_Plus" ->
                    do! emitLine (sprintf "%s = arith.addi %s, %s : i32" resultSSA leftSSA rightSSA)
                | "op_Subtraction" | "op_Minus" ->
                    do! emitLine (sprintf "%s = arith.subi %s, %s : i32" resultSSA leftSSA rightSSA)
                | "op_Multiply" | "op_Star" ->
                    do! emitLine (sprintf "%s = arith.muli %s, %s : i32" resultSSA leftSSA rightSSA)
                | "op_Division" | "op_Divide" ->
                    do! emitLine (sprintf "%s = arith.divsi %s, %s : i32" resultSSA leftSSA rightSSA)
                | "op_Modulus" | "op_Percent" ->
                    do! emitLine (sprintf "%s = arith.remsi %s, %s : i32" resultSSA leftSSA rightSSA)
                | "op_LessThan" ->
                    do! emitLine (sprintf "%s = arith.cmpi slt, %s, %s : i32" resultSSA leftSSA rightSSA)
                | "op_LessThanOrEqual" ->
                    do! emitLine (sprintf "%s = arith.cmpi sle, %s, %s : i32" resultSSA leftSSA rightSSA)
                | "op_GreaterThan" ->
                    do! emitLine (sprintf "%s = arith.cmpi sgt, %s, %s : i32" resultSSA leftSSA rightSSA)
                | "op_GreaterThanOrEqual" ->
                    do! emitLine (sprintf "%s = arith.cmpi sge, %s, %s : i32" resultSSA leftSSA rightSSA)
                | "op_Equality" ->
                    do! emitLine (sprintf "%s = arith.cmpi eq, %s, %s : i32" resultSSA leftSSA rightSSA)
                | "op_Inequality" ->
                    do! emitLine (sprintf "%s = arith.cmpi ne, %s, %s : i32" resultSSA leftSSA rightSSA)
                | _ ->
                    return! failHard "Binary Operator" (sprintf "Unsupported operator: %s" opId.idText)
                
                return (resultSSA, MLIRTypes.i32)
                
            | SynExpr.Ident pipeIdent when pipeIdent.idText = "op_PipeRight" ->
                // This is a pipe application with Ident
                match argExpr with
                | SynExpr.App(_, _, func2, arg2, _) ->
                    // Pipeline: innerArg |> func2 arg2
                    let! (innerResult, innerType) = generateExpression innerArg
                    
                    // Now apply func2 to both innerResult and arg2
                    match func2 with
                    | SynExpr.LongIdent(_, SynLongIdent(ids, _, _), _, _) ->
                        let funcName = ids |> List.map (fun id -> id.idText) |> String.concat "."
                        let! symbol = resolveSymbol funcName
                        let! (arg2SSA, arg2Type) = generateExpression arg2
                        return! generateSymbolCall symbol [(arg2SSA, arg2Type); (innerResult, innerType)]
                    | SynExpr.Ident ident ->
                        let! symbol = resolveSymbol ident.idText
                        let! (arg2SSA, arg2Type) = generateExpression arg2
                        return! generateSymbolCall symbol [(arg2SSA, arg2Type); (innerResult, innerType)]
                    | _ ->
                        return! failHard "Pipeline" "Complex function in pipeline"
                | _ ->
                    // Simple pipeline: innerArg |> func
                    let! (innerResult, innerType) = generateExpression innerArg
                    match argExpr with
                    | SynExpr.Ident ident ->
                        let! symbol = resolveSymbol ident.idText
                        return! generateSymbolCall symbol [(innerResult, innerType)]
                    | _ ->
                        return! failHard "Pipeline" "Complex expression in pipeline"
                        
            // Handle arithmetic operators as Ident in nested applications
            | SynExpr.Ident opIdent when opIdent.idText.StartsWith("op_") ->
                // This is a binary operator: App(App(op, left), right)
                let! (leftSSA, leftType) = generateExpression innerArg
                let! (rightSSA, rightType) = generateExpression argExpr
                let! resultSSA = nextSSA "binop"
                
                match opIdent.idText with
                | "op_Addition" | "op_Plus" ->
                    do! emitLine (sprintf "%s = arith.addi %s, %s : i32" resultSSA leftSSA rightSSA)
                | "op_Subtraction" | "op_Minus" ->
                    do! emitLine (sprintf "%s = arith.subi %s, %s : i32" resultSSA leftSSA rightSSA)
                | "op_Multiply" | "op_Star" ->
                    do! emitLine (sprintf "%s = arith.muli %s, %s : i32" resultSSA leftSSA rightSSA)
                | "op_Division" | "op_Divide" ->
                    do! emitLine (sprintf "%s = arith.divsi %s, %s : i32" resultSSA leftSSA rightSSA)
                | "op_Modulus" | "op_Percent" ->
                    do! emitLine (sprintf "%s = arith.remsi %s, %s : i32" resultSSA leftSSA rightSSA)
                | "op_LessThan" ->
                    do! emitLine (sprintf "%s = arith.cmpi slt, %s, %s : i32" resultSSA leftSSA rightSSA)
                | "op_LessThanOrEqual" ->
                    do! emitLine (sprintf "%s = arith.cmpi sle, %s, %s : i32" resultSSA leftSSA rightSSA)
                | "op_GreaterThan" ->
                    do! emitLine (sprintf "%s = arith.cmpi sgt, %s, %s : i32" resultSSA leftSSA rightSSA)
                | "op_GreaterThanOrEqual" ->
                    do! emitLine (sprintf "%s = arith.cmpi sge, %s, %s : i32" resultSSA leftSSA rightSSA)
                | "op_Equality" ->
                    do! emitLine (sprintf "%s = arith.cmpi eq, %s, %s : i32" resultSSA leftSSA rightSSA)
                | "op_Inequality" ->
                    do! emitLine (sprintf "%s = arith.cmpi ne, %s, %s : i32" resultSSA leftSSA rightSSA)
                | _ ->
                    return! failHard "Binary Operator" (sprintf "Unsupported operator: %s" opIdent.idText)
                
                return (resultSSA, MLIRTypes.i32)
                
            | _ ->
                // Not a pipe, handle as regular application
                let! (funcSSA, funcType) = generateExpression innerFunc
                let! (argSSA, argType) = generateExpression argExpr
                return! failHard "Nested Application" "Nested function applications not yet supported"
            
        | SynExpr.Ident ident ->
            // Check if it's a built-in function
            if Map.containsKey ident.idText builtInFunctions then
                let! (argSSA, argType) = generateExpression argExpr
                return! generateBuiltInCall ident.idText [(argSSA, argType)]
            else
                // Regular function call
                let! symbol = resolveSymbol ident.idText
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
                (sprintf "Complex function expressions not yet supported: %A" funcExpr)
    }

and generateBinaryOperator (opName: string) (argExpr: SynExpr) : MLIRBuilder<string * MLIRType> =
    mlir {
        // For binary operators, the argument is actually the right operand
        // and we need to look for the left operand in a nested App
        match argExpr with
        | SynExpr.App(_, _, leftExpr, rightExpr, _) ->
            let! (leftSSA, leftType) = generateExpression leftExpr
            let! (rightSSA, rightType) = generateExpression rightExpr
            let! resultSSA = nextSSA "binop"
            
            // Generate the appropriate MLIR operation
            match opName with
            | "op_Addition" | "op_Plus" ->
                do! emitLine (sprintf "%s = arith.addi %s, %s : i32" resultSSA leftSSA rightSSA)
            | "op_Subtraction" | "op_Minus" ->
                do! emitLine (sprintf "%s = arith.subi %s, %s : i32" resultSSA leftSSA rightSSA)
            | "op_Multiply" | "op_Star" ->
                do! emitLine (sprintf "%s = arith.muli %s, %s : i32" resultSSA leftSSA rightSSA)
            | "op_Division" | "op_Divide" ->
                do! emitLine (sprintf "%s = arith.divsi %s, %s : i32" resultSSA leftSSA rightSSA)
            | "op_Modulus" | "op_Percent" ->
                do! emitLine (sprintf "%s = arith.remsi %s, %s : i32" resultSSA leftSSA rightSSA)
            | "op_LessThan" ->
                do! emitLine (sprintf "%s = arith.cmpi slt, %s, %s : i32" resultSSA leftSSA rightSSA)
            | "op_LessThanOrEqual" ->
                do! emitLine (sprintf "%s = arith.cmpi sle, %s, %s : i32" resultSSA leftSSA rightSSA)
            | "op_GreaterThan" ->
                do! emitLine (sprintf "%s = arith.cmpi sgt, %s, %s : i32" resultSSA leftSSA rightSSA)
            | "op_GreaterThanOrEqual" ->
                do! emitLine (sprintf "%s = arith.cmpi sge, %s, %s : i32" resultSSA leftSSA rightSSA)
            | "op_Equality" ->
                do! emitLine (sprintf "%s = arith.cmpi eq, %s, %s : i32" resultSSA leftSSA rightSSA)
            | "op_Inequality" ->
                do! emitLine (sprintf "%s = arith.cmpi ne, %s, %s : i32" resultSSA leftSSA rightSSA)
            | _ ->
                return! failHard "Binary Operator" (sprintf "Unsupported operator: %s" opName)
            
            return (resultSSA, MLIRTypes.i32)
        | _ ->
            // Not a binary operator application
            return! failHard "Binary Operator" 
                (sprintf "Expected binary operator application for %s" opName)
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
        
        // Handle known methods
        match fieldName with
        | "AsSpan" ->
            // Just return the buffer itself for now
            return (targetSSA, targetType)
        | "ToString" ->
            // For now, just return the input as-is (spans are already string-like)
            return (targetSSA, targetType)
        | "Pointer" ->
            // Return the buffer pointer
            return (targetSSA, targetType)
        | "Length" ->
            // Return a constant for now
            let! lengthSSA = nextSSA "len"
            do! emitLine (sprintf "%s = arith.constant 256 : i32" lengthSSA)
            return (lengthSSA, MLIRTypes.i32)
        | _ ->
            return! failHard "Field Access" 
                (sprintf "Field or method '%s' not implemented" fieldName)
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

/// Generate MLIR function with parameters
let generateFunction (functionName: string) (attributes: SynAttributes) (pattern: SynPat) (expression: SynExpr) : MLIRBuilder<unit> =
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
            
            // Unwrap parentheses if present
            let rec unwrapParen expr =
                match expr with
                | SynExpr.Paren(inner, _, _, _) -> unwrapParen inner
                | _ -> expr
            
            let unwrappedExpr = unwrapParen expression
            
            // Debug output for stackBuffer
            if functionName = "stackBuffer" then
                do! emit (sprintf "\n    // DEBUG: Pattern type: %A\n" (pattern.GetType().Name))
                do! emit (sprintf "    // DEBUG: Unwrapped expr type: %A\n" (unwrappedExpr.GetType().Name))
            
            // Extract parameters from pattern
            let parameters = 
                match pattern with
                | SynPat.LongIdent(_, _, _, SynArgPats.Pats pats, _, _) ->
                    // Parameters in pattern
                    pats |> List.mapi (fun i pat ->
                        match pat with
                        | SynPat.Named(SynIdent(ident, _), _, _, _) ->
                            Some (ident.idText, sprintf "%%arg%d" i, MLIRTypes.i32)
                        | SynPat.Paren(SynPat.Named(SynIdent(ident, _), _, _, _), _) ->
                            Some (ident.idText, sprintf "%%arg%d" i, MLIRTypes.i32)
                        | SynPat.Typed(SynPat.Named(SynIdent(ident, _), _, _, _), _, _) ->
                            Some (ident.idText, sprintf "%%arg%d" i, MLIRTypes.i32)
                        | SynPat.Paren(SynPat.Typed(SynPat.Named(SynIdent(ident, _), _, _, _), _, _), _) ->
                            Some (ident.idText, sprintf "%%arg%d" i, MLIRTypes.i32)
                        | _ -> None)
                    |> List.choose id
                | SynPat.Named(SynIdent(ident, _), _, _, _) ->
                    // Simple binding - check if expression is a lambda
                    match unwrappedExpr with
                    | SynExpr.Lambda(_, _, args, _, _, _, _) ->
                        let (SynSimplePats.SimplePats(pats, _, _)) = args
                        pats |> List.mapi (fun i pat ->
                            match pat with
                            | SynSimplePat.Id(ident, _, _, _, _, _) ->
                                Some (ident.idText, sprintf "%%arg%d" i, MLIRTypes.i32)
                            | SynSimplePat.Typed(SynSimplePat.Id(ident, _, _, _, _, _), _, _) ->
                                Some (ident.idText, sprintf "%%arg%d" i, MLIRTypes.i32)
                            | _ -> None)
                        |> List.choose id
                    | _ -> []
                | _ -> []
            
            // Additional debug for stackBuffer if no parameters found
            if functionName = "stackBuffer" && parameters.IsEmpty then
                match pattern with
                | SynPat.LongIdent(_, _, _, args, _, _) ->
                    do! emit (sprintf "    // DEBUG: SynArgPats: %A\n" args)
                | _ -> ()
            
            // Build function signature
            let paramList = 
                if parameters.IsEmpty then ""
                else
                    let paramStrs = parameters |> List.map (fun (_, ssa, typ) ->
                        sprintf "%s: %s" ssa (mlirTypeToStringWithLLVM typ))
                    sprintf "(%s)" (String.concat ", " paramStrs)
            
            // Register this function for local resolution
            let paramTypes = parameters |> List.map (fun (_, _, typ) -> typ)
            let funcType = MLIRTypes.func paramTypes MLIRTypes.i32
            do! updateState (fun s -> { s with LocalFunctions = Map.add functionName funcType s.LocalFunctions })
            
            do! emitLine (sprintf "func.func %s%s -> i32 {" mlirName paramList)
            do! indent (
                mlir {
                    // Bind parameters and set up function state in one update
                    do! updateState (fun s -> 
                        let withFunctionState = { s with 
                                                    CurrentFunction = Some qualifiedName
                                                    LocalVars = Map.empty
                                                    SSACounter = 0 }
                        // Bind parameters to the local vars
                        parameters |> List.fold (fun state (name, ssa, typ) ->
                            { state with LocalVars = Map.add name (ssa, typ) state.LocalVars }
                        ) withFunctionState
                    )
                    
                    try
                        // If expression is a lambda, generate its body
                        let bodyExpr = 
                            match unwrappedExpr with
                            | SynExpr.Lambda(_, _, _, body, _, _, _) when not parameters.IsEmpty ->
                                body
                            | _ -> unwrappedExpr
                        
                        let! (resultSSA, resultType) = generateExpression bodyExpr
                        
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

/// Process binding - passes pattern to generateFunction
let processBinding (SynBinding(_, _, _, _, attributes, _, _, pattern, _, expression, _, _, _)) : MLIRBuilder<unit> =
    mlir {
        match pattern with
        | SynPat.Named(SynIdent(ident, _), _, _, _) ->
            do! generateFunction ident.idText attributes pattern expression
        | SynPat.LongIdent(SynLongIdent(ids, _, _), _, _, _, _, _) ->
            let name = ids |> List.map (fun id -> id.idText) |> List.last
            do! generateFunction name attributes pattern expression
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
                
        | SynModuleDecl.Open(_, _) ->
            // Already processed in first pass
            ()
            
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
        
        // First pass: collect all Open declarations and function signatures
        for decl in declarations do
            match decl with
            | SynModuleDecl.Open(target, _) ->
                match target with
                | SynOpenDeclTarget.ModuleOrNamespace(SynLongIdent(ids, _, _), _) ->
                    let namespacePath = ids |> List.map (fun id -> id.idText)
                    do! updateState (fun s -> { s with OpenedNamespaces = namespacePath :: s.OpenedNamespaces })
                | _ -> ()
                
            | SynModuleDecl.Let(_, bindings, _) ->
                // Register function names before generating bodies
                for binding in bindings do
                    let (SynBinding(_, _, _, _, _, _, _, pattern, _, _, _, _, _)) = binding
                    match pattern with
                    | SynPat.Named(SynIdent(ident, _), _, _, _) ->
                        let funcType = MLIRTypes.func [] MLIRTypes.i32  // Simplified
                        do! updateState (fun s -> { s with LocalFunctions = Map.add ident.idText funcType s.LocalFunctions })
                    | SynPat.LongIdent(SynLongIdent(ids, _, _), _, _, _, _, _) ->
                        let name = ids |> List.map (fun id -> id.idText) |> List.last
                        let funcType = MLIRTypes.func [] MLIRTypes.i32  // Simplified
                        do! updateState (fun s -> { s with LocalFunctions = Map.add name funcType s.LocalFunctions })
                    | _ -> ()
            | _ -> ()
        
        // Second pass: generate all declarations
        for decl in declarations do
            match decl with
            | SynModuleDecl.Open(_, _) -> () // Already processed
            | _ -> do! processDeclaration decl
    }

/// Process input file
let processInputFile (parsedInput: ParsedInput) : MLIRBuilder<unit> =
    mlir {
        match parsedInput with
        | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
            // First pass: collect all open declarations
            for module' in modules do
                let (SynModuleOrNamespace(longId, _, _, declarations, _, _, _, _, _)) = module'
                let modulePath = longId |> List.map (fun ident -> ident.idText)
                do! updateState (fun s -> { s with CurrentModule = modulePath })
                
                // Process only Open declarations first
                for decl in declarations do
                    match decl with
                    | SynModuleDecl.Open(target, _) ->
                        match target with
                        | SynOpenDeclTarget.ModuleOrNamespace(SynLongIdent(ids, _, _), _) ->
                            let namespacePath = ids |> List.map (fun id -> id.idText)
                            do! updateState (fun s -> { s with OpenedNamespaces = namespacePath :: s.OpenedNamespaces })
                        | _ -> ()
                    | _ -> ()
            
            // Second pass: process all other declarations
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
    
    let mutable capturedState = None
    
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
        LocalFunctions = Map.empty
        CurrentModule = []
        OpenedNamespaces = []
        HasErrors = false
        ExpectedType = None  // Initialize the new field
    }
    
    let builder = mlir {
        do! emitLine (sprintf "module @%s {" programName)
        
        try
            for (_, parsedInput) in reachableInputs do
                do! processInputFile parsedInput
            
            do! emitExternalDeclarations
            
        with
        | MLIRGenerationException(msg, funcOpt) ->
            // Capture state before re-throwing
            let! state = getState
            capturedState <- Some state
            
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
        // Use captured state if available
        let partialMLIR = 
            match capturedState with
            | Some state -> state.Output.ToString()
            | None ->
                // Try to run builder to get partial output
                try
                    let (_, state) = builder initialState
                    state.Output.ToString()
                with _ -> ""
        
        // Create a custom exception that includes the partial MLIR
        let fullMessage = sprintf "%s\n\n===PARTIAL_MLIR_START===\n%s\n===PARTIAL_MLIR_END===" msg partialMLIR
        raise (System.Exception(fullMessage))
    | ex ->
        // For other exceptions, try to get partial output
        let partialMLIR = 
            try
                let (_, state) = builder initialState
                state.Output.ToString()
            with _ -> ""
            
        if not (System.String.IsNullOrWhiteSpace(partialMLIR)) then
            let fullMessage = sprintf "%s\n\n===PARTIAL_MLIR_START===\n%s\n===PARTIAL_MLIR_END===" ex.Message partialMLIR
            raise (System.Exception(fullMessage))
        else
            reraise()