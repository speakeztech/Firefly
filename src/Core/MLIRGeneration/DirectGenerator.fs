module Core.MLIRGeneration.DirectGenerator

open System.Text
open FSharp.Compiler.Syntax
open FSharp.Compiler.Symbols
open Core.MLIRGeneration.TypeMapping
open Core.MLIRGeneration.TypeSystem
open Core.MLIRGeneration.Dialect
open Core.XParsec.Foundation

// Explicitly qualify to avoid ambiguity
type ResolvedSymbol = Dabbit.Bindings.SymbolRegistry.ResolvedSymbol
type SymbolRegistry = Dabbit.Bindings.SymbolRegistry.SymbolRegistry

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
}

/// Generate SSA value name
let generateSSAName prefix state =
    let ssaName = sprintf "%%%s%d" prefix state.SSACounter
    let updatedState = { state with SSACounter = state.SSACounter + 1 }
    (ssaName, updatedState)

/// Emit MLIR operation with proper indentation
let emitLine mlirCode state =
    let indentation = String.replicate state.IndentLevel "  "
    state.Output.AppendLine(indentation + mlirCode) |> ignore
    state

/// Generate MLIR for a complete program with all reachable modules
let rec generateProgram (programName: string) (typeCtx: TypeContext) (symbolRegistry: SymbolRegistry) 
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
    }
    
    // Start MLIR module
    let stateAfterModuleStart = emitLine (sprintf "module @%s {" programName) initialState
    let stateWithIndent = { stateAfterModuleStart with IndentLevel = 1 }
    
    // Process all reachable input files
    let stateAfterProcessingInputs = 
        reachableInputs 
        |> List.fold (fun currentState (filePath, parsedInput) ->
            processInputFile currentState filePath parsedInput
        ) stateWithIndent
    
    // Emit external function declarations before closing module
    let stateAfterExternals = emitExternalFunctionDeclarations stateAfterProcessingInputs
    
    // Close MLIR module
    let finalState = emitLine "}" { stateAfterExternals with IndentLevel = 0 }
    
    finalState.Output.ToString()

/// Process a single parsed input file
and processInputFile state filePath parsedInput =
    match parsedInput with
    | ParsedInput.ImplFile(ParsedImplFileInput(fileName, isScript, qualName, pragmas, directives, modules, isLast, trivia, ids)) ->
        // Process each module in the file
        modules |> List.fold processModuleOrNamespace state
    | ParsedInput.SigFile(_) -> 
        // Skip signature files for now
        state

/// Process a module or namespace
and processModuleOrNamespace state (SynModuleOrNamespace(longId, isRec, kind, declarations, xmlDoc, attrs, access, range, trivia)) =
    let modulePath = longId |> List.map (fun ident -> ident.idText) |> String.concat "."
    let stateWithModulePath = { state with CurrentModulePath = modulePath }
    
    // Process all declarations in the module
    declarations |> List.fold processModuleDeclaration stateWithModulePath

/// Process different types of module declarations
and processModuleDeclaration state = function
    | SynModuleDecl.Let(isRec, bindings, range) ->
        // Process top-level function bindings
        bindings |> List.fold processFunctionBinding state
    
    | SynModuleDecl.NestedModule(componentInfo, isRec, nestedDeclarations, range, trivia, moduleKeyword) ->
        let (SynComponentInfo(_, _, _, longId, _, _, _, _)) = componentInfo
        let nestedModuleName = longId |> List.map (fun ident -> ident.idText) |> String.concat "."
        let fullNestedPath = 
            if state.CurrentModulePath = "" then nestedModuleName 
            else state.CurrentModulePath + "." + nestedModuleName
        
        let stateWithNestedPath = { state with CurrentModulePath = fullNestedPath }
        nestedDeclarations |> List.fold processModuleDeclaration stateWithNestedPath
    
    | SynModuleDecl.Open(target, range) ->
        // Track opened modules for symbol resolution (future enhancement)
        state
    
    | SynModuleDecl.Types(typeDefs, range) ->
        // Skip type definitions for now - will handle in later phases
        state
    
    | _ -> state

/// Process a function binding and generate MLIR function
and processFunctionBinding state binding =
    let (SynBinding(access, kind, isInline, isMutable, attributes, xmlDoc, valData, pattern, returnInfo, expression, range, sp, trivia)) = binding
    
    match pattern with
    | SynPat.Named(SynIdent(identifier, _), _, _, _) ->
        generateFunctionFromBinding state identifier attributes expression
    
    | SynPat.LongIdent(SynLongIdent(identifiers, _, _), _, _, _, _, _) ->
        // Handle qualified function names
        let functionName = identifiers |> List.map (fun ident -> ident.idText) |> List.last
        let nameIdent = Ident(functionName, range)
        generateFunctionFromBinding state nameIdent attributes expression
    
    | _ -> state

/// Generate MLIR function from binding
and generateFunctionFromBinding state functionIdentifier attributes expression =
    let functionName = functionIdentifier.idText
    let qualifiedName = 
        if state.CurrentModulePath = "" then functionName 
        else state.CurrentModulePath + "." + functionName
    
    // Check if this is the entry point
    let isMainFunction = attributes |> List.exists (fun attributeList ->
        attributeList.Attributes |> List.exists (fun attribute ->
            match attribute.TypeName with
            | SynLongIdent([ident], _, _) -> ident.idText = "EntryPoint"
            | _ -> false))
    
    let mlirFunctionName = if isMainFunction then "@main" else "@" + functionName
    
    // Start function declaration
    let stateAfterFunctionStart = emitLine (sprintf "func.func %s() -> i32 {" mlirFunctionName) state
    let stateWithFunctionIndent = { stateAfterFunctionStart with IndentLevel = state.IndentLevel + 1 }
    
    // Generate function body
    let (resultSSA, resultType, stateAfterBody) = generateExpression stateWithFunctionIndent expression
    
    // Add return statement
    let stateAfterReturn = 
        if isMainFunction then
            // Main always returns 0
            let (zeroSSA, stateWithZero) = generateSSAName "zero" stateAfterBody
            let stateAfterZeroConst = emitLine (sprintf "%s = arith.constant 0 : i32" zeroSSA) stateWithZero
            emitLine (sprintf "func.return %s : i32" zeroSSA) stateAfterZeroConst
        else
            emitLine (sprintf "func.return %s : %s" resultSSA (mlirTypeToString resultType)) stateAfterBody
    
    // Close function
    let stateWithOriginalIndent = { stateAfterReturn with IndentLevel = state.IndentLevel }
    emitLine "}" stateWithOriginalIndent

/// Generate MLIR for expressions
and generateExpression state expr =
    match expr with
    | SynExpr.Const(SynConst.Int32 n, _) ->
        generateIntConstant state n
    
    | SynExpr.Const(SynConst.String(text, _, _), _) ->
        generateStringConstant state text
    
    | SynExpr.Const(SynConst.Unit, _) ->
        // Unit type - no operation needed
        let (unitSSA, stateWithUnit) = generateSSAName "unit" state
        (unitSSA, MLIRTypes.void_, stateWithUnit)
    
    | SynExpr.Ident identifier ->
        resolveIdentifier state identifier
    
    | SynExpr.App(_, _, functionExpr, argumentExpr, _) ->
        generateFunctionApplication state functionExpr argumentExpr
    
    | SynExpr.TypeApp(baseExpr, lessRange, typeArgs, commaRanges, greaterRange, typeArgsRange, range) ->
        // For type applications like stackBuffer<byte>, process the base expression
        // Type arguments will be handled in the application
        generateExpression state baseExpr
    
    | SynExpr.LetOrUse(isRec, isUse, bindings, bodyExpr, range, trivia) ->
        generateLetBinding state bindings bodyExpr
    
    | SynExpr.Sequential(debugPoint, isTrueSeq, firstExpr, secondExpr, range, trivia) ->
        generateSequentialExpressions state firstExpr secondExpr
    
    | SynExpr.Match(spMatch, matchExpr, clauses, range, trivia) ->
        generateMatchExpression state matchExpr clauses
    
    | SynExpr.Paren(innerExpr, _, _, _) ->
        generateExpression state innerExpr
    
    | SynExpr.LongIdent(isOptional, SynLongIdent(identifiers, _, _), altNameRefCell, range) ->
        // Handle qualified identifiers like String.format
        let qualifiedName = identifiers |> List.map (fun ident -> ident.idText) |> String.concat "."
        resolveQualifiedIdentifier state qualifiedName
    
    | _ ->
        // Placeholder for unhandled expressions
        let (todoSSA, stateWithTodo) = generateSSAName "unhandled" state
        let exprTypeName = expr.GetType().Name
        let stateAfterComment = emitLine (sprintf "%s = arith.constant 0 : i32 // TODO: %s" todoSSA exprTypeName) stateWithTodo
        (todoSSA, MLIRTypes.i32, stateAfterComment)

/// Generate integer constant
and generateIntConstant state value =
    let (constantSSA, stateWithSSA) = generateSSAName "const" state
    let stateAfterEmit = emitLine (sprintf "%s = arith.constant %d : i32" constantSSA value) stateWithSSA
    (constantSSA, MLIRTypes.i32, stateAfterEmit)

/// Generate string constant
and generateStringConstant state text =
    let (stringSSA, stateWithSSA) = generateSSAName "str" state
    // For now, use a placeholder - real implementation would create global string constants
    let stateAfterEmit = emitLine (sprintf "%s = llvm.mlir.addressof @str_%d : !llvm.ptr<i8>" stringSSA state.SSACounter) stateWithSSA
    (stringSSA, MLIRTypes.memref MLIRTypes.i8, stateAfterEmit)

/// Resolve identifier to SSA value
and resolveIdentifier state identifier =
    match Map.tryFind identifier.idText state.LocalVariables with
    | Some ssaValue -> 
        // Local variable already has an SSA value
        (ssaValue, MLIRTypes.i32, state)  // TODO: track actual types
    | None ->
        // Try to resolve through symbol registry
        match Dabbit.Bindings.SymbolRegistry.PublicInterface.resolveFunctionCall identifier.idText [] "%unused" state.SymbolRegistry with
        | Success (operations, updatedRegistry) ->
            // Track if this requires external functions
            let stateWithRegistry = { state with SymbolRegistry = updatedRegistry }
            let (functionSSA, stateWithSSA) = generateSSAName "func" stateWithRegistry
            
            // Get return type from registry
            let returnType = 
                match Dabbit.Bindings.SymbolRegistry.PublicInterface.getSymbolType identifier.idText state.SymbolRegistry with
                | Some mlirType -> mlirType
                | None -> MLIRTypes.i32  // Default
            
            (functionSSA, returnType, stateWithSSA)
        | CompilerFailure _ ->
            // Unknown identifier
            let (unknownSSA, stateWithSSA) = generateSSAName "unknown" state
            let stateAfterComment = emitLine (sprintf "%s = arith.constant 0 : i32 // unknown: %s" unknownSSA identifier.idText) stateWithSSA
            (unknownSSA, MLIRTypes.i32, stateAfterComment)

/// Resolve qualified identifier
and resolveQualifiedIdentifier state qualifiedName =
    // For now, treat as function reference
    let (funcRefSSA, stateWithSSA) = generateSSAName "funcref" state
    (funcRefSSA, MLIRTypes.i32, stateWithSSA)

/// Generate function application
and generateFunctionApplication state functionExpr argumentExpr =
    match functionExpr with
    | SynExpr.Ident functionIdent ->
        generateKnownFunctionCall state functionIdent.idText argumentExpr
    
    | SynExpr.LongIdent(_, SynLongIdent(ids, _, _), _, _) ->
        let functionName = ids |> List.map (fun id -> id.idText) |> List.last
        generateKnownFunctionCall state functionName argumentExpr
    
    | _ ->
        // Generic function application
        let (functionSSA, functionType, stateAfterFunction) = generateExpression state functionExpr
        let (argumentSSA, argumentType, stateAfterArgument) = generateExpression stateAfterFunction argumentExpr
        let (resultSSA, stateWithResult) = generateSSAName "call_result" stateAfterArgument
        let stateAfterCall = emitLine (sprintf "%s = func.call %s(%s) : (%s) -> i32" 
                                        resultSSA functionSSA argumentSSA (mlirTypeToString argumentType)) stateWithResult
        (resultSSA, MLIRTypes.i32, stateAfterCall)

/// Generate calls to known functions
and generateKnownFunctionCall state functionName argumentExpr =
    match functionName with
    | "stackBuffer" ->
        generateStackBufferAllocation state argumentExpr
    
    | "prompt" | "writeLine" ->
        generateConsoleOutput state functionName argumentExpr
    
    | "readInto" ->
        generateConsoleInput state argumentExpr
    
    | "format" ->
        generateStringFormat state argumentExpr
    
    | "spanToString" ->
        generateSpanToString state argumentExpr
    
    | _ ->
        // Unknown function - generate generic call
        let (argSSA, argType, stateAfterArg) = generateExpression state argumentExpr
        let (resultSSA, stateWithResult) = generateSSAName "call" stateAfterArg
        let stateAfterCall = emitLine (sprintf "%s = func.call @%s(%s) : (%s) -> i32" 
                                        resultSSA functionName argSSA (mlirTypeToString argType)) stateWithResult
        (resultSSA, MLIRTypes.i32, stateAfterCall)

/// Generate stack buffer allocation
and generateStackBufferAllocation state sizeExpr =
    match sizeExpr with
    | SynExpr.Const(SynConst.Int32 size, _) ->
        let (bufferSSA, stateWithSSA) = generateSSAName "buffer" state
        let stateAfterAlloc = emitLine (sprintf "%s = memref.alloca() : memref<%dxi8>" bufferSSA size) stateWithSSA
        (bufferSSA, MLIRTypes.memref MLIRTypes.i8, stateAfterAlloc)
    | _ ->
        // Dynamic size - need to generate size calculation first
        let (sizeSSA, sizeType, stateAfterSize) = generateExpression state sizeExpr
        let (bufferSSA, stateWithBuffer) = generateSSAName "buffer" stateAfterSize
        let stateAfterAlloc = emitLine (sprintf "%s = memref.alloca(%s) : memref<?xi8>" bufferSSA sizeSSA) stateWithBuffer
        (bufferSSA, MLIRTypes.memref MLIRTypes.i8, stateAfterAlloc)

/// Generate console output operations
and generateConsoleOutput state functionName argumentExpr =
    let (argSSA, argType, stateAfterArg) = generateExpression state argumentExpr
    let stateWithExternal = { stateAfterArg with RequiredExternals = Set.add "printf" stateAfterArg.RequiredExternals }
    let (resultSSA, stateWithResult) = generateSSAName "print_result" stateWithExternal
    let stateAfterCall = emitLine (sprintf "%s = func.call @printf(%s) : (!llvm.ptr<i8>) -> i32" resultSSA argSSA) stateWithResult
    (resultSSA, MLIRTypes.i32, stateAfterCall)

/// Generate console input operations
and generateConsoleInput state bufferExpr =
    let (bufferSSA, bufferType, stateAfterBuffer) = generateExpression state bufferExpr
    let stateWithExternal = { stateAfterBuffer with RequiredExternals = Set.add "fgets" stateAfterBuffer.RequiredExternals }
    let (sizeSSA, stateWithSize) = generateSSAName "buffer_size" stateWithExternal
    let stateAfterSize = emitLine (sprintf "%s = arith.constant 256 : i32" sizeSSA) stateWithSize
    let (stdinSSA, stateWithStdin) = generateSSAName "stdin" stateAfterSize
    let stateAfterStdin = emitLine (sprintf "%s = llvm.mlir.addressof @stdin : !llvm.ptr<i8>" stdinSSA) stateWithStdin
    let (resultSSA, stateWithResult) = generateSSAName "read_result" stateAfterStdin
    let stateAfterCall = emitLine (sprintf "%s = func.call @fgets(%s, %s, %s) : (!llvm.ptr<i8>, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>" 
                                    resultSSA bufferSSA sizeSSA stdinSSA) stateWithResult
    (resultSSA, MLIRTypes.memref MLIRTypes.i8, stateAfterCall)

/// Generate string format operation
and generateStringFormat state argumentExpr =
    // For now, simplified handling
    let (argSSA, argType, stateAfterArg) = generateExpression state argumentExpr
    (argSSA, MLIRTypes.memref MLIRTypes.i8, stateAfterArg)

/// Generate span to string conversion
and generateSpanToString state spanExpr =
    let (spanSSA, spanType, stateAfterSpan) = generateExpression state spanExpr
    // For now, just return the span as-is
    (spanSSA, MLIRTypes.memref MLIRTypes.i8, stateAfterSpan)

/// Generate let binding
and generateLetBinding state bindings bodyExpr =
    let stateAfterBindings = bindings |> List.fold processLetBinding state
    generateExpression stateAfterBindings bodyExpr

/// Process a single let binding
and processLetBinding state binding =
    let (SynBinding(access, kind, isInline, isMutable, attrs, xmlDoc, valData, pattern, returnInfo, expr, range, sp, trivia)) = binding
    match pattern with
    | SynPat.Named(SynIdent(varIdent, _), _, _, _) ->
        let (valueSSA, valueType, stateAfterValue) = generateExpression state expr
        { stateAfterValue with LocalVariables = Map.add varIdent.idText valueSSA stateAfterValue.LocalVariables }
    | _ -> state

/// Generate sequential expressions
and generateSequentialExpressions state firstExpr secondExpr =
    let (_, _, stateAfterFirst) = generateExpression state firstExpr
    generateExpression stateAfterFirst secondExpr

/// Generate match expression
and generateMatchExpression state matchExpr clauses =
    let (matchValueSSA, matchType, stateAfterMatch) = generateExpression state matchExpr
    
    // For HelloWorld, we have Ok/Error pattern
    match clauses with
    | [okClause; errorClause] ->
        // Generate simplified branch structure for now
        let (resultSSA, stateWithResult) = generateSSAName "match_result" stateAfterMatch
        let stateWithComment = emitLine (sprintf "// Match on %s - simplified for now" matchValueSSA) stateWithResult
        (resultSSA, MLIRTypes.i32, stateWithComment)
    | _ ->
        let (defaultSSA, stateWithDefault) = generateSSAName "match_default" stateAfterMatch
        (defaultSSA, MLIRTypes.i32, stateWithDefault)

/// Emit external function declarations
and emitExternalFunctionDeclarations state =
    let emitSingleExternal currentState functionName =
        match functionName with
        | "printf" -> emitLine "func.func private @printf(!llvm.ptr<i8>, ...) -> i32" currentState
        | "fgets" -> emitLine "func.func private @fgets(!llvm.ptr<i8>, i32, !llvm.ptr<i8>) -> !llvm.ptr<i8>" currentState
        | "stdin" -> emitLine "llvm.mlir.global external @stdin() : !llvm.ptr<i8>" currentState
        | _ -> currentState
    
    state.RequiredExternals |> Set.fold emitSingleExternal state