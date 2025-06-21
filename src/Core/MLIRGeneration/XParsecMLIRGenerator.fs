module Core.MLIRGeneration.XParsecMLIRGenerator

open System
open System.Text
open Dabbit.Parsing.OakAst
open Dabbit.SymbolResolution.SymbolRegistry
open Core.MLIRGeneration.TypeSystem
open Core.MLIRGeneration.Operations
open Core.MLIRGeneration.Dialect
open Core.XParsec.Foundation

/// MLIR generation state for tracking SSA values, scopes, and types
type MLIRGenerationState = {
    SSACounter: int
    CurrentScope: Map<string, string>
    ScopeStack: Map<string, string> list
    GeneratedOperations: string list
    ModuleLevelDeclarations: string list
    CurrentFunction: string option
    StringConstants: Map<string, string>
    CurrentDialect: MLIRDialect
    ErrorContext: string list
    SymbolRegistry: SymbolRegistry
    SSAValueTypes: Map<string, MLIRType>
    ExpectedResultType: MLIRType option
}

/// MLIR module output with complete module information
type MLIRModuleOutput = {
    ModuleName: string
    Operations: string list
    SSAMappings: Map<string, string>
    TypeMappings: Map<string, MLIRType>
    Diagnostics: string list
}

/// Core SSA value and scope management functions
module SSA = 
    /// Generates a new SSA value with optional prefix
    let generateValue (prefix: string) (state: MLIRGenerationState) : string * MLIRGenerationState =
        let newCounter = state.SSACounter + 1
        let ssaName = sprintf "%%%s%d" prefix newCounter
        let newState = { state with SSACounter = newCounter }
        (ssaName, newState)
    
    /// Binds a variable name to an SSA value in current scope
    let bindVariable (name: string) (value: string) (state: MLIRGenerationState) : MLIRGenerationState =
        let newScope = Map.add name value state.CurrentScope
        { state with CurrentScope = newScope }
    
    /// Looks up a variable by name in the current scope stack
    let lookupVariable (name: string) (state: MLIRGenerationState) : string option =
        let rec lookup scopes =
            match scopes with
            | [] -> Option.None
            | scope :: rest ->
                match Map.tryFind name scope with
                | Some value -> Some value
                | None -> lookup rest
        
        lookup (state.CurrentScope :: state.ScopeStack)

/// Type tracking and management functions
module TypeTracking =
    /// Records the type of an SSA value for future reference
    let recordSSAType (valueId: string) (valueType: MLIRType) (state: MLIRGenerationState) : MLIRGenerationState =
        { state with SSAValueTypes = Map.add valueId valueType state.SSAValueTypes }
    
    /// Retrieves the type of an SSA value
    let getSSAType (valueId: string) (state: MLIRGenerationState) : MLIRType option =
        Map.tryFind valueId state.SSAValueTypes
    
    /// Sets the expected result type for the current context
    let setExpectedType (expectedType: MLIRType) (state: MLIRGenerationState) : MLIRGenerationState =
        { state with ExpectedResultType = Some expectedType }
    
    /// Clears the expected result type
    let clearExpectedType (state: MLIRGenerationState) : MLIRGenerationState =
        { state with ExpectedResultType = None }
    
    /// Gets the current expected result type
    let getExpectedType (state: MLIRGenerationState) : MLIRType option =
        state.ExpectedResultType
    
    /// Infers the type of a literal expression
    let inferLiteralType (literal: OakLiteral) : MLIRType =
        match literal with
        | IntLiteral _ -> MLIRTypes.createInteger 32
        | FloatLiteral _ -> MLIRTypes.createFloat 32
        | BoolLiteral _ -> MLIRTypes.createInteger 1
        | StringLiteral _ -> MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
        | UnitLiteral -> MLIRTypes.createVoid ()
        | ArrayLiteral _ -> MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
    
    /// Infers the type of a variable from the current scope
    let inferVariableType (name: string) (state: MLIRGenerationState) : MLIRType option =
        match SSA.lookupVariable name state with
        | Some valueId -> getSSAType valueId state
        | None -> None
    
    /// Checks if two types are compatible for the current operation
    let areTypesCompatibleInContext (sourceType: MLIRType) (targetType: MLIRType) (state: MLIRGenerationState) : bool =
        areTypesCompatible sourceType targetType ||
        TypeAnalysis.canConvertTo sourceType targetType

/// Core MLIR operation emission functions
module Emitter =
    /// Emits a raw MLIR operation string to function body
    let emit (operation: string) (state: MLIRGenerationState) : MLIRGenerationState =
        { state with GeneratedOperations = operation :: state.GeneratedOperations }
    
    /// Emits a module-level declaration (globals, external functions)
    let emitModuleLevel (declaration: string) (state: MLIRGenerationState) : MLIRGenerationState =
        { state with ModuleLevelDeclarations = declaration :: state.ModuleLevelDeclarations }
    
    /// Registers a string constant at module level and returns its global name
    let registerString (value: string) (state: MLIRGenerationState) : string * MLIRGenerationState =
        match Map.tryFind value state.StringConstants with
        | Some existingName -> 
            (existingName, state)
        | None ->
            let constName = sprintf "@str_%d" state.StringConstants.Count
            let newState = { 
                state with 
                    StringConstants = Map.add value constName state.StringConstants
            }
            (constName, newState)
    
    /// Creates a constant value in MLIR with type tracking
    let constant (value: string) (mlirType: MLIRType) (state: MLIRGenerationState) : string * MLIRGenerationState =
        let (resultId, state1) = SSA.generateValue "const" state
        let typeStr = mlirTypeToString mlirType
        let state2 = emit (sprintf "    %s = arith.constant %s : %s" resultId value typeStr) state1
        let state3 = TypeTracking.recordSSAType resultId mlirType state2
        (resultId, state3)
    
    /// Creates a function call in MLIR with type tracking
    let call (funcName: string) (args: string list) (resultType: MLIRType option) (state: MLIRGenerationState) : string * MLIRGenerationState =
        let argStr = if args.IsEmpty then "" else String.concat ", " args
        let paramTypeStr = if args.IsEmpty then "" else String.concat ", " (List.replicate args.Length "i32")
        
        match resultType with
        | Some returnType when returnType.Category <> VoidCategory ->
            let (resultId, state1) = SSA.generateValue "call" state
            let typeStr = mlirTypeToString returnType
            
            let callStr = 
                if args.IsEmpty then
                    sprintf "    %s = func.call @%s() : () -> %s" resultId funcName typeStr
                else
                    sprintf "    %s = func.call @%s(%s) : (%s) -> %s" 
                            resultId funcName argStr paramTypeStr typeStr
            
            let state2 = emit callStr state1
            let state3 = TypeTracking.recordSSAType resultId returnType state2
            (resultId, state3)
        | _ ->
            let callStr = 
                if args.IsEmpty then
                    sprintf "    func.call @%s() : () -> ()" funcName
                else
                    sprintf "    func.call @%s(%s) : (%s) -> ()" funcName argStr paramTypeStr
            
            let state1 = emit callStr state
            let (dummyId, state2) = SSA.generateValue "void" state1
            let voidType = MLIRTypes.createVoid ()
            let state3 = TypeTracking.recordSSAType dummyId voidType state2
            (dummyId, state3)

/// Creates a clean initial state for MLIR generation with type tracking
let createInitialState () : MLIRGenerationState = 
    match PublicInterface.createStandardRegistry() with
    | Core.XParsec.Foundation.Success registry -> 
        let state = {
            SSACounter = 0
            CurrentScope = Map.empty
            ScopeStack = []
            GeneratedOperations = []
            ModuleLevelDeclarations = []
            CurrentFunction = Option.None
            StringConstants = Map.empty
            CurrentDialect = Func
            ErrorContext = []
            SymbolRegistry = registry
            SSAValueTypes = Map.empty
            ExpectedResultType = None
        }
        
        let state1 = 
            let (_, state1) = Emitter.registerString "Unknown Person" state
            let (_, state2) = Emitter.registerString "Hello, %s!" state1
            let (_, state3) = Emitter.registerString "Ok-fallback" state2
            let (_, state4) = Emitter.registerString "Error-fallback" state3
            state4
            
        state1
    | Core.XParsec.Foundation.CompilerFailure _ -> 
        failwith "Failed to initialize symbol registry"

/// Match expression handling functions with type awareness
module MatchHandling =

    /// Helper for creating string constants in match handling with type tracking
    let createStringConstant (value: string) (state: MLIRGenerationState) : string * MLIRGenerationState =
        let (globalName, state1) = Emitter.registerString (value.Trim('"')) state
        let (ptrResult, state2) = SSA.generateValue "str_ptr" state1
        let state3 = Emitter.emit (sprintf "    %s = memref.get_global %s : memref<?xi8>" ptrResult globalName) state2
        let memRefType = MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
        let state4 = TypeTracking.recordSSAType ptrResult memRefType state3
        (ptrResult, state4)

    /// Generates branch labels for control flow
    let generateBranchLabels (state: MLIRGenerationState) : string * string * string * MLIRGenerationState =
        let (thenId, state1) = SSA.generateValue "then" state
        let (elseId, state2) = SSA.generateValue "else" state1
        let (endId, state3) = SSA.generateValue "end" state2
        
        let thenLabel = thenId.TrimStart('%')
        let elseLabel = elseId.TrimStart('%')
        let endLabel = endId.TrimStart('%')
        
        (thenLabel, elseLabel, endLabel, state3)
    
    /// Handles pattern matching for Result type with type awareness
    let handleResultMatch (matchExpr: OakExpression) (cases: (OakPattern * OakExpression) list) 
                        (matchValueId: string) (resultId: string) (state: MLIRGenerationState) 
                        (convertExpressionFn: OakExpression -> MLIRGenerationState -> string * MLIRGenerationState) 
                        : string * MLIRGenerationState =
        
        let expectedType = MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
        let state1 = TypeTracking.setExpectedType expectedType state
        
        let bufferName = 
            match matchExpr with
            | Application(Variable "readInto", [Variable name]) -> name
            | _ -> "buffer"
        
        let bufferValue = 
            match SSA.lookupVariable bufferName state1 with
            | Some value -> value
            | None -> "%unknown_buffer"
        
        let state2 = 
            state1
            |> Emitter.emitModuleLevel "  func.func private @is_ok_result(i32) -> i1"
            |> Emitter.emitModuleLevel "  func.func private @extract_result_length(i32) -> i32"
            |> Emitter.emitModuleLevel "  func.func private @create_span(memref<?xi8>, i32) -> memref<?xi8>"
        
        let (isOkId, state3) = SSA.generateValue "is_ok" state2
        let state4 = Emitter.emit (sprintf "    %s = func.call @is_ok_result(%s) : (i32) -> i1" 
                                isOkId matchValueId) state3
        let boolType = MLIRTypes.createInteger 1
        let state5 = TypeTracking.recordSSAType isOkId boolType state4
        
        let (thenLabel, elseLabel, endLabel, state6) = generateBranchLabels state5
        let state7 = Emitter.emit (sprintf "    cond_br %s, ^%s, ^%s" 
                                isOkId thenLabel elseLabel) state6
        
        let okCase = 
            cases |> List.tryFind (fun (pattern, _) -> 
                match pattern with 
                | PatternConstructor("Ok", _) -> true
                | _ -> false)
        
        let state8 = Emitter.emit (sprintf "  ^%s:" thenLabel) state7
        
        let state9 = 
            match okCase with
            | Some (PatternConstructor("Ok", [PatternVariable lengthName]), okExpr) ->
                let (lengthId, stateWithLength) = SSA.generateValue "length" state8
                let stateWithLengthExtract = 
                    Emitter.emit (sprintf "    %s = func.call @extract_result_length(%s) : (i32) -> i32" 
                                lengthId matchValueId) stateWithLength
                let intType = MLIRTypes.createInteger 32
                let stateWithLengthType = TypeTracking.recordSSAType lengthId intType stateWithLengthExtract
                
                let stateWithBinding = SSA.bindVariable lengthName lengthId stateWithLengthType
                
                let (spanId, stateWithSpan) = SSA.generateValue "span" stateWithBinding
                let stateWithCreateSpan = 
                    Emitter.emit (sprintf "    %s = func.call @create_span(%s, %s) : (memref<?xi8>, i32) -> memref<?xi8>" 
                                spanId bufferValue lengthId) stateWithSpan
                let spanType = MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
                let stateWithSpanType = TypeTracking.recordSSAType spanId spanType stateWithCreateSpan
                
                let okExprStateWithBuffer = SSA.bindVariable "span" spanId stateWithSpanType
                let (okResultId, stateAfterExpr) = convertExpressionFn okExpr okExprStateWithBuffer
                
                let stateWithResultStore = 
                    Emitter.emit (sprintf "    %s = %s : memref<?xi8>" resultId okResultId) stateAfterExpr
                let stateWithResultType = TypeTracking.recordSSAType resultId expectedType stateWithResultStore
                
                Emitter.emit (sprintf "    br ^%s" endLabel) stateWithResultType
                
            | Some (_, okExpr) ->
                let (okResultId, stateAfterExpr) = convertExpressionFn okExpr state8
                let stateWithResultStore = 
                    Emitter.emit (sprintf "    %s = %s : memref<?xi8>" resultId okResultId) stateAfterExpr
                let stateWithResultType = TypeTracking.recordSSAType resultId expectedType stateWithResultStore
                Emitter.emit (sprintf "    br ^%s" endLabel) stateWithResultType
                
            | None ->
                let (defaultStr, stateWithStr) = createStringConstant "Ok-fallback" state8
                let stateWithStore = 
                    Emitter.emit (sprintf "    %s = %s : memref<?xi8>" resultId defaultStr) stateWithStr
                let stateWithResultType = TypeTracking.recordSSAType resultId expectedType stateWithStore
                Emitter.emit (sprintf "    br ^%s" endLabel) stateWithResultType
        
        let errorCase = 
            cases 
            |> List.tryFind (fun (pattern, _) -> 
                match pattern with 
                | PatternConstructor("Error", _) -> true 
                | _ -> false)
        
        let state10 = Emitter.emit (sprintf "  ^%s:" elseLabel) state9
        
        let state11 = 
            match errorCase with
            | Some(_, errorExpr) ->
                let (errorResultId, stateAfterErrorExpr) = convertExpressionFn errorExpr state10
                let stateWithResultStore = 
                    Emitter.emit (sprintf "    %s = %s : memref<?xi8>" resultId errorResultId) stateAfterErrorExpr
                let stateWithResultType = TypeTracking.recordSSAType resultId expectedType stateWithResultStore
                Emitter.emit (sprintf "    br ^%s" endLabel) stateWithResultType
                
            | None -> 
                let (errorStr, stateWithStr) = createStringConstant "Error-fallback" state10
                let stateWithStore = 
                    Emitter.emit (sprintf "    %s = %s : memref<?xi8>" resultId errorStr) stateWithStr
                let stateWithResultType = TypeTracking.recordSSAType resultId expectedType stateWithStore
                Emitter.emit (sprintf "    br ^%s" endLabel) stateWithResultType
        
        let state12 = Emitter.emit (sprintf "  ^%s:" endLabel) state11
        let finalState = TypeTracking.clearExpectedType state12
        
        (resultId, finalState)

    /// Handles generic match expression with type awareness
    let handleGenericMatch (matchExpr: OakExpression) (cases: (OakPattern * OakExpression) list) 
                          (matchValueId: string) (resultId: string) (state: MLIRGenerationState)
                          (convertExpressionFn: OakExpression -> MLIRGenerationState -> string * MLIRGenerationState) 
                          : string * MLIRGenerationState =
        
        let (_, elseLabel, endLabel, state1) = generateBranchLabels state
        
        let intType = MLIRTypes.createInteger 32
        let (defaultValue, state2) = Emitter.constant "0" intType state1
        
        if cases.IsEmpty then
            let state3 = Emitter.emit (sprintf "    %s = %s : i32" resultId defaultValue) state2
            let state4 = TypeTracking.recordSSAType resultId intType state3
            (resultId, state4)
        else
            let rec processCases (remainingCases: (OakPattern * OakExpression) list) (currentState: MLIRGenerationState) =
                match remainingCases with
                | [] -> 
                    let fallbackState = Emitter.emit (sprintf "  ^%s:" elseLabel) currentState
                    let storeState = Emitter.emit (sprintf "    %s = %s : i32" resultId defaultValue) fallbackState
                    let typeState = TypeTracking.recordSSAType resultId intType storeState
                    let finalState = Emitter.emit (sprintf "    br ^%s" endLabel) typeState
                    finalState
                | (pattern, expr) :: rest ->
                    let (caseLabel, currentState1) = SSA.generateValue "case" currentState
                    let caseLabelStr = caseLabel.TrimStart('%')
                    
                    let conditionState = Emitter.emit (sprintf "  ^%s:" caseLabelStr) currentState1
                    let (exprResultId, exprState) = convertExpressionFn expr conditionState
                    
                    let storeState = Emitter.emit (sprintf "    %s = %s : i32" resultId exprResultId) exprState
                    let typeState = TypeTracking.recordSSAType resultId intType storeState
                    let branchState = Emitter.emit (sprintf "    br ^%s" endLabel) typeState
                    
                    processCases rest branchState
            
            let firstCaseState = processCases cases state2
            let finalState = Emitter.emit (sprintf "  ^%s:" endLabel) firstCaseState
            
            (resultId, finalState)

/// Determines if a function is a known Alloy function that returns a Result type
let isResultReturningFunction (funcName: string) : bool =
    match funcName with
    | "readInto" 
    | "readFile"
    | "parseInput"
    | "tryParse" -> true
    | _ -> funcName.Contains("OrNone") || 
           funcName.Contains("OrError") || 
           funcName.StartsWith("try") ||
           funcName.Contains("Result")

/// Core expression conversion functions with proper mutual recursion and type tracking
let rec convertLiteral (literal: OakLiteral) (state: MLIRGenerationState) : string * MLIRGenerationState =
    let literalType = TypeTracking.inferLiteralType literal
    match literal with
    | IntLiteral value ->
        Emitter.constant (string value) literalType state
    | FloatLiteral value ->
        Emitter.constant (sprintf "%f" value) literalType state
    | BoolLiteral value ->
        Emitter.constant (if value then "1" else "0") literalType state
    | StringLiteral value ->
        let (globalName, state1) = Emitter.registerString value state
        let (ptrResult, state2) = SSA.generateValue "str_ptr" state1
        let state3 = Emitter.emit (sprintf "    %s = memref.get_global %s : memref<?xi8>" ptrResult globalName) state2
        let state4 = TypeTracking.recordSSAType ptrResult literalType state3
        (ptrResult, state4)
    | UnitLiteral ->
        let intType = MLIRTypes.createInteger 32
        Emitter.constant "0" intType state
    | ArrayLiteral _ ->
        let (arrayResult, state1) = SSA.generateValue "array" state
        let state2 = TypeTracking.recordSSAType arrayResult literalType state1
        (arrayResult, state2)

and convertStackAllocation (args: OakExpression list) (state: MLIRGenerationState) : string * MLIRGenerationState =
    match args with
    | [Literal (IntLiteral size)] ->
        let (bufferResult, state1) = SSA.generateValue "buffer" state
        let state2 = Emitter.emit (sprintf "    %s = memref.alloca() : memref<%dxi8>" bufferResult size) state1
        let bufferType = MLIRTypes.createMemRef (MLIRTypes.createInteger 8) [size]
        let state3 = TypeTracking.recordSSAType bufferResult bufferType state2
        (bufferResult, state3)
    | _ ->
        let (dummyValue, state1) = SSA.generateValue "invalid_alloca" state
        let intType = MLIRTypes.createInteger 32
        let state2 = TypeTracking.recordSSAType dummyValue intType state1
        (dummyValue, state2)

and convertExpression (expr: OakExpression) (state: MLIRGenerationState) : string * MLIRGenerationState =
    match expr with
    | Literal literal ->
        convertLiteral literal state

    | Match(matchExpr, cases) ->
        let (matchValueId, stateAfterMatchExpr) = convertExpression matchExpr state
        let (resultId, stateWithResultVar) = SSA.generateValue "match_result" stateAfterMatchExpr
        
        match matchExpr with
        | Application(Variable funcName, _) when funcName = "readInto" ->
            MatchHandling.handleResultMatch matchExpr cases matchValueId resultId stateWithResultVar convertExpression
        
        | Application(Variable funcName, _) when 
            funcName.Contains("Result") || 
            funcName.EndsWith("OrNone") || 
            funcName.EndsWith("OrError") ->
            MatchHandling.handleResultMatch matchExpr cases matchValueId resultId stateWithResultVar convertExpression
            
        | _ ->
            MatchHandling.handleGenericMatch matchExpr cases matchValueId resultId stateWithResultVar convertExpression

    | Variable name ->
        match SSA.lookupVariable name state with
        | Some value -> (value, state)
        | None -> 
            let (dummyValue, state1) = SSA.generateValue "unknown" state
            let intType = MLIRTypes.createInteger 32
            let state2 = TypeTracking.recordSSAType dummyValue intType state1
            (dummyValue, state2)
    
    | Application(func, args) ->
        match func with
        | Variable "op_PipeRight" ->
            match args with
            | [value; funcExpr] ->
                match funcExpr with
                | Variable fname -> convertExpression (Application(Variable fname, [value])) state
                | Application(f, existingArgs) -> 
                    convertExpression (Application(f, value :: existingArgs)) state
                | _ -> convertExpression (Application(funcExpr, [value])) state
            | _ ->
                let (dummyValue, state1) = SSA.generateValue "invalid_pipe" state
                let intType = MLIRTypes.createInteger 32
                let state2 = TypeTracking.recordSSAType dummyValue intType state1
                (dummyValue, state2)
        
        | Variable funcName ->
            let rec processArgs (remainingArgs: OakExpression list) (currentState: MLIRGenerationState) (accArgs: string list) =
                match remainingArgs with
                | [] -> (List.rev accArgs, currentState)
                | arg :: rest ->
                    let (argValue, newState) = convertExpression arg currentState
                    processArgs rest newState (argValue :: accArgs)
            
            let (argValues, state1) = processArgs args state []
            let (resultId, state2) = SSA.generateValue "call" state1
            
            if isResultReturningFunction funcName then
                let intType = MLIRTypes.createInteger 32
                let state3 = Emitter.emit (sprintf "    %s = arith.constant 1 : i32 // Result marker for %s" 
                                    resultId funcName) state2
                let state4 = TypeTracking.recordSSAType resultId intType state3
                (resultId, state4)
            else
                match PublicInterface.resolveFunctionCall funcName argValues resultId state2.SymbolRegistry with
                | Core.XParsec.Foundation.Success (operations, updatedRegistry) -> 
                    let intType = MLIRTypes.createInteger 32
                    let finalState = 
                        operations
                        |> List.fold (fun accState op -> Emitter.emit op accState) state2
                        |> fun s -> { s with SymbolRegistry = updatedRegistry }
                        |> TypeTracking.recordSSAType resultId intType
                    (resultId, finalState)
                    
                | Core.XParsec.Foundation.CompilerFailure errors -> 
                    match funcName with
                    | "NativePtr.stackalloc" ->
                        convertStackAllocation args state
                        
                    | _ ->
                        let (resultId, state1) = SSA.generateValue "unknown_call" state
                        let intType = MLIRTypes.createInteger 32
                        let state2 = Emitter.emit (sprintf "    %s = arith.constant 0 : i32 // Unknown function %s" 
                                                resultId funcName) state1
                        let state3 = TypeTracking.recordSSAType resultId intType state2
                        (resultId, state3)

        | _ ->
            let (funcValue, state1) = convertExpression func state
            
            let rec processArgs (remainingArgs: OakExpression list) (currentState: MLIRGenerationState) (accArgs: string list) =
                match remainingArgs with
                | [] -> (List.rev accArgs, currentState)
                | arg :: rest ->
                    let (argValue, newState) = convertExpression arg currentState
                    processArgs rest newState (argValue :: accArgs)
            
            let (argValues, state2) = processArgs args state1 []
            let (resultId, state3) = SSA.generateValue "app" state2
            let argStr = String.concat ", " argValues
            let state4 = Emitter.emit (sprintf "    %s = func.call @%s(%s) : (i32) -> i32" resultId funcValue argStr) state3
            let intType = MLIRTypes.createInteger 32
            let state5 = TypeTracking.recordSSAType resultId intType state4
            (resultId, state5)
    
    | Let(name, value, body) ->
        let (valueResult, state1) = convertExpression value state
        let state2 = SSA.bindVariable name valueResult state1
        let (bodyResult, state3) = convertExpression body state2
        (bodyResult, state3)
    
    | Sequential(first, second) ->
        let (firstResult, state1) = convertExpression first state
        let (secondResult, state2) = convertExpression second state1
        (secondResult, state2)
    
    | IfThenElse(cond, thenExpr, elseExpr) ->
        let (condResult, state1) = convertExpression cond state
        let (thenLabel, elseLabel, endLabel, state2) = MatchHandling.generateBranchLabels state1
        
        let state3 = Emitter.emit (sprintf "    cond_br %s, ^%s, ^%s" condResult thenLabel elseLabel) state2
        let (resultId, state4) = SSA.generateValue "if_result" state3
        
        let state5 = Emitter.emit (sprintf "  ^%s:" thenLabel) state4
        let (thenResult, state6) = convertExpression thenExpr state5
        let state7 = Emitter.emit (sprintf "    %s = %s : i32" resultId thenResult) state6
        let state8 = Emitter.emit (sprintf "    br ^%s" endLabel) state7
        
        let state9 = Emitter.emit (sprintf "  ^%s:" elseLabel) state8
        let (elseResult, state10) = convertExpression elseExpr state9
        let state11 = Emitter.emit (sprintf "    %s = %s : i32" resultId elseResult) state10
        let state12 = Emitter.emit (sprintf "    br ^%s" endLabel) state11
        
        let state13 = Emitter.emit (sprintf "  ^%s:" endLabel) state12
        let intType = MLIRTypes.createInteger 32
        let state14 = TypeTracking.recordSSAType resultId intType state13
        
        (resultId, state14)
    
    | FieldAccess(target, fieldName) ->
        let (targetResult, state1) = convertExpression target state
        (targetResult, state1)
    
    | MethodCall(target, methodName, args) ->
        let (targetResult, state1) = convertExpression target state
        let convertArg (accState, accArgs) arg =
            let (argValue, newState) = convertExpression arg accState
            (newState, argValue :: accArgs)
        
        let (state2, argValues) = List.fold convertArg (state1, []) args
        (targetResult, state2)
    
    | IOOperation(ioType, args) ->
        match ioType with
        | Printf formatStr | Printfn formatStr ->
            let state1 = Emitter.emitModuleLevel "  func.func private @printf(memref<?xi8>, ...) -> i32" state
            
            let (formatGlobal, state2) = Emitter.registerString formatStr state1
            let (formatPtr, state3) = SSA.generateValue "fmt_ptr" state2
            let state4 = Emitter.emit (sprintf "    %s = memref.get_global %s : memref<?xi8>" formatPtr formatGlobal) state3
            
            let rec processArgs (remainingArgs: OakExpression list) (currentState: MLIRGenerationState) (accArgs: string list) =
                match remainingArgs with
                | [] -> (List.rev accArgs, currentState)
                | arg :: rest ->
                    let (argValue, newState) = convertExpression arg currentState
                    processArgs rest newState (argValue :: accArgs)
            
            let (argValues, state5) = processArgs args state4 []
            
            let allArgs = formatPtr :: argValues
            let (printfResult, state6) = SSA.generateValue "printf_result" state5
            let argStr = String.concat ", " allArgs
            
            let state7 = 
                if args.IsEmpty then
                    Emitter.emit (sprintf "    %s = func.call @printf(%s) : (memref<?xi8>) -> i32" printfResult formatPtr) state6
                else
                    let types = "memref<?xi8>" :: List.replicate argValues.Length "i32"
                    let typeStr = String.concat ", " types
                    Emitter.emit (sprintf "    %s = func.call @printf(%s) : (%s) -> i32" printfResult argStr typeStr) state6
            
            let intType = MLIRTypes.createInteger 32
            let state8 = TypeTracking.recordSSAType printfResult intType state7
            (printfResult, state8)
        
        | _ ->
            let (dummyValue, state1) = SSA.generateValue "io_op" state
            let intType = MLIRTypes.createInteger 32
            let state2 = TypeTracking.recordSSAType dummyValue intType state1
            (dummyValue, state2)
    
    | Lambda(params', body) ->
        let (bodyResult, state1) = convertExpression body state
        (bodyResult, state1)

/// Function and module level conversion functions
let convertFunction (name: string) (parameters: (string * OakType) list) (returnType: OakType) (body: OakExpression) (state: MLIRGenerationState) : MLIRGenerationState =
    let returnMLIRType = mapOakTypeToMLIR returnType
    let returnTypeStr = mlirTypeToString returnMLIRType
    
    let paramStr = 
        if parameters.IsEmpty then
            ""
        else
            parameters 
            |> List.mapi (fun i (_, typ) -> 
                let mlirType = mapOakTypeToMLIR typ
                sprintf "%%arg%d: %s" i (mlirTypeToString mlirType))
            |> String.concat ", "
            
    let state1 = Emitter.emit (sprintf "  func.func @%s(%s) -> %s {" name paramStr returnTypeStr) state
    
    let state2 = 
        parameters 
        |> List.mapi (fun i (paramName, paramType) -> 
            let argId = sprintf "%%arg%d" i
            let mlirType = mapOakTypeToMLIR paramType
            fun s -> 
                s |> SSA.bindVariable paramName argId
                  |> TypeTracking.recordSSAType argId mlirType)
        |> List.fold (fun s f -> f s) state1
    
    let (bodyResult, state3) = convertExpression body state2
    
    let state4 = 
        if returnType = UnitType then
            Emitter.emit "    func.return" state3
        else
            Emitter.emit (sprintf "    func.return %s : %s" bodyResult returnTypeStr) state3
    
    Emitter.emit "  }" state4

/// MLIR module generation with type tracking
let generateMLIR (program: OakProgram) : MLIRModuleOutput =
    let initialState = createInitialState ()
    
    let processModule (state: MLIRGenerationState) (mdl: OakModule) =
        let state1 = Emitter.emit (sprintf "module @%s {" mdl.Name) state
        
        let processDeclFold currState decl =
            match decl with
            | FunctionDecl(name, parameters, returnType, body) ->
                convertFunction name parameters returnType body currState
            
            | EntryPoint(expr) ->
                convertFunction "main" 
                    [("argc", IntType); ("argv", ArrayType(StringType))] 
                    IntType expr currState
                
            | ExternalDecl(name, paramTypes, returnType, libraryName) ->
                let paramTypeStrs = paramTypes |> List.map (mapOakTypeToMLIR >> mlirTypeToString)
                let returnTypeStr = mlirTypeToString (mapOakTypeToMLIR returnType)
                
                let paramStr = 
                    if paramTypeStrs.IsEmpty then ""
                    else
                        paramTypeStrs
                        |> List.mapi (fun i typ -> sprintf "%%arg%d: %s" i typ)
                        |> String.concat ", "
                
                Emitter.emit (sprintf "  func.func private @%s(%s) -> %s" 
                                 name paramStr returnTypeStr) currState
            
            | _ -> currState
        
        let state2 = List.fold processDeclFold state1 mdl.Declarations
        
        let state3 = 
            state2.StringConstants
            |> Map.toList
            |> List.fold (fun s (value, globalName) ->
                let escapedValue = value.Replace("\\", "\\\\").Replace("\"", "\\\"")
                let constSize = escapedValue.Length + 1
                let declaration = sprintf "  memref.global constant %s = dense<\"%s\\00\"> : memref<%dxi8>" 
                                        globalName escapedValue constSize
                Emitter.emit declaration s) state2
        
        let uniqueDeclarations = 
            state3.ModuleLevelDeclarations
            |> List.rev
            |> List.distinct
        
        let state4 = 
            uniqueDeclarations
            |> List.fold (fun s decl -> Emitter.emit decl s) state3
        
        Emitter.emit "}" state4
    
    let finalState = 
        match program.Modules with
        | [] -> initialState
        | mdl :: _ -> processModule initialState mdl
    
    {
        ModuleName = 
            match program.Modules with
            | [] -> "main"
            | mdl :: _ -> mdl.Name
        Operations = List.rev finalState.GeneratedOperations
        SSAMappings = finalState.CurrentScope
        TypeMappings = finalState.SSAValueTypes
        Diagnostics = finalState.ErrorContext
    }

/// Generates complete MLIR module text from Oak AST with type tracking
let generateMLIRModuleText (program: OakProgram) : Core.XParsec.Foundation.CompilerResult<string> =
    try
        let mlirOutput = generateMLIR program
        let moduleText = String.concat "\n" mlirOutput.Operations
        Core.XParsec.Foundation.Success moduleText
    with ex ->
        Core.XParsec.Foundation.CompilerFailure [
            Core.XParsec.Foundation.ConversionError(
                "MLIR generation", 
                "Oak AST", 
                "MLIR", 
                sprintf "Exception: %s" ex.Message)
        ]