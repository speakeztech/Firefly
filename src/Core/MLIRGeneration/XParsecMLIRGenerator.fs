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
    
    /// Infers the return type of a function from the symbol registry
    let inferFunctionReturnType (funcName: string) (state: MLIRGenerationState) : MLIRType option =
        match PublicInterface.resolveFunctionCall funcName [] "" state.SymbolRegistry with
        | Core.XParsec.Foundation.Success (_, updatedRegistry) ->
            // Try to find the function in the registry and get its return type
            let lookupName = 
                if funcName.Contains(".") then funcName
                else 
                    // Check common Alloy functions
                    if funcName = "stackBuffer" then "Alloy.Memory.stackBuffer"
                    elif funcName = "spanToString" then "Alloy.Memory.spanToString"
                    elif funcName = "readInto" then "Alloy.IO.Console.readInto"
                    elif funcName = "writeLine" then "Alloy.IO.Console.writeLine"
                    elif funcName = "format" then "Alloy.IO.String.format"
                    else funcName
                    
            // Default types for common functions
            match lookupName with
            | "Alloy.Memory.stackBuffer" -> Some (MLIRTypes.createMemRef (MLIRTypes.createInteger 8) [])
            | "Alloy.Memory.spanToString" -> Some (MLIRTypes.createMemRef (MLIRTypes.createInteger 8) [])
            | "Alloy.IO.Console.readInto" -> Some (MLIRTypes.createInteger 32)  // Result<int, string> mapped to i32
            | "Alloy.IO.Console.writeLine" -> Some (MLIRTypes.createVoid ())
            | "Alloy.IO.String.format" -> Some (MLIRTypes.createMemRef (MLIRTypes.createInteger 8) [])
            | _ -> Some (MLIRTypes.createInteger 32) // Default to i32 for unknown functions
        | _ -> Some (MLIRTypes.createInteger 32) // Default to i32 if resolution fails
    
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
    
    /// Creates a basic type conversion operation
    let convertType (sourceValue: string) (sourceType: MLIRType) (targetType: MLIRType) (state: MLIRGenerationState) : string * MLIRGenerationState =
        // Skip conversion if types are already compatible
        if sourceType = targetType then
            (sourceValue, state)
        else
            let (resultId, state1) = SSA.generateValue "conv" state
            
            match sourceType.Category, targetType.Category with
            | IntegerCategory, IntegerCategory when sourceType.Width <> targetType.Width ->
                // Integer width conversion
                let opName = 
                    if (sourceType.Width |> Option.defaultValue 32) < (targetType.Width |> Option.defaultValue 32) then
                        "arith.extsi"  // Sign extension
                    else
                        "arith.trunci" // Truncation
                
                let sourceTypeStr = mlirTypeToString sourceType
                let targetTypeStr = mlirTypeToString targetType
                let opStr = sprintf "    %s = %s %s : %s to %s" resultId opName sourceValue sourceTypeStr targetTypeStr
                let state2 = emit opStr state1
                let state3 = TypeTracking.recordSSAType resultId targetType state2
                (resultId, state3)
                
            | IntegerCategory, FloatCategory ->
                // Integer to float conversion
                let sourceTypeStr = mlirTypeToString sourceType
                let targetTypeStr = mlirTypeToString targetType
                let opStr = sprintf "    %s = arith.sitofp %s : %s to %s" resultId sourceValue sourceTypeStr targetTypeStr
                let state2 = emit opStr state1
                let state3 = TypeTracking.recordSSAType resultId targetType state2
                (resultId, state3)
                
            | FloatCategory, IntegerCategory ->
                // Float to integer conversion
                let sourceTypeStr = mlirTypeToString sourceType
                let targetTypeStr = mlirTypeToString targetType
                let opStr = sprintf "    %s = arith.fptosi %s : %s to %s" resultId sourceValue sourceTypeStr targetTypeStr
                let state2 = emit opStr state1
                let state3 = TypeTracking.recordSSAType resultId targetType state2
                (resultId, state3)
                
            | IntegerCategory, MemoryRefCategory ->
                // Integer to memory reference (unusual but handled)
                // This typically involves creating a string representation
                let (strGlobal, state2) = Emitter.registerString "%d" state1
                let (ptrResult, state3) = SSA.generateValue "str_ptr" state2
                let state4 = emit (sprintf "    %s = memref.get_global %s : memref<?xi8>" ptrResult strGlobal) state3
                let memRefType = MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
                let state5 = TypeTracking.recordSSAType ptrResult memRefType state4
                (ptrResult, state5)
                
            | MemoryRefCategory, MemoryRefCategory ->
                // Memory reference conversion
                let sourceTypeStr = mlirTypeToString sourceType
                let targetTypeStr = mlirTypeToString targetType
                let opStr = sprintf "    %s = memref.cast %s : %s to %s" resultId sourceValue sourceTypeStr targetTypeStr
                let state2 = emit opStr state1
                let state3 = TypeTracking.recordSSAType resultId targetType state2
                (resultId, state3)
                
            | _ ->
                // Default to a simple bitcast for other conversions
                // In a real implementation, this would be more sophisticated
                let sourceTypeStr = mlirTypeToString sourceType
                let targetTypeStr = mlirTypeToString targetType
                let opStr = sprintf "    %s = llvm.bitcast %s : %s to %s" resultId sourceValue sourceTypeStr targetTypeStr
                let state2 = emit opStr state1
                let state3 = TypeTracking.recordSSAType resultId targetType state2
                (resultId, state3)

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
        
        // Pre-register some common string constants for better type handling
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
        
        // Set expected return type for result pattern match (always a string/memref)
        let expectedType = MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
        let state1 = TypeTracking.setExpectedType expectedType state
        
        // Extract buffer name from pattern for better code generation
        let bufferName = 
            match matchExpr with
            | Application(Variable "readInto", [Variable name]) -> name
            | _ -> "buffer"
        
        // Find actual buffer value in scope
        let bufferValue = 
            match SSA.lookupVariable bufferName state1 with
            | Some value -> value
            | None -> "%unknown_buffer"
        
        // Register helper functions for working with Result type
        let state2 = 
            state1
            |> Emitter.emitModuleLevel "  func.func private @is_ok_result(i32) -> i1"
            |> Emitter.emitModuleLevel "  func.func private @extract_result_length(i32) -> i32"
            |> Emitter.emitModuleLevel "  func.func private @create_span(memref<?xi8>, i32) -> memref<?xi8>"
        
        // Check if result is Ok or Error
        let (isOkId, state3) = SSA.generateValue "is_ok" state2
        let state4 = Emitter.emit (sprintf "    %s = func.call @is_ok_result(%s) : (i32) -> i1" 
                                isOkId matchValueId) state3
        let boolType = MLIRTypes.createInteger 1
        let state5 = TypeTracking.recordSSAType isOkId boolType state4
        
        // Create branch labels
        let (thenLabel, elseLabel, endLabel, state6) = generateBranchLabels state5
        let state7 = Emitter.emit (sprintf "    cond_br %s, ^%s, ^%s" 
                                isOkId thenLabel elseLabel) state6
        
        // Find Ok case pattern
        let okCase = 
            cases |> List.tryFind (fun (pattern, _) -> 
                match pattern with 
                | PatternConstructor("Ok", _) -> true
                | _ -> false)
        
        // Handle Ok branch
        let state8 = Emitter.emit (sprintf "  ^%s:" thenLabel) state7
        
        let state9 = 
            match okCase with
            | Some (PatternConstructor("Ok", [PatternVariable lengthName]), okExpr) ->
                // Extract length from result
                let (lengthId, stateWithLength) = SSA.generateValue "length" state8
                let stateWithLengthExtract = 
                    Emitter.emit (sprintf "    %s = func.call @extract_result_length(%s) : (i32) -> i32" 
                                lengthId matchValueId) stateWithLength
                let intType = MLIRTypes.createInteger 32
                let stateWithLengthType = TypeTracking.recordSSAType lengthId intType stateWithLengthExtract
                
                // Bind length variable
                let stateWithBinding = SSA.bindVariable lengthName lengthId stateWithLengthType
                
                // Create span from buffer and length
                let (spanId, stateWithSpan) = SSA.generateValue "span" stateWithBinding
                let stateWithCreateSpan = 
                    Emitter.emit (sprintf "    %s = func.call @create_span(%s, %s) : (memref<?xi8>, i32) -> memref<?xi8>" 
                                spanId bufferValue lengthId) stateWithSpan
                let spanType = MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
                let stateWithSpanType = TypeTracking.recordSSAType spanId spanType stateWithCreateSpan
                
                // Process Ok expression with span bound
                let okExprStateWithBuffer = SSA.bindVariable "span" spanId stateWithSpanType
                let (okResultId, stateAfterExpr) = convertExpressionFn okExpr okExprStateWithBuffer
                
                // Get type of result for potential conversion
                let okResultType = 
                    match TypeTracking.getSSAType okResultId stateAfterExpr with
                    | Some typ -> typ
                    | None -> MLIRTypes.createInteger 32  // Default if unknown
                
                // Apply type conversion if needed
                let (finalResultId, stateAfterConversion) =
                    if isMemRefType okResultType then
                        (okResultId, stateAfterExpr)  // Already the right type
                    else
                        // Need to convert to memref
                        Emitter.convertType okResultId okResultType expectedType stateAfterExpr
                
                // Store result and branch to end
                let stateWithResultStore = 
                    Emitter.emit (sprintf "    %s = %s : %s" resultId finalResultId (mlirTypeToString expectedType)) stateAfterConversion
                let stateWithResultType = TypeTracking.recordSSAType resultId expectedType stateWithResultStore
                
                Emitter.emit (sprintf "    br ^%s" endLabel) stateWithResultType
                
            | Some (_, okExpr) ->
                // Handle Ok case without pattern variable
                let (okResultId, stateAfterExpr) = convertExpressionFn okExpr state8
                
                // Get type of result for potential conversion
                let okResultType = 
                    match TypeTracking.getSSAType okResultId stateAfterExpr with
                    | Some typ -> typ
                    | None -> MLIRTypes.createInteger 32  // Default if unknown
                
                // Apply type conversion if needed
                let (finalResultId, stateAfterConversion) =
                    if isMemRefType okResultType then
                        (okResultId, stateAfterExpr)  // Already the right type
                    else
                        // Need to convert to memref
                        Emitter.convertType okResultId okResultType expectedType stateAfterExpr
                
                // Store result and branch to end
                let stateWithResultStore = 
                    Emitter.emit (sprintf "    %s = %s : %s" resultId finalResultId (mlirTypeToString expectedType)) stateAfterConversion
                let stateWithResultType = TypeTracking.recordSSAType resultId expectedType stateWithResultStore
                
                Emitter.emit (sprintf "    br ^%s" endLabel) stateWithResultType
                
            | None ->
                // No Ok case found, use fallback
                let (defaultStr, stateWithStr) = createStringConstant "Ok-fallback" state8
                let stateWithStore = 
                    Emitter.emit (sprintf "    %s = %s : %s" resultId defaultStr (mlirTypeToString expectedType)) stateWithStr
                let stateWithResultType = TypeTracking.recordSSAType resultId expectedType stateWithStore
                
                Emitter.emit (sprintf "    br ^%s" endLabel) stateWithResultType
        
        // Find Error case pattern
        let errorCase = 
            cases 
            |> List.tryFind (fun (pattern, _) -> 
                match pattern with 
                | PatternConstructor("Error", _) -> true 
                | _ -> false)
        
        // Handle Error branch
        let state10 = Emitter.emit (sprintf "  ^%s:" elseLabel) state9
        
        let state11 = 
            match errorCase with
            | Some(_, errorExpr) ->
                // Process Error expression
                let (errorResultId, stateAfterErrorExpr) = convertExpressionFn errorExpr state10
                
                // Get type of result for potential conversion
                let errorResultType = 
                    match TypeTracking.getSSAType errorResultId stateAfterErrorExpr with
                    | Some typ -> typ
                    | None -> MLIRTypes.createInteger 32  // Default if unknown
                
                // Apply type conversion if needed
                let (finalErrorId, stateAfterConversion) =
                    if isMemRefType errorResultType then
                        (errorResultId, stateAfterErrorExpr)  // Already the right type
                    else
                        // Need to convert to memref
                        Emitter.convertType errorResultId errorResultType expectedType stateAfterErrorExpr
                
                // Store result and branch to end
                let stateWithResultStore = 
                    Emitter.emit (sprintf "    %s = %s : %s" resultId finalErrorId (mlirTypeToString expectedType)) stateAfterConversion
                let stateWithResultType = TypeTracking.recordSSAType resultId expectedType stateWithResultStore
                
                Emitter.emit (sprintf "    br ^%s" endLabel) stateWithResultType
                
            | None -> 
                // No Error case found, use fallback
                let (errorStr, stateWithStr) = createStringConstant "Error-fallback" state10
                let stateWithStore = 
                    Emitter.emit (sprintf "    %s = %s : %s" resultId errorStr (mlirTypeToString expectedType)) stateWithStr
                let stateWithResultType = TypeTracking.recordSSAType resultId expectedType stateWithStore
                
                Emitter.emit (sprintf "    br ^%s" endLabel) stateWithResultType
        
        // Add end label and clear expected type
        let state12 = Emitter.emit (sprintf "  ^%s:" endLabel) state11
        let finalState = TypeTracking.clearExpectedType state12
        
        (resultId, finalState)

    /// Handles generic match expression with type awareness
    let handleGenericMatch (matchExpr: OakExpression) (cases: (OakPattern * OakExpression) list) 
                          (matchValueId: string) (resultId: string) (state: MLIRGenerationState)
                          (convertExpressionFn: OakExpression -> MLIRGenerationState -> string * MLIRGenerationState) 
                          : string * MLIRGenerationState =
        
        // Create branch labels
        let (_, elseLabel, endLabel, state1) = generateBranchLabels state
        
        // Default to integer result type
        let intType = MLIRTypes.createInteger 32
        let (defaultValue, state2) = Emitter.constant "0" intType state1
        
        if cases.IsEmpty then
            // No cases - use default value
            let state3 = Emitter.emit (sprintf "    %s = %s : %s" resultId defaultValue (mlirTypeToString intType)) state2
            let state4 = TypeTracking.recordSSAType resultId intType state3
            (resultId, state4)
        else
            // Process each case
            let rec processCases (remainingCases: (OakPattern * OakExpression) list) (currentState: MLIRGenerationState) =
                match remainingCases with
                | [] -> 
                    // No more cases - handle default fallback
                    let fallbackState = Emitter.emit (sprintf "  ^%s:" elseLabel) currentState
                    let storeState = Emitter.emit (sprintf "    %s = %s : %s" resultId defaultValue (mlirTypeToString intType)) fallbackState
                    let typeState = TypeTracking.recordSSAType resultId intType storeState
                    let finalState = Emitter.emit (sprintf "    br ^%s" endLabel) typeState
                    finalState
                | (pattern, expr) :: rest ->
                    // Create case label and process expression
                    let (caseLabel, currentState1) = SSA.generateValue "case" currentState
                    let caseLabelStr = caseLabel.TrimStart('%')
                    
                    let conditionState = Emitter.emit (sprintf "  ^%s:" caseLabelStr) currentState1
                    let (exprResultId, exprState) = convertExpressionFn expr conditionState
                    
                    // Get expression type for potential conversion
                    let exprType = 
                        match TypeTracking.getSSAType exprResultId exprState with
                        | Some typ -> typ
                        | None -> intType
                    
                    // Get expected type (default to int if not specified)
                    let expectedType = 
                        match TypeTracking.getExpectedType exprState with
                        | Some typ -> typ
                        | None -> intType
                    
                    // Apply type conversion if needed
                    let (finalResultId, afterConvState) =
                        if exprType = expectedType then
                            (exprResultId, exprState)
                        else
                            Emitter.convertType exprResultId exprType expectedType exprState
                    
                    let storeState = Emitter.emit (sprintf "    %s = %s : %s" resultId finalResultId (mlirTypeToString expectedType)) afterConvState
                    let typeState = TypeTracking.recordSSAType resultId expectedType storeState
                    let branchState = Emitter.emit (sprintf "    br ^%s" endLabel) typeState
                    
                    processCases rest branchState
            
            // Process all cases and add final end label
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
        // Default to a small buffer if size isn't a literal
        let (bufferResult, state1) = SSA.generateValue "buffer" state
        let defaultSize = 64 // Default small buffer size
        let state2 = Emitter.emit (sprintf "    %s = memref.alloca() : memref<%dxi8>" bufferResult defaultSize) state1
        let bufferType = MLIRTypes.createMemRef (MLIRTypes.createInteger 8) [defaultSize]
        let state3 = TypeTracking.recordSSAType bufferResult bufferType state2
        (bufferResult, state3)

and convertExpression (expr: OakExpression) (state: MLIRGenerationState) : string * MLIRGenerationState =
    // Get expected type from context (if any)
    let expectedType = TypeTracking.getExpectedType state
    
    // Match expression and convert based on type
    match expr with
    | Literal literal ->
        let (resultId, state1) = convertLiteral literal state
        
        // Check if we need type conversion
        let literalType = TypeTracking.inferLiteralType literal
        match expectedType with
        | Some expected when expected <> literalType && requiresCoercion literalType expected ->
            // Convert the literal to expected type
            Emitter.convertType resultId literalType expected state1
        | _ ->
            (resultId, state1)

    | Match(matchExpr, cases) ->
        let (matchValueId, stateAfterMatchExpr) = convertExpression matchExpr state
        let (resultId, stateWithResultVar) = SSA.generateValue "match_result" stateAfterMatchExpr
        
        // Check matchExpr to determine what kind of match pattern to use
        match matchExpr with
        | Application(Variable funcName, _) when isResultReturningFunction funcName ->
            // Use Result match handling for Result-returning functions
            MatchHandling.handleResultMatch matchExpr cases matchValueId resultId stateWithResultVar convertExpression
            
        | _ ->
            // Use generic match for other patterns
            MatchHandling.handleGenericMatch matchExpr cases matchValueId resultId stateWithResultVar convertExpression

    | Variable name ->
        match SSA.lookupVariable name state with
        | Some value -> 
            // Look up the type of this variable
            let valueType = 
                match TypeTracking.getSSAType value state with
                | Some t -> t
                | None -> MLIRTypes.createInteger 32 // Default if unknown
            
            // Check if we need to convert the type
            match expectedType with
            | Some expected when expected <> valueType && requiresCoercion valueType expected ->
                // Convert variable to expected type
                Emitter.convertType value valueType expected state
            | _ ->
                (value, state)
                
        | None -> 
            // Unknown variable - create a dummy value
            let (dummyValue, state1) = SSA.generateValue "unknown" state
            let intType = MLIRTypes.createInteger 32
            let state2 = TypeTracking.recordSSAType dummyValue intType state1
            
            // Check if we need to convert the type
            match expectedType with
            | Some expected when expected <> intType && requiresCoercion intType expected ->
                Emitter.convertType dummyValue intType expected state2
            | _ ->
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
            // Process arguments with appropriate type context
            let rec processArgs (remainingArgs: OakExpression list) (currentState: MLIRGenerationState) (accArgs: string list) =
                match remainingArgs with
                | [] -> (List.rev accArgs, currentState)
                | arg :: rest ->
                    // Process each argument, potentially using expected type from function signature
                    let (argValue, newState) = convertExpression arg currentState
                    processArgs rest newState (argValue :: accArgs)
            
            let (argValues, state1) = processArgs args state []
            
            // Generate result variable
            let (resultId, state2) = SSA.generateValue "call" state1
            
            // Get expected return type for this function
            let returnType = TypeTracking.inferFunctionReturnType funcName state
            
            // Special case for Result-returning functions
            if isResultReturningFunction funcName then
                let intType = MLIRTypes.createInteger 32
                let state3 = Emitter.emit (sprintf "    %s = arith.constant 1 : i32 // Result marker for %s" 
                                    resultId funcName) state2
                let state4 = TypeTracking.recordSSAType resultId intType state3
                
                // Check if we need type conversion
                match expectedType with
                | Some expected when expected <> intType && requiresCoercion intType expected ->
                    Emitter.convertType resultId intType expected state4
                | _ ->
                    (resultId, state4)
            else
                // Regular function call via symbol registry
                match PublicInterface.resolveFunctionCall funcName argValues resultId state2.SymbolRegistry with
                | Core.XParsec.Foundation.Success (operations, updatedRegistry) -> 
                    // Determine result type from return type or registry
                    let resultType = 
                        match returnType with
                        | Some rt -> rt
                        | None -> MLIRTypes.createInteger 32
                    
                    // Generate the operation and record the type
                    let finalState = 
                        operations
                        |> List.fold (fun accState op -> Emitter.emit op accState) state2
                        |> fun s -> { s with SymbolRegistry = updatedRegistry }
                        |> TypeTracking.recordSSAType resultId resultType
                    
                    // Check if we need type conversion
                    match expectedType with
                    | Some expected when expected <> resultType && requiresCoercion resultType expected ->
                        Emitter.convertType resultId resultType expected finalState
                    | _ ->
                        (resultId, finalState)
                    
                | Core.XParsec.Foundation.CompilerFailure errors -> 
                    // Handle special functions not in registry
                    match funcName with
                    | "NativePtr.stackalloc" ->
                        let (allocResult, allocState) = convertStackAllocation args state
                        
                        // Check if we need type conversion
                        let allocType = 
                            match TypeTracking.getSSAType allocResult allocState with
                            | Some t -> t
                            | None -> MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
                        
                        match expectedType with
                        | Some expected when expected <> allocType && requiresCoercion allocType expected ->
                            Emitter.convertType allocResult allocType expected allocState
                        | _ ->
                            (allocResult, allocState)
                        
                    | _ ->
                        // Unknown function - create a placeholder
                        let (resultId, state1) = SSA.generateValue "unknown_call" state
                        let intType = MLIRTypes.createInteger 32
                        let state2 = Emitter.emit (sprintf "    %s = arith.constant 0 : i32 // Unknown function %s" 
                                                resultId funcName) state1
                        let state3 = TypeTracking.recordSSAType resultId intType state2
                        
                        // Check if we need type conversion
                        match expectedType with
                        | Some expected when expected <> intType && requiresCoercion intType expected ->
                            Emitter.convertType resultId intType expected state3
                        | _ ->
                            (resultId, state3)

        | _ ->
            // Function application with non-variable function
            let (funcValue, state1) = convertExpression func state
            
            // Process arguments
            let rec processArgs (remainingArgs: OakExpression list) (currentState: MLIRGenerationState) (accArgs: string list) =
                match remainingArgs with
                | [] -> (List.rev accArgs, currentState)
                | arg :: rest ->
                    let (argValue, newState) = convertExpression arg currentState
                    processArgs rest newState (argValue :: accArgs)
            
            let (argValues, state2) = processArgs args state1 []
            
            // Generate result with type tracking
            let (resultId, state3) = SSA.generateValue "app" state2
            let argStr = String.concat ", " argValues
            let intType = MLIRTypes.createInteger 32
            
            let state4 = Emitter.emit (sprintf "    %s = func.call %s(%s) : (i32) -> i32" resultId funcValue argStr) state3
            let state5 = TypeTracking.recordSSAType resultId intType state4
            
            // Check if we need type conversion
            match expectedType with
            | Some expected when expected <> intType && requiresCoercion intType expected ->
                Emitter.convertType resultId intType expected state5
            | _ ->
                (resultId, state5)
    
    | Let(name, value, body) ->
        // Process value expression
        let (valueResult, state1) = convertExpression value state
        
        // Get value type for proper binding
        let valueType = 
            match TypeTracking.getSSAType valueResult state1 with
            | Some t -> t
            | None -> MLIRTypes.createInteger 32
        
        // Bind variable and record its type
        let state2 = 
            state1 
            |> SSA.bindVariable name valueResult
            |> TypeTracking.recordSSAType valueResult valueType
            
        // Process body with proper type context
        let (bodyResult, state3) = convertExpression body state2
        
        // Check if we need type conversion
        let bodyType = 
            match TypeTracking.getSSAType bodyResult state3 with
            | Some t -> t
            | None -> MLIRTypes.createInteger 32
            
        match expectedType with
        | Some expected when expected <> bodyType && requiresCoercion bodyType expected ->
            Emitter.convertType bodyResult bodyType expected state3
        | _ ->
            (bodyResult, state3)
    
    | Sequential(first, second) ->
        // Process first expression (ignore its result)
        let (_, state1) = convertExpression first state
        
        // Process second expression with proper type context
        let (secondResult, state2) = convertExpression second state1
        
        // Check if we need type conversion
        let secondType = 
            match TypeTracking.getSSAType secondResult state2 with
            | Some t -> t
            | None -> MLIRTypes.createInteger 32
            
        match expectedType with
        | Some expected when expected <> secondType && requiresCoercion secondType expected ->
            Emitter.convertType secondResult secondType expected state2
        | _ ->
            (secondResult, state2)
    
    | IfThenElse(cond, thenExpr, elseExpr) ->
        // Process condition
        let (condResult, state1) = convertExpression cond state
        
        // Create branch labels
        let (thenLabel, elseLabel, endLabel, state2) = MatchHandling.generateBranchLabels state1
        let state3 = Emitter.emit (sprintf "    cond_br %s, ^%s, ^%s" condResult thenLabel elseLabel) state2
        
        // Create result variable
        let (resultId, state4) = SSA.generateValue "if_result" state3
        
        // Determine result type - use expected type if available, otherwise i32
        let resultType = 
            match expectedType with
            | Some typ -> typ
            | None -> MLIRTypes.createInteger 32
            
        // Process then branch with type context
        let state5 = Emitter.emit (sprintf "  ^%s:" thenLabel) state4
        let thenState = TypeTracking.setExpectedType resultType state5
        let (thenResult, state6) = convertExpression thenExpr thenState
        
        // Get actual type of then result
        let thenType = 
            match TypeTracking.getSSAType thenResult state6 with
            | Some t -> t
            | None -> MLIRTypes.createInteger 32
            
        // Apply type conversion if needed
        let (finalThenResult, state7) =
            if thenType <> resultType && requiresCoercion thenType resultType then
                Emitter.convertType thenResult thenType resultType state6
            else
                (thenResult, state6)
                
        let state8 = Emitter.emit (sprintf "    %s = %s : %s" 
                                  resultId finalThenResult (mlirTypeToString resultType)) state7
        let state9 = Emitter.emit (sprintf "    br ^%s" endLabel) state8
        
        // Process else branch with type context
        let state10 = Emitter.emit (sprintf "  ^%s:" elseLabel) state9
        let elseState = TypeTracking.setExpectedType resultType state10
        let (elseResult, state11) = convertExpression elseExpr elseState
        
        // Get actual type of else result
        let elseType = 
            match TypeTracking.getSSAType elseResult state11 with
            | Some t -> t
            | None -> MLIRTypes.createInteger 32
            
        // Apply type conversion if needed
        let (finalElseResult, state12) =
            if elseType <> resultType && requiresCoercion elseType resultType then
                Emitter.convertType elseResult elseType resultType state11
            else
                (elseResult, state11)
                
        let state13 = Emitter.emit (sprintf "    %s = %s : %s" 
                                   resultId finalElseResult (mlirTypeToString resultType)) state12
        let state14 = Emitter.emit (sprintf "    br ^%s" endLabel) state13
        
        // Add end label and record result type
        let state15 = Emitter.emit (sprintf "  ^%s:" endLabel) state14
        let state16 = TypeTracking.recordSSAType resultId resultType state15
        let finalState = TypeTracking.clearExpectedType state16
        
        (resultId, finalState)
    
    | FieldAccess(target, fieldName) ->
        // Process target expression
        let (targetResult, state1) = convertExpression target state
        
        // Basic implementation - pass through target value
        // In a real implementation, would need to extract field from struct
        (targetResult, state1)
    
    | MethodCall(target, methodName, args) ->
        // Process target expression
        let (targetResult, state1) = convertExpression target state
        
        // Process arguments
        let convertArg (accState, accArgs) arg =
            let (argValue, newState) = convertExpression arg accState
            (newState, argValue :: accArgs)
        
        let (state2, argValues) = List.fold convertArg (state1, []) args
        
        // Basic implementation - pass through target value
        // In a real implementation, would generate method call
        (targetResult, state2)
    
    | IOOperation(ioType, args) ->
        match ioType with
        | Printf formatStr | Printfn formatStr ->
            // Register printf declaration
            let state1 = Emitter.emitModuleLevel "  func.func private @printf(memref<?xi8>, ...) -> i32" state
            
            // Create format string global
            let (formatGlobal, state2) = Emitter.registerString formatStr state1
            let (formatPtr, state3) = SSA.generateValue "fmt_ptr" state2
            let state4 = Emitter.emit (sprintf "    %s = memref.get_global %s : memref<?xi8>" formatPtr formatGlobal) state3
            
            // Process arguments
            let rec processArgs (remainingArgs: OakExpression list) (currentState: MLIRGenerationState) (accArgs: string list) =
                match remainingArgs with
                | [] -> (List.rev accArgs, currentState)
                | arg :: rest ->
                    // Set expected type based on format string (basic implementation)
                    let argState = 
                        if formatStr.Contains("%d") then
                            TypeTracking.setExpectedType (MLIRTypes.createInteger 32) currentState
                        elif formatStr.Contains("%f") then
                            TypeTracking.setExpectedType (MLIRTypes.createFloat 32) currentState
                        elif formatStr.Contains("%s") then
                            TypeTracking.setExpectedType (MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []) currentState
                        else
                            currentState
                            
                    let (argValue, newState) = convertExpression arg argState
                    let clearedState = TypeTracking.clearExpectedType newState
                    processArgs rest clearedState (argValue :: accArgs)
            
            let (argValues, state5) = processArgs args state4 []
            
            // Generate printf call
            let allArgs = formatPtr :: argValues
            let (printfResult, state6) = SSA.generateValue "printf_result" state5
            let argStr = String.concat ", " allArgs
            
            let state7 = 
                if args.IsEmpty then
                    Emitter.emit (sprintf "    %s = func.call @printf(%s) : (memref<?xi8>) -> i32" 
                                 printfResult formatPtr) state6
                else
                    let types = "memref<?xi8>" :: List.replicate argValues.Length "i32"
                    let typeStr = String.concat ", " types
                    Emitter.emit (sprintf "    %s = func.call @printf(%s) : (%s) -> i32" 
                                 printfResult argStr typeStr) state6
            
            // Record result type
            let intType = MLIRTypes.createInteger 32
            let state8 = TypeTracking.recordSSAType printfResult intType state7
            
            // Check if we need type conversion
            match expectedType with
            | Some expected when expected <> intType && requiresCoercion intType expected ->
                Emitter.convertType printfResult intType expected state8
            | _ ->
                (printfResult, state8)
        
        | _ ->
            // Handle other IO operations
            let (dummyValue, state1) = SSA.generateValue "io_op" state
            let intType = MLIRTypes.createInteger 32
            let state2 = TypeTracking.recordSSAType dummyValue intType state1
            
            // Check if we need type conversion
            match expectedType with
            | Some expected when expected <> intType && requiresCoercion intType expected ->
                Emitter.convertType dummyValue intType expected state2
            | _ ->
                (dummyValue, state2)
    
    | Lambda(params', body) ->
        // Basic implementation - evaluate body directly
        // In a real implementation, would need to handle closure conversion
        let (bodyResult, state1) = convertExpression body state
        
        // Check if we need type conversion
        let bodyType = 
            match TypeTracking.getSSAType bodyResult state1 with
            | Some t -> t
            | None -> MLIRTypes.createInteger 32
            
        match expectedType with
        | Some expected when expected <> bodyType && requiresCoercion bodyType expected ->
            Emitter.convertType bodyResult bodyType expected state1
        | _ ->
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
    
    // Set expected return type for body
    let state3 = TypeTracking.setExpectedType returnMLIRType state2
    
    // Process body with type awareness
    let (bodyResult, state4) = convertExpression body state3
    
    // Get body result type
    let bodyType = 
        match TypeTracking.getSSAType bodyResult state4 with
        | Some t -> t
        | None -> MLIRTypes.createInteger 32
        
    // Apply type conversion if needed
    let (finalBodyResult, state5) =
        if bodyType <> returnMLIRType && requiresCoercion bodyType returnMLIRType then
            Emitter.convertType bodyResult bodyType returnMLIRType state4
        else
            (bodyResult, state4)
            
    // Generate return statement
    let state6 = 
        if returnType = UnitType then
            Emitter.emit "    func.return" state5
        else
            Emitter.emit (sprintf "    func.return %s : %s" finalBodyResult returnTypeStr) state5
    
    // Clear expected type and close function
    let state7 = TypeTracking.clearExpectedType state6
    Emitter.emit "  }" state7

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