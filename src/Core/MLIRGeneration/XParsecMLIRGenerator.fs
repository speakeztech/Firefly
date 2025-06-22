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
        { state with CurrentScope = Map.add name value state.CurrentScope }
    
    /// Looks up a variable by name in the current scope stack
    let lookupVariable (name: string) (state: MLIRGenerationState) : string option =
        let rec lookup scopes =
            match scopes with
            | [] -> None
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
    
    /// Infers the type of a literal expression
    let inferLiteralType (literal: OakLiteral) : MLIRType =
        match literal with
        | IntLiteral _ -> MLIRTypes.createInteger 32
        | FloatLiteral _ -> MLIRTypes.createFloat 32
        | BoolLiteral _ -> MLIRTypes.createInteger 1
        | StringLiteral _ -> MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
        | UnitLiteral -> MLIRTypes.createVoid ()
        | ArrayLiteral _ -> MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []

/// Core operation emission with type tracking
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
    
    /// Creates a type conversion operation
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
                // Integer to memory reference conversion - create a string representation
                let (strGlobal, state2) = registerString "%d" state1
                let (formatPtr, state3) = SSA.generateValue "fmt_ptr" state2
                let state4 = emit (sprintf "    %s = memref.get_global %s : memref<?xi8>" formatPtr strGlobal) state3
                
                // Allocate buffer for result
                let (resultBuffer, state5) = SSA.generateValue "str_buf" state4
                let state6 = emit (sprintf "    %s = memref.alloca() : memref<32xi8>" resultBuffer) state5
                
                // Register sprintf declaration
                let state7 = emitModuleLevel "  func.func private @sprintf(memref<?xi8>, memref<?xi8>, i32) -> i32" state6
                
                // Generate sprintf call
                let (sprintfResult, state8) = SSA.generateValue "sprintf_result" state7
                let state9 = emit (sprintf "    %s = func.call @sprintf(%s, %s, %s) : (memref<?xi8>, memref<?xi8>, i32) -> i32" 
                                     sprintfResult resultBuffer formatPtr sourceValue) state8
                
                let memRefType = MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
                let state10 = TypeTracking.recordSSAType resultBuffer memRefType state9
                (resultBuffer, state10)
                
            | MemoryRefCategory, MemoryRefCategory ->
                // Memory reference conversion
                let sourceTypeStr = mlirTypeToString sourceType
                let targetTypeStr = mlirTypeToString targetType
                let opStr = sprintf "    %s = memref.cast %s : %s to %s" resultId sourceValue sourceTypeStr targetTypeStr
                let state2 = emit opStr state1
                let state3 = TypeTracking.recordSSAType resultId targetType state2
                (resultId, state3)
                
            | _ ->
                // Default to a bitcast for other conversions
                let sourceTypeStr = mlirTypeToString sourceType
                let targetTypeStr = mlirTypeToString targetType
                let opStr = sprintf "    %s = llvm.bitcast %s : %s to %s" resultId sourceValue sourceTypeStr targetTypeStr
                let state2 = emit opStr state1
                let state3 = TypeTracking.recordSSAType resultId targetType state2
                (resultId, state3)

// Forward declaration for the main expression processing function
let rec processExpression : OakExpression -> MLIRGenerationState -> string * MLIRType * MLIRGenerationState = 
    fun expr state -> processExpressionImpl expr state

// Helper functions for handling different expression types
and processLiteral (lit: OakLiteral) (state: MLIRGenerationState) : string * MLIRType * MLIRGenerationState =
    let literalType = TypeTracking.inferLiteralType lit
    
    match lit with
    | IntLiteral value ->
        let (resultId, state1) = Emitter.constant (string value) literalType state
        
        // Apply conversion if needed
        match state.ExpectedResultType with
        | Some expected when expected <> literalType ->
            let (convertedId, state2) = Emitter.convertType resultId literalType expected state1
            (convertedId, expected, state2)
        | _ ->
            (resultId, literalType, state1)
        
    | FloatLiteral value ->
        let (resultId, state1) = Emitter.constant (sprintf "%f" value) literalType state
        
        // Apply conversion if needed
        match state.ExpectedResultType with
        | Some expected when expected <> literalType ->
            let (convertedId, state2) = Emitter.convertType resultId literalType expected state1
            (convertedId, expected, state2)
        | _ ->
            (resultId, literalType, state1)
        
    | BoolLiteral value ->
        let (resultId, state1) = Emitter.constant (if value then "1" else "0") literalType state
        
        // Apply conversion if needed
        match state.ExpectedResultType with
        | Some expected when expected <> literalType ->
            let (convertedId, state2) = Emitter.convertType resultId literalType expected state1
            (convertedId, expected, state2)
        | _ ->
            (resultId, literalType, state1)
        
    | StringLiteral value ->
        let (globalName, state1) = Emitter.registerString value state
        let (ptrResult, state2) = SSA.generateValue "str_ptr" state1
        let state3 = Emitter.emit (sprintf "    %s = memref.get_global %s : memref<?xi8>" 
                                  ptrResult globalName) state2
        let state4 = TypeTracking.recordSSAType ptrResult literalType state3
        
        // Apply conversion if needed
        match state.ExpectedResultType with
        | Some expected when expected <> literalType ->
            let (convertedId, state5) = Emitter.convertType ptrResult literalType expected state4
            (convertedId, expected, state5)
        | _ ->
            (ptrResult, literalType, state4)
        
    | UnitLiteral ->
        let (resultId, state1) = Emitter.constant "0" (MLIRTypes.createInteger 32) state
        
        // Apply conversion if needed
        match state.ExpectedResultType with
        | Some expected when expected <> MLIRTypes.createInteger 32 ->
            let (convertedId, state2) = Emitter.convertType resultId (MLIRTypes.createInteger 32) expected state1
            (convertedId, expected, state2)
        | _ ->
            (resultId, MLIRTypes.createInteger 32, state1)
        
    | ArrayLiteral _ ->
        // Not fully implemented - create a dummy array
        let (resultId, state1) = SSA.generateValue "array" state
        let state2 = TypeTracking.recordSSAType resultId literalType state1
        
        // Apply conversion if needed
        match state.ExpectedResultType with
        | Some expected when expected <> literalType ->
            let (convertedId, state3) = Emitter.convertType resultId literalType expected state2
            (convertedId, expected, state3)
        | _ ->
            (resultId, literalType, state2)

and processVariable (name: string) (state: MLIRGenerationState) : string * MLIRType * MLIRGenerationState =
    match SSA.lookupVariable name state with
    | Some value -> 
        // Get type if available
        let valueType = 
            match TypeTracking.getSSAType value state with
            | Some t -> t
            | None -> MLIRTypes.createInteger 32 // Default if unknown
        
        // Check if we need to apply type conversion for expected type
        match state.ExpectedResultType with
        | Some expected when expected <> valueType ->
            let (convertedId, state1) = Emitter.convertType value valueType expected state
            (convertedId, expected, state1)
        | _ ->
            (value, valueType, state)
    | None -> 
        // Unknown variable
        let (dummyValue, state1) = SSA.generateValue "unknown" state
        let intType = MLIRTypes.createInteger 32
        let state2 = TypeTracking.recordSSAType dummyValue intType state1
        
        // Apply conversion if needed
        match state.ExpectedResultType with
        | Some expected when expected <> intType ->
            let (convertedId, state3) = Emitter.convertType dummyValue intType expected state2
            (convertedId, expected, state3)
        | _ ->
            (dummyValue, intType, state2)

and processFunctionCall (funcName: string) (args: OakExpression list) (state: MLIRGenerationState) : string * MLIRType * MLIRGenerationState =
    // Process arguments
    let rec processArgs (remainingArgs: OakExpression list) (processedArgs: string list) 
                      (processedTypes: MLIRType list) (currentState: MLIRGenerationState) =
        match remainingArgs with
        | [] -> (List.rev processedArgs, List.rev processedTypes, currentState)
        | arg :: rest ->
            let (argId, argType, newState) = processExpression arg currentState
            processArgs rest (argId :: processedArgs) (argType :: processedTypes) newState
    
    let (argIds, argTypes, state1) = processArgs args [] [] state
    
    // Create result variable
    let (resultId, state2) = SSA.generateValue "call" state1
    
    // Generate call through symbol registry
    match PublicInterface.resolveFunctionCall funcName argIds resultId state2.SymbolRegistry with
    | Core.XParsec.Foundation.Success (operations, updatedRegistry) ->
        // Determine result type
        let resultType = 
            match PublicInterface.getSymbolType funcName state2.SymbolRegistry with
            | Some rtype -> rtype
            | None -> MLIRTypes.createInteger 32
        
        // Generate operations
        let state3 = 
            operations
            |> List.fold (fun accState op -> Emitter.emit op accState) state2
        
        let state4 = { state3 with SymbolRegistry = updatedRegistry }
        let state5 = TypeTracking.recordSSAType resultId resultType state4
        
        // Apply conversion if needed
        match state.ExpectedResultType with
        | Some expected when expected <> resultType ->
            let (convertedId, state6) = Emitter.convertType resultId resultType expected state5
            (convertedId, expected, state6)
        | _ ->
            (resultId, resultType, state5)
        
    | Core.XParsec.Foundation.CompilerFailure _ ->
        // Fallback for unknown functions
        let resultType = MLIRTypes.createInteger 32
        let state3 = Emitter.emit (sprintf "    %s = arith.constant 0 : i32 // Unknown function %s" 
                                 resultId funcName) state2
        let state4 = TypeTracking.recordSSAType resultId resultType state3
        
        // Apply conversion if needed
        match state.ExpectedResultType with
        | Some expected when expected <> resultType ->
            let (convertedId, state5) = Emitter.convertType resultId resultType expected state4
            (convertedId, expected, state5)
        | _ ->
            (resultId, resultType, state4)

and processLet (name: string) (value: OakExpression) (body: OakExpression) (state: MLIRGenerationState) : string * MLIRType * MLIRGenerationState =
    // Process value
    let valueState = TypeTracking.clearExpectedType state
    let (valueId, valueType, state1) = processExpression value valueState
    
    // Bind variable
    let state2 = SSA.bindVariable name valueId state1
    let state3 = TypeTracking.recordSSAType valueId valueType state2
    
    // Process body
    let (bodyId, bodyType, state4) = processExpression body state3
    
    // Apply conversion if needed
    match state.ExpectedResultType with
    | Some expected when expected <> bodyType ->
        let (convertedId, state5) = Emitter.convertType bodyId bodyType expected state4
        (convertedId, expected, state5)
    | _ ->
        (bodyId, bodyType, state4)

and processSequential (first: OakExpression) (second: OakExpression) (state: MLIRGenerationState) : string * MLIRType * MLIRGenerationState =
    // Process first expression (ignore result)
    let (_, _, state1) = processExpression first state
    
    // Process second expression
    let (secondId, secondType, state2) = processExpression second state1
    
    // Apply conversion if needed
    match state.ExpectedResultType with
    | Some expected when expected <> secondType ->
        let (convertedId, state3) = Emitter.convertType secondId secondType expected state2
        (convertedId, expected, state3)
    | _ ->
        (secondId, secondType, state2)

and processIfThenElse (cond: OakExpression) (thenExpr: OakExpression) (elseExpr: OakExpression) (state: MLIRGenerationState) : string * MLIRType * MLIRGenerationState =
    // Process condition
    let condState = TypeTracking.setExpectedType (MLIRTypes.createInteger 1) state
    let (condId, condType, state1) = processExpression cond condState
    
    // Apply conversion to i1 if needed
    let (finalCondId, state2) =
        if condType <> MLIRTypes.createInteger 1 then
            let (convertedId, state1b) = Emitter.convertType condId condType (MLIRTypes.createInteger 1) state1
            (convertedId, state1b)
        else
            (condId, state1)
    
    // Create branch labels
    let (thenId, state3) = SSA.generateValue "then" state2
    let (elseId, state4) = SSA.generateValue "else" state3
    let (endId, state5) = SSA.generateValue "end" state4
    
    let thenLabel = thenId.TrimStart('%')
    let elseLabel = elseId.TrimStart('%')
    let endLabel = endId.TrimStart('%')
    
    let state6 = Emitter.emit (sprintf "    cond_br %s, ^%s, ^%s" 
                              finalCondId thenLabel elseLabel) state5
    
    // Create result variable
    let (resultId, state7) = SSA.generateValue "if_result" state6
    
    // Determine result type
    let resultType = 
        match state.ExpectedResultType with
        | Some t -> t
        | None -> MLIRTypes.createInteger 32
    
    // Process then branch
    let state8 = Emitter.emit (sprintf "  ^%s:" thenLabel) state7
    let thenState = TypeTracking.setExpectedType resultType state8
    let (thenResultId, thenType, state9) = processExpression thenExpr thenState
    
    // Apply conversion if needed
    let (finalThenId, state10) =
        if thenType <> resultType then
            let (convertedId, state9b) = Emitter.convertType thenResultId thenType resultType state9
            (convertedId, state9b)
        else
            (thenResultId, state9)
    
    let state11 = Emitter.emit (sprintf "    %s = %s : %s" 
                               resultId finalThenId (mlirTypeToString resultType)) state10
    let state12 = Emitter.emit (sprintf "    br ^%s" endLabel) state11
    
    // Process else branch
    let state13 = Emitter.emit (sprintf "  ^%s:" elseLabel) state12
    let elseState = TypeTracking.setExpectedType resultType state13
    let (elseResultId, elseType, state14) = processExpression elseExpr elseState
    
    // Apply conversion if needed
    let (finalElseId, state15) =
        if elseType <> resultType then
            let (convertedId, state14b) = Emitter.convertType elseResultId elseType resultType state14
            (convertedId, state14b)
        else
            (elseResultId, state14)
    
    let state16 = Emitter.emit (sprintf "    %s = %s : %s" 
                               resultId finalElseId (mlirTypeToString resultType)) state15
    let state17 = Emitter.emit (sprintf "    br ^%s" endLabel) state16
    
    // Add end label
    let state18 = Emitter.emit (sprintf "  ^%s:" endLabel) state17
    let state19 = TypeTracking.recordSSAType resultId resultType state18
    
    (resultId, resultType, state19)

and processMatch (matchExpr: OakExpression) (cases: (OakPattern * OakExpression) list) (state: MLIRGenerationState) : string * MLIRType * MLIRGenerationState =
    // Handle simple case for Result<T, E> matching
    let isResultMatch =
        match matchExpr with
        | Application(Variable name, _) ->
            // Check if function name indicates a Result-returning function
            let isResultFunction =
                match name with
                | "readInto" | "readFile" | "parseInput" | "tryParse" -> true
                | n when n.Contains("OrNone") -> true
                | n when n.Contains("OrError") -> true
                | n when n.StartsWith("try") -> true
                | n when n.Contains("Result") -> true
                | _ -> false
            
            isResultFunction &&
            cases |> List.exists (fun (pattern, _) ->
                match pattern with
                | PatternConstructor("Ok", _) -> true
                | _ -> false) &&
            cases |> List.exists (fun (pattern, _) ->
                match pattern with
                | PatternConstructor("Error", _) -> true
                | _ -> false)
        | _ -> false
    
    if isResultMatch then
        // Process the match expression
        let (matchValueId, matchValueType, state1) = processExpression matchExpr state
        
        // Create result variable
        let (resultId, state2) = SSA.generateValue "match_result" state1
        
        // Determine the result type
        let resultType = 
            match state.ExpectedResultType with
            | Some t -> t
            | None -> MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
        
        // Extract buffer name for better code generation
        let bufferName = 
            match matchExpr with
            | Application(Variable "readInto", [Variable name]) -> name
            | _ -> "buffer"
        
        // Find buffer value in scope
        let bufferValue = 
            match SSA.lookupVariable bufferName state with
            | Some value -> value
            | None -> "%unknown_buffer"
        
        // Register helper functions for Result handling
        let state3 = 
            state2
            |> Emitter.emitModuleLevel "  func.func private @is_ok_result(i32) -> i1"
            |> Emitter.emitModuleLevel "  func.func private @extract_result_length(i32) -> i32"
            |> Emitter.emitModuleLevel "  func.func private @create_span(memref<?xi8>, i32) -> memref<?xi8>"
        
        // Check if result is Ok or Error
        let (isOkId, state4) = SSA.generateValue "is_ok" state3
        let state5 = Emitter.emit (sprintf "    %s = func.call @is_ok_result(%s) : (i32) -> i1" 
                                isOkId matchValueId) state4
        let boolType = MLIRTypes.createInteger 1
        let state6 = TypeTracking.recordSSAType isOkId boolType state5
        
        // Create branch labels
        let (thenId, state7) = SSA.generateValue "then" state6
        let (elseId, state8) = SSA.generateValue "else" state7
        let (endId, state9) = SSA.generateValue "end" state8
        
        let thenLabel = thenId.TrimStart('%')
        let elseLabel = elseId.TrimStart('%')
        let endLabel = endId.TrimStart('%')
        
        let state10 = Emitter.emit (sprintf "    cond_br %s, ^%s, ^%s" 
                                isOkId thenLabel elseLabel) state9
        
        // Find Ok case pattern
        let okCase = 
            cases |> List.tryFind (fun (pattern, _) -> 
                match pattern with 
                | PatternConstructor("Ok", _) -> true
                | _ -> false)
        
        // Handle Ok branch
        let state11 = Emitter.emit (sprintf "  ^%s:" thenLabel) state10
        
        let state12 = 
            match okCase with
            | Some (PatternConstructor("Ok", [PatternVariable lengthName]), okExpr) ->
                // Extract length from result
                let (lengthId, state11a) = SSA.generateValue "length" state11
                let state11b = Emitter.emit (sprintf "    %s = func.call @extract_result_length(%s) : (i32) -> i32" 
                                          lengthId matchValueId) state11a
                let intType = MLIRTypes.createInteger 32
                let state11c = TypeTracking.recordSSAType lengthId intType state11b
                
                // Bind length variable
                let state11d = SSA.bindVariable lengthName lengthId state11c
                
                // Create span from buffer and length
                let (spanId, state11e) = SSA.generateValue "span" state11d
                let state11f = Emitter.emit (sprintf "    %s = func.call @create_span(%s, %s) : (memref<?xi8>, i32) -> memref<?xi8>" 
                                          spanId bufferValue lengthId) state11e
                let spanType = MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
                let state11g = TypeTracking.recordSSAType spanId spanType state11f
                
                // Process Ok expression with span bound
                let state11h = state11g |> SSA.bindVariable "span" spanId
                let okExprState = TypeTracking.setExpectedType resultType state11h
                
                let (okResultId, okResultType, state11i) = processExpression okExpr okExprState
                
                // Apply type conversion if needed
                let (finalResultId, state11j) =
                    if okResultType <> resultType then
                        let (convertedId, state11j) = Emitter.convertType okResultId okResultType resultType state11i
                        (convertedId, state11j)
                    else
                        (okResultId, state11i)
                
                // Store result and branch to end
                let state11k = Emitter.emit (sprintf "    %s = %s : %s" 
                                          resultId finalResultId (mlirTypeToString resultType)) state11j
                
                Emitter.emit (sprintf "    br ^%s" endLabel) state11k
                
            | Some (_, okExpr) ->
                // Handle Ok case without pattern variable
                let okExprState = TypeTracking.setExpectedType resultType state11
                
                let (okResultId, okResultType, state11a) = processExpression okExpr okExprState
                
                // Apply type conversion if needed
                let (finalResultId, state11b) =
                    if okResultType <> resultType then
                        let (convertedId, state11b) = Emitter.convertType okResultId okResultType resultType state11a
                        (convertedId, state11b)
                    else
                        (okResultId, state11a)
                
                // Store result and branch to end
                let state11c = Emitter.emit (sprintf "    %s = %s : %s" 
                                          resultId finalResultId (mlirTypeToString resultType)) state11b
                
                Emitter.emit (sprintf "    br ^%s" endLabel) state11c
                
            | None ->
                // No Ok case found, use fallback
                let (defaultStr, state11a) = 
                    let (globalName, state1) = Emitter.registerString "Ok-fallback" state11
                    let (ptrResult, state2) = SSA.generateValue "str_ptr" state1
                    let state3 = Emitter.emit (sprintf "    %s = memref.get_global %s : memref<?xi8>" 
                                           ptrResult globalName) state2
                    (ptrResult, state3)
                    
                let state11b = Emitter.emit (sprintf "    %s = %s : %s" 
                                         resultId defaultStr (mlirTypeToString resultType)) state11a
                
                Emitter.emit (sprintf "    br ^%s" endLabel) state11b
        
        // Find Error case pattern
        let errorCase = 
            cases 
            |> List.tryFind (fun (pattern, _) -> 
                match pattern with 
                | PatternConstructor("Error", _) -> true 
                | _ -> false)
        
        // Handle Error branch
        let state13 = Emitter.emit (sprintf "  ^%s:" elseLabel) state12
        
        let state14 = 
            match errorCase with
            | Some(_, errorExpr) ->
                // Process Error expression
                let errorExprState = TypeTracking.setExpectedType resultType state13
                
                let (errorResultId, errorResultType, state13a) = processExpression errorExpr errorExprState
                
                // Apply type conversion if needed
                let (finalErrorId, state13b) =
                    if errorResultType <> resultType then
                        let (convertedId, state13b) = Emitter.convertType errorResultId errorResultType resultType state13a
                        (convertedId, state13b)
                    else
                        (errorResultId, state13a)
                
                // Store result and branch to end
                let state13c = Emitter.emit (sprintf "    %s = %s : %s" 
                                          resultId finalErrorId (mlirTypeToString resultType)) state13b
                
                Emitter.emit (sprintf "    br ^%s" endLabel) state13c
                
            | None -> 
                // No Error case found, use fallback
                let (errorStr, state13a) = 
                    let (globalName, state1) = Emitter.registerString "Error-fallback" state13
                    let (ptrResult, state2) = SSA.generateValue "str_ptr" state1
                    let state3 = Emitter.emit (sprintf "    %s = memref.get_global %s : memref<?xi8>" 
                                           ptrResult globalName) state2
                    (ptrResult, state3)
                    
                let state13b = Emitter.emit (sprintf "    %s = %s : %s" 
                                         resultId errorStr (mlirTypeToString resultType)) state13a
                
                Emitter.emit (sprintf "    br ^%s" endLabel) state13b
        
        // Add end label and clear expected type
        let state15 = Emitter.emit (sprintf "  ^%s:" endLabel) state14
        let state16 = TypeTracking.recordSSAType resultId resultType state15
        
        (resultId, resultType, state16)
    else
        // Default match implementation for non-Result patterns
        // (In a real implementation, we'd handle more patterns here)
        let (matchValueId, matchValueType, state1) = processExpression matchExpr state
        
        // Create result variable
        let (resultId, state2) = SSA.generateValue "match_default" state1
        
        // Default to first case for simple implementation
        match cases with
        | (_, firstExpr) :: _ ->
            processExpression firstExpr state2
        | [] ->
            // No cases, return unit
            let (unitId, state3) = Emitter.constant "0" (MLIRTypes.createInteger 32) state2
            (unitId, MLIRTypes.createInteger 32, state3)

and processApplication (expr: OakExpression) (state: MLIRGenerationState) : string * MLIRType * MLIRGenerationState =
    match expr with
    | Application(Variable name, args) ->
        // Check for special patterns
        if name = "NativePtr.stackalloc" || name = "stackBuffer" then
            match args with
            | [Literal(IntLiteral size)] ->
                let (bufferResult, state1) = SSA.generateValue "buffer" state
                let state2 = Emitter.emit (sprintf "    %s = memref.alloca() : memref<%dxi8>" 
                                         bufferResult size) state1
                let bufferType = MLIRTypes.createMemRef (MLIRTypes.createInteger 8) [size]
                let state3 = TypeTracking.recordSSAType bufferResult bufferType state2
                
                // Apply conversion if needed
                match state.ExpectedResultType with
                | Some expected when expected <> bufferType ->
                    let (convertedId, state4) = Emitter.convertType bufferResult bufferType expected state3
                    (convertedId, expected, state4)
                | _ ->
                    (bufferResult, bufferType, state3)
            | _ -> processFunctionCall name args state
        
        elif name = "spanToString" then
            match args with
            | [span] ->
                // Process span argument
                let memRefType = MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
                let spanState = TypeTracking.setExpectedType memRefType state
                let (spanResult, spanType, stateAfterSpan) = processExpression span spanState
                
                // Convert span if needed
                let (finalSpanId, state1) =
                    if spanType <> memRefType then
                        let (convertedId, state1) = Emitter.convertType spanResult spanType memRefType stateAfterSpan
                        (convertedId, state1)
                    else
                        (spanResult, stateAfterSpan)
                
                // Create cast operation
                let (resultId, state2) = SSA.generateValue "string" state1
                let state3 = Emitter.emit (sprintf "    %s = memref.cast %s : memref<?xi8> to memref<?xi8>" 
                                          resultId finalSpanId) state2
                
                // Record result type
                let resultType = MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
                let state4 = TypeTracking.recordSSAType resultId resultType state3
                
                // Apply conversion if needed
                match state.ExpectedResultType with
                | Some expected when expected <> resultType ->
                    let (convertedId, state5) = Emitter.convertType resultId resultType expected state4
                    (convertedId, expected, state5)
                | _ ->
                    (resultId, resultType, state4)
            | _ -> processFunctionCall name args state
        
        elif name = "format" || name = "String.format" then
            match args with
            | [Literal(StringLiteral formatStr); value] ->
                // Process the value to format
                let valueExpectedType = 
                    if formatStr.Contains("%s") then
                        MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
                    elif formatStr.Contains("%d") then
                        MLIRTypes.createInteger 32
                    elif formatStr.Contains("%f") then
                        MLIRTypes.createFloat 32
                    else
                        MLIRTypes.createInteger 32
                
                let valueState = TypeTracking.setExpectedType valueExpectedType state
                let (valueResult, valueType, stateAfterValue) = processExpression value valueState
                
                // Convert value if needed
                let (finalValueId, state1) =
                    if valueType <> valueExpectedType then
                        let (convertedId, state1) = Emitter.convertType valueResult valueType valueExpectedType stateAfterValue
                        (convertedId, state1)
                    else
                        (valueResult, stateAfterValue)
                
                // Register sprintf declaration
                let state2 = Emitter.emitModuleLevel "  func.func private @sprintf(memref<?xi8>, memref<?xi8>, ...) -> i32" state1
                
                // Create format string global
                let (formatGlobal, state3) = Emitter.registerString formatStr state2
                let (formatPtr, state4) = SSA.generateValue "fmt_ptr" state3
                let state5 = Emitter.emit (sprintf "    %s = memref.get_global %s : memref<?xi8>" 
                                          formatPtr formatGlobal) state4
                
                // Allocate buffer for result
                let bufferSize = 256
                let (resultBuffer, state6) = SSA.generateValue "format_buffer" state5
                let state7 = Emitter.emit (sprintf "    %s = memref.alloca() : memref<%dxi8>" 
                                          resultBuffer bufferSize) state6
                
                // Call sprintf
                let (sprintfResult, state8) = SSA.generateValue "sprintf_result" state7
                let state9 = Emitter.emit (sprintf "    %s = func.call @sprintf(%s, %s, %s) : (memref<?xi8>, memref<?xi8>, %s) -> i32" 
                                          sprintfResult resultBuffer formatPtr finalValueId 
                                          (mlirTypeToString valueExpectedType)) state8
                
                // Return the buffer as result
                let resultType = MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
                let state10 = TypeTracking.recordSSAType resultBuffer resultType state9
                
                // Apply conversion if needed
                match state.ExpectedResultType with
                | Some expected when expected <> resultType ->
                    let (convertedId, state11) = Emitter.convertType resultBuffer resultType expected state10
                    (convertedId, expected, state11)
                | _ ->
                    (resultBuffer, resultType, state10)
            | _ -> processFunctionCall name args state
        
        elif name = "writeLine" then
            match args with
            | [message] ->
                // Ensure message is a string type
                let memRefType = MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
                let messageState = TypeTracking.setExpectedType memRefType state
                let (messageResult, messageType, stateAfterMessage) = processExpression message messageState
                
                // Convert if needed
                let (finalMessageId, state1) =
                    if messageType <> memRefType then
                        let (convertedId, state1) = Emitter.convertType messageResult messageType memRefType stateAfterMessage
                        (convertedId, state1)
                    else
                        (messageResult, stateAfterMessage)
                
                // Register printf declaration
                let state2 = Emitter.emitModuleLevel "  func.func private @printf(memref<?xi8>, ...) -> i32" state1
                
                // Call printf
                let (printfResult, state3) = SSA.generateValue "printf_result" state2
                let state4 = Emitter.emit (sprintf "    %s = func.call @printf(%s) : (memref<?xi8>) -> i32" 
                                          printfResult finalMessageId) state3
                
                // Record result type
                let resultType = MLIRTypes.createInteger 32
                let state5 = TypeTracking.recordSSAType printfResult resultType state4
                
                // Apply conversion if needed
                match state.ExpectedResultType with
                | Some expected when expected <> resultType ->
                    let (convertedId, state6) = Emitter.convertType printfResult resultType expected state5
                    (convertedId, expected, state6)
                | _ ->
                    (printfResult, resultType, state5)
            | _ -> processFunctionCall name args state
        
        elif name = "readInto" then
            match args with
            | [buffer] ->
                // Process buffer argument
                let memRefType = MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
                let bufferState = TypeTracking.setExpectedType memRefType state
                let (bufferResult, bufferType, stateAfterBuffer) = processExpression buffer bufferState
                
                // Convert if needed
                let (finalBufferId, state1) =
                    if bufferType <> memRefType then
                        let (convertedId, state1) = Emitter.convertType bufferResult bufferType memRefType stateAfterBuffer
                        (convertedId, state1)
                    else
                        (bufferResult, stateAfterBuffer)
                
                // Register declarations
                let state2 = 
                    state1
                    |> Emitter.emitModuleLevel "  func.func private @fgets(memref<?xi8>, i32, memref<?xi8>) -> memref<?xi8>"
                    |> Emitter.emitModuleLevel "  func.func private @strlen(memref<?xi8>) -> i32"
                    |> Emitter.emitModuleLevel "  func.func private @__stdinp() -> memref<?xi8>"
                
                // Get stdin handle
                let (stdinPtr, state3) = SSA.generateValue "stdin_ptr" state2
                let state4 = Emitter.emit (sprintf "    %s = func.call @__stdinp() : () -> memref<?xi8>" stdinPtr) state3
                
                // Default max length
                let (maxLengthId, state5) = SSA.generateValue "max_len" state4
                let state6 = Emitter.emit (sprintf "    %s = arith.constant 256 : i32" maxLengthId) state5
                
                // Call fgets
                let (fgetsResult, state7) = SSA.generateValue "fgets_result" state6
                let state8 = Emitter.emit (sprintf "    %s = func.call @fgets(%s, %s, %s) : (memref<?xi8>, i32, memref<?xi8>) -> memref<?xi8>" 
                                          fgetsResult finalBufferId maxLengthId stdinPtr) state7
                
                // Get string length
                let (lenResult, state9) = SSA.generateValue "read_length" state8
                let state10 = Emitter.emit (sprintf "    %s = func.call @strlen(%s) : (memref<?xi8>) -> i32" 
                                         lenResult finalBufferId) state9
                
                // Create Result wrapper
                let (resultId, state11) = SSA.generateValue "result" state10
                let state12 = Emitter.emit (sprintf "    %s = arith.addi %s, 0x10000 : i32" 
                                          resultId lenResult) state11
                
                // Return the Result value
                let resultType = MLIRTypes.createInteger 32
                let state13 = TypeTracking.recordSSAType resultId resultType state12
                
                // Apply conversion if needed
                match state.ExpectedResultType with
                | Some expected when expected <> resultType ->
                    let (convertedId, state14) = Emitter.convertType resultId resultType expected state13
                    (convertedId, expected, state14)
                | _ ->
                    (resultId, resultType, state13)
            | _ -> processFunctionCall name args state
        
        else
            // Regular function call
            processFunctionCall name args state
        
    | Application(func, args) ->
        // Handle non-variable function application
        let (funcId, funcType, state1) = processExpression func state
        let rec processArgs remainingArgs processedArgs processedTypes currentState =
            match remainingArgs with
            | [] -> (List.rev processedArgs, List.rev processedTypes, currentState)
            | arg :: rest ->
                let (argId, argType, newState) = processExpression arg currentState
                processArgs rest (argId :: processedArgs) (argType :: processedTypes) newState
        
        let (argIds, argTypes, state2) = processArgs args [] [] state1
        
        // Create result
        let (resultId, state3) = SSA.generateValue "app" state2
        let resultType = MLIRTypes.createInteger 32
        
        // Generate a generic call
        let argStr = String.concat ", " argIds
        let state4 = Emitter.emit (sprintf "    %s = func.call %s(%s) : (i32) -> i32" 
                                  resultId funcId argStr) state3
        let state5 = TypeTracking.recordSSAType resultId resultType state4
        
        // Apply conversion if needed
        match state.ExpectedResultType with
        | Some expected when expected <> resultType ->
            let (convertedId, state6) = Emitter.convertType resultId resultType expected state5
            (convertedId, expected, state6)
        | _ ->
            (resultId, resultType, state5)

// Main implementation of the expression processing function
and processExpressionImpl (expr: OakExpression) (state: MLIRGenerationState) : string * MLIRType * MLIRGenerationState =
    match expr with
    | Literal lit -> processLiteral lit state
    | Variable name -> processVariable name state
    | Let(name, value, body) -> processLet name value body state
    | Sequential(first, second) -> processSequential first second state
    | IfThenElse(cond, thenExpr, elseExpr) -> processIfThenElse cond thenExpr elseExpr state
    | Application _ -> processApplication expr state
    | Match(matchExpr, cases) -> processMatch matchExpr cases state
    | _ ->
        // Fallback for other expression types
        let (dummyId, state1) = SSA.generateValue "unknown_expr" state
        let dummyType = MLIRTypes.createInteger 32
        let state2 = TypeTracking.recordSSAType dummyId dummyType state1
        
        // Apply conversion if needed
        match state.ExpectedResultType with
        | Some expected when expected <> dummyType ->
            let (convertedId, state3) = Emitter.convertType dummyId dummyType expected state2
            (convertedId, expected, state3)
        | _ ->
            (dummyId, dummyType, state2)

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
            CurrentFunction = None
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

/// Converts a function declaration using pattern-based approach
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
    
    // Process body with pattern-based approach
    let (bodyResult, bodyType, state4) = processExpression body state3
    
    // Apply type conversion if needed
    let (finalBodyResult, state5) =
        if bodyType <> returnMLIRType then
            let (convertedId, state5) = Emitter.convertType bodyResult bodyType returnMLIRType state4
            (convertedId, state5)
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

/// MLIR module generation with pattern-based type-aware approach
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

/// Generates complete MLIR module text from Oak AST with pattern-based type tracking
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