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
                // Integer to memory reference conversion
                // This typically involves creating a string representation
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

/// Defines a type-aware pattern-matching system for MLIR generation
module TypedPatterns =
    /// Represents a type-aware MLIR generation pattern
    type MLIRPattern = {
        /// Pattern name for diagnostic purposes
        Name: string
        /// Description of what the pattern matches
        Description: string
        /// Function to check if an expression matches this pattern
        Matcher: OakExpression -> bool
        /// Generator function that produces MLIR with type awareness
        Generator: OakExpression -> MLIRType option -> MLIRGenerationState -> string * MLIRType * MLIRGenerationState
    }

       
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
    
    /// Create a Result match pattern that properly handles Ok/Error pattern matching
    let resultMatchPattern : MLIRPattern = {
        Name = "result-match-pattern"
        Description = "Match expression on Result type with Ok/Error cases"
        Matcher = function
            | Match(Application(Variable funcName, _), cases) when isResultReturningFunction funcName ->
                // Check if it has Ok/Error patterns
                cases |> List.exists (fun (pattern, _) ->
                    match pattern with
                    | PatternConstructor("Ok", _) -> true
                    | _ -> false) &&
                cases |> List.exists (fun (pattern, _) ->
                    match pattern with
                    | PatternConstructor("Error", _) -> true
                    | _ -> false)
            | _ -> false
        Generator = fun expr expectedType state ->
            match expr with
            | Match(matchExpr, cases) ->
                // Process the match expression
                let rec processExpression (expr: OakExpression) (state: MLIRGenerationState) =
                    // Dispatch to appropriate pattern handler based on expression type
                    let pattern = 
                        patternRegistry 
                        |> List.tryFind (fun p -> p.Matcher expr)
                    
                    match pattern with
                    | Some p -> p.Generator expr (TypeTracking.getExpectedType state) state
                    | None -> 
                        // Fallback for expressions without specific patterns
                        processExpressionFallback expr state
                
                and processExpressionFallback (expr: OakExpression) (state: MLIRGenerationState) =
                    match expr with
                    | Literal lit ->
                        let literalType = TypeTracking.inferLiteralType lit
                        match lit with
                        | IntLiteral value ->
                            let (resultId, newState) = Emitter.constant (string value) literalType state
                            (resultId, literalType, newState)
                        | StringLiteral value ->
                            let (globalName, state1) = Emitter.registerString value state
                            let (ptrResult, state2) = SSA.generateValue "str_ptr" state1
                            let state3 = Emitter.emit (sprintf "    %s = memref.get_global %s : memref<?xi8>" 
                                                      ptrResult globalName) state2
                            (ptrResult, literalType, state3)
                        | _ -> 
                            let (resultId, newState) = Emitter.constant "0" literalType state
                            (resultId, literalType, newState)
                    
                    | Variable name ->
                        match SSA.lookupVariable name state with
                        | Some value -> 
                            // Get type if available
                            let valueType = 
                                match TypeTracking.getSSAType value state with
                                | Some t -> t
                                | None -> MLIRTypes.createInteger 32 // Default if unknown
                            (value, valueType, state)
                        | None -> 
                            // Unknown variable
                            let (dummyValue, state1) = SSA.generateValue "unknown" state
                            let intType = MLIRTypes.createInteger 32
                            let state2 = TypeTracking.recordSSAType dummyValue intType state1
                            (dummyValue, intType, state2)
                    
                    | _ ->
                        // Default fallback
                        let (dummyValue, state1) = SSA.generateValue "fallback" state
                        let intType = MLIRTypes.createInteger 32
                        let state2 = TypeTracking.recordSSAType dummyValue intType state1
                        (dummyValue, intType, state2)
                
                // Process the match expression
                let (matchValueId, matchValueType, stateAfterMatchExpr) = processExpression matchExpr state
                let (resultId, stateWithResultVar) = SSA.generateValue "match_result" stateAfterMatchExpr
                
                // Determine the result type - either expected type or memref by default
                let resultType = 
                    match expectedType with
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
                let state1 = 
                    stateWithResultVar
                    |> Emitter.emitModuleLevel "  func.func private @is_ok_result(i32) -> i1"
                    |> Emitter.emitModuleLevel "  func.func private @extract_result_length(i32) -> i32"
                    |> Emitter.emitModuleLevel "  func.func private @create_span(memref<?xi8>, i32) -> memref<?xi8>"
                
                // Check if result is Ok or Error
                let (isOkId, state2) = SSA.generateValue "is_ok" state1
                let state3 = Emitter.emit (sprintf "    %s = func.call @is_ok_result(%s) : (i32) -> i1" 
                                        isOkId matchValueId) state2
                let boolType = MLIRTypes.createInteger 1
                let state4 = TypeTracking.recordSSAType isOkId boolType state3
                
                // Create branch labels
                let (thenId, state5) = SSA.generateValue "then" state4
                let (elseId, state6) = SSA.generateValue "else" state5
                let (endId, state7) = SSA.generateValue "end" state6
                
                let thenLabel = thenId.TrimStart('%')
                let elseLabel = elseId.TrimStart('%')
                let endLabel = endId.TrimStart('%')
                
                let state8 = Emitter.emit (sprintf "    cond_br %s, ^%s, ^%s" 
                                        isOkId thenLabel elseLabel) state7
                
                // Find Ok case pattern
                let okCase = 
                    cases |> List.tryFind (fun (pattern, _) -> 
                        match pattern with 
                        | PatternConstructor("Ok", _) -> true
                        | _ -> false)
                
                // Handle Ok branch
                let state9 = Emitter.emit (sprintf "  ^%s:" thenLabel) state8
                
                let state10 = 
                    match okCase with
                    | Some (PatternConstructor("Ok", [PatternVariable lengthName]), okExpr) ->
                        // Extract length from result
                        let (lengthId, stateWithLength) = SSA.generateValue "length" state9
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
                        // Set expected type for the expression
                        let okExprStateWithBuffer = 
                            stateWithSpanType 
                            |> SSA.bindVariable "span" spanId
                            |> TypeTracking.setExpectedType resultType
                        
                        // Process the Ok expression
                        let (okResultId, okResultType, stateAfterExpr) = 
                            processExpression okExpr okExprStateWithBuffer
                        
                        // Apply type conversion if needed to ensure result type matches expected
                        let (finalResultId, stateAfterConversion) =
                            if okResultType <> resultType then
                                let (convertedId, convertedState) = 
                                    Emitter.convertType okResultId okResultType resultType stateAfterExpr
                                (convertedId, convertedState)
                            else
                                (okResultId, stateAfterExpr)
                        
                        // Store result and branch to end
                        let stateWithResultStore = 
                            Emitter.emit (sprintf "    %s = %s : %s" 
                                        resultId finalResultId (mlirTypeToString resultType)) stateAfterConversion
                        
                        Emitter.emit (sprintf "    br ^%s" endLabel) stateWithResultStore
                        
                    | Some (_, okExpr) ->
                        // Handle Ok case without pattern variable
                        // Set expected type for the expression
                        let okExprState = TypeTracking.setExpectedType resultType state9
                        
                        // Process the Ok expression
                        let (okResultId, okResultType, stateAfterExpr) = 
                            processExpression okExpr okExprState
                        
                        // Apply type conversion if needed
                        let (finalResultId, stateAfterConversion) =
                            if okResultType <> resultType then
                                let (convertedId, convertedState) = 
                                    Emitter.convertType okResultId okResultType resultType stateAfterExpr
                                (convertedId, convertedState)
                            else
                                (okResultId, stateAfterExpr)
                        
                        // Store result and branch to end
                        let stateWithResultStore = 
                            Emitter.emit (sprintf "    %s = %s : %s" 
                                        resultId finalResultId (mlirTypeToString resultType)) stateAfterConversion
                        
                        Emitter.emit (sprintf "    br ^%s" endLabel) stateWithResultStore
                        
                    | None ->
                        // No Ok case found, use fallback
                        let (defaultStr, stateWithStr) = 
                            let (globalName, state1) = Emitter.registerString "Ok-fallback" state9
                            let (ptrResult, state2) = SSA.generateValue "str_ptr" state1
                            let state3 = Emitter.emit (sprintf "    %s = memref.get_global %s : memref<?xi8>" 
                                                     ptrResult globalName) state2
                            (ptrResult, state3)
                            
                        let stateWithStore = 
                            Emitter.emit (sprintf "    %s = %s : %s" 
                                         resultId defaultStr (mlirTypeToString resultType)) stateWithStr
                        
                        Emitter.emit (sprintf "    br ^%s" endLabel) stateWithStore
                
                // Find Error case pattern
                let errorCase = 
                    cases 
                    |> List.tryFind (fun (pattern, _) -> 
                        match pattern with 
                        | PatternConstructor("Error", _) -> true 
                        | _ -> false)
                
                // Handle Error branch
                let state11 = Emitter.emit (sprintf "  ^%s:" elseLabel) state10
                
                let state12 = 
                    match errorCase with
                    | Some(_, errorExpr) ->
                        // Process Error expression with expected type
                        let errorExprState = TypeTracking.setExpectedType resultType state11
                        
                        // Process the Error expression
                        let (errorResultId, errorResultType, stateAfterErrorExpr) = 
                            processExpression errorExpr errorExprState
                        
                        // Apply type conversion if needed
                        let (finalErrorId, stateAfterConversion) =
                            if errorResultType <> resultType then
                                let (convertedId, convertedState) = 
                                    Emitter.convertType errorResultId errorResultType resultType stateAfterErrorExpr
                                (convertedId, convertedState)
                            else
                                (errorResultId, stateAfterErrorExpr)
                        
                        // Store result and branch to end
                        let stateWithResultStore = 
                            Emitter.emit (sprintf "    %s = %s : %s" 
                                        resultId finalErrorId (mlirTypeToString resultType)) stateAfterConversion
                        
                        Emitter.emit (sprintf "    br ^%s" endLabel) stateWithResultStore
                        
                    | None -> 
                        // No Error case found, use fallback
                        let (errorStr, stateWithStr) = 
                            let (globalName, state1) = Emitter.registerString "Error-fallback" state11
                            let (ptrResult, state2) = SSA.generateValue "str_ptr" state1
                            let state3 = Emitter.emit (sprintf "    %s = memref.get_global %s : memref<?xi8>" 
                                                     ptrResult globalName) state2
                            (ptrResult, state3)
                            
                        let stateWithStore = 
                            Emitter.emit (sprintf "    %s = %s : %s" 
                                         resultId errorStr (mlirTypeToString resultType)) stateWithStr
                        
                        Emitter.emit (sprintf "    br ^%s" endLabel) stateWithStore
                
                // Add end label and clear expected type
                let state13 = Emitter.emit (sprintf "  ^%s:" endLabel) state12
                let finalState = TypeTracking.recordSSAType resultId resultType state13
                
                (resultId, resultType, finalState)
            | _ ->
                // Should never happen due to pattern check
                let (dummyId, state1) = SSA.generateValue "invalid" state
                let dummyType = MLIRTypes.createInteger 32
                let state2 = TypeTracking.recordSSAType dummyId dummyType state1
                (dummyId, dummyType, state2)
    }
    
    /// Stack allocation pattern
    let stackAllocPattern : MLIRPattern = {
        Name = "stack-alloc-pattern"
        Description = "Stack allocation of fixed-size buffer using NativePtr.stackalloc or stackBuffer"
        Matcher = function
            | Application(Variable fname, [Literal(IntLiteral _)]) 
                when fname = "NativePtr.stackalloc" || fname = "stackBuffer" -> true
            | _ -> false
        Generator = fun expr expectedType state ->
            match expr with
            | Application(_, [Literal(IntLiteral size)]) ->
                let (bufferResult, state1) = SSA.generateValue "buffer" state
                let state2 = Emitter.emit (sprintf "    %s = memref.alloca() : memref<%dxi8>" 
                                         bufferResult size) state1
                let bufferType = MLIRTypes.createMemRef (MLIRTypes.createInteger 8) [size]
                let state3 = TypeTracking.recordSSAType bufferResult bufferType state2
                (bufferResult, bufferType, state3)
            | _ ->
                // Should never happen due to pattern check
                let (dummyId, state1) = SSA.generateValue "invalid" state
                let dummyType = MLIRTypes.createInteger 32
                let state2 = TypeTracking.recordSSAType dummyId dummyType state1
                (dummyId, dummyType, state2)
    }
    
    /// String format pattern
    let stringFormatPattern : MLIRPattern = {
        Name = "string-format-pattern"
        Description = "String formatting with String.format or format function"
        Matcher = function
            | Application(Variable fname, [Literal(StringLiteral _); _])
                when fname = "format" || fname = "String.format" -> true
            | _ -> false
        Generator = fun expr expectedType state ->
            match expr with
            | Application(_, [Literal(StringLiteral formatStr); value]) ->
                let rec processExpression (expr: OakExpression) (state: MLIRGenerationState) =
                    let pattern = 
                        patternRegistry 
                        |> List.tryFind (fun p -> p.Matcher expr)
                    
                    match pattern with
                    | Some p -> p.Generator expr (TypeTracking.getExpectedType state) state
                    | None -> 
                        processExpressionFallback expr state
                
                and processExpressionFallback (expr: OakExpression) (state: MLIRGenerationState) =
                    match expr with
                    | Literal lit ->
                        let literalType = TypeTracking.inferLiteralType lit
                        match lit with
                        | IntLiteral value ->
                            let (resultId, newState) = Emitter.constant (string value) literalType state
                            (resultId, literalType, newState)
                        | StringLiteral value ->
                            let (globalName, state1) = Emitter.registerString value state
                            let (ptrResult, state2) = SSA.generateValue "str_ptr" state1
                            let state3 = Emitter.emit (sprintf "    %s = memref.get_global %s : memref<?xi8>" 
                                                      ptrResult globalName) state2
                            (ptrResult, literalType, state3)
                        | _ -> 
                            let (resultId, newState) = Emitter.constant "0" literalType state
                            (resultId, literalType, newState)
                    
                    | Variable name ->
                        match SSA.lookupVariable name state with
                        | Some value -> 
                            let valueType = 
                                match TypeTracking.getSSAType value state with
                                | Some t -> t
                                | None -> MLIRTypes.createInteger 32
                            (value, valueType, state)
                        | None -> 
                            let (dummyValue, state1) = SSA.generateValue "unknown" state
                            let intType = MLIRTypes.createInteger 32
                            let state2 = TypeTracking.recordSSAType dummyValue intType state1
                            (dummyValue, intType, state2)
                    
                    | _ ->
                        let (dummyValue, state1) = SSA.generateValue "fallback" state
                        let intType = MLIRTypes.createInteger 32
                        let state2 = TypeTracking.recordSSAType dummyValue intType state1
                        (dummyValue, intType, state2)
                
                // Process the value to format - ensure it's a memory reference if string format contains %s
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
                
                // Apply type conversion if needed
                let (finalValueId, stateAfterConversion) =
                    if valueType <> valueExpectedType then
                        let (convertedId, convertedState) = 
                            Emitter.convertType valueResult valueType valueExpectedType stateAfterValue
                        (convertedId, convertedState)
                    else
                        (valueResult, stateAfterValue)
                
                // Register sprintf declaration
                let state1 = Emitter.emitModuleLevel "  func.func private @sprintf(memref<?xi8>, memref<?xi8>, ...) -> i32" stateAfterConversion
                
                // Create format string global
                let (formatGlobal, state2) = Emitter.registerString formatStr state1
                let (formatPtr, state3) = SSA.generateValue "fmt_ptr" state2
                let state4 = Emitter.emit (sprintf "    %s = memref.get_global %s : memref<?xi8>" 
                                          formatPtr formatGlobal) state3
                
                // Allocate buffer for result
                let bufferSize = 256 // Default reasonable size for formatted strings
                let (resultBuffer, state5) = SSA.generateValue "format_buffer" state4
                let state6 = Emitter.emit (sprintf "    %s = memref.alloca() : memref<%dxi8>" 
                                          resultBuffer bufferSize) state5
                
                // Call sprintf
                let (sprintfResult, state7) = SSA.generateValue "sprintf_result" state6
                let state8 = Emitter.emit (sprintf "    %s = func.call @sprintf(%s, %s, %s) : (memref<?xi8>, memref<?xi8>, %s) -> i32" 
                                          sprintfResult resultBuffer formatPtr finalValueId 
                                          (mlirTypeToString valueExpectedType)) state7
                
                // Return the buffer as result
                let resultType = MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
                let state9 = TypeTracking.recordSSAType resultBuffer resultType state8
                
                (resultBuffer, resultType, state9)
                
            | _ ->
                // Should never happen due to pattern check
                let (dummyId, state1) = SSA.generateValue "invalid" state
                let dummyType = MLIRTypes.createInteger 32
                let state2 = TypeTracking.recordSSAType dummyId dummyType state1
                (dummyId, dummyType, state2)
    }
  
    /// Console writeLine pattern
    let writeLinePattern : MLIRPattern = {
        Name = "writeline-pattern"
        Description = "Console.writeLine function for output"
        Matcher = function
            | Application(Variable "writeLine", [_]) -> true
            | _ -> false
        Generator = fun expr expectedType state ->
            match expr with
            | Application(_, [message]) ->
                let rec processExpression (expr: OakExpression) (state: MLIRGenerationState) =
                    let pattern = 
                        patternRegistry 
                        |> List.tryFind (fun p -> p.Matcher expr)
                    
                    match pattern with
                    | Some p -> p.Generator expr (TypeTracking.getExpectedType state) state
                    | None -> 
                        processExpressionFallback expr state
                
                and processExpressionFallback (expr: OakExpression) (state: MLIRGenerationState) =
                    match expr with
                    | Literal lit ->
                        let literalType = TypeTracking.inferLiteralType lit
                        match lit with
                        | IntLiteral value ->
                            let (resultId, newState) = Emitter.constant (string value) literalType state
                            (resultId, literalType, newState)
                        | StringLiteral value ->
                            let (globalName, state1) = Emitter.registerString value state
                            let (ptrResult, state2) = SSA.generateValue "str_ptr" state1
                            let state3 = Emitter.emit (sprintf "    %s = memref.get_global %s : memref<?xi8>" 
                                                      ptrResult globalName) state2
                            (ptrResult, literalType, state3)
                        | _ -> 
                            let (resultId, newState) = Emitter.constant "0" literalType state
                            (resultId, literalType, newState)
                    
                    | Variable name ->
                        match SSA.lookupVariable name state with
                        | Some value -> 
                            let valueType = 
                                match TypeTracking.getSSAType value state with
                                | Some t -> t
                                | None -> MLIRTypes.createInteger 32
                            (value, valueType, state)
                        | None -> 
                            let (dummyValue, state1) = SSA.generateValue "unknown" state
                            let intType = MLIRTypes.createInteger 32
                            let state2 = TypeTracking.recordSSAType dummyValue intType state1
                            (dummyValue, intType, state2)
                    
                    | _ ->
                        let (dummyValue, state1) = SSA.generateValue "fallback" state
                        let intType = MLIRTypes.createInteger 32
                        let state2 = TypeTracking.recordSSAType dummyValue intType state1
                        (dummyValue, intType, state2)
                        
                // Ensure message is a string type
                let memRefType = MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
                let messageState = TypeTracking.setExpectedType memRefType state
                let (messageResult, messageType, stateAfterMessage) = processExpression message messageState
                
                // Apply type conversion if needed
                let (finalMessageId, stateAfterConversion) =
                    if messageType <> memRefType then
                        let (convertedId, convertedState) = 
                            Emitter.convertType messageResult messageType memRefType stateAfterMessage
                        (convertedId, convertedState)
                    else
                        (messageResult, stateAfterMessage)
                
                // Register printf declaration
                let state1 = Emitter.emitModuleLevel "  func.func private @printf(memref<?xi8>, ...) -> i32" stateAfterConversion
                
                // Call printf
                let (printfResult, state2) = SSA.generateValue "printf_result" state1
                let state3 = Emitter.emit (sprintf "    %s = func.call @printf(%s) : (memref<?xi8>) -> i32" 
                                          printfResult finalMessageId) state2
                
                // Record result type
                let resultType = MLIRTypes.createInteger 32
                let state4 = TypeTracking.recordSSAType printfResult resultType state3
                
                (printfResult, resultType, state4)
                
            | _ ->
                // Should never happen due to pattern check
                let (dummyId, state1) = SSA.generateValue "invalid" state
                let dummyType = MLIRTypes.createInteger 32
                let state2 = TypeTracking.recordSSAType dummyId dummyType state1
                (dummyId, dummyType, state2)
    }
    
    /// readInto pattern - Result-returning function for reading into buffer
    let readIntoPattern : MLIRPattern = {
        Name = "readinto-pattern"
        Description = "readInto function that returns Result<int, string>"
        Matcher = function
            | Application(Variable "readInto", [_]) -> true
            | _ -> false
        Generator = fun expr expectedType state ->
            match expr with
            | Application(_, [buffer]) ->
                let rec processExpression (expr: OakExpression) (state: MLIRGenerationState) =
                    let pattern = 
                        patternRegistry 
                        |> List.tryFind (fun p -> p.Matcher expr)
                    
                    match pattern with
                    | Some p -> p.Generator expr (TypeTracking.getExpectedType state) state
                    | None -> 
                        processExpressionFallback expr state
                
                and processExpressionFallback (expr: OakExpression) (state: MLIRGenerationState) =
                    match expr with
                    | Literal lit ->
                        let literalType = TypeTracking.inferLiteralType lit
                        match lit with
                        | IntLiteral value ->
                            let (resultId, newState) = Emitter.constant (string value) literalType state
                            (resultId, literalType, newState)
                        | StringLiteral value ->
                            let (globalName, state1) = Emitter.registerString value state
                            let (ptrResult, state2) = SSA.generateValue "str_ptr" state1
                            let state3 = Emitter.emit (sprintf "    %s = memref.get_global %s : memref<?xi8>" 
                                                      ptrResult globalName) state2
                            (ptrResult, literalType, state3)
                        | _ -> 
                            let (resultId, newState) = Emitter.constant "0" literalType state
                            (resultId, literalType, newState)
                    
                    | Variable name ->
                        match SSA.lookupVariable name state with
                        | Some value -> 
                            let valueType = 
                                match TypeTracking.getSSAType value state with
                                | Some t -> t
                                | None -> MLIRTypes.createInteger 32
                            (value, valueType, state)
                        | None -> 
                            let (dummyValue, state1) = SSA.generateValue "unknown" state
                            let intType = MLIRTypes.createInteger 32
                            let state2 = TypeTracking.recordSSAType dummyValue intType state1
                            (dummyValue, intType, state2)
                    
                    | _ ->
                        let (dummyValue, state1) = SSA.generateValue "fallback" state
                        let intType = MLIRTypes.createInteger 32
                        let state2 = TypeTracking.recordSSAType dummyValue intType state1
                        (dummyValue, intType, state2)
                
                // Process buffer argument
                let memRefType = MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
                let bufferState = TypeTracking.setExpectedType memRefType state
                let (bufferResult, bufferType, stateAfterBuffer) = processExpression buffer bufferState
                
                // Apply type conversion if needed
                let (finalBufferId, stateAfterConversion) =
                    if bufferType <> memRefType then
                        let (convertedId, convertedState) = 
                            Emitter.convertType bufferResult bufferType memRefType stateAfterBuffer
                        (convertedId, convertedState)
                    else
                        (bufferResult, stateAfterBuffer)
                
                // Register scanf/fgets declarations
                let state1 = 
                    stateAfterConversion
                    |> Emitter.emitModuleLevel "  func.func private @fgets(memref<?xi8>, i32, memref<?xi8>) -> memref<?xi8>"
                    |> Emitter.emitModuleLevel "  func.func private @strlen(memref<?xi8>) -> i32"
                    |> Emitter.emitModuleLevel "  func.func private @__stdinp() -> memref<?xi8>"
                
                // Get stdin handle
                let (stdinPtr, state2) = SSA.generateValue "stdin_ptr" state1
                let state3 = Emitter.emit (sprintf "    %s = func.call @__stdinp() : () -> memref<?xi8>" stdinPtr) state2
                
                // Default max length
                let (maxLengthId, state4) = SSA.generateValue "max_len" state3
                let state5 = Emitter.emit (sprintf "    %s = arith.constant 256 : i32" maxLengthId) state4
                
                // Call fgets
                let (fgetsResult, state6) = SSA.generateValue "fgets_result" state5
                let state7 = Emitter.emit (sprintf "    %s = func.call @fgets(%s, %s, %s) : (memref<?xi8>, i32, memref<?xi8>) -> memref<?xi8>" 
                                          fgetsResult finalBufferId maxLengthId stdinPtr) state6
                
                // Get string length
                let (lenResult, state8) = SSA.generateValue "read_length" state7
                let state9 = Emitter.emit (sprintf "    %s = func.call @strlen(%s) : (memref<?xi8>) -> i32" 
                                         lenResult finalBufferId) state8
                
                // Create Result wrapper - typically 1 (success) plus the length
                let (resultId, state10) = SSA.generateValue "result" state9
                let state11 = Emitter.emit (sprintf "    %s = arith.addi %s, 0x10000 : i32" 
                                          resultId lenResult) state10
                
                // Return the Result value (i32 encoding both success status and length)
                let resultType = MLIRTypes.createInteger 32
                let state12 = TypeTracking.recordSSAType resultId resultType state11
                
                (resultId, resultType, state12)
                
            | _ ->
                // Should never happen due to pattern check
                let (dummyId, state1) = SSA.generateValue "invalid" state
                let dummyType = MLIRTypes.createInteger 32
                let state2 = TypeTracking.recordSSAType dummyId dummyType state1
                (dummyId, dummyType, state2)
    }
    
    /// Variable pattern for basic variable lookup with type awareness
    let variablePattern : MLIRPattern = {
        Name = "variable-pattern"
        Description = "Basic variable reference with type tracking"
        Matcher = function
            | Variable _ -> true
            | _ -> false
        Generator = fun expr expectedType state ->
            match expr with
            | Variable name ->
                match SSA.lookupVariable name state with
                | Some value -> 
                    // Get type if available
                    let valueType = 
                        match TypeTracking.getSSAType value state with
                        | Some t -> t
                        | None -> MLIRTypes.createInteger 32 // Default if unknown
                    
                    // Check if we need to apply type conversion for expected type
                    match expectedType with
                    | Some expected when expected <> valueType ->
                        let (convertedId, convertedState) = 
                            Emitter.convertType value valueType expected state
                        (convertedId, expected, convertedState)
                    | _ ->
                        (value, valueType, state)
                | None -> 
                    // Unknown variable
                    let (dummyValue, state1) = SSA.generateValue "unknown" state
                    let intType = MLIRTypes.createInteger 32
                    let state2 = TypeTracking.recordSSAType dummyValue intType state1
                    
                    // Apply conversion if needed
                    match expectedType with
                    | Some expected when expected <> intType ->
                        let (convertedId, convertedState) = 
                            Emitter.convertType dummyValue intType expected state2
                        (convertedId, expected, convertedState)
                    | _ ->
                        (dummyValue, intType, state2)
            | _ ->
                // Should never happen due to pattern check
                let (dummyId, state1) = SSA.generateValue "invalid" state
                let dummyType = MLIRTypes.createInteger 32
                let state2 = TypeTracking.recordSSAType dummyId dummyType state1
                (dummyId, dummyType, state2)
    }
    
    /// Literal pattern for handling literals with proper types
    let literalPattern : MLIRPattern = {
        Name = "literal-pattern"
        Description = "Literal values with type awareness"
        Matcher = function
            | Literal _ -> true
            | _ -> false
        Generator = fun expr expectedType state ->
            match expr with
            | Literal lit ->
                let literalType = TypeTracking.inferLiteralType lit
                
                match lit with
                | IntLiteral value ->
                    let (resultId, state1) = Emitter.constant (string value) literalType state
                    
                    // Apply conversion if needed
                    match expectedType with
                    | Some expected when expected <> literalType ->
                        let (convertedId, convertedState) = 
                            Emitter.convertType resultId literalType expected state1
                        (convertedId, expected, convertedState)
                    | _ ->
                        (resultId, literalType, state1)
                
                | FloatLiteral value ->
                    let (resultId, state1) = Emitter.constant (sprintf "%f" value) literalType state
                    
                    // Apply conversion if needed
                    match expectedType with
                    | Some expected when expected <> literalType ->
                        let (convertedId, convertedState) = 
                            Emitter.convertType resultId literalType expected state1
                        (convertedId, expected, convertedState)
                    | _ ->
                        (resultId, literalType, state1)
                
                | BoolLiteral value ->
                    let (resultId, state1) = Emitter.constant (if value then "1" else "0") literalType state
                    
                    // Apply conversion if needed
                    match expectedType with
                    | Some expected when expected <> literalType ->
                        let (convertedId, convertedState) = 
                            Emitter.convertType resultId literalType expected state1
                        (convertedId, expected, convertedState)
                    | _ ->
                        (resultId, literalType, state1)
                
                | StringLiteral value ->
                    let (globalName, state1) = Emitter.registerString value state
                    let (ptrResult, state2) = SSA.generateValue "str_ptr" state1
                    let state3 = Emitter.emit (sprintf "    %s = memref.get_global %s : memref<?xi8>" 
                                              ptrResult globalName) state2
                    let state4 = TypeTracking.recordSSAType ptrResult literalType state3
                    
                    // Apply conversion if needed
                    match expectedType with
                    | Some expected when expected <> literalType ->
                        let (convertedId, convertedState) = 
                            Emitter.convertType ptrResult literalType expected state4
                        (convertedId, expected, convertedState)
                    | _ ->
                        (ptrResult, literalType, state4)
                
                | UnitLiteral ->
                    let (resultId, state1) = Emitter.constant "0" (MLIRTypes.createInteger 32) state
                    
                    // Apply conversion if needed
                    match expectedType with
                    | Some expected when expected <> MLIRTypes.createInteger 32 ->
                        let (convertedId, convertedState) = 
                            Emitter.convertType resultId (MLIRTypes.createInteger 32) expected state1
                        (convertedId, expected, convertedState)
                    | _ ->
                        (resultId, MLIRTypes.createInteger 32, state1)
                
                | ArrayLiteral _ ->
                    // Not fully implemented - create a dummy array
                    let (resultId, state1) = SSA.generateValue "array" state
                    let state2 = TypeTracking.recordSSAType resultId literalType state1
                    
                    // Apply conversion if needed
                    match expectedType with
                    | Some expected when expected <> literalType ->
                        let (convertedId, convertedState) = 
                            Emitter.convertType resultId literalType expected state2
                        (convertedId, expected, convertedState)
                    | _ ->
                        (resultId, literalType, state2)
            
            | _ ->
                // Should never happen due to pattern check
                let (dummyId, state1) = SSA.generateValue "invalid" state
                let dummyType = MLIRTypes.createInteger 32
                let state2 = TypeTracking.recordSSAType dummyId dummyType state1
                (dummyId, dummyType, state2)
    }
    
    /// Register all patterns in the "Library of Alexandria"
    let patternRegistry : MLIRPattern list = [
        resultMatchPattern
        stackAllocPattern
        stringFormatPattern
        writeLinePattern
        readIntoPattern
        variablePattern
        literalPattern
    ]
    
    /// Finds a matching pattern for an expression
    let findMatchingPattern (expr: OakExpression) : MLIRPattern option =
        patternRegistry |> List.tryFind (fun pattern -> pattern.Matcher expr)
    
    /// Processes an expression using the pattern library
    let rec processExpression (expr: OakExpression) (state: MLIRGenerationState) : string * MLIRType * MLIRGenerationState =
        // Get expected type from context (if any)
        let expectedType = TypeTracking.getExpectedType state
        
        // Find matching pattern
        let pattern = findMatchingPattern expr
        
        match pattern with
        | Some p -> 
            // Use the pattern's generator
            p.Generator expr expectedType state
        | None -> 
            // Handle expressions without specific patterns
            processExpressionFallback expr state
    
    /// Fallback for expressions without specific patterns
    and processExpressionFallback (expr: OakExpression) (state: MLIRGenerationState) : string * MLIRType * MLIRGenerationState =
        // Get expected type from context (if any)
        let expectedType = TypeTracking.getExpectedType state
        
        match expr with
        | Literal lit ->
            let literalType = TypeTracking.inferLiteralType lit
            match lit with
            | IntLiteral value ->
                let (resultId, newState) = Emitter.constant (string value) literalType state
                
                // Apply conversion if needed
                match expectedType with
                | Some expected when expected <> literalType ->
                    let (convertedId, convertedState) = 
                        Emitter.convertType resultId literalType expected newState
                    (convertedId, expected, convertedState)
                | _ ->
                    (resultId, literalType, newState)
                    
            | StringLiteral value ->
                let (globalName, state1) = Emitter.registerString value state
                let (ptrResult, state2) = SSA.generateValue "str_ptr" state1
                let state3 = Emitter.emit (sprintf "    %s = memref.get_global %s : memref<?xi8>" 
                                          ptrResult globalName) state2
                
                // Apply conversion if needed
                match expectedType with
                | Some expected when expected <> literalType ->
                    let (convertedId, convertedState) = 
                        Emitter.convertType ptrResult literalType expected state3
                    (convertedId, expected, convertedState)
                | _ ->
                    (ptrResult, literalType, state3)
                    
            | _ -> 
                let (resultId, newState) = Emitter.constant "0" literalType state
                
                // Apply conversion if needed
                match expectedType with
                | Some expected when expected <> literalType ->
                    let (convertedId, convertedState) = 
                        Emitter.convertType resultId literalType expected newState
                    (convertedId, expected, convertedState)
                | _ ->
                    (resultId, literalType, newState)
        
        | Variable name ->
            match SSA.lookupVariable name state with
            | Some value -> 
                let valueType = 
                    match TypeTracking.getSSAType value state with
                    | Some t -> t
                    | None -> MLIRTypes.createInteger 32
                    
                // Apply conversion if needed
                match expectedType with
                | Some expected when expected <> valueType ->
                    let (convertedId, convertedState) = 
                        Emitter.convertType value valueType expected state
                    (convertedId, expected, convertedState)
                | _ ->
                    (value, valueType, state)
                    
            | None -> 
                let (dummyValue, state1) = SSA.generateValue "unknown" state
                let intType = MLIRTypes.createInteger 32
                let state2 = TypeTracking.recordSSAType dummyValue intType state1
                
                // Apply conversion if needed
                match expectedType with
                | Some expected when expected <> intType ->
                    let (convertedId, convertedState) = 
                        Emitter.convertType dummyValue intType expected state2
                    (convertedId, expected, convertedState)
                | _ ->
                    (dummyValue, intType, state2)
        
        | Let(name, value, body) ->
            // Process value
            let valueState = TypeTracking.clearExpectedType state
            let (valueId, valueType, state1) = processExpression value valueState
            
            // Bind variable
            let state2 = SSA.bindVariable name valueId state1
            let state3 = TypeTracking.recordSSAType valueId valueType state2
            
            // Process body
            let (bodyId, bodyType, state4) = processExpression body state3
            
            // Apply conversion if needed
            match expectedType with
            | Some expected when expected <> bodyType ->
                let (convertedId, convertedState) = 
                    Emitter.convertType bodyId bodyType expected state4
                (convertedId, expected, convertedState)
            | _ ->
                (bodyId, bodyType, state4)
        
        | Sequential(first, second) ->
            // Process first expression (ignore result)
            let (_, _, state1) = processExpression first state
            
            // Process second expression
            let (secondId, secondType, state2) = processExpression second state1
            
            // Apply conversion if needed
            match expectedType with
            | Some expected when expected <> secondType ->
                let (convertedId, convertedState) = 
                    Emitter.convertType secondId secondType expected state2
                (convertedId, expected, convertedState)
            | _ ->
                (secondId, secondType, state2)
        
        | Application(func, args) ->
            match func with
            | Variable funcName ->
                // Process arguments
                let processArg (accState, accArgs, accTypes) arg =
                    let (argId, argType, newState) = processExpression arg accState
                    (newState, argId :: accArgs, argType :: accTypes)
                
                let (state1, argIds, argTypes) = 
                    List.fold processArg (state, [], []) args
                
                let argIds = List.rev argIds
                let argTypes = List.rev argTypes
                
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
                    match expectedType with
                    | Some expected when expected <> resultType ->
                        let (convertedId, convertedState) = 
                            Emitter.convertType resultId resultType expected state5
                        (convertedId, expected, convertedState)
                    | _ ->
                        (resultId, resultType, state5)
                    
                | Core.XParsec.Foundation.CompilerFailure _ ->
                    // Fallback for unknown functions
                    let resultType = MLIRTypes.createInteger 32
                    let state3 = Emitter.emit (sprintf "    %s = arith.constant 0 : i32 // Unknown function %s" 
                                             resultId funcName) state2
                    let state4 = TypeTracking.recordSSAType resultId resultType state3
                    
                    // Apply conversion if needed
                    match expectedType with
                    | Some expected when expected <> resultType ->
                        let (convertedId, convertedState) = 
                            Emitter.convertType resultId resultType expected state4
                        (convertedId, expected, convertedState)
                    | _ ->
                        (resultId, resultType, state4)
            
            | _ ->
                // Handle non-variable function application
                let (funcId, funcType, state1) = processExpression func state
                
                // Process arguments
                let processArg (accState, accArgs, accTypes) arg =
                    let (argId, argType, newState) = processExpression arg accState
                    (newState, argId :: accArgs, argType :: accTypes)
                
                let (state2, argIds, argTypes) = 
                    List.fold processArg (state1, [], []) args
                
                let argIds = List.rev argIds
                let argTypes = List.rev argTypes
                
                // Create result
                let (resultId, state3) = SSA.generateValue "app" state2
                let resultType = MLIRTypes.createInteger 32
                
                // Generate a generic call
                let argStr = String.concat ", " argIds
                let state4 = Emitter.emit (sprintf "    %s = func.call %s(%s) : (i32) -> i32" 
                                          resultId funcId argStr) state3
                let state5 = TypeTracking.recordSSAType resultId resultType state4
                
                // Apply conversion if needed
                match expectedType with
                | Some expected when expected <> resultType ->
                    let (convertedId, convertedState) = 
                        Emitter.convertType resultId resultType expected state5
                    (convertedId, expected, convertedState)
                | _ ->
                    (resultId, resultType, state5)
        
        | IfThenElse(cond, thenExpr, elseExpr) ->
            // Process condition
            let condState = TypeTracking.setExpectedType (MLIRTypes.createInteger 1) state
            let (condId, condType, state1) = processExpression cond condState
            
            // Apply conversion to i1 if needed
            let (finalCondId, state2) =
                if condType <> MLIRTypes.createInteger 1 then
                    let (convertedId, convertedState) = 
                        Emitter.convertType condId condType (MLIRTypes.createInteger 1) state1
                    (convertedId, convertedState)
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
                match expectedType with
                | Some t -> t
                | None -> MLIRTypes.createInteger 32
            
            // Process then branch
            let state8 = Emitter.emit (sprintf "  ^%s:" thenLabel) state7
            let thenState = TypeTracking.setExpectedType resultType state8
            let (thenResultId, thenType, state9) = processExpression thenExpr thenState
            
            // Apply conversion if needed
            let (finalThenId, state10) =
                if thenType <> resultType then
                    let (convertedId, convertedState) = 
                        Emitter.convertType thenResultId thenType resultType state9
                    (convertedId, convertedState)
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
                    let (convertedId, convertedState) = 
                        Emitter.convertType elseResultId elseType resultType state14
                    (convertedId, convertedState)
                else
                    (elseResultId, state14)
            
            let state16 = Emitter.emit (sprintf "    %s = %s : %s" 
                                       resultId finalElseId (mlirTypeToString resultType)) state15
            let state17 = Emitter.emit (sprintf "    br ^%s" endLabel) state16
            
            // Add end label
            let state18 = Emitter.emit (sprintf "  ^%s:" endLabel) state17
            let state19 = TypeTracking.recordSSAType resultId resultType state18
            
            (resultId, resultType, state19)
        
        | _ ->
            // Fallback for other expression types
            let (dummyId, state1) = SSA.generateValue "unknown_expr" state
            let dummyType = MLIRTypes.createInteger 32
            let state2 = TypeTracking.recordSSAType dummyId dummyType state1
            
            // Apply conversion if needed
            match expectedType with
            | Some expected when expected <> dummyType ->
                let (convertedId, convertedState) = 
                    Emitter.convertType dummyId dummyType expected state2
                (convertedId, expected, convertedState)
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
    let (bodyResult, bodyType, state4) = TypedPatterns.processExpression body state3
    
    // Apply type conversion if needed
    let (finalBodyResult, state5) =
        if bodyType <> returnMLIRType then
            let (convertedId, convertedState) = 
                Emitter.convertType bodyResult bodyType returnMLIRType state4
            (convertedId, convertedState)
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