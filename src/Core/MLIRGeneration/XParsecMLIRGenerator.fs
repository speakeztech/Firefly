module Core.MLIRGeneration.XParsecMLIRGenerator

open System
open System.Text
open Dabbit.Parsing.OakAst
open Core.MLIRGeneration.TypeSystem
open Core.MLIRGeneration.Dialect
open Core.XParsec.Foundation

/// MLIR generation state for tracking SSA values and types
type MLIRGenerationState = {
    SSACounter: int
    CurrentScope: Map<string, string>
    ScopeStack: Map<string, string> list
    GeneratedOperations: string list
    ModuleLevelDeclarations: string list
    CurrentFunction: string option
    StringConstants: Map<string, string>
    SSAValueTypes: Map<string, MLIRType>
}

/// MLIR module output with complete information
type MLIRModuleOutput = {
    ModuleName: string
    Operations: string list
    Diagnostics: string list
}

/// Generates a new SSA value with optional prefix
let generateSSAValue (prefix: string) (state: MLIRGenerationState) : string * MLIRGenerationState =
    let newCounter = state.SSACounter + 1
    let ssaName = sprintf "%%%s%d" prefix newCounter
    let newState = { state with SSACounter = newCounter }
    (ssaName, newState)

/// Binds a variable name to an SSA value in current scope
let bindVariable (name: string) (value: string) (state: MLIRGenerationState) : MLIRGenerationState =
    { state with CurrentScope = Map.add name value state.CurrentScope }

/// Looks up a variable by name in the current scope stack
let lookupVariable (name: string) (state: MLIRGenerationState) : string option =
    match Map.tryFind name state.CurrentScope with
    | Some value -> Some value
    | None -> 
        state.ScopeStack 
        |> List.tryPick (fun scope -> Map.tryFind name scope)

/// Records the type of an SSA value
let recordSSAType (valueId: string) (valueType: MLIRType) (state: MLIRGenerationState) : MLIRGenerationState =
    { state with SSAValueTypes = Map.add valueId valueType state.SSAValueTypes }

/// Emits a raw MLIR operation string
let emit (operation: string) (state: MLIRGenerationState) : MLIRGenerationState =
    { state with GeneratedOperations = operation :: state.GeneratedOperations }

/// Emits a module-level declaration
let emitModuleLevel (declaration: string) (state: MLIRGenerationState) : MLIRGenerationState =
    { state with ModuleLevelDeclarations = declaration :: state.ModuleLevelDeclarations }

/// Registers a string constant at module level
let registerString (value: string) (state: MLIRGenerationState) : string * MLIRGenerationState =
    match Map.tryFind value state.StringConstants with
    | Some existingName -> (existingName, state)
    | None ->
        let constName = sprintf "@str_%d" state.StringConstants.Count
        let newState = { state with StringConstants = Map.add value constName state.StringConstants }
        (constName, newState)

/// Processes a literal expression to MLIR
let processLiteral (lit: OakLiteral) (state: MLIRGenerationState) : string * MLIRType * MLIRGenerationState =
    match lit with
    | IntLiteral value ->
        let (resultId, state1) = generateSSAValue "const" state
        let intType = MLIRTypeUtils.createInteger 32
        let state2 = emit (sprintf "    %s = arith.constant %d : i32" resultId value) state1
        let state3 = recordSSAType resultId intType state2
        (resultId, intType, state3)
        
    | FloatLiteral value ->
        let (resultId, state1) = generateSSAValue "const" state
        let floatType = MLIRTypeUtils.createFloat 32
        let state2 = emit (sprintf "    %s = arith.constant %f : f32" resultId value) state1
        let state3 = recordSSAType resultId floatType state2
        (resultId, floatType, state3)
        
    | BoolLiteral value ->
        let (resultId, state1) = generateSSAValue "const" state
        let boolType = MLIRTypeUtils.createInteger 1
        let state2 = emit (sprintf "    %s = arith.constant %d : i1" resultId (if value then 1 else 0)) state1
        let state3 = recordSSAType resultId boolType state2
        (resultId, boolType, state3)
        
    | StringLiteral value ->
        let (globalName, state1) = registerString value state
        let (ptrResult, state2) = generateSSAValue "str" state1
        let memrefType = MLIRTypeUtils.createMemRef (MLIRTypeUtils.createInteger 8)
        let state3 = emit (sprintf "    %s = memref.get_global %s : memref<?xi8>" ptrResult globalName) state2
        let state4 = recordSSAType ptrResult memrefType state3
        (ptrResult, memrefType, state4)
        
    | _ -> 
        // Unit and other literals default to i32 constant 0
        let (resultId, state1) = generateSSAValue "const" state
        let intType = MLIRTypeUtils.createInteger 32
        let state2 = emit (sprintf "    %s = arith.constant 0 : i32" resultId) state1
        let state3 = recordSSAType resultId intType state2
        (resultId, intType, state3)

/// Main recursive function to process expressions to MLIR
let rec processExpression (expr: OakExpression) (state: MLIRGenerationState) : string * MLIRType * MLIRGenerationState =
    match expr with
    | Literal lit -> processLiteral lit state
    
    | Variable name ->
        match lookupVariable name state with
        | Some value -> 
            // Get variable type if known
            let valueType = 
                match Map.tryFind value state.SSAValueTypes with
                | Some t -> t
                | None -> MLIRTypeUtils.createInteger 32 // Default type if unknown
            (value, valueType, state)
        | None ->
            // Unknown variable - create a placeholder
            let (dummyId, state1) = generateSSAValue "unknown" state
            let intType = MLIRTypeUtils.createInteger 32
            let state2 = emit (sprintf "    %s = arith.constant 0 : i32 // Unknown variable: %s" dummyId name) state1
            let state3 = recordSSAType dummyId intType state2
            (dummyId, intType, state3)
    
    | Let(name, value, body) ->
        // Process value
        let (valueId, valueType, state1) = processExpression value state
        
        // Bind variable
        let state2 = bindVariable name valueId state1
        
        // Process body
        let (bodyId, bodyType, state3) = processExpression body state2
        (bodyId, bodyType, state3)
    
    | Sequential(first, second) ->
        // Process first expression (ignore result)
        let (_, _, state1) = processExpression first state
        
        // Process second expression
        let (secondId, secondType, state2) = processExpression second state1
        (secondId, secondType, state2)
    
    | Application(func, args) ->
        // Simplified function application
        let (funcId, _, state1) = processExpression func state
        
        // Process arguments
        let rec processArgs remainingArgs processedArgs currentState =
            match remainingArgs with
            | [] -> (List.rev processedArgs, currentState)
            | arg :: rest ->
                let (argId, _, newState) = processExpression arg currentState
                processArgs rest (argId :: processedArgs) newState
        
        let (processedArgs, state2) = processArgs args [] state1
        
        // Create a basic function call
        let (resultId, state3) = generateSSAValue "call" state2
        let resultType = MLIRTypeUtils.createInteger 32 // Default return type
        
        // Simple call operation
        let argsStr = String.concat ", " processedArgs
        let state4 = emit (sprintf "    %s = func.call %s(%s) : (i32) -> i32" resultId funcId argsStr) state3
        let state5 = recordSSAType resultId resultType state4
        
        (resultId, resultType, state5)
    
    | IfThenElse(cond, thenExpr, elseExpr) ->
        // Process condition
        let (condId, _, state1) = processExpression cond state
        
        // Create branch labels
        let (resultId, state2) = generateSSAValue "if_result" state1
        let thenLabel = sprintf "then_%d" state2.SSACounter
        let elseLabel = sprintf "else_%d" state2.SSACounter
        let endLabel = sprintf "end_%d" state2.SSACounter
        
        // Create conditional branch
        let state3 = emit (sprintf "    cond_br %s, ^%s, ^%s" condId thenLabel elseLabel) state2
        
        // Then block
        let state4 = emit (sprintf "  ^%s:" thenLabel) state3
        let (thenResultId, thenType, state5) = processExpression thenExpr state4
        let state6 = emit (sprintf "    %s = %s : %s" resultId thenResultId (mlirTypeToString thenType)) state5
        let state7 = emit (sprintf "    br ^%s" endLabel) state6
        
        // Else block
        let state8 = emit (sprintf "  ^%s:" elseLabel) state7
        let (elseResultId, elseType, state9) = processExpression elseExpr state8
        let state10 = emit (sprintf "    %s = %s : %s" resultId elseResultId (mlirTypeToString elseType)) state9
        let state11 = emit (sprintf "    br ^%s" endLabel) state10
        
        // End block
        let state12 = emit (sprintf "  ^%s:" endLabel) state11
        
        // Assume result type is the then branch type (simplified)
        let state13 = recordSSAType resultId thenType state12
        
        (resultId, thenType, state13)
    
    | Match(matchExpr, cases) ->
        // Simple implementation for match expressions
        let (matchValueId, matchValueType, state1) = processExpression matchExpr state
        
        // Create result variable 
        let (resultId, state2) = generateSSAValue "match" state1
        let resultType = MLIRTypeUtils.createInteger 32 // Default result type
        
        // Process first case only (simplified)
        match cases with
        | (_, firstExpr) :: _ ->
            let (caseResultId, caseResultType, state3) = processExpression firstExpr state2
            let state4 = emit (sprintf "    %s = %s : %s" resultId caseResultId (mlirTypeToString caseResultType)) state3
            let state5 = recordSSAType resultId caseResultType state4
            (resultId, caseResultType, state5)
        | [] ->
            // No cases, return unit
            let state3 = emit (sprintf "    %s = arith.constant 0 : i32 // Empty match" resultId) state2
            let state4 = recordSSAType resultId resultType state3
            (resultId, resultType, state4)
    
    | _ ->
        // For other expressions, generate a placeholder
        let (dummyId, state1) = generateSSAValue "unhandled" state
        let intType = MLIRTypeUtils.createInteger 32
        let state2 = emit (sprintf "    %s = arith.constant 0 : i32 // Unhandled expression type" dummyId) state1
        let state3 = recordSSAType dummyId intType state2
        (dummyId, intType, state3)

/// Creates initial state for MLIR generation
let createInitialState() : MLIRGenerationState = {
    SSACounter = 0
    CurrentScope = Map.empty
    ScopeStack = []
    GeneratedOperations = []
    ModuleLevelDeclarations = []
    CurrentFunction = None
    StringConstants = Map.empty
    SSAValueTypes = Map.empty
}

/// Converts a function declaration to MLIR
let convertFunction (name: string) (parameters: (string * OakType) list) (returnType: OakType) 
                   (body: OakExpression) (state: MLIRGenerationState) : MLIRGenerationState =
    let returnMLIRType = mapOakTypeToMLIR returnType
    let returnTypeStr = mlirTypeToString returnMLIRType
    
    let paramStr = 
        parameters 
        |> List.mapi (fun i (_, typ) -> 
            let mlirType = mapOakTypeToMLIR typ
            sprintf "%%arg%d: %s" i (mlirTypeToString mlirType))
        |> String.concat ", "
    
    let state1 = emit (sprintf "  func.func @%s(%s) -> %s {" name paramStr returnTypeStr) state
    
    // Bind parameters to scope
    let state2 = 
        parameters 
        |> List.mapi (fun i (paramName, paramType) -> 
            let argId = sprintf "%%arg%d" i
            let mlirType = mapOakTypeToMLIR paramType
            (paramName, argId, mlirType))
        |> List.fold (fun s (name, argId, mlirType) -> 
            s |> bindVariable name argId |> recordSSAType argId mlirType) state1
    
    // Process body
    let (bodyId, bodyType, state3) = processExpression body state2
    
    // Generate return
    let state4 = 
        if returnType = UnitType then
            emit "    func.return" state3
        else
            emit (sprintf "    func.return %s : %s" bodyId returnTypeStr) state3
    
    // Close function
    emit "  }" state4

/// Generate MLIR module from an Oak program
let generateMLIR (program: OakProgram) : MLIRModuleOutput =
    let initialState = createInitialState()
    
    let processModule (state: MLIRGenerationState) (mdl: OakModule) =
        let state1 = emit (sprintf "module @%s {" mdl.Name) state
        
        // Process each declaration
        let state2 = 
            mdl.Declarations
            |> List.fold (fun currState decl ->
                match decl with
                | FunctionDecl(name, parameters, returnType, body) ->
                    convertFunction name parameters returnType body currState
                
                | EntryPoint(expr) ->
                    convertFunction "main" [("argc", IntType); ("argv", ArrayType(StringType))] IntType expr currState
                    
                | ExternalDecl(name, paramTypes, returnType, _) ->
                    let paramTypeStrs = paramTypes |> List.map (mapOakTypeToMLIR >> mlirTypeToString)
                    let returnTypeStr = mlirTypeToString (mapOakTypeToMLIR returnType)
                    let paramStr = String.concat ", " paramTypeStrs
                    emitModuleLevel (sprintf "  func.func private @%s(%s) -> %s" name paramStr returnTypeStr) currState
                
                | _ -> currState
            ) state1
        
        // Generate string constants at module level
        let state3 = 
            state2.StringConstants
            |> Map.toList
            |> List.fold (fun s (value, globalName) ->
                let escapedValue = value.Replace("\\", "\\\\").Replace("\"", "\\\"")
                let constSize = escapedValue.Length + 1
                let declaration = sprintf "  memref.global constant %s = dense<\"%s\\00\"> : memref<%dxi8>" 
                                        globalName escapedValue constSize
                emit declaration s) state2
        
        // Add module-level declarations
        let state4 = 
            state3.ModuleLevelDeclarations
            |> List.rev
            |> List.distinct
            |> List.fold (fun s decl -> emit decl s) state3
        
        // Close module
        emit "}" state4
    
    // Process first module (or empty if none)
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
        Diagnostics = []
    }

/// Generates complete MLIR module text from Oak AST
let generateMLIRModuleText (program: OakProgram) : CompilerResult<string> =
    try
        let mlirOutput = generateMLIR program
        let moduleText = String.concat "\n" mlirOutput.Operations
        Success moduleText
    with ex ->
        CompilerFailure [
            ConversionError(
                "MLIR generation", 
                "Oak AST", 
                "MLIR", 
                sprintf "Exception: %s" ex.Message)
        ]