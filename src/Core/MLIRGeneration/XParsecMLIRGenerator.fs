module Core.MLIRGeneration.XParsecMLIRGenerator

open System
open System.Text
open Dabbit.Parsing.OakAst
open Core.MLIRGeneration.TypeSystem
open Core.MLIRGeneration.Dialect
open Core.XParsec.Foundation

module MLIROperationBuilder =
    // Generate arithmetic operation
    let buildArithmeticOp (op: string) (lhs: string) (rhs: string) (resultId: string) (resultType: MLIRType) : string =
        sprintf "    %s = arith.%s %s, %s : %s" resultId op lhs rhs (mlirTypeToString resultType)
    
    // Generate constant operation
    let buildConstantOp (value: string) (resultId: string) (mlirType: MLIRType) : string =
        sprintf "    %s = arith.constant %s : %s" resultId value (mlirTypeToString mlirType)
        
    /// Creates call operation string
    let buildCallOp (funcName: string) (args: string list) (resultId: string option) (resultType: MLIRType option) =
        // Determine argument types based on name patterns
        let argTypes = 
            args |> List.map (fun arg ->
                if arg.StartsWith("%str") || arg.Contains("global") then
                    // String arguments
                    "memref<?xi8>"
                else
                    // Default to integer
                    "i32")
        
        // Format arguments
        let argsStr = String.concat ", " args
        let argTypesStr = String.concat ", " argTypes
        
        // Format return type
        let returnTypeStr = 
            match resultType with
            | Some typ -> mlirTypeToString typ
            | None -> "()"
        
        // Generate call
        match resultId with
        | Some id ->
            sprintf "    %s = func.call @%s(%s) : (%s) -> %s" 
                id funcName argsStr argTypesStr returnTypeStr
        | None ->
            sprintf "    func.call @%s(%s) : (%s) -> %s" 
                funcName argsStr argTypesStr returnTypeStr
    
    // Generate return operation
    let buildReturnOp (values: string list) : string =
        if values.IsEmpty then
            "    func.return"
        else
            sprintf "    func.return %s" (String.concat ", " values)
            
    // Generate memory allocation operation
    let buildAllocaOp (size: string) (resultId: string) (elementType: MLIRType) : string =
        sprintf "    %s = memref.alloca(%s) : memref<?x%s>" 
            resultId size (mlirTypeToString elementType)
            
    // Generate memory load operation
    let buildLoadOp (memref: string) (resultId: string) (elementType: MLIRType) : string =
        sprintf "    %s = memref.load %s[] : memref<?x%s>" resultId memref (mlirTypeToString elementType)
        
    // Generate memory store operation
    let buildStoreOp (value: string) (memref: string) (elementType: MLIRType) : string =
        sprintf "    memref.store %s, %s[] : memref<?x%s>" value memref (mlirTypeToString elementType)
        
    // Generate global string constant
    let buildGlobalStringOp (name: string) (value: string) : string =
        let escapedValue = value.Replace("\\", "\\\\").Replace("\"", "\\\"")
        let constSize = escapedValue.Length + 1
        sprintf "  memref.global constant %s = dense<\"%s\\00\"> : memref<%dxi8>" 
            name escapedValue constSize
            
    // Generate reference to global
    let buildGetGlobalOp (globalName: string) (resultId: string) : string =
        sprintf "    %s = memref.get_global %s : memref<?xi8>" resultId globalName
        
    // Generate conditional branch
    let buildCondBranchOp (condition: string) (trueDest: string) (falseDest: string) : string =
        sprintf "    cond_br %s, ^%s, ^%s" condition trueDest falseDest
        
    // Generate unconditional branch
    let buildBranchOp (dest: string) : string =
        sprintf "    br ^%s" dest
        
    // Generate block label
    let buildBlockLabel (label: string) : string =
        sprintf "  ^%s:" label

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
        let constOp = MLIROperationBuilder.buildConstantOp (string value) resultId intType
        let state2 = emit constOp state1
        let state3 = recordSSAType resultId intType state2
        (resultId, intType, state3)
        
    | FloatLiteral value ->
        let (resultId, state1) = generateSSAValue "const" state
        let floatType = MLIRTypeUtils.createFloat 32
        let constOp = MLIROperationBuilder.buildConstantOp (string value) resultId floatType
        let state2 = emit constOp state1
        let state3 = recordSSAType resultId floatType state2
        (resultId, floatType, state3)
        
    | BoolLiteral value ->
        let (resultId, state1) = generateSSAValue "const" state
        let boolType = MLIRTypeUtils.createInteger 1
        let constOp = MLIROperationBuilder.buildConstantOp (if value then "1" else "0") resultId boolType
        let state2 = emit constOp state1
        let state3 = recordSSAType resultId boolType state2
        (resultId, boolType, state3)
        
    | StringLiteral value ->
        let (globalName, state1) = registerString value state
        let (ptrResult, state2) = generateSSAValue "str" state1
        let memrefType = MLIRTypeUtils.createMemRef (MLIRTypeUtils.createInteger 8)
        let getGlobalOp = MLIROperationBuilder.buildGetGlobalOp globalName ptrResult
        let state3 = emit getGlobalOp state2
        let state4 = recordSSAType ptrResult memrefType state3
        (ptrResult, memrefType, state4)
        
    | _ -> 
        // Unit and other literals default to i32 constant 0
        let (resultId, state1) = generateSSAValue "const" state
        let intType = MLIRTypeUtils.createInteger 32
        let constOp = MLIROperationBuilder.buildConstantOp "0" resultId intType
        let state2 = emit constOp state1
        let state3 = recordSSAType resultId intType state2
        (resultId, intType, state3)

/// Main recursive function to process expressions to MLIR
let rec processExpression (expr: OakExpression) (state: MLIRGenerationState) : string * MLIRType * MLIRGenerationState =
    match expr with
    | Literal lit -> processLiteral lit state
    
    | Variable name ->
        match lookupVariable name state with
        | Some value -> 
            // Existing logic for local variables
            let valueType = 
                match Map.tryFind value state.SSAValueTypes with
                | Some t -> t
                | None -> MLIRTypeUtils.createInteger 32
            (value, valueType, state)
        | None ->
            // Check if this is an Alloy module function
            if List.contains name ["stackBuffer"; "prompt"; "readInto"; "spanToString"; 
                                "format"; "String.format"; "writeLine"; "readLine"] then
                // Generate a proper function reference instead of a placeholder
                let funcName = sprintf "@%s" name
                let (funcId, state1) = generateSSAValue "func" state
                let funcType = MLIRTypeUtils.createFunction [MLIRTypeUtils.createInteger 32] 
                                                            (MLIRTypeUtils.createInteger 32)
                // Create a reference to the function
                let state2 = emit (sprintf "    %s = func.constant %s : %s" 
                                    funcId funcName (mlirTypeToString funcType)) state1
                let state3 = recordSSAType funcId funcType state2
                
                // Also generate a module-level function declaration
                let state4 = emitModuleLevel (sprintf "  func.func @%s(%s) -> %s" 
                                    name "memref<?xi8>" "i32") state3
                                
                (funcId, funcType, state4)
            else
                // Existing logic for unknown variables
                let (dummyId, state1) = generateSSAValue "unknown" state
                let intType = MLIRTypeUtils.createInteger 32
                let state2 = emit (sprintf "    %s = arith.constant 0 : i32 // Unknown variable: %s" 
                                    dummyId name) state1
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
        // Process function expression
        let (funcId, _, state1) = processExpression func state
        
        // Process arguments
        let rec processArgs remainingArgs processedArgs currentState =
            match remainingArgs with
            | [] -> (List.rev processedArgs, currentState)
            | arg :: rest ->
                let (argId, _, newState) = processExpression arg currentState
                processArgs rest (argId :: processedArgs) newState
        
        let (processedArgs, state2) = processArgs args [] state1
        
        // Create function call
        let (resultId, state3) = generateSSAValue "call" state2
        let resultType = MLIRTypeUtils.createInteger 32 // Default return type
        
        // Generate call operation
        let funcName = funcId.TrimStart('%')
        let callOp = MLIROperationBuilder.buildCallOp funcName processedArgs (Some resultId) (Some resultType)
        let state4 = emit callOp state3
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
        let branchOp = MLIROperationBuilder.buildCondBranchOp condId thenLabel elseLabel
        let state3 = emit branchOp state2
        
        // Then block
        let state4 = emit (MLIROperationBuilder.buildBlockLabel thenLabel) state3
        let (thenResultId, thenType, state5) = processExpression thenExpr state4
        let resultAssignOp = sprintf "    %s = %s : %s" resultId thenResultId (mlirTypeToString thenType)
        let state6 = emit resultAssignOp state5
        let state7 = emit (MLIROperationBuilder.buildBranchOp endLabel) state6
        
        // Else block
        let state8 = emit (MLIROperationBuilder.buildBlockLabel elseLabel) state7
        let (elseResultId, elseType, state9) = processExpression elseExpr state8
        let resultAssignOp2 = sprintf "    %s = %s : %s" resultId elseResultId (mlirTypeToString elseType)
        let state10 = emit resultAssignOp2 state9
        let state11 = emit (MLIROperationBuilder.buildBranchOp endLabel) state10
        
        // End block
        let state12 = emit (MLIROperationBuilder.buildBlockLabel endLabel) state11
        
        // Assume result type is the then branch type (simplified)
        let state13 = recordSSAType resultId thenType state12
        
        (resultId, thenType, state13)
    
    | Match(matchExpr, cases) ->
        // Process match expression
        let (matchValueId, matchValueType, state1) = processExpression matchExpr state
        
        // Create result variable 
        let (resultId, state2) = generateSSAValue "match" state1
        let resultType = MLIRTypeUtils.createInteger 32 // Default result type
        
        // Process first case only (simplified)
        match cases with
        | (_, firstExpr) :: _ ->
            let (caseResultId, caseResultType, state3) = processExpression firstExpr state2
            let assignOp = sprintf "    %s = %s : %s" resultId caseResultId (mlirTypeToString caseResultType)
            let state4 = emit assignOp state3
            let state5 = recordSSAType resultId caseResultType state4
            (resultId, caseResultType, state5)
        | [] ->
            // No cases, return unit
            let constOp = MLIROperationBuilder.buildConstantOp "0" resultId resultType
            let state3 = emit (constOp + " // Empty match") state2
            let state4 = recordSSAType resultId resultType state3
            (resultId, resultType, state4)
    
    | _ ->
        // For other expressions, generate a placeholder
        let (dummyId, state1) = generateSSAValue "unhandled" state
        let intType = MLIRTypeUtils.createInteger 32
        let constOp = MLIROperationBuilder.buildConstantOp "0" dummyId intType
        let state2 = emit (constOp + " // Unhandled expression type") state1
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
            emit (MLIROperationBuilder.buildReturnOp []) state3
        else
            emit (MLIROperationBuilder.buildReturnOp [bodyId]) state3
    
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
                let globalOp = MLIROperationBuilder.buildGlobalStringOp globalName value
                emit globalOp s) state2
        
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