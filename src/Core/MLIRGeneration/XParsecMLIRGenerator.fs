module Core.MLIRGeneration.XParsecMLIRGenerator

open System
open Firefly.Core.XParsec.Foundation
open Dabbit.Parsing.OakAst
open Core.MLIRGeneration.TypeSystem
open Core.MLIRGeneration.Operations

/// MLIR generation state for tracking SSA values and scopes
type MLIRGenerationState = {
    SSACounter: int
    CurrentScope: Map<string, string>  // Variable name -> SSA value
    ScopeStack: Map<string, string> list
    GeneratedOperations: string list
    CurrentFunction: string option
    ExternalDeclarations: Set<string>
    StringConstants: Map<string, int>
}

/// MLIR output with complete module information
type MLIRModuleOutput = {
    ModuleName: string
    Operations: string list
    SSAMappings: Map<string, string>
    TypeMappings: Map<string, MLIRType>
    Diagnostics: string list
}

/// SSA value generation and variable management
module SSAGeneration =
    
    /// Generates a unique SSA value name
    let generateSSAValue (prefix: string) (state: MLIRGenerationState) : string * MLIRGenerationState =
        let newCounter = state.SSACounter + 1
        let ssaName = sprintf "%%%s%d" prefix newCounter
        let newState = { state with SSACounter = newCounter }
        (ssaName, newState)
    
    /// Records a variable binding in current scope
    let bindVariable (varName: string) (ssaValue: string) (state: MLIRGenerationState) : MLIRGenerationState =
        let newScope = Map.add varName ssaValue state.CurrentScope
        { state with CurrentScope = newScope }
    
    /// Looks up a variable in the scope stack
    let lookupVariable (varName: string) (state: MLIRGenerationState) : Result<string, string> =
        let rec findInScopes scopes =
            match scopes with
            | [] -> None
            | scope :: rest ->
                match Map.tryFind varName scope with
                | Some value -> Some value
                | None -> findInScopes rest
        
        match findInScopes (state.CurrentScope :: state.ScopeStack) with
        | Some ssaValue -> Ok ssaValue
        | None -> Error (sprintf "Undefined variable '%s' in current scope" varName)
    
    /// Pushes a new scope onto the stack
    let pushScope (state: MLIRGenerationState) : MLIRGenerationState =
        let newScopeStack = state.CurrentScope :: state.ScopeStack
        { state with 
            ScopeStack = newScopeStack
            CurrentScope = Map.empty 
        }
    
    /// Pops a scope from the stack
    let popScope (state: MLIRGenerationState) : Result<MLIRGenerationState, string> =
        match state.ScopeStack with
        | prevScope :: rest ->
            Ok { state with 
                    CurrentScope = prevScope
                    ScopeStack = rest 
                }
        | [] ->
            Error "Cannot pop scope - scope stack is empty"

/// MLIR operation emission functions
module OperationEmission =
    
    /// Emits a single MLIR operation
    let emitOperation (operation: string) (state: MLIRGenerationState) : MLIRGenerationState =
        let newOps = operation :: state.GeneratedOperations
        { state with GeneratedOperations = newOps }
    
    /// Records an external declaration to avoid duplicates
    let recordExternalDeclaration (funcName: string) (state: MLIRGenerationState) : MLIRGenerationState =
        { state with ExternalDeclarations = Set.add funcName state.ExternalDeclarations }
    
    /// Emits an external function declaration
    let emitExternalDeclaration (funcName: string) (signature: string) (state: MLIRGenerationState) : MLIRGenerationState =
        if Set.contains funcName state.ExternalDeclarations then
            state  // Already declared
        else
            let declaration = sprintf "func.func private @%s%s" funcName signature
            let newState = recordExternalDeclaration funcName state
            { newState with GeneratedOperations = declaration :: newState.GeneratedOperations }
    
    /// Registers a string constant and returns its global name
    let registerStringConstant (value: string) (state: MLIRGenerationState) : string * MLIRGenerationState =
        let existingConst = 
            state.StringConstants 
            |> Map.tryFindKey (fun k _ -> k = value)
        
        match existingConst with
        | Some existingValue ->
            let constIndex = state.StringConstants.[existingValue]
            let globalName = sprintf "@str_const_%d" constIndex
            (globalName, state)
        | None ->
            let newIndex = state.StringConstants.Count
            let globalName = sprintf "@str_const_%d" newIndex
            let newConstants = Map.add value newIndex state.StringConstants
            
            // Emit global string constant declaration
            let escapedValue = value.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n")
            let length = value.Length + 1  // Include null terminator
            let globalDecl = sprintf "memref.global constant @str_const_%d : memref<%dxi8> = dense<\"%s\\00\">" 
                                    newIndex length escapedValue
            
            let newState = { 
                state with 
                    StringConstants = newConstants
                    GeneratedOperations = globalDecl :: state.GeneratedOperations
            }
            (globalName, newState)
    
    /// Emits an MLIR constant operation
    let emitConstant (value: string) (mlirType: MLIRType) (state: MLIRGenerationState) : string * MLIRGenerationState =
        let (ssaValue, newState) = SSAGeneration.generateSSAValue "const" state
        let typeStr = mlirTypeToString mlirType
        let operation = sprintf "  %s = arith.constant %s : %s" ssaValue value typeStr
        let finalState = emitOperation operation newState
        (ssaValue, finalState)

/// Enhanced literal to MLIR conversion
module LiteralConversion =
    
    /// Converts Oak literal to MLIR constant
    let convertLiteral (literal: OakLiteral) (state: MLIRGenerationState) : Result<string * MLIRGenerationState, string> =
        match literal with
        | IntLiteral value ->
            let (ssaValue, newState) = OperationEmission.emitConstant (string value) (Integer 32) state
            Ok (ssaValue, newState)
        
        | FloatLiteral value ->
            let (ssaValue, newState) = OperationEmission.emitConstant (sprintf "%f" value) (Float 32) state
            Ok (ssaValue, newState)
        
        | BoolLiteral value ->
            let boolValue = if value then "1" else "0"
            let (ssaValue, newState) = OperationEmission.emitConstant boolValue (Integer 1) state
            Ok (ssaValue, newState)
        
        | StringLiteral value ->
            let (globalName, state1) = OperationEmission.registerStringConstant value state
            let (addrSSA, state2) = SSAGeneration.generateSSAValue "str_addr" state1
            let addressOp = sprintf "  %s = memref.get_global %s : memref<%dxi8>" 
                                   addrSSA globalName (value.Length + 1)
            let finalState = OperationEmission.emitOperation addressOp state2
            Ok (addrSSA, finalState)
        
        | UnitLiteral ->
            let (ssaValue, newState) = OperationEmission.emitConstant "0" (Integer 32) state
            Ok (ssaValue, newState)
        
        | ArrayLiteral _ ->
            Error "Array literals not yet implemented in MLIR generation"

/// I/O operation conversion
module IOOperationConversion =
    
    /// Converts I/O operation to MLIR with proper external function calls
    let convertIOOperation (ioType: IOOperationType) (args: OakExpression list) 
                          (convertExpression: OakExpression -> MLIRGenerationState -> Result<string * MLIRGenerationState, string>)
                          (state: MLIRGenerationState) : Result<string * MLIRGenerationState, string> =
        match ioType with
        | Printf(formatString) ->
            // Ensure printf is declared
            let state1 = OperationEmission.emitExternalDeclaration "printf" "(memref<?xi8>, ...) -> i32" state
            
            // Register format string constant
            let (formatGlobal, state2) = OperationEmission.registerStringConstant formatString state1
            
            // Get address of format string
            let (formatPtr, state3) = SSAGeneration.generateSSAValue "fmt_ptr" state2
            let getFormatOp = sprintf "  %s = memref.get_global %s : memref<%dxi8>" 
                                     formatPtr formatGlobal (formatString.Length + 1)
            let state4 = OperationEmission.emitOperation getFormatOp state3
            
            // Convert format arguments if any
            if args.IsEmpty then
                // Simple printf with no arguments
                let (resultSSA, state5) = SSAGeneration.generateSSAValue "printf_result" state4
                let callOp = sprintf "  %s = func.call @printf(%s) : (memref<?xi8>) -> i32" 
                                    resultSSA formatPtr
                let finalState = OperationEmission.emitOperation callOp state5
                Ok (resultSSA, finalState)
            else
                // Process arguments one by one, building up the state
                let rec processArgs remainingArgs (currentState: MLIRGenerationState) (accArgs: string list) =
                    match remainingArgs with
                    | [] -> Ok (List.rev accArgs, currentState)
                    | arg :: rest ->
                        match convertExpression arg currentState with
                        | Ok (argResult, newState) -> processArgs rest newState (argResult :: accArgs)
                        | Error e -> Error e
                
                match processArgs args state4 [] with
                | Ok (argSSAValues, state5) ->
                    let (resultSSA, state6) = SSAGeneration.generateSSAValue "printf_result" state5
                    let allArgs = formatPtr :: argSSAValues
                    let argTypes = "memref<?xi8>" :: (List.replicate argSSAValues.Length "i32")
                    let callOp = sprintf "  %s = func.call @printf(%s) : (%s) -> i32" 
                                        resultSSA 
                                        (String.concat ", " allArgs)
                                        (String.concat ", " argTypes)
                    let finalState = OperationEmission.emitOperation callOp state6
                    Ok (resultSSA, finalState)
                | Error e -> Error e
        
        | Printfn(formatString) ->
            let formatWithNewline = formatString + "\n"
            convertIOOperation (Printf(formatWithNewline)) args convertExpression state
            
        | ReadLine ->
            Error "ReadLine operation not implemented"
            
        | Scanf _ ->
            Error "Scanf operation not implemented"
            
        | WriteFile _ | ReadFile _ ->
            Error "File I/O operations not implemented"

/// Expression conversion recursively calls itself
let rec convertExpression (expr: OakExpression) (state: MLIRGenerationState) : Result<string * MLIRGenerationState, string> =
    match expr with
    | Literal literal ->
        LiteralConversion.convertLiteral literal state
    
    | Variable varName ->
        match SSAGeneration.lookupVariable varName state with
        | Ok ssaValue -> Ok (ssaValue, state)
        | Error e -> Error e
    
    | Let(name, value, body) ->
        match convertExpression value state with
        | Ok (valueSSA, state1) ->
            let state2 = SSAGeneration.bindVariable name valueSSA state1
            convertExpression body state2
        | Error e -> Error e
    
    | Sequential(first, second) ->
        match convertExpression first state with
        | Ok (_, state1) -> convertExpression second state1
        | Error e -> Error e
    
    | IOOperation(ioType, args) ->
        IOOperationConversion.convertIOOperation ioType args convertExpression state
    
    | _ ->
        Error (sprintf "Expression type not yet implemented: %A" expr)

/// Function declaration conversion
let convertFunctionDeclaration (name: string) (parameters: (string * OakType) list) (returnType: OakType) 
                             (body: OakExpression) (state: MLIRGenerationState) : Result<MLIRGenerationState, string> =
    // Emit function signature
    let paramTypeStrs = 
        if parameters.IsEmpty then []
        else parameters |> List.map (snd >> mapOakTypeToMLIR >> mlirTypeToString)
    
    let returnTypeStr = mlirTypeToString (mapOakTypeToMLIR returnType)
    let paramList = 
        if parameters.IsEmpty then ""
        else 
            parameters 
            |> List.mapi (fun i (paramName, t) -> 
                sprintf "%%arg%d: %s" i (mlirTypeToString (mapOakTypeToMLIR t)))
            |> String.concat ", "
            |> sprintf "(%s)"
    
    let funcSig = sprintf "func.func @%s%s -> %s {" 
                         name 
                         (if parameters.IsEmpty then "()" else paramList)
                         returnTypeStr
    
    let state1 = OperationEmission.emitOperation funcSig state
    let state2 = SSAGeneration.pushScope state1
    
    // Bind parameters to entry block arguments
    let state3 = 
        parameters 
        |> List.mapi (fun i (paramName, _) -> 
            let argSSA = sprintf "%%arg%d" i
            SSAGeneration.bindVariable paramName argSSA state2)
        |> fun s -> s
    
    // Convert function body
    match convertExpression body state3 with
    | Ok (bodySSA, state4) ->
        // Emit return
        let returnOp = 
            if returnType = UnitType then
                "  func.return"
            else
                sprintf "  func.return %s : %s" bodySSA (mlirTypeToString (mapOakTypeToMLIR returnType))
        
        let state5 = OperationEmission.emitOperation returnOp state4
        let state6 = OperationEmission.emitOperation "}" state5
        
        match SSAGeneration.popScope state6 with
        | Ok finalState -> Ok finalState
        | Error e -> Error e
    | Error e -> Error e

/// Entry point conversion
let convertEntryPoint (expr: OakExpression) (state: MLIRGenerationState) : Result<MLIRGenerationState, string> =
    // For Windows command-line apps, main takes argc and argv
    let state1 = OperationEmission.emitOperation "func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 {" state
    let state2 = SSAGeneration.pushScope state1
    
    // Bind argc and argv if needed
    let state3 = SSAGeneration.bindVariable "argc" "%arg0" state2
    let state4 = SSAGeneration.bindVariable "argv" "%arg1" state3
    
    // Convert the entry point expression
    match convertExpression expr state4 with
    | Ok (exprSSA, state5) ->
        // Entry point must return i32 (exit code)
        let finalSSA = 
            match expr with
            | Literal(IntLiteral(_)) -> exprSSA
            | _ ->
                // If expression doesn't return int, assume it returns unit and we return 0
                let (constSSA, newState) = OperationEmission.emitConstant "0" (Integer 32) state5
                constSSA
        
        let state6 = OperationEmission.emitOperation (sprintf "  func.return %s : i32" finalSSA) state5
        let state7 = OperationEmission.emitOperation "}" state6
        
        match SSAGeneration.popScope state7 with
        | Ok finalState -> Ok finalState
        | Error e -> Error e
    | Error e -> Error e

/// Convert a declaration
let convertDeclaration (decl: OakDeclaration) (state: MLIRGenerationState) : Result<MLIRGenerationState, string> =
    match decl with
    | FunctionDecl(name, parameters, returnType, body) ->
        convertFunctionDeclaration name parameters returnType body state
    | EntryPoint(expr) ->
        convertEntryPoint expr state
    | TypeDecl(name, _) ->
        let comment = sprintf "// Type declaration: %s" name
        Ok (OperationEmission.emitOperation comment state)
    | ExternalDecl(name, paramTypes, returnType, _) ->
        let mlirParamTypes = paramTypes |> List.map (mapOakTypeToMLIR >> mlirTypeToString)
        let mlirReturnType = mapOakTypeToMLIR returnType |> mlirTypeToString
        let signature = sprintf "(%s) -> %s" (String.concat ", " mlirParamTypes) mlirReturnType
        Ok (OperationEmission.emitExternalDeclaration name signature state)

/// Main MLIR generation function
let generateMLIR (program: OakProgram) : Result<MLIRModuleOutput, string> =
    if program.Modules.IsEmpty then
        Error "Program must contain at least one module"
    else
        let mainModule = program.Modules.[0]
        
        if mainModule.Declarations.IsEmpty then
            Error (sprintf "Module '%s' must contain at least one declaration" mainModule.Name)
        else
            let initialState = {
                SSACounter = 0
                CurrentScope = Map.empty
                ScopeStack = []
                GeneratedOperations = []
                CurrentFunction = None
                ExternalDeclarations = Set.empty
                StringConstants = Map.empty
            }
            
            // Generate module header
            let moduleHeader = sprintf "module %s {" mainModule.Name
            let state1 = OperationEmission.emitOperation moduleHeader initialState
            
            // Convert all declarations, accumulating state
            let rec processDeclarations (declarations: OakDeclaration list) (currentState: MLIRGenerationState) =
                match declarations with
                | [] -> Ok currentState
                | decl :: rest ->
                    match convertDeclaration decl currentState with
                    | Ok newState -> processDeclarations rest newState
                    | Error e -> Error e
            
            match processDeclarations mainModule.Declarations state1 with
            | Ok state2 ->
                // Add module closing
                let finalState = OperationEmission.emitOperation "}" state2
                let operations = List.rev finalState.GeneratedOperations
                
                Ok {
                    ModuleName = mainModule.Name
                    Operations = operations
                    SSAMappings = finalState.CurrentScope
                    TypeMappings = Map.empty
                    Diagnostics = []
                }
            | Error e -> Error e

/// Generates complete MLIR module text from Oak AST
let generateMLIRModuleText (program: OakProgram) : Result<string, string> =
    match generateMLIR program with
    | Ok mlirOutput -> 
        let moduleText = String.concat "\n" mlirOutput.Operations
        Ok moduleText
    | Error e -> Error e