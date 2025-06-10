module Core.MLIRGeneration.XParsecMLIRGenerator

open System
open System.Text
open Dabbit.Parsing.OakAst
open Core.MLIRGeneration.TypeSystem
open Core.MLIRGeneration.Operations
open Core.MLIRGeneration.Dialect

/// MLIR generation state for tracking SSA values and scopes
type MLIRGenerationState = {
    SSACounter: int
    CurrentScope: Map<string, string>
    ScopeStack: Map<string, string> list
    GeneratedOperations: string list
    CurrentFunction: string option
    StringConstants: Map<string, string>
    CurrentDialect: MLIRDialect
    ErrorContext: string list
}

/// MLIR module output with complete module information
type MLIRModuleOutput = {
    ModuleName: string
    Operations: string list
    SSAMappings: Map<string, string>
    TypeMappings: Map<string, MLIRType>
    Diagnostics: string list
}

/// Creates a clean initial state for MLIR generation
let createInitialState () : MLIRGenerationState = 
    {
        SSACounter = 0
        CurrentScope = Map.empty
        ScopeStack = []
        GeneratedOperations = []
        CurrentFunction = None
        StringConstants = Map.empty
        CurrentDialect = Func
        ErrorContext = []
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
    
    /// Pushes current scope onto stack and creates a new empty scope
    let pushScope (state: MLIRGenerationState) : MLIRGenerationState =
        { 
            state with 
                ScopeStack = state.CurrentScope :: state.ScopeStack
                CurrentScope = Map.empty 
        }
    
    /// Pops scope from stack and restores previous scope
    let popScope (state: MLIRGenerationState) : MLIRGenerationState option =
        match state.ScopeStack with
        | scope :: rest ->
            Some { 
                state with 
                    CurrentScope = scope
                    ScopeStack = rest 
            }
        | [] -> None

/// Core MLIR operation emission functions
module Emitter =
    /// Emits a raw MLIR operation string
    let emit (operation: string) (state: MLIRGenerationState) : MLIRGenerationState =
        { state with GeneratedOperations = operation :: state.GeneratedOperations }
    
    /// Declares an external function in MLIR
    let declareExternal (name: string) (signature: string) (state: MLIRGenerationState) : MLIRGenerationState =
        emit (sprintf "func.func private @%s %s" name signature) state
    
    /// Registers a string constant and returns its global name
    let registerString (value: string) (state: MLIRGenerationState) : string * MLIRGenerationState =
        match Map.tryFind value state.StringConstants with
        | Some existingName -> 
            (existingName, state)
        | None ->
            let constName = sprintf "@str_%d" state.StringConstants.Count
            
            // Emit global string constant
            let escapedValue = value.Replace("\\", "\\\\").Replace("\"", "\\\"")
            let constSize = escapedValue.Length + 1 // +1 for null terminator
            let declaration = 
                sprintf "memref.global constant %s = dense<\"%s\\00\"> : memref<%dxi8>" 
                        constName escapedValue constSize
            
            // Add to state
            let newState = { 
                state with 
                    StringConstants = Map.add value constName state.StringConstants
                    GeneratedOperations = declaration :: state.GeneratedOperations
            }
            (constName, newState)
    
    /// Creates a constant value in MLIR
    let constant (value: string) (mlirType: MLIRType) (state: MLIRGenerationState) : string * MLIRGenerationState =
        let (resultId, state1) = SSA.generateValue "const" state
        let typeStr = mlirTypeToString mlirType
        let state2 = emit (sprintf "  %s = arith.constant %s : %s" resultId value typeStr) state1
        (resultId, state2)
    
    /// Creates an arithmetic operation in MLIR
    let arithmetic (op: string) (lhs: string) (rhs: string) (resultType: MLIRType) (state: MLIRGenerationState) : string * MLIRGenerationState =
        let (resultId, state1) = SSA.generateValue "arith" state
        let typeStr = mlirTypeToString resultType
        let state2 = emit (sprintf "  %s = arith.%s %s, %s : %s" resultId op lhs rhs typeStr) state1
        (resultId, state2)
    
    /// Creates a function call in MLIR
    let call (funcName: string) (args: string list) (resultType: MLIRType option) (state: MLIRGenerationState) : string * MLIRGenerationState =
        match resultType with
        | Some returnType ->
            let (resultId, state1) = SSA.generateValue "call" state
            let argStr = String.concat ", " args
            let typeStr = mlirTypeToString returnType
            
            // Create parameter type list from argument types (simplified)
            let paramTypeStr = String.concat ", " (List.replicate args.Length "i32")
            
            let state2 = emit (sprintf "  %s = func.call @%s(%s) : (%s) -> %s" 
                                     resultId funcName argStr paramTypeStr typeStr) state1
            (resultId, state2)
        | None ->
            // Function with no return value
            let argStr = String.concat ", " args
            let paramTypeStr = String.concat ", " (List.replicate args.Length "i32")
            
            let state1 = emit (sprintf "  func.call @%s(%s) : (%s) -> ()" 
                                   funcName argStr paramTypeStr) state
            let (dummyId, state2) = SSA.generateValue "void" state1
            (dummyId, state2)

/// Core expression conversion functions
module ExpressionConversion =
    /// Converts Oak literal to MLIR constant
    let rec convertLiteral (literal: OakLiteral) (state: MLIRGenerationState) : string * MLIRGenerationState =
        match literal with
        | IntLiteral value ->
            Emitter.constant (string value) (Integer 32) state
        | FloatLiteral value ->
            Emitter.constant (sprintf "%f" value) (Float 32) state
        | BoolLiteral value ->
            Emitter.constant (if value then "1" else "0") (Integer 1) state
        | StringLiteral value ->
            // Register string as global and get pointer
            let (globalName, state1) = Emitter.registerString value state
            let (ptrResult, state2) = SSA.generateValue "str_ptr" state1
            let state3 = Emitter.emit (sprintf "  %s = memref.get_global %s : memref<?xi8>" ptrResult globalName) state2
            (ptrResult, state3)
        | UnitLiteral ->
            // Unit type has no value in MLIR - return a dummy constant
            Emitter.constant "0" (Integer 32) state
        | ArrayLiteral elements ->
            // Create stack-allocated array and initialize elements
            let elementType = 
                if elements.IsEmpty then Integer 32
                else 
                    // Try to infer element type from first element
                    match elements.Head with
                    | Literal(IntLiteral _) -> Integer 32
                    | Literal(FloatLiteral _) -> Float 32
                    | Literal(BoolLiteral _) -> Integer 1
                    | _ -> Integer 32
            
            let arraySize = elements.Length
            
            // Allocate array
            let (arrayResult, state1) = SSA.generateValue "array" state
            let state2 = Emitter.emit (sprintf "  %s = memref.alloca(%d) : memref<%dx%s>" 
                               arrayResult arraySize arraySize (mlirTypeToString elementType)) state1
            
            // Initialize elements (simplified approach)
            let initializeElement (idx, element) (currentState: MLIRGenerationState) =
                let (elemValue, state1) = convertExpression element currentState
                let (idxResult, state2) = SSA.generateValue "idx" state1
                let (idxConst, state3) = Emitter.constant (string idx) (Integer 32) state2
                let state4 = Emitter.emit (sprintf "  memref.store %s, %s[%s] : %s, memref<%dx%s>" 
                                    elemValue arrayResult idxConst 
                                    (mlirTypeToString elementType) 
                                    arraySize 
                                    (mlirTypeToString elementType)) state3
                state4
            
            let state3 = List.fold (fun s (idx, elem) -> initializeElement (idx, elem) s) 
                           state2 (List.indexed elements)
            
            (arrayResult, state3)
    
    /// Converts Oak expression to MLIR operations with simplified approach
    and convertExpression (expr: OakExpression) (state: MLIRGenerationState) : string * MLIRGenerationState =
        match expr with
        | Literal literal ->
            convertLiteral literal state
        
        | Variable name ->
            // Look up variable in scope
            match SSA.lookupVariable name state with
            | Some value -> (value, state)
            | None -> 
                // Variable not found - create a dummy value and diagnostic
                let (dummyValue, state1) = SSA.generateValue "unknown" state
                let state2 = { state1 with ErrorContext = sprintf "Variable '%s' not found" name :: state1.ErrorContext }
                (dummyValue, state2)
        
        | Application(func, args) ->
            match func with
            | Variable funcName ->
                // Convert arguments
                let convertArg (accState, accArgs) arg =
                    let (argValue, newState) = convertExpression arg accState
                    (newState, argValue :: accArgs)
                
                let (state1, argValues) = List.fold convertArg (state, []) args
                
                // Call function (simplified - always returns i32)
                Emitter.call funcName (List.rev argValues) (Some (Integer 32)) state1
            | _ ->
                // Unsupported function expression - return dummy value
                let (dummyValue, state1) = SSA.generateValue "unsupported" state
                let state2 = { state1 with ErrorContext = "Only direct function calls supported" :: state1.ErrorContext }
                (dummyValue, state2)
        
        | Let(name, value, body) ->
            // Convert the bound value
            let (valueResult, state1) = convertExpression value state
            
            // Bind in current scope
            let state2 = SSA.bindVariable name valueResult state1
            
            // Convert body
            convertExpression body state2
        
        | IfThenElse(cond, thenExpr, elseExpr) ->
            // Generate unique labels for branches
            let thenLabel = sprintf "then_%d" (state.SSACounter + 1)
            let elseLabel = sprintf "else_%d" (state.SSACounter + 2)
            let endLabel = sprintf "endif_%d" (state.SSACounter + 3)
            
            // Evaluate condition
            let (condResult, state1) = convertExpression cond state
            
            // Create a result variable
            let (resultVar, state2) = SSA.generateValue "if_result" state1
            
            // Emit conditional branch
            let state3 = Emitter.emit (sprintf "  cf.cond_br %s, ^%s, ^%s" condResult thenLabel elseLabel) state2
            
            // Then branch
            let state4 = Emitter.emit (sprintf "^%s:" thenLabel) state3
            let (thenResult, state5) = convertExpression thenExpr state4
            let state6 = Emitter.emit (sprintf "  cf.br ^%s(%s : i32)" endLabel thenResult) state5
            
            // Else branch
            let state7 = Emitter.emit (sprintf "^%s:" elseLabel) state6
            let (elseResult, state8) = convertExpression elseExpr state7
            let state9 = Emitter.emit (sprintf "  cf.br ^%s(%s : i32)" endLabel elseResult) state8
            
            // Join point
            let state10 = Emitter.emit (sprintf "^%s(%s: i32):" endLabel resultVar) state9
            
            (resultVar, state10)
        
        | Sequential(first, second) ->
            // Evaluate first expression and discard result
            let (_, state1) = convertExpression first state
            
            // Evaluate and return second expression
            convertExpression second state1
        
        | FieldAccess(target, fieldName) ->
            let (targetResult, state1) = convertExpression target state
            
            // Extract field using GEP (simplified approach)
            let (fieldResult, state2) = SSA.generateValue "field" state1
            
            // This is a simplification - real implementation would need type information
            let state3 = Emitter.emit (sprintf "  %s = llvm.extractvalue %s[0] : !llvm.struct<i32, i32>" 
                                   fieldResult targetResult) state2
            
            (fieldResult, state3)
        
        | MethodCall(target, methodName, args) ->
            // Convert target and arguments (simplified)
            let (targetResult, state1) = convertExpression target state
            
            // Convert arguments
            let convertArg (accState, accArgs) arg =
                let (argValue, newState) = convertExpression arg accState
                (newState, argValue :: accArgs)
            
            let (state2, argValues) = List.fold convertArg (state1, []) args
            
            // Call method as a regular function (simplified)
            let fullMethodName = sprintf "%s_%s" (targetResult.TrimStart('%')) methodName
            Emitter.call fullMethodName (List.rev argValues) (Some (Integer 32)) state2
            
        | IOOperation(ioType, args) ->
            match ioType with
            | Printf formatStr | Printfn formatStr ->
                // Register format string
                let (formatGlobal, state1) = Emitter.registerString formatStr state
                
                // Get format string pointer
                let (formatPtr, state2) = SSA.generateValue "fmt_ptr" state1
                let state3 = Emitter.emit (sprintf "  %s = memref.get_global %s : memref<?xi8>" formatPtr formatGlobal) state2
                
                // Convert arguments
                let convertArg (accState, accArgs) arg =
                    let (argValue, newState) = convertExpression arg accState
                    (newState, argValue :: accArgs)
                
                let (state4, argValues) = List.fold convertArg (state3, []) args
                
                // Declare printf
                let state5 = Emitter.declareExternal "printf" "(memref<?xi8>, ...) -> i32" state4
                
                // Call printf with format and args
                let allArgs = formatPtr :: List.rev argValues
                let (printfResult, state6) = SSA.generateValue "printf_result" state5
                
                // Build parameter type string based on arg count
                let paramTypes = "memref<?xi8>" :: List.replicate argValues.Length "i32"
                let paramTypeStr = String.concat ", " paramTypes
                
                let state7 = Emitter.emit (sprintf "  %s = func.call @printf(%s) : (%s) -> i32" 
                                       printfResult (String.concat ", " allArgs) paramTypeStr) state6
                
                // For printfn, also print newline
                if ioType = Printfn formatStr then
                    let (nlGlobal, state8) = Emitter.registerString "\n" state7
                    let (nlPtr, state9) = SSA.generateValue "nl_ptr" state8
                    let state10 = Emitter.emit (sprintf "  %s = memref.get_global %s : memref<?xi8>" nlPtr nlGlobal) state9
                    
                    let (nlResult, state11) = SSA.generateValue "nl_result" state10
                    let state12 = Emitter.emit (sprintf "  %s = func.call @printf(%s) : (memref<?xi8>) -> i32" nlResult nlPtr) state11
                    (printfResult, state12)
                else
                    (printfResult, state7)
            
            | ReadLine ->
                // Declare fgets and stdin
                let state1 = Emitter.declareExternal "fgets" "(memref<?xi8>, i32, memref<?xi8>) -> memref<?xi8>" state
                let state2 = Emitter.declareExternal "__stdinp" "() -> memref<?xi8>" state1
                
                // Allocate buffer
                let (bufferResult, state3) = SSA.generateValue "buffer" state2
                let state4 = Emitter.emit (sprintf "  %s = memref.alloca() : memref<256xi8>" bufferResult) state3
                
                // Get buffer size constant
                let (sizeResult, state5) = SSA.generateValue "buf_size" state4
                let (_, state6) = Emitter.constant "256" (Integer 32) state5
                
                // Get stdin handle
                let (stdinResult, state7) = SSA.generateValue "stdin" state6
                let state8 = Emitter.emit (sprintf "  %s = func.call @__stdinp() : () -> memref<?xi8>" stdinResult) state7
                
                // Call fgets
                let (fgetsResult, state9) = SSA.generateValue "fgets_result" state8
                let state10 = Emitter.emit (sprintf "  %s = func.call @fgets(%s, %s, %s) : (memref<256xi8>, i32, memref<?xi8>) -> memref<?xi8>" 
                                       fgetsResult bufferResult sizeResult stdinResult) state9
                
                (bufferResult, state10)
            
            | _ ->
                // Unsupported I/O operation
                let (dummyValue, state1) = SSA.generateValue "unsupported_io" state
                let state2 = { state1 with ErrorContext = "Unsupported I/O operation" :: state1.ErrorContext }
                (dummyValue, state2)

/// Function and module level conversion functions
module DeclarationConversion =
    /// Converts function declaration to MLIR function
    let convertFunction (name: string) (parameters: (string * OakType) list) (returnType: OakType) (body: OakExpression) (state: MLIRGenerationState) : MLIRGenerationState =
        // Generate function signature
        let paramTypes = parameters |> List.map (snd >> mapOakTypeToMLIR >> mlirTypeToString)
        let returnTypeStr = mlirTypeToString (mapOakTypeToMLIR returnType)
        
        // Start function definition
        let paramStr = 
            parameters 
            |> List.mapi (fun i (name, typ) -> 
                sprintf "%%arg%d: %s" i (mlirTypeToString (mapOakTypeToMLIR typ)))
            |> String.concat ", "
            
        let state1 = Emitter.emit (sprintf "func.func @%s(%s) -> %s {" name paramStr returnTypeStr) state
        
        // Create new scope for function body
        let state2 = SSA.pushScope state1
        
        // Bind parameters to arguments
        let bindParams (state: MLIRGenerationState) =
            parameters 
            |> List.mapi (fun i (paramName, _) -> 
                SSA.bindVariable paramName (sprintf "%%arg%d" i) state)
            |> List.fold (fun s f -> f) state
        
        let state3 = bindParams state2
        
        // Convert function body
        let (bodyResult, state4) = ExpressionConversion.convertExpression body state3
        
        // Generate return statement
        let state5 = 
            if returnType = UnitType then
                Emitter.emit "  func.return" state4
            else
                Emitter.emit (sprintf "  func.return %s : %s" bodyResult returnTypeStr) state4
        
        // End function
        let state6 = Emitter.emit "}" state5
        
        // Restore previous scope
        match SSA.popScope state6 with
        | Some state7 -> state7
        | None -> state6 // This shouldn't happen with balanced push/pop

/// Main entry point for MLIR generation
let generateMLIR (program: OakProgram) : MLIRModuleOutput =
    let initialState = createInitialState ()
    
    // Process each module
    let processModule (state: MLIRGenerationState) (mdl: OakModule) =
        // Module header
        let state1 = Emitter.emit (sprintf "module @%s {" mdl.Name) state
        
        // Process each declaration
        let processDeclFold currState decl =
            match decl with
            | FunctionDecl(name, params', returnType, body) ->
                DeclarationConversion.convertFunction name params' returnType body currState
            
            | TypeDecl(name, _) ->
                // Ignore type declarations - they don't generate MLIR code
                currState
                
            | EntryPoint(expr) ->
                // Generate a main function
                DeclarationConversion.convertFunction "main" 
                    [("argc", IntType); ("argv", ArrayType(StringType))] 
                    IntType expr currState
                
            | ExternalDecl(name, paramTypes, returnType, libraryName) ->
                // Generate external function declaration
                let paramTypeStrs = paramTypes |> List.map (mapOakTypeToMLIR >> mlirTypeToString)
                let returnTypeStr = mlirTypeToString (mapOakTypeToMLIR returnType)
                
                let paramStr = 
                    paramTypeStrs
                    |> List.mapi (fun i typ -> sprintf "%%arg%d: %s" i typ)
                    |> String.concat ", "
                
                Emitter.emit (sprintf "func.func private @%s(%s) -> %s attributes {ffi.library = \"%s\"}" 
                                 name paramStr returnTypeStr libraryName) currState
        
        let state2 = List.fold processDeclFold state1 mdl.Declarations
        
        // Close module
        Emitter.emit "}" state2
    
    // Process all modules (typically just one)
    let finalState = 
        match program.Modules with
        | [] -> initialState
        | mdl :: _ -> processModule initialState mdl
    
    // Return the final output
    {
        ModuleName = 
            match program.Modules with
            | [] -> "main"
            | mdl :: _ -> mdl.Name
        Operations = List.rev finalState.GeneratedOperations
        SSAMappings = finalState.CurrentScope
        TypeMappings = Map.empty
        Diagnostics = finalState.ErrorContext
    }

/// Generates complete MLIR module text from Oak AST
let generateMLIRModuleText (program: OakProgram) : Core.XParsec.Foundation.CompilerResult<string> =
    try
        let mlirOutput = generateMLIR program
        
        // Join operations with newlines
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