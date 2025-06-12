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
    ModuleLevelDeclarations: string list  // Add this for module-level items
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
        ModuleLevelDeclarations = []
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
    /// Emits a raw MLIR operation string to function body
    let emit (operation: string) (state: MLIRGenerationState) : MLIRGenerationState =
        { state with GeneratedOperations = operation :: state.GeneratedOperations }
    
    /// Emits a module-level declaration (globals, external functions)
    let emitModuleLevel (declaration: string) (state: MLIRGenerationState) : MLIRGenerationState =
        { state with ModuleLevelDeclarations = declaration :: state.ModuleLevelDeclarations }
    
    /// Declares an external function at module level
    let declareExternal (name: string) (signature: string) (state: MLIRGenerationState) : MLIRGenerationState =
        emitModuleLevel (sprintf "func.func private @%s %s" name signature) state
    
    /// Registers a string constant at module level and returns its global name
    let registerString (value: string) (state: MLIRGenerationState) : string * MLIRGenerationState =
        match Map.tryFind value state.StringConstants with
        | Some existingName -> 
            (existingName, state)
        | None ->
            let constName = sprintf "@str_%d" state.StringConstants.Count
            
            // Emit global string constant at module level
            let escapedValue = value.Replace("\\", "\\\\").Replace("\"", "\\\"")
            let constSize = escapedValue.Length + 1 // +1 for null terminator
            let declaration = 
                sprintf "memref.global constant %s = dense<\"%s\\00\"> : memref<%dxi8>" 
                        constName escapedValue constSize
            
            // Add to module-level declarations
            let newState = { 
                state with 
                    StringConstants = Map.add value constName state.StringConstants
                    ModuleLevelDeclarations = declaration :: state.ModuleLevelDeclarations
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
            let elementType = Integer 32  // Simplified
            let arraySize = elements.Length
            
            // Allocate array
            let (arrayResult, state1) = SSA.generateValue "array" state
            let state2 = Emitter.emit (sprintf "  %s = memref.alloca(%d) : memref<%dx%s>" 
                               arrayResult arraySize arraySize (mlirTypeToString elementType)) state1
            
            (arrayResult, state2)
    
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
                // Variable not found - create a dummy value
                let (dummyValue, state1) = SSA.generateValue "unknown" state
                (dummyValue, state1)
        
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
                (dummyValue, state1)
        
        | Let(name, value, body) ->
            // Convert the bound value
            let (valueResult, state1) = convertExpression value state
            
            // Bind in current scope
            let state2 = SSA.bindVariable name valueResult state1
            
            // Convert body
            convertExpression body state2
        
        | Sequential(first, second) ->
            // Evaluate first expression and discard result
            let (_, state1) = convertExpression first state
            
            // Evaluate and return second expression
            convertExpression second state1
        
        | IOOperation(ioType, args) ->
            match ioType with
            | Printf formatStr | Printfn formatStr ->
                // Register format string at module level
                let (formatGlobal, state1) = Emitter.registerString formatStr state
                
                // Declare printf at module level if not already done
                let state2 = Emitter.declareExternal "printf" "(memref<?xi8>, ...) -> i32" state1
                
                // Get format string pointer
                let (formatPtr, state3) = SSA.generateValue "fmt_ptr" state2
                let state4 = Emitter.emit (sprintf "  %s = memref.get_global %s : memref<?xi8>" formatPtr formatGlobal) state3
                
                // Call printf
                let (printfResult, state5) = SSA.generateValue "printf_result" state4
                let state6 = Emitter.emit (sprintf "  %s = func.call @printf(%s) : (memref<?xi8>) -> i32" printfResult formatPtr) state5
                
                (printfResult, state6)
            
            | _ ->
                // Unsupported I/O operation
                let (dummyValue, state1) = SSA.generateValue "unsupported_io" state
                (dummyValue, state1)
        
        | _ ->
            // Handle other expression types with dummy implementation
            let (dummyValue, state1) = SSA.generateValue "todo" state
            (dummyValue, state1)

/// Function and module level conversion functions
module DeclarationConversion =
    /// Converts function declaration to MLIR function with minimal output
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
        | None -> state6

/// Main entry point for MLIR generation with correct structure
let generateMLIR (program: OakProgram) : MLIRModuleOutput =
    let initialState = createInitialState ()
    
    // Process each module
    let processModule (state: MLIRGenerationState) (mdl: OakModule) =
        // Start with module header
        let state1 = Emitter.emit (sprintf "module @%s {" mdl.Name) state
        
        // Process each declaration to collect module-level items and functions
        let processDeclFold currState decl =
            match decl with
            | FunctionDecl(name, params', returnType, body) ->
                DeclarationConversion.convertFunction name params' returnType body currState
            
            | EntryPoint(expr) ->
                DeclarationConversion.convertFunction "main" 
                    [("argc", IntType); ("argv", ArrayType(StringType))] 
                    IntType expr currState
                
            | ExternalDecl(name, paramTypes, returnType, libraryName) ->
                let paramTypeStrs = paramTypes |> List.map (mapOakTypeToMLIR >> mlirTypeToString)
                let returnTypeStr = mlirTypeToString (mapOakTypeToMLIR returnType)
                
                let paramStr = 
                    paramTypeStrs
                    |> List.mapi (fun i typ -> sprintf "%%arg%d: %s" i typ)
                    |> String.concat ", "
                
                Emitter.emitModuleLevel (sprintf "func.func private @%s(%s) -> %s attributes {ffi.library = \"%s\"}" 
                                 name paramStr returnTypeStr libraryName) currState
            
            | _ -> currState
        
        let state2 = List.fold processDeclFold state1 mdl.Declarations
        
        // Close module
        let state3 = Emitter.emit "}" state2
        state3
    
    // Process all modules (typically just one)
    let finalState = 
        match program.Modules with
        | [] -> initialState
        | mdl :: _ -> processModule initialState mdl
    
    // Combine module-level declarations with function operations in correct order
    let moduleHeader = sprintf "module @%s {" (
        match program.Modules with
        | [] -> "main"
        | mdl :: _ -> mdl.Name
    )
    
    let moduleFooter = "}"
    
    // Structure: module header, module-level declarations, functions, module footer
    let allOperations = 
        [moduleHeader] @
        (List.rev finalState.ModuleLevelDeclarations) @
        (List.rev finalState.GeneratedOperations |> List.tail |> List.rev |> List.tail) @  // Remove duplicate module header/footer
        [moduleFooter]
    
    // Return the final output with correct operations
    {
        ModuleName = 
            match program.Modules with
            | [] -> "main"
            | mdl :: _ -> mdl.Name
        Operations = allOperations
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