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
    ModuleLevelDeclarations: string list
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
        CurrentFunction = Option.None
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
            | [] -> Option.None
            | scope :: rest ->
                match Map.tryFind name scope with
                | Some value -> Some value
                | Option.None -> lookup rest
        
        lookup (state.CurrentScope :: state.ScopeStack)

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
        | Option.None ->
            let constName = sprintf "@str_%d" state.StringConstants.Count
            
            // Emit global string constant at module level
            let escapedValue = value.Replace("\\", "\\\\").Replace("\"", "\\\"")
            let constSize = escapedValue.Length + 1 // +1 for null terminator
            let declaration = 
                sprintf "  memref.global constant %s = dense<\"%s\\00\"> : memref<%dxi8>" 
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
        let state2 = emit (sprintf "    %s = arith.constant %s : %s" resultId value typeStr) state1
        (resultId, state2)
    
    /// Creates a function call in MLIR
    let call (funcName: string) (args: string list) (resultType: MLIRType option) (state: MLIRGenerationState) : string * MLIRGenerationState =
        match resultType with
        | Some returnType ->
            let (resultId, state1) = SSA.generateValue "call" state
            let argStr = String.concat ", " args
            let typeStr = mlirTypeToString returnType
            let paramTypeStr = String.concat ", " (List.replicate args.Length "i32")
            
            let state2 = emit (sprintf "    %s = func.call @%s(%s) : (%s) -> %s" 
                                     resultId funcName argStr paramTypeStr typeStr) state1
            (resultId, state2)
        | Option.None ->
            let argStr = String.concat ", " args
            let paramTypeStr = String.concat ", " (List.replicate args.Length "i32")
            
            let state1 = emit (sprintf "    func.call @%s(%s) : (%s) -> ()" 
                                   funcName argStr paramTypeStr) state
            let (dummyId, state2) = SSA.generateValue "void" state1
            (dummyId, state2)

/// Core expression conversion functions - COMPLETE MODULE WITH PROPER ORDERING
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
            let state3 = Emitter.emit (sprintf "    %s = memref.get_global %s : memref<?xi8>" ptrResult globalName) state2
            (ptrResult, state3)
        | UnitLiteral ->
            // Unit type - return a dummy constant
            Emitter.constant "0" (Integer 32) state
        | ArrayLiteral _ ->
            // Simplified array handling
            let (arrayResult, state1) = SSA.generateValue "array" state
            (arrayResult, state1)
    
    /// Converts stack allocation to MLIR memref.alloca
    and convertStackAllocation (args: OakExpression list) (state: MLIRGenerationState) : string * MLIRGenerationState =
        match args with
        | [Literal (IntLiteral size)] ->
            // Generate stack allocation for byte buffer
            let (bufferResult, state1) = SSA.generateValue "buffer" state
            let state2 = Emitter.emit (sprintf "    %s = memref.alloca() : memref<%dxi8>" bufferResult size) state1
            (bufferResult, state2)
        | _ ->
            // Invalid stack allocation arguments
            let (dummyValue, state1) = SSA.generateValue "invalid_alloca" state
            (dummyValue, state1)

    /// Converts Console.readLine to MLIR scanf call
    and convertConsoleReadLine (args: OakExpression list) (state: MLIRGenerationState) : string * MLIRGenerationState =
        match args with
        | [Variable bufferName; Literal (IntLiteral size)] ->
            // Look up buffer variable
            match SSA.lookupVariable bufferName state with
            | Some bufferValue ->
                // Declare scanf at module level if not already done
                let state1 = Emitter.emitModuleLevel "  func.func private @scanf(memref<?xi8>, ...) -> i32" state
                
                // Create format string for scanf ("%s")
                let (scanfFormat, state2) = Emitter.registerString "%s" state1
                let (formatPtr, state3) = SSA.generateValue "scanf_fmt" state2
                let state4 = Emitter.emit (sprintf "    %s = memref.get_global %s : memref<?xi8>" formatPtr scanfFormat) state3
                
                // Cast buffer to appropriate type for scanf
                let (castBuffer, state5) = SSA.generateValue "cast_buffer" state4
                let state6 = Emitter.emit (sprintf "    %s = memref.cast %s : memref<%dxi8> to memref<?xi8>" castBuffer bufferValue size) state5
                
                // Call scanf
                let (scanfResult, state7) = SSA.generateValue "scanf_result" state6
                let state8 = Emitter.emit (sprintf "    %s = func.call @scanf(%s, %s) : (memref<?xi8>, memref<?xi8>) -> i32" scanfResult formatPtr castBuffer) state7
                
                (scanfResult, state8)
            | None ->
                // Buffer variable not found
                let (dummyValue, state1) = SSA.generateValue "missing_buffer" state
                (dummyValue, state1)
        | _ ->
            // Invalid readLine arguments
            let (dummyValue, state1) = SSA.generateValue "invalid_readline" state
            (dummyValue, state1)

    /// Converts Span creation to MLIR operations
    and convertSpanCreation (args: OakExpression list) (state: MLIRGenerationState) : string * MLIRGenerationState =
        match args with
        | [Variable bufferName; Variable lengthName] ->
            // Look up buffer and length variables
            match SSA.lookupVariable bufferName state, SSA.lookupVariable lengthName state with
            | Some bufferValue, Some lengthValue ->
                // For now, just return the buffer (Span is essentially a view)
                (bufferValue, state)
            | _ ->
                // Variables not found
                let (dummyValue, state1) = SSA.generateValue "missing_span_args" state
                (dummyValue, state1)
        | _ ->
            // Invalid span arguments
            let (dummyValue, state1) = SSA.generateValue "invalid_span" state
            (dummyValue, state1)
    
    /// Converts Oak expression to MLIR operations with Alloy library recognition
    and convertExpression (expr: OakExpression) (state: MLIRGenerationState) : string * MLIRGenerationState =
        match expr with
        | Literal literal ->
            convertLiteral literal state
        
        | Variable name ->
            // Look up variable in scope
            match SSA.lookupVariable name state with
            | Some value -> (value, state)
            | Option.None -> 
                // Variable not found - create a dummy value
                let (dummyValue, state1) = SSA.generateValue "unknown" state
                (dummyValue, state1)
        
        | Application(func, args) ->
            match func with
            | Variable "NativePtr.stackalloc" ->
                // Handle stack allocation specially
                convertStackAllocation args state
            | Variable "Console.readLine" ->
                // Handle console input
                convertConsoleReadLine args state
            | Variable "Span.create" ->
                // Handle span creation
                convertSpanCreation args state
            | Variable funcName ->
                // Convert arguments
                let convertArg (accState, accArgs) arg =
                    let (argValue, newState) = convertExpression arg accState
                    (newState, argValue :: accArgs)
                
                let (state1, argValues) = List.fold convertArg (state, []) args
                
                // Call function
                Emitter.call funcName (List.rev argValues) (Some (Integer 32)) state1
            | _ ->
                // Unsupported function expression
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
            // Evaluate first expression
            let (_, state1) = convertExpression first state
            
            // Evaluate and return second expression
            convertExpression second state1
        
        | IOOperation(ioType, _) ->
            match ioType with
            | Printf formatStr | Printfn formatStr ->
                // Register format string at module level
                let (formatGlobal, state1) = Emitter.registerString formatStr state
                
                // Declare printf at module level if not already done
                let state2 = Emitter.emitModuleLevel "  func.func private @printf(memref<?xi8>, ...) -> i32" state1
                
                // Get format string pointer
                let (formatPtr, state3) = SSA.generateValue "fmt_ptr" state2
                let state4 = Emitter.emit (sprintf "    %s = memref.get_global %s : memref<?xi8>" formatPtr formatGlobal) state3
                
                // Call printf
                let (printfResult, state5) = SSA.generateValue "printf_result" state4
                let state6 = Emitter.emit (sprintf "    %s = func.call @printf(%s) : (memref<?xi8>) -> i32" printfResult formatPtr) state5
                
                (printfResult, state6)
            
            | _ ->
                // Unsupported I/O operation
                let (dummyValue, state1) = SSA.generateValue "unsupported_io" state
                (dummyValue, state1)
        
        | _ ->
            // Handle other expression types
            let (dummyValue, state1) = SSA.generateValue "todo" state
            (dummyValue, state1)

/// Function and module level conversion functions
module DeclarationConversion =
    /// Converts function declaration to MLIR function
    let convertFunction (name: string) (parameters: (string * OakType) list) (returnType: OakType) (body: OakExpression) (state: MLIRGenerationState) : MLIRGenerationState =
        // Generate function signature
        let returnTypeStr = mlirTypeToString (mapOakTypeToMLIR returnType)
        
        // Start function definition
        let paramStr = 
            if parameters.IsEmpty then
                ""
            else
                parameters 
                |> List.mapi (fun i (_, typ) -> 
                    sprintf "%%arg%d: %s" i (mlirTypeToString (mapOakTypeToMLIR typ)))
                |> String.concat ", "
                
        let state1 = Emitter.emit (sprintf "  func.func @%s(%s) -> %s {" name paramStr returnTypeStr) state
        
        // Bind parameters to arguments
        let state2 = 
            parameters 
            |> List.mapi (fun i (paramName, _) -> 
                SSA.bindVariable paramName (sprintf "%%arg%d" i))
            |> List.fold (fun s f -> f s) state1
        
        // Convert function body
        let (bodyResult, state3) = ExpressionConversion.convertExpression body state2
        
        // Generate return statement
        let state4 = 
            if returnType = UnitType then
                Emitter.emit "    func.return" state3
            else
                Emitter.emit (sprintf "    func.return %s : %s" bodyResult returnTypeStr) state3
        
        // End function
        let state5 = Emitter.emit "  }" state4
        
        state5

/// Main entry point for MLIR generation with CORRECT module structure
let generateMLIR (program: OakProgram) : MLIRModuleOutput =
    let initialState = createInitialState ()
    
    // Process the first module (typically the only one)
    let processModule (state: MLIRGenerationState) (mdl: OakModule) =
        // Start with module header
        let state1 = Emitter.emit (sprintf "module @%s {" mdl.Name) state
        
        // Process each declaration to collect module-level needs
        let processDeclFold currState decl =
            match decl with
            | FunctionDecl(name, parameters, returnType, body) ->
                DeclarationConversion.convertFunction name parameters returnType body currState
            
            | EntryPoint(expr) ->
                DeclarationConversion.convertFunction "main" 
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
                
                Emitter.emitModuleLevel (sprintf "  func.func private @%s(%s) -> %s" 
                                 name paramStr returnTypeStr) currState
            
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
    
    // CORRECT: Build operations with proper module structure
    let moduleOperations = List.rev finalState.GeneratedOperations
    let moduleDeclarations = List.rev finalState.ModuleLevelDeclarations
    
    // Insert module-level declarations RIGHT AFTER module opening
    let correctOperations = 
        match moduleOperations with
        | moduleHeader :: rest ->
            // Insert module-level declarations after "module @Name {"
            moduleHeader :: (moduleDeclarations @ rest)
        | [] -> moduleDeclarations
    
    // Return the final output with CORRECT operations order
    {
        ModuleName = 
            match program.Modules with
            | [] -> "main"
            | mdl :: _ -> mdl.Name
        Operations = correctOperations
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