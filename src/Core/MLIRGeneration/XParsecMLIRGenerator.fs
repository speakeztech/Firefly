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
            let newState = { 
                state with 
                    StringConstants = Map.add value constName state.StringConstants
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
        let argStr = if args.IsEmpty then "" else String.concat ", " args
        let paramTypeStr = if args.IsEmpty then "" else String.concat ", " (List.replicate args.Length "i32")
        
        match resultType with
        | Some returnType when returnType <> Void ->
            let (resultId, state1) = SSA.generateValue "call" state
            let typeStr = mlirTypeToString returnType
            
            let callStr = 
                if args.IsEmpty then
                    sprintf "    %s = func.call @%s() : () -> %s" resultId funcName typeStr
                else
                    sprintf "    %s = func.call @%s(%s) : (%s) -> %s" 
                            resultId funcName argStr paramTypeStr typeStr
            
            let state2 = emit callStr state1
            (resultId, state2)
        | _ ->
            let callStr = 
                if args.IsEmpty then
                    sprintf "    func.call @%s() : () -> ()" funcName
                else
                    sprintf "    func.call @%s(%s) : (%s) -> ()" funcName argStr paramTypeStr
            
            let state1 = emit callStr state
            let (dummyId, state2) = SSA.generateValue "void" state1
            (dummyId, state2)

/// Core expression conversion functions with proper mutual recursion
let rec convertLiteral (literal: OakLiteral) (state: MLIRGenerationState) : string * MLIRGenerationState =
    match literal with
    | IntLiteral value ->
        Emitter.constant (string value) (Integer 32) state
    | FloatLiteral value ->
        Emitter.constant (sprintf "%f" value) (Float 32) state
    | BoolLiteral value ->
        Emitter.constant (if value then "1" else "0") (Integer 1) state
    | StringLiteral value ->
        let (globalName, state1) = Emitter.registerString value state
        let (ptrResult, state2) = SSA.generateValue "str_ptr" state1
        let state3 = Emitter.emit (sprintf "    %s = memref.get_global %s : memref<?xi8>" ptrResult globalName) state2
        (ptrResult, state3)
    | UnitLiteral ->
        Emitter.constant "0" (Integer 32) state
    | ArrayLiteral _ ->
        let (arrayResult, state1) = SSA.generateValue "array" state
        (arrayResult, state1)

and convertStackAllocation (args: OakExpression list) (state: MLIRGenerationState) : string * MLIRGenerationState =
    match args with
    | [Literal (IntLiteral size)] ->
        let (bufferResult, state1) = SSA.generateValue "buffer" state
        let state2 = Emitter.emit (sprintf "    %s = memref.alloca() : memref<%dxi8>" bufferResult size) state1
        (bufferResult, state2)
    | _ ->
        let (dummyValue, state1) = SSA.generateValue "invalid_alloca" state
        (dummyValue, state1)

and convertConsoleReadLine (args: OakExpression list) (state: MLIRGenerationState) : string * MLIRGenerationState =
    printfn "DEBUG: convertConsoleReadLine called with args: %A" args
    
    match args with
    | [Variable bufferName; Literal (IntLiteral size)] ->
        printfn "DEBUG: Processing Console.readLine(%s, %d)" bufferName size
        
        match SSA.lookupVariable bufferName state with
        | Some bufferValue ->
            printfn "DEBUG: Found buffer variable: %s -> %s" bufferName bufferValue
            
            // Declare scanf with proper types
            let state1 = Emitter.emitModuleLevel "  func.func private @scanf(memref<?xi8>, memref<?xi8>) -> i32" state
            
            // Format string that limits the input size for safety
            let formatStr = sprintf "%%%ds" (size - 1)
            let (scanfFormat, state2) = Emitter.registerString formatStr state1
            
            let (formatPtr, state3) = SSA.generateValue "scanf_fmt" state2
            let state4 = Emitter.emit (sprintf "    %s = memref.get_global %s : memref<?xi8>" formatPtr scanfFormat) state3
            
            // Cast buffer to correct type for scanf
            let (castBuffer, state5) = SSA.generateValue "scanf_buffer" state4
            let state6 = Emitter.emit (sprintf "    %s = memref.cast %s : memref<%dxi8> to memref<?xi8>" castBuffer bufferValue size) state5
            
            let (scanfResult, state7) = SSA.generateValue "scanf_result" state6
            let state8 = Emitter.emit (sprintf "    %s = func.call @scanf(%s, %s) : (memref<?xi8>, memref<?xi8>) -> i32" 
                                      scanfResult formatPtr castBuffer) state7
            
            (scanfResult, state8)
        
        | None ->
            let (dummyValue, state1) = SSA.generateValue "missing_buffer" state
            (dummyValue, state1)
    
    | _ ->
        let (dummyValue, state1) = SSA.generateValue "invalid_readline" state
        (dummyValue, state1)

and convertSpanCreation (args: OakExpression list) (state: MLIRGenerationState) : string * MLIRGenerationState =
    printfn "DEBUG: Span creation with args: %A" args
    match args with
    | [Variable bufferName; Variable lengthName] ->
        printfn "DEBUG: Span creation with buffer %s and length %s" bufferName lengthName
        match SSA.lookupVariable bufferName state, SSA.lookupVariable lengthName state with
        | Some bufferValue, Some lengthValue ->
            // For Span<byte>, we just pass the buffer pointer directly
            printfn "DEBUG: Resolved buffer %s to %s and length %s to %s" bufferName bufferValue lengthName lengthValue
            (bufferValue, state)
        | Some bufferValue, None ->
            printfn "DEBUG: Resolved buffer %s to %s, but length %s not found" bufferName bufferValue lengthName
            (bufferValue, state)
        | None, _ ->
            printfn "DEBUG: Buffer %s not found in scope" bufferName
            let (dummyValue, state1) = SSA.generateValue "missing_span_args" state
            (dummyValue, state1)
    | [expr1; expr2] ->
        printfn "DEBUG: Span creation with non-variable expressions"
        let (bufferValue, state1) = convertExpression expr1 state
        let (lengthValue, state2) = convertExpression expr2 state1
        (bufferValue, state2)
    | _ ->
        printfn "DEBUG: Invalid Span arguments: %A" args
        let (dummyValue, state1) = SSA.generateValue "invalid_span" state
        (dummyValue, state1)

and convertExpression (expr: OakExpression) (state: MLIRGenerationState) : string * MLIRGenerationState =
    match expr with
    | Literal literal ->
        convertLiteral literal state
    
    | Variable name ->
        match SSA.lookupVariable name state with
        | Some value -> (value, state)
        | None -> 
            let (dummyValue, state1) = SSA.generateValue "unknown" state
            (dummyValue, state1)
    
    | Application(func, args) ->
        match func with
        // Handle pipe operator: x |> f becomes f(x)
        | Variable "op_PipeRight" ->
            match args with
            | [value; funcExpr] ->
                // Convert pipe to regular application
                convertExpression (Application(funcExpr, [value])) state
            | _ ->
                // Invalid pipe operator usage
                let (dummyValue, state1) = SSA.generateValue "invalid_pipe" state
                (dummyValue, state1)
        // Handle curried Console.readLine: Console.readLine(buffer)(size)
        | Application(Variable "Console.readLine", innerArgs) ->
            printfn "DEBUG: Found curried Console.readLine"
            printfn "DEBUG: Inner args: %A" innerArgs
            printfn "DEBUG: Outer args: %A" args
            
            let allArgs = innerArgs @ args
            printfn "DEBUG: Combined args for Console.readLine: %A" allArgs
            
            convertConsoleReadLine allArgs state
        
        // Handle Span<byte> constructor properly
        | Variable "Span<byte>" ->
            printfn "DEBUG: Found Span<byte> constructor with args: %A" args
            convertSpanCreation args state
        
        // Handle IOOperation wrapped in Application: Application(IOOperation(...), args)
        | IOOperation(ioType, ioArgs) ->
            printfn "DEBUG: Found IOOperation wrapped in Application: %A with args: %A" ioType args
            
            // Process the IOOperation with its original args plus the application args
            let combinedArgs = ioArgs @ args
            convertExpression (IOOperation(ioType, combinedArgs)) state
        
        | Variable "Console.readLine" ->
            printfn "DEBUG: Direct Console.readLine call with args: %A" args
            convertConsoleReadLine args state
            
        | Variable "NativePtr.stackalloc" ->
            printfn "DEBUG: NativePtr.stackalloc with args: %A" args
            convertStackAllocation args state
            
        | Variable "Span.create" ->
            printfn "DEBUG: Span.create with args: %A" args
            convertSpanCreation args state
    
    | Let(name, value, body) ->
        printfn "DEBUG: Converting Let - name: %s" name
        printfn "DEBUG: Converting Let - value: %A" value
        printfn "DEBUG: Converting Let - body: %A" body
        
        // Special handling for the match expression pattern
        match value with
        | Literal UnitLiteral ->
            // This seems to be a placeholder from AST conversion
            // Just bind unit and continue
            let (unitValue, state1) = Emitter.constant "0" (Integer 32) state
            let state2 = SSA.bindVariable name unitValue state1
            convertExpression body state2
        | _ ->
            let (valueResult, state1) = convertExpression value state
            printfn "DEBUG: Let value converted, result: %s" valueResult
            
            let state2 = SSA.bindVariable name valueResult state1
            printfn "DEBUG: Variable '%s' bound to '%s'" name valueResult
            
            let (bodyResult, state3) = convertExpression body state2
            printfn "DEBUG: Let body converted, result: %s" bodyResult
            
            (bodyResult, state3)
    
    | Sequential(first, second) ->
        printfn "DEBUG: Converting Sequential - first: %A" first
        printfn "DEBUG: Converting Sequential - second: %A" second
        
        let (firstResult, state1) = convertExpression first state
        printfn "DEBUG: First expression converted, result: %s" firstResult
        
        let state1_debug = Emitter.emit (sprintf "    ; Sequential: completed first part, result: %s" firstResult) state1
        
        printfn "DEBUG: About to convert second expression..."
        let (secondResult, state2) = convertExpression second state1_debug
        printfn "DEBUG: Second expression converted, result: %s" secondResult
        
        let state3 = Emitter.emit (sprintf "    ; Sequential: completed second part, result: %s" secondResult) state2
        
        (secondResult, state3)
    
    | IfThenElse(cond, thenExpr, elseExpr) ->
        let (condResult, state1) = convertExpression cond state
        let (thenResult, state2) = convertExpression thenExpr state1
        let (elseResult, state3) = convertExpression elseExpr state2
        (thenResult, state3)
    
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
            printfn "DEBUG: Processing IO operation: %A with args: %A" ioType args
            
            // Check if format string has placeholders for strings
            let hasStringPlaceholder = formatStr.Contains("%s")
            
            let (formatGlobal, state1) = Emitter.registerString formatStr state
            let state2 = Emitter.emitModuleLevel "  func.func private @printf(memref<?xi8>, ...) -> i32" state1
            
            let (formatPtr, state3) = SSA.generateValue "fmt_ptr" state2
            let state4 = Emitter.emit (sprintf "    %s = memref.get_global %s : memref<?xi8>" formatPtr formatGlobal) state3
            
            match args with
            | [] when hasStringPlaceholder ->
                // If format has %s but no buffer is provided, use a dummy buffer
                let (dummyBuffer, state5) = SSA.generateValue "dummy_buffer" state4
                let state6 = Emitter.emit (sprintf "    %s = memref.alloca() : memref<1xi8>" dummyBuffer) state5
                let (printfResult, state7) = SSA.generateValue "printf_result" state6
                let state8 = Emitter.emit (sprintf "    %s = func.call @printf(%s, %s) : (memref<?xi8>, memref<?xi8>) -> i32" 
                                          printfResult formatPtr dummyBuffer) state7
                (printfResult, state8)
            | [] ->
                let (printfResult, state5) = SSA.generateValue "printf_result" state4
                let state6 = Emitter.emit (sprintf "    %s = func.call @printf(%s) : (memref<?xi8>) -> i32" printfResult formatPtr) state5
                (printfResult, state6)
            | [bufferArg] when hasStringPlaceholder ->
                // Special case for string format with buffer argument
                let (bufferValue, state5) = convertExpression bufferArg state4
                let (printfResult, state6) = SSA.generateValue "printf_result" state5
                let state7 = Emitter.emit (sprintf "    %s = func.call @printf(%s, %s) : (memref<?xi8>, memref<?xi8>) -> i32" 
                                          printfResult formatPtr bufferValue) state6
                (printfResult, state7)
            | argList ->
                // Multiple arguments
                let convertArg (accState, accArgs) arg =
                    let (argValue, newState) = convertExpression arg accState
                    (newState, argValue :: accArgs)
                
                let (state5, convertedArgs) = List.fold convertArg (state4, []) argList
                let allArgs = formatPtr :: (List.rev convertedArgs)
                let (printfResult, state6) = SSA.generateValue "printf_result" state5
                
                let argTypes = "memref<?xi8>" :: (List.replicate convertedArgs.Length "memref<?xi8>")
                let argTypeStr = String.concat ", " argTypes
                let argStr = String.concat ", " allArgs
                
                let state7 = Emitter.emit (sprintf "    %s = func.call @printf(%s) : (%s) -> i32" printfResult argStr argTypeStr) state6
                (printfResult, state7)
        
        | ReadLine ->
            let (dummyValue, state1) = SSA.generateValue "readline" state
            (dummyValue, state1)
        
        | Scanf formatStr ->
            let (dummyValue, state1) = SSA.generateValue "scanf" state
            (dummyValue, state1)
        
        | WriteFile path ->
            let (dummyValue, state1) = SSA.generateValue "writefile" state
            (dummyValue, state1)
        
        | ReadFile path ->
            let (dummyValue, state1) = SSA.generateValue "readfile" state
            (dummyValue, state1)
    
    | Lambda(params', body) ->
        let (bodyResult, state1) = convertExpression body state
        (bodyResult, state1)
    
    | _ ->
        printfn "DEBUG: Unhandled expression type: %A" expr
        let (dummyValue, state1) = SSA.generateValue "unhandled" state
        (dummyValue, state1)

/// Function and module level conversion functions
let convertFunction (name: string) (parameters: (string * OakType) list) (returnType: OakType) (body: OakExpression) (state: MLIRGenerationState) : MLIRGenerationState =
    printfn "DEBUG: Converting function %s" name
    printfn "DEBUG: Function body: %A" body
    
    let returnTypeStr = mlirTypeToString (mapOakTypeToMLIR returnType)
    
    let paramStr = 
        if parameters.IsEmpty then
            ""
        else
            parameters 
            |> List.mapi (fun i (_, typ) -> 
                sprintf "%%arg%d: %s" i (mlirTypeToString (mapOakTypeToMLIR typ)))
            |> String.concat ", "
            
    let state1 = Emitter.emit (sprintf "  func.func @%s(%s) -> %s {" name paramStr returnTypeStr) state
    
    let state2 = 
        parameters 
        |> List.mapi (fun i (paramName, _) -> 
            SSA.bindVariable paramName (sprintf "%%arg%d" i))
        |> List.fold (fun s f -> f s) state1
    
    printfn "DEBUG: About to convert function body for %s" name
    let (bodyResult, state3) = convertExpression body state2
    printfn "DEBUG: Function body conversion completed for %s, result: %s" name bodyResult
    
    let state4 = 
        if returnType = UnitType then
            printfn "DEBUG: Adding void return for %s" name
            Emitter.emit "    func.return" state3
        else
            printfn "DEBUG: Adding value return for %s: %s" name bodyResult
            Emitter.emit (sprintf "    func.return %s : %s" bodyResult returnTypeStr) state3
    
    let state5 = Emitter.emit "  }" state4
    
    printfn "DEBUG: Function %s conversion completed" name
    state5

/// MLIR module generation
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
        TypeMappings = Map.empty
        Diagnostics = finalState.ErrorContext
    }

/// Generates complete MLIR module text from Oak AST
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