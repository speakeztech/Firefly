module Core.MLIRGeneration.XParsecMLIRGenerator

open System
open System.Text
open XParsec
open Core.XParsec.Foundation
open Core.XParsec.Foundation.Combinators
open Core.XParsec.Foundation.ErrorHandling
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
let createInitialState() = {
    SSACounter = 0
    CurrentScope = Map.empty
    ScopeStack = []
    GeneratedOperations = []
    CurrentFunction = None
    StringConstants = Map.empty
    CurrentDialect = Func
    ErrorContext = []
}

/// Core SSA value and scope management module
module SSA = 
    /// Generates a new SSA value with optional prefix
    let generateValue (prefix: string) : Parser<string, MLIRGenerationState> =
        fun state ->
            let newCounter = state.SSACounter + 1
            let ssaName = sprintf "%%%s%d" prefix newCounter
            let newState = { state with SSACounter = newCounter }
            Reply(Ok ssaName, newState)
    
    /// Binds a variable name to an SSA value in current scope
    let bindVariable (name: string) (value: string) : Parser<unit, MLIRGenerationState> =
        fun state ->
            let newScope = Map.add name value state.CurrentScope
            let newState = { state with CurrentScope = newScope }
            Reply(Ok (), newState)
    
    /// Looks up a variable by name in the current scope stack
    let lookupVariable (name: string) : Parser<string, MLIRGenerationState> =
        fun state ->
            let rec lookup scopes =
                match scopes with
                | [] -> None
                | scope :: rest ->
                    match Map.tryFind name scope with
                    | Some value -> Some value
                    | None -> lookup rest
            
            match lookup (state.CurrentScope :: state.ScopeStack) with
            | Some value -> Reply(Ok value, state)
            | None -> Reply(Error, sprintf "Variable '%s' not found in scope" name)
    
    /// Pushes current scope onto stack and creates a new empty scope
    let pushScope : Parser<unit, MLIRGenerationState> =
        fun state ->
            let newState = { 
                state with 
                    ScopeStack = state.CurrentScope :: state.ScopeStack
                    CurrentScope = Map.empty 
            }
            Reply(Ok (), newState)
    
    /// Pops scope from stack and restores previous scope
    let popScope : Parser<unit, MLIRGenerationState> =
        fun state ->
            match state.ScopeStack with
            | scope :: rest ->
                let newState = { 
                    state with 
                        CurrentScope = scope
                        ScopeStack = rest 
                }
                Reply(Ok (), newState)
            | [] -> 
                Reply(Error, "Cannot pop scope - scope stack is empty")

/// Core MLIR operation emission module
module Operations =
    /// Emits a raw MLIR operation string
    let emit (operation: string) : Parser<unit, MLIRGenerationState> =
        fun state ->
            let newState = { 
                state with 
                    GeneratedOperations = operation :: state.GeneratedOperations 
            }
            Reply(Ok (), newState)
    
    /// Declares an external function in MLIR
    let declareExternal (name: string) (signature: string) : Parser<unit, MLIRGenerationState> =
        emit (sprintf "func.func private @%s %s" name signature)
    
    /// Registers a string constant and returns its global name
    let registerString (value: string) : Parser<string, MLIRGenerationState> =
        fun state ->
            match Map.tryFind value state.StringConstants with
            | Some existingName -> 
                Reply(Ok existingName, state)
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
                Reply(Ok constName, newState)
    
    /// Creates a constant value in MLIR
    let constant (value: string) (mlirType: MLIRType) : Parser<string, MLIRGenerationState> =
        SSA.generateValue "const" >>= fun result ->
        let typeStr = mlirTypeToString mlirType
        emit (sprintf "  %s = arith.constant %s : %s" result value typeStr) >>= fun _ ->
        succeed result
    
    /// Creates an arithmetic operation in MLIR
    let arithmetic (op: string) (lhs: string) (rhs: string) (resultType: MLIRType) : Parser<string, MLIRGenerationState> =
        SSA.generateValue "arith" >>= fun result ->
        let typeStr = mlirTypeToString resultType
        emit (sprintf "  %s = arith.%s %s, %s : %s" result op lhs rhs typeStr) >>= fun _ ->
        succeed result
    
    /// Creates a function call in MLIR
    let call (funcName: string) (args: string list) (resultType: MLIRType option) : Parser<string, MLIRGenerationState> =
        match resultType with
        | Some returnType ->
            SSA.generateValue "call" >>= fun result ->
            let argStr = String.concat ", " args
            let typeStr = mlirTypeToString returnType
            
            // Create parameter type list from argument types (simplified)
            let paramTypeStr = String.concat ", " (List.replicate args.Length "i32")
            
            emit (sprintf "  %s = func.call @%s(%s) : (%s) -> %s" 
                         result funcName argStr paramTypeStr typeStr) >>= fun _ ->
            succeed result
        | None ->
            // Function with no return value
            let argStr = String.concat ", " args
            let paramTypeStr = String.concat ", " (List.replicate args.Length "i32")
            
            emit (sprintf "  func.call @%s(%s) : (%s) -> ()" 
                         funcName argStr paramTypeStr) >>= fun _ ->
            SSA.generateValue "void" // Return a dummy value for consistency

/// Core expression conversion using simplified XParsec patterns
module ExpressionConversion =
    /// Converts Oak literal to MLIR constant
    let rec convertLiteral (literal: OakLiteral) : Parser<string, MLIRGenerationState> =
        match literal with
        | IntLiteral value ->
            Operations.constant (string value) (Integer 32)
        | FloatLiteral value ->
            Operations.constant (sprintf "%f" value) (Float 32)
        | BoolLiteral value ->
            Operations.constant (if value then "1" else "0") (Integer 1)
        | StringLiteral value ->
            // Register string as global and get pointer
            Operations.registerString value >>= fun globalName ->
            SSA.generateValue "str_ptr" >>= fun ptrResult ->
            Operations.emit (sprintf "  %s = memref.get_global %s : memref<?xi8>" ptrResult globalName) >>= fun _ ->
            succeed ptrResult
        | UnitLiteral ->
            // Unit type has no value in MLIR - return a dummy constant
            Operations.constant "0" (Integer 32)
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
            SSA.generateValue "array" >>= fun arrayResult ->
            Operations.emit (sprintf "  %s = memref.alloca(%d) : memref<%dx%s>" 
                                   arrayResult arraySize arraySize (mlirTypeToString elementType)) >>= fun _ ->
            
            // Initialize elements
            elements 
            |> List.mapi (fun idx element ->
                convertExpression element >>= fun elemValue ->
                SSA.generateValue "idx" >>= fun idxResult ->
                Operations.constant (string idx) (Integer 32) >>= fun idxConst ->
                Operations.emit (sprintf "  memref.store %s, %s[%s] : %s, memref<%dx%s>" 
                                        elemValue arrayResult idxConst 
                                        (mlirTypeToString elementType) 
                                        arraySize 
                                        (mlirTypeToString elementType)) >>= fun _ ->
                succeed ())
            |> List.fold (>>=) (succeed ())
            >>= fun _ ->
            
            succeed arrayResult
    
    /// Converts Oak expression to MLIR operations
    and convertExpression (expr: OakExpression) : Parser<string, MLIRGenerationState> =
        match expr with
        | Literal literal ->
            convertLiteral literal
            |> withErrorContext "literal conversion"
        
        | Variable name ->
            SSA.lookupVariable name
            |> withErrorContext (sprintf "variable '%s'" name)
        
        | Binary(op, lhs, rhs) ->
            // Convert operands
            convertExpression lhs >>= fun lhsResult ->
            convertExpression rhs >>= fun rhsResult ->
            
            // Apply operation
            match op with
            | Add -> Operations.arithmetic "addi" lhsResult rhsResult (Integer 32)
            | Subtract -> Operations.arithmetic "subi" lhsResult rhsResult (Integer 32)
            | Multiply -> Operations.arithmetic "muli" lhsResult rhsResult (Integer 32)
            | Divide -> Operations.arithmetic "divsi" lhsResult rhsResult (Integer 32)
            | Equal -> 
                // Comparison with predicate
                SSA.generateValue "cmp" >>= fun cmpResult ->
                Operations.emit (sprintf "  %s = arith.cmpi eq, %s, %s : i32" 
                                       cmpResult lhsResult rhsResult) >>= fun _ ->
                succeed cmpResult
            | _ ->
                // Other comparisons/operations would be handled similarly
                Operations.arithmetic "addi" lhsResult rhsResult (Integer 32)
            |> withErrorContext "binary operation"
        
        | Application(func, args) ->
            match func with
            | Variable funcName ->
                // Convert arguments
                args 
                |> List.map convertExpression
                |> List.fold (fun accM argM ->
                    accM >>= fun accArgs ->
                    argM >>= fun argResult ->
                    succeed (argResult :: accArgs)
                ) (succeed [])
                >>= fun argResults ->
                
                // Call function
                Operations.call funcName (List.rev argResults) (Some (Integer 32))
                |> withErrorContext (sprintf "function call '%s'" funcName)
            | _ ->
                fail "Only direct function calls supported"
                |> withErrorContext "complex function application"
        
        | Let(name, value, body) ->
            // Evaluate the bound value
            convertExpression value >>= fun valueResult ->
            
            // Bind in current scope and evaluate body
            SSA.bindVariable name valueResult >>= fun _ ->
            convertExpression body
            |> withErrorContext (sprintf "let binding '%s'" name)
        
        | IfThenElse(cond, thenExpr, elseExpr) ->
            // Generate unique labels for branches
            let thenLabel = sprintf "then_%d" (Guid.NewGuid().GetHashCode() |> abs)
            let elseLabel = sprintf "else_%d" (Guid.NewGuid().GetHashCode() |> abs)
            let endLabel = sprintf "endif_%d" (Guid.NewGuid().GetHashCode() |> abs)
            
            // Evaluate condition
            convertExpression cond >>= fun condResult ->
            
            // Create a result variable
            SSA.generateValue "if_result" >>= fun resultVar ->
            
            // Emit conditional branch
            Operations.emit (sprintf "  cf.cond_br %s, ^%s, ^%s" condResult thenLabel elseLabel) >>= fun _ ->
            
            // Then branch
            Operations.emit (sprintf "^%s:" thenLabel) >>= fun _ ->
            convertExpression thenExpr >>= fun thenResult ->
            Operations.emit (sprintf "  cf.br ^%s(%s : i32)" endLabel thenResult) >>= fun _ ->
            
            // Else branch
            Operations.emit (sprintf "^%s:" elseLabel) >>= fun _ ->
            convertExpression elseExpr >>= fun elseResult ->
            Operations.emit (sprintf "  cf.br ^%s(%s : i32)" endLabel elseResult) >>= fun _ ->
            
            // Join point
            Operations.emit (sprintf "^%s(%s: i32):" endLabel resultVar) >>= fun _ ->
            
            succeed resultVar
            |> withErrorContext "if-then-else expression"
        
        | Sequential(first, second) ->
            // Evaluate first expression and discard result
            convertExpression first >>= fun _ ->
            
            // Evaluate and return second expression
            convertExpression second
            |> withErrorContext "sequential expression"
        
        | Lambda(parameters', body) ->
            fail "Lambda expressions should be eliminated by closure transformation"
            |> withErrorContext "lambda expression"
        
        | FieldAccess(target, fieldName) ->
            convertExpression target >>= fun targetResult ->
            
            // Extract field using GEP
            SSA.generateValue "field" >>= fun fieldResult ->
            
            // This is a simplification - real implementation would need type information
            Operations.emit (sprintf "  %s = llvm.extractvalue %s[0] : !llvm.struct<i32, i32>" 
                                   fieldResult targetResult) >>= fun _ ->
            
            succeed fieldResult
            |> withErrorContext (sprintf "field access '%s'" fieldName)
        
        | IndexedAccess(target, indices) ->
            // Convert target to get array pointer
            convertExpression target >>= fun targetResult ->
            
            // Convert indices
            indices 
            |> List.map convertExpression
            |> List.fold (fun accM idxM ->
                accM >>= fun accIndices ->
                idxM >>= fun idxResult ->
                succeed (idxResult :: accIndices)
            ) (succeed [])
            >>= fun indexResults ->
            
            // Get element (simplified - would need proper type info)
            SSA.generateValue "element" >>= fun elementResult ->
            
            // Use first index only for now - multidimensional would be more complex
            let indexResult = List.rev indexResults |> List.tryHead |> Option.defaultValue "0"
            
            Operations.emit (sprintf "  %s = memref.load %s[%s] : memref<?xi32>" 
                                   elementResult targetResult indexResult) >>= fun _ ->
            
            succeed elementResult
            |> withErrorContext "indexed access"
        
        | IOOperation(ioType, args) ->
            match ioType with
            | Printf formatStr | Printfn formatStr ->
                // Register format string
                Operations.registerString formatStr >>= fun formatGlobal ->
                
                // Get format string pointer
                SSA.generateValue "fmt_ptr" >>= fun formatPtr ->
                Operations.emit (sprintf "  %s = memref.get_global %s : memref<?xi8>" formatPtr formatGlobal) >>= fun _ ->
                
                // Convert arguments
                args 
                |> List.map convertExpression
                |> List.fold (fun accM argM ->
                    accM >>= fun accArgs ->
                    argM >>= fun argResult ->
                    succeed (argResult :: accArgs)
                ) (succeed [])
                >>= fun argResults ->
                
                // Declare printf
                Operations.declareExternal "printf" "(memref<?xi8>, ...) -> i32" >>= fun _ ->
                
                // Call printf with format and args
                let allArgs = formatPtr :: List.rev argResults
                SSA.generateValue "printf_result" >>= fun printfResult ->
                
                // Build parameter type string based on arg count
                let paramTypes = "memref<?xi8>" :: List.replicate argResults.Length "i32"
                let paramTypeStr = String.concat ", " paramTypes
                
                Operations.emit (sprintf "  %s = func.call @printf(%s) : (%s) -> i32" 
                                       printfResult (String.concat ", " allArgs) paramTypeStr) >>= fun _ ->
                
                // For printfn, also print newline
                (if ioType = Printfn formatStr then
                    Operations.registerString "\n" >>= fun nlGlobal ->
                    SSA.generateValue "nl_ptr" >>= fun nlPtr ->
                    Operations.emit (sprintf "  %s = memref.get_global %s : memref<?xi8>" nlPtr nlGlobal) >>= fun _ ->
                    
                    SSA.generateValue "nl_result" >>= fun nlResult ->
                    Operations.emit (sprintf "  %s = func.call @printf(%s) : (memref<?xi8>) -> i32" nlResult nlPtr) >>= fun _ ->
                    succeed printfResult
                 else
                    succeed printfResult)
                |> withErrorContext "printf operation"
            
            | ReadLine ->
                // Declare fgets and stdin
                Operations.declareExternal "fgets" "(memref<?xi8>, i32, memref<?xi8>) -> memref<?xi8>" >>= fun _ ->
                Operations.declareExternal "__stdinp" "() -> memref<?xi8>" >>= fun _ ->
                
                // Allocate buffer
                SSA.generateValue "buffer" >>= fun bufferResult ->
                Operations.emit (sprintf "  %s = memref.alloca() : memref<256xi8>" bufferResult) >>= fun _ ->
                
                // Get buffer size constant
                SSA.generateValue "buf_size" >>= fun sizeResult ->
                Operations.constant "256" (Integer 32) >>= fun _ ->
                
                // Get stdin handle
                SSA.generateValue "stdin" >>= fun stdinResult ->
                Operations.emit (sprintf "  %s = func.call @__stdinp() : () -> memref<?xi8>" stdinResult) >>= fun _ ->
                
                // Call fgets
                SSA.generateValue "fgets_result" >>= fun fgetsResult ->
                Operations.emit (sprintf "  %s = func.call @fgets(%s, %s, %s) : (memref<256xi8>, i32, memref<?xi8>) -> memref<?xi8>" 
                                       fgetsResult bufferResult sizeResult stdinResult) >>= fun _ ->
                
                succeed bufferResult
                |> withErrorContext "readline operation"
            
            | _ ->
                fail "Unsupported I/O operation"
                |> withErrorContext "I/O operation"
        
        | _ ->
            fail "Unsupported expression type"
            |> withErrorContext "unknown expression"

/// Function and module level conversion functions
module DeclarationConversion =
    /// Converts function declaration to MLIR function
    let convertFunction (name: string) (parameters': (string * OakType) list) (returnType: OakType) (body: OakExpression) : Parser<unit, MLIRGenerationState> =
        // Generate function signature
        let paramTypes = parameters' |> List.map (snd >> mlirTypeToString)
        let returnTypeStr = mlirTypeToString (mapOakTypeToMLIR returnType)
        
        // Start function definition
        let paramStr = 
            parameters' 
            |> List.mapi (fun i (name, typ) -> 
                sprintf "%%arg%d: %s" i (mlirTypeToString (mapOakTypeToMLIR typ)))
            |> String.concat ", "
            
        Operations.emit (sprintf "func.func @%s(%s) -> %s {" name paramStr returnTypeStr) >>= fun _ ->
        
        // Create new scope for function body
        SSA.pushScope >>= fun _ ->
        
        // Bind parameters to arguments
        parameters' 
        |> List.mapi (fun i (paramName, _) -> 
            SSA.bindVariable paramName (sprintf "%%arg%d" i))
        |> List.fold (>>=) (succeed ())
        >>= fun _ ->
        
        // Convert function body
        ExpressionConversion.convertExpression body >>= fun bodyResult ->
        
        // Generate return statement
        (if returnType = UnitType then
            Operations.emit "  func.return"
         else
            Operations.emit (sprintf "  func.return %s : %s" bodyResult returnTypeStr))
        >>= fun _ ->
        
        // End function
        Operations.emit "}" >>= fun _ ->
        
        // Restore previous scope
        SSA.popScope
    
    /// Converts Oak entry point to MLIR main function
    let convertEntryPoint (expr: OakExpression) : Parser<unit, MLIRGenerationState> =
        // Generate main function signature
        Operations.emit "func.func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {" >>= fun _ ->
        
        // Create new scope
        SSA.pushScope >>= fun _ ->
        
        // Bind argc/argv
        SSA.bindVariable "argc" "%arg0" >>= fun _ ->
        SSA.bindVariable "argv" "%arg1" >>= fun _ ->
        
        // Convert body expression
        ExpressionConversion.convertExpression expr >>= fun bodyResult ->
        
        // Return 0 if expr doesn't return a value
        (match expr with
         | Literal(IntLiteral _) -> 
             Operations.emit (sprintf "  func.return %s : i32" bodyResult)
         | _ ->
             SSA.generateValue "exit_code" >>= fun exitCode ->
             Operations.constant "0" (Integer 32) >>= fun _ ->
             Operations.emit (sprintf "  func.return %s : i32" exitCode))
        >>= fun _ ->
        
        // End function
        Operations.emit "}" >>= fun _ ->
        
        // Restore previous scope
        SSA.popScope
    
    /// Converts external declaration to MLIR function declaration
    let convertExternalDecl (name: string) (paramTypes: OakType list) (returnType: OakType) (libraryName: string) : Parser<unit, MLIRGenerationState> =
        // Map Oak types to MLIR types
        let mlirParamTypes = paramTypes |> List.map (mapOakTypeToMLIR >> mlirTypeToString)
        let mlirReturnType = mlirTypeToString (mapOakTypeToMLIR returnType)
        
        // Generate parameter list
        let paramStr = 
            mlirParamTypes 
            |> List.mapi (fun i typ -> sprintf "%%arg%d: %s" i typ)
            |> String.concat ", "
        
        // Emit external declaration
        Operations.emit (sprintf "func.func private @%s(%s) -> %s attributes {ffi.library = \"%s\"}" 
                              name paramStr mlirReturnType libraryName)

/// Main MLIR generation function using XParsec combinators - simplified
let generateMLIR (program: OakProgram) : CompilerResult<MLIRModuleOutput> =
    if program.Modules.IsEmpty then
        CompilerFailure [TransformError("MLIR generation", "empty program", "MLIR module", "Program has no modules")]
    else
        // Get the main module
        let mainModule = program.Modules.[0]
        
        // Create initial state
        let initialState = createInitialState()
        
        // Build the conversion pipeline
        let pipelineResult =
            // Generate module header
            Operations.emit (sprintf "module @%s {" mainModule.Name) >>= fun _ ->
            
            // Convert all declarations
            mainModule.Declarations
            |> List.fold (fun stateM decl ->
                stateM >>= fun _ ->
                match decl with
                | FunctionDecl(name, parameters', returnType, body) ->
                    DeclarationConversion.convertFunction name parameters' returnType body
                | TypeDecl(name, oakType) ->
                    // Simple comment for type declarations
                    Operations.emit (sprintf "// Type declaration: %s" name)
                | EntryPoint(expr) ->
                    DeclarationConversion.convertEntryPoint expr
                | ExternalDecl(name, paramTypes, returnType, libraryName) ->
                    DeclarationConversion.convertExternalDecl name paramTypes returnType libraryName
            ) (succeed ())
            >>= fun _ ->
            
            // Close module
            Operations.emit "}"
        
        // Execute pipeline
        match pipelineResult initialState with
        | Reply(Ok (), finalState) ->
            // Success - build module output
            Success {
                ModuleName = mainModule.Name
                Operations = List.rev finalState.GeneratedOperations
                SSAMappings = finalState.CurrentScope
                TypeMappings = Map.empty
                Diagnostics = []
            }
        | Reply(Error, errorMsg) ->
            // Failure - report error
            CompilerFailure [TransformError("MLIR generation", "Oak AST", "MLIR", errorMsg)]

/// Generates complete MLIR module text from Oak AST with proper error handling
let generateMLIRModuleText (program: OakProgram) : CompilerResult<string> =
    generateMLIR program >>= fun mlirOutput ->
    
    // Join operations with newlines
    let moduleText = String.concat "\n" mlirOutput.Operations
    
    Success moduleText

/// Core MLIR operations builder API - stateless and simple to use
module MLIRBuilder =
    /// Creates a function declaration
    let createFunction name parameters returnType =
        sprintf "func.func @%s(%s) -> %s" name parameters returnType
    
    /// Creates a function call
    let createCall resultId funcName args resultType =
        sprintf "%s = func.call @%s(%s) : %s" resultId funcName args resultType
    
    /// Creates an arithmetic operation
    let createArithOp resultId op lhs rhs resultType =
        sprintf "%s = arith.%s %s, %s : %s" resultId op lhs rhs resultType
    
    /// Creates a memory allocation
    let createAlloca resultId size elementType =
        sprintf "%s = memref.alloca(%s) : memref<%s>" resultId size elementType
    
    /// Creates a memory load
    let createLoad resultId memref indices resultType =
        sprintf "%s = memref.load %s[%s] : %s" resultId memref indices resultType
    
    /// Creates a memory store
    let createStore value memref indices valueType =
        sprintf "memref.store %s, %s[%s] : %s" value memref indices valueType

/// Direct MLIR generation from Oak AST for simple cases - no parser combinators
let generateSimpleMLIR (program: OakProgram) : CompilerResult<string> =
    try
        if program.Modules.IsEmpty then
            CompilerFailure [TransformError("MLIR generation", "empty program", "MLIR module", "Program has no modules")]
        else
            let mainModule = program.Modules.[0]
            let sb = StringBuilder()
            
            // Add module header
            sb.AppendLine(sprintf "module @%s {" mainModule.Name) |> ignore
            
            // Process declarations
            mainModule.Declarations
            |> List.iter (function
                | FunctionDecl(name, parameters', returnType, _) ->
                    // Simple placeholder for function declaration
                    let paramStr = 
                        parameters' 
                        |> List.mapi (fun i (paramName, _) -> sprintf "%%arg%d: i32" i)
                        |> String.concat ", "
                    
                    sb.AppendLine(sprintf "  func.func @%s(%s) -> i32 {" name paramStr) |> ignore
                    sb.AppendLine("    // Function body would be generated here") |> ignore
                    sb.AppendLine("    func.return %c0_i32 : i32") |> ignore
                    sb.AppendLine("  }") |> ignore
                    
                | EntryPoint(_) ->
                    // Main function
                    sb.AppendLine("  func.func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {") |> ignore
                    sb.AppendLine("    // Entry point body would be generated here") |> ignore
                    sb.AppendLine("    func.return %c0_i32 : i32") |> ignore
                    sb.AppendLine("  }") |> ignore
                    
                | _ -> ())
                
            // Close module
            sb.AppendLine("}") |> ignore
            
            Success (sb.ToString())
    with ex ->
        CompilerFailure [TransformError("MLIR generation", "Oak AST", "MLIR", ex.Message)]