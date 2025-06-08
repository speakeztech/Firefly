module Core.MLIRGeneration.XParsecMLIRGenerator

open System
open XParsec
open Core.XParsec.Foundation
open Core.XParsec.Foundation.Combinators
open Core.XParsec.Foundation.ErrorHandling
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
    ErrorContext: string list
}

/// MLIR output with complete module information
type MLIRModuleOutput = {
    ModuleName: string
    Operations: string list
    SSAMappings: Map<string, string>
    TypeMappings: Map<string, MLIRType>
    Diagnostics: string list
}

/// SSA value generation with XParsec state management
module SSAGeneration =
    
    /// Generates a unique SSA value name
    let generateSSAValue (prefix: string) : Parser<string, MLIRGenerationState> =
        fun state ->
            let newCounter = state.SSACounter + 1
            let ssaName = sprintf "%%%s%d" prefix newCounter
            let newState = { state with SSACounter = newCounter }
            Reply(Ok ssaName, newState)
    
    /// Records a variable binding in current scope
    let bindVariable (varName: string) (ssaValue: string) : Parser<unit, MLIRGenerationState> =
        fun state ->
            let newScope = Map.add varName ssaValue state.CurrentScope
            let newState = { state with CurrentScope = newScope }
            Reply(Ok (), newState)
    
    /// Looks up a variable in the scope stack
    let lookupVariable (varName: string) : Parser<string, MLIRGenerationState> =
        fun state ->
            let rec findInScopes scopes =
                match scopes with
                | [] -> None
                | scope :: rest ->
                    match Map.tryFind varName scope with
                    | Some value -> Some value
                    | None -> findInScopes rest
            
            match findInScopes (state.CurrentScope :: state.ScopeStack) with
            | Some ssaValue -> Reply(Ok ssaValue, state)
            | None -> 
                let error = sprintf "Undefined variable '%s' in current scope" varName
                Reply(Error, error)
    
    /// Pushes a new scope onto the stack
    let pushScope : Parser<unit, MLIRGenerationState> =
        fun state ->
            let newScopeStack = state.CurrentScope :: state.ScopeStack
            let newState = { 
                state with 
                    ScopeStack = newScopeStack
                    CurrentScope = Map.empty 
            }
            Reply(Ok (), newState)
    
    /// Pops a scope from the stack
    let popScope : Parser<unit, MLIRGenerationState> =
        fun state ->
            match state.ScopeStack with
            | prevScope :: rest ->
                let newState = { 
                    state with 
                        CurrentScope = prevScope
                        ScopeStack = rest 
                }
                Reply(Ok (), newState)
            | [] ->
                Reply(Error, "Cannot pop scope - scope stack is empty")

/// MLIR operation emission with XParsec combinators
module OperationEmission =
    
    /// Emits a single MLIR operation
    let emitOperation (operation: string) : Parser<unit, MLIRGenerationState> =
        fun state ->
            let newOps = operation :: state.GeneratedOperations
            let newState = { state with GeneratedOperations = newOps }
            Reply(Ok (), newState)
    
    /// Emits multiple MLIR operations
    let emitOperations (operations: string list) : Parser<unit, MLIRGenerationState> =
        operations
        |> List.map emitOperation
        |> List.fold (>>=) (succeed ())
        |> fun combined -> combined |>> ignore
    
    /// Emits an MLIR constant operation
    let emitConstant (value: string) (mlirType: MLIRType) : Parser<string, MLIRGenerationState> =
        generateSSAValue "const" >>= fun ssaValue ->
        let typeStr = mlirTypeToString mlirType
        let operation = sprintf "  %s = arith.constant %s : %s" ssaValue value typeStr
        emitOperation operation >>= fun _ ->
        succeed ssaValue
    
    /// Emits an MLIR function call operation
    let emitFunctionCall (funcName: string) (args: string list) (resultType: MLIRType) : Parser<string, MLIRGenerationState> =
        generateSSAValue "call" >>= fun ssaValue ->
        let argStr = String.concat ", " args
        let typeStr = mlirTypeToString resultType
        let operation = sprintf "  %s = func.call @%s(%s) : (%s) -> %s" 
                               ssaValue funcName argStr 
                               (String.concat ", " (List.replicate args.Length "i32")) 
                               typeStr
        emitOperation operation >>= fun _ ->
        succeed ssaValue
    
    /// Emits an MLIR arithmetic operation
    let emitArithmeticOp (opName: string) (lhs: string) (rhs: string) (resultType: MLIRType) : Parser<string, MLIRGenerationState> =
        generateSSAValue "arith" >>= fun ssaValue ->
        let typeStr = mlirTypeToString resultType
        let operation = sprintf "  %s = arith.%s %s, %s : %s" ssaValue opName lhs rhs typeStr
        emitOperation operation >>= fun _ ->
        succeed ssaValue

/// Oak literal to MLIR conversion using XParsec
module LiteralConversion =
    
    /// Converts Oak literal to MLIR constant
    let convertLiteral (literal: OakLiteral) : Parser<string, MLIRGenerationState> =
        match literal with
        | IntLiteral value ->
            emitConstant (string value) (Integer 32)
            |> withErrorContext "integer literal conversion"
        
        | FloatLiteral value ->
            emitConstant (sprintf "%f" value) (Float 32)
            |> withErrorContext "float literal conversion"
        
        | BoolLiteral value ->
            let boolValue = if value then "1" else "0"
            emitConstant boolValue (Integer 1)
            |> withErrorContext "boolean literal conversion"
        
        | StringLiteral value ->
            // String literals require more complex handling in MLIR
            let escapedValue = value.Replace("\"", "\\\"")
            generateSSAValue "str" >>= fun ssaValue ->
            let operation = sprintf "  %s = llvm.mlir.addressof @str_const_%d : !llvm.ptr<i8>" 
                                   ssaValue (abs (value.GetHashCode()))
            emitOperation operation >>= fun _ ->
            succeed ssaValue
            |> withErrorContext "string literal conversion"
        
        | UnitLiteral ->
            // Unit type has no runtime representation
            succeed ""
            |> withErrorContext "unit literal conversion"
        
        | ArrayLiteral elements ->
            // Array literals need element-by-element conversion
            compilerFail (TransformError(
                "literal conversion", 
                "ArrayLiteral", 
                "MLIR", 
                "Array literals not yet implemented in MLIR generation"))

/// Oak expression to MLIR conversion using XParsec
module ExpressionConversion =
    
    let rec convertExpression (expr: OakExpression) : Parser<string, MLIRGenerationState> =
        match expr with
        | Literal literal ->
            LiteralConversion.convertLiteral literal
            |> withErrorContext "literal expression"
        
        | Variable varName ->
            lookupVariable varName
            |> withErrorContext (sprintf "variable reference '%s'" varName)
        
        | Application(func, args) ->
            convertFunctionApplication func args
            |> withErrorContext "function application"
        
        | Lambda(params', body) ->
            convertLambdaExpression params' body
            |> withErrorContext "lambda expression"
        
        | Let(name, value, body) ->
            convertLetExpression name value body
            |> withErrorContext "let expression"
        
        | IfThenElse(cond, thenExpr, elseExpr) ->
            convertConditionalExpression cond thenExpr elseExpr
            |> withErrorContext "conditional expression"
        
        | Sequential(first, second) ->
            convertSequentialExpression first second
            |> withErrorContext "sequential expression"
        
        | FieldAccess(target, fieldName) ->
            convertFieldAccess target fieldName
            |> withErrorContext "field access"
        
        | MethodCall(target, methodName, args) ->
            convertMethodCall target methodName args
            |> withErrorContext "method call"
    
    and convertFunctionApplication (func: OakExpression) (args: OakExpression list) : Parser<string, MLIRGenerationState> =
        match func with
        | Variable funcName ->
            // Convert all arguments first
            args 
            |> List.map convertExpression
            |> List.fold (fun acc argParser ->
                acc >>= fun accResults ->
                argParser >>= fun argResult ->
                succeed (argResult :: accResults)
            ) (succeed [])
            >>= fun argSSAValues ->
            let reversedArgs = List.rev argSSAValues
            emitFunctionCall funcName reversedArgs (Integer 32)
        
        | _ ->
            compilerFail (TransformError(
                "function application", 
                "complex function expression", 
                "MLIR", 
                "Only simple function names supported for function calls"))
    
    and convertLambdaExpression (params': (string * OakType) list) (body: OakExpression) : Parser<string, MLIRGenerationState> =
        compilerFail (TransformError(
            "lambda conversion", 
            "lambda expression", 
            "MLIR", 
            "Lambda expressions should be eliminated by closure transformer"))
    
    and convertLetExpression (name: string) (value: OakExpression) (body: OakExpression) : Parser<string, MLIRGenerationState> =
        convertExpression value >>= fun valueSSA ->
        bindVariable name valueSSA >>= fun _ ->
        convertExpression body
    
    and convertConditionalExpression (cond: OakExpression) (thenExpr: OakExpression) (elseExpr: OakExpression) : Parser<string, MLIRGenerationState> =
        convertExpression cond >>= fun condSSA ->
        generateSSAValue "if_result" >>= fun resultSSA ->
        
        // Emit conditional branch structure
        let thenLabel = "then_block"
        let elseLabel = "else_block" 
        let mergeLabel = "merge_block"
        
        emitOperation (sprintf "  cf.cond_br %s, ^%s, ^%s" condSSA thenLabel elseLabel) >>= fun _ ->
        emitOperation (sprintf "^%s:" thenLabel) >>= fun _ ->
        convertExpression thenExpr >>= fun thenSSA ->
        emitOperation (sprintf "  cf.br ^%s(%s : i32)" mergeLabel thenSSA) >>= fun _ ->
        emitOperation (sprintf "^%s:" elseLabel) >>= fun _ ->
        convertExpression elseExpr >>= fun elseSSA ->
        emitOperation (sprintf "  cf.br ^%s(%s : i32)" mergeLabel elseSSA) >>= fun _ ->
        emitOperation (sprintf "^%s(%s: i32):" mergeLabel resultSSA) >>= fun _ ->
        succeed resultSSA
    
    and convertSequentialExpression (first: OakExpression) (second: OakExpression) : Parser<string, MLIRGenerationState> =
        convertExpression first >>= fun _ ->  // Ignore first result
        convertExpression second
    
    and convertFieldAccess (target: OakExpression) (fieldName: string) : Parser<string, MLIRGenerationState> =
        compilerFail (TransformError(
            "field access", 
            sprintf "field access .%s" fieldName, 
            "MLIR", 
            "Field access not implemented - requires struct type information"))
    
    and convertMethodCall (target: OakExpression) (methodName: string) (args: OakExpression list) : Parser<string, MLIRGenerationState> =
        compilerFail (TransformError(
            "method call", 
            sprintf "method call .%s" methodName, 
            "MLIR", 
            "Method calls not implemented - requires object model"))

/// Oak declaration to MLIR conversion using XParsec
module DeclarationConversion =
    
    /// Converts function declaration to MLIR function
    let convertFunctionDeclaration (name: string) (params': (string * OakType) list) (returnType: OakType) (body: OakExpression) : Parser<unit, MLIRGenerationState> =
        let paramTypes = params' |> List.map (snd >> mapOakTypeToMLIR)
        let returnMLIRType = mapOakTypeToMLIR returnType
        
        // Emit function signature
        let paramTypeStrs = paramTypes |> List.map mlirTypeToString
        let returnTypeStr = mlirTypeToString returnMLIRType
        let funcSig = sprintf "func.func @%s(%s) -> %s {" 
                             name 
                             (String.concat ", " paramTypeStrs)
                             returnTypeStr
        
        emitOperation funcSig >>= fun _ ->
        pushScope >>= fun _ ->
        
        // Bind parameters to entry block arguments
        params' 
        |> List.mapi (fun i (paramName, _) -> 
            let argSSA = sprintf "%%arg%d" i
            bindVariable paramName argSSA)
        |> List.fold (>>=) (succeed ())
        >>= fun _ ->
        
        // Convert function body
        convertExpression body >>= fun bodySSA ->
        
        // Emit return
        let returnOp = 
            if returnType = UnitType then
                "  func.return"
            else
                sprintf "  func.return %s : %s" bodySSA (mlirTypeToString returnMLIRType)
        
        emitOperation returnOp >>= fun _ ->
        emitOperation "}" >>= fun _ ->
        popScope
        |> withErrorContext (sprintf "function declaration '%s'" name)
    
    /// Converts entry point to MLIR main function
    let convertEntryPoint (expr: OakExpression) : Parser<unit, MLIRGenerationState> =
        emitOperation "func.func @main() -> i32 {" >>= fun _ ->
        pushScope >>= fun _ ->
        convertExpression expr >>= fun exprSSA ->
        
        // Entry point must return i32 (exit code)
        let returnSSA = 
            if String.IsNullOrEmpty(exprSSA) then
                // Unit expression - return 0
                emitConstant "0" (Integer 32)
            else
                succeed exprSSA
        
        returnSSA >>= fun finalSSA ->
        emitOperation (sprintf "  func.return %s : i32" finalSSA) >>= fun _ ->
        emitOperation "}" >>= fun _ ->
        popScope
        |> withErrorContext "entry point conversion"
    
    /// Converts type declaration (placeholder for now)
    let convertTypeDeclaration (name: string) (oakType: OakType) : Parser<unit, MLIRGenerationState> =
        let comment = sprintf "// Type declaration: %s = %s" name (oakType.ToString())
        emitOperation comment
        |> withErrorContext (sprintf "type declaration '%s'" name)
    
    /// Converts any declaration
    let convertDeclaration (decl: OakDeclaration) : Parser<unit, MLIRGenerationState> =
        match decl with
        | FunctionDecl(name, params', returnType, body) ->
            convertFunctionDeclaration name params' returnType body
        | EntryPoint(expr) ->
            convertEntryPoint expr
        | TypeDecl(name, oakType) ->
            convertTypeDeclaration name oakType

/// Main MLIR generation using XParsec - NO FALLBACKS
let generateMLIR (program: OakProgram) : CompilerResult<MLIRModuleOutput> =
    if program.Modules.IsEmpty then
        CompilerFailure [TransformError(
            "MLIR generation", 
            "empty program", 
            "MLIR module", 
            "Program must contain at least one module")]
    else
        let mainModule = program.Modules.[0]
        
        if mainModule.Declarations.IsEmpty then
            CompilerFailure [TransformError(
                "MLIR generation", 
                sprintf "empty module '%s'" mainModule.Name, 
                "MLIR module", 
                "Module must contain at least one declaration")]
        else
            let initialState = {
                SSACounter = 0
                CurrentScope = Map.empty
                ScopeStack = []
                GeneratedOperations = []
                CurrentFunction = None
                ErrorContext = []
            }
            
            // Generate module header
            let moduleHeader = sprintf "module @%s {" mainModule.Name
            
            // Convert all declarations
            let conversionParser = 
                emitOperation moduleHeader >>= fun _ ->
                mainModule.Declarations
                |> List.map convertDeclaration
                |> List.fold (>>=) (succeed ())
                >>= fun _ ->
                emitOperation "}"
            
            match conversionParser initialState with
            | Reply(Ok _, finalState) ->
                let operations = List.rev finalState.GeneratedOperations
                Success {
                    ModuleName = mainModule.Name
                    Operations = operations
                    SSAMappings = finalState.CurrentScope
                    TypeMappings = Map.empty  // Would be populated with type info
                    Diagnostics = []
                }
            
            | Reply(Error, errorMsg) ->
                CompilerFailure [TransformError(
                    "MLIR generation", 
                    "Oak AST", 
                    "MLIR", 
                    errorMsg)]

/// Generates complete MLIR module text from Oak AST
let generateMLIRModuleText (program: OakProgram) : CompilerResult<string> =
    generateMLIR program >>= fun mlirOutput ->
    let moduleText = String.concat "\n" mlirOutput.Operations
    Success moduleText