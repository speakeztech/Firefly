module Core.MLIRGeneration.XParsecMLIRGenerator

open System
open Firefly.Core.XParsec.Foundation
open Dabbit.Parsing.OakAst
open Core.MLIRGeneration.TypeSystem
open Core.MLIRGeneration.Operations

// ======================================
// MLIR Generation State
// ======================================

/// State for MLIR generation with SSA tracking
type MLIRState = {
    SSACounter: int
    CurrentScope: Map<string, string>
    ScopeStack: Map<string, string> list
    Operations: string list
    ExternalDecls: Set<string>
    StringConstants: Map<string, int>
}

/// Parser type specialized for MLIR generation
type MLIRParser<'T> = Parser<'T>

/// Creates initial MLIR generation state
let createMLIRState() : MLIRState =
    {
        SSACounter = 0
        CurrentScope = Map.empty
        ScopeStack = []
        Operations = []
        ExternalDecls = Set.empty
        StringConstants = Map.empty
    }

// ======================================
// State Management Combinators
// ======================================

/// Gets the current MLIR state from parser metadata
let getMLIRState : MLIRParser<MLIRState> =
    getMetadata "mlir_state" 
    |>> function 
        | Some state -> state :?> MLIRState
        | None -> createMLIRState()

/// Sets the MLIR state in parser metadata
let setMLIRState (state: MLIRState) : MLIRParser<unit> =
    addMetadata "mlir_state" state

/// Updates the MLIR state using a function
let updateMLIRState (f: MLIRState -> MLIRState) : MLIRParser<unit> =
    getMLIRState >>= fun state ->
    setMLIRState (f state)

// ======================================
// SSA Value Generation
// ======================================

module SSA =
    
    /// Generates a unique SSA value name
    let generateValue (prefix: string) : MLIRParser<string> =
        getMLIRState >>= fun state ->
        let newCounter = state.SSACounter + 1
        let ssaName = sprintf "%%%s%d" prefix newCounter
        let newState = { state with SSACounter = newCounter }
        setMLIRState newState >>= fun _ ->
        succeed ssaName
    
    /// Binds a variable to an SSA value in current scope
    let bindVariable (varName: string) (ssaValue: string) : MLIRParser<unit> =
        updateMLIRState (fun state ->
            { state with CurrentScope = Map.add varName ssaValue state.CurrentScope })
    
    /// Looks up a variable in the scope stack
    let lookupVariable (varName: string) : MLIRParser<string> =
        getMLIRState >>= fun state ->
        let rec findInScopes scopes =
            match scopes with
            | [] -> None
            | scope :: rest ->
                match Map.tryFind varName scope with
                | Some value -> Some value
                | None -> findInScopes rest
        
        match findInScopes (state.CurrentScope :: state.ScopeStack) with
        | Some ssaValue -> succeed ssaValue
        | None -> fail (sprintf "Undefined variable: %s" varName)
    
    /// Pushes a new scope onto the stack
    let pushScope : MLIRParser<unit> =
        updateMLIRState (fun state ->
            { state with 
                ScopeStack = state.CurrentScope :: state.ScopeStack
                CurrentScope = Map.empty })
    
    /// Pops a scope from the stack
    let popScope : MLIRParser<unit> =
        getMLIRState >>= fun state ->
        match state.ScopeStack with
        | prevScope :: rest ->
            setMLIRState { state with 
                            CurrentScope = prevScope
                            ScopeStack = rest }
        | [] -> fail "Cannot pop scope: stack is empty"

// ======================================
// Operation Emission
// ======================================

module Operations =
    
    /// Emits a single MLIR operation
    let emit (operation: string) : MLIRParser<unit> =
        updateMLIRState (fun state ->
            { state with Operations = operation :: state.Operations })
    
    /// Emits multiple operations
    let emitMany (operations: string list) : MLIRParser<unit> =
        operations 
        |> List.fold (fun acc op -> acc >>= fun _ -> emit op) (succeed ())
    
    /// Records an external function declaration
    let declareExternal (funcName: string) (signature: string) : MLIRParser<unit> =
        getMLIRState >>= fun state ->
        if Set.contains funcName state.ExternalDecls then
            succeed ()  // Already declared
        else
            let declaration = sprintf "func.func private @%s%s" funcName signature
            let newState = { state with ExternalDecls = Set.add funcName state.ExternalDecls }
            setMLIRState newState >>= fun _ ->
            emit declaration
    
    /// Registers a string constant and returns its global name
    let registerString (value: string) : MLIRParser<string> =
        getMLIRState >>= fun state ->
        let existing = state.StringConstants |> Map.tryFindKey (fun k _ -> k = value)
        
        match existing with
        | Some _ ->
            let constIndex = state.StringConstants.[value]
            let globalName = sprintf "@str_const_%d" constIndex
            succeed globalName
        | None ->
            let newIndex = state.StringConstants.Count
            let globalName = sprintf "@str_const_%d" newIndex
            let newConstants = Map.add value newIndex state.StringConstants
            
            let escapedValue = value.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n")
            let length = value.Length + 1
            let globalDecl = sprintf "memref.global constant @str_const_%d : memref<%dxi8> = dense<\"%s\\00\">" 
                                    newIndex length escapedValue
            
            updateMLIRState (fun s -> { s with StringConstants = newConstants }) >>= fun _ ->
            emit globalDecl >>= fun _ ->
            succeed globalName

// ======================================
// Literal Conversion
// ======================================

module Literals =
    
    /// Converts Oak literal to MLIR constant
    let convert (literal: OakLiteral) : MLIRParser<string> =
        match literal with
        | IntLiteral value ->
            SSA.generateValue "const" >>= fun ssaValue ->
            let operation = sprintf "  %s = arith.constant %d : i32" ssaValue value
            Operations.emit operation >>= fun _ ->
            succeed ssaValue
        
        | FloatLiteral value ->
            SSA.generateValue "const" >>= fun ssaValue ->
            let operation = sprintf "  %s = arith.constant %f : f32" ssaValue value
            Operations.emit operation >>= fun _ ->
            succeed ssaValue
        
        | BoolLiteral value ->
            SSA.generateValue "const" >>= fun ssaValue ->
            let boolValue = if value then "1" else "0"
            let operation = sprintf "  %s = arith.constant %s : i1" ssaValue boolValue
            Operations.emit operation >>= fun _ ->
            succeed ssaValue
        
        | StringLiteral value ->
            Operations.registerString value >>= fun globalName ->
            SSA.generateValue "str_addr" >>= fun addrSSA ->
            let addressOp = sprintf "  %s = memref.get_global %s : memref<%dxi8>" 
                                   addrSSA globalName (value.Length + 1)
            Operations.emit addressOp >>= fun _ ->
            succeed addrSSA
        
        | UnitLiteral ->
            SSA.generateValue "const" >>= fun ssaValue ->
            let operation = sprintf "  %s = arith.constant 0 : i32" ssaValue
            Operations.emit operation >>= fun _ ->
            succeed ssaValue
        
        | ArrayLiteral _ ->
            fail "Array literals not implemented"

// ======================================
// I/O Operations
// ======================================

module IOOperations =
    
    /// Converts printf operation
    let convertPrintf (formatString: string) (args: OakExpression list) 
                     (convertExpr: OakExpression -> MLIRParser<string>) : MLIRParser<string> =
        
        Operations.declareExternal "printf" "(memref<?xi8>, ...) -> i32" >>= fun _ ->
        Operations.registerString formatString >>= fun formatGlobal ->
        SSA.generateValue "fmt_ptr" >>= fun formatPtr ->
        
        let getFormatOp = sprintf "  %s = memref.get_global %s : memref<%dxi8>" 
                                 formatPtr formatGlobal (formatString.Length + 1)
        Operations.emit getFormatOp >>= fun _ ->
        
        if args.IsEmpty then
            SSA.generateValue "printf_result" >>= fun resultSSA ->
            let callOp = sprintf "  %s = func.call @printf(%s) : (memref<?xi8>) -> i32" 
                                resultSSA formatPtr
            Operations.emit callOp >>= fun _ ->
            succeed resultSSA
        else
            // Convert arguments sequentially
            let rec convertArgs remainingArgs accArgs =
                match remainingArgs with
                | [] -> succeed (List.rev accArgs)
                | arg :: rest ->
                    convertExpr arg >>= fun argResult ->
                    convertArgs rest (argResult :: accArgs)
            
            convertArgs args [] >>= fun argSSAValues ->
            SSA.generateValue "printf_result" >>= fun resultSSA ->
            let allArgs = formatPtr :: argSSAValues
            let argTypes = "memref<?xi8>" :: (List.replicate argSSAValues.Length "i32")
            let callOp = sprintf "  %s = func.call @printf(%s) : (%s) -> i32" 
                                resultSSA 
                                (String.concat ", " allArgs)
                                (String.concat ", " argTypes)
            Operations.emit callOp >>= fun _ ->
            succeed resultSSA
    
    /// Converts printfn operation (adds newline)
    let convertPrintfn (formatString: string) (args: OakExpression list) 
                      (convertExpr: OakExpression -> MLIRParser<string>) : MLIRParser<string> =
        let formatWithNewline = formatString + "\n"
        convertPrintf formatWithNewline args convertExpr
    
    /// Converts I/O operation based on type
    let convert (ioType: IOOperationType) (args: OakExpression list) 
               (convertExpr: OakExpression -> MLIRParser<string>) : MLIRParser<string> =
        match ioType with
        | Printf(formatString) -> convertPrintf formatString args convertExpr
        | Printfn(formatString) -> convertPrintfn formatString args convertExpr
        | ReadLine -> fail "ReadLine operation not implemented"
        | Scanf _ -> fail "Scanf operation not implemented"
        | WriteFile _ | ReadFile _ -> fail "File I/O operations not implemented"

// ======================================
// Expression Conversion
// ======================================

/// Converts Oak expressions to MLIR
let rec convertExpression (expr: OakExpression) : MLIRParser<string> =
    match expr with
    | Literal literal ->
        Literals.convert literal
    
    | Variable varName ->
        SSA.lookupVariable varName
    
    | Let(name, value, body) ->
        convertExpression value >>= fun valueSSA ->
        SSA.bindVariable name valueSSA >>= fun _ ->
        convertExpression body
    
    | Sequential(first, second) ->
        convertExpression first >>= fun _ ->
        convertExpression second
    
    | IOOperation(ioType, args) ->
        IOOperations.convert ioType args convertExpression
    
    | Application(func, args) ->
        match func with
        | Variable funcName ->
            let rec convertArgs remainingArgs accArgs =
                match remainingArgs with
                | [] -> succeed (List.rev accArgs)
                | arg :: rest ->
                    convertExpression arg >>= fun argResult ->
                    convertArgs rest (argResult :: accArgs)
            
            convertArgs args [] >>= fun argSSAValues ->
            SSA.generateValue "call" >>= fun resultSSA ->
            let argStr = String.concat ", " argSSAValues
            let argTypes = List.replicate argSSAValues.Length "i32" |> String.concat ", "
            let operation = sprintf "  %s = func.call @%s(%s) : (%s) -> i32" 
                                   resultSSA funcName argStr argTypes
            Operations.emit operation >>= fun _ ->
            succeed resultSSA
        | _ ->
            fail "Only simple function names supported for calls"
    
    | IfThenElse(cond, thenExpr, elseExpr) ->
        convertExpression cond >>= fun condSSA ->
        SSA.generateValue "if_result" >>= fun resultSSA ->
        
        let thenLabel = "then_block"
        let elseLabel = "else_block" 
        let mergeLabel = "merge_block"
        
        Operations.emit (sprintf "  cf.cond_br %s, ^%s, ^%s" condSSA thenLabel elseLabel) >>= fun _ ->
        Operations.emit (sprintf "^%s:" thenLabel) >>= fun _ ->
        convertExpression thenExpr >>= fun thenSSA ->
        Operations.emit (sprintf "  cf.br ^%s(%s : i32)" mergeLabel thenSSA) >>= fun _ ->
        Operations.emit (sprintf "^%s:" elseLabel) >>= fun _ ->
        convertExpression elseExpr >>= fun elseSSA ->
        Operations.emit (sprintf "  cf.br ^%s(%s : i32)" mergeLabel elseSSA) >>= fun _ ->
        Operations.emit (sprintf "^%s(%s: i32):" mergeLabel resultSSA) >>= fun _ ->
        succeed resultSSA
    
    | FieldAccess(_, fieldName) ->
        fail (sprintf "Field access .%s not implemented" fieldName)
    
    | MethodCall(_, methodName, _) ->
        fail (sprintf "Method call .%s not implemented" methodName)
    
    | Lambda(_, _) ->
        fail "Lambda expressions should be eliminated by closure transformer"

// ======================================
// Declaration Conversion
// ======================================

module Declarations =
    
    /// Converts function declaration
    let convertFunction (name: string) (parameters: (string * OakType) list) 
                       (returnType: OakType) (body: OakExpression) : MLIRParser<unit> =
        
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
        
        Operations.emit funcSig >>= fun _ ->
        SSA.pushScope >>= fun _ ->
        
        // Bind parameters to entry block arguments
        parameters
        |> List.indexed
        |> List.fold (fun acc (i, (paramName, _)) -> 
            acc >>= fun _ ->
            let argSSA = sprintf "%%arg%d" i
            SSA.bindVariable paramName argSSA) (succeed ()) >>= fun _ ->
        
        convertExpression body >>= fun bodySSA ->
        
        let returnOp = 
            if returnType = UnitType then
                "  func.return"
            else
                sprintf "  func.return %s : %s" bodySSA (mlirTypeToString (mapOakTypeToMLIR returnType))
        
        Operations.emit returnOp >>= fun _ ->
        Operations.emit "}" >>= fun _ ->
        SSA.popScope
    
    /// Converts entry point
    let convertEntryPoint (expr: OakExpression) : MLIRParser<unit> =
        Operations.emit "func.func @main(%arg0: i32, %arg1: memref<?xmemref<?xi8>>) -> i32 {" >>= fun _ ->
        SSA.pushScope >>= fun _ ->
        SSA.bindVariable "argc" "%arg0" >>= fun _ ->
        SSA.bindVariable "argv" "%arg1" >>= fun _ ->
        
        convertExpression expr >>= fun exprSSA ->
        
        let returnSSA = 
            match expr with
            | Literal(IntLiteral(_)) -> succeed exprSSA
            | _ ->
                SSA.generateValue "const" >>= fun constSSA ->
                Operations.emit (sprintf "  %s = arith.constant 0 : i32" constSSA) >>= fun _ ->
                succeed constSSA
        
        returnSSA >>= fun finalSSA ->
        Operations.emit (sprintf "  func.return %s : i32" finalSSA) >>= fun _ ->
        Operations.emit "}" >>= fun _ ->
        SSA.popScope
    
    /// Converts type declaration (placeholder)
    let convertType (name: string) (oakType: OakType) : MLIRParser<unit> =
        let comment = sprintf "// Type declaration: %s = %s" name (oakType.ToString())
        Operations.emit comment
    
    /// Converts external declaration
    let convertExternal (name: string) (paramTypes: OakType list) 
                       (returnType: OakType) (libraryName: string) : MLIRParser<unit> =
        let mlirParamTypes = paramTypes |> List.map (mapOakTypeToMLIR >> mlirTypeToString)
        let mlirReturnType = mapOakTypeToMLIR returnType |> mlirTypeToString
        let signature = sprintf "(%s) -> %s" (String.concat ", " mlirParamTypes) mlirReturnType
        Operations.declareExternal name signature
    
    /// Converts any declaration
    let convert (decl: OakDeclaration) : MLIRParser<unit> =
        match decl with
        | FunctionDecl(name, parameters, returnType, body) ->
            convertFunction name parameters returnType body
        | EntryPoint(expr) ->
            convertEntryPoint expr
        | TypeDecl(name, oakType) ->
            convertType name oakType
        | ExternalDecl(name, paramTypes, returnType, libraryName) ->
            convertExternal name paramTypes returnType libraryName

// ======================================
// Main Generation Functions
// ======================================

/// Generates MLIR for a complete program
let generateMLIR (program: OakProgram) : Result<string list * Map<string, string>, string> =
    if program.Modules.IsEmpty then
        Error "Program must contain at least one module"
    else
        let mainModule = program.Modules.[0]
        
        if mainModule.Declarations.IsEmpty then
            Error (sprintf "Module '%s' must contain at least one declaration" mainModule.Name)
        else
            let moduleHeader = sprintf "module %s {" mainModule.Name
            let moduleFooter = "}"
            
            let generationParser = 
                setMLIRState (createMLIRState()) >>= fun _ ->
                Operations.emit moduleHeader >>= fun _ ->
                
                // Convert all declarations
                mainModule.Declarations
                |> List.fold (fun acc decl -> 
                    acc >>= fun _ -> Declarations.convert decl) (succeed ()) >>= fun _ ->
                
                Operations.emit moduleFooter >>= fun _ ->
                getMLIRState
            
            match runParser generationParser "" with
            | Ok state ->
                let operations = List.rev state.Operations
                let ssaMappings = state.CurrentScope
                Ok (operations, ssaMappings)
            | Error e -> Error e

/// Generates complete MLIR module text
let generateMLIRModuleText (program: OakProgram) : Result<string, string> =
    match generateMLIR program with
    | Ok (operations, _) -> Ok (String.concat "\n" operations)
    | Error e -> Error e