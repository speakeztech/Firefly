module Dabbit.Closures.ClosureTransformer

open System
open Core.XParsec.Foundation
open Dabbit.Parsing.OakAst

/// Lifted closure function with minimal metadata
type LiftedFunction = {
    Name: string
    Parameters: (string * OakType) list
    Body: OakExpression
    ReturnType: OakType
}

/// Simplified closure analysis state
type ClosureState = {
    LiftedFunctions: LiftedFunction list
    FunctionCounter: int
    CurrentScope: Set<string>
    GlobalFunctions: Set<string>
}

/// Creates initial closure transformation state
let createInitialState (globalFunctions: Set<string>) : ClosureState = {
    LiftedFunctions = []
    FunctionCounter = 0
    CurrentScope = globalFunctions
    GlobalFunctions = globalFunctions
}

/// Core closure analysis functions
module ClosureAnalysis =
    
    /// Finds free variables in an expression
    let rec findFreeVariables (scope: Set<string>) (expr: OakExpression) : Set<string> =
        match expr with
        | Variable name ->
            if Set.contains name scope then Set.empty
            else Set.singleton name
        
        | Application(func, args) ->
            let funcFree = findFreeVariables scope func
            let argsFree = args |> List.map (findFreeVariables scope) |> Set.unionMany
            Set.union funcFree argsFree
        
        | Lambda(params', body) ->
            let paramNames = params' |> List.map fst |> Set.ofList
            let extendedScope = Set.union scope paramNames
            findFreeVariables extendedScope body
        
        | Let(name, value, body) ->
            let valueFree = findFreeVariables scope value
            let extendedScope = Set.add name scope
            let bodyFree = findFreeVariables extendedScope body
            Set.union valueFree bodyFree
        
        | IfThenElse(cond, thenExpr, elseExpr) ->
            [cond; thenExpr; elseExpr]
            |> List.map (findFreeVariables scope)
            |> Set.unionMany
        
        | Sequential(first, second) ->
            Set.union (findFreeVariables scope first) (findFreeVariables scope second)
        
        | FieldAccess(target, _) ->
            findFreeVariables scope target
        
        | MethodCall(target, _, args) ->
            let targetFree = findFreeVariables scope target
            let argsFree = args |> List.map (findFreeVariables scope) |> Set.unionMany
            Set.union targetFree argsFree
        
        | Literal _ ->
            Set.empty
        
        | IOOperation(_, args) ->
            args |> List.map (findFreeVariables scope) |> Set.unionMany
    
    /// Checks if an expression contains any closures
    let rec hasClosures (expr: OakExpression) : bool =
        match expr with
        | Lambda(_, _) -> true
        | Application(func, args) -> hasClosures func || List.exists hasClosures args
        | Let(_, value, body) -> hasClosures value || hasClosures body
        | IfThenElse(cond, thenExpr, elseExpr) -> hasClosures cond || hasClosures thenExpr || hasClosures elseExpr
        | Sequential(first, second) -> hasClosures first || hasClosures second
        | FieldAccess(target, _) -> hasClosures target
        | MethodCall(target, _, args) -> hasClosures target || List.exists hasClosures args
        | IOOperation(_, args) -> List.exists hasClosures args
        | Variable _ | Literal _ -> false

/// Simple closure transformation without excessive XParsec
module ClosureTransformation =
    
    /// Generates a unique closure function name
    let generateClosureName (state: ClosureState) : string * ClosureState =
        let name = sprintf "_closure_%d" (state.FunctionCounter + 1)
        let newState = { state with FunctionCounter = state.FunctionCounter + 1 }
        (name, newState)
    
    /// Lifts a lambda to a top-level function
    let liftLambda (lambda: OakExpression) (state: ClosureState) : (string * ClosureState) =
        match lambda with
        | Lambda(params', body) ->
            let (closureName, state1) = generateClosureName state
            let freeVars = ClosureAnalysis.findFreeVariables state.CurrentScope lambda |> Set.toList
            let capturedParams = freeVars |> List.map (fun name -> (name, UnitType))
            let allParams = capturedParams @ params'
            
            let liftedFunction = {
                Name = closureName
                Parameters = allParams
                Body = body
                ReturnType = UnitType
            }
            
            let state2 = { state1 with LiftedFunctions = liftedFunction :: state1.LiftedFunctions }
            (closureName, state2)
        
        | _ -> failwith "Expected lambda expression"
    
    /// Transforms an expression by eliminating closures
    let rec transformExpression (expr: OakExpression) (state: ClosureState) : (OakExpression * ClosureState) =
        match expr with
        | Lambda(params', body) ->
            let (closureName, state1) = liftLambda expr state
            let freeVars = ClosureAnalysis.findFreeVariables state.CurrentScope expr |> Set.toList
            let capturedArgs = freeVars |> List.map Variable
            
            if capturedArgs.IsEmpty then
                (Variable(closureName), state1)
            else
                (Application(Variable(closureName), capturedArgs), state1)
        
        | Application(func, args) ->
            let (transformedFunc, state1) = transformExpression func state
            let (transformedArgs, finalState) = 
                List.fold (fun (accArgs, accState) arg ->
                    let (transformedArg, newState) = transformExpression arg accState
                    (transformedArg :: accArgs, newState)
                ) ([], state1) args
            (Application(transformedFunc, List.rev transformedArgs), finalState)
        
        | Let(name, value, body) ->
            let (transformedValue, state1) = transformExpression value state
            let extendedScope = Set.add name state1.CurrentScope
            let state2 = { state1 with CurrentScope = extendedScope }
            let (transformedBody, state3) = transformExpression body state2
            let state4 = { state3 with CurrentScope = state.CurrentScope }
            (Let(name, transformedValue, transformedBody), state4)
        
        | IfThenElse(cond, thenExpr, elseExpr) ->
            let (transformedCond, state1) = transformExpression cond state
            let (transformedThen, state2) = transformExpression thenExpr state1
            let (transformedElse, state3) = transformExpression elseExpr state2
            (IfThenElse(transformedCond, transformedThen, transformedElse), state3)
        
        | Sequential(first, second) ->
            let (transformedFirst, state1) = transformExpression first state
            let (transformedSecond, state2) = transformExpression second state1
            (Sequential(transformedFirst, transformedSecond), state2)
        
        | FieldAccess(target, fieldName) ->
            let (transformedTarget, state1) = transformExpression target state
            (FieldAccess(transformedTarget, fieldName), state1)
        
        | MethodCall(target, methodName, args) ->
            let (transformedTarget, state1) = transformExpression target state
            let (transformedArgs, finalState) = 
                List.fold (fun (accArgs, accState) arg ->
                    let (transformedArg, newState) = transformExpression arg accState
                    (transformedArg :: accArgs, newState)
                ) ([], state1) args
            (MethodCall(transformedTarget, methodName, List.rev transformedArgs), finalState)
        
        | IOOperation(ioType, args) ->
            let (transformedArgs, finalState) = 
                List.fold (fun (accArgs, accState) arg ->
                    let (transformedArg, newState) = transformExpression arg accState
                    (transformedArg :: accArgs, newState)
                ) ([], state) args
            (IOOperation(ioType, List.rev transformedArgs), finalState)
        
        | Variable _ | Literal _ ->
            (expr, state)

/// Declaration transformation
module DeclarationTransformation =
    
    /// Transforms a function declaration
    let transformFunctionDeclaration (name: string) (params': (string * OakType) list) (returnType: OakType) (body: OakExpression) (state: ClosureState) : (OakDeclaration list * ClosureState) =
        let paramNames = params' |> List.map fst |> Set.ofList
        let functionScope = Set.union state.CurrentScope paramNames
        let state1 = { state with CurrentScope = functionScope }
        
        let (transformedBody, state2) = ClosureTransformation.transformExpression body state1
        let mainFunction = FunctionDecl(name, params', returnType, transformedBody)
        
        let liftedDeclarations = 
            state2.LiftedFunctions 
            |> List.map (fun lf -> FunctionDecl(lf.Name, lf.Parameters, lf.ReturnType, lf.Body))
        
        let allDeclarations = mainFunction :: liftedDeclarations
        let clearedState = { state2 with LiftedFunctions = []; CurrentScope = state.CurrentScope }
        
        (allDeclarations, clearedState)
    
    /// Transforms an entry point declaration
    let transformEntryPointDeclaration (expr: OakExpression) (state: ClosureState) : (OakDeclaration list * ClosureState) =
        let (transformedExpr, state1) = ClosureTransformation.transformExpression expr state
        let mainEntry = EntryPoint(transformedExpr)
        
        let liftedDeclarations = 
            state1.LiftedFunctions 
            |> List.map (fun lf -> FunctionDecl(lf.Name, lf.Parameters, lf.ReturnType, lf.Body))
        
        let allDeclarations = mainEntry :: liftedDeclarations
        let clearedState = { state1 with LiftedFunctions = [] }
        
        (allDeclarations, clearedState)
    
    /// Transforms any declaration
    let transformDeclaration (decl: OakDeclaration) (state: ClosureState) : (OakDeclaration list * ClosureState) =
        match decl with
        | FunctionDecl(name, params', returnType, body) ->
            transformFunctionDeclaration name params' returnType body state
        
        | EntryPoint(expr) ->
            transformEntryPointDeclaration expr state
        
        | TypeDecl(_, _) | ExternalDecl(_, _, _, _) ->
            ([decl], state)

/// Module transformation
module ModuleTransformation =
    
    /// Builds global scope from function declarations
    let buildGlobalScope (declarations: OakDeclaration list) : Set<string> =
        declarations
        |> List.choose (function
            | FunctionDecl(name, _, _, _) -> Some name
            | ExternalDecl(name, _, _, _) -> Some name
            | _ -> None)
        |> Set.ofList
    
    /// Transforms a complete module
    let transformModule (module': OakModule) : CompilerResult<OakModule> =
        let globalScope = buildGlobalScope module'.Declarations
        let initialState = createInitialState globalScope
        
        try
            let (transformedDeclarations, _) = 
                module'.Declarations
                |> List.fold (fun (accDecls, accState) decl ->
                    let (newDecls, newState) = DeclarationTransformation.transformDeclaration decl accState
                    (accDecls @ newDecls, newState)
                ) ([], initialState)
            
            Success { module' with Declarations = transformedDeclarations }
        
        with ex ->
            CompilerFailure [ConversionError("closure-elimination", module'.Name, "transformed module", ex.Message)]

/// Validation functions
module ClosureValidation =
    
    /// Validates that no closures remain in an expression
    let rec validateNoClosures (expr: OakExpression) : CompilerResult<unit> =
        match expr with
        | Lambda(_, _) ->
            CompilerFailure [ConversionError("closure-validation", "lambda expression", "eliminated closure", "Lambda expression found after closure elimination")]
        
        | Application(func, args) ->
            match validateNoClosures func with
            | Success () ->
                let argResults = args |> List.map validateNoClosures
                let combinedResult = ResultHelpers.combineResults argResults
                match combinedResult with
                | Success _ -> Success ()
                | CompilerFailure errors -> CompilerFailure errors
            | CompilerFailure errors -> CompilerFailure errors
        
        | Let(_, value, body) ->
            match validateNoClosures value with
            | Success () -> validateNoClosures body
            | CompilerFailure errors -> CompilerFailure errors
        
        | IfThenElse(cond, thenExpr, elseExpr) ->
            match validateNoClosures cond with
            | Success () ->
                match validateNoClosures thenExpr with
                | Success () -> validateNoClosures elseExpr
                | CompilerFailure errors -> CompilerFailure errors
            | CompilerFailure errors -> CompilerFailure errors
        
        | Sequential(first, second) ->
            match validateNoClosures first with
            | Success () -> validateNoClosures second
            | CompilerFailure errors -> CompilerFailure errors
        
        | FieldAccess(target, _) ->
            validateNoClosures target
        
        | MethodCall(target, _, args) ->
            match validateNoClosures target with
            | Success () ->
                let argResults = args |> List.map validateNoClosures
                let combinedResult = ResultHelpers.combineResults argResults
                match combinedResult with
                | Success _ -> Success ()
                | CompilerFailure errors -> CompilerFailure errors
            | CompilerFailure errors -> CompilerFailure errors
        
        | IOOperation(_, args) ->
            let argResults = args |> List.map validateNoClosures
            let combinedResult = ResultHelpers.combineResults argResults
            match combinedResult with
            | Success _ -> Success ()
            | CompilerFailure errors -> CompilerFailure errors
        
        | Variable _ | Literal _ ->
            Success ()
    
    /// Validates that a declaration contains no closures
    let validateDeclarationNoClosures (decl: OakDeclaration) : CompilerResult<unit> =
        match decl with
        | FunctionDecl(_, _, _, body) -> validateNoClosures body
        | EntryPoint(expr) -> validateNoClosures expr
        | TypeDecl(_, _) | ExternalDecl(_, _, _, _) -> Success ()

/// Main closure elimination entry point
let eliminateClosures (program: OakProgram) : CompilerResult<OakProgram> =
    if program.Modules.IsEmpty then
        CompilerFailure [ConversionError("closure-elimination", "empty program", "transformed program", "Program must contain at least one module")]
    else
        // Transform all modules
        program.Modules
        |> List.map ModuleTransformation.transformModule
        |> ResultHelpers.combineResults
        >>= fun transformedModules ->
        
        let transformedProgram = { program with Modules = transformedModules }
        
        // Validate that no closures remain
        let validationResults = 
            transformedProgram.Modules
            |> List.collect (fun m -> m.Declarations)
            |> List.map ClosureValidation.validateDeclarationNoClosures
        
        let combinedValidation = ResultHelpers.combineResults validationResults
        
        match combinedValidation with
        | Success _ -> Success transformedProgram
        | CompilerFailure errors -> CompilerFailure errors

/// Analyzes closure usage in a program for diagnostics
let analyzeClosureUsage (program: OakProgram) : CompilerResult<Map<string, int>> =
    let rec countClosures (expr: OakExpression) : int =
        match expr with
        | Lambda(_, body) -> 1 + countClosures body
        | Application(func, args) -> countClosures func + (args |> List.sumBy countClosures)
        | Let(_, value, body) -> countClosures value + countClosures body
        | IfThenElse(cond, thenExpr, elseExpr) -> countClosures cond + countClosures thenExpr + countClosures elseExpr
        | Sequential(first, second) -> countClosures first + countClosures second
        | FieldAccess(target, _) -> countClosures target
        | MethodCall(target, _, args) -> countClosures target + (args |> List.sumBy countClosures)
        | IOOperation(_, args) -> args |> List.sumBy countClosures
        | Variable _ | Literal _ -> 0
    
    let countClosuresInDeclaration (decl: OakDeclaration) : (string * int) =
        match decl with
        | FunctionDecl(name, _, _, body) -> (name, countClosures body)
        | EntryPoint(_) -> ("__entry__", 0)
        | TypeDecl(name, _) -> (name, 0)
        | ExternalDecl(name, _, _, _) -> (name, 0)
    
    try
        let closureCounts = 
            program.Modules
            |> List.collect (fun m -> m.Declarations)
            |> List.map countClosuresInDeclaration
            |> Map.ofList
        
        Success closureCounts
    
    with ex ->
        CompilerFailure [ConversionError("closure-analysis", "program", "closure usage statistics", ex.Message)]