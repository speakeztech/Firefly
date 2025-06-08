module Dabbit.Closures.ClosureTransformer

open System
open Dabbit.Parsing.OakAst

/// Represents a captured variable in a closure
type CapturedVariable = {
    Name: string
    Type: OakType
    OriginalName: string
}

/// Represents a lifted closure function
type LiftedClosure = {
    Name: string
    Parameters: (string * OakType) list
    CapturedVars: CapturedVariable list
    Body: OakExpression
    ReturnType: OakType
}

/// Counter for generating unique closure names
let mutable private closureCounter = 0

/// Generates a unique closure function name
let private generateClosureName() =
    closureCounter <- closureCounter + 1
    sprintf "_closure_%d" closureCounter

/// Finds all free variables in an expression (variables not bound locally)
let rec private findFreeVariables (expr: OakExpression) (boundVars: Set<string>) : Set<string> =
    match expr with
    | Variable name -> 
        if boundVars.Contains(name) then Set.empty else Set.singleton name
    
    | Application(func, args) ->
        let funcVars = findFreeVariables func boundVars
        let argVars = args |> List.map (fun arg -> findFreeVariables arg boundVars) |> Set.unionMany
        Set.union funcVars argVars
    
    | Lambda(params', body) ->
        let paramNames = params' |> List.map fst |> Set.ofList
        let newBoundVars = Set.union boundVars paramNames
        findFreeVariables body newBoundVars
    
    | Let(name, value, body) ->
        let valueVars = findFreeVariables value boundVars
        let newBoundVars = Set.add name boundVars
        let bodyVars = findFreeVariables body newBoundVars
        Set.union valueVars bodyVars
    
    | IfThenElse(cond, thenExpr, elseExpr) ->
        let condVars = findFreeVariables cond boundVars
        let thenVars = findFreeVariables thenExpr boundVars
        let elseVars = findFreeVariables elseExpr boundVars
        Set.unionMany [condVars; thenVars; elseVars]
    
    | Sequential(first, second) ->
        let firstVars = findFreeVariables first boundVars
        let secondVars = findFreeVariables second boundVars
        Set.union firstVars secondVars
    
    | FieldAccess(target, _) ->
        findFreeVariables target boundVars
    
    | MethodCall(target, _, args) ->
        let targetVars = findFreeVariables target boundVars
        let argVars = args |> List.map (fun arg -> findFreeVariables arg boundVars) |> Set.unionMany
        Set.union targetVars argVars
    
    | Literal _ -> Set.empty

/// Transforms a lambda expression into a lifted function and application
let private transformLambda (lambda: OakExpression) (capturedVars: CapturedVariable list) : LiftedClosure * OakExpression =
    match lambda with
    | Lambda(params', body) ->
        let closureName = generateClosureName()
        
        // Add captured variables as additional parameters
        let capturedParams = capturedVars |> List.map (fun cv -> (cv.Name, cv.Type))
        let allParams = params' @ capturedParams
        
        let liftedClosure = {
            Name = closureName
            Parameters = allParams
            CapturedVars = capturedVars
            Body = body
            ReturnType = UnitType // Simplified - would need type inference
        }
        
        // Create a partial application with captured variables
        let capturedArgs = capturedVars |> List.map (fun cv -> Variable(cv.OriginalName))
        let closureRef = Variable(closureName)
        let partialApp = 
            if capturedArgs.IsEmpty then
                closureRef
            else
                Application(closureRef, capturedArgs)
        
        (liftedClosure, partialApp)
    
    | _ -> failwith "Expected lambda expression"

/// Transforms an expression by eliminating closures
let rec private transformExpression (expr: OakExpression) (globalScope: Set<string>) : OakExpression * LiftedClosure list =
    match expr with
    | Lambda(params', body) ->
        let paramNames = params' |> List.map fst |> Set.ofList
        let freeVars = findFreeVariables expr (Set.union globalScope paramNames)
        
        if freeVars.IsEmpty then
            // No captured variables - can be lifted as-is
            let closureName = generateClosureName()
            let liftedClosure = {
                Name = closureName
                Parameters = params'
                CapturedVars = []
                Body = body
                ReturnType = UnitType
            }
            (Variable(closureName), [liftedClosure])
        else
            // Has captured variables - need to transform
            let capturedVars = 
                freeVars 
                |> Set.toList 
                |> List.map (fun name -> { Name = name; Type = UnitType; OriginalName = name })
            
            let (liftedClosure, transformedExpr) = transformLambda expr capturedVars
            (transformedExpr, [liftedClosure])
    
    | Application(func, args) ->
        let (transformedFunc, funcClosures) = transformExpression func globalScope
        let argResults = args |> List.map (fun arg -> transformExpression arg globalScope)
        let transformedArgs = argResults |> List.map fst
        let argClosures = argResults |> List.map snd |> List.concat
        let allClosures = funcClosures @ argClosures
        (Application(transformedFunc, transformedArgs), allClosures)
    
    | Let(name, value, body) ->
        let (transformedValue, valueClosures) = transformExpression value globalScope
        let newGlobalScope = Set.add name globalScope
        let (transformedBody, bodyClosures) = transformExpression body newGlobalScope
        let allClosures = valueClosures @ bodyClosures
        (Let(name, transformedValue, transformedBody), allClosures)
    
    | IfThenElse(cond, thenExpr, elseExpr) ->
        let (transformedCond, condClosures) = transformExpression cond globalScope
        let (transformedThen, thenClosures) = transformExpression thenExpr globalScope
        let (transformedElse, elseClosures) = transformExpression elseExpr globalScope
        let allClosures = condClosures @ thenClosures @ elseClosures
        (IfThenElse(transformedCond, transformedThen, transformedElse), allClosures)
    
    | Sequential(first, second) ->
        let (transformedFirst, firstClosures) = transformExpression first globalScope
        let (transformedSecond, secondClosures) = transformExpression second globalScope
        let allClosures = firstClosures @ secondClosures
        (Sequential(transformedFirst, transformedSecond), allClosures)
    
    | FieldAccess(target, fieldName) ->
        let (transformedTarget, targetClosures) = transformExpression target globalScope
        (FieldAccess(transformedTarget, fieldName), targetClosures)
    
    | MethodCall(target, methodName, args) ->
        let (transformedTarget, targetClosures) = transformExpression target globalScope
        let argResults = args |> List.map (fun arg -> transformExpression arg globalScope)
        let transformedArgs = argResults |> List.map fst
        let argClosures = argResults |> List.map snd |> List.concat
        let allClosures = targetClosures @ argClosures
        (MethodCall(transformedTarget, methodName, transformedArgs), allClosures)
    
    | Variable _ | Literal _ -> (expr, [])

/// Transforms a declaration by eliminating closures
let private transformDeclaration (decl: OakDeclaration) (globalScope: Set<string>) : OakDeclaration list =
    match decl with
    | FunctionDecl(name, params', returnType, body) ->
        let paramNames = params' |> List.map fst |> Set.ofList
        let declScope = Set.union globalScope paramNames
        let (transformedBody, liftedClosures) = transformExpression body declScope
        
        let mainFunction = FunctionDecl(name, params', returnType, transformedBody)
        let closureFunctions = 
            liftedClosures 
            |> List.map (fun lc -> FunctionDecl(lc.Name, lc.Parameters, lc.ReturnType, lc.Body))
        
        mainFunction :: closureFunctions
    
    | EntryPoint(expr) ->
        let (transformedExpr, liftedClosures) = transformExpression expr globalScope
        let mainEntry = EntryPoint(transformedExpr)
        let closureFunctions = 
            liftedClosures 
            |> List.map (fun lc -> FunctionDecl(lc.Name, lc.Parameters, lc.ReturnType, lc.Body))
        
        mainEntry :: closureFunctions
    
    | TypeDecl(_, _) -> [decl] // Type declarations unchanged

/// Transforms a module by eliminating closures
let private transformModule (module': OakModule) : OakModule =
    // Build global scope from function declarations
    let globalScope = 
        module'.Declarations
        |> List.choose (function
            | FunctionDecl(name, _, _, _) -> Some name
            | _ -> None)
        |> Set.ofList
    
    let transformedDeclarations = 
        module'.Declarations
        |> List.collect (fun decl -> transformDeclaration decl globalScope)
    
    { module' with Declarations = transformedDeclarations }

/// Transforms closures in Oak AST to use explicit parameters instead of captures,
/// eliminating heap allocations for closures
let eliminateClosures (program: OakProgram) : OakProgram =
    let transformedModules = 
        program.Modules 
        |> List.map transformModule
    
    { program with Modules = transformedModules }