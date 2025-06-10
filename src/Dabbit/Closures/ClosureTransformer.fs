module Dabbit.Closures.ClosureTransformer

open System
open Dabbit.Parsing.OakAst

/// Represents an error during closure transformation
type TransformError = {
    Phase: string
    Message: string
    Details: string option
}

/// Result type for closure transformation operations
type TransformResult<'T> = 
    | Success of 'T
    | Failure of TransformError list

/// Computation expression builder for TransformResult
type TransformResultBuilder() =
    member _.Return(x) = Success x
    member _.ReturnFrom(m: TransformResult<'T>) = m
    member _.Bind(m: TransformResult<'T>, f: 'T -> TransformResult<'U>) =
        match m with
        | Success x -> f x
        | Failure errors -> Failure errors
    member _.Zero() = Success ()
    member _.Combine(m1: TransformResult<unit>, m2: TransformResult<'T>) =
        match m1 with
        | Success () -> m2
        | Failure errors -> Failure errors
    member _.Delay(f: unit -> TransformResult<'T>) = f
    member _.Run(f: unit -> TransformResult<'T>) = f()
    member _.TryWith(m: unit -> TransformResult<'T>, h: exn -> TransformResult<'T>) =
        try m() with e -> h e
    member _.TryFinally(m: unit -> TransformResult<'T>, compensation: unit -> unit) =
        try m() finally compensation()
    member this.Using(res: 'T when 'T :> IDisposable, body: 'T -> TransformResult<'U>) =
        try body res
        finally if not (isNull (box res)) then res.Dispose()
    member this.While(guard: unit -> bool, body: unit -> TransformResult<unit>) =
        if guard() then 
            match body() with
            | Success () -> this.While(guard, body)
            | Failure errors -> Failure errors
        else Success ()
    member this.For(sequence: seq<'T>, body: 'T -> TransformResult<unit>) =
        this.Using(sequence.GetEnumerator(), fun enum ->
            this.While(enum.MoveNext, 
                this.Delay(fun () -> body enum.Current)))

let transform = TransformResultBuilder()

/// Closure analysis state for tracking captured variables and transformations
type ClosureAnalysisState = {
    GlobalScope: Set<string>
    CurrentScope: Set<string>
    ScopeStack: Set<string> list
    CapturedVariables: Map<string, CapturedVariable list>
    LiftedFunctions: LiftedClosure list
    TransformationMappings: Map<string, string>
}

/// Represents a captured variable in a closure with full type information
and CapturedVariable = {
    Name: string
    Type: OakType
    OriginalName: string
    CaptureContext: string
    IsParameter: bool
}

/// Represents a lifted closure function with complete transformation metadata
and LiftedClosure = {
    Name: string
    OriginalLambda: OakExpression
    Parameters: (string * OakType) list
    CapturedVars: CapturedVariable list
    Body: OakExpression
    ReturnType: OakType
    CallSites: string list
}

/// Module for scope management
module ScopeManagement =
    /// Pushes a new scope with parameters
    let pushScopeWithParams (parameters: Set<string>) (state: ClosureAnalysisState) : ClosureAnalysisState =
        let newScopeStack = state.CurrentScope :: state.ScopeStack
        let newCurrentScope = Set.union state.CurrentScope parameters
        { state with 
            CurrentScope = newCurrentScope
            ScopeStack = newScopeStack 
        }
    
    /// Pops the current scope
    let popScope (state: ClosureAnalysisState) : TransformResult<ClosureAnalysisState> =
        match state.ScopeStack with
        | prevScope :: rest ->
            Success { state with 
                              CurrentScope = prevScope
                              ScopeStack = rest }
        | [] ->
            Failure [{ Phase = "scope management"; Message = "Cannot pop scope"; Details = Some "Scope stack is empty" }]
    
    /// Binds a variable in the current scope
    let bindVariable (varName: string) (state: ClosureAnalysisState) : ClosureAnalysisState =
        let newCurrentScope = Set.add varName state.CurrentScope
        { state with CurrentScope = newCurrentScope }

/// Module for free variable analysis
module FreeVariableAnalysis =
    /// Analyzes an expression for free variables
    let rec analyzeFreeVariables (expr: OakExpression) (state: ClosureAnalysisState) : TransformResult<Set<string>> =
        match expr with
        | Variable name ->
            if Set.contains name (Set.union state.GlobalScope state.CurrentScope) then
                Success Set.empty
            else
                Success (Set.singleton name)
        
        | Application(func, args) ->
            // Analyze function expression
            analyzeFreeVariables func state
            |> (fun funcFreeResult ->
                match funcFreeResult with
                | Success funcFree ->
                    // Analyze all arguments and combine results
                    let rec analyzeArgs remainingArgs accFree =
                        match remainingArgs with
                        | [] -> Success accFree
                        | arg :: rest ->
                            analyzeFreeVariables arg state
                            |> (fun argFreeResult ->
                                match argFreeResult with
                                | Success argFree ->
                                    analyzeArgs rest (Set.union accFree argFree)
                                | Failure errors -> Failure errors)
                    
                    analyzeArgs args funcFree
                | Failure errors -> Failure errors)
        
        | Lambda(params', body) ->
            let paramNames = params' |> List.map fst |> Set.ofList
            let stateWithParams = ScopeManagement.pushScopeWithParams paramNames state
            
            analyzeFreeVariables body stateWithParams
            |> (fun result ->
                match result with
                | Success bodyFree -> 
                    Success (Set.difference bodyFree paramNames)
                | Failure errors -> Failure errors)
        
        | Let(name, value, body) ->
            analyzeFreeVariables value state
            |> (fun valueFreeResult ->
                match valueFreeResult with
                | Success valueFree ->
                    let stateWithBinding = ScopeManagement.bindVariable name state
                    analyzeFreeVariables body stateWithBinding
                    |> (fun bodyFreeResult ->
                        match bodyFreeResult with
                        | Success bodyFree ->
                            Success (Set.union valueFree bodyFree)
                        | Failure errors -> Failure errors)
                | Failure errors -> Failure errors)
        
        | IfThenElse(cond, thenExpr, elseExpr) ->
            analyzeFreeVariables cond state
            |> (fun condFreeResult ->
                match condFreeResult with
                | Success condFree ->
                    analyzeFreeVariables thenExpr state
                    |> (fun thenFreeResult ->
                        match thenFreeResult with
                        | Success thenFree ->
                            analyzeFreeVariables elseExpr state
                            |> (fun elseFreeResult ->
                                match elseFreeResult with
                                | Success elseFree ->
                                    Success (Set.unionMany [condFree; thenFree; elseFree])
                                | Failure errors -> Failure errors)
                        | Failure errors -> Failure errors)
                | Failure errors -> Failure errors)
        
        | Sequential(first, second) ->
            analyzeFreeVariables first state
            |> (fun firstFreeResult ->
                match firstFreeResult with
                | Success firstFree ->
                    analyzeFreeVariables second state
                    |> (fun secondFreeResult ->
                        match secondFreeResult with
                        | Success secondFree ->
                            Success (Set.union firstFree secondFree)
                        | Failure errors -> Failure errors)
                | Failure errors -> Failure errors)
        
        | FieldAccess(target, _) ->
            analyzeFreeVariables target state
        
        | MethodCall(target, _, args) ->
            analyzeFreeVariables target state
            |> (fun targetFreeResult ->
                match targetFreeResult with
                | Success targetFree ->
                    // Analyze all arguments
                    let rec analyzeArgs remainingArgs accFree =
                        match remainingArgs with
                        | [] -> Success accFree
                        | arg :: rest ->
                            analyzeFreeVariables arg state
                            |> (fun argFreeResult ->
                                match argFreeResult with
                                | Success argFree ->
                                    analyzeArgs rest (Set.union accFree argFree)
                                | Failure errors -> Failure errors)
                    
                    analyzeArgs args targetFree
                | Failure errors -> Failure errors)
        
        | Literal _ | IOOperation(_, _) ->
            Success Set.empty

/// Module for closure lifting
module ClosureLifting =
    /// Generates a unique closure function name
    let generateClosureName (state: ClosureAnalysisState) : string * ClosureAnalysisState =
        let existingCount = state.LiftedFunctions.Length
        let name = sprintf "_closure_%d" (existingCount + 1)
        (name, state)
    
    /// Creates captured variable metadata
    let createCapturedVariable (name: string) (originalContext: string) (ty: OakType) : CapturedVariable =
        {
            Name = name
            Type = ty
            OriginalName = name
            CaptureContext = originalContext
            IsParameter = false
        }
    
    /// Records a lifted closure
    let recordLiftedClosure (closure: LiftedClosure) (state: ClosureAnalysisState) : ClosureAnalysisState =
        { state with LiftedFunctions = closure :: state.LiftedFunctions }
    
    /// Transforms a lambda expression to a lifted function call
    let transformLambdaToCall (lambda: OakExpression) (state: ClosureAnalysisState) : TransformResult<OakExpression * ClosureAnalysisState> =
        match lambda with
        | Lambda(params', body) ->
            let (closureName, state1) = generateClosureName state
            
            FreeVariableAnalysis.analyzeFreeVariables lambda state1
            |> (fun freeVarsResult ->
                match freeVarsResult with
                | Success freeVars ->
                    // Create captured variables for all free variables
                    let capturedVars = 
                        freeVars
                        |> Set.toList
                        |> List.map (fun varName -> 
                            createCapturedVariable varName "lambda-capture" UnitType)
                    
                    // Create lifted closure
                    let capturedParams = capturedVars |> List.map (fun cv -> (cv.Name, cv.Type))
                    let allParams = params' @ capturedParams
                    let liftedClosure = {
                        Name = closureName
                        OriginalLambda = lambda
                        Parameters = allParams
                        CapturedVars = capturedVars
                        Body = body
                        ReturnType = UnitType  // Would need type inference
                        CallSites = []
                    }
                    
                    let state2 = recordLiftedClosure liftedClosure state1
                    
                    // Create function call expression
                    let capturedArgs = capturedVars |> List.map (fun cv -> Variable(cv.OriginalName))
                    let functionCall = 
                        if capturedArgs.IsEmpty then
                            Variable(closureName)
                        else
                            Application(Variable(closureName), capturedArgs)
                    
                    Success (functionCall, state2)
                | Failure errors -> Failure errors)
        
        | _ ->
            Failure [{ Phase = "lambda transformation"; Message = "Expected lambda expression"; Details = Some "Non-lambda expression provided for transformation" }]

/// Module for expression transformation
module ExpressionTransformation =
    /// Transforms an expression by eliminating closures
    let rec transformExpression (expr: OakExpression) (state: ClosureAnalysisState) : TransformResult<OakExpression * ClosureAnalysisState> =
        match expr with
        | Lambda(_, _) ->
            // Transform lambda to lifted function call
            ClosureLifting.transformLambdaToCall expr state
        
        | Application(func, args) ->
            transformExpression func state
            |> (fun funcResult ->
                match funcResult with
                | Success (transformedFunc, state1) ->
                    // Transform all arguments
                    let rec transformArgs remainingArgs accArgs currentState =
                        match remainingArgs with
                        | [] -> Success (List.rev accArgs, currentState)
                        | arg :: rest ->
                            transformExpression arg currentState
                            |> (fun argResult ->
                                match argResult with
                                | Success (transformedArg, newState) ->
                                    transformArgs rest (transformedArg :: accArgs) newState
                                | Failure errors -> Failure errors)
                    
                    transformArgs args [] state1
                    |> (fun argsResult ->
                        match argsResult with
                        | Success (transformedArgs, finalState) ->
                            Success (Application(transformedFunc, transformedArgs), finalState)
                        | Failure errors -> Failure errors)
                | Failure errors -> Failure errors)
        
        | Let(name, value, body) ->
            transformExpression value state
            |> (fun valueResult ->
                match valueResult with
                | Success (transformedValue, state1) ->
                    let state2 = ScopeManagement.bindVariable name state1
                    transformExpression body state2
                    |> (fun bodyResult ->
                        match bodyResult with
                        | Success (transformedBody, state3) ->
                            Success (Let(name, transformedValue, transformedBody), state3)
                        | Failure errors -> Failure errors)
                | Failure errors -> Failure errors)
        
        | IfThenElse(cond, thenExpr, elseExpr) ->
            transformExpression cond state
            |> (fun condResult ->
                match condResult with
                | Success (transformedCond, state1) ->
                    transformExpression thenExpr state1
                    |> (fun thenResult ->
                        match thenResult with
                        | Success (transformedThen, state2) ->
                            transformExpression elseExpr state2
                            |> (fun elseResult ->
                                match elseResult with
                                | Success (transformedElse, state3) ->
                                    Success (IfThenElse(transformedCond, transformedThen, transformedElse), state3)
                                | Failure errors -> Failure errors)
                        | Failure errors -> Failure errors)
                | Failure errors -> Failure errors)
        
        | Sequential(first, second) ->
            transformExpression first state
            |> (fun firstResult ->
                match firstResult with
                | Success (transformedFirst, state1) ->
                    transformExpression second state1
                    |> (fun secondResult ->
                        match secondResult with
                        | Success (transformedSecond, state2) ->
                            Success (Sequential(transformedFirst, transformedSecond), state2)
                        | Failure errors -> Failure errors)
                | Failure errors -> Failure errors)
        
        | FieldAccess(target, fieldName) ->
            transformExpression target state
            |> (fun targetResult ->
                match targetResult with
                | Success (transformedTarget, state1) ->
                    Success (FieldAccess(transformedTarget, fieldName), state1)
                | Failure errors -> Failure errors)
        
        | MethodCall(target, methodName, args) ->
            transformExpression target state
            |> (fun targetResult ->
                match targetResult with
                | Success (transformedTarget, state1) ->
                    // Transform all arguments
                    let rec transformArgs remainingArgs accArgs currentState =
                        match remainingArgs with
                        | [] -> Success (List.rev accArgs, currentState)
                        | arg :: rest ->
                            transformExpression arg currentState
                            |> (fun argResult ->
                                match argResult with
                                | Success (transformedArg, newState) ->
                                    transformArgs rest (transformedArg :: accArgs) newState
                                | Failure errors -> Failure errors)
                    
                    transformArgs args [] state1
                    |> (fun argsResult ->
                        match argsResult with
                        | Success (transformedArgs, finalState) ->
                            Success (MethodCall(transformedTarget, methodName, transformedArgs), finalState)
                        | Failure errors -> Failure errors)
                | Failure errors -> Failure errors)
        
        | Variable _ | Literal _ | IOOperation(_, _) ->
            // These expressions don't need transformation
            Success (expr, state)

/// Module for declaration transformation
module DeclarationTransformation =
    /// Transforms a function declaration by eliminating closures
    let transformFunctionDeclaration (name: string) (params': (string * OakType) list) (returnType: OakType) 
                                     (body: OakExpression) (state: ClosureAnalysisState) 
                                     : TransformResult<OakDeclaration list * ClosureAnalysisState> =
        // Set up function scope
        let paramNames = params' |> List.map fst |> Set.ofList
        let stateWithParams = ScopeManagement.pushScopeWithParams paramNames state
        
        // Transform function body
        ExpressionTransformation.transformExpression body stateWithParams
        |> (fun bodyResult ->
            match bodyResult with
            | Success (transformedBody, state1) ->
                // Create main function declaration
                let mainFunction = FunctionDecl(name, params', returnType, transformedBody)
                
                // Get lifted closures and create their declarations
                let liftedDeclarations = 
                    state1.LiftedFunctions 
                    |> List.map (fun lc -> FunctionDecl(lc.Name, lc.Parameters, lc.ReturnType, lc.Body))
                
                let allDeclarations = mainFunction :: liftedDeclarations
                
                // Clean up scope
                ScopeManagement.popScope state1
                |> (fun popResult ->
                    match popResult with
                    | Success state2 -> Success (allDeclarations, state2)
                    | Failure errors -> Failure errors)
            | Failure errors -> Failure errors)
    
    /// Transforms an entry point declaration
    let transformEntryPointDeclaration (expr: OakExpression) (state: ClosureAnalysisState) 
                                       : TransformResult<OakDeclaration list * ClosureAnalysisState> =
        ExpressionTransformation.transformExpression expr state
        |> (fun exprResult ->
            match exprResult with
            | Success (transformedExpr, state1) ->
                let mainEntry = EntryPoint(transformedExpr)
                
                // Get lifted closures and create their declarations
                let liftedDeclarations = 
                    state1.LiftedFunctions 
                    |> List.map (fun lc -> FunctionDecl(lc.Name, lc.Parameters, lc.ReturnType, lc.Body))
                
                let allDeclarations = mainEntry :: liftedDeclarations
                Success (allDeclarations, state1)
            | Failure errors -> Failure errors)
    
    /// Transforms any declaration
    let transformDeclaration (decl: OakDeclaration) (state: ClosureAnalysisState) 
                             : TransformResult<OakDeclaration list * ClosureAnalysisState> =
        match decl with
        | FunctionDecl(name, params', returnType, body) ->
            transformFunctionDeclaration name params' returnType body state
        
        | EntryPoint(expr) ->
            transformEntryPointDeclaration expr state
        
        | TypeDecl(_, _) | ExternalDecl(_, _, _, _) ->
            // Type declarations and external declarations are unchanged
            Success ([decl], state)

/// Module for module transformation
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
    let transformModule (module': OakModule) : TransformResult<OakModule> =
        let globalScope = buildGlobalScope module'.Declarations
        
        let initialState = { 
            GlobalScope = globalScope
            CurrentScope = globalScope
            ScopeStack = []
            CapturedVariables = Map.empty
            LiftedFunctions = []
            TransformationMappings = Map.empty
        }
        
        // Transform all declarations in the module
        let rec transformDeclarations (remaining: OakDeclaration list) (accumulated: OakDeclaration list) (currentState: ClosureAnalysisState) =
            match remaining with
            | [] -> Success (List.rev accumulated, currentState)
            | decl :: rest ->
                DeclarationTransformation.transformDeclaration decl currentState
                |> (fun declResult ->
                    match declResult with
                    | Success (transformedDecls, newState) ->
                        transformDeclarations rest (List.append transformedDecls accumulated) newState
                    | Failure errors -> Failure errors)
        
        transformDeclarations module'.Declarations [] initialState
        |> (fun result ->
            match result with
            | Success (transformedDeclarations, _) ->
                Success { module' with Declarations = transformedDeclarations }
            | Failure errors -> Failure errors)

/// Module for closure validation
module ClosureValidation =
    /// Validates that no closures remain in an expression
    let rec validateNoClosures (expr: OakExpression) : TransformResult<unit> =
        match expr with
        | Lambda(_, _) ->
            Failure [{ 
                Phase = "closure validation"
                Message = "Lambda expression found after closure elimination"
                Details = Some "Lambda expressions should have been transformed to function calls" 
            }]
        
        | Application(func, args) ->
            validateNoClosures func
            |> (fun funcResult ->
                match funcResult with
                | Success () ->
                    let rec validateArgs remainingArgs =
                        match remainingArgs with
                        | [] -> Success ()
                        | arg :: rest ->
                            validateNoClosures arg
                            |> (fun argResult ->
                                match argResult with
                                | Success () -> validateArgs rest
                                | Failure errors -> Failure errors)
                    
                    validateArgs args
                | Failure errors -> Failure errors)
        
        | Let(_, value, body) ->
            validateNoClosures value
            |> (fun valueResult ->
                match valueResult with
                | Success () -> validateNoClosures body
                | Failure errors -> Failure errors)
        
        | IfThenElse(cond, thenExpr, elseExpr) ->
            validateNoClosures cond
            |> (fun condResult ->
                match condResult with
                | Success () ->
                    validateNoClosures thenExpr
                    |> (fun thenResult ->
                        match thenResult with
                        | Success () -> validateNoClosures elseExpr
                        | Failure errors -> Failure errors)
                | Failure errors -> Failure errors)
        
        | Sequential(first, second) ->
            validateNoClosures first
            |> (fun firstResult ->
                match firstResult with
                | Success () -> validateNoClosures second
                | Failure errors -> Failure errors)
        
        | FieldAccess(target, _) ->
            validateNoClosures target
        
        | MethodCall(target, _, args) ->
            validateNoClosures target
            |> (fun targetResult ->
                match targetResult with
                | Success () ->
                    let rec validateArgs remainingArgs =
                        match remainingArgs with
                        | [] -> Success ()
                        | arg :: rest ->
                            validateNoClosures arg
                            |> (fun argResult ->
                                match argResult with
                                | Success () -> validateArgs rest
                                | Failure errors -> Failure errors)
                    
                    validateArgs args
                | Failure errors -> Failure errors)
        
        | Variable _ | Literal _ | IOOperation(_, _) ->
            Success ()
    
    /// Validates that a declaration contains no closures
    let validateDeclarationNoClosures (decl: OakDeclaration) : TransformResult<unit> =
        match decl with
        | FunctionDecl(_, _, _, body) -> validateNoClosures body
        | EntryPoint(expr) -> validateNoClosures expr
        | TypeDecl(_, _) | ExternalDecl(_, _, _, _) -> Success ()
    
    /// Validates that a program contains no closures
    let validateProgramNoClosures (program: OakProgram) : TransformResult<unit> =
        let rec validateModules (modules: OakModule list) =
            match modules with
            | [] -> Success ()
            | m :: rest ->
                let declarations = m.Declarations
                let rec validateDeclarations (declarations: OakDeclaration list) =
                    match declarations with
                    | [] -> Success ()
                    | decl :: rest ->
                        validateDeclarationNoClosures decl
                        |> (fun result ->
                            match result with
                            | Success () -> validateDeclarations rest
                            | Failure errors -> Failure errors)
                
                validateDeclarations declarations
                |> (fun result ->
                    match result with
                    | Success () -> validateModules rest
                    | Failure errors -> Failure errors)
        
        validateModules program.Modules

/// Main closure elimination entry point
let eliminateClosures (program: OakProgram) : Result<OakProgram, string> =
    if program.Modules.IsEmpty then
        Error "Program must contain at least one module"
    else
        // Transform all modules
        let moduleResults = 
            program.Modules
            |> List.map ModuleTransformation.transformModule
        
        // Check if any transformations failed
        let failures = 
            moduleResults 
            |> List.choose (function 
                | Failure errors -> 
                    Some (errors |> List.map (fun e -> sprintf "%s: %s" e.Phase e.Message) |> String.concat "; ")
                | Success _ -> None)
        
        if not failures.IsEmpty then
            Error (String.concat "\n" failures)
        else
            // All transformations succeeded, collect results
            let transformedModules = 
                moduleResults 
                |> List.choose (function 
                    | Success m -> Some m
                    | Failure _ -> None)
            
            let transformedProgram = { program with Modules = transformedModules }
            
            // Validate that no closures remain
            match ClosureValidation.validateProgramNoClosures transformedProgram with
            | Success () -> Ok transformedProgram
            | Failure errors -> 
                Error (errors |> List.map (fun e -> sprintf "%s: %s" e.Phase e.Message) |> String.concat "; ")

/// Analyzes closure usage in a program for diagnostics
let analyzeClosureUsage (program: OakProgram) : Result<Map<string, int>, string> =
    let rec countClosures (expr: OakExpression) : int =
        match expr with
        | Lambda(_, body) -> 1 + countClosures body
        | Application(func, args) -> countClosures func + (args |> List.sumBy countClosures)
        | Let(_, value, body) -> countClosures value + countClosures body
        | IfThenElse(cond, thenExpr, elseExpr) -> countClosures cond + countClosures thenExpr + countClosures elseExpr
        | Sequential(first, second) -> countClosures first + countClosures second
        | FieldAccess(target, _) -> countClosures target
        | MethodCall(target, _, args) -> countClosures target + (args |> List.sumBy countClosures)
        | Variable _ | Literal _ | IOOperation(_, _) -> 0
    
    let countClosuresInDeclaration (decl: OakDeclaration) : (string * int) =
        match decl with
        | FunctionDecl(name, _, _, body) -> (name, countClosures body)
        | EntryPoint(_) -> ("__entry__", 0)
        | TypeDecl(name, _) -> (name, 0)
        | ExternalDecl(name, _, _, _) -> (name, 0)
    
    let closureCounts = 
        program.Modules
        |> List.collect (fun m -> m.Declarations)
        |> List.map countClosuresInDeclaration
        |> Map.ofList
    
    Ok closureCounts

/// Adapter to convert from TransformResult to Core.XParsec.Foundation result types
module CompilerIntegration =
    /// Converts a TransformResult to a format expected by the rest of the compiler
    let convertToCompilerResult (result: TransformResult<'T>) : Result<'T, string> =
        match result with
        | Success value -> Ok value
        | Failure errors -> 
            let errorMessage = 
                errors 
                |> List.map (fun e -> 
                    match e.Details with
                    | Some details -> sprintf "%s: %s (%s)" e.Phase e.Message details
                    | None -> sprintf "%s: %s" e.Phase e.Message)
                |> String.concat "\n"
            Error errorMessage