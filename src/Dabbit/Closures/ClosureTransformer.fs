module Dabbit.Closures.ClosureTransformer

open System
open XParsec
open Core.XParsec.Foundation
open Core.XParsec.Foundation.Combinators
open Core.XParsec.Foundation.ErrorHandling
open Dabbit.Parsing.OakAst

/// Closure analysis state for tracking captured variables and transformations
type ClosureAnalysisState = {
    GlobalScope: Set<string>
    CurrentScope: Set<string>
    ScopeStack: Set<string> list
    CapturedVariables: Map<string, CapturedVariable list>
    LiftedFunctions: LiftedClosure list
    TransformationMappings: Map<string, string>
    ErrorContext: string list
}

/// Represents a captured variable in a closure with full type information
type CapturedVariable = {
    Name: string
    Type: OakType
    OriginalName: string
    CaptureContext: string
    IsParameter: bool
}

/// Represents a lifted closure function with complete transformation metadata
type LiftedClosure = {
    Name: string
    OriginalLambda: OakExpression
    Parameters: (string * OakType) list
    CapturedVars: CapturedVariable list
    Body: OakExpression
    ReturnType: OakType
    CallSites: string list
}

/// Closure transformation patterns using XParsec combinators
module ClosureAnalysisParsers =
    
    /// Analyzes an expression for free variables using XParsec patterns
    let rec analyzeFreeVariables (expr: OakExpression) : Parser<Set<string>, ClosureAnalysisState> =
        match expr with
        | Variable name ->
            fun state ->
                if Set.contains name (Set.union state.GlobalScope state.CurrentScope) then
                    Reply(Ok Set.empty, state)
                else
                    Reply(Ok (Set.singleton name), state)
        
        | Application(func, args) ->
            analyzeFreeVariables func >>= fun funcFree ->
            args 
            |> List.map analyzeFreeVariables
            |> List.fold (fun acc argParser ->
                acc >>= fun accSet ->
                argParser >>= fun argSet ->
                succeed (Set.union accSet argSet)
            ) (succeed Set.empty)
            >>= fun argsFree ->
            succeed (Set.union funcFree argsFree)
        
        | Lambda(params', body) ->
            let paramNames = params' |> List.map fst |> Set.ofList
            pushScopeWithParams paramNames >>= fun _ ->
            analyzeFreeVariables body >>= fun bodyFree ->
            popScope >>= fun _ ->
            succeed (Set.difference bodyFree paramNames)
        
        | Let(name, value, body) ->
            analyzeFreeVariables value >>= fun valueFree ->
            bindVariable name >>= fun _ ->
            analyzeFreeVariables body >>= fun bodyFree ->
            succeed (Set.union valueFree bodyFree)
        
        | IfThenElse(cond, thenExpr, elseExpr) ->
            analyzeFreeVariables cond >>= fun condFree ->
            analyzeFreeVariables thenExpr >>= fun thenFree ->
            analyzeFreeVariables elseExpr >>= fun elseFree ->
            succeed (Set.unionMany [condFree; thenFree; elseFree])
        
        | Sequential(first, second) ->
            analyzeFreeVariables first >>= fun firstFree ->
            analyzeFreeVariables second >>= fun secondFree ->
            succeed (Set.union firstFree secondFree)
        
        | FieldAccess(target, _) ->
            analyzeFreeVariables target
        
        | MethodCall(target, _, args) ->
            analyzeFreeVariables target >>= fun targetFree ->
            args 
            |> List.map analyzeFreeVariables
            |> List.fold (fun acc argParser ->
                acc >>= fun accSet ->
                argParser >>= fun argSet ->
                succeed (Set.union accSet argSet)
            ) (succeed Set.empty)
            >>= fun argsFree ->
            succeed (Set.union targetFree argsFree)
        
        | Literal _ ->
            succeed Set.empty
    
    /// Pushes a new scope with parameters
    and pushScopeWithParams (params: Set<string>) : Parser<unit, ClosureAnalysisState> =
        fun state ->
            let newScopeStack = state.CurrentScope :: state.ScopeStack
            let newCurrentScope = Set.union state.CurrentScope params
            let newState = { 
                state with 
                    CurrentScope = newCurrentScope
                    ScopeStack = newScopeStack 
            }
            Reply(Ok (), newState)
    
    /// Pops the current scope
    and popScope : Parser<unit, ClosureAnalysisState> =
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
    
    /// Binds a variable in the current scope
    and bindVariable (varName: string) : Parser<unit, ClosureAnalysisState> =
        fun state ->
            let newCurrentScope = Set.add varName state.CurrentScope
            let newState = { state with CurrentScope = newCurrentScope }
            Reply(Ok (), newState)

/// Closure lifting using XParsec combinators
module ClosureLiftingParsers =
    
    /// Generates a unique closure function name
    let generateClosureName : Parser<string, ClosureAnalysisState> =
        fun state ->
            let existingCount = state.LiftedFunctions.Length
            let name = sprintf "_closure_%d" (existingCount + 1)
            Reply(Ok name, state)
    
    /// Creates captured variable metadata
    let createCapturedVariable (name: string) (originalContext: string) : Parser<CapturedVariable, ClosureAnalysisState> =
        fun state ->
            let capturedVar = {
                Name = name
                Type = UnitType  // Would need type inference in real implementation
                OriginalName = name
                CaptureContext = originalContext
                IsParameter = false
            }
            Reply(Ok capturedVar, state)
    
    /// Records a lifted closure
    let recordLiftedClosure (closure: LiftedClosure) : Parser<unit, ClosureAnalysisState> =
        fun state ->
            let newState = { 
                state with 
                    LiftedFunctions = closure :: state.LiftedFunctions 
            }
            Reply(Ok (), newState)
    
    /// Transforms a lambda expression to a lifted function call
    let transformLambdaToCall (lambda: OakExpression) : Parser<(LiftedClosure * OakExpression), ClosureAnalysisState> =
        match lambda with
        | Lambda(params', body) ->
            generateClosureName >>= fun closureName ->
            analyzeFreeVariables lambda >>= fun freeVars ->
            
            // Create captured variables for all free variables
            freeVars
            |> Set.toList
            |> List.map (fun varName -> createCapturedVariable varName "lambda-capture")
            |> List.fold (fun acc varParser ->
                acc >>= fun accVars ->
                varParser >>= fun var ->
                succeed (var :: accVars)
            ) (succeed [])
            >>= fun capturedVars ->
            
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
            
            recordLiftedClosure liftedClosure >>= fun _ ->
            
            // Create function call expression
            let capturedArgs = capturedVars |> List.map (fun cv -> Variable(cv.OriginalName))
            let functionCall = 
                if capturedArgs.IsEmpty then
                    Variable(closureName)
                else
                    Application(Variable(closureName), capturedArgs)
            
            succeed (liftedClosure, functionCall)
        
        | _ ->
            compilerFail (TransformError("lambda transformation", "non-lambda expression", "lifted function", "Expected lambda expression"))

/// Expression transformation using XParsec combinators
module ExpressionTransformationParsers =
    
    /// Transforms an expression by eliminating closures
    let rec transformExpression (expr: OakExpression) : Parser<OakExpression, ClosureAnalysisState> =
        match expr with
        | Lambda(params', body) ->
            // Transform lambda to lifted function call
            transformLambdaToCall expr >>= fun (_, callExpr) ->
            succeed callExpr
            |> withErrorContext "lambda expression transformation"
        
        | Application(func, args) ->
            transformExpression func >>= fun transformedFunc ->
            args 
            |> List.map transformExpression
            |> List.fold (fun acc argParser ->
                acc >>= fun accArgs ->
                argParser >>= fun transformedArg ->
                succeed (transformedArg :: accArgs)
            ) (succeed [])
            >>= fun transformedArgs ->
            succeed (Application(transformedFunc, List.rev transformedArgs))
            |> withErrorContext "application expression transformation"
        
        | Let(name, value, body) ->
            transformExpression value >>= fun transformedValue ->
            bindVariable name >>= fun _ ->
            transformExpression body >>= fun transformedBody ->
            succeed (Let(name, transformedValue, transformedBody))
            |> withErrorContext "let expression transformation"
        
        | IfThenElse(cond, thenExpr, elseExpr) ->
            transformExpression cond >>= fun transformedCond ->
            transformExpression thenExpr >>= fun transformedThen ->
            transformExpression elseExpr >>= fun transformedElse ->
            succeed (IfThenElse(transformedCond, transformedThen, transformedElse))
            |> withErrorContext "conditional expression transformation"
        
        | Sequential(first, second) ->
            transformExpression first >>= fun transformedFirst ->
            transformExpression second >>= fun transformedSecond ->
            succeed (Sequential(transformedFirst, transformedSecond))
            |> withErrorContext "sequential expression transformation"
        
        | FieldAccess(target, fieldName) ->
            transformExpression target >>= fun transformedTarget ->
            succeed (FieldAccess(transformedTarget, fieldName))
            |> withErrorContext "field access transformation"
        
        | MethodCall(target, methodName, args) ->
            transformExpression target >>= fun transformedTarget ->
            args 
            |> List.map transformExpression
            |> List.fold (fun acc argParser ->
                acc >>= fun accArgs ->
                argParser >>= fun transformedArg ->
                succeed (transformedArg :: accArgs)
            ) (succeed [])
            >>= fun transformedArgs ->
            succeed (MethodCall(transformedTarget, methodName, List.rev transformedArgs))
            |> withErrorContext "method call transformation"
        
        | Variable _ | Literal _ ->
            succeed expr

/// Declaration transformation using XParsec combinators
module DeclarationTransformationParsers =
    
    /// Transforms a function declaration by eliminating closures
    let transformFunctionDeclaration (name: string) (params': (string * OakType) list) (returnType: OakType) (body: OakExpression) : Parser<OakDeclaration list, ClosureAnalysisState> =
        // Set up function scope
        let paramNames = params' |> List.map fst |> Set.ofList
        pushScopeWithParams paramNames >>= fun _ ->
        
        // Transform function body
        transformExpression body >>= fun transformedBody ->
        
        popScope >>= fun _ ->
        
        // Create main function declaration
        let mainFunction = FunctionDecl(name, params', returnType, transformedBody)
        
        // Get lifted closures and create their declarations
        fun state ->
            let liftedDeclarations = 
                state.LiftedFunctions 
                |> List.map (fun lc -> FunctionDecl(lc.Name, lc.Parameters, lc.ReturnType, lc.Body))
            
            let allDeclarations = mainFunction :: liftedDeclarations
            Reply(Ok allDeclarations, state)
        |> withErrorContext (sprintf "function declaration transformation '%s'" name)
    
    /// Transforms an entry point declaration
    let transformEntryPointDeclaration (expr: OakExpression) : Parser<OakDeclaration list, ClosureAnalysisState> =
        transformExpression expr >>= fun transformedExpr ->
        
        let mainEntry = EntryPoint(transformedExpr)
        
        fun state ->
            let liftedDeclarations = 
                state.LiftedFunctions 
                |> List.map (fun lc -> FunctionDecl(lc.Name, lc.Parameters, lc.ReturnType, lc.Body))
            
            let allDeclarations = mainEntry :: liftedDeclarations
            Reply(Ok allDeclarations, state)
        |> withErrorContext "entry point transformation"
    
    /// Transforms any declaration
    let transformDeclaration (decl: OakDeclaration) : Parser<OakDeclaration list, ClosureAnalysisState> =
        match decl with
        | FunctionDecl(name, params', returnType, body) ->
            transformFunctionDeclaration name params' returnType body
        
        | EntryPoint(expr) ->
            transformEntryPointDeclaration expr
        
        | TypeDecl(_, _) ->
            succeed [decl]  // Type declarations are unchanged

/// Module transformation using XParsec combinators
module ModuleTransformationParsers =
    
    /// Builds global scope from function declarations
    let buildGlobalScope (declarations: OakDeclaration list) : Set<string> =
        declarations
        |> List.choose (function
            | FunctionDecl(name, _, _, _) -> Some name
            | _ -> None)
        |> Set.ofList
    
    /// Transforms a complete module
    let transformModule (module': OakModule) : Parser<OakModule, ClosureAnalysisState> =
        let globalScope = buildGlobalScope module'.Declarations
        
        fun state ->
            let initialState = { 
                state with 
                    GlobalScope = globalScope
                    CurrentScope = globalScope
            }
            
            // Transform all declarations
            let transformAllDeclarations (declarations: OakDeclaration list) : Parser<OakDeclaration list, ClosureAnalysisState> =
                declarations
                |> List.map transformDeclaration
                |> List.fold (fun acc declParser ->
                    acc >>= fun accDecls ->
                    declParser >>= fun newDecls ->
                    succeed (accDecls @ newDecls)
                ) (succeed [])
            
            match transformAllDeclarations module'.Declarations initialState with
            | Reply(Ok transformedDeclarations, finalState) ->
                let transformedModule = { module' with Declarations = transformedDeclarations }
                Reply(Ok transformedModule, finalState)
            | Reply(Error, error) ->
                Reply(Error, error)
        |> withErrorContext (sprintf "module transformation '%s'" module'.Name)

/// Closure elimination validation
module ClosureValidation =
    
    /// Validates that no closures remain in an expression
    let rec validateNoClosures (expr: OakExpression) : CompilerResult<unit> =
        match expr with
        | Lambda(_, _) ->
            CompilerFailure [TransformError("closure validation", "lambda expression", "eliminated closure", "Lambda expression found after closure elimination")]
        
        | Application(func, args) ->
            validateNoClosures func >>= fun _ ->
            args 
            |> List.map validateNoClosures
            |> List.fold (fun acc result ->
                match acc, result with
                | Success (), Success () -> Success ()
                | CompilerFailure errors, Success () -> CompilerFailure errors
                | Success (), CompilerFailure errors -> CompilerFailure errors
                | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
            ) (Success ())
        
        | Let(_, value, body) ->
            validateNoClosures value >>= fun _ ->
            validateNoClosures body
        
        | IfThenElse(cond, thenExpr, elseExpr) ->
            validateNoClosures cond >>= fun _ ->
            validateNoClosures thenExpr >>= fun _ ->
            validateNoClosures elseExpr
        
        | Sequential(first, second) ->
            validateNoClosures first >>= fun _ ->
            validateNoClosures second
        
        | FieldAccess(target, _) ->
            validateNoClosures target
        
        | MethodCall(target, _, args) ->
            validateNoClosures target >>= fun _ ->
            args 
            |> List.map validateNoClosures
            |> List.fold (fun acc result ->
                match acc, result with
                | Success (), Success () -> Success ()
                | CompilerFailure errors, Success () -> CompilerFailure errors
                | Success (), CompilerFailure errors -> CompilerFailure errors
                | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
            ) (Success ())
        
        | Variable _ | Literal _ ->
            Success ()
    
    /// Validates that a declaration contains no closures
    let validateDeclarationNoClosures (decl: OakDeclaration) : CompilerResult<unit> =
        match decl with
        | FunctionDecl(_, _, _, body) -> validateNoClosures body
        | EntryPoint(expr) -> validateNoClosures expr
        | TypeDecl(_, _) -> Success ()
    
    /// Validates that a program contains no closures
    let validateProgramNoClosures (program: OakProgram) : CompilerResult<unit> =
        program.Modules
        |> List.collect (fun m -> m.Declarations)
        |> List.map validateDeclarationNoClosures
        |> List.fold (fun acc result ->
            match acc, result with
            | Success (), Success () -> Success ()
            | CompilerFailure errors, Success () -> CompilerFailure errors
            | Success (), CompilerFailure errors -> CompilerFailure errors
            | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
        ) (Success ())

/// Main closure elimination entry point - NO FALLBACKS ALLOWED
let eliminateClosures (program: OakProgram) : CompilerResult<OakProgram> =
    if program.Modules.IsEmpty then
        CompilerFailure [TransformError("closure elimination", "empty program", "transformed program", "Program must contain at least one module")]
    else
        let initialState = {
            GlobalScope = Set.empty
            CurrentScope = Set.empty
            ScopeStack = []
            CapturedVariables = Map.empty
            LiftedFunctions = []
            TransformationMappings = Map.empty
            ErrorContext = []
        }
        
        // Transform all modules
        let transformAllModules (modules: OakModule list) : CompilerResult<OakModule list> =
            modules
            |> List.map (fun module' ->
                match transformModule module' initialState with
                | Reply(Ok transformedModule, _) -> Success transformedModule
                | Reply(Error, error) -> CompilerFailure [TransformError("module transformation", module'.Name, "transformed module", error)])
            |> List.fold (fun acc result ->
                match acc, result with
                | Success modules, Success module' -> Success (module' :: modules)
                | CompilerFailure errors, Success _ -> CompilerFailure errors
                | Success _, CompilerFailure errors -> CompilerFailure errors
                | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
            ) (Success [])
            |>> List.rev
        
        transformAllModules program.Modules >>= fun transformedModules ->
        let transformedProgram = { program with Modules = transformedModules }
        
        // Validate that no closures remain
        ClosureValidation.validateProgramNoClosures transformedProgram >>= fun _ ->
        
        Success transformedProgram

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
        | Variable _ | Literal _ -> 0
    
    let countClosuresInDeclaration (decl: OakDeclaration) : (string * int) =
        match decl with
        | FunctionDecl(name, _, _, body) -> (name, countClosures body)
        | EntryPoint(_) -> ("__entry__", 0)  // Entry points shouldn't have closures after parsing
        | TypeDecl(name, _) -> (name, 0)
    
    let closureCounts = 
        program.Modules
        |> List.collect (fun m -> m.Declarations)
        |> List.map countClosuresInDeclaration
        |> Map.ofList
    
    Success closureCounts