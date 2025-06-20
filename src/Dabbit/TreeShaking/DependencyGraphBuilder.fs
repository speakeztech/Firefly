module Dabbit.TreeShaking.DependencyGraphBuilder

open Dabbit.Parsing.OakAst
open Dabbit.TreeShaking.ReachabilityAnalyzer

/// State for building the dependency graph
type GraphBuilderState = {
    CurrentDeclaration: string option
    CurrentModule: string
    Graph: DependencyGraph
}

/// Analyzes an expression to extract dependencies
let rec analyzeDependencies (expr: OakExpression) (state: GraphBuilderState) : Set<Dependency> =
    match expr with
    | Variable name ->
        match state.CurrentDeclaration with
        | Some currentDecl -> Set.singleton (FunctionCall(currentDecl, name))
        | None -> Set.empty
    
    | Application(func, args) ->
        let funcDeps = analyzeDependencies func state
        let argDeps = args |> List.map (fun arg -> analyzeDependencies arg state) |> Set.unionMany
        Set.union funcDeps argDeps
    
    | Let(name, value, body) ->
        let valueDeps = analyzeDependencies value state
        let bodyDeps = analyzeDependencies body state
        Set.union valueDeps bodyDeps
    
    | IfThenElse(cond, thenExpr, elseExpr) ->
        [cond; thenExpr; elseExpr]
        |> List.map (fun e -> analyzeDependencies e state)
        |> Set.unionMany
    
    | Sequential(first, second) ->
        Set.union (analyzeDependencies first state) (analyzeDependencies second state)
    
    | Lambda(params', body) ->
        analyzeDependencies body state
    
    | OakExpression.FieldAccess(target, fieldName) ->
        let targetDeps = analyzeDependencies target state
        match state.CurrentDeclaration with
        | Some currentDecl ->
            let fieldDep = ReachabilityAnalyzer.FieldAccess(currentDecl, "UnknownType", fieldName)
            Set.add fieldDep targetDeps
        | None -> targetDeps
    
    | MethodCall(target, methodName, args) ->
        let targetDeps = analyzeDependencies target state
        let argDeps = args |> List.map (fun arg -> analyzeDependencies arg state) |> Set.unionMany
        match state.CurrentDeclaration with
        | Some currentDecl ->
            let methodDep = FunctionCall(currentDecl, methodName)
            targetDeps |> Set.add methodDep |> Set.union argDeps
        | None ->
            Set.union targetDeps argDeps
    
    | IOOperation(ioType, args) ->
        let ioDeps = 
            match state.CurrentDeclaration with
            | Some currentDecl ->
                match ioType with
                | Printf _ | Printfn _ -> Set.singleton (FunctionCall(currentDecl, "printf"))
                | ReadLine -> Set.singleton (FunctionCall(currentDecl, "readLine"))
                | Scanf _ -> Set.singleton (FunctionCall(currentDecl, "scanf"))
                | WriteFile _ -> Set.singleton (FunctionCall(currentDecl, "writeFile"))
                | ReadFile _ -> Set.singleton (FunctionCall(currentDecl, "readFile"))
            | None -> Set.empty
        
        let argDeps = args |> List.map (fun arg -> analyzeDependencies arg state) |> Set.unionMany
        Set.union ioDeps argDeps
    
    | Literal _ -> Set.empty

/// Builds the dependency graph for a declaration
let analyzeDeclaration (decl: OakDeclaration) (state: GraphBuilderState) : GraphBuilderState =
    match decl with
    | FunctionDecl(name, params', returnType, body) ->
        let qualifiedName = sprintf "%s.%s" state.CurrentModule name
        let declState = { state with CurrentDeclaration = Some qualifiedName }
        let dependencies = analyzeDependencies body declState
        
        let newGraph = {
            state.Graph with
                Dependencies = Map.add qualifiedName dependencies state.Graph.Dependencies
                Declarations = Map.add qualifiedName decl state.Graph.Declarations
                QualifiedNames = Map.add name qualifiedName state.Graph.QualifiedNames
        }
        { state with Graph = newGraph }
    
    | EntryPoint(expr) ->
        let qualifiedName = sprintf "%s.main" state.CurrentModule
        let declState = { state with CurrentDeclaration = Some qualifiedName }
        let dependencies = analyzeDependencies expr declState
        
        let newGraph = {
            state.Graph with
                Dependencies = Map.add qualifiedName dependencies state.Graph.Dependencies
                Declarations = Map.add qualifiedName (FunctionDecl("main", [], UnitType, expr)) state.Graph.Declarations
                EntryPoints = Set.add qualifiedName state.Graph.EntryPoints
        }
        { state with Graph = newGraph }
    
    | TypeDecl(name, oakType) ->
        let qualifiedName = sprintf "%s.%s" state.CurrentModule name
        let newGraph = {
            state.Graph with
                Declarations = Map.add qualifiedName decl state.Graph.Declarations
        }
        { state with Graph = newGraph }
    
    | ExternalDecl(name, params', returnType, libraryName) ->
        let qualifiedName = sprintf "%s.%s" state.CurrentModule name
        let newGraph = {
            state.Graph with
                Declarations = Map.add qualifiedName decl state.Graph.Declarations
                QualifiedNames = Map.add name qualifiedName state.Graph.QualifiedNames
        }
        { state with Graph = newGraph }

/// Builds dependency graph for entire program
let buildDependencyGraph (program: OakProgram) : DependencyGraph =
    let initialGraph = {
        Dependencies = Map.empty
        Declarations = Map.empty
        EntryPoints = Set.empty
        QualifiedNames = Map.empty
    }

    let addAlloyDependencies (graph: DependencyGraph) : DependencyGraph =
        // Add common Alloy functions that might be called
        let alloyFunctions = [
            "stackBuffer", "Alloy.Memory"
            "spanToString", "Alloy.Memory"
            "prompt", "Alloy.IO.Console"
            "readInto", "Alloy.IO.Console"
            "writeLine", "Alloy.IO.Console"
            "String.format", "Alloy.IO.String"
        ]
    
        let updatedGraph = 
            alloyFunctions
            |> List.fold (fun g (funcName, moduleName) ->
                let qualifiedName = sprintf "%s.%s" moduleName funcName
                { g with 
                    Declarations = Map.add qualifiedName (ExternalDecl(funcName, [], UnitType, "Alloy")) g.Declarations
                    QualifiedNames = Map.add funcName qualifiedName g.QualifiedNames
                }) graph
        
        updatedGraph
    
    let processModule (graph: DependencyGraph) (oakModule: OakModule) =
        let initialState = {
            CurrentDeclaration = None
            CurrentModule = oakModule.Name
            Graph = graph
        }
        
        let finalState = 
            oakModule.Declarations
            |> List.fold (fun state decl -> analyzeDeclaration decl state) initialState
        
        finalState.Graph
    
    program.Modules |> List.fold processModule initialGraph |> addAlloyDependencies