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
        | Some currentDecl -> 
            // Check if this is a qualified name with a module prefix
            if name.Contains(".") then
                // Handle module.function references
                let parts = name.Split('.')
                if parts.Length >= 2 then
                    let moduleName = parts |> Array.take (parts.Length - 1) |> Array.toList |> String.concat "."
                    let funcName = parts.[parts.Length - 1]
                    Set.singleton (FunctionCall(currentDecl, name))
                    |> Set.add (ModuleReference(currentDecl, moduleName))
                else
                    Set.singleton (FunctionCall(currentDecl, name))
            else
                // Try to resolve via QualifiedNames map or direct call
                Set.singleton (FunctionCall(currentDecl, name))
        | None -> Set.empty

    | Match(matchExpr, cases) ->
        let exprDeps = analyzeDependencies matchExpr state
        let casesDeps = 
            cases 
            |> List.map (fun (pattern, caseExpr) -> analyzeDependencies caseExpr state)
            |> Set.unionMany
        Set.union exprDeps casesDeps
    
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

/// Helper function to register all declarations first
/// Note: This MUST be defined BEFORE buildDependencyGraph since it's used by it
let registerDeclarations (graph: DependencyGraph) (oakModule: OakModule) =
    oakModule.Declarations
    |> List.fold (fun g decl ->
        match decl with
        | FunctionDecl(name, _, _, _) | ExternalDecl(name, _, _, _) | TypeDecl(name, _) -> 
            let qualifiedName = sprintf "%s.%s" oakModule.Name name
            { g with 
                Declarations = Map.add qualifiedName decl g.Declarations
                QualifiedNames = Map.add name qualifiedName g.QualifiedNames
            }
        | EntryPoint(_) -> 
            let qualifiedName = sprintf "%s.main" oakModule.Name
            { g with 
                Declarations = Map.add qualifiedName decl g.Declarations
                EntryPoints = Set.add qualifiedName g.EntryPoints 
            }
    ) graph

/// Builds dependency graph for entire program
let buildDependencyGraph (program: OakProgram) : DependencyGraph =
    let initialGraph = {
        Dependencies = Map.empty
        Declarations = Map.empty
        EntryPoints = Set.empty
        QualifiedNames = Map.empty
    }

    // Add Alloy dependencies (expanded version with more comprehensive mappings)
    let addAlloyDependencies (graph: DependencyGraph) : DependencyGraph =
        // More comprehensive list of Alloy functions that might be called
        let alloyFunctions = [
            // Memory module
            "stackBuffer", "Alloy.Memory"
            "spanToString", "Alloy.Memory"
            "NativePtr.stackalloc", "Alloy.Memory"
            "Span", "Alloy.Memory"
            "Span.create", "Alloy.Memory"
            "INativeBuffer", "Alloy.Memory"
            "StackBuffer", "Alloy.Memory"
            "StackBuffer.AsSpan", "Alloy.Memory"
            
            // IO module
            "prompt", "Alloy.IO.Console"
            "readInto", "Alloy.IO.Console"
            "readLine", "Alloy.IO.Console"
            "writeLine", "Alloy.IO.Console"
            "format", "Alloy.IO.String"
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
    
    // Improve module processing to better handle qualified names
    let processModule (graph: DependencyGraph) (oakModule: OakModule) =
        let initialState = {
            CurrentDeclaration = None
            CurrentModule = oakModule.Name
            Graph = graph
        }
        
        // Phase 1: First pass to register all declarations for proper qualified name resolution
        let preregisteredGraph = registerDeclarations graph oakModule
        
        // Phase 2: Analyze dependencies after all declarations are registered
        let finalState = 
            oakModule.Declarations
            |> List.fold (fun state decl -> analyzeDeclaration decl { state with Graph = preregisteredGraph }) initialState
        
        finalState.Graph
    
    // First register all modules to build a complete declaration map
    let graphWithDeclarations = 
        program.Modules
        |> List.fold (fun g m -> registerDeclarations g m) initialGraph
    
    // Then process dependencies
    let graphWithDependencies =
        program.Modules
        |> List.fold processModule graphWithDeclarations
    
    // Add Alloy library dependencies
    addAlloyDependencies graphWithDependencies