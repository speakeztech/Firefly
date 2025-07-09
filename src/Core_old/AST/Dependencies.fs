module Core.AST.Dependencies

open System
open FSharp.Compiler.Symbols
open FSharp.Compiler.Text
open Core.XParsec.Foundation    // Import shared types
open Core.AST.Extraction        // Import TypedFunction from here

// ===================================================================
// Dependency Classification Logic
// ===================================================================

/// Classify a dependency based on the target function name
let private classifyDependency (targetFullName: string) : DependencyType =
    if targetFullName.StartsWith("Alloy.") || targetFullName.StartsWith("Fidelity.") then 
        AlloyLibraryCall
    elif targetFullName.Contains("..ctor") then 
        ConstructorCall
    elif targetFullName.StartsWith("Microsoft.FSharp.") then 
        ExternalCall
    else 
        DirectCall

// ===================================================================
// Simplified Expression Analysis - Using ImmediateSubExpressions
// ===================================================================

/// Extract basic dependencies from function calls in expression
let private extractBasicDependencies (containingFunction: string) (expr: FSharpExpr) : Dependency list =
    let deps = ResizeArray<Dependency>()
    
    let addDep targetName range =
        if targetName <> containingFunction && not (String.IsNullOrEmpty targetName) then
            deps.Add({
                From = containingFunction
                To = targetName
                CallSite = range
                Type = classifyDependency targetName
            })
    
    try
        // Use a simplified traversal approach using ImmediateSubExpressions
        let rec traverse (expr: FSharpExpr) =
            // Try to extract function calls and value references from the expression
            // This uses the actual FCS API without non-existent properties
            
            // Check if this expression has a type that represents a function call
            if expr.Type.HasTypeDefinition then
                let typeDef = expr.Type.TypeDefinition
                if typeDef.IsFSharpModule || typeDef.IsClass then
                    // This might be a function call, but we need to be more careful
                    // about how we extract the name
                    let possibleName = typeDef.QualifiedName
                    if not (String.IsNullOrEmpty possibleName) then
                        addDep possibleName expr.Range
            
            // Recursively process sub-expressions using the correct API
            for subExpr in expr.ImmediateSubExpressions do
                traverse subExpr
        
        traverse expr
    with
    | ex ->
        // Log warning but continue - don't let expression analysis block compilation
        eprintfn "Warning: Could not fully analyze expression in %s: %s" containingFunction ex.Message
    
    deps |> List.ofSeq

// ===================================================================
// Public API for Dependency Analysis
// ===================================================================

/// Build dependency graph from a collection of typed functions
let buildDependencies (functions: TypedFunction[]) : Dependency[] =
    functions
    |> Array.collect (fun func ->
        try
            extractBasicDependencies func.FullName func.Body |> Array.ofList
        with
        | ex ->
            // Log error but continue processing other functions
            eprintfn "Warning: Failed to extract dependencies from %s: %s" func.FullName ex.Message
            [||])

/// Group dependencies by source function for efficient lookup
let groupDependenciesBySource (dependencies: Dependency[]) : Map<string, Dependency[]> =
    dependencies
    |> Array.groupBy (_.From)
    |> Map.ofArray

/// Get all direct callees of a function
let getDirectCallees (functionName: string) (dependencyMap: Map<string, Dependency[]>) : string[] =
    match Map.tryFind functionName dependencyMap with
    | Some deps -> deps |> Array.map (_.To)
    | None -> [||]

/// Filter dependencies by type
let filterDependenciesByType (depType: DependencyType) (dependencies: Dependency[]) : Dependency[] =
    dependencies |> Array.filter (fun dep -> dep.Type = depType)

/// Get all external dependencies (non-Alloy, non-direct calls)
let getExternalDependencies (dependencies: Dependency[]) : Dependency[] =
    dependencies |> Array.filter (fun dep -> dep.Type = ExternalCall)

/// Get statistics about dependency types
let getDependencyStatistics (dependencies: Dependency[]) : Map<DependencyType, int> =
    dependencies
    |> Array.groupBy (_.Type)
    |> Array.map (fun (depType, deps) -> depType, deps.Length)
    |> Map.ofArray