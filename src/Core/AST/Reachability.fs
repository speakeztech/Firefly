module Core.AST.Reachability

open System
open Core.XParsec.Foundation    // Import shared types  
open Core.AST.Extraction        // Import TypedFunction from here

// ===================================================================
// Reachability Computation Core
// ===================================================================

/// Compute reachable functions using depth-first traversal
/// This is the primary algorithm for dead code elimination
let computeReachable (dependencies: Dependency[]) (entryPoints: string[]) : Set<string> =
    // Build adjacency map for efficient graph traversal
    let depMap = 
        dependencies 
        |> Array.groupBy (fun dep -> dep.From)
        |> Map.ofArray
        |> Map.map (fun _ deps -> deps |> Array.map (fun dep -> dep.To))
    
    let rec traverse (visited: Set<string>) (current: string) : Set<string> =
        if Set.contains current visited then 
            visited
        else
            let newVisited = Set.add current visited
            match Map.tryFind current depMap with
            | Some callees -> 
                callees |> Array.fold traverse newVisited
            | None -> 
                newVisited
    
    // Start traversal from all entry points
    entryPoints 
    |> Array.fold traverse Set.empty

/// Compute reachable functions with detailed analysis
let computeReachabilityAnalysis (dependencies: Dependency[]) (allFunctions: TypedFunction[]) (entryPoints: string[]) : Map<string, string list> * Set<string> * Set<string> =
    let allFunctionNames = allFunctions |> Array.map (fun f -> f.FullName) |> Set.ofArray
    let reachableFunctions = computeReachable dependencies entryPoints
    let unreachableFunctions = Set.difference allFunctionNames reachableFunctions
    
    // Build call graph for analysis
    let callGraph = 
        dependencies
        |> Array.groupBy (fun dep -> dep.From)
        |> Map.ofArray
        |> Map.map (fun _ deps -> deps |> Array.map (fun dep -> dep.To) |> List.ofArray)
    
    (callGraph, reachableFunctions, unreachableFunctions)

// ===================================================================
// Reachability Analysis Results
// ===================================================================

/// Result of reachability analysis with detailed metrics
type ReachabilityResult = {
    /// Total number of functions analyzed
    TotalFunctions: int
    /// Functions reachable from entry points
    ReachableFunctions: Set<string>
    /// Functions that can be eliminated
    UnreachableFunctions: Set<string>
    /// Call graph representation
    CallGraph: Map<string, string list>
    /// Entry points used for analysis
    EntryPoints: string[]
}

/// Perform complete reachability analysis
let analyzeReachability (functions: TypedFunction[]) (dependencies: Dependency[]) : ReachabilityResult =
    // Identify entry points
    let entryPoints = 
        functions
        |> Array.filter (fun f -> f.IsEntryPoint)
        |> Array.map (fun f -> f.FullName)
    
    // If no explicit entry points, treat all public functions as entry points
    let effectiveEntryPoints = 
        if Array.isEmpty entryPoints then
            functions
            |> Array.filter (fun f -> not (f.FullName.Contains("+"))) // Filter out nested functions
            |> Array.map (fun f -> f.FullName)
        else
            entryPoints
    
    let (callGraph, reachable, unreachable) = 
        computeReachabilityAnalysis dependencies functions effectiveEntryPoints
    
    {
        TotalFunctions = functions.Length
        ReachableFunctions = reachable
        UnreachableFunctions = unreachable
        CallGraph = callGraph
        EntryPoints = effectiveEntryPoints
    }

// ===================================================================
// Utility Functions for Analysis
// ===================================================================

/// Get functions that directly call the specified function
let getCallers (targetFunction: string) (dependencies: Dependency[]) : string[] =
    dependencies
    |> Array.filter (fun dep -> dep.To = targetFunction)
    |> Array.map (fun dep -> dep.From)
    |> Array.distinct

/// Get functions directly called by the specified function
let getCallees (sourceFunction: string) (dependencies: Dependency[]) : string[] =
    dependencies
    |> Array.filter (fun dep -> dep.From = sourceFunction)
    |> Array.map (fun dep -> dep.To)
    |> Array.distinct

/// Compute strongly connected components for cycle detection
let findCycles (dependencies: Dependency[]) : string[][] =
    // Simple cycle detection using Tarjan's algorithm
    let adjMap = 
        dependencies
        |> Array.groupBy (fun dep -> dep.From)
        |> Map.ofArray
        |> Map.map (fun _ deps -> deps |> Array.map (fun dep -> dep.To) |> Set.ofArray)
    
    let visited = System.Collections.Generic.HashSet<string>()
    let stack = System.Collections.Generic.HashSet<string>()
    let cycles = ResizeArray<string[]>()
    
    let rec dfs (node: string) (path: string list) =
        if stack.Contains(node) then
            // Found a cycle
            let cycleStart = path |> List.findIndex ((=) node)
            let cycle = path |> List.skip cycleStart |> List.toArray
            cycles.Add(cycle)
        elif not (visited.Contains(node)) then
            visited.Add(node) |> ignore
            stack.Add(node) |> ignore
            
            match Map.tryFind node adjMap with
            | Some neighbors ->
                for neighbor in neighbors do
                    dfs neighbor (neighbor :: path)
            | None -> ()
            
            stack.Remove(node) |> ignore
    
    // Start DFS from all nodes
    let allNodes = 
        dependencies 
        |> Array.collect (fun dep -> [| dep.From; dep.To |])
        |> Array.distinct
    
    for node in allNodes do
        if not (visited.Contains(node)) then
            dfs node [node]
    
    cycles |> Array.ofSeq

/// Generate reachability report for debugging
let generateReachabilityReport (result: ReachabilityResult) : string =
    let sb = System.Text.StringBuilder()
    
    sb.AppendLine("=== FIREFLY REACHABILITY ANALYSIS ===") |> ignore
    sb.AppendLine($"Total Functions: {result.TotalFunctions}") |> ignore
    sb.AppendLine($"Reachable: {result.ReachableFunctions.Count}") |> ignore
    sb.AppendLine($"Unreachable: {result.UnreachableFunctions.Count}") |> ignore
    sb.AppendLine($"Elimination Rate: {(float result.UnreachableFunctions.Count / float result.TotalFunctions) * 100.0:F1}%%") |> ignore
    sb.AppendLine() |> ignore
    
    sb.AppendLine("Entry Points:") |> ignore
    for ep in result.EntryPoints do
        sb.AppendLine($"  • {ep}") |> ignore
    sb.AppendLine() |> ignore
    
    if not (Set.isEmpty result.UnreachableFunctions) then
        sb.AppendLine("Eliminated Functions:") |> ignore
        for func in result.UnreachableFunctions |> Set.toArray |> Array.sort do
            sb.AppendLine($"  × {func}") |> ignore
    
    sb.ToString()