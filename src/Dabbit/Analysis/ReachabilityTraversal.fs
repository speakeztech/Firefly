module Dabbit.Analysis.ReachabilityTraversal

open Dabbit.Analysis.ReachabilityAnalyzer
open Dabbit.Analysis.DependencyGraphBuilder

/// Determines if a module name is from the Alloy library
let isAlloyModuleName (name: string) : bool =
    name.StartsWith("Alloy.")

/// Performs reachability analysis using worklist algorithm
let analyzeReachability (graph: DependencyGraph) : ReachabilityResult =
    printfn "Starting reachability analysis..."
    
    let mutable reachable = Set.empty<string>
    let mutable worklist = graph.Roots |> Set.toList
    let mutable reachableUnionCases = Map.empty<string, Set<string>>
    let mutable reachableFields = Map.empty<string, Set<string>>
    
    // First add all entry points to the worklist
    printfn "Starting with %d entry points" worklist.Length
    
    // Make sure we have at least one entry point
    if worklist.IsEmpty && not (Map.isEmpty graph.Nodes) then
        printfn "WARNING: No entry points found, using first declaration as seed"
        let firstDecl = graph.Nodes |> Map.toSeq |> Seq.head |> fst
        worklist <- [firstDecl]
    
    // Special handling: Add all Alloy module declarations to reachable set directly
    let alloyReachable =
        graph.Nodes
        |> Map.toSeq
        |> Seq.choose (fun (name, _) ->
            let parts = name.Split('.')
            if parts.Length > 0 && isAlloyModuleName parts.[0] then
                printfn "  Adding Alloy declaration as always reachable: %s" name
                Some name
            else
                None)
        |> Set.ofSeq
    
    reachable <- Set.union reachable alloyReachable
    
    // Process worklist until empty
    let mutable iterations = 0
    while not worklist.IsEmpty do
        iterations <- iterations + 1
        
        if iterations % 100 = 0 then
            printfn "  Reachability iteration %d, worklist size: %d, reachable set: %d" 
                iterations worklist.Length reachable.Count
            
        match worklist with
        | current :: rest ->
            worklist <- rest
            
            if not (Set.contains current reachable) then
                // Only add to reachable if it's an actual declaration
                if Map.containsKey current graph.Nodes then
                    printfn "  Adding to reachable set: %s" current
                    reachable <- Set.add current reachable
                
                // Get dependencies of current item
                match Map.tryFind current graph.Edges with
                | Some deps ->
                    for dep in deps do
                        // CRITICAL FIX: Add ALL possible qualified versions
                        let possibleNames = [
                            dep
                            // Try common Alloy namespaces
                            "Alloy.Memory." + dep
                            "Alloy.IO." + dep
                            "Alloy.IO.Console." + dep
                            "Alloy.IO.String." + dep
                            // Try to find by suffix match in declarations
                            yield! graph.Nodes 
                                   |> Map.toSeq 
                                   |> Seq.choose (fun (k, _) -> 
                                       if k.EndsWith("." + dep) || k = dep then Some k 
                                       else None)
                        ]
                        
                        for name in possibleNames do
                            if Map.containsKey name graph.Nodes && 
                               not (Set.contains name reachable) && 
                               not (List.contains name worklist) then
                                printfn "  Adding to worklist: %s (via dependency %s from %s)" 
                                    name dep current
                                worklist <- name :: worklist
                | None -> ()
        | [] -> ()
    
    printfn "Reachability analysis completed after %d iterations" iterations
    
    // Calculate statistics - only count actual declarations
    let totalDecls = Map.count graph.Nodes
    let reachableCount = Set.count reachable
    let eliminatedCount = totalDecls - reachableCount
    
    // Group by module for breakdown
    let moduleStats = 
        graph.Nodes
        |> Map.toList
        |> List.groupBy (fun (name, _) -> 
            let parts = name.Split('.')
            if parts.Length > 0 then parts.[0] else "Global")
        |> List.map (fun (moduleName, decls) ->
            let moduleTotal = decls.Length
            let moduleReachable = decls |> List.filter (fun (name, _) -> Set.contains name reachable) |> List.length
            let moduleEliminated = moduleTotal - moduleReachable
            
            (moduleName, {
                Module = moduleName
                Total = moduleTotal
                Retained = moduleReachable
                Eliminated = moduleEliminated
            })
        )
        |> Map.ofList
    
    printfn "Reachability statistics:"
    printfn "  Total declarations: %d" totalDecls
    printfn "  Reachable declarations: %d (%.1f%%)" 
        reachableCount 
        (if totalDecls > 0 then float reachableCount / float totalDecls * 100.0 else 0.0)
    printfn "  Eliminated declarations: %d" eliminatedCount
    
    // Print module statistics
    printfn "Module breakdown:"
    for KeyValue(moduleName, stats) in moduleStats do
        printfn "  %s: %d of %d declarations reachable (%.1f%% eliminated)" 
            moduleName 
            stats.Retained 
            stats.Total 
            (if stats.Total > 0 then
                float stats.Eliminated / float stats.Total * 100.0
             else 0.0)
    
    {
        Reachable = reachable
        UnionCases = reachableUnionCases
        Statistics = {
            TotalSymbols = totalDecls
            ReachableSymbols = reachableCount
            EliminatedSymbols = eliminatedCount
            ModuleBreakdown = moduleStats
        }
    }