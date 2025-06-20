module Dabbit.TreeShaking.ReachabilityTraversal

open Dabbit.TreeShaking.ReachabilityAnalyzer
open Dabbit.TreeShaking.DependencyGraphBuilder

/// Performs reachability analysis using worklist algorithm
let analyzeReachability (graph: DependencyGraph) : ReachabilityResult =
    let mutable reachable = Set.empty<string>
    let mutable worklist = graph.EntryPoints |> Set.toList
    let mutable reachableUnionCases = Map.empty<string, Set<string>>
    let mutable reachableFields = Map.empty<string, Set<string>>
    
    // Process worklist until empty
    while not worklist.IsEmpty do
        match worklist with
        | current :: rest ->
            worklist <- rest
            
            if not (Set.contains current reachable) then
                // Only add to reachable if it's an actual declaration
                if Map.containsKey current graph.Declarations then
                    reachable <- Set.add current reachable
                
                // Get dependencies of current item
                match Map.tryFind current graph.Dependencies with
                | Some deps ->
                    for dep in deps do
                        match dep with
                        | FunctionCall(_, callee) ->
                            // Improved qualified name resolution
                            let qualifiedCallee = 
                                if callee.Contains(".") then
                                    // Already a qualified name, see if it exists in declarations
                                    if Map.containsKey callee graph.Declarations then
                                        callee
                                    else
                                        // Try to find by end part
                                        let parts = callee.Split('.')
                                        let funcName = parts.[parts.Length - 1]
                                        match Map.tryFind funcName graph.QualifiedNames with
                                        | Some qname -> qname
                                        | None -> callee
                                else
                                    // Try to find in QualifiedNames map
                                    match Map.tryFind callee graph.QualifiedNames with
                                    | Some qname -> qname
                                    | None -> 
                                        // Try to find by suffix match
                                        graph.Declarations 
                                        |> Map.tryFindKey (fun k _ -> k.EndsWith("." + callee))
                                        |> Option.defaultValue callee
                            
                            if not (Set.contains qualifiedCallee reachable) && not (List.contains qualifiedCallee worklist) then
                                worklist <- qualifiedCallee :: worklist
                        
                        | ModuleReference(_, moduleName) ->
                            // Find all declarations in this module and add to worklist
                            let moduleDeclarations =
                                graph.Declarations
                                |> Map.toSeq
                                |> Seq.choose (fun (name, _) -> 
                                    if name.StartsWith(moduleName + ".") then Some name else None)
                                |> Seq.toList
                            
                            for decl in moduleDeclarations do
                                if not (Set.contains decl reachable) && not (List.contains decl worklist) then
                                    worklist <- decl :: worklist
                        
                        // [Rest of the match cases remain the same]
                | None -> ()
        | [] -> ()
    
    // Calculate statistics - only count actual declarations
    let totalDecls = graph.Declarations.Count
    let reachableCount = reachable.Count
    let eliminatedCount = totalDecls - reachableCount
    
    // Group by module for breakdown
    let moduleStats = 
        graph.Declarations
        |> Map.toList
        |> List.groupBy (fun (name, _) -> name.Split('.') |> Array.head)
        |> List.map (fun (moduleName, decls) ->
            let moduleTotal = decls.Length
            let moduleReachable = decls |> List.filter (fun (name, _) -> Set.contains name reachable) |> List.length
            let moduleEliminated = moduleTotal - moduleReachable
            
            (moduleName, {
                ModuleName = moduleName
                TotalFunctions = moduleTotal
                RetainedFunctions = moduleReachable
                EliminatedFunctions = moduleEliminated
            })
        )
        |> Map.ofList
    
    {
        ReachableDeclarations = reachable
        ReachableUnionCases = reachableUnionCases
        ReachableFields = reachableFields
        EliminationStats = {
            TotalDeclarations = totalDecls
            ReachableDeclarations = reachableCount
            EliminatedDeclarations = eliminatedCount
            ModuleBreakdown = moduleStats
            LargestEliminated = [] // TODO: Calculate based on AST node count
        }
    }