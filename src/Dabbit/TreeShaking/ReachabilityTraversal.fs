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
                            // Resolve qualified name
                            let qualifiedCallee = 
                                match Map.tryFind callee graph.QualifiedNames with
                                | Some qname -> qname
                                | None -> 
                                    // Check if it's already qualified
                                    if callee.Contains(".") then callee
                                    else
                                        // Try to find in any module
                                        graph.Declarations 
                                        |> Map.tryFindKey (fun k _ -> k.EndsWith("." + callee))
                                        |> Option.defaultValue callee
                            
                            if not (Set.contains qualifiedCallee reachable) && not (List.contains qualifiedCallee worklist) then
                                worklist <- qualifiedCallee :: worklist
                        
                        | TypeUsage(_, typeName) ->
                            if not (Set.contains typeName reachable) && not (List.contains typeName worklist) then
                                worklist <- typeName :: worklist
                        
                        | UnionCaseUsage(_, typeName, caseName) ->
                            let cases = 
                                match Map.tryFind typeName reachableUnionCases with
                                | Some existing -> Set.add caseName existing
                                | None -> Set.singleton caseName
                            reachableUnionCases <- Map.add typeName cases reachableUnionCases
                            
                            if not (Set.contains typeName reachable) && not (List.contains typeName worklist) then
                                worklist <- typeName :: worklist
                        
                        | FieldAccess(_, typeName, fieldName) ->
                            let fields = 
                                match Map.tryFind typeName reachableFields with
                                | Some existing -> Set.add fieldName existing
                                | None -> Set.singleton fieldName
                            reachableFields <- Map.add typeName fields reachableFields
                            
                            if not (Set.contains typeName reachable) && not (List.contains typeName worklist) then
                                worklist <- typeName :: worklist
                        
                        | ModuleReference(_, moduleName) ->
                            ()
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