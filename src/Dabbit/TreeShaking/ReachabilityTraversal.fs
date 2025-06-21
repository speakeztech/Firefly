module Dabbit.TreeShaking.ReachabilityTraversal

open Dabbit.TreeShaking.ReachabilityAnalyzer
open Dabbit.TreeShaking.DependencyGraphBuilder

/// Determines if a module name is from the Alloy library
let isAlloyModuleName (name: string) : bool =
    name.StartsWith("Alloy.")

/// Performs reachability analysis using worklist algorithm
let analyzeReachability (graph: DependencyGraph) : ReachabilityResult =
    printfn "Starting reachability analysis..."
    
    let mutable reachable = Set.empty<string>
    let mutable worklist = graph.EntryPoints |> Set.toList
    let mutable reachableUnionCases = Map.empty<string, Set<string>>
    let mutable reachableFields = Map.empty<string, Set<string>>
    
    // First add all entry points to the worklist
    printfn "Starting with %d entry points" worklist.Length
    
    // Make sure we have at least one entry point
    if worklist.IsEmpty && not graph.Declarations.IsEmpty then
        printfn "WARNING: No entry points found, using first declaration as seed"
        let firstDecl = graph.Declarations |> Map.toSeq |> Seq.head |> fst
        worklist <- [firstDecl]
    
    // Special handling: Add all Alloy module declarations to reachable set directly
    let alloyReachable =
        graph.Declarations
        |> Map.toSeq
        |> Seq.choose (fun (name, _) ->
            if isAlloyModuleName (name.Split('.').[0]) then
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
                if Map.containsKey current graph.Declarations then
                    printfn "  Adding to reachable set: %s" current
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
                                printfn "  Adding to worklist: %s (via function call from %s)" 
                                    qualifiedCallee current
                                worklist <- qualifiedCallee :: worklist
                        
                        | ModuleReference(_, moduleName) ->
                            // Find all declarations in this module and add to worklist
                            let moduleDeclarations =
                                graph.Declarations
                                |> Map.toSeq
                                |> Seq.choose (fun (name, _) -> 
                                    if name.StartsWith(moduleName + ".") then 
                                        printfn "  Found module declaration: %s (from module %s)" 
                                            name moduleName
                                        Some name 
                                    else None)
                                |> Seq.toList
                            
                            printfn "  Found %d declarations in module %s" 
                                moduleDeclarations.Length moduleName
                                
                            for decl in moduleDeclarations do
                                if not (Set.contains decl reachable) && not (List.contains decl worklist) then
                                    printfn "  Adding to worklist: %s (via module reference to %s)" 
                                        decl moduleName
                                    worklist <- decl :: worklist
                        
                        | FieldAccess(_, typeName, fieldName) ->
                            // Track field accesses for union cases and records
                            let currentFields = 
                                match Map.tryFind typeName reachableFields with
                                | Some fields -> fields
                                | None -> Set.empty
                            
                            let updatedFields = Set.add fieldName currentFields
                            reachableFields <- Map.add typeName updatedFields reachableFields
                            
                        | UnionCaseUsage(_, typeName, caseName) ->
                            // Track union case usage
                            let currentCases = 
                                match Map.tryFind typeName reachableUnionCases with
                                | Some cases -> cases
                                | None -> Set.empty
                            
                            let updatedCases = Set.add caseName currentCases
                            reachableUnionCases <- Map.add typeName updatedCases reachableUnionCases
                | None -> ()
        | [] -> ()
    
    printfn "Reachability analysis completed after %d iterations" iterations
    
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
            stats.RetainedFunctions 
            stats.TotalFunctions 
            (if stats.TotalFunctions > 0 then
                float stats.EliminatedFunctions / float stats.TotalFunctions * 100.0
             else 0.0)
    
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