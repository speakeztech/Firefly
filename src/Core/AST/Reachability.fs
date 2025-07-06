module Core.AST.Reachability

/// Compute reachability using simple graph traversal - NO DUPLICATES
let computeReachable (dependencies: Dependency[]) (entryPoints: string[]) : Set<string> =
    let depMap = 
        dependencies 
        |> Array.groupBy (_.From)
        |> Map.ofArray
    
    let rec traverse visited current =
        if Set.contains current visited then visited
        else
            let newVisited = Set.add current visited
            match Map.tryFind current depMap with
            | Some deps -> 
                deps |> Array.fold (fun acc dep -> traverse acc dep.To) newVisited
            | None -> newVisited
    
    entryPoints |> Array.fold traverse Set.empty