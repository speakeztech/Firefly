module Dabbit.Analysis.ReachabilityAnalyzer

open FSharp.Compiler.Symbols
open FSharp.Compiler.Syntax

/// Reachability analysis result
type ReachabilityResult = {
    Reachable: Set<string>
    UnionCases: Map<string, Set<string>>  // Type -> used cases
    Statistics: ReachabilityStats
}

and ReachabilityStats = {
    TotalSymbols: int
    ReachableSymbols: int
    EliminatedSymbols: int
    ModuleBreakdown: Map<string, ModuleStats>
}

and ModuleStats = {
    Module: string
    Total: int
    Retained: int
    Eliminated: int
}

/// Build reachability worklist from entry points
let analyze (symbols: Map<string, FSharpSymbol>) (deps: Map<string, Set<string>>) (entries: Set<string>) =
    let rec reach visited queue =
        match queue with
        | [] -> visited
        | sym :: rest when Set.contains sym visited -> reach visited rest
        | sym :: rest ->
            let visited' = Set.add sym visited
            let neighbors = Map.tryFind sym deps |> Option.defaultValue Set.empty
            reach visited' (Set.toList neighbors @ rest)
    
    let reachable = reach Set.empty (Set.toList entries)
    
    // Calculate statistics
    let moduleStats = 
        symbols 
        |> Map.toList
        |> List.groupBy (fun (name, _) -> name.Split('.').[0])
        |> List.map (fun (modName, syms) ->
            let total = syms.Length
            let retained = syms |> List.filter (fun (n, _) -> Set.contains n reachable) |> List.length
            (modName, { Module = modName; Total = total; Retained = retained; Eliminated = total - retained }))
        |> Map.ofList
    
    { Reachable = reachable
      UnionCases = Map.empty  // TODO: Track union case usage
      Statistics = {
          TotalSymbols = Map.count symbols
          ReachableSymbols = Set.count reachable
          EliminatedSymbols = Map.count symbols - Set.count reachable
          ModuleBreakdown = moduleStats
      }}