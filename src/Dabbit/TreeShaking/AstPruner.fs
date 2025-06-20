module Dabbit.TreeShaking.AstPruner

open Dabbit.Parsing.OakAst
open Dabbit.TreeShaking.ReachabilityAnalyzer

/// Prunes unreachable declarations from Oak AST
let pruneUnreachableCode (program: OakProgram) (result: ReachabilityResult) : OakProgram =
    let pruneModule (oakModule: OakModule) : OakModule option =
        let prunedDeclarations = 
            oakModule.Declarations
            |> List.filter (fun decl ->
                match decl with
                | FunctionDecl(name, _, _, _) ->
                    let qualifiedName = sprintf "%s.%s" oakModule.Name name
                    Set.contains qualifiedName result.ReachableDeclarations
                
                | EntryPoint _ -> true // Always keep entry points
                
                | TypeDecl(name, oakType) ->
                    let qualifiedName = sprintf "%s.%s" oakModule.Name name
                    Set.contains qualifiedName result.ReachableDeclarations
                
                | ExternalDecl(name, _, _, _) ->
                    let qualifiedName = sprintf "%s.%s" oakModule.Name name
                    Set.contains qualifiedName result.ReachableDeclarations
            )
        
        if prunedDeclarations.IsEmpty then
            None
        else
            Some { oakModule with Declarations = prunedDeclarations }
    
    let prunedModules = 
        program.Modules 
        |> List.choose pruneModule
    
    { Modules = prunedModules }

/// Generates diagnostic report
let generateDiagnosticReport (stats: EliminationStatistics) : string =
    let sb = System.Text.StringBuilder()
    
    sb.AppendLine("=== Firefly Tree Shaking Analysis ===") |> ignore
    sb.AppendLine() |> ignore
    
    sb.AppendLine("Reachability Summary:") |> ignore
    sb.AppendLine(sprintf "- Total declarations: %d" stats.TotalDeclarations) |> ignore
    sb.AppendLine(sprintf "- Reachable declarations: %d (%.1f%%)" 
        stats.ReachableDeclarations 
        (float stats.ReachableDeclarations / float stats.TotalDeclarations * 100.0)) |> ignore
    sb.AppendLine(sprintf "- Eliminated: %d (%.1f%%)" 
        stats.EliminatedDeclarations 
        (float stats.EliminatedDeclarations / float stats.TotalDeclarations * 100.0)) |> ignore
    sb.AppendLine() |> ignore
    
    sb.AppendLine("Module Breakdown:") |> ignore
    for KeyValue(moduleName, moduleStats) in stats.ModuleBreakdown do
        if moduleStats.TotalFunctions > 0 then
            let eliminationPercent = float moduleStats.EliminatedFunctions / float moduleStats.TotalFunctions * 100.0
            sb.AppendLine(sprintf "  %s: %d of %d functions retained (%.1f%% eliminated)" 
                moduleName 
                moduleStats.RetainedFunctions 
                moduleStats.TotalFunctions 
                eliminationPercent) |> ignore
    
    sb.ToString()