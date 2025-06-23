module Dabbit.TreeShaking.AstPruner

open Dabbit.Parsing.OakAst
open Dabbit.TreeShaking.ReachabilityAnalyzer

/// Determines if a module is part of the Alloy library
let isAlloyModule (moduleName: string) =
    moduleName.StartsWith("Alloy")

/// Prunes unreachable declarations from Oak AST
let pruneUnreachableCode (program: OakProgram) (result: ReachabilityResult) : OakProgram =
    printfn "Pruning unreachable code..."
    
    let pruneModule (oakModule: OakModule) : OakModule option =
        // Always preserve Alloy modules without pruning
        if isAlloyModule oakModule.Name then
            printfn "  Preserving Alloy module: %s with %d declarations" 
                oakModule.Name oakModule.Declarations.Length
            Some oakModule
        else
            let prunedDeclarations = 
                oakModule.Declarations
                |> List.filter (fun decl ->
                    match decl with
                    | FunctionDecl(name, _, _, _) ->
                        let qualifiedName = sprintf "%s.%s" oakModule.Name name
                        let isReachable = Set.contains qualifiedName result.ReachableDeclarations
                        if isReachable then
                            printfn "  Keeping reachable function: %s.%s" oakModule.Name name
                        else
                            printfn "  Pruning unreachable function: %s.%s" oakModule.Name name
                        isReachable
                    
                    | EntryPoint _ -> 
                        printfn "  Keeping entry point in %s" oakModule.Name
                        true // Always keep entry points
                    
                    | TypeDecl(name, oakType) ->
                        let qualifiedName = sprintf "%s.%s" oakModule.Name name
                        let isReachable = Set.contains qualifiedName result.ReachableDeclarations
                        if isReachable then
                            printfn "  Keeping reachable type: %s.%s" oakModule.Name name
                        else
                            printfn "  Pruning unreachable type: %s.%s" oakModule.Name name
                        isReachable
                    
                    | ExternalDecl(name, _, _, _) ->
                        let qualifiedName = sprintf "%s.%s" oakModule.Name name
                        let isReachable = Set.contains qualifiedName result.ReachableDeclarations
                        if isReachable then
                            printfn "  Keeping reachable external: %s.%s" oakModule.Name name
                        else
                            printfn "  Pruning unreachable external: %s.%s" oakModule.Name name
                        isReachable
                )
            
            if prunedDeclarations.IsEmpty then
                printfn "  Module %s has no reachable declarations after pruning" oakModule.Name
                if isAlloyModule oakModule.Name then
                    // Even if empty, preserve Alloy modules with original declarations
                    printfn "  Preserving empty Alloy module %s anyway" oakModule.Name
                    Some oakModule
                else
                    None
            else
                printfn "  Module %s has %d declarations after pruning (was %d)" 
                    oakModule.Name prunedDeclarations.Length oakModule.Declarations.Length
                Some { oakModule with Declarations = prunedDeclarations }
    
    let prunedModules = 
        program.Modules 
        |> List.choose pruneModule
    
    printfn "After pruning: %d modules remain (was %d)" 
        prunedModules.Length program.Modules.Length
    
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
    
    // Add a special note about Alloy modules
    sb.AppendLine() |> ignore
    sb.AppendLine("Note: All Alloy modules are preserved automatically to ensure compilation success.") |> ignore
    
    sb.ToString()

/// Validates that the pruned program still contains required dependencies
let validatePrunedProgram (program: OakProgram) : bool =
    // Check that essential Alloy modules are present
    let essentialModules = [
        "Alloy.Memory"
        "Alloy.IO.Console" 
        "Alloy.IO"
    ]
    
    let moduleNames = program.Modules |> List.map (fun m -> m.Name)
    
    let missingModules = 
        essentialModules 
        |> List.filter (fun essentialName -> 
            not (moduleNames |> List.exists (fun modName -> modName = essentialName)))
    
    if not missingModules.IsEmpty then
        printfn "WARNING: Missing essential modules after pruning: %A" missingModules
        false
    else
        let entryPointCount = 
            program.Modules 
            |> List.sumBy (fun m -> 
                m.Declarations 
                |> List.filter (function | EntryPoint _ -> true | _ -> false) 
                |> List.length)
                
        if entryPointCount = 0 then
            printfn "WARNING: No entry points found in pruned program"
            false
        else
            // Count declarations in Alloy modules
            let alloyDeclarationCount = 
                program.Modules
                |> List.filter (fun m -> isAlloyModule m.Name)
                |> List.sumBy (fun m -> m.Declarations.Length)
                
            if alloyDeclarationCount = 0 then
                printfn "WARNING: Alloy modules exist but have no declarations"
                false
            else
                true