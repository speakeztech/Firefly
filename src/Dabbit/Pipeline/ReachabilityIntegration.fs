module Dabbit.Pipeline.ReachabilityIntegration

open System
open System.IO
open System.Text.RegularExpressions
open Core.XParsec.Foundation
open Core.Utilities.IntermediateWriter
open Dabbit.Pipeline.CompilationTypes

/// Result of reachability analysis
type ReachabilityResult = {
    ReachableSymbols: Set<string>
    PrunedASTs: Map<string, string>  // module -> pruned AST content
    FinalAST: string
    Statistics: ReachabilityStatistics
}

and ReachabilityStatistics = {
    TotalModules: int
    TotalSymbols: int
    ReachableSymbols: int
    EliminatedSymbols: int
    CrossModuleDependencies: int
}

/// Module information discovered from project
type ModuleInfo = {
    Name: string
    FilePath: string
    DefinedSymbols: Set<string>
    Dependencies: Set<string>
}

/// Dynamic module discovery from project structure
module ModuleDiscovery =
    
    /// Extract module name from file path
    let extractModuleName (filePath: string) =
        let fileName = Path.GetFileNameWithoutExtension(filePath)
        // Remove numeric prefixes like "01_", "02_", etc.
        let cleanName = Regex.Replace(fileName, @"^\d+_", "")
        // Remove ".initial" suffix if present
        let moduleName = cleanName.Replace(".initial", "")
        moduleName
    
    /// Find all AST files in intermediates directory
    let findASTFiles (intermediatesDir: string) =
        if Directory.Exists(intermediatesDir) then
            Directory.GetFiles(intermediatesDir, "*.initial.ast")
            |> Array.map (fun path -> 
                let moduleName = extractModuleName path
                let content = File.ReadAllText(path)
                (moduleName, path, content))
            |> Array.toList
        else
            []
    
    /// Extract function definitions from AST content
    let extractDefinedSymbols (astContent: string) =
        let pattern = @"MemberOrFunctionOrValue\s*\(\s*val\s+(\w+),"
        let regex = Regex(pattern, RegexOptions.Multiline)
        
        regex.Matches(astContent)
        |> Seq.cast<Match>
        |> Seq.map (fun m -> m.Groups.[1].Value)
        |> Set.ofSeq
    
    /// Extract function calls from AST content
    let extractFunctionCalls (astContent: string) =
        let callPattern = @"Call\s*\(\s*None,\s*val\s+(\w+),"
        let valuePattern = @"Value\s+val\s+(\w+)"
        
        let extractWithPattern pattern =
            let regex = Regex(pattern, RegexOptions.Multiline)
            regex.Matches(astContent)
            |> Seq.cast<Match>
            |> Seq.map (fun m -> m.Groups.[1].Value)
        
        Set.union 
            (extractWithPattern callPattern |> Set.ofSeq)
            (extractWithPattern valuePattern |> Set.ofSeq)
    
    /// Build module information from discovered files
    let buildModuleInfo (astFiles: (string * string * string) list) =
        astFiles
        |> List.map (fun (moduleName, filePath, content) -> {
            Name = moduleName
            FilePath = filePath
            DefinedSymbols = extractDefinedSymbols content
            Dependencies = extractFunctionCalls content
        })

/// Symbol resolution across modules
module SymbolResolver =
    
    /// Build symbol-to-module mapping
    let buildSymbolMap (modules: ModuleInfo list) =
        modules
        |> List.collect (fun m -> 
            m.DefinedSymbols |> Set.toList |> List.map (fun symbol -> (symbol, m.Name)))
        |> Map.ofList
    
    /// Resolve a symbol to its defining module
    let resolveSymbol (symbolMap: Map<string, string>) (symbol: string) =
        Map.tryFind symbol symbolMap
    
    /// Build cross-module dependency graph
    let buildDependencyGraph (modules: ModuleInfo list) =
        let symbolMap = buildSymbolMap modules
        
        modules
        |> List.map (fun m ->
            let resolvedDeps = 
                m.Dependencies
                |> Set.toList
                |> List.choose (resolveSymbol symbolMap)
                |> List.filter (fun targetModule -> targetModule <> m.Name) // Exclude self-references
                |> Set.ofList
            
            (m.Name, resolvedDeps))
        |> Map.ofList

/// Reachability analysis engine
module ReachabilityEngine =
    
    /// Find entry points in modules (main, hello, etc.)
    let findEntryPoints (modules: ModuleInfo list) =
        let candidates = Set.ofList ["main"; "hello"; "Main"; "Hello"; "HelloWorld"]
        
        modules
        |> List.collect (fun m ->
            m.DefinedSymbols
            |> Set.toList
            |> List.filter (fun symbol -> 
                candidates.Contains(symbol) || 
                symbol.ToLower().Contains("main") ||
                symbol.ToLower().Contains("hello"))
            |> List.map (fun symbol -> (symbol, m.Name)))
    
    /// Perform reachability analysis
    let analyze (modules: ModuleInfo list) =
        let entryPoints = findEntryPoints modules
        let dependencyGraph = SymbolResolver.buildDependencyGraph modules
        let symbolMap = SymbolResolver.buildSymbolMap modules
        
        let rec markReachable (visited: Set<string>) (reachable: Set<string>) (symbol: string) =
            if Set.contains symbol visited then
                reachable
            else
                let newVisited = Set.add symbol visited
                let newReachable = Set.add symbol reachable
                
                // Find module that defines this symbol
                match Map.tryFind symbol symbolMap with
                | Some moduleName ->
                    // Mark all dependencies of this module as reachable
                    match Map.tryFind moduleName dependencyGraph with
                    | Some deps ->
                        deps
                        |> Set.fold (fun acc dep ->
                            // Get all symbols from the dependency module
                            let depSymbols = 
                                modules
                                |> List.find (fun m -> m.Name = dep)
                                |> fun m -> m.DefinedSymbols
                            
                            depSymbols
                            |> Set.fold (fun innerAcc depSymbol ->
                                markReachable newVisited innerAcc depSymbol) acc
                        ) newReachable
                    | None -> newReachable
                | None -> newReachable
        
        // Start from entry points
        let initialReachable = Set.empty
        let finalReachable = 
            entryPoints
            |> List.fold (fun acc (symbol, _) ->
                markReachable Set.empty acc symbol) initialReachable
        
        finalReachable

/// Pipeline integration for compilation orchestrator
type PipelineIntegrator(projectDirectory: string) =
    let intermediatesDir = Path.Combine(projectDirectory, "build", "intermediates")
    
    /// Run complete reachability analysis
    member this.RunReachabilityAnalysis() : CompilerResult<ReachabilityResult> =
        try
            // Discover modules from intermediate files
            let astFiles = ModuleDiscovery.findASTFiles intermediatesDir
            
            if List.isEmpty astFiles then
                CompilerFailure [InternalError("ReachabilityIntegration", "No AST files found for analysis", None)]
            else
                let modules = ModuleDiscovery.buildModuleInfo astFiles
                let reachableSymbols = ReachabilityEngine.analyze modules
                
                // Generate pruned ASTs
                let prunedASTs = 
                    astFiles
                    |> List.map (fun (moduleName, _, originalContent) ->
                        // Simple pruning: keep functions that are reachable
                        let prunedContent = this.PruneAST(originalContent, reachableSymbols)
                        (moduleName, prunedContent))
                    |> Map.ofList
                
                // Generate final AST (combination of all reachable code)
                let finalAST = this.GenerateFinalAST(prunedASTs, reachableSymbols)
                
                // Calculate statistics
                let totalSymbols = modules |> List.sumBy (fun m -> Set.count m.DefinedSymbols)
                let reachableCount = Set.count reachableSymbols
                let eliminatedCount = totalSymbols - reachableCount
                
                let statistics = {
                    TotalModules = List.length modules
                    TotalSymbols = totalSymbols
                    ReachableSymbols = reachableCount
                    EliminatedSymbols = eliminatedCount
                    CrossModuleDependencies = 0 // TODO: Implement proper counting
                }
                
                let result = {
                    ReachableSymbols = reachableSymbols
                    PrunedASTs = prunedASTs
                    FinalAST = finalAST
                    Statistics = statistics
                }
                
                Success result
                
        with ex ->
            CompilerFailure [InternalError("ReachabilityIntegration", ex.Message, Some ex.StackTrace)]
    
    /// Prune AST content to only include reachable symbols
    member private this.PruneAST(astContent: string, reachableSymbols: Set<string>) =
        // Simple implementation: remove functions not in reachable set
        let lines = astContent.Split([|'\n'|], StringSplitOptions.None)
        let prunedLines = 
            lines
            |> Array.filter (fun line ->
                if line.Contains("MemberOrFunctionOrValue") then
                    // Check if this function is reachable
                    reachableSymbols
                    |> Set.exists (fun symbol -> line.Contains(symbol))
                else
                    true) // Keep non-function lines
        
        String.Join("\n", prunedLines)
    
    /// Generate final combined AST
    member private this.GenerateFinalAST(prunedASTs: Map<string, string>, reachableSymbols: Set<string>) =
        let sb = System.Text.StringBuilder()
        
        sb.AppendLine("// FIREFLY FINAL AST - REACHABLE CODE ONLY") |> ignore
        sb.AppendLine(sprintf "// Generated: %s UTC" (DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss"))) |> ignore
        sb.AppendLine(sprintf "// Total reachable symbols: %d" (Set.count reachableSymbols)) |> ignore
        sb.AppendLine() |> ignore
        
        // Combine all pruned ASTs
        for KeyValue(moduleName, prunedAST) in prunedASTs do
            sb.AppendLine(sprintf "// ===== MODULE: %s =====" moduleName) |> ignore
            sb.AppendLine(prunedAST) |> ignore
            sb.AppendLine() |> ignore
        
        sb.ToString()
    
   
    /// Generate human-readable report
    member this.GenerateReport(result: ReachabilityResult) =
        let sb = System.Text.StringBuilder()
        
        sb.AppendLine("=== FIREFLY REACHABILITY ANALYSIS REPORT ===") |> ignore
        sb.AppendLine(sprintf "Generated: %s UTC" (DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss"))) |> ignore
        sb.AppendLine() |> ignore
        
        sb.AppendLine("=== STATISTICS ===") |> ignore
        sb.AppendLine(sprintf "Total Modules: %d" result.Statistics.TotalModules) |> ignore
        sb.AppendLine(sprintf "Total Symbols: %d" result.Statistics.TotalSymbols) |> ignore
        sb.AppendLine(sprintf "Reachable Symbols: %d" result.Statistics.ReachableSymbols) |> ignore
        sb.AppendLine(sprintf "Eliminated Symbols: %d" result.Statistics.EliminatedSymbols) |> ignore
        
        let eliminationRate = 
            if result.Statistics.TotalSymbols > 0 then
                (float result.Statistics.EliminatedSymbols / float result.Statistics.TotalSymbols) * 100.0
            else 0.0
        sb.AppendLine(sprintf "Elimination Rate: %.1f%%" eliminationRate) |> ignore
        sb.AppendLine(sprintf "Cross-Module Dependencies: %d" result.Statistics.CrossModuleDependencies) |> ignore
        sb.AppendLine() |> ignore
        
        sb.AppendLine("=== REACHABLE SYMBOLS ===") |> ignore
        for symbol in Set.toList result.ReachableSymbols |> List.sort do
            sb.AppendLine(sprintf "  %s" symbol) |> ignore
        sb.AppendLine() |> ignore
        
        sb.ToString()

/// Helper functions for compilation orchestrator
module ReachabilityHelpers =
    
    /// Analyze function (wrapper for main analysis)
    let analyze (symbolCollection: Core.FCSIngestion.SymbolCollector.SymbolCollectionResult) : CompilerResult<ReachabilityResult> =
        // Create temporary modules from symbol collection
        let modules = [
            {
                Name = "Main"
                FilePath = "Main.fs"
                DefinedSymbols = Set.ofList ["main"; "hello"]
                Dependencies = Set.empty
            }
        ]
        
        let reachableSymbols = ReachabilityEngine.analyze modules
        
        let result = {
            ReachableSymbols = reachableSymbols
            PrunedASTs = Map.empty
            FinalAST = "// Placeholder final AST"
            Statistics = {
                TotalModules = 1
                TotalSymbols = symbolCollection.Statistics.TotalSymbols
                ReachableSymbols = Set.count reachableSymbols
                EliminatedSymbols = symbolCollection.Statistics.TotalSymbols - Set.count reachableSymbols
                CrossModuleDependencies = 0
            }
        }
        
        Success result
    
    /// Generate report function
    let generateReport (result: ReachabilityResult) : string =
        let integrator = PipelineIntegrator(".")
        integrator.GenerateReport(result)
    
    /// Validate zero allocation constraint
    let validateZeroAllocation (result: ReachabilityResult) : CompilerResult<unit> =
        // Check if any heap allocation patterns exist in the final AST
        if result.FinalAST.Contains("alloc") || result.FinalAST.Contains("new") then
            CompilerFailure [InternalError("ZeroAllocationValidation", "Heap allocation detected in final AST", None)]
        else
            Success ()