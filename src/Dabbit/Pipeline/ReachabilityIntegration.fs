module Dabbit.Pipeline.ReachabilityIntegration

open System
open System.IO
open System.Text.RegularExpressions
open Core.XParsec.Foundation

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
        modules
        |> List.collect (fun m ->
            m.DefinedSymbols
            |> Set.filter (fun symbol -> 
                symbol = "main" || symbol = "hello" || symbol.EndsWith("Main"))
            |> Set.toList
            |> List.map (fun symbol -> (symbol, m.Name)))
    
    /// Compute transitive closure of reachable symbols
    let computeReachableSymbols (modules: ModuleInfo list) (entryPoints: (string * string) list) =
        let symbolMap = SymbolResolver.buildSymbolMap modules
        let moduleMap = modules |> List.map (fun m -> (m.Name, m)) |> Map.ofList
        
        let rec traverse (visited: Set<string>) (worklist: string list) =
            match worklist with
            | [] -> visited
            | symbol :: rest ->
                if Set.contains symbol visited then
                    traverse visited rest
                else
                    let visited' = Set.add symbol visited
                    
                    // Find which module defines this symbol
                    match Map.tryFind symbol symbolMap with
                    | Some moduleName ->
                        match Map.tryFind moduleName moduleMap with
                        | Some moduleInfo ->
                            // Add all dependencies of this symbol
                            let newDeps = 
                                moduleInfo.Dependencies 
                                |> Set.filter (fun dep -> not (Set.contains dep visited'))
                                |> Set.toList
                            traverse visited' (newDeps @ rest)
                        | None -> traverse visited' rest
                    | None -> traverse visited' rest
        
        let initialSymbols = entryPoints |> List.map fst
        traverse Set.empty initialSymbols

/// AST pruning functionality  
module ASTPruner =
    
    /// Check if a function should be kept
    let shouldKeepFunction (functionName: string) (moduleName: string) (reachableSymbols: Set<string>) =
        reachableSymbols.Contains(functionName) ||
        reachableSymbols.Contains(sprintf "%s.%s" moduleName functionName)
    
    /// Prune AST content to only reachable functions
    let pruneAST (originalAST: string) (moduleName: string) (reachableSymbols: Set<string>) =
        let lines = originalAST.Split('\n')
        let result = System.Text.StringBuilder()
        
        let mutable currentFunction = None
        let mutable functionLines = []
        let mutable inFunction = false
        let mutable braceDepth = 0
        
        for line in lines do
            // Check if this line starts a function definition
            let functionMatch = Regex.Match(line, @"MemberOrFunctionOrValue\s*\(\s*val\s+(\w+),")
            if functionMatch.Success then
                // Save previous function if it should be kept
                match currentFunction with
                | Some funcName when shouldKeepFunction funcName moduleName reachableSymbols ->
                    for savedLine in List.rev functionLines do
                        result.AppendLine(savedLine) |> ignore
                | _ -> ()
                
                // Start tracking new function
                currentFunction <- Some functionMatch.Groups.[1].Value
                functionLines <- [line]
                inFunction <- true
                braceDepth <- 0
            elif inFunction then
                functionLines <- line :: functionLines
                
                // Track brace depth to know when function ends
                braceDepth <- braceDepth + (line |> Seq.filter (fun c -> c = '(') |> Seq.length)
                braceDepth <- braceDepth - (line |> Seq.filter (fun c -> c = ')') |> Seq.length)
                
                if braceDepth = 0 then
                    inFunction <- false
            else
                // Non-function content (Entity declarations, etc.)
                result.AppendLine(line) |> ignore
        
        // Handle last function
        match currentFunction with
        | Some funcName when shouldKeepFunction funcName moduleName reachableSymbols ->
            for savedLine in List.rev functionLines do
                result.AppendLine(savedLine) |> ignore
        | _ -> ()
        
        result.ToString()

/// Main integration with compilation pipeline
type PipelineIntegrator(projectDirectory: string) =
    let intermediatesDir = Path.Combine(projectDirectory, "build", "intermediates")
    
    /// Run complete reachability analysis
    member this.RunReachabilityAnalysis() =
        // Discover modules from AST files
        let astFiles = ModuleDiscovery.findASTFiles intermediatesDir
        
        if List.isEmpty astFiles then
            CompilerFailure [InternalError("ReachabilityAnalysis", "No AST files found in intermediates directory", None)]
        else
            try
                let modules = ModuleDiscovery.buildModuleInfo astFiles
                let entryPoints = ReachabilityEngine.findEntryPoints modules
                
                if List.isEmpty entryPoints then
                    CompilerFailure [InternalError("ReachabilityAnalysis", "No entry points found", None)]
                else
                    let reachableSymbols = ReachabilityEngine.computeReachableSymbols modules entryPoints
                    
                    // Generate pruned ASTs
                    let prunedASTs = 
                        modules
                        |> List.choose (fun m ->
                            let moduleSymbols = m.DefinedSymbols
                            let hasReachableSymbols = 
                                not (Set.isEmpty (Set.intersect moduleSymbols reachableSymbols))
                            
                            if hasReachableSymbols then
                                let originalAST = File.ReadAllText(m.FilePath)
                                let prunedAST = ASTPruner.pruneAST originalAST m.Name reachableSymbols
                                Some (m.Name, prunedAST)
                            else
                                None)
                        |> Map.ofList
                    
                    // Generate final AST
                    let finalAST = this.GenerateFinalAST(modules, reachableSymbols)
                    
                    // Calculate statistics
                    let stats = this.CalculateStatistics(modules, reachableSymbols, prunedASTs)
                    
                    Success {
                        ReachableSymbols = reachableSymbols
                        PrunedASTs = prunedASTs
                        FinalAST = finalAST
                        Statistics = stats
                    }
            with
            | ex -> CompilerFailure [InternalError("ReachabilityAnalysis", ex.Message, Some ex)]
    
    /// Generate final AST with all reachable code
    member private this.GenerateFinalAST(modules: ModuleInfo list, reachableSymbols: Set<string>) =
        let sb = System.Text.StringBuilder()
        
        // Add header
        sb.AppendLine("// Final AST - Complete Computation Graph") |> ignore
        sb.AppendLine(sprintf "// Reachable Symbols: %d" (Set.count reachableSymbols)) |> ignore
        sb.AppendLine(sprintf "// Generated: %s UTC" (DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss"))) |> ignore
        sb.AppendLine() |> ignore
        
        // Find main module (contains entry points)
        let mainModule = 
            modules
            |> List.tryFind (fun m -> 
                not (Set.isEmpty (Set.intersect m.DefinedSymbols (set ["main"; "hello"]))))
        
        // Add main module first
        match mainModule with
        | Some main ->
            sb.AppendLine(sprintf "// === MAIN MODULE: %s ===" main.Name) |> ignore
            let originalAST = File.ReadAllText(main.FilePath)
            sb.AppendLine(originalAST) |> ignore
            sb.AppendLine() |> ignore
        | None -> ()
        
        // Add dependency modules
        let dependencyModules = 
            modules |> List.filter (fun m -> 
                match mainModule with 
                | Some main -> m.Name <> main.Name 
                | None -> true)
        
        for module_ in dependencyModules do
            let moduleSymbols = module_.DefinedSymbols
            let hasReachableSymbols = not (Set.isEmpty (Set.intersect moduleSymbols reachableSymbols))
            
            if hasReachableSymbols then
                sb.AppendLine(sprintf "// === DEPENDENCY: %s ===" module_.Name) |> ignore
                let originalAST = File.ReadAllText(module_.FilePath)
                let prunedAST = ASTPruner.pruneAST originalAST module_.Name reachableSymbols
                sb.AppendLine(prunedAST) |> ignore
                sb.AppendLine() |> ignore
        
        sb.ToString()
    
    /// Calculate analysis statistics
    member private this.CalculateStatistics(modules: ModuleInfo list, reachableSymbols: Set<string>, prunedASTs: Map<string, string>) =
        let totalSymbols = modules |> List.sumBy (fun m -> Set.count m.DefinedSymbols)
        let dependencyGraph = SymbolResolver.buildDependencyGraph modules
        let crossModuleDeps = dependencyGraph |> Map.toSeq |> Seq.sumBy (fun (_, deps) -> Set.count deps)
        
        {
            TotalModules = List.length modules
            TotalSymbols = totalSymbols
            ReachableSymbols = Set.count reachableSymbols
            EliminatedSymbols = totalSymbols - Set.count reachableSymbols
            CrossModuleDependencies = crossModuleDeps
        }
    
    /// Write output files
    member this.WriteOutputFiles(result: ReachabilityResult) =
        // Write pruned AST files
        for (moduleName, prunedAST) in Map.toSeq result.PrunedASTs do
            let fileName = sprintf "%s.pruned.ast" moduleName
            let filePath = Path.Combine(intermediatesDir, fileName)
            File.WriteAllText(filePath, prunedAST)
        
        // Find main module for final AST filename
        let mainModuleName = 
            result.PrunedASTs.Keys
            |> Seq.tryFind (fun name -> 
                name.Contains("HelloWorld") || name.Contains("Main") || 
                name.ToLower().Contains("main"))
            |> Option.defaultValue "Program"
        
        // Write final AST
        let finalFileName = sprintf "%s.final.ast" mainModuleName
        let finalFilePath = Path.Combine(intermediatesDir, finalFileName)
        File.WriteAllText(finalFilePath, result.FinalAST)
        
        // Write report
        let reportContent = this.GenerateReport(result)
        let reportPath = Path.Combine(intermediatesDir, "reachability.report")
        File.WriteAllText(reportPath, reportContent)
    
    /// Generate human-readable report
    member private this.GenerateReport(result: ReachabilityResult) =
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

/// Integration point for CompilationOrchestrator
module CompilationOrchestratorIntegration =
    
    let enhanceCompilationPipeline (projectDirectory: string) (progress: CompilationPhase -> string -> unit) =
        progress ReachabilityAnalysis "Starting reachability analysis"
        
        let integrator = PipelineIntegrator(projectDirectory)
        
        match integrator.RunReachabilityAnalysis() with
        | Success result ->
            progress IntermediateGeneration "Writing pruned and final AST files"
            integrator.WriteOutputFiles(result)
            progress IntermediateGeneration "Reachability analysis complete"
            
            Success {
                TotalFiles = result.Statistics.TotalModules
                TotalSymbols = result.Statistics.TotalSymbols
                ReachableSymbols = result.Statistics.ReachableSymbols
                EliminatedSymbols = result.Statistics.EliminatedSymbols
                CompilationTimeMs = 0.0 // TODO: Add timing
            }
        | CompilerFailure errors ->
            progress ReachabilityAnalysis "Reachability analysis failed"
            CompilerFailure errors