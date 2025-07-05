module Dabbit.Analysis.ReachabilityAnalyzer

open System
open Core.XParsec.Foundation
open Core.FCSIngestion.SymbolCollector

/// Reachability analysis result
type ReachabilityResult = {
    Reachable: Set<string>
    Dependencies: Map<string, Set<string>>
    EntryPoints: Set<string>
    Statistics: ReachabilityStatistics
    ReachableSymbols: CollectedSymbol[]
    ProblematicSymbols: ProblematicSymbolReport
}

and ReachabilityStatistics = {
    TotalSymbols: int
    ReachableSymbols: int
    EliminatedSymbols: int
    DependencyEdges: int
}

and ProblematicSymbolReport = {
    /// should trigger errors
    AssemblyReferences: CollectedSymbol[]
    AllocatingSymbols: CollectedSymbol[]
    BCLSymbols: CollectedSymbol[]
    UnknownSymbols: CollectedSymbol[]
}

/// Detect problematic Assembly references that should cause compilation errors
let private detectAssemblyReferences (symbols: CollectedSymbol[]) : CollectedSymbol[] =
    symbols
    |> Array.filter (fun sym ->
        sym.FullName.Contains("Assembly") ||
        sym.DisplayName.Contains("Assembly") ||
        (sym.DeclaringEntity |> Option.exists (fun entity -> entity.Contains("Assembly"))))

/// Convert SymbolDependency array to Map for reachability computation
let private buildDependencyMap (dependencies: SymbolDependency[]) : Map<string, Set<string>> =
    dependencies
    |> Array.groupBy (fun dep -> dep.FromSymbol)
    |> Array.map (fun (fromSymbol, deps) ->
        let targets = deps |> Array.map (fun d -> d.ToSymbol) |> Set.ofArray
        (fromSymbol, targets))
    |> Map.ofArray

/// Compute transitive closure for reachability analysis
let private computeTransitiveClosure (dependencies: Map<string, Set<string>>) (entryPoints: Set<string>) : Set<string> =
    let rec loop (reachable: Set<string>) (worklist: Set<string>) =
        if Set.isEmpty worklist then
            reachable
        else
            let current = Set.minElement worklist
            let remaining = Set.remove current worklist
            
            match Map.tryFind current dependencies with
            | Some deps ->
                let newDeps = Set.difference deps reachable
                loop (Set.union reachable newDeps) (Set.union remaining newDeps)
            | None ->
                loop reachable remaining
    
    loop entryPoints entryPoints

/// MAIN ANALYSIS FUNCTION - works with SymbolCollectionResult
let analyze (symbolCollection: SymbolCollectionResult) : CompilerResult<ReachabilityResult> =
    try
        // Check for Assembly references first - these should be compilation errors
        let assemblyReferences = detectAssemblyReferences (symbolCollection.AllSymbols |> Map.toArray |> Array.map snd)
        
        if assemblyReferences.Length > 0 then
            let assemblyErrors = 
                assemblyReferences
                |> Array.map (fun sym ->
                    ConversionError(
                        "reachability", 
                        sym.FullName, 
                        "Fidelity framework", 
                        sprintf "Assembly reference detected: %s. Fidelity is source-code based and does not use .NET assemblies." sym.FullName))
                |> Array.toList
            CompilerFailure assemblyErrors
        else
            // Build dependency map from collected dependencies
            let dependencyMap = buildDependencyMap symbolCollection.Dependencies
            
            // Extract entry points as set of full names
            let entryPointNames = 
                symbolCollection.EntryPoints 
                |> Array.map (fun ep -> ep.FullName) 
                |> Set.ofArray
            
            // If no entry points found, look for main-like functions
            let actualEntryPoints = 
                if Set.isEmpty entryPointNames then
                    symbolCollection.AllSymbols
                    |> Map.filter (fun name _ -> 
                        name.EndsWith(".main") || name.EndsWith(".Main") || name = "main")
                    |> Map.toSeq
                    |> Seq.map fst
                    |> Set.ofSeq
                else entryPointNames
            
            // Compute reachable symbols
            let reachableNames = computeTransitiveClosure dependencyMap actualEntryPoints
            
            // Filter to get reachable symbol objects
            let reachableSymbols = 
                symbolCollection.AllSymbols
                |> Map.filter (fun name _ -> Set.contains name reachableNames)
                |> Map.toArray
                |> Array.map snd
            
            // Build problematic symbol report
            let problematicReport = {
                AssemblyReferences = assemblyReferences
                AllocatingSymbols = 
                    reachableSymbols 
                    |> Array.filter (fun sym -> sym.IsAllocation)
                BCLSymbols = 
                    reachableSymbols 
                    |> Array.filter (fun sym -> sym.Category = BCLType)
                UnknownSymbols = 
                    reachableSymbols 
                    |> Array.filter (fun sym -> sym.Category = Unknown)
            }
            
            // Calculate statistics
            let stats = {
                TotalSymbols = symbolCollection.Statistics.TotalSymbols
                ReachableSymbols = reachableSymbols.Length
                EliminatedSymbols = symbolCollection.Statistics.TotalSymbols - reachableSymbols.Length
                DependencyEdges = symbolCollection.Dependencies.Length
            }
            
            let result = {
                Reachable = reachableNames
                Dependencies = dependencyMap
                EntryPoints = actualEntryPoints
                Statistics = stats
                ReachableSymbols = reachableSymbols
                ProblematicSymbols = problematicReport
            }
            
            Success result
        
    with ex ->
        CompilerFailure [InternalError("analyze", ex.Message, Some ex.StackTrace)]

/// Check if reachability analysis passes zero-allocation requirements
let validateZeroAllocation (result: ReachabilityResult) : CompilerResult<unit> =
    let problems = result.ProblematicSymbols
    
    // Check for allocating symbols without Alloy replacements
    let unfixableAllocations = 
        problems.AllocatingSymbols
        |> Array.filter (fun sym -> sym.AlloyReplacement.IsNone)
    
    if unfixableAllocations.Length > 0 then
        let errors = 
            unfixableAllocations
            |> Array.map (fun sym ->
                ConversionError(
                    "allocation-check", 
                    sym.FullName, 
                    "zero-allocation", 
                    sprintf "Allocating function %s has no Alloy replacement. Zero-allocation guarantee violated." sym.DisplayName))
            |> Array.toList
        CompilerFailure errors
    else
        Success ()

/// Generate reachability report for diagnostics
let generateReport (result: ReachabilityResult) : string =
    let sb = System.Text.StringBuilder()
    
    sb.AppendLine("Firefly Reachability Analysis Report") |> ignore
    sb.AppendLine("====================================") |> ignore
    sb.AppendLine() |> ignore
    
    sb.AppendLine(sprintf "Total symbols: %d" result.Statistics.TotalSymbols) |> ignore
    sb.AppendLine(sprintf "Reachable symbols: %d" result.Statistics.ReachableSymbols) |> ignore
    sb.AppendLine(sprintf "Eliminated symbols: %d" result.Statistics.EliminatedSymbols) |> ignore
    sb.AppendLine(sprintf "Entry points: %d" (Set.count result.EntryPoints)) |> ignore
    sb.AppendLine() |> ignore
    
    sb.AppendLine("Entry Points:") |> ignore
    result.EntryPoints |> Set.iter (fun ep -> sb.AppendLine(sprintf "  %s" ep) |> ignore)
    sb.AppendLine() |> ignore
    
    let problems = result.ProblematicSymbols
    
    if problems.AllocatingSymbols.Length > 0 then
        sb.AppendLine("Allocating Symbols (require Alloy replacements):") |> ignore
        problems.AllocatingSymbols
        |> Array.iter (fun sym ->
            match sym.AlloyReplacement with
            | Some replacement ->
                sb.AppendLine(sprintf "  %s → %s" sym.DisplayName replacement) |> ignore
            | None ->
                sb.AppendLine(sprintf "  %s (⚠️ NO REPLACEMENT)" sym.DisplayName) |> ignore)
        sb.AppendLine() |> ignore
    
    if problems.BCLSymbols.Length > 0 then
        sb.AppendLine("BCL References (should be avoided):") |> ignore
        problems.BCLSymbols
        |> Array.truncate 10
        |> Array.iter (fun sym -> sb.AppendLine(sprintf "  %s" sym.FullName) |> ignore)
        if problems.BCLSymbols.Length > 10 then
            sb.AppendLine(sprintf "  ... and %d more" (problems.BCLSymbols.Length - 10)) |> ignore
        sb.AppendLine() |> ignore
    
    if problems.UnknownSymbols.Length > 0 then
        sb.AppendLine("Unknown Symbols (need classification):") |> ignore
        problems.UnknownSymbols
        |> Array.truncate 10
        |> Array.iter (fun sym -> sb.AppendLine(sprintf "  %s" sym.FullName) |> ignore)
        sb.AppendLine() |> ignore
    
    sb.ToString()

/// DEPRECATED: Old function signature for backward compatibility
let analyzeFromParsedInputs (parsedInputs: (string * FSharp.Compiler.Syntax.ParsedInput) list) : ReachabilityResult =
    failwith "analyzeFromParsedInputs is deprecated. Use analyze(SymbolCollectionResult) instead."