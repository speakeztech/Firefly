module Core.IngestionPipeline

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization
open FSharp.Compiler.CodeAnalysis
open Core.FCS.ProjectContext
open Core.PSG.Types
open Core.PSG.Builder
open Core.PSG.Correlation
open Core.PSG.DebugOutput
open Core.PSG.Reachability
open Core.Utilities.IntermediateWriter
open Core.Utilities.RemoveIntermediates

/// Configure JSON serialization with F# support
let private createJsonOptions() =
    let options = JsonSerializerOptions(WriteIndented = true)
    options.Converters.Add(JsonFSharpConverter())
    options

/// Global JSON options for the pipeline
let private jsonOptions = createJsonOptions()

/// Pipeline configuration
type PipelineConfig = {
    CacheStrategy: CacheStrategy
    TemplateName: string option
    CustomTemplateDir: string option
    EnableCouplingAnalysis: bool
    EnableMemoryOptimization: bool
    OutputIntermediates: bool
    IntermediatesDir: string option
}

/// Pipeline result
type PipelineResult = {
    Success: bool
    ProjectResults: ProjectResults option
    ProgramSemanticGraph: ProgramSemanticGraph option
    ReachabilityAnalysis: LibraryAwareReachability option
    Diagnostics: Diagnostic list
}

and Diagnostic = {
    Severity: DiagnosticSeverity
    Message: string
    Location: string option
}

and DiagnosticSeverity = Info | Warning | Error

/// Default pipeline configuration
let defaultConfig = {
    CacheStrategy = Conservative
    TemplateName = None
    CustomTemplateDir = None
    EnableCouplingAnalysis = true
    EnableMemoryOptimization = true
    OutputIntermediates = true
    IntermediatesDir = None
}

/// Write the symbolic AST using F#'s native representation 
let private writeSymbolicAst (parseResults: FSharpParseFileResults[]) (intermediatesDir: string) =
    parseResults |> Array.iter (fun pr ->
        let baseName = Path.GetFileNameWithoutExtension(pr.FileName)
        let astPath = Path.Combine(intermediatesDir, $"{baseName}.sym.ast")
        File.WriteAllText(astPath, sprintf "%A" pr.ParseTree)
        printfn "  Wrote %s (%d bytes)" (Path.GetFileName astPath) (FileInfo(astPath).Length)
    )

/// Write project analysis summary
let private writeProjectSummary (projectResults: ProjectResults) (intermediatesDir: string) =
    let typeCheckData = {|
        Files = projectResults.CompilationOrder
        SymbolCount = projectResults.SymbolUses.Length
        Timestamp = DateTime.UtcNow
        HasErrors = projectResults.CheckResults.HasCriticalErrors
    |}
    let typeCheckJson = JsonSerializer.Serialize(typeCheckData, jsonOptions)
    let typeCheckPath = Path.Combine(intermediatesDir, "project.analysis.json")
    writeFileToPath typeCheckPath typeCheckJson
    printfn "  Wrote %s (%d bytes)" (Path.GetFileName typeCheckPath) typeCheckJson.Length

/// Write PSG summary information
let private writePSGSummary (psg: ProgramSemanticGraph) (intermediatesDir: string) =
    let psgSummary = {|
        NodeCount = psg.Nodes.Count
        EdgeCount = psg.Edges.Length
        EntryPoints = psg.EntryPoints |> List.map (fun ep -> ep.Value)
        SymbolCount = psg.SymbolTable.Count
        FileCount = psg.SourceFiles.Count
        CompilationOrder = psg.CompilationOrder |> List.map Path.GetFileName
    |}
    let psgSummaryPath = Path.Combine(intermediatesDir, "psg.summary.json")
    writeFileToPath psgSummaryPath (JsonSerializer.Serialize(psgSummary, jsonOptions))
    printfn "  Wrote PSG summary (%d nodes, %d edges)" psg.Nodes.Count psg.Edges.Length

/// Generate pruned symbol data from PSG and reachability results
let private generatePrunedSymbolData (psg: ProgramSemanticGraph) (result: LibraryAwareReachability) =
    psg.Nodes
    |> Map.toSeq
    |> Seq.choose (fun (_, node) -> 
        match node.Symbol with
        | Some symbol ->
            let symbolId = symbol.FullName
            Some {|
                SymbolName = symbol.DisplayName
                SymbolKind = symbol.GetType().Name
                SymbolHash = symbol.GetHashCode()
                Range = {|
                    File = node.SourceFile
                    StartLine = node.Range.Start.Line
                    StartColumn = node.Range.Start.Column
                    EndLine = node.Range.End.Line
                    EndColumn = node.Range.End.Column
                |}
                IsReachable = Set.contains symbolId result.BasicResult.ReachableSymbols
                LibraryCategory = Map.tryFind symbolId result.LibraryCategories |> Option.map (sprintf "%A")
                NodeId = node.Id.Value
                SyntaxKind = node.SyntaxKind
            |}
        | None -> None
    )
    |> Seq.filter (fun symbolData -> symbolData.IsReachable)
    |> Seq.toArray

/// Generate comparison data from PSG and reachability results
let private generateComparisonData (psg: ProgramSemanticGraph) (result: LibraryAwareReachability) =
    let originalSymbolCount = 
        psg.Nodes
        |> Map.toSeq
        |> Seq.sumBy (fun (_, node) -> if node.Symbol.IsSome then 1 else 0)
    
    {|
        Summary = {|
            OriginalSymbolCount = originalSymbolCount
            ReachableSymbolCount = result.PruningStatistics.ReachableSymbols
            EliminatedCount = result.PruningStatistics.EliminatedSymbols
            EliminationRate = 
                if originalSymbolCount > 0 then
                    (float result.PruningStatistics.EliminatedSymbols / float originalSymbolCount) * 100.0
                else 0.0
            ComputationTimeMs = result.PruningStatistics.ComputationTimeMs
        |}
        EntryPoints = result.BasicResult.EntryPoints |> List.map (fun ep -> ep.DisplayName)
        CallGraph = result.BasicResult.CallGraph
        PSGMetadata = {|
            TotalNodes = Map.count psg.Nodes
            TotalEdges = psg.Edges.Length
            SourceFiles = psg.SourceFiles |> Map.keys |> Seq.map System.IO.Path.GetFileName |> Seq.toArray
            CompilationOrder = psg.CompilationOrder |> List.map System.IO.Path.GetFileName |> List.toArray
        |}
    |}

/// Generate call graph visualization data from reachability results
let private generateCallGraphData (result: LibraryAwareReachability) =
    let nodes = 
        result.BasicResult.ReachableSymbols 
        |> Set.toArray 
        |> Array.map (fun symbolId -> {|
            Id = symbolId
            Name = symbolId
            IsReachable = true
            Category = Map.tryFind symbolId result.LibraryCategories |> Option.map (sprintf "%A") |> Option.defaultValue "Unknown"
        |})
    
    let edges = 
        result.BasicResult.CallGraph 
        |> Map.toSeq
        |> Seq.collect (fun (from, targets) ->
            targets |> List.map (fun target -> {|
                Source = from
                Target = target
                Kind = "CallsOrReferences"
            |}))
        |> Seq.toArray
    
    {|
        Nodes = nodes
        Edges = edges
        NodeCount = nodes.Length
        EdgeCount = edges.Length
        EntryPoints = result.BasicResult.EntryPoints |> List.map (fun ep -> ep.FullName)
    |}

/// Generate library boundary analysis from reachability results
let private generateLibraryBoundaryData (result: LibraryAwareReachability) =
    let categorySummary =
        result.LibraryCategories
        |> Map.toSeq
        |> Seq.groupBy (fun (_, category) -> category)
        |> Seq.map (fun (category, symbols) -> 
            let symbolIds = symbols |> Seq.map fst |> Set.ofSeq
            let reachableCount = 
                symbolIds 
                |> Set.intersect result.BasicResult.ReachableSymbols 
                |> Set.count
            
            {|
                LibraryCategory = sprintf "%A" category
                TotalSymbols = Set.count symbolIds
                ReachableSymbols = reachableCount
                EliminatedSymbols = Set.count symbolIds - reachableCount
                EliminationRate = 
                    if Set.count symbolIds > 0 then
                        (float (Set.count symbolIds - reachableCount) / float (Set.count symbolIds)) * 100.0
                    else 0.0
                IncludedInAnalysis = 
                    match category with
                    | UserCode | AlloyLibrary -> true
                    | FSharpCore -> false
                    | Other _ -> false
            |})
        |> Seq.toArray
    
    {|
        CategorySummary = categorySummary
        OverallStatistics = {|
            TotalCategories = categorySummary.Length
            CategoriesWithEliminations = categorySummary |> Array.filter (fun c -> c.EliminatedSymbols > 0) |> Array.length
            HighestEliminationRate = 
                if categorySummary.Length > 0 then
                    categorySummary |> Array.map (fun c -> c.EliminationRate) |> Array.max
                else 0.0
        |}
    |}

/// Run the complete ingestion and analysis pipeline
let runPipeline (projectPath: string) (config: PipelineConfig) : Async<PipelineResult> = async {
    let diagnostics = ResizeArray<Diagnostic>()
    
    try
        // Step 1: Prepare intermediate outputs
        if config.OutputIntermediates && config.IntermediatesDir.IsSome then
            printfn "[Pipeline] Preparing intermediates directory..."
            prepareIntermediatesDirectory config.IntermediatesDir
        
        // Step 2: Load and analyze project with FCS
        printfn "[Pipeline] Loading project: %s" projectPath
        let! ctx = loadProject projectPath config.CacheStrategy
        
        printfn "[Pipeline] Analyzing project with FCS..."
        let! projectResults = getProjectResults ctx
        
        // Validate project compilation
        if projectResults.CheckResults.HasCriticalErrors then
            diagnostics.Add {
                Severity = Error
                Message = "Project has critical compilation errors"
                Location = Some projectPath
            }
            return {
                Success = false
                ProjectResults = Some projectResults
                ProgramSemanticGraph = None
                ReachabilityAnalysis = None
                Diagnostics = List.ofSeq diagnostics
            }
        else
            // Step 3: Generate initial intermediate outputs
            if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                printfn "[Pipeline] Writing project analysis intermediates..."
                writeSymbolicAst projectResults.ParseResults config.IntermediatesDir.Value
                writeProjectSummary projectResults config.IntermediatesDir.Value
            
            // Step 4: Build Program Semantic Graph
            printfn "[Pipeline] Building Program Semantic Graph..."

            // Enable nanopass intermediate emission if outputting intermediates
            if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                Core.PSG.Construction.Main.emitNanopassIntermediates <- true
                Core.PSG.Construction.Main.nanopassOutputDir <- config.IntermediatesDir.Value

            let psg = buildProgramSemanticGraph projectResults.CheckResults projectResults.ParseResults

            // Reset nanopass emission flags
            Core.PSG.Construction.Main.emitNanopassIntermediates <- false
            
            // Basic validation of PSG structure
            if psg.Nodes.Count = 0 then
                diagnostics.Add {
                    Severity = Error
                    Message = "PSG construction resulted in empty graph"
                    Location = Some projectPath
                }
                return {
                    Success = false
                    ProjectResults = Some projectResults
                    ProgramSemanticGraph = None
                    ReachabilityAnalysis = None
                    Diagnostics = List.ofSeq diagnostics
                }
            else
                // Add warning if no entry points, but continue execution
                if psg.EntryPoints.Length = 0 then
                    diagnostics.Add {
                        Severity = Warning
                        Message = "PSG has no entry points detected"
                        Location = None
                    }
                
                printfn "[Pipeline] PSG construction complete: %d nodes, %d edges, %d entry points" 
                    psg.Nodes.Count psg.Edges.Length psg.EntryPoints.Length
                
                // Step 5: Generate initial PSG debug outputs (before reachability)
                if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                    printfn "[Pipeline] Writing initial PSG debug outputs..."
                    
                    let correlationContext = createContext projectResults.CheckResults
                    let correlations = 
                        projectResults.ParseResults
                        |> Array.collect (fun pr -> correlateFile pr.ParseTree correlationContext)
                    let stats = generateStats correlationContext correlations
                    
                    // Generate correlations and initial summary
                    generateCorrelationDebugOutput correlations config.IntermediatesDir.Value
                    writePSGSummary psg config.IntermediatesDir.Value
                
                // Step 6: Perform reachability analysis
                printfn "[Pipeline] Performing PSG-based reachability analysis..."
                let reachabilityResult = performReachabilityAnalysis psg
                
                printfn "[Pipeline] Reachability analysis complete: %d/%d symbols reachable (%.1f%% eliminated)" 
                    reachabilityResult.PruningStatistics.ReachableSymbols
                    reachabilityResult.PruningStatistics.TotalSymbols
                    ((float reachabilityResult.PruningStatistics.EliminatedSymbols / float reachabilityResult.PruningStatistics.TotalSymbols) * 100.0)
                
                // Step 7: Generate pruned PSG debug assets
                if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                    printfn "[Pipeline] Writing pruned PSG debug assets..."
                    
                    // Use the marked PSG from reachability analysis
                    let markedPSG = reachabilityResult.MarkedPSG
                    
                    // Generate debug outputs with the marked PSG
                    generatePSGDebugOutput markedPSG config.IntermediatesDir.Value
                    
                    let prunedSymbols = generatePrunedSymbolData markedPSG reachabilityResult
                    let comparisonData = generateComparisonData markedPSG reachabilityResult
                    let callGraphData = generateCallGraphData reachabilityResult
                    let libraryBoundaryData = generateLibraryBoundaryData reachabilityResult
                    
                    writeJsonAsset config.IntermediatesDir.Value "psg.pruned.symbols.json" prunedSymbols
                    writeJsonAsset config.IntermediatesDir.Value "reachability.analysis.json" comparisonData
                    writeJsonAsset config.IntermediatesDir.Value "psg.callgraph.pruned.json" callGraphData
                    writeJsonAsset config.IntermediatesDir.Value "library.boundaries.json" libraryBoundaryData
                    
                    printfn "[Pipeline] Pruned PSG debug assets written successfully"
                
                // Return successful pipeline result
                return {
                    Success = true
                    ProjectResults = Some projectResults
                    ProgramSemanticGraph = Some psg
                    ReachabilityAnalysis = Some reachabilityResult
                    Diagnostics = List.ofSeq diagnostics
                }
            
    with ex ->
        diagnostics.Add {
            Severity = Error
            Message = sprintf "Pipeline execution failed: %s" ex.Message
            Location = Some projectPath
        }
        return {
            Success = false
            ProjectResults = None
            ProgramSemanticGraph = None
            ReachabilityAnalysis = None
            Diagnostics = List.ofSeq diagnostics
        }
}