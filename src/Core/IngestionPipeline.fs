module Core.IngestionPipeline

open System
open System.IO
open System.Text.Json
open FSharp.Native.Compiler.CodeAnalysis
open Core.CompilerConfig
open Core.FCS.ProjectContext
open Core.PSG.Types
open Core.PSG.Builder
open Core.PSG.Correlation
open Core.PSG.DebugOutput
open Core.PSG.Reachability
open Core.Utilities.IntermediateWriter
open Core.Utilities.RemoveIntermediates
open Core.PSG.Nanopass.ValidateNativeTypes

/// JSON options with F# support (reuse from IntermediateWriter)
let private jsonOptions = jsonOptionsWithFSharpSupport

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
    /// Baker enrichment result (Phase 4: member body mappings)
    BakerResult: Baker.Baker.BakerEnrichmentResult option
    Diagnostics: Diagnostic list
}

and Diagnostic = {
    Severity: DiagnosticSeverity
    Message: string
    Location: string option
}

and DiagnosticSeverity = Info | Warning | DiagError

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
                SyntaxKind = SyntaxKindT.toString node.Kind
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
            prepareIntermediatesDirectory config.IntermediatesDir

        // Step 2: Load and analyze project with FCS
        let! ctx = loadProject projectPath config.CacheStrategy
        let! projectResults = getProjectResults ctx
        
        // Validate project compilation
        if projectResults.CheckResults.HasCriticalErrors then
            diagnostics.Add {
                Severity = DiagError
                Message = "Project has critical compilation errors"
                Location = Some projectPath
            }
            return {
                Success = false
                ProjectResults = Some projectResults
                ProgramSemanticGraph = None
                ReachabilityAnalysis = None
                BakerResult = None
                Diagnostics = List.ofSeq diagnostics
            }
        else
            // Step 3: Generate initial intermediate outputs
            if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                writeSymbolicAst projectResults.ParseResults config.IntermediatesDir.Value
                writeProjectSummary projectResults config.IntermediatesDir.Value

            // Step 4: Build Program Semantic Graph

            // Enable nanopass intermediate emission if outputting intermediates
            if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                enableNanopassIntermediates config.IntermediatesDir.Value

            let psg = buildProgramSemanticGraph projectResults.CheckResults projectResults.ParseResults

            // Reset nanopass emission flags
            disableNanopassIntermediates()
            
            // Basic validation of PSG structure
            if psg.Nodes.Count = 0 then
                diagnostics.Add {
                    Severity = DiagError
                    Message = "PSG construction resulted in empty graph"
                    Location = Some projectPath
                }
                return {
                    Success = false
                    ProjectResults = Some projectResults
                    ProgramSemanticGraph = None
                    ReachabilityAnalysis = None
                    BakerResult = None
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
                
                // Step 5: Generate initial PSG debug outputs (before reachability)
                if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                    let correlationContext = createContext projectResults.CheckResults
                    let correlations =
                        projectResults.ParseResults
                        |> Array.collect (fun pr -> correlateFile pr.ParseTree correlationContext)
                    // Generate stats (result not currently used but function may have side effects)
                    generateStats correlationContext correlations |> ignore

                    // Generate correlations and initial summary
                    generateCorrelationDebugOutput correlations config.IntermediatesDir.Value
                    writePSGSummary psg config.IntermediatesDir.Value
                
                // Step 5.5: Extract SRTP call relationships (before reachability!)
                // This returns additional call relationships that get merged into the call graph
                let srtpResult = Core.PSG.Nanopass.ExtractSRTPEdges.run projectResults.CheckResults
                let additionalCalls =
                    if Map.isEmpty srtpResult.AdditionalCalls then None
                    else Some srtpResult.AdditionalCalls

                // Step 6: Perform reachability analysis (narrows the graph)
                // Pass in SRTP-discovered call relationships to merge into the call graph
                let reachabilityResult = performReachabilityAnalysis psg additionalCalls

                // Enrichment nanopasses on narrowed graph
                if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                    enableNanopassIntermediates config.IntermediatesDir.Value

                let enrichedPSG = runEnrichmentPasses reachabilityResult.MarkedPSG projectResults.CheckResults

                disableNanopassIntermediates()

                // Step 7: Generate pruned PSG debug assets
                if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                    // Generate debug outputs with the enriched PSG
                    generatePSGDebugOutput enrichedPSG config.IntermediatesDir.Value

                    let prunedSymbols = generatePrunedSymbolData enrichedPSG reachabilityResult
                    let comparisonData = generateComparisonData enrichedPSG reachabilityResult
                    let callGraphData = generateCallGraphData reachabilityResult
                    let libraryBoundaryData = generateLibraryBoundaryData reachabilityResult

                    writeJsonAsset config.IntermediatesDir.Value "psg.pruned.symbols.json" prunedSymbols
                    writeJsonAsset config.IntermediatesDir.Value "reachability.analysis.json" comparisonData
                    writeJsonAsset config.IntermediatesDir.Value "psg.callgraph.pruned.json" callGraphData
                    writeJsonAsset config.IntermediatesDir.Value "library.boundaries.json" libraryBoundaryData

                // Step 7.5: Validate native types (halt on non-native BCL types)
                eprintfn "[PIPELINE] About to call validateReachable"
                let nativeValidation = validateReachable enrichedPSG
                eprintfn "[PIPELINE] validateReachable returned HasErrors=%b" nativeValidation.HasErrors

                if nativeValidation.HasErrors then
                    // Add all native type errors as diagnostics
                    nativeValidation.Errors |> List.iter (fun err ->
                        diagnostics.Add {
                            Severity = DiagError
                            Message = err.Message
                            Location = Some (sprintf "Node %s (%s)" err.NodeId err.SyntaxKind)
                        })

                    return {
                        Success = false
                        ProjectResults = Some projectResults
                        ProgramSemanticGraph = Some enrichedPSG
                        ReachabilityAnalysis = Some reachabilityResult
                        BakerResult = None
                        Diagnostics = List.ofSeq diagnostics
                    }
                else
                    // Step 8: Run Baker enrichment
                    let bakerResult = Baker.Baker.enrich enrichedPSG projectResults.CheckResults

                    // Return successful pipeline result
                    return {
                        Success = true
                        ProjectResults = Some projectResults
                        ProgramSemanticGraph = Some enrichedPSG
                        ReachabilityAnalysis = Some reachabilityResult
                        BakerResult = Some bakerResult
                        Diagnostics = List.ofSeq diagnostics
                    }
            
    with ex ->
        diagnostics.Add {
            Severity = DiagError
            Message = sprintf "Pipeline execution failed: %s" ex.Message
            Location = Some projectPath
        }
        return {
            Success = false
            ProjectResults = None
            ProgramSemanticGraph = None
            ReachabilityAnalysis = None
            BakerResult = None
            Diagnostics = List.ofSeq diagnostics
        }
}

// ============================================================================
// PIPELINE WITH PRE-LOADED RESULTS
// ============================================================================
// This entry point accepts already-loaded FCS results (from FidprojLoader)
// and runs the enrichment pipeline without re-loading the project.

/// Run the pipeline with pre-loaded FCS results (used by orchestrator after FidprojLoader)
let runPipelineWithResults
    (checkResults: FSharpCheckProjectResults)
    (parseResults: FSharpParseFileResults[])
    (config: PipelineConfig)
    : PipelineResult =

    let diagnostics = ResizeArray<Diagnostic>()

    try
        // Step 1: Prepare intermediate outputs
        if config.OutputIntermediates && config.IntermediatesDir.IsSome then
            prepareIntermediatesDirectory config.IntermediatesDir

        // Create a ProjectResults-like structure for compatibility
        // Note: We don't have full ProjectResults since we came from FidprojLoader
        let compilationOrder =
            parseResults
            |> Array.map (fun pr -> pr.FileName)

        // Validate project compilation
        if checkResults.HasCriticalErrors then
            diagnostics.Add {
                Severity = DiagError
                Message = "Project has critical compilation errors"
                Location = None
            }
            {
                Success = false
                ProjectResults = None
                ProgramSemanticGraph = None
                ReachabilityAnalysis = None
                BakerResult = None
                Diagnostics = List.ofSeq diagnostics
            }
        else
            // EARLY VALIDATION: Check for BCL types BEFORE building PSG
            // FAIL FAST - stops immediately on first BCL type detected
            let earlyValidationResult =
                try
                    validateEarlyFailFast checkResults
                    None  // No error
                with
                | BclTypeDetectedException(file, line, col, typeName) ->
                    Some (file, line, col, typeName)

            match earlyValidationResult with
            | Some (file, line, col, typeName) ->
                // Clean, controlled error output - BCL types should NEVER appear
                eprintfn ""
                eprintfn "BCL TYPE ERROR: Type requires .NET runtime"
                eprintfn ""
                eprintfn "  File: %s" file
                eprintfn "  Location: line %d, column %d" line col
                eprintfn "  Type: %s" typeName
                eprintfn ""
                eprintfn "BCL types cannot appear in the Firefly compilation pipeline."
                eprintfn "They require the .NET garbage collector and managed runtime."
                eprintfn ""
                eprintfn "This is a bug - BCL types should never reach this point:"
                eprintfn "  - If in library code: The library must use native types (NativeStr, voption, nativeptr, etc.)"
                eprintfn "  - If in user code: Use Alloy's API which provides native alternatives"
                eprintfn ""
                eprintfn "Allowed FSharp.Core types: int, byte, bool, unit, voption, nativeptr, etc."
                eprintfn "Forbidden BCL types: string, option, list, System.*, etc."
                eprintfn ""

                {
                    Success = false
                    ProjectResults = None
                    ProgramSemanticGraph = None
                    ReachabilityAnalysis = None
                    BakerResult = None
                    Diagnostics = [{
                        Severity = DiagError
                        Message = sprintf "BCL type '%s' at %s:%d:%d" typeName file line col
                        Location = Some file
                    }]
                }

            | None ->
            // Step 2: Generate initial intermediate outputs
            if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                writeSymbolicAst parseResults config.IntermediatesDir.Value

            // Step 3: Build Program Semantic Graph
            if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                enableNanopassIntermediates config.IntermediatesDir.Value

            let psg = buildProgramSemanticGraph checkResults parseResults

            disableNanopassIntermediates()

            // Basic validation of PSG structure
            if psg.Nodes.Count = 0 then
                diagnostics.Add {
                    Severity = DiagError
                    Message = "PSG construction resulted in empty graph"
                    Location = None
                }
                {
                    Success = false
                    ProjectResults = None
                    ProgramSemanticGraph = None
                    ReachabilityAnalysis = None
                    BakerResult = None
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

                // Step 4: Generate initial PSG debug outputs (before reachability)
                if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                    let correlationContext = createContext checkResults
                    let correlations =
                        parseResults
                        |> Array.collect (fun pr -> correlateFile pr.ParseTree correlationContext)
                    generateStats correlationContext correlations |> ignore
                    generateCorrelationDebugOutput correlations config.IntermediatesDir.Value
                    writePSGSummary psg config.IntermediatesDir.Value

                // Step 5: Extract SRTP call relationships (before reachability!)
                let srtpResult = Core.PSG.Nanopass.ExtractSRTPEdges.run checkResults
                let additionalCalls =
                    if Map.isEmpty srtpResult.AdditionalCalls then None
                    else Some srtpResult.AdditionalCalls

                // Step 6: Perform reachability analysis (narrows the graph)
                let reachabilityResult = performReachabilityAnalysis psg additionalCalls

                // Step 7: Enrichment nanopasses on narrowed graph
                if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                    enableNanopassIntermediates config.IntermediatesDir.Value

                let enrichedPSG = runEnrichmentPasses reachabilityResult.MarkedPSG checkResults

                disableNanopassIntermediates()

                // Step 8: Generate pruned PSG debug assets
                if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                    generatePSGDebugOutput enrichedPSG config.IntermediatesDir.Value

                    let prunedSymbols = generatePrunedSymbolData enrichedPSG reachabilityResult
                    let comparisonData = generateComparisonData enrichedPSG reachabilityResult
                    let callGraphData = generateCallGraphData reachabilityResult
                    let libraryBoundaryData = generateLibraryBoundaryData reachabilityResult

                    writeJsonAsset config.IntermediatesDir.Value "psg.pruned.symbols.json" prunedSymbols
                    writeJsonAsset config.IntermediatesDir.Value "reachability.analysis.json" comparisonData
                    writeJsonAsset config.IntermediatesDir.Value "psg.callgraph.pruned.json" callGraphData
                    writeJsonAsset config.IntermediatesDir.Value "library.boundaries.json" libraryBoundaryData

                // Step 9: Validate native types
                let nativeValidation = validateReachable enrichedPSG

                if nativeValidation.HasErrors then
                    nativeValidation.Errors |> List.iter (fun err ->
                        diagnostics.Add {
                            Severity = DiagError
                            Message = err.Message
                            Location = Some (sprintf "Node %s (%s)" err.NodeId err.SyntaxKind)
                        })

                    {
                        Success = false
                        ProjectResults = None
                        ProgramSemanticGraph = Some enrichedPSG
                        ReachabilityAnalysis = Some reachabilityResult
                        BakerResult = None
                        Diagnostics = List.ofSeq diagnostics
                    }
                else
                    // Step 10: Run Baker enrichment
                    let bakerResult = Baker.Baker.enrich enrichedPSG checkResults

                    {
                        Success = true
                        ProjectResults = None
                        ProgramSemanticGraph = Some enrichedPSG
                        ReachabilityAnalysis = Some reachabilityResult
                        BakerResult = Some bakerResult
                        Diagnostics = List.ofSeq diagnostics
                    }

    with ex ->
        diagnostics.Add {
            Severity = DiagError
            Message = sprintf "Pipeline execution failed: %s" ex.Message
            Location = None
        }
        {
            Success = false
            ProjectResults = None
            ProgramSemanticGraph = None
            ReachabilityAnalysis = None
            BakerResult = None
            Diagnostics = List.ofSeq diagnostics
        }

// ============================================================================
// OPTIMIZED PIPELINE (Structural reachability pre-filter)
// ============================================================================
// This pipeline uses structural reachability BEFORE type-checking to minimize
// the amount of code that needs expensive FCS type analysis.
//
// Flow:
// 1. Parse all files (fast)
// 2. Build structural PSG (fast, no symbols)
// 3. Structural reachability (fast, narrows file set)
// 4. Type-check only reachable files (much faster!)
// 5. Build correlated PSG with symbols
// 6. Full reachability (narrows node set)
// 7. Run enrichment nanopasses on narrowed graph
// 8. Baker enrichment

/// Run the optimized ingestion pipeline with structural pre-filtering
/// Uses structural reachability to minimize type-checking scope
let runOptimizedPipeline (projectPath: string) (config: PipelineConfig) : Async<PipelineResult> = async {
    let diagnostics = ResizeArray<Diagnostic>()

    try
        // Step 1: Prepare intermediate outputs
        if config.OutputIntermediates && config.IntermediatesDir.IsSome then
            prepareIntermediatesDirectory config.IntermediatesDir

        // Step 2: Load project and get PARSE-ONLY results (fast)
        let! ctx = loadProject projectPath config.CacheStrategy
        let! parseOnly = getParseOnlyResults ctx

        diagnostics.Add {
            Severity = Info
            Message = sprintf "Parsed %d files" parseOnly.ParseResults.Length
            Location = None
        }

        // Step 3: Build STRUCTURAL PSG (fast, no type checking)
        let structuralPSG = Core.PSG.Construction.Main.buildStructuralGraph parseOnly.ParseResults

        diagnostics.Add {
            Severity = Info
            Message = sprintf "Built structural PSG with %d nodes" structuralPSG.Nodes.Count
            Location = None
        }

        // Step 4: Structural reachability (fast)
        let structuralReach, markedPSG = performStructuralReachabilityAnalysis structuralPSG

        diagnostics.Add {
            Severity = Info
            Message = sprintf "Structural reachability: %d reachable functions, %d reachable nodes"
                structuralReach.ReachableFunctions.Count
                structuralReach.ReachableNodes.Count
            Location = None
        }

        // Step 5: Get reachable files
        let reachableFiles = getReachableFiles markedPSG

        diagnostics.Add {
            Severity = Info
            Message = sprintf "Reachable files: %d of %d total"
                reachableFiles.Count
                parseOnly.SourceFiles.Length
            Location = None
        }

        // Step 6: Type-check ONLY reachable files (the key optimization!)
        let! projectResults = getSelectiveCheckResults parseOnly.Context reachableFiles

        if projectResults.CheckResults.HasCriticalErrors then
            diagnostics.Add {
                Severity = DiagError
                Message = "Project has critical compilation errors"
                Location = Some projectPath
            }
            return {
                Success = false
                ProjectResults = Some projectResults
                ProgramSemanticGraph = None
                ReachabilityAnalysis = None
                BakerResult = None
                Diagnostics = List.ofSeq diagnostics
            }
        else
            // Step 7: Generate intermediate outputs
            if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                writeSymbolicAst projectResults.ParseResults config.IntermediatesDir.Value
                writeProjectSummary projectResults config.IntermediatesDir.Value

            // Step 8: Build full PSG with symbol correlation (on reduced file set)
            if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                enableNanopassIntermediates config.IntermediatesDir.Value

            let psg = buildProgramSemanticGraph projectResults.CheckResults projectResults.ParseResults

            disableNanopassIntermediates()

            if psg.Nodes.Count = 0 then
                diagnostics.Add {
                    Severity = DiagError
                    Message = "PSG construction resulted in empty graph"
                    Location = Some projectPath
                }
                return {
                    Success = false
                    ProjectResults = Some projectResults
                    ProgramSemanticGraph = None
                    ReachabilityAnalysis = None
                    BakerResult = None
                    Diagnostics = List.ofSeq diagnostics
                }
            else
                if psg.EntryPoints.Length = 0 then
                    diagnostics.Add {
                        Severity = Warning
                        Message = "PSG has no entry points detected"
                        Location = None
                    }

                // Step 8.5: Extract SRTP call relationships (before reachability!)
                let srtpResult = Core.PSG.Nanopass.ExtractSRTPEdges.run projectResults.CheckResults
                let additionalCalls =
                    if Map.isEmpty srtpResult.AdditionalCalls then None
                    else Some srtpResult.AdditionalCalls

                // Step 9: Full (symbol-aware) reachability with SRTP-discovered calls
                let reachabilityResult = performReachabilityAnalysis psg additionalCalls

                // Step 9.5: Run enrichment nanopasses on the NARROWED graph
                let enrichedPSG = runEnrichmentPasses reachabilityResult.MarkedPSG projectResults.CheckResults

                // Step 10: Generate debug outputs
                if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                    let correlationContext = createContext projectResults.CheckResults
                    let correlations =
                        projectResults.ParseResults
                        |> Array.collect (fun pr -> correlateFile pr.ParseTree correlationContext)
                    // Generate stats (result not currently used but function may have side effects)
                    generateStats correlationContext correlations |> ignore

                    generateCorrelationDebugOutput correlations config.IntermediatesDir.Value
                    writePSGSummary psg config.IntermediatesDir.Value
                    generatePSGDebugOutput enrichedPSG config.IntermediatesDir.Value

                    let prunedSymbols = generatePrunedSymbolData enrichedPSG reachabilityResult
                    let comparisonData = generateComparisonData enrichedPSG reachabilityResult
                    let callGraphData = generateCallGraphData reachabilityResult
                    let libraryBoundaryData = generateLibraryBoundaryData reachabilityResult

                    writeJsonAsset config.IntermediatesDir.Value "psg.pruned.symbols.json" prunedSymbols
                    writeJsonAsset config.IntermediatesDir.Value "reachability.analysis.json" comparisonData
                    writeJsonAsset config.IntermediatesDir.Value "psg.callgraph.pruned.json" callGraphData
                    writeJsonAsset config.IntermediatesDir.Value "library.boundaries.json" libraryBoundaryData

                // Step 11: Validate native types
                let nativeValidation = validateReachable enrichedPSG

                if nativeValidation.HasErrors then
                    nativeValidation.Errors |> List.iter (fun err ->
                        diagnostics.Add {
                            Severity = DiagError
                            Message = err.Message
                            Location = Some (sprintf "Node %s (%s)" err.NodeId err.SyntaxKind)
                        })

                    return {
                        Success = false
                        ProjectResults = Some projectResults
                        ProgramSemanticGraph = Some enrichedPSG
                        ReachabilityAnalysis = Some reachabilityResult
                        BakerResult = None
                        Diagnostics = List.ofSeq diagnostics
                    }
                else
                    // Step 12: Baker enrichment
                    let bakerResult = Baker.Baker.enrich enrichedPSG projectResults.CheckResults

                    diagnostics.Add {
                        Severity = Info
                        Message = sprintf "Baker: %d member bodies, %d correlated with PSG"
                            bakerResult.Statistics.TotalMembers
                            bakerResult.Statistics.MembersCorrelatedWithPSG
                        Location = None
                    }

                    return {
                        Success = true
                        ProjectResults = Some projectResults
                        ProgramSemanticGraph = Some enrichedPSG
                        ReachabilityAnalysis = Some reachabilityResult
                        BakerResult = Some bakerResult
                        Diagnostics = List.ofSeq diagnostics
                    }

    with ex ->
        diagnostics.Add {
            Severity = DiagError
            Message = sprintf "Optimized pipeline failed: %s" ex.Message
            Location = Some projectPath
        }
        return {
            Success = false
            ProjectResults = None
            ProgramSemanticGraph = None
            ReachabilityAnalysis = None
            BakerResult = None
            Diagnostics = List.ofSeq diagnostics
        }
}