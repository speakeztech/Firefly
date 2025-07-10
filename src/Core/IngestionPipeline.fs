module Core.IngestionPipeline

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization
open FSharp.Compiler.CodeAnalysis
open Core.FCS.ProjectContext
open Core.FCS.SymbolAnalysis
open Core.PSG.Types
open Core.PSG.Builder
open Core.PSG.Correlation
open Core.PSG.DebugOutput
open Core.Analysis.CouplingCohesion
open Core.Analysis.Reachability
open Core.Analysis.MemoryLayout
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
    CouplingAnalysis: CouplingAnalysisResult option
    ReachabilityAnalysis: LibraryAwareReachability option
    MemoryLayout: LayoutStrategy option
    Diagnostics: Diagnostic list
}

and CouplingAnalysisResult = {
    Components: CodeComponent list
    Couplings: Coupling list
    Report: {| TotalUnits: int; ComponentCount: int; AverageCohesion: float |}
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

/// Run the ingestion pipeline
let runPipeline (projectPath: string) (config: PipelineConfig) : Async<PipelineResult> = async {
    let diagnostics = ResizeArray<Diagnostic>()
    
    try
        // Step 0: Clean intermediates directory if enabled
        if config.OutputIntermediates && config.IntermediatesDir.IsSome then
            printfn "[IntermediateGeneration] Preparing intermediates directory..."
            prepareIntermediatesDirectory config.IntermediatesDir
        
        // Step 1: Load project
        printfn "[Pipeline] Loading project: %s" projectPath
        let! ctx = loadProject projectPath config.CacheStrategy
        
        // Step 2: Get complete project results
        printfn "[Pipeline] Analyzing project with FCS..."
        let! projectResults = getProjectResults ctx
        
        // Write intermediate files if enabled
        if config.OutputIntermediates && config.IntermediatesDir.IsSome then
            // Write symbolic AST
            printfn "[IntermediateGeneration] Writing symbolic AST..."
            writeSymbolicAst projectResults.ParseResults config.IntermediatesDir.Value
        
            // Write type-checked results summary
            let typeCheckData = {|
                Files = projectResults.CompilationOrder
                SymbolCount = projectResults.SymbolUses.Length
                Timestamp = DateTime.UtcNow
            |}
            let typeCheckJson = JsonSerializer.Serialize(typeCheckData, jsonOptions)
            let typeCheckPath = Path.Combine(config.IntermediatesDir.Value, "typeChecked.json")
            writeFileToPath typeCheckPath typeCheckJson
            printfn "  Wrote %s (%d bytes)" (Path.GetFileName typeCheckPath) typeCheckJson.Length
        
        // Continue with rest of pipeline...
        if projectResults.CheckResults.HasCriticalErrors then
            diagnostics.Add {
                Severity = Error
                Message = "Project has critical errors"
                Location = Some projectPath
            }
            return {
                Success = false
                ProjectResults = Some projectResults
                ProgramSemanticGraph = None
                CouplingAnalysis = None
                ReachabilityAnalysis = None
                MemoryLayout = None
                Diagnostics = List.ofSeq diagnostics
            }
        else
            // Step 3: Extract symbol relationships
            printfn "[Pipeline] Extracting symbol relationships..."
            let relationships = extractRelationships projectResults.SymbolUses
            
            // Step 4: Build Program Semantic Graph
            printfn "[PSG] Building Program Semantic Graph..."
            let psgResult = buildProgramSemanticGraph projectResults.CheckResults projectResults.ParseResults
            
            match psgResult with
            | PSGResult.Failure errors ->
                errors |> List.iter (fun error ->
                    diagnostics.Add {
                        Severity = Error
                        Message = error.Message
                        Location = error.Location |> Option.map (fun r -> r.FileName)
                    }
                )
                return {
                    Success = false
                    ProjectResults = Some projectResults
                    ProgramSemanticGraph = None
                    CouplingAnalysis = None
                    ReachabilityAnalysis = None
                    MemoryLayout = None
                    Diagnostics = List.ofSeq diagnostics
                }
                
            | PSGResult.Success psg ->
                // Validate PSG structure
                match validateGraph psg with
                | PSGResult.Failure validationErrors ->
                    validationErrors |> List.iter (fun error ->
                        diagnostics.Add {
                            Severity = Warning
                            Message = sprintf "PSG validation: %s" error.Message
                            Location = None
                        }
                    )
                | _ -> ()
                
                // Generate PSG debug outputs if intermediates enabled
                if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                    // Create correlation context for statistics
                    let correlationContext = createContext projectResults.CheckResults
                    
                    // Collect all correlations from parsed files
                    let correlations = 
                        projectResults.ParseResults
                        |> Array.collect (fun pr -> 
                            correlateFile pr.ParseTree correlationContext
                        )
                    
                    // Generate correlation statistics
                    let stats = generateStats correlationContext correlations
                    
                    // Generate all debug outputs
                    generateDebugOutputs psg correlations stats config.IntermediatesDir.Value
                    
                    // Also write a PSG summary
                    let psgSummary = {|
                        NodeCount = psg.Nodes.Count
                        EdgeCount = psg.Edges.Length
                        EntryPoints = psg.EntryPoints |> List.map (fun ep -> ep.Value)
                        SymbolCount = psg.SymbolTable.Count
                        FileCount = psg.SourceFiles.Count
                    |}
                    let psgSummaryPath = Path.Combine(config.IntermediatesDir.Value, "psg.summary.json")
                    writeFileToPath psgSummaryPath (JsonSerializer.Serialize(psgSummary, jsonOptions))
                
                printfn "[PSG] PSG construction complete: %d nodes, %d edges, %d entry points" 
                    psg.Nodes.Count psg.Edges.Length psg.EntryPoints.Length
                
                // Step 5: Coupling/Cohesion Analysis (if enabled)
                let couplingAnalysis = 
                    if config.EnableCouplingAnalysis then
                        printfn "[Analysis] Performing coupling/cohesion analysis..."
                        // TODO: Implement coupling analysis using PSG
                        None
                    else
                        None
                
                // Step 6: Reachability Analysis with Library Boundary Awareness
                let reachabilityAnalysis =
                    printfn "[Analysis] Performing reachability analysis with library boundaries..."
                    
                    // Run enhanced reachability analysis
                    let reachabilityResult = analyzeReachabilityWithBoundaries projectResults.SymbolUses
                    
                    // Log analysis results
                    printfn "[Analysis] Reachability complete: %d/%d symbols reachable (%.1f%% eliminated)" 
                        reachabilityResult.PruningStatistics.ReachableSymbols
                        reachabilityResult.PruningStatistics.TotalSymbols
                        ((float reachabilityResult.PruningStatistics.EliminatedSymbols / float reachabilityResult.PruningStatistics.TotalSymbols) * 100.0)
                    
                    // Generate debug assets if intermediates enabled
                    if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                        printfn "[DebugAssets] Generating pruned PSG debug assets..."
                        
                        // Generate all debug data structures (logic stays here, near the analysis)
                        let debugAssets = generatePrunedPSGAssets projectResults.SymbolUses reachabilityResult
                        
                        // Simple IO calls to write the prepared data
                        writeJsonAsset config.IntermediatesDir.Value "psg.corr.pruned.json" debugAssets.CorrelationData
                        writeJsonAsset config.IntermediatesDir.Value "reachability.analysis.json" debugAssets.ComparisonData  
                        writeJsonAsset config.IntermediatesDir.Value "psg.callgraph.pruned.json" debugAssets.CallGraphData
                        writeJsonAsset config.IntermediatesDir.Value "library.boundaries.json" debugAssets.LibraryBoundaryData
                        
                        printfn "[DebugAssets] Pruned PSG debug assets written successfully"
                    
                    Some reachabilityResult
                
                // Step 7: Memory Layout Analysis (if enabled)
                let memoryLayout =
                    if config.EnableMemoryOptimization then
                        printfn "[Analysis] Determining memory layout..."
                        // TODO: Implement memory layout using PSG and coupling data
                        None
                    else
                        None
                
                // Step 8: Generate MLIR (placeholder - will use PSG)
                printfn "[MLIRGeneration] Generating MLIR dialects..."
                if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                    let mlirContent = sprintf ";; MLIR generation from PSG with %d nodes\nmodule @main {\n  func.func @main() -> i32 {\n    %%0 = arith.constant 0 : i32\n    return %%0 : i32\n  }\n}" psg.Nodes.Count
                    let mlirPath = Path.Combine(config.IntermediatesDir.Value, "output.mlir")
                    writeFileToPath mlirPath mlirContent
                
                // Step 9: Generate LLVM (placeholder)
                printfn "[LLVMGeneration] Lowering MLIR to LLVM IR..."
                if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                    let llvmContent = "; LLVM IR placeholder\ndefine i32 @main() {\n  ret i32 0\n}"
                    let llvmPath = Path.Combine(config.IntermediatesDir.Value, "output.ll")
                    writeFileToPath llvmPath llvmContent
                
                // Return success with PSG and reachability analysis
                return {
                    Success = true
                    ProjectResults = Some projectResults
                    ProgramSemanticGraph = Some psg
                    CouplingAnalysis = couplingAnalysis
                    ReachabilityAnalysis = reachabilityAnalysis
                    MemoryLayout = memoryLayout
                    Diagnostics = List.ofSeq diagnostics
                }
            
    with ex ->
        diagnostics.Add {
            Severity = Error
            Message = sprintf "Pipeline error: %s" ex.Message
            Location = Some projectPath
        }
        return {
            Success = false
            ProjectResults = None
            ProgramSemanticGraph = None
            CouplingAnalysis = None
            ReachabilityAnalysis = None
            MemoryLayout = None
            Diagnostics = List.ofSeq diagnostics
        }
}