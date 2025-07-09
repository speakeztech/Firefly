module Core.IngestionPipeline

open System
open System.IO
open FSharp.Compiler.CodeAnalysis
open Core.FCS.ProjectContext
open Core.FCS.SymbolAnalysis
open Core.FCS.TypedASTAccess
open Core.Analysis.CouplingCohesion
open Core.Analysis.Reachability
open Core.Analysis.MemoryLayout
open Core.Templates.TemplateTypes
open Core.Templates.TemplateLoader
open Core.Meta.AlloyHints

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
    CouplingAnalysis: CouplingAnalysisResult option
    ReachabilityAnalysis: EnhancedReachability option
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
    CacheStrategy = Balanced
    TemplateName = None
    CustomTemplateDir = None
    EnableCouplingAnalysis = true
    EnableMemoryOptimization = true
    OutputIntermediates = false
    IntermediatesDir = None
}

/// Run the complete compilation pipeline
let runPipeline (projectPath: string) (config: PipelineConfig) = async {
    let diagnostics = ResizeArray<Diagnostic>()
    
    try
        // Step 1: Load project with FCS
        printfn "[Pipeline] Loading project: %s" projectPath
        let! ctx = loadProject projectPath config.CacheStrategy
        
        // Step 2: Get complete project results
        printfn "[Pipeline] Analyzing project with FCS..."
        let! projectResults = getProjectResults ctx
        
        if projectResults.CheckResults.HasCriticalErrors then
            diagnostics.Add {
                Severity = Error
                Message = "Project has critical errors"
                Location = Some projectPath
            }
            return {
                Success = false
                ProjectResults = Some projectResults
                CouplingAnalysis = None
                ReachabilityAnalysis = None
                MemoryLayout = None
                Diagnostics = List.ofSeq diagnostics
            }
        else
            // Step 3: Extract symbol relationships
            printfn "[Pipeline] Extracting symbol relationships..."
            let relationships = extractRelationships projectResults.SymbolUses
            
            // Step 4: Coupling/Cohesion Analysis (if enabled)
            let couplingAnalysis = 
                if config.EnableCouplingAnalysis then
                    printfn "[Pipeline] Performing coupling/cohesion analysis..."
                    let units = identifySemanticUnits projectResults.SymbolUses relationships
                    let components = detectComponents units relationships 0.5
                    let couplings = 
                        [for u1 in units do
                            for u2 in units do
                                if u1 <> u2 then
                                    yield calculateCoupling u1 u2 relationships]
                        |> List.filter (fun c -> c.Strength > 0.0)
                    
                    let report = generateAnalysisReport components units
                    // Extract only the fields we need from the report
                    let simplifiedReport = {|
                        TotalUnits = report.TotalUnits
                        ComponentCount = report.ComponentCount
                        AverageCohesion = report.AverageCohesion
                    |}
                    Some {
                        Components = components
                        Couplings = couplings
                        Report = simplifiedReport
                    }
                else None
            
            // Step 5: Reachability Analysis
            printfn "[Pipeline] Performing reachability analysis..."
            let basicReachability = analyzeReachability projectResults.SymbolUses relationships
            let enhancedReachability = 
                match couplingAnalysis with
                | Some ca -> analyzeComponentReachability basicReachability ca.Components
                | None -> { BasicResult = basicReachability; ComponentReachability = Map.empty }
            
            // Step 6: Get elimination opportunities (needed for report)
            let eliminationOpportunities = 
                match couplingAnalysis with
                | Some ca -> identifyEliminationOpportunities basicReachability ca.Components projectResults.SymbolUses
                | None -> identifyEliminationOpportunities basicReachability [] projectResults.SymbolUses
            
            // Step 7: Memory Layout Optimization (if enabled)
            let memoryLayout = 
                if config.EnableMemoryOptimization then
                    match config.TemplateName, couplingAnalysis with
                    | Some templateName, Some ca ->
                        printfn "[Pipeline] Optimizing memory layout for platform: %s" templateName
                        let registry = initializeRegistry config.CustomTemplateDir
                        let selector = { 
                            Platform = Some templateName
                            MinMemory = None
                            RequiredCapabilities = []
                            Profile = None 
                        }
                        match selectTemplate selector registry with
                        | Some template ->
                            let layout = calculateMemoryLayout ca.Components ca.Couplings template
                            let violations = validateMemorySafety layout template
                            
                            for v in violations do
                                diagnostics.Add {
                                    Severity = Warning
                                    Message = v.Description
                                    Location = None
                                }
                            
                            Some layout
                        | None ->
                            diagnostics.Add {
                                Severity = Warning
                                Message = $"Template '{templateName}' not found"
                                Location = None
                            }
                            None
                    | _ -> None
                else None
            
            // Step 8: Output intermediates (if enabled)
            if config.OutputIntermediates then
                match config.IntermediatesDir with
                | Some dir ->
                    printfn "[Pipeline] Writing intermediate files to: %s" dir
                    Directory.CreateDirectory(dir) |> ignore
                    
                    // Write coupling analysis
                    match couplingAnalysis with
                    | Some ca ->
                        let couplingJson = System.Text.Json.JsonSerializer.Serialize(ca.Report)
                        File.WriteAllText(Path.Combine(dir, "coupling.json"), couplingJson)
                    | None -> ()
                    
                    // Write reachability analysis
                    let reachReport = generateReport enhancedReachability eliminationOpportunities
                    let reachJson = System.Text.Json.JsonSerializer.Serialize(reachReport)
                    File.WriteAllText(Path.Combine(dir, "reachability.json"), reachJson)
                    
                    // Write memory layout
                    match memoryLayout with
                    | Some layout ->
                        let layoutReport = generateLayoutReport layout []
                        let layoutJson = System.Text.Json.JsonSerializer.Serialize(layoutReport)
                        File.WriteAllText(Path.Combine(dir, "memory_layout.json"), layoutJson)
                    | None -> ()
                | None -> ()
            
            printfn "[Pipeline] Analysis complete"
            return {
                Success = true
                ProjectResults = Some projectResults
                CouplingAnalysis = couplingAnalysis
                ReachabilityAnalysis = Some enhancedReachability
                MemoryLayout = memoryLayout
                Diagnostics = List.ofSeq diagnostics
            }
            
    with ex ->
        diagnostics.Add {
            Severity = Error
            Message = $"Pipeline failed: {ex.Message}"
            Location = None
        }
        return {
            Success = false
            ProjectResults = None
            CouplingAnalysis = None
            ReachabilityAnalysis = None
            MemoryLayout = None
            Diagnostics = List.ofSeq diagnostics
        }
}

/// Generate compilation summary
let generateSummary (result: PipelineResult) =
    let summary = System.Text.StringBuilder()
    
    summary.AppendLine("=== FIREFLY COMPILATION SUMMARY ===") |> ignore
    summary.AppendLine($"Success: {result.Success}") |> ignore
    
    // Project statistics
    match result.ProjectResults with
    | Some pr ->
        summary.AppendLine($"Files: {pr.CompilationOrder.Length}") |> ignore
        summary.AppendLine($"Symbols: {pr.SymbolUses.Length}") |> ignore
    | None -> ()
    
    // Coupling analysis
    match result.CouplingAnalysis with
    | Some ca ->
        summary.AppendLine($"Components: {ca.Report.ComponentCount}") |> ignore
        summary.AppendLine($"Average Cohesion: {ca.Report.AverageCohesion:F2}") |> ignore
    | None -> ()
    
    // Reachability
    match result.ReachabilityAnalysis with
    | Some ra ->
        let basic = ra.BasicResult
        let totalReachable = basic.ReachableSymbols.Count
        let totalUnreachable = basic.UnreachableSymbols.Count
        let total = totalReachable + totalUnreachable
        let rate = 
            if total > 0 then
                float totalUnreachable / float total * 100.0
            else 0.0
        summary.AppendLine($"Dead Code: {rate:F1}%%") |> ignore
    | None -> ()
    
    // Memory layout
    match result.MemoryLayout with
    | Some layout ->
        summary.AppendLine($"Memory Regions: {layout.Regions.Length}") |> ignore
        summary.AppendLine($"Cross-Region Links: {layout.CrossRegionLinks.Length}") |> ignore
    | None -> ()
    
    // Diagnostics
    let errors = result.Diagnostics |> List.filter (fun d -> d.Severity = Error) |> List.length
    let warnings = result.Diagnostics |> List.filter (fun d -> d.Severity = Warning) |> List.length
    summary.AppendLine($"Errors: {errors}, Warnings: {warnings}") |> ignore
    
    summary.ToString()

/// Run pipeline with progress reporting
let runPipelineWithProgress (projectPath: string) (config: PipelineConfig) (progress: string -> unit) = async {
    progress "Starting Firefly compilation pipeline..."
    
    let! result = runPipeline projectPath config
    
    progress "Generating summary..."
    let summary = generateSummary result
    progress summary
    
    return result
}