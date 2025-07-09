module Core.IngestionPipeline

open System
open System.IO
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Symbols
open FSharp.Compiler.Syntax
open Core.FCS.ProjectContext
open Core.FCS.SymbolAnalysis
open Core.FCS.TypedASTAccess
open Core.Analysis.CouplingCohesion
open Core.Analysis.Reachability
open Core.Analysis.MemoryLayout
open Core.Templates.TemplateTypes
open Core.Templates.TemplateLoader
open Core.Meta.AlloyHints
open Core.Utilities.IntermediateWriter

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
        // Ensure intermediates directory exists if needed
        if config.OutputIntermediates && config.IntermediatesDir.IsSome then
            Directory.CreateDirectory(config.IntermediatesDir.Value) |> ignore
        
        // Step 1: Load project with FCS
        printfn "[Pipeline] Loading project: %s" projectPath
        let! ctx = loadProject projectPath config.CacheStrategy
        
        // Step 2: Get complete project results
        printfn "[Pipeline] Analyzing project with FCS..."
        let! projectResults = getProjectResults ctx
        
        // Write parsed AST immediately with full details
        if config.OutputIntermediates && config.IntermediatesDir.IsSome then
            // Write summary
            let parseData = {|
                files = projectResults.ParseResults |> Array.map (fun pr -> {|
                    fileName = pr.FileName
                    hasErrors = pr.Diagnostics.Length > 0
                    diagnosticCount = pr.Diagnostics.Length
                |})
                parseCount = projectResults.ParseResults.Length
            |}
            let parseJson = System.Text.Json.JsonSerializer.Serialize(parseData, 
                System.Text.Json.JsonSerializerOptions(WriteIndented = true))
            let parsePath = Path.Combine(config.IntermediatesDir.Value, "parse.ast.json")
            writeFileToPath parsePath parseJson
            
            // Write detailed AST for each file
            projectResults.ParseResults
            |> Array.iter (fun pr ->
                match pr.ParseTree with
                | ParsedInput.ImplFile(ParsedImplFileInput(fileName, isScript, qualName, _, _, modules, _, _, _)) ->
                    let detailedAst = {|
                        fileName = fileName
                        isScript = isScript
                        qualifiedName = qualName
                        modules = 
                            modules |> List.map (fun m ->
                                match m with
                                | SynModuleOrNamespace(longId, isRecursive, kind, decls, _, attrs, _, _, _) ->
                                    {|
                                        name = longId |> List.map (fun id -> id.idText) |> String.concat "."
                                        isRecursive = isRecursive
                                        isModule = match kind with SynModuleOrNamespaceKind.NamedModule -> true | _ -> false
                                        declarationCount = decls.Length
                                        declarations = 
                                            decls |> List.map (fun decl ->
                                                match decl with
                                                | SynModuleDecl.Let(isRec, bindings, _) ->
                                                    {| 
                                                        kind = "Let"
                                                        isRecursive = isRec
                                                        bindingCount = bindings.Length
                                                        bindings = bindings |> List.map (fun binding ->
                                                            match binding with
                                                            | SynBinding(_, _, _, _, _, _, _, pat, _, _, _, _, _) ->
                                                                match pat with
                                                                | SynPat.Named(synIdent, _, _, _) -> 
                                                                    match synIdent with
                                                                    | SynIdent(ident, _) -> {| name = ident.idText; pattern = "Named" |}
                                                                | SynPat.LongIdent(longIdentPat, _, _, _, _, _) ->
                                                                    let ids = longIdentPat.LongIdent  // This is already LongIdent (Ident list)
                                                                    {| name = ids |> List.map (fun id -> id.idText) |> String.concat "."; pattern = "LongIdent" |}
                                                                | _ -> 
                                                                    {| name = "<complex>"; pattern = pat.GetType().Name |})
                                                    |}
                                                | SynModuleDecl.Types(typeDefns, _) ->
                                                    {| 
                                                        kind = "Types"
                                                        isRecursive = false
                                                        bindingCount = typeDefns.Length
                                                        bindings = typeDefns |> List.map (fun td ->
                                                            match td with
                                                            | SynTypeDefn(componentInfo, _, _, _, _, _) ->
                                                                match componentInfo with
                                                                | SynComponentInfo(_, _, _, longId, _, _, _, _) ->
                                                                    // longId is already LongIdent (Ident list)
                                                                    {| name = longId |> List.map (fun id -> id.idText) |> String.concat "."; pattern = "TypeDefn" |})
                                                    |}
                                                | SynModuleDecl.NestedModule(componentInfo, _, _, _, _, _) ->
                                                    match componentInfo with
                                                    | SynComponentInfo(_, _, _, longId, _, _, _, _) ->
                                                        // longId is already LongIdent (Ident list)
                                                        {| 
                                                            kind = "NestedModule"
                                                            isRecursive = false
                                                            bindingCount = 0
                                                            bindings = [{| name = longId |> List.map (fun id -> id.idText) |> String.concat "."; pattern = "Module" |}]
                                                        |}
                                                | SynModuleDecl.Open(target, _) ->
                                                    let openName = 
                                                        match target with
                                                        | SynOpenDeclTarget.ModuleOrNamespace(longId, _) ->
                                                            longId |> List.map (fun id -> id.idText) |> String.concat "."
                                                        | _ -> "<type>"
                                                    {| 
                                                        kind = "Open"
                                                        isRecursive = false
                                                        bindingCount = 0
                                                        bindings = [{| name = openName; pattern = "Open" |}]
                                                    |}
                                                | _ ->
                                                    {| 
                                                        kind = decl.GetType().Name
                                                        isRecursive = false
                                                        bindingCount = 0
                                                        bindings = []
                                                    |})
                                    |})
                    |}
                    let fileName = Path.GetFileNameWithoutExtension(pr.FileName)
                    let detailPath = Path.Combine(config.IntermediatesDir.Value, $"{fileName}.parsed.ast.json")
                    let detailJson = System.Text.Json.JsonSerializer.Serialize(detailedAst,
                        System.Text.Json.JsonSerializerOptions(WriteIndented = true))
                    writeFileToPath detailPath detailJson
                | ParsedInput.SigFile(_) -> ()
            )
        
        // Write type-checked results immediately with full details
        if config.OutputIntermediates && config.IntermediatesDir.IsSome then
            // Write summary
            let typeCheckData = {|
                Files = projectResults.CompilationOrder
                SymbolCount = projectResults.SymbolUses.Length
                Timestamp = DateTime.UtcNow
            |}
            let typeCheckJson = System.Text.Json.JsonSerializer.Serialize(typeCheckData,
                System.Text.Json.JsonSerializerOptions(WriteIndented = true))
            let typeCheckPath = Path.Combine(config.IntermediatesDir.Value, "typeChecked.json")
            writeFileToPath typeCheckPath typeCheckJson
            
            // Write detailed typed AST per file
            projectResults.CheckResults.AssemblyContents.ImplementationFiles
            |> List.iter (fun implFile ->
                let detailedTypedAst = {|
                    fileName = implFile.FileName
                    qualifiedName = implFile.QualifiedName
                    hasEntryPoint = 
                        implFile.Declarations
                        |> List.exists (fun decl ->
                            match decl with
                            | FSharpImplementationFileDeclaration.MemberOrFunctionOrValue(mfv, _, _) ->
                                mfv.Attributes
                                |> Seq.exists (fun attr -> 
                                    attr.AttributeType.BasicQualifiedName = "EntryPointAttribute" ||
                                    attr.AttributeType.BasicQualifiedName = "System.EntryPointAttribute")
                            | _ -> false)
                    declarations = 
                        implFile.Declarations
                        |> List.map (fun decl ->
                            match decl with
                            | FSharpImplementationFileDeclaration.Entity(entity, subDecls) ->
                                {|
                                    kind = "Entity"
                                    name = entity.DisplayName
                                    fullName = entity.FullName
                                    isModule = entity.IsFSharpModule
                                    isNamespace = entity.IsNamespace
                                    isClass = entity.IsClass
                                    isUnion = entity.IsFSharpUnion
                                    isRecord = entity.IsFSharpRecord
                                    memberCount = 
                                        try entity.MembersFunctionsAndValues.Count
                                        with _ -> 0
                                    members = 
                                        try
                                            entity.MembersFunctionsAndValues
                                            |> Seq.map (fun m -> {| 
                                                name = m.DisplayName
                                                isFunction = m.IsFunction
                                                isValue = m.IsValue
                                                isMutable = m.IsMutable
                                                paramCount = m.CurriedParameterGroups.Count
                                            |})
                                            |> Seq.toList
                                        with _ -> []
                                    subDeclarationCount = subDecls.Length
                                |}
                            | FSharpImplementationFileDeclaration.MemberOrFunctionOrValue(mfv, args, _) ->
                                {|
                                    kind = "MemberOrFunctionOrValue"
                                    name = mfv.DisplayName
                                    fullName = mfv.FullName
                                    isModule = false
                                    isNamespace = false
                                    isClass = false
                                    isUnion = false
                                    isRecord = false
                                    memberCount = 0
                                    members = []
                                    subDeclarationCount = 0
                                |}
                            | FSharpImplementationFileDeclaration.InitAction(_) ->
                                {|
                                    kind = "InitAction"
                                    name = "<init>"
                                    fullName = "<init>"
                                    isModule = false
                                    isNamespace = false
                                    isClass = false
                                    isUnion = false
                                    isRecord = false
                                    memberCount = 0
                                    members = []
                                    subDeclarationCount = 0
                                |})
                |}
                let fileName = Path.GetFileNameWithoutExtension(implFile.FileName)
                let detailPath = Path.Combine(config.IntermediatesDir.Value, $"{fileName}.typed.ast.json")
                let detailJson = System.Text.Json.JsonSerializer.Serialize(detailedTypedAst,
                    System.Text.Json.JsonSerializerOptions(WriteIndented = true))
                writeFileToPath detailPath detailJson
            )
            
            // Write symbol uses per file
            let symbolUsesByFile = 
                projectResults.SymbolUses
                |> Array.groupBy (fun su -> su.Range.FileName)
            
            symbolUsesByFile
            |> Array.iter (fun (fileName, symbolUses) ->
                let detailedSymbols = {|
                    fileName = fileName
                    symbolCount = symbolUses.Length
                    symbols = 
                        symbolUses
                        |> Array.map (fun su -> {|
                            name = su.Symbol.DisplayName
                            fullName = su.Symbol.FullName
                            kind = su.Symbol.GetType().Name
                            isDefinition = su.IsFromDefinition
                            isFromType = su.IsFromType
                            isFromAttribute = su.IsFromAttribute
                            isFromPattern = su.IsFromPattern
                            location = {|
                                startLine = su.Range.Start.Line
                                startColumn = su.Range.Start.Column
                                endLine = su.Range.End.Line
                                endColumn = su.Range.End.Column
                            |}
                        |})
                        |> Array.truncate 1000  // Limit to prevent huge files
                |}
                let baseFileName = Path.GetFileNameWithoutExtension(fileName)
                let symbolPath = Path.Combine(config.IntermediatesDir.Value, $"{baseFileName}.symbols.json")
                let symbolJson = System.Text.Json.JsonSerializer.Serialize(detailedSymbols,
                    System.Text.Json.JsonSerializerOptions(WriteIndented = true))
                writeFileToPath symbolPath symbolJson
            )
        
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
            
            // Write symbol relationships immediately
            if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                let symbolData = {|
                    totalRelationships = relationships.Length
                    uniqueSymbols = 
                        relationships 
                        |> Array.collect (fun r -> [|r.From; r.To|])
                        |> Array.distinct
                        |> Array.length
                |}
                let symbolJson = System.Text.Json.JsonSerializer.Serialize(symbolData,
                    System.Text.Json.JsonSerializerOptions(WriteIndented = true))
                let symbolPath = Path.Combine(config.IntermediatesDir.Value, "symbols.json")
                writeFileToPath symbolPath symbolJson
            
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
                    let simplifiedReport = {|
                        TotalUnits = report.TotalUnits
                        ComponentCount = report.ComponentCount
                        AverageCohesion = report.AverageCohesion
                    |}
                    
                    // Write coupling analysis immediately
                    if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                        let couplingJson = System.Text.Json.JsonSerializer.Serialize(simplifiedReport,
                            System.Text.Json.JsonSerializerOptions(WriteIndented = true))
                        let couplingPath = Path.Combine(config.IntermediatesDir.Value, "coupling.json")
                        writeFileToPath couplingPath couplingJson
                    
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
            
            // Get elimination opportunities for report
            let eliminationOpportunities = 
                match couplingAnalysis with
                | Some ca -> identifyEliminationOpportunities basicReachability ca.Components projectResults.SymbolUses
                | None -> identifyEliminationOpportunities basicReachability [] projectResults.SymbolUses
            
            // Write reachability analysis immediately
            if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                let reachReport = generateReport enhancedReachability eliminationOpportunities
                let reachJson = System.Text.Json.JsonSerializer.Serialize(reachReport,
                    System.Text.Json.JsonSerializerOptions(WriteIndented = true))
                let reachPath = Path.Combine(config.IntermediatesDir.Value, "reachability.json")
                writeFileToPath reachPath reachJson
            
            // Step 6: Memory Layout Optimization (if enabled)
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
                            
                            // Write memory layout immediately
                            if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                                let layoutReport = generateLayoutReport layout []
                                let layoutJson = System.Text.Json.JsonSerializer.Serialize(layoutReport,
                                    System.Text.Json.JsonSerializerOptions(WriteIndented = true))
                                let layoutPath = Path.Combine(config.IntermediatesDir.Value, "memory_layout.json")
                                writeFileToPath layoutPath layoutJson
                            
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
            
            // Done - all intermediates have been written immediately
            if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                printfn "[Pipeline] All intermediate files written to: %s" config.IntermediatesDir.Value
            
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