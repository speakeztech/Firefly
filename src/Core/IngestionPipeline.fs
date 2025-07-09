module Core.IngestionPipeline

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization
open FSharp.Compiler.Syntax
open FSharp.Compiler.Text
open FSharp.Compiler.CodeAnalysis  // Required for FSharpParseFileResults
open Core.FCS.ProjectContext
open Core.FCS.SymbolAnalysis
open Core.Analysis.CouplingCohesion
open Core.Analysis.Reachability
open Core.Analysis.MemoryLayout
open Core.Utilities.IntermediateWriter

/// Configure JSON serialization with F# support
let private createJsonOptions() =
    let options = JsonSerializerOptions(WriteIndented = true)
    options.Converters.Add(JsonFSharpConverter())
    options

/// Global JSON options for the pipeline
let private jsonOptions = createJsonOptions()

/// Extract the string value from QualifiedNameOfFile
let private extractQualifiedName (qualName: QualifiedNameOfFile) =
    match qualName with
    | QualifiedNameOfFile ident -> ident.idText

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
    CacheStrategy = Conservative
    TemplateName = None
    CustomTemplateDir = None
    EnableCouplingAnalysis = true
    EnableMemoryOptimization = true
    OutputIntermediates = true
    IntermediatesDir = None
}

/// Convert range to JSON-friendly format
let private rangeToJson (range: range) =
    {| 
        fileName = range.FileName
        start = {| line = range.Start.Line; column = range.Start.Column |}
        ``end`` = {| line = range.End.Line; column = range.End.Column |}
    |}

/// Convert range to string for output
let private rangeToString (range: range) =
    sprintf "[%d:%d-%d:%d]" range.Start.Line range.Start.Column range.End.Line range.End.Column

/// Write the raw FCS AST using F#'s native representation
let private writeRawFCSOutput (parseResults: FSharpParseFileResults[]) (intermediatesDir: string) =
    parseResults |> Array.iter (fun pr ->
        let baseName = Path.GetFileNameWithoutExtension(pr.FileName)
        let astPath = Path.Combine(intermediatesDir, $"{baseName}.fcs.ast")
        File.WriteAllText(astPath, sprintf "%A" pr.ParseTree)
        printfn "  Wrote %s (%d bytes)" (Path.GetFileName astPath) (FileInfo(astPath).Length)
    )

/// Write module summary in S-expression format
let private writeModuleSummary (parseResults: FSharpParseFileResults[]) (intermediatesDir: string) =
    parseResults |> Array.iter (fun pr ->
        match pr.ParseTree with
        | ParsedInput.ImplFile(ParsedImplFileInput(fileName, _, qualName, _, _, modules, _, _, _)) ->
            let baseName = Path.GetFileNameWithoutExtension(pr.FileName)
            let summaryPath = Path.Combine(intermediatesDir, $"{baseName}.summary.sexp")
            
            use writer = new StreamWriter(summaryPath)
            writer.WriteLine(sprintf ";; Module Summary: %s" fileName)
            writer.WriteLine(sprintf ";; Qualified Name: %s" (extractQualifiedName qualName))
            writer.WriteLine()
            
            modules |> List.iter (fun m ->
                match m with
                | SynModuleOrNamespace(longId, isRec, kind, decls, _, attrs, _, range, _) ->
                    let moduleName = longId |> List.map (fun id -> id.idText) |> String.concat "."
                    writer.WriteLine(sprintf "(module %s" moduleName)
                    writer.WriteLine(sprintf "  :kind %A" kind)
                    writer.WriteLine(sprintf "  :range %s" (rangeToString range))
                    writer.WriteLine(sprintf "  :declarations %d" decls.Length)
                    
                    // Summary statistics
                    let letCount = decls |> List.filter (function SynModuleDecl.Let _ -> true | _ -> false) |> List.length
                    let openCount = decls |> List.filter (function SynModuleDecl.Open _ -> true | _ -> false) |> List.length
                    let typeCount = decls |> List.filter (function SynModuleDecl.Types _ -> true | _ -> false) |> List.length
                    let nestedModuleCount = decls |> List.filter (function SynModuleDecl.NestedModule _ -> true | _ -> false) |> List.length
                    
                    writer.WriteLine(sprintf "  :let-bindings %d" letCount)
                    writer.WriteLine(sprintf "  :open-declarations %d" openCount)
                    writer.WriteLine(sprintf "  :type-definitions %d" typeCount)
                    writer.WriteLine(sprintf "  :nested-modules %d" nestedModuleCount)
                    writer.WriteLine(")")
            )
            
            printfn "  Wrote %s (%d bytes)" (Path.GetFileName summaryPath) (FileInfo(summaryPath).Length)
            
        | ParsedInput.SigFile(_) -> 
            ()
    )

/// Write structured JSON representation of the AST
let private writeStructuredJSON (parseResults: FSharpParseFileResults[]) (intermediatesDir: string) =
    // Summary file
    let parseData = {|
        files = parseResults |> Array.map (fun pr -> {|
            fileName = pr.FileName
            hasErrors = pr.ParseHadErrors
            diagnosticCount = pr.Diagnostics.Length
        |})
        parseCount = parseResults.Length
    |}
    let parseJson = JsonSerializer.Serialize(parseData, jsonOptions)
    let parsePath = Path.Combine(intermediatesDir, "parse.ast.json")
    writeFileToPath parsePath parseJson
    
    // Detailed JSON for each file
    parseResults |> Array.iter (fun pr ->
        match pr.ParseTree with
        | ParsedInput.ImplFile(ParsedImplFileInput(fileName, isScript, qualName, _, _, modules, _, _, _)) ->
            let detailedAst = {|
                fileName = fileName
                isScript = isScript
                qualifiedName = extractQualifiedName qualName
                modules = 
                    modules |> List.map (fun m ->
                        match m with
                        | SynModuleOrNamespace(longId, isRecursive, kind, decls, _, attrs, _, range, _) ->
                            {|
                                name = longId |> List.map (fun id -> id.idText) |> String.concat "."
                                isModule = (match kind with SynModuleOrNamespaceKind.NamedModule -> true | _ -> false)
                                isRecursive = isRecursive
                                range = rangeToJson range
                                declarationCount = decls.Length
                                declarations = decls |> List.map (fun decl ->
                                    match decl with
                                    | SynModuleDecl.Let(isRec, bindings, range) ->
                                        {| 
                                            kind = "Let"
                                            isRecursive = isRec
                                            range = rangeToJson range
                                            bindingCount = bindings.Length
                                            bindings = bindings |> List.map (fun binding ->
                                                match binding with
                                                | SynBinding(_, _, _, _, _, _, _, pat, _, _, range, _, _) ->
                                                    let name = 
                                                        match pat with
                                                        | SynPat.Named(SynIdent(ident, _), _, _, _) -> ident.idText
                                                        | SynPat.LongIdent(SynLongIdent(ids, _, _), _, _, _, _, _) ->
                                                            ids |> List.map (fun id -> id.idText) |> String.concat "."
                                                        | _ -> "<pattern>"
                                                    {|
                                                        pattern = pat.GetType().Name
                                                        name = name
                                                        expression = {| kind = "Expression" |}
                                                    |}
                                            )
                                        |}
                                    
                                    | SynModuleDecl.Open(SynOpenDeclTarget.ModuleOrNamespace(SynLongIdent(ids, _, _), _), range) ->
                                        {| 
                                            kind = "Open"
                                            isRecursive = false
                                            range = rangeToJson range
                                            bindingCount = 0
                                            bindings = []
                                        |}
                                    
                                    | SynModuleDecl.NestedModule(SynComponentInfo(_, _, _, longId, _, _, _, _), _, decls, _, range, _) ->
                                        {| 
                                            kind = "NestedModule"
                                            isRecursive = false
                                            range = rangeToJson range
                                            bindingCount = 0
                                            bindings = []
                                        |}
                                    
                                    | _ ->
                                        {| 
                                            kind = decl.GetType().Name.Replace("SynModuleDecl+", "")
                                            isRecursive = false
                                            range = rangeToJson decl.Range
                                            bindingCount = 0
                                            bindings = []
                                        |}
                                )
                            |}
                    )
            |}
            
            let baseName = Path.GetFileNameWithoutExtension(pr.FileName)
            let detailPath = Path.Combine(intermediatesDir, $"{baseName}.parsed.ast.json")
            let detailJson = JsonSerializer.Serialize(detailedAst, jsonOptions)
            writeFileToPath detailPath detailJson
            
        | ParsedInput.SigFile(_) -> ()
    )

/// Run the ingestion pipeline
let runPipeline (projectPath: string) (config: PipelineConfig) : Async<PipelineResult> = async {
    let diagnostics = ResizeArray<Diagnostic>()
    
    try
        // Step 1: Load project
        printfn "[Pipeline] Loading project: %s" projectPath
        let! ctx = loadProject projectPath config.CacheStrategy
        
        // Step 2: Get complete project results
        printfn "[Pipeline] Analyzing project with FCS..."
        let! projectResults = getProjectResults ctx
        
        // Write parsed AST immediately with full details
        if config.OutputIntermediates && config.IntermediatesDir.IsSome then
            // 1. Raw FCS output - exactly what FCS parsed
            writeRawFCSOutput projectResults.ParseResults config.IntermediatesDir.Value
            
            // 2. Module summary in S-expression format
            writeModuleSummary projectResults.ParseResults config.IntermediatesDir.Value
            
            // 3. Structured JSON format
            writeStructuredJSON projectResults.ParseResults config.IntermediatesDir.Value
        
        // Write type-checked results 
        if config.OutputIntermediates && config.IntermediatesDir.IsSome then
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
                CouplingAnalysis = None
                ReachabilityAnalysis = None
                MemoryLayout = None
                Diagnostics = List.ofSeq diagnostics
            }
        else
            // Step 3: Extract symbol relationships
            printfn "[Pipeline] Extracting symbol relationships..."
            let relationships = extractRelationships projectResults.SymbolUses
            
            // Step 4: Build PSG (placeholder for now)
            printfn "[ASTTransformation] Building Program Semantic Graph..."
            if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                let psgData = {|
                    entryPoints = []
                    functions = 100  // Placeholder
                    modules = 6      // Placeholder
                    types = 50       // Placeholder
                |}
                let psgPath = Path.Combine(config.IntermediatesDir.Value, "semantic.psg.json")
                writeFileToPath psgPath (JsonSerializer.Serialize(psgData, jsonOptions))
            
            // Step 5: Generate MLIR (placeholder)
            printfn "[MLIRGeneration] Generating MLIR dialects..."
            if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                let mlirContent = ";; MLIR generation placeholder\nmodule @main {\n  func.func @main() -> i32 {\n    %0 = arith.constant 0 : i32\n    return %0 : i32\n  }\n}"
                let mlirPath = Path.Combine(config.IntermediatesDir.Value, "output.mlir")
                writeFileToPath mlirPath mlirContent
            
            // Step 6: Generate LLVM (placeholder)
            printfn "[LLVMGeneration] Lowering MLIR to LLVM IR..."
            if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                let llvmContent = "; LLVM IR placeholder\ndefine i32 @main() {\n  ret i32 0\n}"
                let llvmPath = Path.Combine(config.IntermediatesDir.Value, "output.ll")
                writeFileToPath llvmPath llvmContent
            
            // Return success
            return {
                Success = true
                ProjectResults = Some projectResults
                CouplingAnalysis = None  // TODO: Implement
                ReachabilityAnalysis = None  // TODO: Implement
                MemoryLayout = None  // TODO: Implement
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
            CouplingAnalysis = None
            ReachabilityAnalysis = None
            MemoryLayout = None
            Diagnostics = List.ofSeq diagnostics
        }
}