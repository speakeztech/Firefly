module Core.IngestionPipeline

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization
open FSharp.Compiler.Syntax
open FSharp.Compiler.Text
open FSharp.Compiler.CodeAnalysis  // Required for FSharpParseFileResults
open FSharp.Compiler.Symbols
open Core.FCS.ProjectContext
open Core.FCS.SymbolAnalysis
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

/// Write the raw FCS AST using F#'s native representation (symbolic AST)
let private writeRawFCSOutput (parseResults: FSharpParseFileResults[]) (intermediatesDir: string) =
    parseResults |> Array.iter (fun pr ->
        let baseName = Path.GetFileNameWithoutExtension(pr.FileName)
        let astPath = Path.Combine(intermediatesDir, $"{baseName}.fcs.ast")
        File.WriteAllText(astPath, sprintf "%A" pr.ParseTree)
        printfn "  Wrote %s (%d bytes)" (Path.GetFileName astPath) (FileInfo(astPath).Length)
    )

/// Write typed AST from check results
let private writeTypedAST (checkResults: FSharpCheckProjectResults) (intermediatesDir: string) =
    // Get all implementation files from the project
    checkResults.AssemblyContents.ImplementationFiles
    |> List.iter (fun implFile ->
        let baseName = Path.GetFileNameWithoutExtension(implFile.FileName)
        let typedPath = Path.Combine(intermediatesDir, $"{baseName}.typed.ast")
        
        // Write the typed AST representation
        let content = sprintf "%A" implFile
        File.WriteAllText(typedPath, content)
        printfn "  Wrote %s (%d bytes)" (Path.GetFileName typedPath) (FileInfo(typedPath).Length)
    )

/// Write symbol uses with type information
let private writeSymbolUses (symbolUses: FSharpSymbolUse[]) (intermediatesDir: string) =
    // Group symbol uses by file
    let symbolsByFile = 
        symbolUses 
        |> Array.groupBy (fun symbolUse -> symbolUse.FileName)
    
    symbolsByFile
    |> Array.iter (fun (fileName, uses) ->
        let baseName = Path.GetFileNameWithoutExtension(fileName)
        let symbolPath = Path.Combine(intermediatesDir, $"{baseName}.symbols")
        
        let content = 
            uses
            |> Array.map (fun symbolUse ->
                let typeInfo = 
                    match symbolUse.Symbol with
                    | :? FSharpMemberOrFunctionOrValue as mfv -> 
                        try mfv.FullType.Format(symbolUse.DisplayContext) with _ -> "<type unavailable>"
                    | :? FSharpEntity as entity ->
                        entity.DisplayName
                    | _ -> "<no type info>"
                
                sprintf "Symbol: %s\n  Location: %s\n  Type: %s\n  IsFromDefinition: %b\n  IsFromAttribute: %b\n  IsFromComputationExpression: %b\n  IsFromDispatchSlotImplementation: %b\n  IsFromPattern: %b\n  IsFromType: %b\n"
                    symbolUse.Symbol.DisplayName
                    (symbolUse.Range.ToString())
                    typeInfo
                    symbolUse.IsFromDefinition
                    symbolUse.IsFromAttribute
                    symbolUse.IsFromComputationExpression
                    symbolUse.IsFromDispatchSlotImplementation
                    symbolUse.IsFromPattern
                    symbolUse.IsFromType
            )
            |> String.concat "\n"
            
        File.WriteAllText(symbolPath, content)
        printfn "  Wrote %s (%d bytes)" (Path.GetFileName symbolPath) (FileInfo(symbolPath).Length)
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

// Removed writeStructuredJSON - we agreed to discard the simplified JSON representation

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
        
        // Write parsed AST immediately with full details
        if config.OutputIntermediates && config.IntermediatesDir.IsSome then
            // 1. Raw FCS output - exactly what FCS parsed (symbolic AST)
            writeRawFCSOutput projectResults.ParseResults config.IntermediatesDir.Value
            
            // 2. Module summary in S-expression format
            writeModuleSummary projectResults.ParseResults config.IntermediatesDir.Value
            
            // 3. Typed AST from check results
            printfn "[IntermediateGeneration] Writing type-checked AST..."
            writeTypedAST projectResults.CheckResults config.IntermediatesDir.Value
            
            // 4. Symbol uses with type information
            printfn "[IntermediateGeneration] Writing symbol uses..."
            writeSymbolUses projectResults.SymbolUses config.IntermediatesDir.Value
        
        // Write type-checked results summary
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