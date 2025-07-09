module Core.IngestionPipeline

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization
open FSharp.Compiler.CodeAnalysis
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

/// Write the symbolic AST using F#'s native representation 
let private writeSymbolicAst (parseResults: FSharpParseFileResults[]) (intermediatesDir: string) =
    parseResults |> Array.iter (fun pr ->
        let baseName = Path.GetFileNameWithoutExtension(pr.FileName)
        let astPath = Path.Combine(intermediatesDir, $"{baseName}.sym.ast")
        File.WriteAllText(astPath, sprintf "%A" pr.ParseTree)
        printfn "  Wrote %s (%d bytes)" (Path.GetFileName astPath) (FileInfo(astPath).Length)
    )

/// Write typed AST from check results
let private writeTypedAst (checkResults: FSharpCheckProjectResults) (intermediatesDir: string) =
    // Get all implementation files from the project
    checkResults.AssemblyContents.ImplementationFiles
    |> List.iter (fun implFile ->
        let baseName = Path.GetFileNameWithoutExtension(implFile.FileName)
        let typedPath = Path.Combine(intermediatesDir, $"{baseName}.typ.ast")
        
        // Write the complete typed AST representation with all details
        use writer = new StreamWriter(typedPath)
        // Format with same depth and detail as symbolic AST
        let content = sprintf "%A" implFile
        writer.Write(content)
        
        printfn "  Wrote %s (%d bytes)" (Path.GetFileName typedPath) (FileInfo(typedPath).Length)
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
            
            // Write typed AST
            printfn "[IntermediateGeneration] Writing typed AST..."
            writeTypedAst projectResults.CheckResults config.IntermediatesDir.Value
        
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