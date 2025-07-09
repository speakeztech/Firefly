module Dabbit.Pipeline.CompilationOrchestrator

open System
open System.IO
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Text
open Core.XParsec.Foundation
open Core.IngestionPipeline
open Core.FCS.ProjectContext
open Core.Meta.AlloyHints
open Core.Analysis.MemoryLayout
open Core.Utilities.IntermediateWriter
open Dabbit.Pipeline.CompilationTypes

// ===================================================================
// Pipeline Integration Types
// ===================================================================

/// Extended compilation result with MLIR/LLVM outputs
type CompilationResult = {
    Success: bool
    Diagnostics: CompilerError list
    Statistics: CompilationStatistics
    Intermediates: IntermediateOutputs
    MLIROutput: string option
    LLVMOutput: string option
}

/// Intermediate file outputs
and IntermediateOutputs = {
    TypeCheckedAST: string option
    ReachabilityAnalysis: string option
    PSGRepresentation: string option
    MLIRDialects: string option
    LLVMAssembly: string option
}

/// Empty intermediates for initialization
let emptyIntermediates = {
    TypeCheckedAST = None
    ReachabilityAnalysis = None
    PSGRepresentation = None
    MLIRDialects = None
    LLVMAssembly = None
}

// ===================================================================
// Pipeline Configuration Bridge
// ===================================================================

/// Convert ingestion pipeline config to compilation config
let createPipelineConfig (config: CompilationConfig) (intermediatesDir: string option) : PipelineConfig = {
    CacheStrategy = Balanced
    TemplateName = None
    CustomTemplateDir = None
    EnableCouplingAnalysis = config.EnableReachabilityAnalysis
    EnableMemoryOptimization = config.EnableStackAllocation
    OutputIntermediates = config.PreserveIntermediateASTs
    IntermediatesDir = intermediatesDir
}

// ===================================================================
// Type Conversions
// ===================================================================

/// Convert IngestionPipeline.DiagnosticSeverity to CompilationTypes.ErrorSeverity
let convertSeverity (severity: DiagnosticSeverity) : ErrorSeverity =
    match severity with
    | DiagnosticSeverity.Error -> ErrorSeverity.Error
    | DiagnosticSeverity.Warning -> ErrorSeverity.Warning
    | DiagnosticSeverity.Info -> ErrorSeverity.Info

/// Convert IngestionPipeline.Diagnostic to CompilerError
let convertDiagnostic (diag: Diagnostic) : CompilerError = {
    Phase = "Ingestion"
    Message = diag.Message
    Location = diag.Location
    Severity = convertSeverity diag.Severity
}

/// Convert FireflyError to CompilerError
let convertFireflyError (error: FireflyError) : CompilerError =
    match error with
    | SyntaxError(pos, msg, _) -> {
        Phase = "Syntax"
        Message = msg
        Location = Some $"{pos.File}:{pos.Line}:{pos.Column}"
        Severity = ErrorSeverity.Error
      }
    | ConversionError(phase, _, _, msg) -> {
        Phase = phase
        Message = msg
        Location = None
        Severity = ErrorSeverity.Error
      }
    | TypeCheckError(construct, msg, pos) -> {
        Phase = "TypeCheck"
        Message = $"{construct}: {msg}"
        Location = Some $"{pos.File}:{pos.Line}:{pos.Column}"
        Severity = ErrorSeverity.Error
      }
    | InternalError(phase, msg, _) -> {
        Phase = phase
        Message = msg
        Location = None
        Severity = ErrorSeverity.Error
      }
    | ParseError(pos, msg) -> {
        Phase = "Parse"
        Message = msg
        Location = Some $"{pos.File}:{pos.Line}:{pos.Column}"
        Severity = ErrorSeverity.Error
      }
    | DependencyResolutionError(symbol, msg) -> {
        Phase = "DependencyResolution"
        Message = $"{symbol}: {msg}"
        Location = None
        Severity = ErrorSeverity.Error
      }

// ===================================================================
// Progress Reporting
// ===================================================================

/// Report compilation phase progress
let reportPhase (progress: ProgressCallback) (phase: CompilationPhase) (message: string) =
    progress phase message

// ===================================================================
// AST Transformation Pipeline
// ===================================================================

/// Transform FCS AST to PSG representation
let transformToSemanticGraph 
    (projectResults: ProjectResults) 
    (config: PipelineConfig) 
    (progress: ProgressCallback) = async {
    
    reportPhase progress CompilationPhase.ASTTransformation "Building Program Semantic Graph..."
    
    // TODO: Implement actual PSG construction
    // For now, return a placeholder with structured JSON
    let psgData = {|
        modules = projectResults.CompilationOrder.Length
        types = 50  // placeholder
        functions = 100  // placeholder
        entryPoints = []
    |}
    
    let psgJson = System.Text.Json.JsonSerializer.Serialize(psgData, 
        System.Text.Json.JsonSerializerOptions(WriteIndented = true))
    
    return Success psgJson
}

/// Generate MLIR from semantic graph
let generateMLIR 
    (psgData: string) 
    (memoryLayout: LayoutStrategy option)
    (config: PipelineConfig) 
    (progress: ProgressCallback) = async {
    
    reportPhase progress CompilationPhase.MLIRGeneration "Generating MLIR dialects..."
    
    // TODO: Implement actual MLIR generation
    // For now, return a placeholder
    let mlir = """
module {
    func.func @main() -> i32 {
        %0 = arith.constant 0 : i32
        return %0 : i32
    }
}
"""
    return Success mlir
}

/// Generate LLVM from MLIR
let generateLLVM 
    (mlirCode: string) 
    (config: PipelineConfig) 
    (progress: ProgressCallback) = async {
    
    reportPhase progress CompilationPhase.LLVMGeneration "Lowering MLIR to LLVM IR..."
    
    // TODO: Implement actual LLVM generation via MLIR lowering
    // For now, return a placeholder
    let llvm = """
define i32 @main() {
entry:
    ret i32 0
}
"""
    return Success llvm
}

// ===================================================================
// Intermediate File Writing
// ===================================================================

/// Write intermediate JSON data
let writeIntermediateJson (filePath: string) (data: obj) =
    let json = System.Text.Json.JsonSerializer.Serialize(data, 
        System.Text.Json.JsonSerializerOptions(WriteIndented = true))
    writeFileToPath filePath json |> ignore

// ===================================================================
// Main Compilation Entry Points
// ===================================================================

/// Compile a project file with deep FCS integration
let compileProject 
    (projectPath: string) 
    (outputPath: string)
    (projectOptions: FSharpProjectOptions)
    (compilationConfig: CompilationConfig)
    (intermediatesDir: string option)
    (progress: ProgressCallback) = async {
    
    let startTime = DateTime.UtcNow
    let mutable intermediates = emptyIntermediates
    
    // Convert CompilationConfig to PipelineConfig
    let pipelineConfig = createPipelineConfig compilationConfig intermediatesDir
    
    try
        // Step 1: Run FCS ingestion pipeline
        reportPhase progress CompilationPhase.Initialization "Starting FCS ingestion pipeline..."
        let! pipelineResult = runPipeline projectPath pipelineConfig
        
        if not pipelineResult.Success then
            let errors = pipelineResult.Diagnostics |> List.map convertDiagnostic
            return {
                Success = false
                Diagnostics = errors
                Statistics = CompilationStatistics.empty
                Intermediates = intermediates
                MLIROutput = None
                LLVMOutput = None
            }
        else
        
        let projectResults = pipelineResult.ProjectResults.Value
        
        // Write type-checked AST if requested
        if pipelineConfig.OutputIntermediates && intermediatesDir.IsSome then
            reportPhase progress CompilationPhase.IntermediateGeneration "Writing type-checked AST..."
            let astPath = Path.Combine(intermediatesDir.Value, "typeChecked.json")
            
            // Create simplified representation for serialization
            let astData = {|
                Files = projectResults.CompilationOrder
                SymbolCount = projectResults.SymbolUses.Length
                Timestamp = DateTime.UtcNow
            |}
            writeIntermediateJson astPath astData
            intermediates <- { intermediates with TypeCheckedAST = Some astPath }
        
        // Step 2: Transform to PSG
        let! psgResult = transformToSemanticGraph projectResults pipelineConfig progress
        
        match psgResult with
        | CompilerFailure errors -> 
            return {
                Success = false
                Diagnostics = errors |> List.map convertFireflyError
                Statistics = CompilationStatistics.empty
                Intermediates = intermediates
                MLIROutput = None
                LLVMOutput = None
            }
        | Success psgData ->
            
            // Write PSG if requested
            if pipelineConfig.OutputIntermediates && intermediatesDir.IsSome then
                let psgPath = Path.Combine(intermediatesDir.Value, "semantic.psg.json")
                writeFileToPath psgPath psgData |> ignore
                intermediates <- { intermediates with PSGRepresentation = Some psgPath }
            
            // Step 3: Generate MLIR
            let! mlirResult = generateMLIR psgData pipelineResult.MemoryLayout pipelineConfig progress
            
            match mlirResult with
            | CompilerFailure errors ->
                return {
                    Success = false
                    Diagnostics = errors |> List.map convertFireflyError
                    Statistics = CompilationStatistics.empty
                    Intermediates = intermediates
                    MLIROutput = None
                    LLVMOutput = None
                }
            | Success mlirCode ->
                
                // Write MLIR if requested
                if pipelineConfig.OutputIntermediates && intermediatesDir.IsSome then
                    let mlirPath = Path.Combine(intermediatesDir.Value, "output.mlir")
                    writeFileToPath mlirPath mlirCode |> ignore
                    intermediates <- { intermediates with MLIRDialects = Some mlirPath }
                
                // Step 4: Generate LLVM
                let! llvmResult = generateLLVM mlirCode pipelineConfig progress
                
                match llvmResult with
                | CompilerFailure errors ->
                    return {
                        Success = false
                        Diagnostics = errors |> List.map convertFireflyError
                        Statistics = CompilationStatistics.empty
                        Intermediates = intermediates
                        MLIROutput = Some mlirCode
                        LLVMOutput = None
                    }
                | Success llvmCode ->
                    
                    // Write LLVM if requested
                    if pipelineConfig.OutputIntermediates && intermediatesDir.IsSome then
                        let llvmPath = Path.Combine(intermediatesDir.Value, "output.ll")
                        writeFileToPath llvmPath llvmCode |> ignore
                        intermediates <- { intermediates with LLVMAssembly = Some llvmPath }
                    
                    // Calculate statistics
                    let endTime = DateTime.UtcNow
                    let stats = {
                        TotalFiles = projectResults.CompilationOrder.Length
                        TotalSymbols = projectResults.SymbolUses.Length
                        ReachableSymbols = 
                            match pipelineResult.ReachabilityAnalysis with
                            | Some ra -> ra.BasicResult.ReachableSymbols.Count
                            | None -> projectResults.SymbolUses.Length
                        EliminatedSymbols = 
                            match pipelineResult.ReachabilityAnalysis with
                            | Some ra -> ra.BasicResult.UnreachableSymbols.Count
                            | None -> 0
                        CompilationTimeMs = (endTime - startTime).TotalMilliseconds
                    }
                    
                    reportPhase progress CompilationPhase.Finalization "Compilation completed successfully"
                    
                    return {
                        Success = true
                        Diagnostics = []
                        Statistics = stats
                        Intermediates = intermediates
                        MLIROutput = Some mlirCode
                        LLVMOutput = Some llvmCode
                    }
        
    with ex ->
        return {
            Success = false
            Diagnostics = [{
                Phase = "Compilation"
                Message = ex.Message
                Location = None
                Severity = ErrorSeverity.Error
            }]
            Statistics = CompilationStatistics.empty
            Intermediates = intermediates
            MLIROutput = None
            LLVMOutput = None
        }
}

/// Simplified entry point using file path
let compile 
    (projectPath: string) 
    (intermediatesDir: string option) 
    (progress: ProgressCallback) = async {
    
    // Create default compilation configuration
    let compilationConfig : CompilationConfig = {
        EnableClosureElimination = true
        EnableStackAllocation = true
        EnableReachabilityAnalysis = true
        PreserveIntermediateASTs = intermediatesDir.IsSome
        VerboseOutput = false
    }
    
    // Create F# checker and load project
    let checker = FSharpChecker.Create()
    
    // Read the project file content and create ISourceText
    let content = File.ReadAllText(projectPath)
    let sourceText = SourceText.ofString content
    
    // Get project options from script
    let! (projectOptions, diagnostics) = checker.GetProjectOptionsFromScript(projectPath, sourceText)
    
    // Check for critical errors in diagnostics
    if diagnostics.Length > 0 then
        printfn "Project loading diagnostics:"
        for diag in diagnostics do
            printfn "  %s" diag.Message
    
    // Use a default output path
    let outputPath = Path.ChangeExtension(projectPath, ".exe")
    
    return! compileProject projectPath outputPath projectOptions compilationConfig intermediatesDir progress
}