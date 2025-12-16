/// Main PSG builder entry point - orchestrates all building phases
module Core.PSG.Construction.Main

open System.IO
open FSharp.Compiler.Syntax
open FSharp.Compiler.CodeAnalysis
open Core.PSG.Types
open Core.PSG.Correlation
open Core.PSG.TypeIntegration
open Core.PSG.Nanopass.IntermediateEmission
open Core.PSG.Nanopass.FlattenApplications
open Core.PSG.Nanopass.ReducePipeOperators
open Core.PSG.Nanopass.ReduceAlloyOperators
open Core.PSG.Nanopass.DefUseEdges
open Core.PSG.Nanopass.ConstantPropagation
open Core.PSG.Nanopass.LowerStringLength
open Core.PSG.Nanopass.ParameterAnnotation
open Core.PSG.Nanopass.ClassifyOperations
open Core.PSG.Nanopass.LowerInterpolatedStrings
open Core.PSG.Construction.Types
open Core.PSG.Construction.DeclarationProcessing

/// Flag to enable nanopass intermediate emission (set via --verbose or -k flags)
let mutable emitNanopassIntermediates = false

/// Output directory for nanopass intermediates
let mutable nanopassOutputDir = ""

/// Symbol validation helper (no-op in production)
let private validateSymbolCapture (_graph: ProgramSemanticGraph) = ()

/// Build source files map from parse results
let private buildSourceFiles (parseResults: FSharpParseFileResults[]) =
    parseResults
    |> Array.map (fun pr ->
        let content =
            if File.Exists pr.FileName then
                File.ReadAllText pr.FileName
            else ""
        pr.FileName, content
    )
    |> Map.ofArray

/// Build structural PSG from parse results only (Phase 1)
/// This is fast - no type checking required
/// Used for reachability analysis before type checking
let buildStructuralGraph (parseResults: FSharpParseFileResults[]) : ProgramSemanticGraph =
    // Force initialization of BindingProcessing module
    Core.PSG.Construction.BindingProcessing.ensureInitialized ()

    let sourceFiles = buildSourceFiles parseResults

    // Create structural-only context (no symbol correlation)
    let context = createStructuralContext parseResults sourceFiles

    // Process each file and merge results
    let graphs =
        parseResults
        |> Array.choose (fun pr ->
            match pr.ParseTree with
            | ParsedInput.ImplFile implFile ->
                let (ParsedImplFileInput(contents = modules)) = implFile
                let emptyGraph = {
                    Nodes = Map.empty
                    Edges = []
                    SymbolTable = Map.empty
                    EntryPoints = []
                    SourceFiles = sourceFiles
                    CompilationOrder = []
                    StringLiterals = Map.empty
                }
                let processedGraph =
                    modules |> List.fold (fun acc implFile ->
                        processImplFile implFile context acc) emptyGraph
                Some processedGraph
            | _ -> None
        )

    // Merge all graphs
    let structuralGraph =
        if Array.isEmpty graphs then
            {
                Nodes = Map.empty
                Edges = []
                SymbolTable = Map.empty
                EntryPoints = []
                SourceFiles = sourceFiles
                CompilationOrder = []
                StringLiterals = Map.empty
            }
        else
            graphs |> Array.reduce (fun g1 g2 ->
                {
                    Nodes = Map.fold (fun acc k v -> Map.add k v acc) g1.Nodes g2.Nodes
                    Edges = g1.Edges @ g2.Edges
                    SymbolTable = Map.fold (fun acc k v -> Map.add k v acc) g1.SymbolTable g2.SymbolTable
                    EntryPoints = g1.EntryPoints @ g2.EntryPoints
                    SourceFiles = Map.fold (fun acc k v -> Map.add k v acc) g1.SourceFiles g2.SourceFiles
                    CompilationOrder = g1.CompilationOrder @ g2.CompilationOrder
                    StringLiterals = Map.fold (fun acc k v -> Map.add k v acc) g1.StringLiterals g2.StringLiterals
                }
            )

    // Emit Phase 1 intermediate (parallel)
    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediateAsync structuralGraph "1_structural" nanopassOutputDir

    { structuralGraph with
        CompilationOrder = parseResults |> Array.map (fun pr -> pr.FileName) |> List.ofArray
    }

/// Build correlated PSG from project results (structural + symbol correlation only)
/// This is the first major pass - returns graph ready for reachability analysis
/// IMPORTANT: Call reachability BEFORE runEnrichmentPasses to narrow the graph
let buildProgramSemanticGraph
    (checkResults: FSharpCheckProjectResults)
    (parseResults: FSharpParseFileResults[]) : ProgramSemanticGraph =

    // Force initialization of BindingProcessing module to register the circular dependency handler
    Core.PSG.Construction.BindingProcessing.ensureInitialized ()

    let correlationContext = createContext checkResults
    let sourceFiles = buildSourceFiles parseResults

    let context = createFullContext checkResults parseResults correlationContext sourceFiles

    // Process each file and merge results
    let graphs =
        parseResults
        |> Array.choose (fun pr ->
            match pr.ParseTree with
            | ParsedInput.ImplFile implFile ->
                let (ParsedImplFileInput(contents = modules)) = implFile
                let emptyGraph = {
                    Nodes = Map.empty
                    Edges = []
                    SymbolTable = Map.empty
                    EntryPoints = []
                    SourceFiles = sourceFiles
                    CompilationOrder = []
                    StringLiterals = Map.empty
                }
                let processedGraph =
                    modules |> List.fold (fun acc implFile ->
                        processImplFile implFile context acc) emptyGraph
                Some processedGraph
            | _ -> None
        )

    // Merge all graphs
    let structuralGraph =
        if Array.isEmpty graphs then
            {
                Nodes = Map.empty
                Edges = []
                SymbolTable = Map.empty
                EntryPoints = []
                SourceFiles = sourceFiles
                CompilationOrder = []
                StringLiterals = Map.empty
            }
        else
            graphs |> Array.reduce (fun g1 g2 ->
                {
                    Nodes = Map.fold (fun acc k v -> Map.add k v acc) g1.Nodes g2.Nodes
                    Edges = g1.Edges @ g2.Edges
                    SymbolTable = Map.fold (fun acc k v -> Map.add k v acc) g1.SymbolTable g2.SymbolTable
                    EntryPoints = g1.EntryPoints @ g2.EntryPoints
                    SourceFiles = Map.fold (fun acc k v -> Map.add k v acc) g1.SourceFiles g2.SourceFiles
                    CompilationOrder = g1.CompilationOrder @ g2.CompilationOrder
                    StringLiterals = Map.fold (fun acc k v -> Map.add k v acc) g1.StringLiterals g2.StringLiterals
                }
            )

    // Emit structural intermediate (parallel)
    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediateAsync structuralGraph "1_structural" nanopassOutputDir

    // Validate symbol capture
    validateSymbolCapture structuralGraph

    // Return correlated structural graph - reachability should be called BEFORE enrichment
    { structuralGraph with
        CompilationOrder = parseResults |> Array.map (fun pr -> pr.FileName) |> List.ofArray
    }

/// Run enrichment nanopasses on a PSG (post-reachability)
/// These passes should only be run on the NARROWED (reachable) graph for performance
/// Call this AFTER reachability analysis has marked unreachable nodes
let runEnrichmentPasses
    (graph: ProgramSemanticGraph)
    (checkResults: FSharpCheckProjectResults) : ProgramSemanticGraph =

    // Type integration - apply FCS constraint resolution
    let typeEnhancedGraph = integrateTypesWithCheckResults graph checkResults

    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediateAsync typeEnhancedGraph "2_type_integration" nanopassOutputDir
        emitNanopassDiffAsync graph typeEnhancedGraph "1_structural" "2_type_integration" nanopassOutputDir

    // Resolve SRTP constraints
    let srtpResolvedGraph = Core.PSG.Nanopass.ResolveSRTP.run typeEnhancedGraph checkResults

    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediateAsync srtpResolvedGraph "2a_srtp_resolved" nanopassOutputDir
        emitNanopassDiffAsync typeEnhancedGraph srtpResolvedGraph "2_type_integration" "2a_srtp_resolved" nanopassOutputDir

    // Flatten curried applications
    let flattenedGraph = flattenApplications srtpResolvedGraph

    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediateAsync flattenedGraph "2b_flattened_apps" nanopassOutputDir
        emitNanopassDiffAsync srtpResolvedGraph flattenedGraph "2a_srtp_resolved" "2b_flattened_apps" nanopassOutputDir

    // Reduce pipe operators
    let pipeReducedGraph = reducePipeOperators flattenedGraph

    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediateAsync pipeReducedGraph "2c_pipe_reduced" nanopassOutputDir
        emitNanopassDiffAsync flattenedGraph pipeReducedGraph "2b_flattened_apps" "2c_pipe_reduced" nanopassOutputDir

    // Reduce Alloy operators (currently disabled)
    let alloyReducedGraph = pipeReducedGraph

    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediateAsync alloyReducedGraph "2d_alloy_reduced" nanopassOutputDir
        emitNanopassDiffAsync pipeReducedGraph alloyReducedGraph "2c_pipe_reduced" "2d_alloy_reduced" nanopassOutputDir

    // Add def-use edges
    let defUseGraph = addDefUseEdges alloyReducedGraph

    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediateAsync defUseGraph "3_def_use_edges" nanopassOutputDir
        emitNanopassDiffAsync typeEnhancedGraph defUseGraph "2_type_integration" "3_def_use_edges" nanopassOutputDir

    // Constant propagation
    let constPropGraph = propagateConstants defUseGraph

    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediateAsync constPropGraph "3a_const_prop" nanopassOutputDir
        emitNanopassDiffAsync defUseGraph constPropGraph "3_def_use_edges" "3a_const_prop" nanopassOutputDir

    // Lower string length
    let strlenLoweredGraph = lowerStringLength constPropGraph

    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediateAsync strlenLoweredGraph "3a2_strlen_lowered" nanopassOutputDir
        emitNanopassDiffAsync constPropGraph strlenLoweredGraph "3a_const_prop" "3a2_strlen_lowered" nanopassOutputDir

    // Annotate function parameters
    let paramAnnotatedGraph = annotateParameters strlenLoweredGraph

    // Classify operations
    let classifiedGraph = classifyOperations paramAnnotatedGraph

    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediateAsync classifiedGraph "3c_classified_ops" nanopassOutputDir
        emitNanopassDiffAsync paramAnnotatedGraph classifiedGraph "3b_param_annotated" "3c_classified_ops" nanopassOutputDir

    // Lower interpolated strings
    let loweredGraph = lowerInterpolatedStrings classifiedGraph

    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediateAsync loweredGraph "3d_lowered_interp_strings" nanopassOutputDir
        emitNanopassDiffAsync classifiedGraph loweredGraph "3c_classified_ops" "3d_lowered_interp_strings" nanopassOutputDir

    // Finalize nodes and analyze context
    let finalNodes =
        loweredGraph.Nodes
        |> Map.map (fun _ node ->
            node
            |> ChildrenStateHelpers.finalizeChildren
            |> ReachabilityHelpers.updateNodeContext)

    let finalGraph =
        { loweredGraph with
            Nodes = finalNodes
        }

    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediateAsync finalGraph "4_finalized" nanopassOutputDir
        emitNanopassDiffAsync loweredGraph finalGraph "3d_lowered_interp_strings" "4_finalized" nanopassOutputDir
        // Wait for all parallel writes to complete before returning
        awaitAllWrites()

    validateSymbolCapture finalGraph
    finalGraph
