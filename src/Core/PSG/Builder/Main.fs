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

/// Build complete PSG from project results with ENHANCED symbol correlation (FCS 43.9.300 compatible)
let buildProgramSemanticGraph
    (checkResults: FSharpCheckProjectResults)
    (parseResults: FSharpParseFileResults[]) : ProgramSemanticGraph =

    // Force initialization of BindingProcessing module to register the circular dependency handler
    Core.PSG.Construction.BindingProcessing.ensureInitialized ()

    let correlationContext = createContext checkResults

    let sourceFiles =
        parseResults
        |> Array.map (fun pr ->
            let content =
                if File.Exists pr.FileName then
                    File.ReadAllText pr.FileName
                else ""
            pr.FileName, content
        )
        |> Map.ofArray

    let context = {
        CheckResults = checkResults
        ParseResults = parseResults
        CorrelationContext = correlationContext
        SourceFiles = sourceFiles
    }

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

    // Emit Phase 1 intermediate
    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediate structuralGraph "1_structural" nanopassOutputDir

    // Validate symbol capture
    validateSymbolCapture structuralGraph

    // Phase 2: Apply FCS constraint resolution
    let typeEnhancedGraph = integrateTypesWithCheckResults structuralGraph checkResults

    // Emit Phase 2 intermediate
    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediate typeEnhancedGraph "2_type_integration" nanopassOutputDir
        emitNanopassDiff structuralGraph typeEnhancedGraph "1_structural" "2_type_integration" nanopassOutputDir

    // Phase 2a: Nanopass - Resolve SRTP constraints
    let srtpResolvedGraph = Core.PSG.Nanopass.ResolveSRTP.run typeEnhancedGraph checkResults

    // Emit Phase 2a intermediate
    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediate srtpResolvedGraph "2a_srtp_resolved" nanopassOutputDir
        emitNanopassDiff typeEnhancedGraph srtpResolvedGraph "2_type_integration" "2a_srtp_resolved" nanopassOutputDir

    // Phase 2b: Nanopass - Flatten curried applications
    let flattenedGraph = flattenApplications srtpResolvedGraph

    // Emit Phase 2b intermediate
    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediate flattenedGraph "2b_flattened_apps" nanopassOutputDir
        emitNanopassDiff srtpResolvedGraph flattenedGraph "2a_srtp_resolved" "2b_flattened_apps" nanopassOutputDir

    // Phase 2c: Nanopass - Reduce pipe operators
    let pipeReducedGraph = reducePipeOperators flattenedGraph

    // Emit Phase 2c intermediate
    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediate pipeReducedGraph "2c_pipe_reduced" nanopassOutputDir
        emitNanopassDiff flattenedGraph pipeReducedGraph "2b_flattened_apps" "2c_pipe_reduced" nanopassOutputDir

    // Phase 2d: Nanopass - Reduce Alloy operators (currently disabled)
    let alloyReducedGraph = pipeReducedGraph

    // Emit Phase 2d intermediate
    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediate alloyReducedGraph "2d_alloy_reduced" nanopassOutputDir
        emitNanopassDiff pipeReducedGraph alloyReducedGraph "2c_pipe_reduced" "2d_alloy_reduced" nanopassOutputDir

    // Phase 3: Nanopass - Add def-use edges
    let defUseGraph = addDefUseEdges alloyReducedGraph

    // Emit Phase 3 intermediate
    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediate defUseGraph "3_def_use_edges" nanopassOutputDir
        emitNanopassDiff typeEnhancedGraph defUseGraph "2_type_integration" "3_def_use_edges" nanopassOutputDir

    // Phase 3a: Nanopass - Constant propagation
    let constPropGraph = propagateConstants defUseGraph

    // Emit Phase 3a intermediate
    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediate constPropGraph "3a_const_prop" nanopassOutputDir
        emitNanopassDiff defUseGraph constPropGraph "3_def_use_edges" "3a_const_prop" nanopassOutputDir

    // Phase 3a2: Nanopass - Lower string length
    let strlenLoweredGraph = lowerStringLength constPropGraph

    // Emit Phase 3a2 intermediate
    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediate strlenLoweredGraph "3a2_strlen_lowered" nanopassOutputDir
        emitNanopassDiff constPropGraph strlenLoweredGraph "3a_const_prop" "3a2_strlen_lowered" nanopassOutputDir

    // Phase 3b: Nanopass - Annotate function parameters
    let paramAnnotatedGraph = annotateParameters strlenLoweredGraph

    // Phase 3c: Nanopass - Classify operations
    let classifiedGraph = classifyOperations paramAnnotatedGraph

    // Emit Phase 3c intermediate
    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediate classifiedGraph "3c_classified_ops" nanopassOutputDir
        emitNanopassDiff paramAnnotatedGraph classifiedGraph "3b_param_annotated" "3c_classified_ops" nanopassOutputDir

    // Phase 3d: Nanopass - Lower interpolated strings
    let loweredGraph = lowerInterpolatedStrings classifiedGraph

    // Emit Phase 3d intermediate
    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediate loweredGraph "3d_lowered_interp_strings" nanopassOutputDir
        emitNanopassDiff classifiedGraph loweredGraph "3c_classified_ops" "3d_lowered_interp_strings" nanopassOutputDir

    // Phase 4: Finalize nodes and analyze context
    let finalNodes =
        loweredGraph.Nodes
        |> Map.map (fun _ node ->
            node
            |> ChildrenStateHelpers.finalizeChildren
            |> ReachabilityHelpers.updateNodeContext)

    let finalGraph =
        { loweredGraph with
            Nodes = finalNodes
            CompilationOrder = parseResults |> Array.map (fun pr -> pr.FileName) |> List.ofArray
        }

    // Emit Phase 4 intermediate (final)
    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediate finalGraph "4_finalized" nanopassOutputDir
        emitNanopassDiff loweredGraph finalGraph "3d_lowered_interp_strings" "4_finalized" nanopassOutputDir

    validateSymbolCapture finalGraph
    finalGraph
