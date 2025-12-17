/// Main PSG builder entry point - orchestrates all building phases
module Core.PSG.Construction.Main

open System.IO
open FSharp.Compiler.Syntax
open FSharp.Compiler.CodeAnalysis
open Core.CompilerConfig
open Core.PSG.Types
open Core.PSG.Correlation
open Core.PSG.TypeIntegration
open Core.PSG.Nanopass.IntermediateEmission
open Core.PSG.Nanopass.FlattenApplications
open Core.PSG.Nanopass.ReducePipeOperators
open Core.PSG.Nanopass.DefUseEdges
open Core.PSG.Nanopass.ConstantPropagation
open Core.PSG.Nanopass.LowerStringLength
open Core.PSG.Nanopass.LowerStructConstructors
open Core.PSG.Nanopass.ParameterAnnotation
open Core.PSG.Nanopass.ClassifyOperations
open Core.PSG.Nanopass.LowerInterpolatedStrings
open Core.PSG.Nanopass.DetectPlatformBindings
open Core.PSG.Nanopass.DetectInlineFunctions
open Core.PSG.Construction.Types
open Core.PSG.Construction.DeclarationProcessing

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
    if shouldEmitNanopassIntermediates() then
        emitNanopassIntermediateAsync structuralGraph "1_structural" (getNanopassOutputDir())

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
    if shouldEmitNanopassIntermediates() then
        emitNanopassIntermediateAsync structuralGraph "1_structural" (getNanopassOutputDir())

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

    if shouldEmitNanopassIntermediates() then
        emitNanopassIntermediateAsync typeEnhancedGraph "2_type_integration" (getNanopassOutputDir())
        emitNanopassDiffAsync graph typeEnhancedGraph "1_structural" "2_type_integration" (getNanopassOutputDir())

    // Resolve SRTP constraints
    let srtpResolvedGraph = Core.PSG.Nanopass.ResolveSRTP.run typeEnhancedGraph checkResults

    if shouldEmitNanopassIntermediates() then
        emitNanopassIntermediateAsync srtpResolvedGraph "2a_srtp_resolved" (getNanopassOutputDir())
        emitNanopassDiffAsync typeEnhancedGraph srtpResolvedGraph "2_type_integration" "2a_srtp_resolved" (getNanopassOutputDir())

    // Reduce pipe operators FIRST (creates curried structures)
    let pipeReducedGraph = reducePipeOperators srtpResolvedGraph

    if shouldEmitNanopassIntermediates() then
        emitNanopassIntermediateAsync pipeReducedGraph "2b_pipe_reduced" (getNanopassOutputDir())
        emitNanopassDiffAsync srtpResolvedGraph pipeReducedGraph "2a_srtp_resolved" "2b_pipe_reduced" (getNanopassOutputDir())

    // Flatten curried applications SECOND (flattens structures from pipe reduction)
    let flattenedGraph = flattenApplications pipeReducedGraph

    if shouldEmitNanopassIntermediates() then
        emitNanopassIntermediateAsync flattenedGraph "2c_flattened_apps" (getNanopassOutputDir())
        emitNanopassDiffAsync pipeReducedGraph flattenedGraph "2b_pipe_reduced" "2c_flattened_apps" (getNanopassOutputDir())

    // Add def-use edges
    let defUseGraph = addDefUseEdges flattenedGraph

    if shouldEmitNanopassIntermediates() then
        emitNanopassIntermediateAsync defUseGraph "3_def_use_edges" (getNanopassOutputDir())
        emitNanopassDiffAsync typeEnhancedGraph defUseGraph "2_type_integration" "3_def_use_edges" (getNanopassOutputDir())

    // Constant propagation
    let constPropGraph = propagateConstants defUseGraph

    if shouldEmitNanopassIntermediates() then
        emitNanopassIntermediateAsync constPropGraph "3a_const_prop" (getNanopassOutputDir())
        emitNanopassDiffAsync defUseGraph constPropGraph "3_def_use_edges" "3a_const_prop" (getNanopassOutputDir())

    // Lower string length
    let strlenLoweredGraph = lowerStringLength constPropGraph

    if shouldEmitNanopassIntermediates() then
        emitNanopassIntermediateAsync strlenLoweredGraph "3a2_strlen_lowered" (getNanopassOutputDir())
        emitNanopassDiffAsync constPropGraph strlenLoweredGraph "3a_const_prop" "3a2_strlen_lowered" (getNanopassOutputDir())

    // Lower struct constructors to Record nodes (Fidelity memory model)
    let structLoweredGraph = lowerStructConstructors strlenLoweredGraph

    if shouldEmitNanopassIntermediates() then
        emitNanopassIntermediateAsync structLoweredGraph "3a3_struct_lowered" (getNanopassOutputDir())
        emitNanopassDiffAsync strlenLoweredGraph structLoweredGraph "3a2_strlen_lowered" "3a3_struct_lowered" (getNanopassOutputDir())

    // Annotate function parameters
    let paramAnnotatedGraph = annotateParameters structLoweredGraph

    // Classify operations
    let classifiedGraph = classifyOperations paramAnnotatedGraph

    if shouldEmitNanopassIntermediates() then
        emitNanopassIntermediateAsync classifiedGraph "3c_classified_ops" (getNanopassOutputDir())
        emitNanopassDiffAsync paramAnnotatedGraph classifiedGraph "3b_param_annotated" "3c_classified_ops" (getNanopassOutputDir())

    // Detect platform bindings (marks Alloy.Platform.Bindings functions with PlatformBinding)
    let platformBindingsGraph = detectPlatformBindings classifiedGraph

    if shouldEmitNanopassIntermediates() then
        emitNanopassIntermediateAsync platformBindingsGraph "3c2_platform_bindings" (getNanopassOutputDir())
        emitNanopassDiffAsync classifiedGraph platformBindingsGraph "3c_classified_ops" "3c2_platform_bindings" (getNanopassOutputDir())

    // Detect inline functions (marks F# inline functions for exclusion from standalone emission)
    let inlineFunctionsGraph = detectInlineFunctions platformBindingsGraph

    if shouldEmitNanopassIntermediates() then
        emitNanopassIntermediateAsync inlineFunctionsGraph "3c3_inline_functions" (getNanopassOutputDir())
        emitNanopassDiffAsync platformBindingsGraph inlineFunctionsGraph "3c2_platform_bindings" "3c3_inline_functions" (getNanopassOutputDir())

    // Lower interpolated strings
    let loweredGraph = lowerInterpolatedStrings inlineFunctionsGraph

    if shouldEmitNanopassIntermediates() then
        emitNanopassIntermediateAsync loweredGraph "3d_lowered_interp_strings" (getNanopassOutputDir())
        emitNanopassDiffAsync inlineFunctionsGraph loweredGraph "3c3_inline_functions" "3d_lowered_interp_strings" (getNanopassOutputDir())

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

    if shouldEmitNanopassIntermediates() then
        emitNanopassIntermediateAsync finalGraph "4_finalized" (getNanopassOutputDir())
        emitNanopassDiffAsync loweredGraph finalGraph "3d_lowered_interp_strings" "4_finalized" (getNanopassOutputDir())
        // Wait for all parallel writes to complete before returning
        awaitAllWrites()

    finalGraph
