/// Main PSG builder entry point - orchestrates all building phases
module Core.PSG.Construction.Main

open System.IO
open FSharp.Compiler.Syntax
open FSharp.Compiler.CodeAnalysis
open Core.PSG.Types
open Core.PSG.Correlation
open Core.PSG.TypeIntegration
open Core.PSG.Nanopass.IntermediateEmission
open Core.PSG.Nanopass.DefUseEdges
open Core.PSG.Construction.Types
open Core.PSG.Construction.DeclarationProcessing

/// Flag to enable nanopass intermediate emission (set via --verbose or -k flags)
let mutable emitNanopassIntermediates = false

/// Output directory for nanopass intermediates
let mutable nanopassOutputDir = ""

/// Symbol validation helper
let private validateSymbolCapture (graph: ProgramSemanticGraph) =
    let expectedSymbols = [
        "stackBuffer"; "AsReadOnlySpan"; "spanToString";
        "readInto"; "sprintf"; "Ok"; "Error"; "Write"; "WriteLine"
    ]

    printfn "[VALIDATION] === Symbol Capture Validation ==="
    expectedSymbols |> List.iter (fun expected ->
        let found =
            graph.SymbolTable
            |> Map.exists (fun _ symbol ->
                symbol.DisplayName.Contains(expected) ||
                symbol.FullName.Contains(expected))

        if found then
            printfn "[VALIDATION] ✓ Found expected symbol: %s" expected
        else
            printfn "[VALIDATION] ✗ Missing expected symbol: %s" expected
    )

    printfn "[VALIDATION] Total symbols captured: %d" graph.SymbolTable.Count

/// Build complete PSG from project results with ENHANCED symbol correlation (FCS 43.9.300 compatible)
let buildProgramSemanticGraph
    (checkResults: FSharpCheckProjectResults)
    (parseResults: FSharpParseFileResults[]) : ProgramSemanticGraph =

    // Force initialization of BindingProcessing module to register the circular dependency handler
    // This must happen before any expression processing occurs
    Core.PSG.Construction.BindingProcessing.ensureInitialized ()

    printfn "[BUILDER] Starting ENHANCED PSG construction (FCS 43.9.300 compatible)"

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

    printfn "[BUILDER] Phase 1: Building structural nodes with enhanced correlation from %d files" parseResults.Length

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
                }
            )

    printfn "[BUILDER] Phase 1 complete: Enhanced PSG built with %d nodes, %d entry points"
        structuralGraph.Nodes.Count structuralGraph.EntryPoints.Length

    // Emit Phase 1 intermediate
    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediate structuralGraph "1_structural" nanopassOutputDir

    // Validate symbol capture
    validateSymbolCapture structuralGraph

    // Phase 2: Apply FCS constraint resolution
    printfn "[BUILDER] Phase 2: Applying FCS constraint resolution"
    let typeEnhancedGraph = integrateTypesWithCheckResults structuralGraph checkResults

    // Emit Phase 2 intermediate
    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediate typeEnhancedGraph "2_type_integration" nanopassOutputDir
        emitNanopassDiff structuralGraph typeEnhancedGraph "1_structural" "2_type_integration" nanopassOutputDir

    // Phase 3: Nanopass - Add def-use edges
    // This makes variable binding relationships explicit in the PSG structure,
    // eliminating the need for scope tracking in the emitter.
    printfn "[BUILDER] Phase 3: Running def-use nanopass"
    let defUseGraph = addDefUseEdges typeEnhancedGraph

    // Emit Phase 3 intermediate
    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediate defUseGraph "3_def_use_edges" nanopassOutputDir
        emitNanopassDiff typeEnhancedGraph defUseGraph "2_type_integration" "3_def_use_edges" nanopassOutputDir

    // Phase 4: Finalize nodes and analyze context
    printfn "[BUILDER] Phase 4: Finalizing PSG nodes and analyzing context"
    let finalNodes =
        defUseGraph.Nodes
        |> Map.map (fun _ node ->
            node
            |> ChildrenStateHelpers.finalizeChildren
            |> ReachabilityHelpers.updateNodeContext)

    let finalGraph =
        { defUseGraph with
            Nodes = finalNodes
            CompilationOrder = parseResults |> Array.map (fun pr -> pr.FileName) |> List.ofArray
        }

    // Emit Phase 4 intermediate (final)
    if emitNanopassIntermediates && nanopassOutputDir <> "" then
        emitNanopassIntermediate finalGraph "4_finalized" nanopassOutputDir
        emitNanopassDiff defUseGraph finalGraph "3_def_use_edges" "4_finalized" nanopassOutputDir

    printfn "[BUILDER] ENHANCED PSG construction complete (FCS 43.9.300 compatible)"
    printfn "[BUILDER] Final PSG: %d nodes, %d edges, %d entry points, %d symbols"
        finalGraph.Nodes.Count finalGraph.Edges.Length finalGraph.EntryPoints.Length finalGraph.SymbolTable.Count

    // Final validation
    validateSymbolCapture finalGraph

    finalGraph
