/// IntermediateEmission - Emit labeled PSG intermediates for each nanopass phase
///
/// This module provides functions to dump the PSG state after each nanopass,
/// enabling analysis of individual transformations and their effects.
///
/// Reference: Nanopass Framework (Sarkar, Waddell, Dybvig, Keep)
module Core.PSG.Nanopass.IntermediateEmission

open System
open System.IO
open System.Text.Json
open System.Threading.Tasks
open FSharp.Compiler.Symbols
open Core.PSG.Types
open Core.Utilities.IntermediateWriter

/// JSON options with F# support (reuse from IntermediateWriter)
let private jsonOptions = jsonOptionsWithFSharpSupport

// ═══════════════════════════════════════════════════════════════════════════
// Labeled Nanopass Intermediate Emission
// ═══════════════════════════════════════════════════════════════════════════

/// Emit a labeled PSG intermediate for a specific nanopass phase.
/// This allows each transform to be analyzed independently.
let emitNanopassIntermediate (psg: ProgramSemanticGraph) (phaseLabel: string) (outputDir: string) =
    try
        Directory.CreateDirectory(outputDir) |> ignore

        // Create phase-specific filename
        let safeLabel = phaseLabel.Replace(" ", "_").Replace("/", "-")
        let filename = sprintf "psg_phase_%s.json" safeLabel
        let filepath = Path.Combine(outputDir, filename)

        // Prepare compact PSG summary for this phase
        let edgesByKind =
            psg.Edges
            |> List.groupBy (fun e -> e.Kind.ToString())
            |> List.map (fun (kind, edges) -> kind, edges.Length)
            |> Map.ofList

        let nodesWithSymbols =
            psg.Nodes |> Map.filter (fun _ n -> n.Symbol.IsSome) |> Map.count

        let nodesWithTypes =
            psg.Nodes |> Map.filter (fun _ n -> n.Type.IsSome) |> Map.count

        let reachableNodes =
            psg.Nodes |> Map.filter (fun _ n -> n.IsReachable) |> Map.count

        let phaseData = {|
            Phase = phaseLabel
            Timestamp = DateTime.UtcNow.ToString("o")
            Summary = {|
                TotalNodes = psg.Nodes.Count
                NodesWithSymbols = nodesWithSymbols
                NodesWithTypes = nodesWithTypes
                ReachableNodes = reachableNodes
                TotalEdges = psg.Edges.Length
                EdgesByKind = edgesByKind
                EntryPoints = psg.EntryPoints.Length
            |}
            Nodes =
                psg.Nodes
                |> Map.toSeq
                |> Seq.map (fun (id, node) -> {|
                    Id = id
                    SyntaxKind = SyntaxKindT.toString node.Kind
                    SymbolName = node.Symbol |> Option.map (fun s -> s.DisplayName)
                    SymbolFullName = node.Symbol |> Option.map (fun s -> s.FullName)
                    TypeName = node.Type |> Option.map (fun t ->
                        try t.Format(FSharpDisplayContext.Empty)
                        with _ -> "?")
                    ParentId = node.ParentId |> Option.map (fun n -> n.Value)
                    IsReachable = node.IsReachable
                    Operation = node.Operation |> Option.map (fun op -> op.ToString())
                    PlatformBinding = node.PlatformBinding |> Option.map (fun pb -> pb.EntryPoint)
                    Range = {|
                        File = Path.GetFileName(node.Range.FileName)
                        Line = node.Range.StartLine
                        Col = node.Range.StartColumn
                    |}
                |})
                |> Seq.toArray
            Edges =
                psg.Edges
                |> List.map (fun e -> {|
                    Source = e.Source.Value
                    Target = e.Target.Value
                    Kind = e.Kind.ToString()
                |})
                |> List.toArray
        |}

        File.WriteAllText(filepath, JsonSerializer.Serialize(phaseData, jsonOptions))

    with ex ->
        // Log emission failure to stderr, don't crash compilation
        eprintfn "[EMIT] Warning: Failed to emit nanopass intermediate '%s': %s" phaseLabel ex.Message

/// Emit a diff summary between two PSG phases
let emitNanopassDiff (before: ProgramSemanticGraph) (after: ProgramSemanticGraph) (fromPhase: string) (toPhase: string) (outputDir: string) =
    try
        Directory.CreateDirectory(outputDir) |> ignore

        let safeFromLabel = fromPhase.Replace(" ", "_").Replace("/", "-")
        let safeToLabel = toPhase.Replace(" ", "_").Replace("/", "-")
        let filename = sprintf "psg_diff_%s_to_%s.json" safeFromLabel safeToLabel
        let filepath = Path.Combine(outputDir, filename)

        // Compute edge differences
        let beforeEdgeSet =
            before.Edges
            |> List.map (fun e -> (e.Source.Value, e.Target.Value, e.Kind.ToString()))
            |> Set.ofList

        let afterEdgeSet =
            after.Edges
            |> List.map (fun e -> (e.Source.Value, e.Target.Value, e.Kind.ToString()))
            |> Set.ofList

        let addedEdges = Set.difference afterEdgeSet beforeEdgeSet |> Set.toList
        let removedEdges = Set.difference beforeEdgeSet afterEdgeSet |> Set.toList

        // Compute node differences
        let beforeNodeIds = before.Nodes |> Map.keys |> Set.ofSeq
        let afterNodeIds = after.Nodes |> Map.keys |> Set.ofSeq

        let addedNodes = Set.difference afterNodeIds beforeNodeIds |> Set.count
        let removedNodes = Set.difference beforeNodeIds afterNodeIds |> Set.count

        // Compute edge kind breakdown for added edges
        let addedEdgesByKind =
            addedEdges
            |> List.groupBy (fun (_, _, kind) -> kind)
            |> List.map (fun (kind, edges) -> kind, edges.Length)
            |> Map.ofList

        let diffData = {|
            FromPhase = fromPhase
            ToPhase = toPhase
            Timestamp = DateTime.UtcNow.ToString("o")
            NodeChanges = {|
                Before = before.Nodes.Count
                After = after.Nodes.Count
                Added = addedNodes
                Removed = removedNodes
            |}
            EdgeChanges = {|
                Before = before.Edges.Length
                After = after.Edges.Length
                Added = addedEdges.Length
                Removed = removedEdges.Length
                AddedByKind = addedEdgesByKind
            |}
            AddedEdges =
                addedEdges
                |> List.take (min 100 addedEdges.Length) // Limit output size
                |> List.map (fun (src, tgt, kind) -> {| Source = src; Target = tgt; Kind = kind |})
            RemovedEdges =
                removedEdges
                |> List.take (min 100 removedEdges.Length)
                |> List.map (fun (src, tgt, kind) -> {| Source = src; Target = tgt; Kind = kind |})
        |}

        File.WriteAllText(filepath, JsonSerializer.Serialize(diffData, jsonOptions))

    with ex ->
        // Log diff emission failure to stderr, don't crash compilation
        eprintfn "[EMIT] Warning: Failed to emit diff '%s' -> '%s': %s" fromPhase toPhase ex.Message

// ═══════════════════════════════════════════════════════════════════════════
// Parallel write task collection
// ═══════════════════════════════════════════════════════════════════════════

/// Collected write tasks - call awaitAllWrites() before process exit
let private pendingWrites = System.Collections.Concurrent.ConcurrentBag<Task>()

/// Spin off intermediate emission as a background task (non-blocking)
let emitNanopassIntermediateAsync (psg: ProgramSemanticGraph) (phaseLabel: string) (outputDir: string) : unit =
    let task = Task.Run(fun () -> emitNanopassIntermediate psg phaseLabel outputDir)
    pendingWrites.Add(task)

/// Spin off diff emission as a background task (non-blocking)
let emitNanopassDiffAsync (before: ProgramSemanticGraph) (after: ProgramSemanticGraph) (fromPhase: string) (toPhase: string) (outputDir: string) : unit =
    let task = Task.Run(fun () -> emitNanopassDiff before after fromPhase toPhase outputDir)
    pendingWrites.Add(task)

/// Wait for all pending write tasks to complete
let awaitAllWrites () : unit =
    let tasks = pendingWrites.ToArray()
    if tasks.Length > 0 then
        Task.WaitAll(tasks)
        pendingWrites.Clear()
