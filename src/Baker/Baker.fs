/// Baker - Post-Reachability Type Resolution Component Library
///
/// Baker is the symmetric companion to Alex in the Firefly architecture:
/// - Baker: Consolidates type-level transforms on the "front" of the PSG
/// - Alex: Consolidates code-level transforms on the "back" of the PSG
///
/// CRITICAL: Baker operates AFTER reachability analysis (Phase 3).
/// It ONLY processes the narrowed compute graph - nodes marked IsReachable = true.
///
/// This module provides the main entry point for Baker enrichment.
module Baker.Baker

open System
open FSharp.Compiler.CodeAnalysis
open Core.PSG.Types
open Baker.Types
open Baker.MemberBodyMapper

/// Result of Baker enrichment including updated PSG and statistics
type BakerEnrichmentResult = {
    /// The enriched PSG (with member bodies index)
    EnrichedPSG: ProgramSemanticGraph
    /// Member body mappings for body lookup
    MemberBodies: Map<string, MemberBodyMapping>
    /// Statistics from Baker processing
    Statistics: BakerStatistics
}

/// Run Baker enrichment on a PSG that has completed reachability analysis (Phase 3)
///
/// PRECONDITION: PSG must have reachability marks (IsReachable field populated)
/// POST: PSG is enriched with member body mappings
///
/// This is the Phase 4 (Typed Tree Overlay) step in the nanopass pipeline.
let enrich
    (psg: ProgramSemanticGraph)
    (checkResults: FSharpCheckProjectResults)
    : BakerEnrichmentResult =

    let startTime = DateTime.UtcNow

    // Validate precondition: PSG should have reachability marks
    let reachableCount =
        psg.Nodes
        |> Map.filter (fun _ node -> node.IsReachable)
        |> Map.count

    if reachableCount = 0 then
        printfn "[BAKER] WARNING: No reachable nodes found. Is reachability analysis complete?"

    printfn "[BAKER] Starting enrichment on PSG with %d nodes (%d reachable)"
        psg.Nodes.Count reachableCount

    // Run MemberBodyMapper to extract and correlate member bodies
    let mapperResult = MemberBodyMapper.run psg checkResults

    let elapsed = (DateTime.UtcNow - startTime).TotalMilliseconds

    printfn "[BAKER] Enrichment complete in %.1fms" elapsed
    printfn "[BAKER]   Member bodies extracted: %d" mapperResult.Statistics.MembersWithBodies
    printfn "[BAKER]   Correlated with PSG: %d" mapperResult.Statistics.MembersCorrelatedWithPSG

    {
        EnrichedPSG = psg  // PSG structure unchanged; bodies stored separately
        MemberBodies = mapperResult.MemberBodies
        Statistics = mapperResult.Statistics
    }

/// Look up a member body by full name
/// This is the primary API for Alex to find function bodies for inlining
let tryLookupBody
    (memberBodies: Map<string, MemberBodyMapping>)
    (fullName: string)
    : MemberBodyMapping option =

    // Try exact match first
    match Map.tryFind fullName memberBodies with
    | Some mapping -> Some mapping
    | None ->
        // Try with op_Dollar format for operators
        let altName = fullName.Replace("($)", "op_Dollar")
        Map.tryFind altName memberBodies

/// Look up a member body by declaring entity and member name
let tryLookupBodyByKey
    (memberBodies: Map<string, MemberBodyMapping>)
    (declaringEntity: string)
    (memberName: string)
    : MemberBodyMapping option =

    let fullName = sprintf "%s.%s" declaringEntity memberName
    tryLookupBody memberBodies fullName
