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

open FSharp.Compiler.CodeAnalysis
open Core.PSG.Types
open Baker.Types
open Baker.MemberBodyMapper
open Baker.TypedTreeZipper

/// Result of Baker enrichment including correlation state for emission
type BakerEnrichmentResult = {
    /// The enriched PSG (structure unchanged; lookups via CorrelationState)
    EnrichedPSG: ProgramSemanticGraph
    /// Member body mappings for body lookup (fullName -> mapping)
    MemberBodies: Map<string, MemberBodyMapping>
    /// Correlation state from TypedTreeZipper
    /// Contains range-indexed maps for O(1) lookup of:
    /// - FieldAccessByRange: field access info for PropertyAccess nodes
    /// - TypesByRange: resolved types
    /// - MemberBodies: member bodies (duplicated in MemberBodies above for convenience)
    CorrelationState: CorrelationState
    /// Statistics from Baker processing
    Statistics: BakerStatistics
}

/// Run Baker enrichment on a PSG that has completed reachability analysis (Phase 3)
///
/// PRECONDITION: PSG must have reachability marks (IsReachable field populated)
/// POST: PSG is enriched with member body mappings and correlation state
///
/// This is the Phase 4 (Typed Tree Overlay) step in the nanopass pipeline.
let enrich
    (psg: ProgramSemanticGraph)
    (checkResults: FSharpCheckProjectResults)
    : BakerEnrichmentResult =

    // Run MemberBodyMapper to extract member bodies and correlation state
    let mapperResult = MemberBodyMapper.run psg checkResults

    {
        EnrichedPSG = psg  // PSG structure unchanged; enrichment is in CorrelationState
        MemberBodies = mapperResult.MemberBodies
        CorrelationState = mapperResult.CorrelationState
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

/// Look up field access info by PSG node (O(1) via range-indexed map)
/// Use this when emitting PropertyAccess nodes
let tryLookupFieldAccess
    (correlationState: CorrelationState)
    (node: PSGNode)
    : FieldAccessInfo option =
    lookupFieldAccess correlationState node

/// Look up resolved type by PSG node (O(1) via range-indexed map)
let tryLookupType
    (correlationState: CorrelationState)
    (node: PSGNode)
    : FSharp.Compiler.Symbols.FSharpType option =
    lookupType correlationState node
