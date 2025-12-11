/// Baker Types - Type definitions for the post-reachability type resolution component library
///
/// Baker operates AFTER reachability analysis (Phase 3) and ONLY processes the narrowed
/// compute graph - nodes marked IsReachable = true.
///
/// This module defines:
/// - MemberBodyMapping: Maps member symbols to their FSharpExpr bodies
/// - BakerContext: Accumulated state during Baker processing
module Baker.Types

open FSharp.Compiler.Text
open FSharp.Compiler.Symbols
open Core.PSG.Types

/// Maps a member (function, static member, etc.) to its typed expression body.
/// Used to look up bodies for inlining during MLIR emission.
type MemberBodyMapping = {
    /// The member symbol (from FCS)
    Member: FSharpMemberOrFunctionOrValue
    /// Declaration source range (for correlation)
    DeclarationRange: range
    /// The typed expression body (FSharpExpr from FCS typed tree)
    Body: FSharpExpr
    /// Corresponding PSG binding node ID (if correlated)
    PSGBindingId: NodeId option
}

/// Key for looking up member bodies.
/// Uses (DeclaringEntityName, MemberName) for fast lookup.
type MemberBodyKey = {
    /// Declaring type or module full name
    DeclaringEntityFullName: string
    /// Member display name
    MemberDisplayName: string
}

module MemberBodyKey =
    /// Create a key from a member symbol
    let fromMember (mfv: FSharpMemberOrFunctionOrValue) : MemberBodyKey option =
        try
            let declaringEntity =
                match mfv.DeclaringEntity with
                | Some entity ->
                    match entity.TryFullName with
                    | Some name -> name
                    | None -> entity.DisplayName
                | None -> ""
            if declaringEntity = "" then None
            else
                Some {
                    DeclaringEntityFullName = declaringEntity
                    MemberDisplayName = mfv.DisplayName
                }
        with _ -> None

    /// Create a key from a full name string like "Module.Type.Member"
    let fromFullName (fullName: string) : MemberBodyKey option =
        let parts = fullName.Split('.')
        if parts.Length >= 2 then
            let memberName = parts.[parts.Length - 1]
            let declaringEntity = String.concat "." (parts |> Array.take (parts.Length - 1))
            Some {
                DeclaringEntityFullName = declaringEntity
                MemberDisplayName = memberName
            }
        else None

/// Result of Baker enrichment phase
type BakerResult = {
    /// Member body mappings (key -> mapping)
    MemberBodies: Map<string, MemberBodyMapping>
    /// Statistics about the enrichment
    Statistics: BakerStatistics
}

/// Statistics about Baker processing
and BakerStatistics = {
    /// Total members processed
    TotalMembers: int
    /// Members with bodies extracted
    MembersWithBodies: int
    /// Members correlated with PSG bindings
    MembersCorrelatedWithPSG: int
    /// Processing time in milliseconds
    ProcessingTimeMs: float
}

module BakerStatistics =
    let empty = {
        TotalMembers = 0
        MembersWithBodies = 0
        MembersCorrelatedWithPSG = 0
        ProcessingTimeMs = 0.0
    }
