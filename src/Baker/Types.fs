/// Baker Types - Type definitions for the post-reachability type resolution component library
///
/// Baker operates AFTER reachability analysis (Phase 3) and ONLY processes the narrowed
/// compute graph - nodes marked IsReachable = true.
///
/// This module defines:
/// - FieldAccessInfo: Field access information from typed tree
/// - MemberBodyMapping: Maps member symbols to their FSharpExpr bodies
/// - BakerStatistics: Processing statistics
module Baker.Types

open FSharp.Compiler.Text
open FSharp.Compiler.Symbols
open Core.PSG.Types

// ═══════════════════════════════════════════════════════════════════════════
// Field Access Types - Extracted from typed tree for struct field emission
// ═══════════════════════════════════════════════════════════════════════════

/// Field access information from FSharpFieldGet
/// Used for emitting llvm.extractvalue operations
type FieldAccessInfo = {
    /// The containing type (e.g., NativeStr)
    ContainingType: FSharpType
    /// The field being accessed
    Field: FSharpField
    /// Field name (e.g., "Length", "Pointer")
    FieldName: string
    /// Field index in the struct (for extraction)
    FieldIndex: int
}

// ═══════════════════════════════════════════════════════════════════════════
// SRTP Resolution Types - For future use
// ═══════════════════════════════════════════════════════════════════════════

/// SRTP resolution details (for future TraitCall handling)
type SRTPResolutionInfo = {
    /// The trait name (e.g., "op_Dollar")
    TraitName: string
    /// Source types with the constraint
    SourceTypes: FSharpType list
    /// Argument types
    ArgTypes: FSharpType list
    /// Return types
    ReturnTypes: FSharpType list
}

// ═══════════════════════════════════════════════════════════════════════════
// Member Body Types
// ═══════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════
// Statistics
// ═══════════════════════════════════════════════════════════════════════════

/// Statistics about Baker processing
type BakerStatistics = {
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
