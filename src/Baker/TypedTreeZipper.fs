/// TypedTreeZipper - Two-tree zipper correlating FSharpExpr with PSGNode
///
/// This module walks the FCS typed tree and extracts semantic information
/// that the syntax-based PSG doesn't have: resolved types, SRTP resolutions,
/// field access info.
///
/// ARCHITECTURE NOTE (Dec 2024):
/// The original design attempted synchronized dual-tree navigation with
/// range-based correlation at every expression. This didn't work because:
/// 1. The typed tree and PSG have DIFFERENT structures (semantic vs syntactic)
/// 2. Range-based searching is O(n) per expression = O(n²) overall
/// 3. Ranges differ subtly between representations
///
/// Current approach:
/// - Phase A: Walk declarations, extract member bodies (for inlining)
/// - Phase B: Walk typed tree, build range-indexed info map
/// - At emission: O(1) lookup by PSG node range
///
/// CRITICAL: This operates AFTER reachability analysis (Phase 3).
module Baker.TypedTreeZipper

open FSharp.Compiler.Symbols
open FSharp.Compiler.Symbols.FSharpExprPatterns
open FSharp.Compiler.Text
open Core.PSG.Types
open Baker.Types

// ═══════════════════════════════════════════════════════════════════════════
// Range Key for Indexing
// ═══════════════════════════════════════════════════════════════════════════

/// Normalized range key for O(1) lookup
type RangeKey = {
    FileName: string
    StartLine: int
    StartColumn: int
    EndLine: int
    EndColumn: int
}

module RangeKey =
    let fromRange (r: range) : RangeKey =
        {
            FileName = System.IO.Path.GetFileName r.FileName
            StartLine = r.StartLine
            StartColumn = r.StartColumn
            EndLine = r.EndLine
            EndColumn = r.EndColumn
        }

    let fromPSGNode (node: PSGNode) : RangeKey =
        fromRange node.Range

// ═══════════════════════════════════════════════════════════════════════════
// Correlation State
// ═══════════════════════════════════════════════════════════════════════════

/// Accumulated state during traversal
type CorrelationState = {
    /// Member name -> (member, body) for function inlining
    MemberBodies: Map<string, FSharpMemberOrFunctionOrValue * FSharpExpr>
    /// Range -> FieldAccessInfo for struct field access
    FieldAccessByRange: Map<RangeKey, FieldAccessInfo>
    /// Range -> Resolved type
    TypesByRange: Map<RangeKey, FSharpType>
}

module CorrelationState =
    let empty = {
        MemberBodies = Map.empty
        FieldAccessByRange = Map.empty
        TypesByRange = Map.empty
    }

    let addMemberBody fullName mfv body state =
        { state with MemberBodies = Map.add fullName (mfv, body) state.MemberBodies }

    let addFieldAccess range info state =
        { state with FieldAccessByRange = Map.add range info state.FieldAccessByRange }

    let addType range ftype state =
        { state with TypesByRange = Map.add range ftype state.TypesByRange }

// ═══════════════════════════════════════════════════════════════════════════
// Field Access Extraction
// ═══════════════════════════════════════════════════════════════════════════

/// Find the field index for a field in its containing type
let private getFieldIndex (containingType: FSharpType) (field: FSharpField) : int =
    if containingType.HasTypeDefinition then
        let entity = containingType.TypeDefinition
        if entity.IsFSharpRecord || entity.IsValueType then
            entity.FSharpFields
            |> Seq.tryFindIndex (fun f -> f.Name = field.Name)
            |> Option.defaultValue 0
        else 0
    else 0

/// Extract FieldAccessInfo from FSharpFieldGet expression
let private extractFieldAccess (expr: FSharpExpr) : FieldAccessInfo option =
    match expr with
    | FSharpFieldGet(_, containingType, field) ->
        Some {
            ContainingType = containingType
            Field = field
            FieldName = field.Name
            FieldIndex = getFieldIndex containingType field
        }
    | _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// Expression Tree Walking (No PSG correlation - just extraction)
// ═══════════════════════════════════════════════════════════════════════════

/// Walk an expression tree and extract interesting info into state
let rec private walkExpression (expr: FSharpExpr) (state: CorrelationState) : CorrelationState =
    try
        let rangeKey = RangeKey.fromRange expr.Range

        // Record type for this expression (may throw on unsolved constraints)
        let state =
            try
                CorrelationState.addType rangeKey expr.Type state
            with _ -> state

        // Check for field access
        let state =
            match extractFieldAccess expr with
            | Some fieldInfo -> CorrelationState.addFieldAccess rangeKey fieldInfo state
            | None -> state

        // Recurse into children (may throw on expressions with constraint issues)
        try
            expr.ImmediateSubExpressions
            |> List.fold (fun s child -> walkExpression child s) state
        with _ -> state
    with _ -> state

// ═══════════════════════════════════════════════════════════════════════════
// Declaration Walking
// ═══════════════════════════════════════════════════════════════════════════

/// Process a single declaration
let rec private processDeclaration (decl: FSharpImplementationFileDeclaration) (state: CorrelationState) : CorrelationState =
    try
        match decl with
        | FSharpImplementationFileDeclaration.MemberOrFunctionOrValue(mfv, _args, body) ->
            // Record member body for inlining lookup
            let state =
                try
                    CorrelationState.addMemberBody mfv.FullName mfv body state
                with _ -> state

            // Walk the body to extract expression-level info
            walkExpression body state

        | FSharpImplementationFileDeclaration.Entity(_entity, decls) ->
            // Process nested declarations
            decls |> List.fold (fun s d -> processDeclaration d s) state

        | FSharpImplementationFileDeclaration.InitAction expr ->
            // Walk init action expression
            walkExpression expr state
    with _ -> state

/// Process all declarations in a file
let private processFile (file: FSharpImplementationFileContents) (state: CorrelationState) : CorrelationState =
    try
        file.Declarations
        |> List.fold (fun s decl -> processDeclaration decl s) state
    with _ -> state

// ═══════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════

/// Walk all implementation files and extract typed tree info
let extractTypedTreeInfo (implFiles: FSharpImplementationFileContents list) : CorrelationState =
    implFiles
    |> List.fold (fun state file -> processFile file state) CorrelationState.empty

/// Look up field access info by PSG node range
let lookupFieldAccess (state: CorrelationState) (node: PSGNode) : FieldAccessInfo option =
    let key = RangeKey.fromPSGNode node
    Map.tryFind key state.FieldAccessByRange

/// Look up resolved type by PSG node range
let lookupType (state: CorrelationState) (node: PSGNode) : FSharpType option =
    let key = RangeKey.fromPSGNode node
    Map.tryFind key state.TypesByRange

/// Look up member body by full name
let lookupMemberBody (state: CorrelationState) (fullName: string) : (FSharpMemberOrFunctionOrValue * FSharpExpr) option =
    Map.tryFind fullName state.MemberBodies
