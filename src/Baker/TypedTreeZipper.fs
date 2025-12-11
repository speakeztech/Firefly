/// TypedTreeZipper - Two-tree zipper correlating FSharpExpr with PSGNode
///
/// This zipper walks BOTH the FCS typed tree and the PSG in parallel,
/// maintaining synchronized position through structural context.
///
/// Key insight: The zipper's PATH provides the context needed for correlation.
/// We don't search for matching ranges - we navigate both trees in tandem,
/// and the structural position determines correspondence.
///
/// CRITICAL: This operates AFTER reachability analysis (Phase 3).
/// Only processes the narrowed compute graph.
///
/// Phase: Baker (Phase 4) - post-reachability type resolution
module Baker.TypedTreeZipper

open System
open FSharp.Compiler.Symbols
open FSharp.Compiler.Text
open Core.PSG.Types

// ═══════════════════════════════════════════════════════════════════════════
// Types for Typed Tree Navigation
// ═══════════════════════════════════════════════════════════════════════════

/// What kind of typed tree node we're focused on
type TypedFocus =
    /// At a file level (list of declarations)
    | AtFile of FSharpImplementationFileContents
    /// At a declaration (Entity, MemberOrFunctionOrValue, InitAction)
    | AtDeclaration of FSharpImplementationFileDeclaration
    /// At an expression within a member body
    | AtExpression of FSharpExpr

/// Path element in the typed tree - tracks where we came from
type TypedPathElement =
    /// Inside a file, at declaration index i of n
    | InFile of file: FSharpImplementationFileContents * index: int * siblings: FSharpImplementationFileDeclaration list
    /// Inside an entity (type/module), at declaration index i
    | InEntity of entity: FSharpEntity * index: int * siblings: FSharpImplementationFileDeclaration list
    /// Inside a member body, at child expression index i
    | InMember of mfv: FSharpMemberOrFunctionOrValue * args: FSharpMemberOrFunctionOrValue list list
    /// Inside an expression, at child index i
    | InExpression of parent: FSharpExpr * index: int * siblings: FSharpExpr list

/// Path through the typed tree (stack of crumbs back to root)
type TypedPath = TypedPathElement list

// ═══════════════════════════════════════════════════════════════════════════
// Types for PSG Navigation (simplified - uses existing PSGPath concepts)
// ═══════════════════════════════════════════════════════════════════════════

/// Path element in the PSG
type PSGPathElement =
    /// At a child of a parent node
    | PSGChild of parent: PSGNode * index: int * siblings: NodeId list

/// Path through the PSG
type PSGPath = PSGPathElement list

// ═══════════════════════════════════════════════════════════════════════════
// Correlation State - What we're collecting
// ═══════════════════════════════════════════════════════════════════════════

/// Information extracted from correlation
type CorrelationInfo = {
    /// PSG node ID
    NodeId: NodeId
    /// Resolved type from typed tree (if available)
    ResolvedType: FSharpType option
    /// SRTP resolution (if this is a TraitCall)
    SRTPResolution: SRTPResolutionInfo option
    /// Member body (if this is a binding with a body)
    MemberBody: FSharpExpr option
}

/// SRTP resolution details
and SRTPResolutionInfo = {
    /// The trait name (e.g., "op_Dollar")
    TraitName: string
    /// Source types with the constraint
    SourceTypes: FSharpType list
    /// Argument types
    ArgTypes: FSharpType list
    /// Return types
    ReturnTypes: FSharpType list
}

/// Accumulated correlations during traversal
type CorrelationState = {
    /// NodeId -> CorrelationInfo for nodes we've correlated
    Correlations: Map<string, CorrelationInfo>
    /// Member name -> (member, body) for bodies we've found
    MemberBodies: Map<string, FSharpMemberOrFunctionOrValue * FSharpExpr>
}

module CorrelationState =
    let empty = {
        Correlations = Map.empty
        MemberBodies = Map.empty
    }

    let addCorrelation nodeId info state =
        { state with Correlations = Map.add nodeId.Value info state.Correlations }

    let addMemberBody fullName mfv body state =
        { state with MemberBodies = Map.add fullName (mfv, body) state.MemberBodies }

// ═══════════════════════════════════════════════════════════════════════════
// The Two-Tree Zipper
// ═══════════════════════════════════════════════════════════════════════════

/// The TypedTreeZipper maintains synchronized position in both trees
type TypedTreeZipper = {
    /// Current focus in the typed tree
    TypedFocus: TypedFocus
    /// Path back to root in typed tree
    TypedPath: TypedPath
    /// Current focus in PSG (if correlated)
    PSGFocus: PSGNode option
    /// Path back to root in PSG
    PSGPath: PSGPath
    /// The full PSG for lookups
    Graph: ProgramSemanticGraph
    /// Accumulated correlation state
    State: CorrelationState
}

// ═══════════════════════════════════════════════════════════════════════════
// Zipper Creation
// ═══════════════════════════════════════════════════════════════════════════

module TypedTreeZipper =

    /// Create a zipper at the start of a file, correlated with PSG
    let createAtFile (file: FSharpImplementationFileContents) (psg: ProgramSemanticGraph) : TypedTreeZipper =
        // Find the root module node in PSG by matching the qualified name
        let moduleNode =
            psg.Nodes
            |> Map.toSeq
            |> Seq.tryFind (fun (_, node) ->
                node.SyntaxKind.StartsWith("Module") &&
                node.IsReachable)
            |> Option.map snd

        {
            TypedFocus = AtFile file
            TypedPath = []
            PSGFocus = moduleNode
            PSGPath = []
            Graph = psg
            State = CorrelationState.empty
        }

    // ─────────────────────────────────────────────────────────────────────
    // Context Queries - The "attention" of the zipper
    // ─────────────────────────────────────────────────────────────────────

    /// Get the current typed tree range (for correlation validation)
    let typedRange (zipper: TypedTreeZipper) : range option =
        match zipper.TypedFocus with
        | AtFile _ -> None
        | AtDeclaration decl ->
            match decl with
            | FSharpImplementationFileDeclaration.MemberOrFunctionOrValue (mfv, _, _) ->
                Some mfv.DeclarationLocation
            | FSharpImplementationFileDeclaration.Entity (entity, _) ->
                Some entity.DeclarationLocation
            | FSharpImplementationFileDeclaration.InitAction expr ->
                Some expr.Range
        | AtExpression expr ->
            Some expr.Range

    /// Get the current PSG range (for correlation validation)
    let psgRange (zipper: TypedTreeZipper) : range option =
        zipper.PSGFocus |> Option.map (fun n -> n.Range)

    /// Check if current positions are correlated (ranges overlap/contain)
    let isCorrelated (zipper: TypedTreeZipper) : bool =
        match typedRange zipper, psgRange zipper with
        | Some tr, Some pr ->
            // Check if they're on the same line and the typed range is within PSG range
            let sameFile = System.IO.Path.GetFileName(tr.FileName) = System.IO.Path.GetFileName(pr.FileName)
            let lineOverlap = tr.StartLine >= pr.StartLine && tr.EndLine <= pr.EndLine
            sameFile && lineOverlap
        | _ -> false

    /// Get the containing entity name from path
    let containingEntityName (zipper: TypedTreeZipper) : string option =
        zipper.TypedPath
        |> List.tryPick (function
            | InEntity (entity, _, _) -> entity.TryFullName
            | _ -> None)

    // ─────────────────────────────────────────────────────────────────────
    // PSG Correlation Helpers
    // ─────────────────────────────────────────────────────────────────────

    /// Find PSG binding node that contains the given range
    /// NOTE: We do NOT filter by IsReachable here because SRTP-resolved members
    /// may not be directly reachable through call graph edges, but we still need
    /// their bodies for inlining.
    let private findPSGBindingContaining (range: range) (psg: ProgramSemanticGraph) : PSGNode option =
        let fileName = System.IO.Path.GetFileName range.FileName
        psg.Nodes
        |> Map.toSeq
        |> Seq.tryFind (fun (_, node) ->
            node.SyntaxKind.StartsWith("Binding") &&
            let nodeFile = System.IO.Path.GetFileName node.Range.FileName
            nodeFile = fileName &&
            node.Range.StartLine <= range.StartLine &&
            node.Range.EndLine >= range.EndLine &&
            (node.Range.StartLine < range.StartLine || node.Range.StartColumn <= range.StartColumn))
        |> Option.map snd

    /// Find PSG child node within current focus that matches typed expr range
    let private findPSGChildMatching (range: range) (parentNode: PSGNode) (psg: ProgramSemanticGraph) : PSGNode option =
        let fileName = System.IO.Path.GetFileName range.FileName
        let children = ChildrenStateHelpers.getChildrenList parentNode
        children
        |> List.tryPick (fun childId ->
            match Map.tryFind childId.Value psg.Nodes with
            | Some child when child.IsReachable ->
                let childFile = System.IO.Path.GetFileName child.Range.FileName
                if childFile = fileName &&
                   child.Range.StartLine = range.StartLine &&
                   child.Range.StartColumn <= range.StartColumn &&
                   child.Range.EndColumn >= range.EndColumn then
                    Some child
                else
                    None
            | _ -> None)

    // ─────────────────────────────────────────────────────────────────────
    // Navigation: Moving into declarations
    // ─────────────────────────────────────────────────────────────────────

    /// Move into the declarations of a file
    let enterDeclarations (zipper: TypedTreeZipper) : TypedTreeZipper option =
        match zipper.TypedFocus with
        | AtFile file ->
            match file.Declarations with
            | [] -> None
            | first :: _ ->
                Some {
                    zipper with
                        TypedFocus = AtDeclaration first
                        TypedPath = InFile(file, 0, file.Declarations) :: zipper.TypedPath
                }
        | _ -> None

    /// Move into nested declarations of an entity
    let enterEntityDeclarations (zipper: TypedTreeZipper) : TypedTreeZipper option =
        match zipper.TypedFocus with
        | AtDeclaration (FSharpImplementationFileDeclaration.Entity (entity, decls)) ->
            match decls with
            | [] -> None
            | first :: _ ->
                Some {
                    zipper with
                        TypedFocus = AtDeclaration first
                        TypedPath = InEntity(entity, 0, decls) :: zipper.TypedPath
                }
        | _ -> None

    /// Move into the body of a member
    let enterMemberBody (zipper: TypedTreeZipper) : TypedTreeZipper option =
        match zipper.TypedFocus with
        | AtDeclaration (FSharpImplementationFileDeclaration.MemberOrFunctionOrValue (mfv, args, body)) ->
            // Find PSG binding that contains this member's declaration
            let psgBinding =
                match typedRange zipper with
                | Some range -> findPSGBindingContaining range zipper.Graph
                | None -> None

            Some {
                zipper with
                    TypedFocus = AtExpression body
                    TypedPath = InMember(mfv, args) :: zipper.TypedPath
                    PSGFocus = psgBinding
                    PSGPath =
                        match zipper.PSGFocus, psgBinding with
                        | Some parent, Some _ ->
                            let children = ChildrenStateHelpers.getChildrenList parent
                            PSGChild(parent, 0, children) :: zipper.PSGPath
                        | _ -> zipper.PSGPath
            }
        | _ -> None

    // ─────────────────────────────────────────────────────────────────────
    // Navigation: Moving to siblings
    // ─────────────────────────────────────────────────────────────────────

    /// Move to the next sibling declaration
    let nextDeclaration (zipper: TypedTreeZipper) : TypedTreeZipper option =
        match zipper.TypedPath with
        | InFile(file, index, siblings) :: rest ->
            if index + 1 < List.length siblings then
                let next = List.item (index + 1) siblings
                Some {
                    zipper with
                        TypedFocus = AtDeclaration next
                        TypedPath = InFile(file, index + 1, siblings) :: rest
                }
            else None
        | InEntity(entity, index, siblings) :: rest ->
            if index + 1 < List.length siblings then
                let next = List.item (index + 1) siblings
                Some {
                    zipper with
                        TypedFocus = AtDeclaration next
                        TypedPath = InEntity(entity, index + 1, siblings) :: rest
                }
            else None
        | _ -> None

    // ─────────────────────────────────────────────────────────────────────
    // Navigation: Moving up
    // ─────────────────────────────────────────────────────────────────────

    /// Move up to parent
    let up (zipper: TypedTreeZipper) : TypedTreeZipper option =
        match zipper.TypedPath with
        | InFile(file, _, _) :: rest ->
            Some {
                zipper with
                    TypedFocus = AtFile file
                    TypedPath = rest
            }
        | InEntity(entity, _, siblings) :: InFile(file, index, fileDecls) :: rest ->
            Some {
                zipper with
                    TypedFocus = AtDeclaration (FSharpImplementationFileDeclaration.Entity(entity, siblings))
                    TypedPath = InFile(file, index, fileDecls) :: rest
            }
        | InEntity(entity, _, siblings) :: InEntity(parentEntity, parentIndex, parentSiblings) :: rest ->
            // Nested entity within another entity
            Some {
                zipper with
                    TypedFocus = AtDeclaration (FSharpImplementationFileDeclaration.Entity(entity, siblings))
                    TypedPath = InEntity(parentEntity, parentIndex, parentSiblings) :: rest
            }
        | InEntity(entity, _, siblings) :: rest ->
            // Entity not directly under file - handle generically
            Some {
                zipper with
                    TypedFocus = AtDeclaration (FSharpImplementationFileDeclaration.Entity(entity, siblings))
                    TypedPath = rest
            }
        | InMember(_mfv, _args) :: rest ->
            // Reconstruct the declaration - we need the body, but we're inside it
            // This is tricky; for now, just pop the path
            Some {
                zipper with
                    TypedPath = rest
                    PSGPath = match zipper.PSGPath with _ :: t -> t | [] -> []
            }
        | InExpression(parent, _, _) :: rest ->
            Some {
                zipper with
                    TypedFocus = AtExpression parent
                    TypedPath = rest
            }
        | [] -> None

    // ─────────────────────────────────────────────────────────────────────
    // State Updates - Recording correlations
    // ─────────────────────────────────────────────────────────────────────

    /// Record that we found a member body
    let recordMemberBody (mfv: FSharpMemberOrFunctionOrValue) (body: FSharpExpr) (zipper: TypedTreeZipper) : TypedTreeZipper =
        try
            let fullName = mfv.FullName
            { zipper with State = CorrelationState.addMemberBody fullName mfv body zipper.State }
        with _ ->
            zipper

    /// Record correlation with PSG node
    let recordCorrelation (info: CorrelationInfo) (zipper: TypedTreeZipper) : TypedTreeZipper =
        { zipper with State = CorrelationState.addCorrelation info.NodeId info zipper.State }

    // ─────────────────────────────────────────────────────────────────────
    // Extraction - Getting SRTP info from TraitCall
    // ─────────────────────────────────────────────────────────────────────

    /// Check if current expression is a TraitCall and extract SRTP info
    let extractSRTPInfo (zipper: TypedTreeZipper) : SRTPResolutionInfo option =
        match zipper.TypedFocus with
        | AtExpression expr ->
            // FSharpExpr doesn't directly expose TraitCall as a pattern in the public API
            // We need to check the expression's string representation or use reflection
            // For now, we check if it's a call that might be SRTP
            // TODO: Use proper FCS API for TraitCall detection
            None
        | _ -> None

    // ─────────────────────────────────────────────────────────────────────
    // Traversal - Walk all declarations and collect member bodies
    // ─────────────────────────────────────────────────────────────────────

    /// Process a single declaration, recording member bodies
    let rec private processDeclaration (zipper: TypedTreeZipper) : TypedTreeZipper =
        match zipper.TypedFocus with
        | AtDeclaration (FSharpImplementationFileDeclaration.MemberOrFunctionOrValue (mfv, args, body)) ->
            // Record this member body
            let zipper' = recordMemberBody mfv body zipper

            // Also try to find and correlate with PSG binding
            match typedRange zipper with
            | Some range ->
                match findPSGBindingContaining range zipper.Graph with
                | Some psgNode ->
                    let info = {
                        NodeId = psgNode.Id
                        ResolvedType = Some body.Type
                        SRTPResolution = None
                        MemberBody = Some body
                    }
                    recordCorrelation info zipper'
                | None -> zipper'
            | None -> zipper'

        | AtDeclaration (FSharpImplementationFileDeclaration.Entity (entity, decls)) ->
            // Process nested declarations
            match enterEntityDeclarations zipper with
            | Some innerZipper ->
                let processed = processAllDeclarations innerZipper
                // Move back up and continue
                match up processed with
                | Some upZipper -> upZipper
                | None -> processed
            | None -> zipper

        | AtDeclaration (FSharpImplementationFileDeclaration.InitAction _) ->
            // Init actions don't have named bodies to correlate
            zipper

        | _ -> zipper

    /// Process all declarations at current level
    and private processAllDeclarations (zipper: TypedTreeZipper) : TypedTreeZipper =
        let zipper' = processDeclaration zipper
        match nextDeclaration zipper' with
        | Some next -> processAllDeclarations next
        | None -> zipper'

    /// Walk the entire typed tree and collect correlations
    let walkAndCorrelate (zipper: TypedTreeZipper) : TypedTreeZipper =
        match enterDeclarations zipper with
        | Some declZipper -> processAllDeclarations declZipper
        | None -> zipper

// ═══════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════

/// Walk a file's typed tree and correlate with PSG
let correlateFile (file: FSharpImplementationFileContents) (psg: ProgramSemanticGraph) : CorrelationState =
    let zipper = TypedTreeZipper.createAtFile file psg
    let result = TypedTreeZipper.walkAndCorrelate zipper
    result.State

/// Walk all implementation files and collect correlations
let correlateAll (implFiles: FSharpImplementationFileContents list) (psg: ProgramSemanticGraph) : CorrelationState =
    implFiles
    |> List.fold (fun state file ->
        let fileState = correlateFile file psg
        // Merge states
        {
            Correlations = Map.fold (fun acc k v -> Map.add k v acc) state.Correlations fileState.Correlations
            MemberBodies = Map.fold (fun acc k v -> Map.add k v acc) state.MemberBodies fileState.MemberBodies
        }
    ) CorrelationState.empty
