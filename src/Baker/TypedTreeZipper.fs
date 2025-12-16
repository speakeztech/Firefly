/// TypedTreeZipper - Two-tree zipper for FSharpExpr/PSG correlation
///
/// This module implements the dual-tree zipper described in the Baker architecture:
/// - Maintains synchronized position in BOTH typed tree AND PSG
/// - Navigation is O(1): up, down, left, right
/// - Structural position determines correspondence (no range searching)
///
/// The zipper "carries context" - as we navigate, we know where we are in both trees.
module Baker.TypedTreeZipper

open FSharp.Compiler.Symbols
open FSharp.Compiler.Symbols.FSharpExprPatterns
open FSharp.Compiler.Text
open Core.PSG.Types
open Baker.Types

// ═══════════════════════════════════════════════════════════════════════════
// Typed Tree Zipper - Navigation through FSharpExpr
// ═══════════════════════════════════════════════════════════════════════════

/// Breadcrumb for typed tree navigation - records how we got here
type TypedCrumb = {
    /// Parent expression
    Parent: FSharpExpr
    /// Index of current child in parent's ImmediateSubExpressions
    ChildIndex: int
    /// Siblings before current position
    Before: FSharpExpr list
    /// Siblings after current position
    After: FSharpExpr list
}

/// Zipper for navigating FSharpExpr tree
type TypedZipper = {
    /// Current focused expression
    Focus: FSharpExpr
    /// Path back to root (stack of breadcrumbs)
    Path: TypedCrumb list
}

module TypedZipper =
    /// Create zipper focused on an expression (at root)
    let create (expr: FSharpExpr) : TypedZipper =
        { Focus = expr; Path = [] }

    /// Move down to first child, if any
    /// Protected against FCS constraint solver exceptions
    let down (z: TypedZipper) : TypedZipper option =
        try
            let children = z.Focus.ImmediateSubExpressions
            match children with
            | [] -> None
            | first :: rest ->
                Some {
                    Focus = first
                    Path = {
                        Parent = z.Focus
                        ChildIndex = 0
                        Before = []
                        After = rest
                    } :: z.Path
                }
        with _ -> None

    /// Move up to parent
    let up (z: TypedZipper) : TypedZipper option =
        match z.Path with
        | [] -> None
        | crumb :: rest ->
            Some { Focus = crumb.Parent; Path = rest }

    /// Move to next sibling (right)
    let right (z: TypedZipper) : TypedZipper option =
        match z.Path with
        | [] -> None
        | crumb :: rest ->
            match crumb.After with
            | [] -> None
            | next :: afterNext ->
                Some {
                    Focus = next
                    Path = {
                        Parent = crumb.Parent
                        ChildIndex = crumb.ChildIndex + 1
                        Before = z.Focus :: crumb.Before
                        After = afterNext
                    } :: rest
                }

    /// Move to previous sibling (left)
    let left (z: TypedZipper) : TypedZipper option =
        match z.Path with
        | [] -> None
        | crumb :: rest ->
            match crumb.Before with
            | [] -> None
            | prev :: beforePrev ->
                Some {
                    Focus = prev
                    Path = {
                        Parent = crumb.Parent
                        ChildIndex = crumb.ChildIndex - 1
                        Before = beforePrev
                        After = z.Focus :: crumb.After
                    } :: rest
                }

// ═══════════════════════════════════════════════════════════════════════════
// PSG Navigation - For Baker's internal use (distinct from Alex.Traversal.PSGZipper)
// ═══════════════════════════════════════════════════════════════════════════

/// Breadcrumb for PSG navigation
type BakerPSGCrumb = {
    /// Parent node ID
    ParentId: NodeId
    /// Index of current child among parent's children
    ChildIndex: int
    /// Sibling IDs before current
    Before: NodeId list
    /// Sibling IDs after current
    After: NodeId list
}

/// Baker's internal PSG zipper (separate from Alex.Traversal.PSGZipper)
type BakerPSGZipper = {
    /// Current focused node
    FocusNode: PSGNode
    /// Path back to root
    Crumbs: BakerPSGCrumb list
    /// The full graph (for node lookups)
    FullGraph: ProgramSemanticGraph
}

module BakerPSGZipper =
    /// Get children of a node from the graph (nodes where edge.Target = nodeId)
    let private getChildren (graph: ProgramSemanticGraph) (nodeId: NodeId) : PSGNode list =
        graph.Edges
        |> List.choose (fun edge ->
            match edge.Kind with
            | ChildOf when edge.Target = nodeId ->
                Map.tryFind edge.Source.Value graph.Nodes
            | _ -> None)
        |> List.sortBy (fun n -> n.Range.StartLine, n.Range.StartColumn)

    /// Create zipper focused on a node
    let create (graph: ProgramSemanticGraph) (node: PSGNode) : BakerPSGZipper =
        { FocusNode = node; Crumbs = []; FullGraph = graph }

    /// Move down to first child
    let down (z: BakerPSGZipper) : BakerPSGZipper option =
        let children = getChildren z.FullGraph z.FocusNode.Id
        match children with
        | [] -> None
        | first :: rest ->
            Some {
                FocusNode = first
                Crumbs = {
                    ParentId = z.FocusNode.Id
                    ChildIndex = 0
                    Before = []
                    After = rest |> List.map (fun n -> n.Id)
                } :: z.Crumbs
                FullGraph = z.FullGraph
            }

    /// Move up to parent
    let up (z: BakerPSGZipper) : BakerPSGZipper option =
        match z.Crumbs with
        | [] -> None
        | crumb :: rest ->
            match Map.tryFind crumb.ParentId.Value z.FullGraph.Nodes with
            | None -> None
            | Some parent ->
                Some { FocusNode = parent; Crumbs = rest; FullGraph = z.FullGraph }

    /// Move to next sibling
    let right (z: BakerPSGZipper) : BakerPSGZipper option =
        match z.Crumbs with
        | [] -> None
        | crumb :: rest ->
            match crumb.After with
            | [] -> None
            | nextId :: afterNext ->
                match Map.tryFind nextId.Value z.FullGraph.Nodes with
                | None -> None
                | Some next ->
                    Some {
                        FocusNode = next
                        Crumbs = {
                            ParentId = crumb.ParentId
                            ChildIndex = crumb.ChildIndex + 1
                            Before = z.FocusNode.Id :: crumb.Before
                            After = afterNext
                        } :: rest
                        FullGraph = z.FullGraph
                    }

// ═══════════════════════════════════════════════════════════════════════════
// Correlation State
// ═══════════════════════════════════════════════════════════════════════════

/// Correlation state accumulated during traversal
type CorrelationState = {
    /// NodeId.Value -> FieldAccessInfo for PropertyAccess nodes
    FieldAccess: Map<string, FieldAccessInfo>
    /// NodeId.Value -> resolved FSharpType
    Types: Map<string, FSharpType>
    /// Member fullName -> (member, body) for inlining
    MemberBodies: Map<string, FSharpMemberOrFunctionOrValue * FSharpExpr>
}

module CorrelationState =
    let empty : CorrelationState = {
        FieldAccess = Map.empty
        Types = Map.empty
        MemberBodies = Map.empty
    }

    let addFieldAccess nodeId info state =
        { state with FieldAccess = Map.add nodeId info state.FieldAccess }

    let addType nodeId ftype state =
        { state with Types = Map.add nodeId ftype state.Types }

    let addMemberBody fullName mfv body state =
        { state with MemberBodies = Map.add fullName (mfv, body) state.MemberBodies }

/// The dual-tree zipper - maintains synchronized position in both trees
type DualZipper = {
    /// Position in typed tree
    Typed: TypedZipper
    /// Position in PSG (None if no correlation at this point)
    PSG: BakerPSGZipper option
    /// Accumulated correlations
    State: CorrelationState
}

// ═══════════════════════════════════════════════════════════════════════════
// Field Access Extraction
// ═══════════════════════════════════════════════════════════════════════════

/// Find field index in containing type (protected against FCS exceptions)
let private getFieldIndex (containingType: FSharpType) (field: FSharpField) : int =
    try
        if containingType.HasTypeDefinition then
            let entity = containingType.TypeDefinition
            if entity.IsFSharpRecord || entity.IsValueType then
                entity.FSharpFields
                |> Seq.tryFindIndex (fun f -> f.Name = field.Name)
                |> Option.defaultValue 0
            else 0
        else 0
    with _ -> 0

/// Extract FieldAccessInfo from current typed tree focus (protected against FCS exceptions)
let private extractFieldAccess (expr: FSharpExpr) : FieldAccessInfo option =
    try
        match expr with
        | FSharpFieldGet(_, containingType, field) ->
            Some {
                ContainingType = containingType
                Field = field
                FieldName = field.Name
                FieldIndex = getFieldIndex containingType field
            }
        | _ -> None
    with _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// Correlation Logic
// ═══════════════════════════════════════════════════════════════════════════

/// Record correlation at current synchronized position
/// Protected against FCS constraint solver exceptions
let private recordCorrelation (dual: DualZipper) : CorrelationState =
    try
        match dual.PSG with
        | None -> dual.State
        | Some psgZipper ->
            let nodeId = psgZipper.FocusNode.Id.Value
            let expr = dual.Typed.Focus

            // Record type (may throw for SRTP expressions)
            let state =
                try
                    CorrelationState.addType nodeId expr.Type dual.State
                with _ -> dual.State

            // Record field access if applicable
            match extractFieldAccess expr with
            | Some fieldInfo -> CorrelationState.addFieldAccess nodeId fieldInfo state
            | None -> state
    with _ ->
        // If anything fails, just return the unchanged state
        dual.State

// ═══════════════════════════════════════════════════════════════════════════
// Synchronized Traversal
// ═══════════════════════════════════════════════════════════════════════════

/// Try to find initial PSG correlation for a typed tree expression by range
let private findPSGNodeByRange (graph: ProgramSemanticGraph) (range: range) : PSGNode option =
    graph.Nodes
    |> Map.tryPick (fun _ node ->
        if node.Range.FileName = range.FileName &&
           node.Range.StartLine = range.StartLine &&
           node.Range.StartColumn = range.StartColumn then
            Some node
        else None)

/// Walk the dual zipper, recording correlations at each step
let rec private walkDual (dual: DualZipper) : CorrelationState =
    // Record correlation at current position
    let state = recordCorrelation { dual with State = dual.State }

    // Try to descend into children
    match TypedZipper.down dual.Typed with
    | None ->
        // No children, we're at a leaf - return accumulated state
        state
    | Some typedChild ->
        // Descend in PSG too if we have correlation
        let psgChild = dual.PSG |> Option.bind BakerPSGZipper.down

        // Walk the first child
        let stateAfterChild = walkDual {
            Typed = typedChild
            PSG = psgChild
            State = state
        }

        // Walk remaining siblings
        walkSiblings { dual with Typed = typedChild; PSG = psgChild; State = stateAfterChild }

and private walkSiblings (dual: DualZipper) : CorrelationState =
    match TypedZipper.right dual.Typed with
    | None ->
        // No more siblings
        dual.State
    | Some typedNext ->
        // Move right in PSG too
        let psgNext = dual.PSG |> Option.bind BakerPSGZipper.right

        // Record correlation for this sibling
        let state = recordCorrelation { Typed = typedNext; PSG = psgNext; State = dual.State }

        // Walk this sibling's children
        let stateAfterChildren =
            match TypedZipper.down typedNext with
            | None -> state
            | Some typedChild ->
                let psgChild = psgNext |> Option.bind BakerPSGZipper.down
                walkDual { Typed = typedChild; PSG = psgChild; State = state }

        // Continue to next sibling
        walkSiblings { Typed = typedNext; PSG = psgNext; State = stateAfterChildren }

// ═══════════════════════════════════════════════════════════════════════════
// Declaration Processing
// ═══════════════════════════════════════════════════════════════════════════

/// Process a member declaration - correlate body with PSG
/// Protected against FCS exceptions
/// OPTIMIZATION: Skip unreachable nodes to focus processing on reachable code only
let private processMember
    (graph: ProgramSemanticGraph)
    (mfv: FSharpMemberOrFunctionOrValue)
    (body: FSharpExpr)
    (state: CorrelationState)
    : CorrelationState =
    try
        // Try to find corresponding PSG node for the body
        let psgNode =
            try findPSGNodeByRange graph body.Range
            with _ -> None

        // OPTIMIZATION: Skip unreachable nodes - they won't be emitted anyway
        // This significantly reduces processing time for large libraries like Alloy
        match psgNode with
        | Some node when not node.IsReachable ->
            // Skip unreachable member - no need to record body or walk tree
            state
        | _ ->
            // Record member body for inlining (only for reachable or unknown nodes)
            let state =
                try
                    CorrelationState.addMemberBody mfv.FullName mfv body state
                with _ -> state

            // Create dual zipper and walk
            let dual = {
                Typed = TypedZipper.create body
                PSG = psgNode |> Option.map (BakerPSGZipper.create graph)
                State = state
            }

            walkDual dual
    with _ -> state

/// Process all declarations in a file
/// Protected against FCS exceptions
/// OPTIMIZATION: Skip unreachable nodes
let rec private processDeclaration
    (graph: ProgramSemanticGraph)
    (decl: FSharpImplementationFileDeclaration)
    (state: CorrelationState)
    : CorrelationState =
    try
        match decl with
        | FSharpImplementationFileDeclaration.MemberOrFunctionOrValue(mfv, _args, body) ->
            processMember graph mfv body state

        | FSharpImplementationFileDeclaration.Entity(_entity, decls) ->
            decls |> List.fold (fun s d -> processDeclaration graph d s) state

        | FSharpImplementationFileDeclaration.InitAction body ->
            let psgNode =
                try findPSGNodeByRange graph body.Range
                with _ -> None

            // OPTIMIZATION: Skip unreachable init actions
            match psgNode with
            | Some node when not node.IsReachable ->
                state
            | _ ->
                let dual = {
                    Typed = TypedZipper.create body
                    PSG = psgNode |> Option.map (BakerPSGZipper.create graph)
                    State = state
                }
                walkDual dual
    with _ -> state

/// Process a single file
let private processFile
    (graph: ProgramSemanticGraph)
    (file: FSharpImplementationFileContents)
    (state: CorrelationState)
    : CorrelationState =

    file.Declarations
    |> List.fold (fun s d -> processDeclaration graph d s) state

// ═══════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════

/// Correlate typed tree with PSG using dual-tree zipper traversal
let correlate
    (graph: ProgramSemanticGraph)
    (implFiles: FSharpImplementationFileContents list)
    : CorrelationState =

    implFiles
    |> List.fold (fun state file -> processFile graph file state) CorrelationState.empty

/// Look up field access info by PSG node ID
let lookupFieldAccess (state: CorrelationState) (nodeId: NodeId) : FieldAccessInfo option =
    Map.tryFind nodeId.Value state.FieldAccess

/// Look up resolved type by PSG node ID
let lookupType (state: CorrelationState) (nodeId: NodeId) : FSharpType option =
    Map.tryFind nodeId.Value state.Types

/// Look up member body by full name
let lookupMemberBody (state: CorrelationState) (fullName: string) : (FSharpMemberOrFunctionOrValue * FSharpExpr) option =
    Map.tryFind fullName state.MemberBodies
