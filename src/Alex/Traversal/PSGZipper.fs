/// PSG Zipper - Context-preserving tree traversal for the Program Semantic Graph
///
/// A zipper maintains both a focus node and a path back to the root,
/// enabling bidirectional traversal while preserving context.
///
/// References:
/// - Huet, "The Zipper" (1997)
/// - Tomas Petricek's tree zipper implementation
/// - "Hyping Hypergraphs" blog post on temporal zippers
module Alex.Traversal.PSGZipper

open Core.PSG.Types
open FSharp.Compiler.Symbols

// ═══════════════════════════════════════════════════════════════════════════
// Symbol SSA Context - Symbol-keyed SSA value tracking for Zipper traversal
// ═══════════════════════════════════════════════════════════════════════════

/// SSA information for an emitted node
type SSAInfo = {
    Value: string      // e.g., "%v42"
    Type: string       // e.g., "!llvm.ptr", "i32"
}

/// Symbol-keyed SSA context built during Zipper traversal
/// Maps PSG nodes and symbols to their emitted SSA values
/// This replaces string-based local variable tracking with symbol-based lookup
type SymbolSSAContext = {
    /// Map from NodeId to SSA info - the primary lookup for emitted nodes
    NodeSSA: Map<string, SSAInfo>
    /// Map from Symbol to defining NodeId - for resolving Ident references
    SymbolDefs: Map<string, string>  // symbol key -> NodeId
}

module SymbolSSAContext =
    let empty = {
        NodeSSA = Map.empty
        SymbolDefs = Map.empty
    }

    /// Generate a stable key for a symbol
    /// Uses DeclarationLocation for local variables (stable across symbol instances)
    /// Falls back to DisplayName + hash for other symbols
    let symbolKey (sym: FSharpSymbol) : string =
        match sym with
        | :? FSharpMemberOrFunctionOrValue as mfv ->
            // For local variables and functions, use declaration location as unique identifier
            // This is stable because there's exactly one definition site
            let loc = mfv.DeclarationLocation
            sprintf "%s@%s:%d:%d" mfv.DisplayName loc.FileName loc.StartLine loc.StartColumn
        | _ ->
            // Fallback for other symbol types
            sprintf "%s_%d" sym.DisplayName (sym.GetHashCode())

    /// Record that a node produced an SSA value
    let recordNodeSSA (nodeId: NodeId) (ssa: SSAInfo) (ctx: SymbolSSAContext) : SymbolSSAContext =
        { ctx with NodeSSA = Map.add nodeId.Value ssa ctx.NodeSSA }

    /// Record that a symbol is defined at a node (for Binding nodes)
    let recordSymbolDef (sym: FSharpSymbol) (nodeId: NodeId) (ctx: SymbolSSAContext) : SymbolSSAContext =
        let key = symbolKey sym
        { ctx with SymbolDefs = Map.add key nodeId.Value ctx.SymbolDefs }

    /// Record both the SSA value and symbol definition in one call
    let recordBinding (sym: FSharpSymbol) (nodeId: NodeId) (ssa: SSAInfo) (ctx: SymbolSSAContext) : SymbolSSAContext =
        ctx
        |> recordNodeSSA nodeId ssa
        |> recordSymbolDef sym nodeId

    /// Look up SSA info for a node by its NodeId
    let lookupNode (nodeId: NodeId) (ctx: SymbolSSAContext) : SSAInfo option =
        Map.tryFind nodeId.Value ctx.NodeSSA

    /// Look up SSA info for an identifier by finding its defining node
    let lookupSymbol (sym: FSharpSymbol) (ctx: SymbolSSAContext) : SSAInfo option =
        let key = symbolKey sym
        match Map.tryFind key ctx.SymbolDefs with
        | Some defNodeId -> Map.tryFind defNodeId ctx.NodeSSA
        | None -> None

// ═══════════════════════════════════════════════════════════════════════════
// PSG Path - Zipper navigation context
// ═══════════════════════════════════════════════════════════════════════════

/// The path from the current focus back to the root.
/// Each step records what we "left behind" when descending.
type PSGPath =
    | Top
    /// Inside a binding: parent node, which child index we're at, siblings list
    | BindingChild of parent: PSGNode * childIndex: int * siblings: NodeId list * path: PSGPath
    /// Inside a function application: the function node, arg index, other args
    | AppArg of func: PSGNode * argIndex: int * otherArgs: NodeId list * path: PSGPath
    /// Inside a sequential expression: preceding nodes, following nodes
    | SequenceItem of preceding: NodeId list * following: NodeId list * parent: PSGNode * path: PSGPath
    /// Inside a match expression: scrutinee, case index, other cases
    | MatchCase of scrutinee: PSGNode * caseIndex: int * otherCases: NodeId list * path: PSGPath
    /// Generic child position for other node types
    | Child of parent: PSGNode * childIndex: int * siblings: NodeId list * path: PSGPath

/// Emission state carried during traversal
type EmissionState = {
    /// SSA counter for unique value names
    SSACounter: int
    /// Label counter for unique block labels
    LabelCounter: int
    /// Symbol-keyed SSA context - primary mechanism for variable resolution
    SymbolSSA: SymbolSSAContext
    /// Map from SSA name to MLIR type (for type lookups during emission)
    SSATypes: Map<string, string>
    /// Accumulated string literals: content -> global name
    StringLiterals: (string * string) list
    /// Current function name being emitted
    CurrentFunction: string option
    /// Block labels that have been emitted
    EmittedBlocks: Set<string>
    /// Map from mutable F# variable name to its stack slot SSA name (pointer)
    /// Mutable variables are stored on the stack to avoid SSA complications
    MutableSlots: Map<string, string * string>  // name -> (slot_ptr, element_type)
    // DEPRECATED: These are being replaced by SSAContext
    // Kept temporarily for backwards compatibility during migration
    Locals: Map<string, string>
    LocalTypes: Map<string, string>
}

module EmissionState =
    let empty = {
        SSACounter = 0
        LabelCounter = 0
        SymbolSSA = SymbolSSAContext.empty
        SSATypes = Map.empty
        StringLiterals = []
        CurrentFunction = None
        EmittedBlocks = Set.empty
        MutableSlots = Map.empty
        // DEPRECATED
        Locals = Map.empty
        LocalTypes = Map.empty
    }

    /// Generate next SSA value name
    let nextSSA state =
        let name = sprintf "%%v%d" state.SSACounter
        { state with SSACounter = state.SSACounter + 1 }, name

    /// Generate next SSA value with type tracking
    let nextSSAWithType mlirType state =
        let name = sprintf "%%v%d" state.SSACounter
        { state with
            SSACounter = state.SSACounter + 1
            SSATypes = Map.add name mlirType state.SSATypes }, name

    /// Generate next block label
    let nextLabel state =
        let name = sprintf "bb%d" state.LabelCounter
        { state with LabelCounter = state.LabelCounter + 1 }, name

    // ═══════════════════════════════════════════════════════════════════
    // NEW: Symbol-based SSA binding (preferred)
    // ═══════════════════════════════════════════════════════════════════

    /// Bind a symbol to an SSA value at a specific node
    /// Use this when emitting Binding nodes
    let bindSymbolSSA (sym: FSharpSymbol) (nodeId: NodeId) (ssaValue: string) (mlirType: string) (state: EmissionState) : EmissionState =
        let ssaInfo = { Value = ssaValue; Type = mlirType }
        { state with
            SymbolSSA = SymbolSSAContext.recordBinding sym nodeId ssaInfo state.SymbolSSA
            SSATypes = Map.add ssaValue mlirType state.SSATypes }

    /// Record an SSA value for a node (without symbol binding)
    /// Use this for intermediate expressions that don't define variables
    let recordNodeSSA (nodeId: NodeId) (ssaValue: string) (mlirType: string) (state: EmissionState) : EmissionState =
        let ssaInfo = { Value = ssaValue; Type = mlirType }
        { state with
            SymbolSSA = SymbolSSAContext.recordNodeSSA nodeId ssaInfo state.SymbolSSA
            SSATypes = Map.add ssaValue mlirType state.SSATypes }

    /// Look up SSA info for a symbol (for Ident nodes)
    let lookupSymbolSSA (sym: FSharpSymbol) (state: EmissionState) : SSAInfo option =
        SymbolSSAContext.lookupSymbol sym state.SymbolSSA

    /// Look up SSA info for a node by its ID
    let lookupNodeSSA (nodeId: NodeId) (state: EmissionState) : SSAInfo option =
        SymbolSSAContext.lookupNode nodeId state.SymbolSSA

    // ═══════════════════════════════════════════════════════════════════
    // DEPRECATED: String-based binding (for backwards compatibility)
    // These will be removed once migration is complete
    // ═══════════════════════════════════════════════════════════════════

    /// Bind an F# local to an SSA name (DEPRECATED - use bindSymbolSSA)
    let bindLocal fsharpName ssaName mlirType state =
        { state with
            Locals = Map.add fsharpName ssaName state.Locals
            LocalTypes = Map.add fsharpName mlirType state.LocalTypes
            SSATypes = Map.add ssaName mlirType state.SSATypes }

    /// Lookup a local variable's SSA name (DEPRECATED - use lookupSymbolSSA)
    let lookupLocal name state =
        Map.tryFind name state.Locals

    /// Lookup a local variable's MLIR type (DEPRECATED)
    let lookupLocalType name state =
        Map.tryFind name state.LocalTypes

    /// Lookup an SSA value's MLIR type
    let lookupSSAType ssaName state =
        Map.tryFind ssaName state.SSATypes

    /// Register a string literal
    let addStringLiteral content state =
        match state.StringLiterals |> List.tryFind (fun (c, _) -> c = content) with
        | Some (_, name) -> state, name
        | None ->
            let name = sprintf "@str%d" (List.length state.StringLiterals)
            { state with StringLiterals = (content, name) :: state.StringLiterals }, name

/// The zipper: focus node + path + graph context + emission state
type PSGZipper = {
    /// The current focus node
    Focus: PSGNode
    /// Path back to root
    Path: PSGPath
    /// The full graph (for node lookups)
    Graph: ProgramSemanticGraph
    /// Emission state
    State: EmissionState
}

/// Result type for navigation operations that may fail
type NavResult<'T> =
    | NavOk of 'T
    | NavFail of string

module PSGZipper =

    /// Create a zipper focused on a node
    let create (graph: ProgramSemanticGraph) (node: PSGNode) : PSGZipper =
        { Focus = node
          Path = Top
          Graph = graph
          State = EmissionState.empty }

    /// Create a zipper focused on a node with initial state
    let createWithState (graph: ProgramSemanticGraph) (node: PSGNode) (state: EmissionState) : PSGZipper =
        { Focus = node
          Path = Top
          Graph = graph
          State = state }

    /// Lookup a node by ID
    let lookupNode (nodeId: NodeId) (zipper: PSGZipper) : PSGNode option =
        Map.tryFind nodeId.Value zipper.Graph.Nodes

    /// Get children of the focus node as NodeIds (in source order)
    /// Note: Children are now in correct source order after PSG finalization
    let children (zipper: PSGZipper) : NodeId list =
        ChildrenStateHelpers.getChildrenList zipper.Focus

    /// Get children of the focus node as actual nodes (in source order)
    let childNodes (zipper: PSGZipper) : PSGNode list =
        children zipper
        |> List.choose (fun id -> lookupNode id zipper)

    // ─────────────────────────────────────────────────────────────
    // Navigation: Moving Down
    // ─────────────────────────────────────────────────────────────

    /// Move to the first child of the focus node
    let down (zipper: PSGZipper) : NavResult<PSGZipper> =
        let kids = children zipper
        match kids with
        | [] -> NavFail "No children to descend into"
        | firstId :: rest ->
            match lookupNode firstId zipper with
            | None -> NavFail (sprintf "Child node %s not found in graph" firstId.Value)
            | Some firstChild ->
                NavOk { zipper with
                         Focus = firstChild
                         Path = Child(zipper.Focus, 0, kids, zipper.Path) }

    /// Move to the nth child of the focus node
    let downTo (index: int) (zipper: PSGZipper) : NavResult<PSGZipper> =
        let kids = children zipper
        if index < 0 || index >= List.length kids then
            NavFail (sprintf "Child index %d out of range (0..%d)" index (List.length kids - 1))
        else
            let childId = List.item index kids
            match lookupNode childId zipper with
            | None -> NavFail (sprintf "Child node %s not found in graph" childId.Value)
            | Some child ->
                NavOk { zipper with
                         Focus = child
                         Path = Child(zipper.Focus, index, kids, zipper.Path) }

    // ─────────────────────────────────────────────────────────────
    // Navigation: Moving Up
    // ─────────────────────────────────────────────────────────────

    /// Move to the parent of the focus node
    let up (zipper: PSGZipper) : NavResult<PSGZipper> =
        match zipper.Path with
        | Top -> NavFail "Already at root"
        | Child(parent, _, _, path) ->
            NavOk { zipper with Focus = parent; Path = path }
        | BindingChild(parent, _, _, path) ->
            NavOk { zipper with Focus = parent; Path = path }
        | AppArg(func, _, _, path) ->
            NavOk { zipper with Focus = func; Path = path }
        | SequenceItem(_, _, parent, path) ->
            NavOk { zipper with Focus = parent; Path = path }
        | MatchCase(scrutinee, _, _, path) ->
            NavOk { zipper with Focus = scrutinee; Path = path }

    /// Navigate to the root of the tree
    let rec top (zipper: PSGZipper) : PSGZipper =
        match zipper.Path with
        | Top -> zipper
        | _ ->
            match up zipper with
            | NavOk parent -> top parent
            | NavFail _ -> zipper  // Should never happen

    // ─────────────────────────────────────────────────────────────
    // Navigation: Moving Sideways
    // ─────────────────────────────────────────────────────────────

    /// Move to the next sibling (right)
    let right (zipper: PSGZipper) : NavResult<PSGZipper> =
        match zipper.Path with
        | Top -> NavFail "At root, no siblings"
        | Child(parent, index, siblings, path) ->
            if index + 1 >= List.length siblings then
                NavFail "No more siblings to the right"
            else
                let nextId = List.item (index + 1) siblings
                match lookupNode nextId zipper with
                | None -> NavFail (sprintf "Sibling node %s not found" nextId.Value)
                | Some nextNode ->
                    NavOk { zipper with
                             Focus = nextNode
                             Path = Child(parent, index + 1, siblings, path) }
        | BindingChild(parent, index, siblings, path) ->
            if index + 1 >= List.length siblings then
                NavFail "No more binding siblings to the right"
            else
                let nextId = List.item (index + 1) siblings
                match lookupNode nextId zipper with
                | None -> NavFail (sprintf "Sibling node %s not found" nextId.Value)
                | Some nextNode ->
                    NavOk { zipper with
                             Focus = nextNode
                             Path = BindingChild(parent, index + 1, siblings, path) }
        | SequenceItem(preceding, following, parent, path) ->
            match following with
            | [] -> NavFail "No more sequence items to the right"
            | nextId :: rest ->
                match lookupNode nextId zipper with
                | None -> NavFail (sprintf "Sequence item %s not found" nextId.Value)
                | Some nextNode ->
                    NavOk { zipper with
                             Focus = nextNode
                             Path = SequenceItem(zipper.Focus.Id :: preceding, rest, parent, path) }
        | _ -> NavFail "Right navigation not supported for this path type"

    /// Move to the previous sibling (left)
    let left (zipper: PSGZipper) : NavResult<PSGZipper> =
        match zipper.Path with
        | Top -> NavFail "At root, no siblings"
        | Child(parent, index, siblings, path) ->
            if index <= 0 then
                NavFail "No more siblings to the left"
            else
                let prevId = List.item (index - 1) siblings
                match lookupNode prevId zipper with
                | None -> NavFail (sprintf "Sibling node %s not found" prevId.Value)
                | Some prevNode ->
                    NavOk { zipper with
                             Focus = prevNode
                             Path = Child(parent, index - 1, siblings, path) }
        | SequenceItem(preceding, following, parent, path) ->
            match preceding with
            | [] -> NavFail "No more sequence items to the left"
            | prevId :: rest ->
                match lookupNode prevId zipper with
                | None -> NavFail (sprintf "Sequence item %s not found" prevId.Value)
                | Some prevNode ->
                    NavOk { zipper with
                             Focus = prevNode
                             Path = SequenceItem(rest, zipper.Focus.Id :: following, parent, path) }
        | _ -> NavFail "Left navigation not supported for this path type"

    // ─────────────────────────────────────────────────────────────
    // Context Queries
    // ─────────────────────────────────────────────────────────────

    /// Check if we're at the root
    let isAtRoot (zipper: PSGZipper) : bool =
        match zipper.Path with
        | Top -> true
        | _ -> false

    /// Get the parent node without moving
    let parent (zipper: PSGZipper) : PSGNode option =
        match zipper.Path with
        | Top -> None
        | Child(p, _, _, _) -> Some p
        | BindingChild(p, _, _, _) -> Some p
        | AppArg(f, _, _, _) -> Some f
        | SequenceItem(_, _, p, _) -> Some p
        | MatchCase(s, _, _, _) -> Some s

    /// Get the index among siblings
    let siblingIndex (zipper: PSGZipper) : int option =
        match zipper.Path with
        | Top -> None
        | Child(_, i, _, _) -> Some i
        | BindingChild(_, i, _, _) -> Some i
        | AppArg(_, i, _, _) -> Some i
        | SequenceItem(preceding, _, _, _) -> Some (List.length preceding)
        | MatchCase(_, i, _, _) -> Some i

    /// Get ancestors as a list (immediate parent first)
    let rec ancestors (zipper: PSGZipper) : PSGNode list =
        match parent zipper with
        | None -> []
        | Some p ->
            match up zipper with
            | NavOk parentZipper -> p :: ancestors parentZipper
            | NavFail _ -> [p]

    /// Check if any ancestor matches a predicate
    let hasAncestorWhere (pred: PSGNode -> bool) (zipper: PSGZipper) : bool =
        ancestors zipper |> List.exists pred

    /// Find the nearest ancestor matching a predicate
    let findAncestor (pred: PSGNode -> bool) (zipper: PSGZipper) : PSGNode option =
        ancestors zipper |> List.tryFind pred

    // ─────────────────────────────────────────────────────────────
    // State Management
    // ─────────────────────────────────────────────────────────────

    /// Update the emission state
    let withState (newState: EmissionState) (zipper: PSGZipper) : PSGZipper =
        { zipper with State = newState }

    /// Map over the emission state
    let mapState (f: EmissionState -> EmissionState) (zipper: PSGZipper) : PSGZipper =
        { zipper with State = f zipper.State }

    /// Generate next SSA value
    let nextSSA (zipper: PSGZipper) : PSGZipper * string =
        let state', name = EmissionState.nextSSA zipper.State
        { zipper with State = state' }, name

    /// Generate next SSA value with type
    let nextSSAWithType mlirType (zipper: PSGZipper) : PSGZipper * string =
        let state', name = EmissionState.nextSSAWithType mlirType zipper.State
        { zipper with State = state' }, name

    /// Bind a local variable
    let bindLocal fsharpName ssaName mlirType (zipper: PSGZipper) : PSGZipper =
        { zipper with State = EmissionState.bindLocal fsharpName ssaName mlirType zipper.State }

    // ─────────────────────────────────────────────────────────────
    // Tree Traversal Combinators
    // ─────────────────────────────────────────────────────────────

    /// Traverse all children, threading state through
    let traverseChildren (f: PSGZipper -> PSGZipper * 'a) (zipper: PSGZipper) : PSGZipper * 'a list =
        let rec go acc currentZipper =
            match down currentZipper with
            | NavFail _ ->
                currentZipper, List.rev acc
            | NavOk childZipper ->
                let rec visitSiblings results z =
                    let z', result = f z
                    match right z' with
                    | NavFail _ ->
                        // Done with siblings - go back up
                        match up z' with
                        | NavOk parentZ -> parentZ, List.rev (result :: results)
                        | NavFail _ -> z', List.rev (result :: results)  // Should not happen
                    | NavOk nextZ ->
                        visitSiblings (result :: results) nextZ
                visitSiblings acc childZipper
        go [] zipper

    /// Fold over the tree in pre-order
    let foldPreOrder (folder: 'acc -> PSGZipper -> 'acc) (initial: 'acc) (zipper: PSGZipper) : 'acc =
        let rec go acc z =
            let acc' = folder acc z
            // Visit children
            let kids = childNodes z
            kids |> List.fold (fun a node ->
                match downTo (List.findIndex (fun n -> n.Id = node.Id) kids) z with
                | NavOk childZ -> go a childZ
                | NavFail _ -> a
            ) acc'
        go initial zipper

    /// Fold over the tree in post-order
    let foldPostOrder (folder: 'acc -> PSGZipper -> 'acc) (initial: 'acc) (zipper: PSGZipper) : 'acc =
        let rec go acc z =
            // Visit children first
            let kids = childNodes z
            let foldChild a node =
                let idx = kids |> List.findIndex (fun n -> n.Id = node.Id)
                match downTo idx z with
                | NavOk childZ -> go a childZ
                | NavFail _ -> a
            let acc' = kids |> List.fold foldChild acc
            // Then visit this node
            folder acc' z
        go initial zipper
