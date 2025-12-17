/// PSG Zipper - Context-preserving tree traversal for the Program Semantic Graph
///
/// A zipper maintains both a focus node and a path back to the root,
/// enabling bidirectional traversal while preserving context.
///
/// The zipper carries minimal emission-time state (SSA counters, string literals).
/// Variable resolution uses PSG def-use edges, NOT imperative scope tracking.
///
/// References:
/// - Huet, "The Zipper" (1997)
/// - Tomas Petricek's tree zipper implementation
/// - "Hyping Hypergraphs" blog post on temporal zippers
module Alex.Traversal.PSGZipper

open Core.PSG.Types
open Core.PSG.NavigationUtils

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
///
/// MINIMAL: Only what's needed for MLIR emission mechanics.
/// Variable resolution uses PSG def-use edges, NOT this state.
type EmissionState = {
    /// SSA counter for unique value names
    SSACounter: int
    /// Label counter for unique block labels
    LabelCounter: int
    /// Map from SSA name to MLIR type (for type lookups during emission)
    SSATypes: Map<string, string>
    /// Accumulated string literals: content -> global name
    StringLiterals: (string * string) list
    /// Current function name being emitted
    CurrentFunction: string option
    /// Block labels that have been emitted
    EmittedBlocks: Set<string>
    /// Map from NodeId to emitted SSA value
    /// When we emit a binding, we record its SSA here.
    /// When we need to resolve a use, we follow PSG def-use edges to find the defining node,
    /// then look up that node's SSA value here.
    NodeSSA: Map<string, string * string>  // NodeId.Value -> (ssa_value, mlir_type)
}

module EmissionState =
    let empty = {
        SSACounter = 0
        LabelCounter = 0
        SSATypes = Map.empty
        StringLiterals = []
        CurrentFunction = None
        EmittedBlocks = Set.empty
        NodeSSA = Map.empty
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

    /// Record that a node was emitted with a given SSA value
    /// Called when emitting Binding nodes
    let recordNodeSSA (nodeId: NodeId) (ssaValue: string) (mlirType: string) (state: EmissionState) : EmissionState =
        { state with
            NodeSSA = Map.add nodeId.Value (ssaValue, mlirType) state.NodeSSA
            SSATypes = Map.add ssaValue mlirType state.SSATypes }

    /// Look up SSA value for a node by its ID
    let lookupNodeSSA (nodeId: NodeId) (state: EmissionState) : (string * string) option =
        Map.tryFind nodeId.Value state.NodeSSA

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
    /// Uses ChildOf edges as the authoritative source of graph structure
    let children (zipper: PSGZipper) : NodeId list =
        getChildIds zipper.Graph zipper.Focus

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

    /// Record that a node was emitted with a given SSA value
    let recordNodeSSA (nodeId: NodeId) (ssaValue: string) (mlirType: string) (zipper: PSGZipper) : PSGZipper =
        { zipper with State = EmissionState.recordNodeSSA nodeId ssaValue mlirType zipper.State }

    /// Look up SSA value for a node by its ID
    let lookupNodeSSA (nodeId: NodeId) (zipper: PSGZipper) : (string * string) option =
        EmissionState.lookupNodeSSA nodeId zipper.State

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
