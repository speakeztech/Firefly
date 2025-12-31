/// FNCS Integration Layer
/// Thin interface between Firefly and F# Native Compiler Services.
///
/// FNCS provides:
/// - Native type checking with types attached during construction
/// - SRTP resolution during type checking (not post-hoc)
/// - Hard-pruned SemanticGraph (only reachable nodes)
/// - No BCL types, no IL imports, no obj
///
/// Firefly receives SemanticGraph and applies:
/// - Lowering nanopasses (FlattenApplications, LowerStrings, etc.)
/// - Alex emission (Zipper + XParsec + Bindings → MLIR)
module Core.FNCS.Integration

// Re-export FNCS types for use throughout Firefly
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.NativeService

/// Type aliases for cleaner code
type FNCSNode = SemanticNode
type FNCSGraph = SemanticGraph
type FNCSType = NativeType
type FNCSKind = SemanticKind
type FNCSNodeId = NodeId
type FNCSLiteralValue = LiteralValue

/// Check result from FNCS
type FNCSCheckResult = CheckResult

/// Extract int from NodeId
let nodeIdToInt (NodeId id) = id

/// Get the ID of a node
let nodeId (node: FNCSNode) : FNCSNodeId = node.Id

/// Get the type of a node
let nodeType (node: FNCSNode) : FNCSType = node.Type

/// Get the kind of a node
let nodeKind (node: FNCSNode) : FNCSKind = node.Kind

/// Get the SRTP resolution if present
let nodeSRTP (node: FNCSNode) : WitnessResolution option = node.SRTPResolution

/// Get children of a node
let nodeChildren (node: FNCSNode) : NodeId list = node.Children

/// Get a node by ID
let getNode (id: NodeId) (graph: FNCSGraph) : FNCSNode option =
    SemanticGraph.tryGetNode id graph

/// Get a node by ID (throws if not found)
let getNodeExn (id: NodeId) (graph: FNCSGraph) : FNCSNode =
    SemanticGraph.getNode id graph

/// Fold over graph in post-order (children before parents - required for SSA emission)
let foldPostOrder (folder: 'State -> FNCSNode -> 'State) (state: 'State) (graph: FNCSGraph) : 'State =
    Traversal.foldPostOrder folder state graph

/// Fold over graph in pre-order
let foldPreOrder (folder: 'State -> FNCSNode -> 'State) (state: 'State) (graph: FNCSGraph) : 'State =
    Traversal.foldPreOrder folder state graph

/// Map over all nodes in the graph
let mapNodes (f: FNCSNode -> FNCSNode) (graph: FNCSGraph) : FNCSGraph =
    Traversal.map f graph

/// Filter nodes in the graph
let filterNodes (predicate: FNCSNode -> bool) (graph: FNCSGraph) : FNCSGraph =
    Traversal.filter predicate graph

/// Get all binding nodes
let bindings (graph: FNCSGraph) : FNCSNode list =
    SemanticGraph.bindings graph

/// Check if result has errors
let hasErrors (result: FNCSCheckResult) : bool =
    CheckResult.hasErrors result

/// Get error diagnostics
let errors (result: FNCSCheckResult) : Diagnostic list =
    CheckResult.errors result

/// Get warning diagnostics
let warnings (result: FNCSCheckResult) : Diagnostic list =
    CheckResult.warnings result

/// Format a type for display
let formatTypeStr (ty: FNCSType) : string =
    formatType ty

/// Check if a node is a platform binding
let isPlatformBinding (node: FNCSNode) : bool =
    match node.Kind with
    | SemanticKind.PlatformBinding _ -> true
    | _ -> false

/// Get platform binding name if present
let platformBindingName (node: FNCSNode) : string option =
    match node.Kind with
    | SemanticKind.PlatformBinding name -> Some name
    | _ -> None

/// Check if a node is a function application
let isApplication (node: FNCSNode) : bool =
    match node.Kind with
    | SemanticKind.Application _ -> true
    | _ -> false

/// Check if a node is a binding (let/do)
let isBinding (node: FNCSNode) : bool =
    match node.Kind with
    | SemanticKind.Binding _ -> true
    | _ -> false

/// Check if a node is a literal
let isLiteral (node: FNCSNode) : bool =
    match node.Kind with
    | SemanticKind.Literal _ -> true
    | _ -> false

/// Get literal value if present
let literalValue (node: FNCSNode) : LiteralValue option =
    match node.Kind with
    | SemanticKind.Literal v -> Some v
    | _ -> None

/// Check if a node is a lambda
let isLambda (node: FNCSNode) : bool =
    match node.Kind with
    | SemanticKind.Lambda _ -> true
    | _ -> false

/// Check if type is a function type
let isFunType (ty: FNCSType) : bool =
    isFunctionType ty

/// Check if type is a type variable
let isTypeVariable (ty: FNCSType) : bool =
    isTypeVar ty
