/// SSA Assignment Pass - Alex preprocessing for MLIR emission
///
/// This pass assigns SSA names to PSG nodes BEFORE MLIR emission.
/// SSA is an MLIR/LLVM concern, not F# semantics, so it lives in Alex.
///
/// Key design:
/// - SSA counter resets at each Lambda boundary (per-function scoping)
/// - Post-order traversal ensures values are assigned before uses
/// - Returns Map<NodeId, string> that FNCSTransfer reads (no generation during emission)
module Alex.Preprocessing.SSAAssignment

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes

/// SSA assignment state for a single function scope
type private FunctionScope = {
    Counter: int
    Assignments: Map<int, string>  // NodeId.value -> SSA name
}

module private FunctionScope =
    let empty = { Counter = 0; Assignments = Map.empty }

    let yieldSSA (scope: FunctionScope) : string * FunctionScope =
        let name = sprintf "%%v%d" scope.Counter
        name, { scope with Counter = scope.Counter + 1 }

    let assign (nodeId: NodeId) (ssaName: string) (scope: FunctionScope) : FunctionScope =
        { scope with Assignments = Map.add (NodeId.value nodeId) ssaName scope.Assignments }

/// Check if a node kind produces an SSA value
let private producesValue (kind: SemanticKind) : bool =
    match kind with
    | SemanticKind.Literal _ -> true
    | SemanticKind.VarRef _ -> true
    | SemanticKind.Application _ -> true
    | SemanticKind.Lambda _ -> true
    | SemanticKind.Binding _ -> true
    | SemanticKind.Sequential _ -> true
    | SemanticKind.IfThenElse _ -> true
    | SemanticKind.Match _ -> true
    | SemanticKind.TupleExpr _ -> true
    | SemanticKind.RecordExpr _ -> true
    | SemanticKind.UnionCase _ -> true
    | SemanticKind.ArrayExpr _ -> true
    | SemanticKind.ListExpr _ -> true
    | SemanticKind.FieldGet _ -> true
    | SemanticKind.IndexGet _ -> true
    | SemanticKind.Upcast _ -> true
    | SemanticKind.Downcast _ -> true
    | SemanticKind.TypeTest _ -> true
    | SemanticKind.AddressOf _ -> true
    | SemanticKind.Deref _ -> true
    | SemanticKind.TraitCall _ -> true
    | SemanticKind.Intrinsic _ -> true
    | SemanticKind.PlatformBinding _ -> true
    | SemanticKind.InterpolatedString _ -> true
    // These don't produce values (statements/void)
    | SemanticKind.Set _ -> false
    | SemanticKind.FieldSet _ -> false
    | SemanticKind.IndexSet _ -> false
    | SemanticKind.NamedIndexedPropertySet _ -> false
    | SemanticKind.WhileLoop _ -> false
    | SemanticKind.ForLoop _ -> false
    | SemanticKind.ForEach _ -> false
    | SemanticKind.TryWith _ -> false
    | SemanticKind.TryFinally _ -> false
    | SemanticKind.Quote _ -> false
    | SemanticKind.ObjectExpr _ -> false
    | SemanticKind.ModuleDef _ -> false
    | SemanticKind.TypeDef _ -> false
    | SemanticKind.MemberDef _ -> false
    | SemanticKind.TypeAnnotation _ -> true  // Passes through the inner value
    | SemanticKind.Error _ -> false

/// Result of SSA assignment pass
type SSAAssignment = {
    /// Map from NodeId.value to SSA name
    NodeSSA: Map<int, string>
    /// Map from Lambda NodeId.value to its function name
    LambdaNames: Map<int, string>
    /// Set of entry point Lambda IDs
    EntryPointLambdas: Set<int>
}

/// Assign SSA names to all nodes in a function body
/// Returns updated scope with assignments
let rec private assignFunctionBody
    (graph: SemanticGraph)
    (scope: FunctionScope)
    (nodeId: NodeId)
    : FunctionScope =

    match Map.tryFind nodeId graph.Nodes with
    | None -> scope
    | Some node ->
        // Post-order: process children first
        let scopeAfterChildren =
            node.Children |> List.fold (fun s childId -> assignFunctionBody graph s childId) scope

        // Special handling for nested Lambdas - they get their own scope
        // (but we still assign this Lambda node an SSA in parent scope)
        match node.Kind with
        | SemanticKind.Lambda(_, bodyId) ->
            // Process Lambda body in a NEW scope (SSA counter resets)
            let _innerScope = assignFunctionBody graph FunctionScope.empty bodyId
            // Lambda itself gets an SSA in the PARENT scope (function pointer)
            if producesValue node.Kind then
                let ssaName, scopeWithSSA = FunctionScope.yieldSSA scopeAfterChildren
                FunctionScope.assign node.Id ssaName scopeWithSSA
            else
                scopeAfterChildren
        | _ ->
            // Regular node - assign SSA if it produces a value
            if producesValue node.Kind then
                let ssaName, scopeWithSSA = FunctionScope.yieldSSA scopeAfterChildren
                FunctionScope.assign node.Id ssaName scopeWithSSA
            else
                scopeAfterChildren

/// Collect all Lambdas in the graph and assign function names
let private collectLambdas (graph: SemanticGraph) : Map<int, string> * Set<int> =
    let mutable lambdaCounter = 0
    let mutable lambdaNames = Map.empty
    let mutable entryPoints = Set.empty

    // First, identify entry point lambdas
    for entryId in graph.EntryPoints do
        match Map.tryFind entryId graph.Nodes with
        | Some node ->
            match node.Kind with
            | SemanticKind.Binding(_, _, _, isEntryPoint) when isEntryPoint ->
                // The binding's first child is typically the Lambda
                match node.Children with
                | lambdaId :: _ -> entryPoints <- Set.add (NodeId.value lambdaId) entryPoints
                | _ -> ()
            | SemanticKind.Lambda _ ->
                entryPoints <- Set.add (NodeId.value entryId) entryPoints
            | _ -> ()
        | None -> ()

    // Now assign names to all Lambdas
    for kvp in graph.Nodes do
        let node = kvp.Value
        match node.Kind with
        | SemanticKind.Lambda _ ->
            let nodeIdVal = NodeId.value node.Id
            let name =
                if Set.contains nodeIdVal entryPoints then
                    "main"
                else
                    let n = sprintf "lambda_%d" lambdaCounter
                    lambdaCounter <- lambdaCounter + 1
                    n
            lambdaNames <- Map.add nodeIdVal name lambdaNames
        | _ -> ()

    lambdaNames, entryPoints

/// Main entry point: assign SSA names to all nodes in the graph
let assignSSA (graph: SemanticGraph) : SSAAssignment =
    let lambdaNames, entryPoints = collectLambdas graph

    // For each Lambda, assign SSAs to its body in its own scope
    let mutable allAssignments = Map.empty

    for kvp in graph.Nodes do
        let node = kvp.Value
        match node.Kind with
        | SemanticKind.Lambda(params', bodyId) ->
            // Start with parameter SSAs (%arg0, %arg1, etc.)
            let paramScope =
                params'
                |> List.mapi (fun i (name, _ty) -> i, name)
                |> List.fold (fun (scope: FunctionScope) (i, _name) ->
                    // Parameters use %argN naming, don't count toward SSA counter
                    scope
                ) FunctionScope.empty

            // Assign SSAs to body nodes
            let bodyScope = assignFunctionBody graph paramScope bodyId

            // Merge into global assignments
            for kvp in bodyScope.Assignments do
                allAssignments <- Map.add kvp.Key kvp.Value allAssignments
        | _ -> ()

    // Also process top-level nodes (module bindings, etc.)
    let topLevelScope =
        graph.EntryPoints
        |> List.fold (fun scope entryId -> assignFunctionBody graph scope entryId) FunctionScope.empty

    for kvp in topLevelScope.Assignments do
        allAssignments <- Map.add kvp.Key kvp.Value allAssignments

    {
        NodeSSA = allAssignments
        LambdaNames = lambdaNames
        EntryPointLambdas = entryPoints
    }

/// Look up the SSA name for a node
let lookupSSA (nodeId: NodeId) (assignment: SSAAssignment) : string option =
    Map.tryFind (NodeId.value nodeId) assignment.NodeSSA

/// Look up the function name for a Lambda
let lookupLambdaName (nodeId: NodeId) (assignment: SSAAssignment) : string option =
    Map.tryFind (NodeId.value nodeId) assignment.LambdaNames

/// Check if a Lambda is an entry point
let isEntryPoint (nodeId: NodeId) (assignment: SSAAssignment) : bool =
    Set.contains (NodeId.value nodeId) assignment.EntryPointLambdas
