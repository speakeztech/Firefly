/// DefUseEdges - Nanopass to add explicit def-use edges to the PSG
///
/// This nanopass transforms a PSG with nodes and ChildOf edges into a PSG
/// with additional SymbolUse edges connecting variable uses to their definitions.
///
/// Nanopass 3a: Build symbol definition index
/// Nanopass 3b: Create SymbolUse edges from uses to definitions
///
/// Reference: Nanopass Framework (Sarkar, Waddell, Dybvig, Keep)
/// See: ~/repos/nanopass-framework-scheme
module Core.PSG.Nanopass.DefUseEdges

open FSharp.Compiler.Symbols
open Core.PSG.Types

// ═══════════════════════════════════════════════════════════════════════════
// Symbol Key Generation
// ═══════════════════════════════════════════════════════════════════════════

/// Generate a stable key for a symbol using its declaration location.
/// For local variables, DeclarationLocation is stable across symbol instances.
/// This is the same key function used in the emitter's SymbolSSAContext,
/// ensuring consistency between PSG edges and any remaining emission logic.
let symbolKey (sym: FSharpSymbol) : string =
    match sym with
    | :? FSharpMemberOrFunctionOrValue as mfv ->
        // Use declaration location for stable identity
        // Some symbols (e.g., BCL types) may not have DeclarationLocation available
        try
            let loc = mfv.DeclarationLocation
            sprintf "%s@%s:%d:%d" mfv.DisplayName loc.FileName loc.StartLine loc.StartColumn
        with _ ->
            // Fallback: use display name + hash for symbols without declaration location
            sprintf "%s_%d" mfv.DisplayName (mfv.GetHashCode())
    | :? FSharpEntity as entity ->
        // For types, use full name
        sprintf "type:%s" entity.FullName
    | :? FSharpUnionCase as uc ->
        // For union cases, use full name
        sprintf "case:%s" uc.FullName
    | :? FSharpField as field ->
        // For fields, use declaring type + field name
        match field.DeclaringEntity with
        | Some entity -> sprintf "field:%s.%s" entity.FullName field.Name
        | None -> sprintf "field:unknown.%s" field.Name
    | _ ->
        // Fallback: display name + hash
        sprintf "%s_%d" sym.DisplayName (sym.GetHashCode())

// ═══════════════════════════════════════════════════════════════════════════
// Nanopass 3a: Build Symbol Definition Index
// ═══════════════════════════════════════════════════════════════════════════

/// Index mapping symbol keys to their defining node IDs.
/// Built by scanning all Binding nodes in the PSG.
type SymbolDefinitionIndex = Map<string, NodeId>

/// Check if a Pattern:Named node is a function parameter (not inside a Binding).
/// Function parameters have Pattern:LongIdent as parent, not Binding.
let private isParameterPattern (psg: ProgramSemanticGraph) (node: PSGNode) : bool =
    match node.ParentId with
    | Some parentId ->
        match Map.tryFind parentId.Value psg.Nodes with
        | Some parentNode -> SyntaxKindT.isLongIdentPattern parentNode.Kind
        | None -> false
    | None -> false

/// Check if a node represents a symbol definition.
/// Definitions include:
/// - Binding nodes (let bindings - SSA recorded here by emitBindingNode)
/// - Pattern:Named nodes ONLY if they are function parameters (SSA recorded by FunctionEmitter)
///   Pattern:Named inside Bindings are NOT indexed - we use the Binding node instead
let private isDefinitionNode (psg: ProgramSemanticGraph) (node: PSGNode) : bool =
    match node.Kind with
    | SKBinding _ -> true
    | SKPattern PNamed ->
        // Only index Pattern:Named that are function parameters
        isParameterPattern psg node
    | _ -> false

/// Build an index of symbol definitions from Binding and parameter Pattern:Named nodes.
/// This is Nanopass 3a - a pure function from PSG to index.
let buildDefinitionIndex (psg: ProgramSemanticGraph) : SymbolDefinitionIndex =
    psg.Nodes
    |> Map.toSeq
    |> Seq.choose (fun (_, node) ->
        // Index Binding nodes and parameter Pattern:Named nodes
        if isDefinitionNode psg node then
            match node.Symbol with
            | Some symbol ->
                let key = symbolKey symbol
                Some (key, node.Id)
            | None -> None
        else
            None)
    |> Map.ofSeq

// ═══════════════════════════════════════════════════════════════════════════
// Nanopass 3b: Create Def-Use Edges
// ═══════════════════════════════════════════════════════════════════════════

/// Check if a node is a variable use (Ident, LongIdent, MutableSet)
/// MutableSet is included because it references a mutable variable for assignment
let private isVariableUse (node: PSGNode) : bool =
    match node.Kind with
    | SKExpr EIdent | SKExpr ELongIdent | SKExpr EMutableSet -> true
    | _ -> false

/// Create def-use edges for all variable uses in the PSG.
/// This is Nanopass 3b - transforms PSG by adding SymbolUse edges.
let createDefUseEdges (defIndex: SymbolDefinitionIndex) (psg: ProgramSemanticGraph) : ProgramSemanticGraph =
    let newEdges =
        psg.Nodes
        |> Map.toSeq
        |> Seq.choose (fun (_, node) ->
            // Only process variable use nodes
            if isVariableUse node then
                match node.Symbol with
                | Some symbol ->
                    let key = symbolKey symbol
                    match Map.tryFind key defIndex with
                    | Some defNodeId ->
                        // Create edge from use to definition
                        Some {
                            Source = node.Id
                            Target = defNodeId
                            Kind = SymbolUse
                        }
                    | None ->
                        // No definition found - this is likely a module-level or external symbol
                        // Not an error, just no local def-use edge
                        None
                | None ->
                    // No symbol on node - can't create edge
                    None
            else
                None)
        |> Seq.toList

    // Add new edges to existing edges
    { psg with Edges = newEdges @ psg.Edges }

// ═══════════════════════════════════════════════════════════════════════════
// Combined Nanopass
// ═══════════════════════════════════════════════════════════════════════════

/// Run both nanopasses: build definition index, then create def-use edges.
/// This is the main entry point for the def-use edge nanopass.
let addDefUseEdges (psg: ProgramSemanticGraph) : ProgramSemanticGraph =
    let defIndex = buildDefinitionIndex psg
    createDefUseEdges defIndex psg

// ═══════════════════════════════════════════════════════════════════════════
// Edge Query Helpers (for downstream use)
// ═══════════════════════════════════════════════════════════════════════════

/// Find the defining node for a variable use node by following SymbolUse edge.
/// Returns None if no def-use edge exists (e.g., external symbol).
let findDefiningNode (psg: ProgramSemanticGraph) (useNode: PSGNode) : PSGNode option =
    psg.Edges
    |> List.tryFind (fun edge ->
        edge.Source = useNode.Id && edge.Kind = SymbolUse)
    |> Option.bind (fun edge ->
        Map.tryFind edge.Target.Value psg.Nodes)

/// Find all uses of a definition node by following SymbolUse edges backwards.
let findUses (psg: ProgramSemanticGraph) (defNode: PSGNode) : PSGNode list =
    psg.Edges
    |> List.filter (fun edge ->
        edge.Target = defNode.Id && edge.Kind = SymbolUse)
    |> List.choose (fun edge ->
        Map.tryFind edge.Source.Value psg.Nodes)
