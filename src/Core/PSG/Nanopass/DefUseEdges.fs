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
        let loc = mfv.DeclarationLocation
        sprintf "%s@%s:%d:%d" mfv.DisplayName loc.FileName loc.StartLine loc.StartColumn
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

/// Build an index of symbol definitions from Binding nodes.
/// This is Nanopass 3a - a pure function from PSG to index.
let buildDefinitionIndex (psg: ProgramSemanticGraph) : SymbolDefinitionIndex =
    psg.Nodes
    |> Map.toSeq
    |> Seq.choose (fun (_, node) ->
        // Only index Binding nodes (where symbols are defined)
        if node.SyntaxKind.StartsWith("Binding") then
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

/// Check if a node is a variable use (Ident, Value, LongIdent)
let private isVariableUse (node: PSGNode) : bool =
    node.SyntaxKind.StartsWith("Ident:") ||
    node.SyntaxKind.StartsWith("Value:") ||
    node.SyntaxKind.StartsWith("LongIdent:")

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

    printfn "[NANOPASS] DefUse: Built definition index with %d entries" (Map.count defIndex)

    let enrichedPsg = createDefUseEdges defIndex psg

    let defUseEdgeCount =
        enrichedPsg.Edges
        |> List.filter (fun e -> e.Kind = SymbolUse)
        |> List.length

    printfn "[NANOPASS] DefUse: Created %d def-use edges" defUseEdgeCount

    enrichedPsg

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
