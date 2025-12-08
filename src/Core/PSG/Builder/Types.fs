/// Build context and node creation for PSG construction
module Core.PSG.Construction.Types

open FSharp.Compiler.Text
open FSharp.Compiler.CodeAnalysis
open Core.PSG.Types
open Core.PSG.Correlation

/// Build context for PSG construction
type BuildContext = {
    CheckResults: FSharpCheckProjectResults
    ParseResults: FSharpParseFileResults[]
    CorrelationContext: CorrelationContext
    SourceFiles: Map<string, string>
}

/// Creates a new PSG node with the given properties
/// DEPRECATED: Use createNodeWithKind for type-driven dispatch
let createNode syntaxKind range fileName symbol parentId =
    let cleanKind = (syntaxKind : string).Replace(":", "_").Replace(" ", "_")
    let uniqueFileName = sprintf "%s_%s.fs" (System.IO.Path.GetFileNameWithoutExtension(fileName : string)) cleanKind
    let nodeId = NodeId.FromRange(uniqueFileName, range)

    ChildrenStateHelpers.createWithNotProcessed nodeId syntaxKind symbol range fileName parentId

/// Creates a new PSG node with typed syntax kind
/// Use this for type-driven dispatch instead of string parsing
let createNodeWithKind syntaxKind (kind: SyntaxKindT) range fileName symbol parentId =
    let cleanKind = (syntaxKind : string).Replace(":", "_").Replace(" ", "_")
    let uniqueFileName = sprintf "%s_%s.fs" (System.IO.Path.GetFileNameWithoutExtension(fileName : string)) cleanKind
    let nodeId = NodeId.FromRange(uniqueFileName, range)

    ChildrenStateHelpers.createWithKind nodeId syntaxKind kind symbol range fileName parentId

/// Add child to parent and return updated graph
let addChildToParent (childId: NodeId) (parentId: NodeId option) (graph: ProgramSemanticGraph) =
    match parentId with
    | None -> graph
    | Some pid ->
        match Map.tryFind pid.Value graph.Nodes with
        | Some parentNode ->
            let updatedParent = ChildrenStateHelpers.addChild childId parentNode

            let childOfEdge = {
                Source = pid
                Target = childId
                Kind = ChildOf
            }

            { graph with
                Nodes = Map.add pid.Value updatedParent graph.Nodes
                Edges = childOfEdge :: graph.Edges }
        | None -> graph
