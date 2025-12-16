/// Build context and node creation for PSG construction
module Core.PSG.Construction.Types

open FSharp.Compiler.Text
open FSharp.Compiler.CodeAnalysis
open Core.PSG.Types
open Core.PSG.Correlation

/// Build context for PSG construction
/// Phase 1 (structural) uses None for CorrelationContext
/// Phase 2 (symbol correlation) provides the full context
type BuildContext = {
    CheckResults: FSharpCheckProjectResults option
    ParseResults: FSharpParseFileResults[]
    CorrelationContext: CorrelationContext option
    SourceFiles: Map<string, string>
}

/// Create a structural-only build context (Phase 1)
let createStructuralContext (parseResults: FSharpParseFileResults[]) (sourceFiles: Map<string, string>) : BuildContext =
    {
        CheckResults = None
        ParseResults = parseResults
        CorrelationContext = None
        SourceFiles = sourceFiles
    }

/// Create a full build context with symbol correlation (Phase 2)
let createFullContext
    (checkResults: FSharpCheckProjectResults)
    (parseResults: FSharpParseFileResults[])
    (correlationContext: CorrelationContext)
    (sourceFiles: Map<string, string>) : BuildContext =
    {
        CheckResults = Some checkResults
        ParseResults = parseResults
        CorrelationContext = Some correlationContext
        SourceFiles = sourceFiles
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
