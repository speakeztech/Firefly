# PSG Implementation Gap Analysis and Remediation Plan

## Executive Summary

The current Program Semantic Graph (PSG) implementation in Firefly fails to create a proper graph structure necessary for reachability analysis, dead code elimination, and MLIR generation. This document provides a comprehensive analysis of the gaps and a concrete multi-pass remediation strategy that aligns with the multi-pass compilation philosophy of MLIR and LLVM.

## Current State Analysis

### Critical Gap 1: No Graph Structure

The current implementation creates isolated nodes without parent-child relationships or proper edges.

**Location**: `src/Core/PSG/Builder.fs`

```fsharp
// Current broken implementation
let createNode (syntaxKind: string) (range: range) (fileName: string) 
               (symbol: FSharpSymbol option) (parentId: NodeId option) =
    {
        Id = nodeId
        SyntaxKind = syntaxKind
        Symbol = symbol
        Range = range
        SourceFile = fileName
        ParentId = parentId
        Children = []  // ❌ NEVER POPULATED!
    }
```

**Impact**: Cannot traverse the graph for reachability analysis or dead code elimination.

### Critical Gap 2: Failed Symbol Correlation

Symbol correlation achieves 0% success rate due to exact range matching requirements.

**Location**: `src/Core/PSG/Correlation.fs`

```fsharp
// Current implementation requires exact position match
match Map.tryFind range.Start context.PositionIndex with
| Some symbolUse -> Some symbolUse.Symbol
| None -> None  // ❌ Falls back to unreliable search
```

**Evidence from Phase 1 Testing**:
- 0 correlated nodes out of 4726 symbols
- Empty correlation JSON output
- All files show 0% coverage

### Critical Gap 3: Self-Referencing Edges

Edge creation logic creates circular references instead of proper symbol resolution.

**Location**: `src/Core/PSG/Builder.fs` (processExpression function)

```fsharp
| SynExpr.Ident ident ->
    let identName = ident.idText
    match Map.tryFind identName graph'.SymbolTable with
    | Some targetSymbol ->
        let edge = {
            Source = exprNode.Id
            Target = NodeId.FromSymbol(targetSymbol)  // ❌ Wrong symbol!
            Kind = SymRef
        }
```

**Impact**: All edges point to themselves (e.g., `sym_message_652804a8` → `sym_message_652804a8`)

### Critical Gap 4: Missing Typed AST Integration

The implementation ignores FCS typed AST, losing all type information.

**Location**: `src/Core/PSG/Builder.fs` (buildProgramSemanticGraph function)

```fsharp
// Current: Only uses parse tree
let graphs = 
    parseResults
    |> Array.choose (fun pr ->
        match pr.ParseTree with
        | ParsedInput.ImplFile implFile ->
            Some (buildFromImplementationFile implFile buildContext)
```

**Missing**: `checkResults.AssemblyContents.ImplementationFiles` is never processed

### Critical Gap 5: Incomplete Edge Types

Only creates `SymRef` edges, missing critical relationship types:
- Call edges (function invocations)
- Type instantiation edges
- Control flow edges
- Data dependency edges

### Critical Gap 6: No Memory Layout Information

PSG nodes lack memory and type preservation information needed for MLIR generation.

**Missing from `PSGNode` type**:
- `Type: FSharpType option`
- `MemoryLayout: MemoryInfo option`
- `MLIRHints: MLIRMetadata option`

## Multi-Pass Remediation Strategy

### Philosophy: Aligning with MLIR/LLVM Multi-Pass Architecture

Just as MLIR employs multiple transformation passes (canonicalization, CSE, inlining) and LLVM uses multiple optimization passes (mem2reg, SROA, DCE), Firefly's PSG construction must embrace a multi-pass approach:

1. **Pass 1**: Structural Construction - Build syntax tree with proper parent/child links
2. **Pass 2**: Type Integration - Merge typed AST information 
3. **Pass 3**: Symbol Resolution - Correlate symbols and resolve references
4. **Pass 4**: Edge Construction - Build all edge types
5. **Pass 5**: Memory Analysis - Compute layouts and add MLIR hints
6. **Pass 6**: Reachability Analysis - Prune unreachable code

### Implementation Plan

#### Phase 1: Fix Core Data Structures

**Location**: `src/Core/PSG/Types.fs`

```fsharp
// Enhanced PSG node with bidirectional links and type info
type PSGNode = {
    Id: NodeId
    SyntaxKind: string
    Symbol: FSharpSymbol option
    Type: FSharpType option  // NEW: Type information
    MemoryLayout: MemoryInfo option  // NEW: Memory requirements
    Range: range
    SourceFile: string
    ParentId: NodeId option
    ChildIds: NodeId list  // NEW: Maintained child list
    IncomingEdges: Set<EdgeId>  // NEW: For efficient traversal
    OutgoingEdges: Set<EdgeId>  // NEW: For reachability
}

// Comprehensive edge types
type EdgeKind =
    | SymbolDef     // Symbol definition site
    | SymbolUse     // Symbol usage
    | FunctionCall  // Direct function invocation
    | TypeInstantiation of typeArgs: FSharpType list
    | ControlFlow of kind: ControlFlowKind
    | DataDependency
    | ModuleContainment

// Memory layout information
type MemoryInfo = {
    Size: int
    Alignment: int
    StorageClass: StorageClass
    MLIRMemRefType: string option
}
```

#### Phase 2: Multi-Pass Builder Architecture

**New Module**: `src/Core/PSG/MultiPassBuilder.fs`

```fsharp
module Core.PSG.MultiPassBuilder

// Pass 1: Structural Construction
let pass1_buildStructure (parseResults: FSharpParseFileResults[]) : StructuralPSG =
    // Build syntax tree with proper parent/child relationships
    let mutable nodeMap = Map.empty
    let mutable rootNodes = []
    
    let rec buildNode (parent: NodeId option) (syntax: obj) =
        let node = createNodeWithChildren syntax parent
        nodeMap <- Map.add node.Id node nodeMap
        
        // Recursively process children and update parent
        let childIds = 
            getChildSyntax syntax 
            |> List.map (buildNode (Some node.Id))
        
        // Update node with children
        let updatedNode = { node with ChildIds = childIds }
        nodeMap <- Map.add node.Id updatedNode nodeMap
        node.Id
    
    // Process each file
    parseResults |> Array.iter (fun pr ->
        match pr.ParseTree with
        | ParsedInput.ImplFile impl ->
            let rootId = buildNode None impl
            rootNodes <- rootId :: rootNodes
        | _ -> ()
    )
    
    { Nodes = nodeMap; RootNodes = rootNodes }

// Pass 2: Type Integration
let pass2_integrateTypes (structuralPSG: StructuralPSG) 
                        (typedFiles: FSharpImplementationFileContents list) : TypedPSG =
    // Create type index from typed AST
    let typeIndex = buildTypeIndex typedFiles
    
    // Update nodes with type information
    let typedNodes = 
        structuralPSG.Nodes 
        |> Map.map (fun nodeId node ->
            match findTypeForNode node typeIndex with
            | Some fsharpType ->
                { node with Type = Some fsharpType }
            | None -> node
        )
    
    { structuralPSG with Nodes = typedNodes }

// Pass 3: Symbol Resolution
let pass3_resolveSymbols (typedPSG: TypedPSG) 
                        (symbolUses: FSharpSymbolUse[]) : SymbolResolvedPSG =
    // Build enhanced correlation context with tolerance
    let correlationContext = buildEnhancedCorrelation symbolUses
    
    // Two sub-passes for symbol resolution
    let pass3a_correlateDefinitions psg =
        // First correlate all definition sites
        symbolUses 
        |> Array.filter (fun su -> su.IsFromDefinition)
        |> Array.fold (correlateDefinition psg) psg
    
    let pass3b_resolveReferences psg =
        // Then resolve all usage sites
        symbolUses
        |> Array.filter (fun su -> su.IsFromUse)
        |> Array.fold (resolveReference psg) psg
    
    typedPSG |> pass3a_correlateDefinitions |> pass3b_resolveReferences

// Pass 4: Edge Construction
let pass4_buildEdges (symbolResolvedPSG: SymbolResolvedPSG) : EdgeEnhancedPSG =
    let mutable edges = Map.empty
    let mutable edgeId = 0L
    
    let createEdge source target kind =
        let edge = {
            Id = EdgeId edgeId
            Source = source
            Target = target
            Kind = kind
        }
        edges <- Map.add edge.Id edge edges
        edgeId <- edgeId + 1L
        edge.Id
    
    // Build different edge types in sub-passes
    let pass4a_symbolEdges psg =
        // Create symbol definition and usage edges
        psg.Nodes |> Map.iter (fun nodeId node ->
            match node.Symbol with
            | Some symbol when isDefinition node ->
                // Find all uses of this symbol
                findSymbolUses symbol psg |> List.iter (fun useNode ->
                    createEdge useNode.Id nodeId SymbolUse |> ignore
                )
            | _ -> ()
        )
    
    let pass4b_callEdges psg =
        // Create function call edges
        psg.Nodes |> Map.iter (fun nodeId node ->
            match node.SyntaxKind with
            | "Application" ->
                match resolveCalledFunction node psg with
                | Some targetFunc ->
                    createEdge nodeId targetFunc.Id FunctionCall |> ignore
                | None -> ()
            | _ -> ()
        )
    
    let pass4c_controlFlowEdges psg =
        // Create control flow edges for match expressions, if-then-else, etc.
        psg.Nodes |> Map.iter (fun nodeId node ->
            match node.SyntaxKind with
            | "Match" -> buildMatchControlFlow node psg createEdge
            | "IfThenElse" -> buildIfControlFlow node psg createEdge
            | _ -> ()
        )
    
    // Execute edge construction sub-passes
    pass4a_symbolEdges symbolResolvedPSG
    pass4b_callEdges symbolResolvedPSG
    pass4c_controlFlowEdges symbolResolvedPSG
    
    { symbolResolvedPSG with Edges = edges }

// Pass 5: Memory Analysis
let pass5_analyzeMemory (edgeEnhancedPSG: EdgeEnhancedPSG) : MemoryAnnotatedPSG =
    // Analyze each node for memory requirements
    let analyzeNode (node: PSGNode) : PSGNode =
        match node.Type with
        | Some fsharpType ->
            let layout = computeMemoryLayout fsharpType
            let mlirHint = generateMLIRMemRefType fsharpType layout
            { node with 
                MemoryLayout = Some layout
                MLIRHints = Some { MemRefType = mlirHint } }
        | None -> node
    
    let memoryAnnotatedNodes =
        edgeEnhancedPSG.Nodes
        |> Map.map (fun _ node -> analyzeNode node)
    
    { edgeEnhancedPSG with Nodes = memoryAnnotatedNodes }

// Pass 6: Reachability Analysis and Pruning
let pass6_reachabilityAnalysis (memoryAnnotatedPSG: MemoryAnnotatedPSG) : FinalPSG =
    // Find entry points
    let entryPoints = findEntryPoints memoryAnnotatedPSG
    
    // Mark reachable nodes using graph traversal
    let reachableNodes = 
        let mutable visited = Set.empty
        let mutable workList = entryPoints |> List.ofSeq
        
        while not (List.isEmpty workList) do
            match workList with
            | nodeId :: rest ->
                workList <- rest
                if not (Set.contains nodeId visited) then
                    visited <- Set.add nodeId visited
                    
                    // Add all nodes reachable via outgoing edges
                    let node = memoryAnnotatedPSG.Nodes.[nodeId]
                    let outgoingTargets =
                        node.OutgoingEdges
                        |> Set.toList
                        |> List.map (fun edgeId -> memoryAnnotatedPSG.Edges.[edgeId].Target)
                    
                    workList <- outgoingTargets @ workList
            | [] -> ()
        
        visited
    
    // Create pruned PSG with only reachable nodes
    let prunedNodes = 
        memoryAnnotatedPSG.Nodes 
        |> Map.filter (fun nodeId _ -> Set.contains nodeId reachableNodes)
    
    let prunedEdges =
        memoryAnnotatedPSG.Edges
        |> Map.filter (fun _ edge ->
            Set.contains edge.Source reachableNodes &&
            Set.contains edge.Target reachableNodes)
    
    {
        InitialGraph = memoryAnnotatedPSG
        PrunedGraph = { memoryAnnotatedPSG with Nodes = prunedNodes; Edges = prunedEdges }
        EliminatedNodeCount = memoryAnnotatedPSG.Nodes.Count - prunedNodes.Count
        ReachabilityReport = generateReachabilityReport reachableNodes memoryAnnotatedPSG
    }
```

#### Phase 3: Enhanced Correlation Strategy

**Location**: `src/Core/PSG/EnhancedCorrelation.fs`

```fsharp
module Core.PSG.EnhancedCorrelation

// Tolerance-based range matching
let private rangeContains (outerRange: range) (innerPos: pos) =
    outerRange.Start.Line <= innerPos.Line && 
    innerPos.Line <= outerRange.End.Line &&
    (innerPos.Line > outerRange.Start.Line || innerPos.Column >= outerRange.Start.Column) &&
    (innerPos.Line < outerRange.End.Line || innerPos.Column <= outerRange.End.Column)

// Build spatial index for efficient correlation
type SpatialIndex = {
    FileIndex: Map<string, SymbolUseTree>
}

and SymbolUseTree = 
    | Leaf of FSharpSymbolUse list
    | Branch of range * left: SymbolUseTree * right: SymbolUseTree

let buildSpatialIndex (symbolUses: FSharpSymbolUse[]) : SpatialIndex =
    let byFile = symbolUses |> Array.groupBy (fun su -> su.Range.FileName)
    
    let buildTree (uses: FSharpSymbolUse[]) =
        // Build R-tree or interval tree for efficient range queries
        match uses.Length with
        | 0 -> Leaf []
        | 1 -> Leaf [uses.[0]]
        | n ->
            let sorted = uses |> Array.sortBy (fun su -> su.Range.Start.Line, su.Range.Start.Column)
            let mid = n / 2
            let left = buildTree sorted.[0..mid-1]
            let right = buildTree sorted.[mid..]
            let bounds = computeBounds sorted
            Branch(bounds, left, right)
    
    { FileIndex = byFile |> Array.map (fun (file, uses) -> file, buildTree uses) |> Map.ofArray }

// Enhanced correlation with multiple strategies
let correlateWithTolerance (node: PSGNode) (spatialIndex: SpatialIndex) : FSharpSymbol option =
    // Strategy 1: Exact match
    match exactMatch node.Range spatialIndex with
    | Some symbol -> Some symbol
    | None ->
        // Strategy 2: Tolerance-based containment
        match containmentMatch node.Range spatialIndex with
        | Some symbol -> Some symbol
        | None ->
            // Strategy 3: Name-based fallback with scope awareness
            match node.SyntaxKind with
            | "Binding" -> nameBasedMatch node spatialIndex
            | _ -> None
```

#### Phase 4: Integration Points

**Modified**: `src/Core/IngestionPipeline.fs`

```fsharp
// Replace single-pass PSG construction with multi-pass
let buildProgramSemanticGraph (checkResults: FSharpCheckProjectResults) 
                             (parseResults: FSharpParseFileResults[]) =
    
    // Execute all passes in sequence
    let structuralPSG = MultiPassBuilder.pass1_buildStructure parseResults
    
    let typedPSG = 
        MultiPassBuilder.pass2_integrateTypes 
            structuralPSG 
            checkResults.AssemblyContents.ImplementationFiles
    
    let symbolResolvedPSG = 
        MultiPassBuilder.pass3_resolveSymbols 
            typedPSG 
            (checkResults.GetAllUsesOfAllSymbols())
    
    let edgeEnhancedPSG = MultiPassBuilder.pass4_buildEdges symbolResolvedPSG
    
    let memoryAnnotatedPSG = MultiPassBuilder.pass5_analyzeMemory edgeEnhancedPSG
    
    let finalPSG = MultiPassBuilder.pass6_reachabilityAnalysis memoryAnnotatedPSG
    
    // Return both initial and pruned graphs
    finalPSG
```

## Validation Strategy

### Unit Tests for Each Pass

**Location**: `tests/Core.PSG.Tests/MultiPassTests.fs`

```fsharp
[<Test>]
let ``Pass 1 creates proper parent-child relationships`` () =
    let parseResult = parseTestFile "SimpleModule.fs"
    let structuralPSG = pass1_buildStructure [|parseResult|]
    
    // Verify every non-root node has a parent
    structuralPSG.Nodes |> Map.forall (fun nodeId node ->
        List.contains nodeId structuralPSG.RootNodes || 
        Option.isSome node.ParentId
    ) |> should equal true
    
    // Verify parent-child consistency
    structuralPSG.Nodes |> Map.forall (fun nodeId node ->
        node.ChildIds |> List.forall (fun childId ->
            let child = structuralPSG.Nodes.[childId]
            child.ParentId = Some nodeId
        )
    ) |> should equal true

[<Test>]
let ``Pass 3 achieves >90% symbol correlation`` () =
    let psg = buildTestPSG "HelloWorld.fs"
    let correlatedNodes = 
        psg.Nodes 
        |> Map.filter (fun _ node -> Option.isSome node.Symbol)
        |> Map.count
    
    let correlationRate = float correlatedNodes / float psg.Nodes.Count
    correlationRate |> should greaterThan 0.9

[<Test>]
let ``Pass 6 correctly identifies unreachable code`` () =
    let psg = buildTestPSG "DeadCodeExample.fs"
    let finalPSG = pass6_reachabilityAnalysis psg
    
    // Verify unused function is eliminated
    finalPSG.PrunedGraph.Nodes 
    |> Map.exists (fun _ node -> 
        node.Symbol |> Option.map (fun s -> s.DisplayName) = Some "unusedFunction"
    ) |> should equal false
```

### Integration Tests

```fsharp
[<Test>]
let ``HelloWorldDirect produces correct PSG structure`` () =
    let psg = buildPSGForProject "01_HelloWorldDirect.fsproj"
    
    // Verify key elements exist
    psg.PrunedGraph.Nodes |> Map.exists (fun _ n -> 
        n.Symbol |> Option.exists (fun s -> s.DisplayName = "hello")
    ) |> should equal true
    
    // Verify stackBuffer call has proper edges
    let stackBufferCalls = 
        psg.PrunedGraph.Edges 
        |> Map.filter (fun _ edge ->
            match edge.Kind with
            | FunctionCall -> 
                let targetNode = psg.PrunedGraph.Nodes.[edge.Target]
                targetNode.Symbol 
                |> Option.exists (fun s -> s.DisplayName.Contains "stackBuffer")
            | _ -> false
        )
    
    Map.count stackBufferCalls |> should equal 1
```

## Success Metrics

1. **Symbol Correlation Rate**: >90% of nodes with expected symbols should have correlation
2. **Graph Connectivity**: Every non-root node must be reachable from a root
3. **Edge Completeness**: All function calls, symbol uses, and control flow must have edges
4. **Type Preservation**: >95% of typed nodes should maintain type information
5. **Reachability Accuracy**: Dead code elimination should match manual analysis

## Deliverables

1. **Enhanced PSG Types** (`src/Core/PSG/Types.fs`)
   - Bidirectional graph structure
   - Complete edge taxonomy
   - Memory layout information

2. **Multi-Pass Builder** (`src/Core/PSG/MultiPassBuilder.fs`)
   - Six-pass construction pipeline
   - Sub-passes for complex operations
   - Progress reporting

3. **Enhanced Correlation** (`src/Core/PSG/EnhancedCorrelation.fs`)
   - Spatial indexing
   - Tolerance-based matching
   - Multiple fallback strategies

4. **Comprehensive Tests** (`tests/Core.PSG.Tests/`)
   - Unit tests per pass
   - Integration tests
   - Performance benchmarks

5. **Updated Pipeline** (`src/Core/IngestionPipeline.fs`)
   - Multi-pass PSG construction
   - Intermediate output options
   - Diagnostic reporting

## Conclusion

This multi-pass approach aligns Firefly's front-end with the compilation philosophy of MLIR and LLVM, where complex transformations are achieved through a series of well-defined passes. Each pass has a clear responsibility and builds upon the previous pass's output, creating a robust pipeline that can handle the complexities of F# while producing a high-quality Program Semantic Graph suitable for MLIR generation.