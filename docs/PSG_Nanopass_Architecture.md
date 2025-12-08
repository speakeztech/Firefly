# PSG Nanopass Architecture

## Overview

This document describes the nanopass architecture for PSG (Program Semantic Graph) construction in Firefly. The nanopass approach, pioneered by Sarkar, Waddell, Dybvig, and Keep at Indiana University, advocates for compilers composed of many small, single-purpose passes rather than a few large, multi-task passes.

**Reference**: See `~/repos/nanopass-framework-scheme` for the canonical Scheme implementation.

## Motivation

The PSG must be the single source of truth for all downstream stages. Each nanopass enriches the PSG with additional information (edges, classifications) so that MLIR generation can simply traverse the enriched structure.

Key principles:
1. **PSG as single source of truth** - All information needed for MLIR generation should be in the PSG
2. **Composability** - XParsec transformations compose with nanopass outputs
3. **No coupling** - MLIR generation is decoupled from binding semantics

## Nanopass Principles Applied to PSG

From the nanopass framework:

> "The idea of writing a compiler as a series of small, single-purpose passes grew out of a course on compiler construction... Passes in a [nanopass] compiler are easy to understand, as each pass is responsible for just one transformation."

For PSG construction, this means:
- Each pass does ONE thing (e.g., "add def-use edges")
- Passes are composable and can be reordered or inserted
- The output of each pass can be inspected and validated
- Each pass transforms a well-defined input language to a well-defined output language

## Architectural Principle

Inspired by Appel's CPS work and the nanopass framework: make data flow explicit in the IR structure itself, not reconstructed during transformation. In SSA/MLIR, this is why values have explicit def-use chains. The PSG should have the same property.

## Nanopass Pipeline Design

### Pipeline

```
Phase 1: Structural Construction (syntax → nodes + ChildOf edges)
Phase 2: Type Integration (FCS types → node.Type field)
Phase 3: Def-Use Edge Construction (nanopass)
  - Nanopass 3a: Build symbol definition index
  - Nanopass 3b: Create SymbolUse edges from uses to definitions
Phase 4: Finalization (finalize children, update context)
→ Reachability Analysis (mark reachable nodes)
→ Alex/Zipper + Bindings (PSG → MLIR)
```

## Phase 3: Def-Use Edge Construction (Nanopass)

### Nanopass 3a: Build Symbol Definition Index

Input: PSG with nodes
Output: `Map<SymbolKey, NodeId>` mapping symbols to their defining nodes

Algorithm:
```
for each node in PSG.Nodes:
  if node.SyntaxKind starts with "Binding":
    if node.Symbol is Some symbol:
      key = symbolKey(symbol)  // Uses DeclarationLocation for stability
      defIndex[key] = node.Id
```

### Nanopass 3b: Create Def-Use Edges

Input: PSG with nodes, definition index
Output: PSG with SymbolUse edges added

Algorithm:
```
for each node in PSG.Nodes:
  if node.SyntaxKind starts with "Ident:" or "LongIdent:" or "Value:":
    if node.Symbol is Some symbol:
      key = symbolKey(symbol)
      if defIndex contains key:
        defNodeId = defIndex[key]
        add edge { Source = node.Id; Target = defNodeId; Kind = SymbolUse }
```

### Symbol Key Function

Use `DeclarationLocation` for local variables (stable across instances):

```fsharp
let symbolKey (sym: FSharpSymbol) : string =
    match sym with
    | :? FSharpMemberOrFunctionOrValue as mfv ->
        let loc = mfv.DeclarationLocation
        sprintf "%s@%s:%d:%d" mfv.DisplayName loc.FileName loc.StartLine loc.StartColumn
    | _ ->
        sprintf "%s_%d" sym.DisplayName (sym.GetHashCode())
```

## Impact on Downstream Components

### MLIR Generation Simplification

With def-use edges in PSG:

```fsharp
// At binding site - just generate MLIR, PSG already has the connection
generateBinding node  // Returns SSA value, recorded in node→SSA map

// At use site - follow edge to definition
let defNode = followSymbolUseEdge node
let ssaValue = nodeSSAMap[defNode.Id]
generate (Value(ssaValue, ...))
```

### Reachability Benefits

With explicit def-use edges, reachability analysis becomes more precise:
- A binding is reachable if it has a reachable use
- Uses naturally follow from their definitions
- No need to guess at scope relationships

## Implementation Status

### Completed

1. **`Core/PSG/Nanopass/DefUseEdges.fs`** - Nanopass implementation ✓
   - `buildDefinitionIndex`: Indexes Binding nodes and parameter Pattern:Named nodes
   - `createDefUseEdges`: Creates SymbolUse edges from Ident/Value/LongIdent to definitions
   - `findDefiningNode`: Query helper to follow SymbolUse edge from use to definition
   - `findUses`: Query helper to find all uses of a definition

2. **Other nanopasses implemented:**
   - `Core/PSG/Nanopass/ReducePipeOperators.fs` - Transforms pipe operators
   - `Core/PSG/Nanopass/FlattenApplications.fs` - Flattens curried applications

### Remaining Work

3. **MutableSet handling via def-use edges**
   - `isVariableUse` in DefUseEdges.fs should check MutableSet
   - MutableSet nodes need edges to their defining Binding:Mutable nodes

## Testing Strategy

1. Compile simple binding: `let x = 1 in x + 1`
   - Verify Binding node for `x` exists
   - Verify Ident node for `x` (in `x + 1`) has SymbolUse edge to Binding

2. Compile 03_HelloWorldHalfCurried
   - Verify `greeting` binding has SymbolUse edge from `greeting |> Console.writeln`
   - Verify no "Variable not found" errors in MLIR

3. **TimeLoop sample** (forcing function for mutable state)
   - Verify MutableSet:counter has edge to Binding:Mutable [counter]
   - Verify Zipper/Bindings follow edge for stack locations

## Future Nanopasses

The nanopass architecture enables future transformations as additional small passes:

- **Continuation edges** (control flow) - Make control flow explicit
- **Environment edges** (closure capture) - Track captured variables
- **Effect edges** (side effect ordering) - Order effectful operations
- **Generic instantiation** - Resolve generic types to concrete types
- **Tail call detection** - Mark tail-recursive calls

Each is a separate nanopass that enriches the PSG with explicit information, keeping downstream Zipper traversal and XParsec transformations composable.

## Nanopass Framework Principles

From Keep's dissertation (2013):

1. **Small passes** - Each pass does one transformation
2. **Formal language specs** - Input/output languages are formally defined
3. **Boilerplate generation** - Framework generates traversal code
4. **Validation** - Output of each pass can be validated against grammar

For Firefly/PSG:
- "Languages" are PSG schemas (what fields/edges exist)
- Nanopasses transform PSG → PSG (adding edges, enriching fields)
- XParsec provides composable pattern matching
- Each pass can be tested in isolation

## Intermediate Output

Each nanopass phase outputs a labeled PSG intermediate to enable analysis of individual transformations. When `--verbose` or `-k` (keep intermediates) is specified, the following files are generated in the intermediates directory:

- `psg_phase_1_structural.json` - After Phase 1 (structural construction)
- `psg_phase_2_type_integration.json` - After Phase 2 (FCS type integration)
- `psg_phase_3_def_use_edges.json` - After Phase 3 (def-use edge nanopass)
- `psg_phase_4_finalized.json` - After Phase 4 (finalization)

Additionally, diff files show what changed between phases:

- `psg_diff_1_structural_to_2_type_integration.json`
- `psg_diff_2_type_integration_to_3_def_use_edges.json`
- `psg_diff_3_def_use_edges_to_4_finalized.json`

This enables debugging and validation of each nanopass in isolation.

---

# Addendum: From PSG to PHG (Program Hypergraph)

## Vision: The Temporal Program Hypergraph

The nanopass architecture lays the foundation for evolving the Program Semantic Graph (PSG) into a full **Program Hypergraph (PHG)**. This evolution is not just nomenclature—it's the architectural insight that will enable Fidelity to produce efficient workflows for everything from LLVM-targeted CPUs to novel dataflow architectures.

**Reference**: See the [Hyping Hypergraphs](../../SpeakEZ/hugo/content/blog/Hyping%20Hypergraphs.md) blog entry for the full vision of temporal hypergraphs and learning systems.

## Why Hypergraphs?

Traditional graphs force us to decompose multi-way relationships into binary edges, losing semantic information. Consider how:

- Async code with delimited continuations creates rich, multi-way dependencies
- Pattern matching has multiple input→output relationships
- Closure capture involves multiple variables entering a single scope

Standard binary edges require auxiliary "join" and "split" nodes to represent these relationships. A hypergraph preserves them naturally:

```fsharp
// Traditional PSG (binary edges)
let closure_binding = NodeId "let_f"
let captured_x = PSGEdge { Source = closure_binding; Target = x_def; Kind = SymbolUse }
let captured_y = PSGEdge { Source = closure_binding; Target = y_def; Kind = SymbolUse }
// Relationship between x and y as co-captured variables is lost

// PHG (hyperedge)
let closure_capture = PHGHyperedge {
    Participants = Set.ofList [x_def; y_def; closure_binding]
    Kind = ClosureCapture
    Semantics = { SharedScope = true; CaptureKind = ByRef }
}
// Multi-way relationship preserved
```

## Nanopass as PHG Foundation

Each nanopass adds edges to the PSG. As the edge vocabulary grows, we're building toward a richer structure:

| Nanopass | Edge Kind Added | PHG Hyperedge Potential |
|----------|-----------------|-------------------------|
| Def-Use | `SymbolUse` | Variable flow hyperedge |
| Continuation | `ControlFlow` | Async join hyperedge |
| Environment | `Captures` | Closure context hyperedge |
| Effect | `EffectOrdering` | Effect chain hyperedge |

When multiple edges share the same semantic relationship, they can be promoted to a hyperedge:

```fsharp
// Nanopass edges
let edges = [
    { Source = use1; Target = def; Kind = SymbolUse }
    { Source = use2; Target = def; Kind = SymbolUse }
    { Source = use3; Target = def; Kind = SymbolUse }
]

// PHG hyperedge (all uses of 'def' as single multi-way relationship)
let hyperedge = {
    Participants = Set.ofList [def; use1; use2; use3]
    Kind = DataflowFanOut
    Direction = Definition def
}
```

## Targeting Multiple Architectures

The PHG vision enables unified compilation across traditional and novel architectures:

```
Program Hypergraph Core
        ↓
    Gradient-Based Analysis
        ↓
   ┌────┼────┬────────┐
   ↓    ↓    ↓        ↓
Harvard Von    Hybrid    Dataflow
   ↓    ↓    ↓        ↓
LLVM  CPU+GPU Spatial  Groq/Tenstorrent
```

The same PHG structure can be analyzed with different "gradients":

- **Control-flow emphasis** → LLVM IR, traditional CPU optimization
- **Dataflow emphasis** → Spatial kernels, streaming pipelines
- **Hybrid** → Heterogeneous CPU+GPU execution

## Temporal Dimension

The final evolution adds a temporal dimension—the PHG learns from each compilation:

```fsharp
type TemporalProgramHypergraph = {
    Current: ProgramHypergraph
    History: TemporalProjection list  // Previous compilations
    LearnedPatterns: CompilationKnowledge
}
```

Each nanopass intermediate we output today becomes training data for the learning PHG of tomorrow. This is why labeled intermediate output matters—we're building the dataset for future optimization learning.

## Roadmap

1. **Current**: Nanopass architecture with binary edges (implemented)
2. **Near-term**: Edge query helpers for Zipper traversal
3. **Mid-term**: Hyperedge promotion for multi-way relationships
4. **Long-term**: Temporal PHG with compilation learning

The nanopass infrastructure we're building today is the foundation for this evolution. Each small, single-purpose pass is a step toward the unified PHG representation that will enable Fidelity to target the full spectrum of computing architectures—from traditional CPUs to the novel dataflow processors that are reshaping the industry.
