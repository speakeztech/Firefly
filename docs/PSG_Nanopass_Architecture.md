# PSG Nanopass Architecture v2: True Nanopass Pipeline

## Executive Summary

This document defines a fundamental architectural shift in Firefly's PSG (Program Semantic Graph) construction. Drawing directly from the nanopass framework principles (Sarkar, Waddell, Dybvig, Keep - Indiana University), we recognize that **PSG construction itself must be a true nanopass pipeline**, not just downstream enrichment.

**The Critical Insight**: The current architecture has a blind spot at the foundation. We were applying nanopass principles to enrichment (def-use edges, operation classification) while the initial PSG construction remained a monolithic operation. This document corrects that oversight.

## Reference

- **Nanopass Framework**: `~/repos/nanopass-framework-scheme`
- **User Guide**: `~/repos/nanopass-framework-scheme/doc/user-guide.pdf`
- **Key Papers**:
  - Sarkar et al. "A Nanopass Infrastructure for Compiler Education" (ICFP 2004)
  - Keep "A Nanopass Framework for Commercial Compiler Development" (ICFP 2013)
- **Baker Architecture**: `docs/Baker_Architecture.md` - Details Phase 4 implementation

## Core Nanopass Principles

From the nanopass framework user guide:

> "The idea of writing a compiler as a series of small, single-purpose passes grew out of a course on compiler construction... Passes in a [nanopass] compiler are easy to understand, as each pass is responsible for just one transformation. The compiler is easier to debug when compared with a traditional compiler composed of a few, multi-task passes."

Key properties:
1. **Each pass does ONE transformation**
2. **Input/output languages are formally defined** (grammar-checked)
3. **Boilerplate is auto-generated** (catamorphisms, traversal code)
4. **Passes are composable** - can be reordered, inserted, removed
5. **Each intermediate is inspectable** - validates correctness at each step

## The Architectural Blind Spot

### What We Had

```
FCS (SynExpr + symbols) → [Monolithic PSG Builder] → PSG
                                    ↓
                              Type Integration (also monolithic)
                                    ↓
                              Nanopasses (FlattenApps, ReducePipe, DefUse, etc.)
                                    ↓
                              Reachability
                                    ↓
                              Alex/Zipper → MLIR
```

The "Monolithic PSG Builder" was doing too much:
- Walking syntax tree (SynExpr)
- Correlating FCS symbols
- Attempting to capture typed tree information
- All in one pass

### What We Need

```
FCS Parse → PSG₀ (Pure Syntax)
              ↓ Pass 1: Structural Construction (FULL LIBRARY)
            PSG₁ (Nodes + ChildOf edges)
              ↓ Pass 2: Symbol Correlation (FULL LIBRARY)
            PSG₂ (+ FSharpSymbol attachments)
              ↓ Pass 3: Reachability (SOFT DELETE - NARROWS SCOPE)
            PSG₃ (+ IsReachable marks, structure intact)
              │
              │ *** GRAPH NOW NARROWED TO APPLICATION SCOPE ***
              │
              ↓ Pass 4: Typed Tree Overlay [BAKER] (NARROWED GRAPH ONLY)
            PSG₄ (+ Type, Constraints, SRTP resolution, member bodies)
              ↓ Pass 5+: Enrichment Nanopasses (NARROWED GRAPH ONLY)
            PSG_n (+ def-use edges, classifications, etc.)
              ↓
            Alex/Zipper → MLIR (NARROWED, ENRICHED GRAPH)
```

**Critical Transition at Phase 3**: Reachability analysis narrows the compute graph to application scope. Phases 4+ (Baker, enrichment nanopasses, Alex) all operate on this narrowed graph. This ensures:
- Performance: Expensive type correlation only on reachable code
- Zipper coherence: Baker and Alex see the same narrowed scope
- Semantic correctness: Only code that will be compiled gets enriched

### Pass Ordering and Dependencies

The nanopass framework allows for **dependency ordering** between passes. This isn't arbitrary coupling - it reflects semantic dependencies in the compilation process:

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5+
                      ↓
              SCOPE NARROWING
                      ↓
            Baker depends on this
```

**Phase 3 (Reachability) is a scope-narrowing pass.** All subsequent passes (Baker, enrichment, Alex) depend on this narrowing. This is proper nanopass design:

1. **Explicit dependency**: Baker's precondition is "reachability complete"
2. **Focused scaffolding**: Work remains focused on the right part of the graph
3. **Composability preserved**: Passes can still be reordered within their dependency constraints

The nanopass framework paper (Keep 2013) discusses this: passes may have ordering constraints when one pass's output is another's precondition. This is not tight coupling - it's semantic dependency made explicit.

## The Typed Tree Zipper

### Why a Zipper?

The typed tree (`FSharpExpr`) must be correlated with the syntax tree (`SynExpr`) during PSG construction. This correlation is NOT a simple traversal - it requires:

1. **Bidirectional navigation** - Move forward and backward through both trees
2. **Context preservation** - Know where we are in both trees
3. **Structural alignment** - Match FSharpExpr nodes to PSG nodes by range
4. **Backtracking** - When misaligned, back up and realign

A zipper provides exactly these capabilities.

### The Correlation Problem

```
SynExpr (syntax)              FSharpExpr (typed)
==================            ===================
App                           Call
├── LongIdent:op_Dollar       ├── TraitCall (SRTP resolved!)
├── Ident:WritableString      │   ├── sourceTypes
└── Ident:s                   │   ├── memberName: "$"
                              │   └── args...
                              └── ...
```

The syntax tree sees `op_Dollar` as a regular identifier. The typed tree knows it's an SRTP-resolved static member. Only by correlating them can we capture the resolution.

### Zipper Design

```fsharp
/// Typed tree zipper for FSharpExpr/PSG correlation
type TypedTreeZipper = {
    /// Current focus in typed tree
    TypedFocus: FSharpExpr
    /// Current focus in PSG
    PSGFocus: PSGNode
    /// Path back to root in typed tree
    TypedPath: TypedTreeCrumb list
    /// Path back to root in PSG
    PSGPath: PSGCrumb list
    /// Accumulated type information
    TypeInfo: Map<NodeId, TypeCorrelation>
}

type TypeCorrelation = {
    ResolvedType: FSharpType
    Constraints: FSharpGenericParameterConstraint list
    /// For TraitCall nodes - the resolved member info
    SRTPResolution: SRTPResolution option
}

type SRTPResolution = {
    TraitName: string           // e.g., "op_Dollar", "op_Addition"
    SourceTypes: FSharpType list
    MemberFlags: SynMemberFlags
    /// The resolved target (extracted from FCS)
    ResolvedMember: FSharpMemberOrFunctionOrValue option
}
```

### XParsec Integration

The zipper combines with XParsec for pattern-based correlation:

```fsharp
/// XParsec combinator for matching FSharpExpr patterns
let traitCall : Parser<TypedTreeZipper, SRTPResolution> =
    pexpr {
        let! focus = getFocus
        match focus.TypedFocus with
        | FSharpExprPatterns.TraitCall(sourceTypes, traitName, flags, paramTypes, retTypes, args) ->
            return {
                TraitName = traitName
                SourceTypes = sourceTypes
                MemberFlags = flags
                ResolvedMember = extractResolvedMember focus  // FCS internals
            }
        | _ ->
            return! fail "Not a trait call"
    }
```

## Soft-Delete Reachability

### Why Soft Delete?

Reachability must run BEFORE typed tree overlay to reduce the correlation workload. But the typed tree zipper needs the FULL structure to navigate - it can't correlate if nodes are missing.

Solution: **Soft delete** - mark nodes as unreachable but preserve structure.

```fsharp
/// Soft-delete reachability - marks but doesn't remove
let markUnreachable (psg: ProgramSemanticGraph) : ProgramSemanticGraph =
    let reachable = computeReachableSet psg
    { psg with
        Nodes = psg.Nodes |> Map.map (fun id node ->
            { node with
                IsReachable = Set.contains id reachable
                // Structure intact - Children, ParentId unchanged
            })
    }
```

### Benefits

1. **Typed tree zipper can navigate** - Full structure available for correlation
2. **Reduced work** - Zipper can skip unreachable nodes during correlation
3. **Deferred pruning** - Hard delete happens after typed overlay, if needed
4. **Debugging** - Can inspect what was marked unreachable

## The New Pipeline

### Phase 0: FCS Parse and Type Check

```
F# Source → FCS → (ParseResults[], CheckProjectResults)
```

FCS provides:
- `SynExpr` syntax trees (parse)
- `FSharpExpr` typed trees (check)
- Symbol resolution (check)

### Phase 1: Structural Construction

**Input**: `ParseResults[]` (SynExpr trees)
**Output**: `PSG₁` with nodes and ChildOf edges

**Language transformation**:
```
L_SynExpr → L_PSG₁
  terminals: (syntaxKind, range, fileName)
  Expr → (syntaxKind, children[], range, fileName)
```

This pass ONLY walks syntax, creating structural nodes. No FCS symbols yet.

### Phase 2: Symbol Correlation

**Input**: `PSG₁`, `CheckProjectResults`
**Output**: `PSG₂` with FSharpSymbol attachments

**Language transformation**:
```
L_PSG₁ → L_PSG₂
  Expr → (syntaxKind, children[], range, fileName, symbol?)
```

Uses FCS `GetAllUsesOfAllSymbols()` to correlate by range.

### Phase 3: Soft-Delete Reachability

**Input**: `PSG₂`
**Output**: `PSG₃` with IsReachable marks

**Language transformation**:
```
L_PSG₂ → L_PSG₃
  Expr → (syntaxKind, children[], range, fileName, symbol?, isReachable)
```

Marks unreachable nodes but preserves ALL structure.

### Phase 4: Typed Tree Overlay (The Zipper Pass)

**Input**: `PSG₃`, `CheckProjectResults` (FSharpExpr trees)
**Output**: `PSG₄` with Type, Constraints, SRTP resolution

**Language transformation**:
```
L_PSG₃ → L_PSG₄
  Expr → (syntaxKind, children[], range, fileName, symbol?, isReachable,
          type?, constraints?, srtpResolution?)
```

This is where the TypedTreeZipper walks FSharpExpr in parallel with PSG, correlating by range and attaching:
- Resolved types (after inference)
- Resolved constraints (after solving)
- SRTP resolution info (TraitCall → resolved member)

**Only processes reachable nodes** - skips `IsReachable = false`.

### Phase 5+: Enrichment Nanopasses

Each subsequent pass enriches `PSG₄`:

```
PSG₄ → FlattenApplications → PSG₅
PSG₅ → ReducePipeOperators → PSG₆
PSG₆ → AddDefUseEdges → PSG₇
PSG₇ → AnnotateParameters → PSG₈
PSG₈ → ClassifyOperations → PSG₉
...
```

### Phase N: Hard Pruning (Optional)

If needed for performance, a final pass can physically remove unreachable nodes:

```fsharp
let pruneUnreachable (psg: ProgramSemanticGraph) : ProgramSemanticGraph =
    { psg with
        Nodes = psg.Nodes |> Map.filter (fun _ n -> n.IsReachable)
        // Rebuild edges to only reference remaining nodes
    }
```

## SRTP Resolution: The Concrete Example

### The Problem

```fsharp
// Alloy/Console.fs
type WritableString =
    | WritableString
    static member inline ($) (WritableString, s: NativeStr) = writeNativeStr s
    static member inline ($) (WritableString, s: string) = writeSystemString s

let inline Write s = WritableString $ s
```

When we call `Console.Write "hello"`:
- Syntax sees: `App [op_Dollar, WritableString, "hello"]`
- Types resolve: `$` → `WritableString.op_Dollar(WritableString, string)` → `writeSystemString`

### The Solution

The typed tree overlay captures this via `FSharpExpr.TraitCall`:

```fsharp
// In TypedTreeZipper pass
match typedFocus with
| TraitCall(sourceTypes, "op_Dollar", flags, paramTypes, retTypes, args) ->
    let srtpInfo = {
        TraitName = "op_Dollar"
        SourceTypes = sourceTypes  // [WritableString]
        MemberFlags = flags
        // FCS has resolved this - extract the target
        ResolvedMember = Some (resolveToWriteSystemString ...)
    }
    attachToPSGNode currentNode srtpInfo
```

Now the PSG node has the SRTP resolution attached. Downstream passes can use it directly - no need to re-resolve.

## Implementation Roadmap

### Immediate (This Iteration)

1. **Restructure Builder/Main.fs** - Separate Phase 1 (structural) from Phase 2 (symbol correlation)
2. **Move reachability earlier** - Phase 3, soft-delete
3. **Stub TypedTreeZipper** - Phase 4 infrastructure

### Near-Term

4. **Implement TypedTreeZipper** - Full correlation logic
5. **Add SRTP capture** - TraitCall handling
6. **Update XParsec** - Combinators for typed patterns

### Validation

7. **Test with WritableString** - SRTP resolution captured
8. **Test with TimeLoop** - Mutable state flows correctly
9. **All samples compile** - End-to-end validation

## Impact on Downstream Components

### Alex/Zipper

With typed overlay complete, Alex's job simplifies:
- SRTP already resolved → follow resolved member
- Types already attached → no re-inference
- Reachable nodes marked → skip dead code

### PSGScribe

The scribe becomes a pure transcription layer:
- Follow PSG structure
- Emit MLIR based on node.Kind, node.Operation, node.SRTPResolution
- No resolution logic - that's done in Phase 4

### Nanopasses

Enrichment nanopasses work on fully-typed PSG:
- DefUseEdges can create more precise edges (with type info)
- ClassifyOperations can see resolved SRTP targets
- Future passes have complete information

## Validation Strategy

Each phase outputs intermediate PSG (when `-k` flag set):

```
target/intermediates/
├── psg_phase_1_structural.json
├── psg_phase_2_symbol_correlated.json
├── psg_phase_3_reachability_marked.json
├── psg_phase_4_typed_overlay.json
├── psg_phase_5_flattened.json
├── psg_phase_6_pipe_reduced.json
├── psg_phase_7_def_use.json
└── ...
```

Each can be inspected independently. Diffs show exactly what each pass added.

## Conclusion

This architectural revision makes PSG construction a **true nanopass pipeline** from the ground up. The critical addition is the **typed tree overlay pass** using a **zipper** for FSharpExpr/PSG correlation.

Key benefits:
1. **SRTP resolution captured at source** - No downstream guessing
2. **Soft-delete reachability** - Zipper can navigate, work is reduced
3. **Each phase is inspectable** - True to nanopass principles
4. **Downstream components simplified** - Work done once, at the right place

This is not incremental improvement - it's fixing a foundational blind spot that was causing second-order problems throughout the pipeline.

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
