# Typed Tree Zipper Design

## Purpose

This document specifies the design of the **Typed Tree Zipper** - the mechanism that correlates F# Compiler Services (FCS) typed trees (`FSharpExpr`) with PSG nodes built from syntax trees (`SynExpr`).

**The Problem**: The syntax tree (`SynExpr`) provides structure, but critical semantic information exists only in the typed tree (`FSharpExpr`). Most importantly:
- **SRTP Resolution**: Statically Resolved Type Parameter calls (like `WritableString $ s`) are resolved in the typed tree to concrete member implementations
- **Type Inference Results**: Generic type parameters filled in with actual types
- **Constraint Solutions**: Type constraints resolved to specific implementations
- **Overload Resolution**: Which overloaded method was chosen

Without the typed tree overlay, the PSG lacks this semantic information, and downstream passes (including Alex/MLIR generation) cannot correctly handle SRTP dispatch.

## Reference: The Huet Zipper

The zipper data structure (Huet, 1997) provides:
1. **Focus**: A current position in a tree
2. **Context/Path**: The "hole" in the tree leading back to the root
3. **Navigation**: Operations to move focus (up, down, left, right)
4. **Modification**: Operations to modify at focus and reconstruct

For our use case, we need a **two-tree zipper** that maintains synchronized focus in both the typed tree and the PSG.

## Architecture

### Core Types

```fsharp
/// A crumb in the typed tree path
type TypedTreeCrumb = {
    /// Parent expression
    Parent: FSharpExpr
    /// Index of current focus in parent's children
    ChildIndex: int
    /// Siblings before current focus
    Before: FSharpExpr list
    /// Siblings after current focus
    After: FSharpExpr list
}

/// A crumb in the PSG path
type PSGCrumb = {
    /// Parent node ID
    ParentId: NodeId
    /// Index in parent's children
    ChildIndex: int
    /// Range of the parent node
    ParentRange: Range
}

/// The two-tree zipper
type TypedTreeZipper = {
    /// Current focus in typed tree
    TypedFocus: FSharpExpr
    /// Current focus in PSG
    PSGFocus: PSGNode
    /// PSG reference for node lookups
    PSG: ProgramSemanticGraph
    /// Path back to root in typed tree
    TypedPath: TypedTreeCrumb list
    /// Path back to root in PSG
    PSGPath: PSGCrumb list
    /// Accumulated type correlations
    Correlations: Map<NodeId, TypeCorrelation>
}
```

### Type Correlation Information

```fsharp
/// Information captured from typed tree for a PSG node
type TypeCorrelation = {
    /// The resolved type (after inference)
    ResolvedType: FSharpType option
    /// Generic parameter constraints (after solving)
    Constraints: FSharpGenericParameterConstraint list
    /// For SRTP calls - the resolved member
    SRTPResolution: SRTPResolution option
    /// The original FSharpExpr for debugging
    TypedExpr: FSharpExpr option
}

/// SRTP resolution details
type SRTPResolution = {
    /// The trait member name (e.g., "op_Dollar", "op_Addition")
    TraitName: string
    /// Source types constraining the trait
    SourceTypes: FSharpType list
    /// Member flags from FCS
    MemberFlags: MemberFlags
    /// The resolved target member (if extractable)
    ResolvedMember: FSharpMemberOrFunctionOrValue option
    /// Argument types (for overload disambiguation)
    ArgumentTypes: FSharpType list
    /// Return type
    ReturnType: FSharpType option
}
```

## Correlation Strategy

### Range-Based Matching

The primary correlation mechanism uses **source ranges**. Both `SynExpr` and `FSharpExpr` carry `Range` information:

```fsharp
/// Attempt to correlate typed focus with PSG focus by range
let correlateByRange (zipper: TypedTreeZipper) : bool =
    let typedRange = zipper.TypedFocus.Range
    let psgRange = zipper.PSGFocus.Range
    rangesOverlap typedRange psgRange
```

However, ranges don't always match exactly due to:
- Desugaring (the typed tree may have synthetic nodes)
- Range adjustments by FCS
- Different granularity between syntax and typed trees

### Structure-Based Matching

When ranges are ambiguous, structure provides disambiguation:

```fsharp
/// Match by structure when ranges overlap multiple candidates
let correlateByStructure (typedExpr: FSharpExpr) (psgNode: PSGNode) : float =
    match typedExpr, psgNode.SyntaxKind with
    | FSharpExpr.Application _, "App:FunctionCall" -> 1.0
    | FSharpExpr.Let _, "Binding:Let" -> 1.0
    | FSharpExpr.IfThenElse _, "IfThenElse" -> 1.0
    | FSharpExpr.Value _, "Ident" -> 0.8  // May need name check
    | FSharpExpr.TraitCall _, "App:FunctionCall" -> 0.9  // SRTP appears as App in syntax
    | _ -> 0.0
```

### SRTP Detection

The critical pattern to detect is `TraitCall`:

```fsharp
/// Extract SRTP information from typed expression
let extractSRTPInfo (expr: FSharpExpr) : SRTPResolution option =
    match expr with
    | FSharpExprPatterns.TraitCall(sourceTypes, traitName, memberFlags, argTypes, retType, _) ->
        Some {
            TraitName = traitName
            SourceTypes = sourceTypes |> List.ofSeq
            MemberFlags = memberFlags
            ResolvedMember = tryExtractResolvedMember expr  // FCS internals
            ArgumentTypes = argTypes |> List.ofSeq
            ReturnType = Some retType
        }
    | _ -> None
```

## Zipper Operations

### Navigation

```fsharp
/// Move down to first child in both trees
let goDown (zipper: TypedTreeZipper) : TypedTreeZipper option =
    let typedChildren = getTypedChildren zipper.TypedFocus
    let psgChildren = getPSGChildren zipper.PSG zipper.PSGFocus

    match typedChildren, psgChildren with
    | firstTyped :: restTyped, firstPSG :: _ ->
        Some {
            TypedFocus = firstTyped
            PSGFocus = firstPSG
            PSG = zipper.PSG
            TypedPath = {
                Parent = zipper.TypedFocus
                ChildIndex = 0
                Before = []
                After = restTyped
            } :: zipper.TypedPath
            PSGPath = {
                ParentId = zipper.PSGFocus.Id
                ChildIndex = 0
                ParentRange = zipper.PSGFocus.Range
            } :: zipper.PSGPath
            Correlations = zipper.Correlations
        }
    | _ -> None

/// Move to next sibling
let goRight (zipper: TypedTreeZipper) : TypedTreeZipper option =
    match zipper.TypedPath, zipper.PSGPath with
    | typedCrumb :: typedRest, psgCrumb :: psgRest ->
        match typedCrumb.After with
        | nextTyped :: afterTyped ->
            let psgSiblings = getPSGSiblings zipper.PSG psgCrumb.ParentId
            let nextPSGIndex = psgCrumb.ChildIndex + 1
            match List.tryItem nextPSGIndex psgSiblings with
            | Some nextPSG ->
                Some {
                    TypedFocus = nextTyped
                    PSGFocus = nextPSG
                    PSG = zipper.PSG
                    TypedPath = {
                        typedCrumb with
                            ChildIndex = typedCrumb.ChildIndex + 1
                            Before = zipper.TypedFocus :: typedCrumb.Before
                            After = afterTyped
                    } :: typedRest
                    PSGPath = {
                        psgCrumb with ChildIndex = nextPSGIndex
                    } :: psgRest
                    Correlations = zipper.Correlations
                }
            | None -> None  // PSG structure doesn't match
        | [] -> None  // No more siblings
    | _ -> None  // At root

/// Move up to parent
let goUp (zipper: TypedTreeZipper) : TypedTreeZipper option =
    match zipper.TypedPath, zipper.PSGPath with
    | typedCrumb :: typedRest, psgCrumb :: psgRest ->
        match Map.tryFind psgCrumb.ParentId zipper.PSG.Nodes with
        | Some parentNode ->
            Some {
                TypedFocus = typedCrumb.Parent
                PSGFocus = parentNode
                PSG = zipper.PSG
                TypedPath = typedRest
                PSGPath = psgRest
                Correlations = zipper.Correlations
            }
        | None -> None
    | _ -> None  // At root
```

### Correlation Capture

```fsharp
/// Capture type information at current focus
let captureCorrelation (zipper: TypedTreeZipper) : TypedTreeZipper =
    let correlation = {
        ResolvedType = tryGetType zipper.TypedFocus
        Constraints = tryGetConstraints zipper.TypedFocus
        SRTPResolution = extractSRTPInfo zipper.TypedFocus
        TypedExpr = Some zipper.TypedFocus
    }
    { zipper with
        Correlations = Map.add zipper.PSGFocus.Id correlation zipper.Correlations
    }
```

## Integration with Nanopass Pipeline

### Phase 4: Typed Tree Overlay

This is the nanopass that runs the zipper:

```fsharp
/// Phase 4: Overlay typed tree information onto PSG
let overlayTypedTree
    (psg: ProgramSemanticGraph)
    (checkResults: FSharpCheckProjectResults)
    : ProgramSemanticGraph =

    // Get typed trees for each implementation file
    let typedTrees =
        checkResults.AssemblyContents.ImplementationFiles
        |> Seq.map (fun f -> f.FileName, f.Declarations)
        |> Map.ofSeq

    // Find entry points in PSG and their corresponding typed trees
    let mutable correlations = Map.empty

    for entryPoint in psg.EntryPoints do
        match Map.tryFind entryPoint.FileName typedTrees with
        | Some declarations ->
            // Create zipper at root
            let rootPSGNode = Map.find entryPoint.NodeId psg.Nodes
            let rootTypedExpr = findMatchingTypedRoot declarations rootPSGNode

            match rootTypedExpr with
            | Some typedRoot ->
                let zipper = {
                    TypedFocus = typedRoot
                    PSGFocus = rootPSGNode
                    PSG = psg
                    TypedPath = []
                    PSGPath = []
                    Correlations = Map.empty
                }

                // Traverse and correlate
                let finalZipper = traverseAndCorrelate zipper
                correlations <- Map.fold (fun acc k v -> Map.add k v acc) correlations finalZipper.Correlations
            | None ->
                printfn "[TYPED-OVERLAY] Warning: No typed tree match for entry point %s" entryPoint.FileName
        | None ->
            printfn "[TYPED-OVERLAY] Warning: No typed declarations for %s" entryPoint.FileName

    // Apply correlations to PSG nodes
    { psg with
        Nodes = psg.Nodes |> Map.map (fun id node ->
            match Map.tryFind id correlations with
            | Some correlation ->
                { node with
                    TypeCorrelation = Some correlation
                    // For SRTP, also update Operation field
                    Operation =
                        match correlation.SRTPResolution with
                        | Some srtp -> Some (SRTPDispatch srtp)
                        | None -> node.Operation
                }
            | None -> node
        )
    }
```

### Traversal Strategy

The traversal must handle mismatches between typed and syntax trees:

```fsharp
/// Traverse both trees, correlating where possible
let rec traverseAndCorrelate (zipper: TypedTreeZipper) : TypedTreeZipper =
    // Skip unreachable nodes (soft-deleted by reachability pass)
    if not zipper.PSGFocus.IsReachable then
        zipper
    else
        // Capture correlation at current position
        let zipper = captureCorrelation zipper

        // Try to descend into children
        match goDown zipper with
        | Some childZipper ->
            // Process all children via sibling traversal
            let rec processChildren z =
                let z = traverseAndCorrelate z
                match goRight z with
                | Some siblingZipper -> processChildren siblingZipper
                | None -> z

            let afterChildren = processChildren childZipper

            // Go back up
            match goUp afterChildren with
            | Some parentZipper -> parentZipper
            | None -> afterChildren  // Should not happen

        | None ->
            // Leaf node, nothing more to do
            zipper
```

## XParsec Integration

The zipper can be used with XParsec for pattern-based matching:

```fsharp
/// Parser state is the zipper
type ZipperParser<'a> = Parser<TypedTreeZipper, 'a>

/// Match SRTP call pattern
let srtpCall : ZipperParser<SRTPResolution> =
    pzipper {
        let! focus = getTypedFocus
        match extractSRTPInfo focus with
        | Some srtp -> return srtp
        | None -> return! fail "Not an SRTP call"
    }

/// Match specific SRTP operator
let srtpOperator (opName: string) : ZipperParser<SRTPResolution> =
    pzipper {
        let! srtp = srtpCall
        if srtp.TraitName = opName then
            return srtp
        else
            return! fail $"Expected SRTP operator {opName}, got {srtp.TraitName}"
    }

/// Match Console.Write SRTP pattern (WritableString $ s)
let consoleWriteSRTP : ZipperParser<SRTPResolution> =
    srtpOperator "op_Dollar" >>= fun srtp ->
        match srtp.SourceTypes with
        | [t] when t.TypeDefinition.DisplayName = "WritableString" -> succeed srtp
        | _ -> fail "Not a WritableString $ pattern"
```

## Impact on Downstream Components

### Alex/PSGScribe

With typed overlay complete, PSGScribe can:

```fsharp
/// Emit MLIR for an operation, using SRTP info if present
let emitOperation (psg: ProgramSemanticGraph) (node: PSGNode) : MLIR<Val> =
    match node.Operation with
    | Some (SRTPDispatch srtp) ->
        // Follow SRTP resolution to concrete implementation
        match srtp.ResolvedMember with
        | Some member ->
            // Emit call to resolved member
            emitResolvedMemberCall psg member node
        | None ->
            // Fallback: try to resolve from trait info
            emitSRTPCall psg srtp node

    | Some (FunctionCall sym) ->
        // Regular function call
        emitFunctionCall psg sym node

    | None ->
        // No operation classification - emit based on structure
        emitByStructure psg node
```

### Validation

The typed overlay can be validated by checking:

```fsharp
/// Validate that all SRTP calls have resolutions
let validateSRTPResolution (psg: ProgramSemanticGraph) : ValidationResult =
    let srtpNodes =
        psg.Nodes
        |> Map.values
        |> Seq.filter (fun n ->
            match n.Operation with
            | Some (SRTPDispatch _) -> true
            | _ -> false)

    let unresolved =
        srtpNodes
        |> Seq.filter (fun n ->
            match n.TypeCorrelation with
            | Some tc -> tc.SRTPResolution.IsNone
            | None -> true)
        |> Seq.toList

    if List.isEmpty unresolved then
        ValidationResult.Success
    else
        ValidationResult.Failure $"Unresolved SRTP calls: {unresolved.Length}"
```

## Implementation Roadmap

1. **Define Core Types** (`Core/PSG/TypedTree/Types.fs`)
   - TypedTreeZipper, TypeCorrelation, SRTPResolution

2. **Implement Navigation** (`Core/PSG/TypedTree/Navigation.fs`)
   - goDown, goUp, goRight, goLeft

3. **Implement Correlation** (`Core/PSG/TypedTree/Correlation.fs`)
   - Range matching, structure matching, SRTP extraction

4. **Implement Phase 4 Nanopass** (`Core/PSG/Nanopass/TypedTreeOverlay.fs`)
   - Main traversal and PSG annotation

5. **XParsec Combinators** (`Core/PSG/TypedTree/XParsec.fs`)
   - Zipper-based parser combinators

6. **Update Alex/PSGScribe** to consume SRTP information

## Conclusion

The Typed Tree Zipper bridges the gap between syntax (structure) and semantics (meaning). By correlating `FSharpExpr` with PSG nodes, we capture SRTP resolutions that are otherwise invisible to downstream passes.

This is Phase 4 of the nanopass pipeline - it runs after soft-delete reachability so the zipper only processes reachable code, but before enrichment nanopasses so they can use type information.
