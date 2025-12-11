# Baker Implementation Plan

## Overview

This document outlines the implementation phases for Baker, the type resolution **consolidation component library** that provides Phase 4 (Typed Tree Overlay) of the PSG nanopass pipeline.

**CRITICAL PRINCIPLE**: Baker operates **AFTER reachability analysis** (Phase 3). It only processes the narrowed compute graph - nodes marked `IsReachable = true`. This is non-negotiable.

## Current State

### Working Infrastructure
- `TypeIntegration.fs` builds a range-based type index from FCS
- `ResolveSRTP.fs` captures TraitCall nodes and infers resolutions
- `CompilationOrchestrator.fs` has type substitution infrastructure for inlining

### The Gap
- Static member bindings don't get symbol correlation (Phase 2 limitation)
- `op_Dollar` body lookup fails because PSG doesn't have the body mapped
- Type substitutions work but body inlining fails

## Implementation Phases

### Phase A: Foundation (Minimum Viable Baker)

**Goal**: Get static member body lookup working for SRTP operators.

**Files to Create**:
```
src/Baker/
├── Baker.fs           # Module entry point
└── MemberBodyMapper.fs # Static member → body mapping
```

**Implementation**:

1. **Create `src/Baker/` directory**

2. **Implement `MemberBodyMapper.fs`**:
   - Walk `FSharpImplementationFileDeclaration.MemberOrFunctionOrValue(mfv, args, expr)`
   - Create `MemberBodyMapping` records
   - Build index: `(DeclaringEntity, MemberName, ParameterTypes) → FSharpExpr body`
   - **CRITICAL: Only process members referenced by reachable nodes**

3. **Extend PSG Types**:
   - Add `MemberBodies: Map<string, MemberBodyMapping>` to `ProgramSemanticGraph`

4. **Integrate with Pipeline** (POST-REACHABILITY):
   - Call `Baker.extractMemberBodies` **AFTER Phase 3 (reachability)**
   - Only extract bodies for members referenced by `IsReachable = true` nodes
   - Store in PSG for Alex to consume

5. **Update Alex body lookup**:
   - Replace `tryGetCalledFunctionBodyByName` to check `psg.MemberBodies`
   - For SRTP-resolved members, look up body directly

**Validation**: `01_HelloWorldDirect` compiles and executes correctly.

### Phase B: SRTP Resolution Capture

**Goal**: Capture SRTP resolutions properly using FCS public API.

**Files**:
```
src/Baker/
├── Types.fs           # SRTPResolution, MemberBodyMapping types
└── SRTPResolver.fs    # TraitCall extraction
```

**Implementation**:

1. **Define types in `Types.fs`**:
   ```fsharp
   type SRTPResolution = {
       CallSite: NodeId
       ResolvedMember: FSharpMemberOrFunctionOrValue
       TypeSubstitutions: Map<string, FSharpType>
   }
   ```

2. **Implement `SRTPResolver.fs`**:
   - Walk typed expressions looking for `FSharpExprPatterns.TraitCall`
   - For each TraitCall, extract constraint and infer resolution
   - Correlate with PSG nodes by range
   - Record in `PSG.SRTPResolutions`

3. **Update Alex emission**:
   - When emitting App node, check `psg.SRTPResolutions`
   - If SRTP resolution found, use it for body lookup and type subs

**Validation**: SRTP operators resolve correctly, body inlining works.

### Phase C: Full TypedTreeZipper

**Goal**: Complete typed tree correlation for all expressions.

**Files**:
```
src/Baker/
└── TypedTreeZipper.fs # Two-tree zipper
```

**Implementation**:

1. **Implement `TypedTreeZipper.fs`**:
   - Zipper state: `(FSharpExpr cursor, PSGNode cursor, Context)`
   - Navigation operations: `moveToChild`, `moveToSibling`, `moveUp`
   - Correlation strategies: range-based, structure-based

2. **Type overlay integration**:
   - As zipper navigates, overlay resolved types onto PSG nodes
   - Replace `TypeIntegration.fs` functionality

3. **Constraint capture**:
   - Extract constraint information from FSharpType
   - Store in `PSGNode.Constraints`

**Validation**: All types correctly resolved, no missing type information.

### Phase D: Consolidation

**Goal**: Clean up duplicated functionality.

**Changes**:
- Remove `Core/PSG/TypeIntegration.fs` (absorbed into Baker)
- Remove `Core/PSG/Nanopass/ResolveSRTP.fs` (absorbed into Baker)
- Update imports/references throughout codebase

**Validation**: All samples compile and execute correctly.

## File Changes Summary

### New Files
| File | Phase | Purpose |
|------|-------|---------|
| `src/Baker/Baker.fs` | A | Entry point, orchestration |
| `src/Baker/MemberBodyMapper.fs` | A | Static member body extraction |
| `src/Baker/Types.fs` | B | Baker-specific types |
| `src/Baker/SRTPResolver.fs` | B | TraitCall resolution |
| `src/Baker/TypedTreeZipper.fs` | C | Two-tree zipper |

### Modified Files
| File | Phase | Changes |
|------|-------|---------|
| `src/Core/PSG/Types.fs` | A | Add MemberBodies, SRTPResolutions fields |
| `src/Firefly.fsproj` | A | Add Baker files |
| `src/Alex/Pipeline/CompilationOrchestrator.fs` | A | Integrate Baker, update body lookup |

### Removed Files (Phase D)
| File | Replacement |
|------|-------------|
| `src/Core/PSG/TypeIntegration.fs` | Baker/TypedTreeZipper.fs |
| `src/Core/PSG/Nanopass/ResolveSRTP.fs` | Baker/SRTPResolver.fs |

## Testing Strategy

### Per-Phase Validation
- **Phase A**: `01_HelloWorldDirect` compiles and prints "Hello, World!"
- **Phase B**: SRTP operators resolve; debug output shows correct body lookup
- **Phase C**: All type information present; `-k` output shows types on all nodes
- **Phase D**: Full sample suite passes; no regressions

### Regression Tests
After each phase, run through samples in order:
1. `01_HelloWorldDirect`
2. `02_HelloWorldSaturated`
3. `03_HelloWorldHalfCurried`
4. `04_HelloWorldFullCurried`
5. `TimeLoop`

## Risk Mitigation

### FCS API Stability
- Use public API (`FSharpExprPatterns`, `FSharpImplementationFileDeclaration`)
- Document any reflection usage (minimize)
- Test against FCS updates

### Performance
- Build indices once during Phase 4
- O(1) lookup during Alex emission
- Profile if needed; range-based lookup is fast

### Complexity
- Keep each component focused (single responsibility)
- Write unit tests for zipper navigation
- Document correlation strategies

## Dependencies

- FSharp.Compiler.Service (existing)
- No new external dependencies
- Alloy library (unchanged)

## Design Invariants (NON-NEGOTIABLE)

These invariants must be maintained throughout implementation:

1. **Post-Reachability Activation**: Baker MUST run after Phase 3 (reachability). It MUST NOT process unreachable nodes.

2. **Narrowed Graph Scope**: All Baker operations work on `IsReachable = true` nodes only.

3. **Zipper Coherence**: Baker's TypedTreeZipper and Alex's PSGZipper MUST operate on the same narrowed scope.

4. **Consolidation Library**: Baker consolidates type-level transforms. Do NOT scatter type resolution logic elsewhere.

5. **No Scope Mismatch**: SRTP resolutions captured by Baker MUST match exactly what Alex will need to emit.

## Timeline Estimate

| Phase | Complexity | Effort |
|-------|------------|--------|
| A | Low | Days |
| B | Medium | Days |
| C | High | Week |
| D | Low | Day |

Phase A provides immediate value (fixes the current blocking issue).
Phases B-D can proceed incrementally.

## Next Steps

1. Create `src/Baker/` directory structure
2. Implement Phase A (MemberBodyMapper)
3. Validate with `01_HelloWorldDirect`
4. Proceed to Phase B if needed
