# Pipeline Debugging Guide

## Critical Working Principle: Zoom Out Before Fixing

> **THIS IS A COMPILER. Compiler development has fundamentally different requirements than typical application development.**

The pipeline is a directed, multi-stage transformation where upstream decisions have cascading downstream effects.

### The Cardinal Rule

**When encountering ANY adverse finding that is NOT a baseline F# syntax error, you MUST stop and review the ENTIRE compiler pipeline before attempting a fix.**

### Why Patching In Place Is Wrong

**DO NOT "patch in place."** The instinct to fix a problem where it manifests is almost always wrong in compiler development.

A symptom appearing in MLIR generation may have its root cause in:
- Missing or stub implementations in Alloy
- Incorrect symbol capture in FCS ingestion
- Failed correlation in PSG construction
- Missing nanopass transformation
- Wrong reachability decisions

**Patching downstream creates technical debt that compounds.** A "fix" that works around a PSG deficiency:
- Masks the real problem
- Creates implicit dependencies on broken behavior
- Makes future fixes harder as the codebase grows
- Violates the architectural separation of concerns

## The Correct Debugging Approach

1. **Observe the symptom** - Note exactly what's wrong (wrong output, missing symbol, incorrect behavior)

2. **Trace upstream** - Walk backwards through the pipeline:
   ```
   Native Binary ← LLVM ← MLIR ← Alex/Zipper ← Nanopasses ← PSG ← FCS ← Alloy ← F# Source
   ```

3. **Find the root cause** - The fix belongs at the EARLIEST point in the pipeline where the defect exists

4. **Fix upstream** - Correct the root cause, then verify the fix propagates correctly through all downstream stages

5. **Validate end-to-end** - Confirm the native binary behaves correctly

## Forcing Questions Before Proposing Any Fix

1. "Have I read the relevant docs in `/docs/`?"
2. "Have I traced this issue through the full pipeline?"
3. "Am I fixing the ROOT CAUSE or patching a SYMPTOM?"
4. "Is my fix at the earliest possible point in the pipeline?"
5. "Will this fix work correctly as the compiler evolves, or am I creating hidden coupling?"
6. **"Am I adding library-specific logic to a layer that shouldn't know about libraries?"**
7. **"Does my fix require code generation to 'know' about specific function names?"**

If you cannot confidently answer all questions, you have not yet understood the problem well enough to fix it.

**If the answer to question 6 or 7 is "yes", STOP. You are about to make the same mistake again.**

## Pipeline Review Checklist

When a non-syntax issue arises:

### 1. Alloy Library Level
- Is the required function actually implemented (not a stub)?
- Does the function decompose to primitives through real F# code?
- Is there a full call graph, not just a placeholder?

### 2. FCS Ingestion Level
- Is the symbol being captured correctly?
- Is the type information complete and accurate?
- Are all dependencies being resolved?

### 3. PSG Construction Level
- Is the function reachable from the entry point?
- Are call edges being created correctly?
- Is symbol correlation working?
- Does the PSG show the full decomposed structure?

### 4. Nanopass Level
- Are def-use edges being created?
- Are operations being classified correctly?
- Is the PSG fully enriched before MLIR generation?

### 5. Alex/Zipper Level
- Is the traversal following the PSG structure?
- Is MLIR generation based on node kinds, not symbol names?
- Are there NO library-specific special cases?

### 6. MLIR/LLVM Level
- Is the generated IR valid?
- Are external declarations correct?
- Is the calling convention appropriate?

## Useful Debugging Commands

```bash
# Compile with verbose output to see pipeline stages
Firefly compile HelloWorld.fidproj --verbose

# Keep intermediate files for inspection
Firefly compile HelloWorld.fidproj -k

# Intermediate files are in target/intermediates/:
# - psg_phase_*.json - PSG at each nanopass phase
# - psg_diff_*.json - Differences between phases
# - *.mlir - Generated MLIR
# - *.ll - LLVM IR
```

## Common Pitfalls

1. **Stub Functions**: Alloy functions that compile but do nothing at runtime. Always verify the implementation decomposes to primitives.

2. **Library-Specific Logic**: Adding `if functionName = "Alloy.X.Y"` logic anywhere in code generation. This is ALWAYS wrong.

3. **Symbol Correlation**: FCS symbol correlation can fail silently. Check `[BUILDER] Warning:` messages in verbose output.

4. **Missing Nanopass**: If the PSG doesn't have the information you need, add a nanopass to enrich it. Don't compute it during MLIR generation.

5. **Layer Violations**: Any time you find yourself importing a module from a different pipeline stage, stop and reconsider.
