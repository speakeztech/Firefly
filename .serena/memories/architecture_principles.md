# Architecture Principles (from CLAUDE.md)

## CRITICAL: Deliberate, Didactic Approach Required

> **SLOW DOWN. Be deliberate. Be didactic. Use agents prolifically to gather context before acting.**

This is a sophisticated compiler project with deep architectural constraints. Fast, intuitive fixes are almost always wrong.

### Required Approach
1. **Use Task Agents PROLIFICALLY Before Acting** - Spawn multiple Task agents (subagent_type=Explore) IN PARALLEL to explore:
   - `~/repos/fsharp` - FCS implementation details
   - `~/repos/Alloy` - Native library patterns
   - `~/triton-cpu` - MLIR dialect patterns
   - `~/repos/fslang-spec` - F# language semantics
   - Related Firefly code via Serena's symbolic tools
   
2. **Confirm Intent** - When encountering ambiguity, stop and confirm with the user rather than making assumptions

3. **Trace Full Pipeline** - Every issue must be traced through the complete compilation pipeline before proposing a fix

4. **Understand Before Implementing** - Synthesize findings from all agent explorations before writing any code

### Anti-Pattern: "Going Too Fast"

**WRONG behavior:**
- Seeing an error and immediately patching where it manifests
- Adding stub implementations to "make it work"
- Using BCL/runtime dependencies when native implementations exist
- Making changes without exploring reference materials first

**CORRECT behavior:**
- Pause and spawn agents to explore context
- Understand WHY something works a certain way before changing it
- Look at how similar problems are solved in reference implementations
- Recognize when Alloy should provide native implementations

## The PSG Nanopass Pipeline (CRITICAL)

> **See: `docs/PSG_Nanopass_Architecture_v2.md` for authoritative details.**

PSG construction is a **true nanopass pipeline**, not a monolithic operation. Each phase does ONE thing:

```
Phase 1: Structural Construction    SynExpr → PSG with nodes + ChildOf edges
Phase 2: Symbol Correlation         + FSharpSymbol attachments (via FCS)
Phase 3: Soft-Delete Reachability   + IsReachable marks (structure preserved!)
Phase 4: Typed Tree Overlay         + Type, Constraints, SRTP resolution (Zipper)
Phase 5+: Enrichment Nanopasses     + def-use edges, operation classification, etc.
```

**CRITICAL Principles:**

1. **Soft-delete reachability** - Mark unreachable nodes but preserve structure. The typed tree zipper needs full structure for navigation.

2. **Typed tree overlay via zipper** - A zipper correlates `FSharpExpr` (typed tree) with PSG nodes by range, capturing:
   - Resolved types (after inference)
   - Resolved constraints (after solving)
   - **SRTP resolution** (TraitCall → resolved member) - THIS IS ESSENTIAL

3. **Each phase is inspectable** - Intermediate PSGs can be examined independently via `-k` flag.

**Why the Typed Tree Matters:**

Without the typed tree overlay, SRTP (Statically Resolved Type Parameters) cannot be resolved:

```fsharp
// Alloy/Console.fs
let inline Write s = WritableString $ s  // $ is SRTP-dispatched!
```

- Syntax tree sees: `App [op_Dollar, WritableString, s]`
- Typed tree knows: `TraitCall` resolving `$` to `WritableString.op_Dollar` → specific implementation

> **See: `docs/TypedTree_Zipper_Design.md` for zipper implementation details.**

## The Layer Separation Principle

> **Each layer in the pipeline has ONE responsibility. Do not mix concerns across layers.**

| Layer | Responsibility | DOES NOT |
|-------|---------------|----------|
| **Alloy** | Provide F# implementations of library functions | Contain stubs that expect compiler magic |
| **FCS** | Parse, type-check, resolve symbols | Transform or generate code |
| **PSG Builder** | Construct semantic graph from FCS output | Make targeting decisions |
| **Nanopasses** | Enrich PSG with edges, classifications | Generate MLIR or know about targets |
| **Alex/Zipper** | Traverse PSG, generate MLIR via bindings | Pattern-match on library names |
| **Bindings** | Platform-specific MLIR generation | Know about F# syntax or Alloy namespaces |
| **MLIR/LLVM** | Lower and optimize IR | Know about F# or Alloy |

## ANTIPATTERN: Emitter Layer

> **NEVER create an "emitter" abstraction between PSG and MLIR.**

The emitter is an antipattern because:
1. It sits between PSG and MLIR making interpretation decisions
2. It would need Alloy awareness (violates layer separation)
3. It circumvents the compositional XParsec + Bindings architecture
4. It accumulates special cases and coupling

See `docs/Emitter_Removal_Rebuild_Plan.md` for why it was removed.

## The Zipper + Bindings Architecture

**Alex generates MLIR through Zipper traversal and platform Bindings.**

The Zipper:
- Traverses the PSG structure bidirectionally
- Carries context (path, state) through traversal
- Provides focus on current node
- Does NOT contain MLIR generation logic

The Bindings:
- Contain platform-specific MLIR generation
- Are looked up by PSG node structure (not library names)
- Handle syscalls, memory operations, etc.
- Are organized by platform (Linux_x86_64, ARM, etc.)

**MLIR generation should NEVER:**
- Pattern-match on function names like "Alloy.Console.Write"
- Have special cases for specific libraries
- Contain conditional logic based on symbol names
- "Know" what Alloy is

## The Extern Primitive Surface

The ONLY acceptable "stubs" are **extern declarations** using `DllImport("__fidelity")`:

```fsharp
// Alloy/Primitives.fs - declarative extern primitives
[<DllImport("__fidelity", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "fidelity_write_bytes")>]
extern int writeBytes(int fd, nativeptr<byte> buffer, int count)

[<DllImport("__fidelity", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "fidelity_read_bytes")>]
extern int readBytes(int fd, nativeptr<byte> buffer, int maxCount)
```

The `"__fidelity"` library name is a marker for Alex. Alex provides platform-specific implementations.
Everything else must decompose to these primitives through real F# code.
