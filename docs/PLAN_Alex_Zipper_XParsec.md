# Implementation Plan: Alex/Zipper XParsec Integration

## CRITICAL: NO EMITTER LAYER

> **The "emitter" is an ANTIPATTERN that was removed.** See `Emitter_Removal_Rebuild_Plan.md`.
>
> Alex traverses PSG via Zipper, uses XParsec to match patterns, and dispatches to Bindings.
> There is NO intermediate "emitter" abstraction.

## Architecture

```
PSG (full semantic graph)
    → Zipper (XParsec pattern matching)
        → Bindings (platform-specific MLIR)
            → MLIR → LLVM → Native
```

### Why This Architecture

1. **Alloy is platform-agnostic** - provides native types and extern primitives only
2. **PSG captures full semantics** - decomposed call graphs, def-use edges, type info
3. **Alex/Zipper does targeting** - structural decisions (stack layout, loop flattening) happen here
4. **Bindings are platform-specific** - syscalls, memory operations, calling conventions

The same Alloy code compiles to different MLIR for different targets:
- x86_64 Linux: alloca, syscalls
- ARM Cortex-M33: stack-only, no syscalls, loop flattening
- WASM: different memory model entirely

## Validation Samples (DO NOT MODIFY)

| Sample | Complexity | Tests |
|--------|------------|-------|
| 01_HelloWorldDirect | Basic | Console.Write, Console.WriteLine with static strings |
| 02_HelloWorldSaturated | Medium | Let bindings, string interpolation, I/O |
| 03_HelloWorldHalfCurried | Advanced | Pipe operators, function values |

---

## Phase 0: Cleanup ✅ COMPLETE

### Task 0.1: Fix Misleading Comment in Alloy ✅
- **File**: `~/repos/Alloy/src/Primitives.fs` line 17
- **Status**: Complete

---

## Phase 1: XParsec Combinator Wrappers ✅ COMPLETE

### Task 1.1: Add Combinators to PSGXParsec.fs ✅
- **File**: `/home/hhh/repos/Firefly/src/Alex/Traversal/PSGXParsec.fs`
- **Added**: pMap, pBind, pLeft, pRight, pAnd, pOr, pMany, pMany1, pChoice, pChoiceL, pOpt, pSkip, pSepBy, pSepBy1, pBetween, pLookAhead, pNotFollowedBy, pSkipMany, pSkipMany1, pTuple2-5, pPipe2-5, pReturn, pFail, pLabel
- **Note**: manyTill, chainl1, chainr1 not wrapped (require equality on PSGNode)

### Task 1.2: Verify XParsec Imports ✅
- XParsec package v0.1.0 referenced in Firefly.fsproj
- System.Collections.Immutable added for ImmutableArray

---

## Phase 2: Zipper Traversal Enhancement

**Goal**: Enhance Zipper to use XParsec combinators for pattern-matching PSG structures and dispatching to Bindings.

**Checkpoint**: Sample 01 compiles and runs

### Task 2.1: Review Current Zipper Capabilities
- Understand existing traversal patterns in PSGZipper.fs
- Identify what XParsec patterns are needed for basic expression handling

### Task 2.2: Implement Basic Expression Traversal
Using XParsec combinators in Zipper traversal for:
- Constants (Int32, String, Bool, Unit)
- Sequential expressions
- Function applications (leading to extern dispatch)

### Task 2.3: Connect to ExternDispatch
When Zipper traversal encounters an extern call (PSG structure shows `DllImport("__fidelity")`):
- Extract entry point from PSG node metadata
- Get target platform from compilation context (fidproj)
- Dispatch to Bindings via ExternDispatch registry

### Task 2.4: Validate Sample 01
```bash
cd /home/hhh/repos/Firefly/samples/console/FidelityHelloWorld/01_HelloWorldDirect
../../../../src/bin/Debug/net9.0/Firefly compile HelloWorldDirect.fidproj -k --verbose
./target/hello_direct
# Expected: "Hello, World!" and exit code 0
```

---

## Phase 3: Function Calls and I/O

**Goal**: Handle full function call decomposition to extern primitives

**Checkpoint**: Sample 02 compiles and runs

### Task 3.1: Trace Console.Write Decomposition
- Console.Write → write (internal) → writeBytes (extern primitive)
- Verify PSG contains full call graph

### Task 3.2: Handle Let Bindings
- Emit MLIR for binding (use def-use edges, not scope tracking)
- Record SSA value in Zipper state

### Task 3.3: String Literal Handling
- Global string constants in MLIR
- NativeStr struct construction (ptr + length)

### Task 3.4: Validate Sample 02

---

## Phase 4: Pipe Operators and Function Values

**Goal**: Handle higher-order patterns

**Checkpoint**: Sample 03 compiles and runs

### Task 4.1: Pipe Operator Handling
- PSG nanopass should have already reduced pipes
- Verify traversal handles the reduced form

### Task 4.2: Function Value References
- Partial application
- Function as argument

### Task 4.3: Validate Sample 03

---

## Key Principles

1. **No Alloy awareness in Alex** - match PSG structure, not symbol names
2. **No emitter layer** - Zipper + XParsec + Bindings only
3. **Target from fidproj** - platform is data, not code organization
4. **Validate incrementally** - each phase has a sample checkpoint
