# Implementation Plan: XParsec Compositional MLIR Generation

## Overview

This plan covers the work to achieve **fully compositional MLIR generation via XParsec**, where:
- Alloy provides platform-agnostic native types and primitives
- FCS ingests and type-checks, producing rich semantic information
- PSG captures full AST + semantics
- XParsec combinators traverse PSG and pattern-match structures
- Alex Bindings produce platform-specific MLIR

**Validation Samples**: FidelityHelloWorld 01, 02, 03 (progressive complexity)

**Checkpoints**: After each phase, pause for user review before proceeding.

---

## Phase 0: Cleanup and Foundation
**Estimated Scope**: Small (1-2 files)
**Checkpoint**: After completion

### Task 0.1: Fix Misleading Comment in Alloy
**File**: `~/repos/Alloy/src/Primitives.fs`
**Line**: 17

**Current**:
```fsharp
/// Platform implementations belong in Alex/Bindings/{Linux,MacOS,Windows,Embedded}/.
```

**Change to**:
```fsharp
/// Alex provides platform-specific implementations via Bindings modules,
/// dispatching by entry point name and target platform to generate
/// platform-appropriate MLIR (syscalls on Linux/macOS, Win32 APIs, etc.).
```

**Rationale**: The current comment suggests platform-specific directories, contradicting the "platform-as-data" architecture.

### Task 0.2: Verify Alloy Platform-Agnostic Status
**Action**: Confirm audit findings - Alloy has no platform-specific code
**Result**: ✅ Confirmed clean (agent audit complete)

---

## Phase 1: XParsec Combinator Wrappers
**Estimated Scope**: Medium (1 file, ~100-150 lines)
**Checkpoint**: After completion

### Task 1.1: Add Missing Combinators to PSGXParsec.fs
**File**: `/home/hhh/repos/Firefly/src/Alex/Traversal/PSGXParsec.fs`

Add PSG-specialized wrappers for XParsec combinators that aren't yet exposed:

```fsharp
// === REPETITION ===

/// Match zero or more children with parser p
let pMany (p: PSGChildParser<'T>) : PSGChildParser<'T list>

/// Match one or more children with parser p
let pMany1 (p: PSGChildParser<'T>) : PSGChildParser<'T list>

/// Match children separated by separator parser
let pSepBy (sep: PSGChildParser<unit>) (p: PSGChildParser<'T>) : PSGChildParser<'T list>

/// Match children separated by separator, at least one required
let pSepBy1 (sep: PSGChildParser<unit>) (p: PSGChildParser<'T>) : PSGChildParser<'T list>

// === CHOICE ===

/// Try each parser in order, return first success
let pChoice (ps: PSGChildParser<'T> list) : PSGChildParser<'T>

/// Choice with custom error message on failure
let pChoiceL (ps: PSGChildParser<'T> list) (label: string) : PSGChildParser<'T>

// === OPTIONAL ===

/// Try parser, return ValueSome on success, ValueNone on failure (never fails)
let pOpt (p: PSGChildParser<'T>) : PSGChildParser<'T voption>

/// Try parser, return default value on failure
let pOptDefault (defaultVal: 'T) (p: PSGChildParser<'T>) : PSGChildParser<'T>

// === SEQUENCING ===

/// Run p1, discard result, run p2, return p2's result
let pThen (p1: PSGChildParser<'a>) (p2: PSGChildParser<'b>) : PSGChildParser<'b>

/// Run p1, run p2, return tuple of both results
let pAnd (p1: PSGChildParser<'a>) (p2: PSGChildParser<'b>) : PSGChildParser<'a * 'b>

/// Run p, apply function to result
let pMap (f: 'a -> 'b) (p: PSGChildParser<'a>) : PSGChildParser<'b>

// === LOOKAHEAD ===

/// Run parser but don't consume input (for peeking)
let pLookAhead (p: PSGChildParser<'T>) : PSGChildParser<'T>

/// Succeed if parser would fail (negative lookahead)
let pNotFollowedBy (p: PSGChildParser<'T>) : PSGChildParser<unit>
```

### Task 1.2: Verify XParsec Import
**File**: `/home/hhh/repos/Firefly/src/Alex/Traversal/PSGXParsec.fs`

Ensure proper imports from `~/repos/XParsec`:
```fsharp
open XParsec
open XParsec.Combinators
open XParsec.Parsers
```

---

## Phase 2: Expression Dispatcher Foundation
**Estimated Scope**: Medium (2 new files, ~200-300 lines)
**Checkpoint**: After completion
**Validation**: Sample 01 (HelloWorldDirect) compiles

### Task 2.1: Create Binding Registry
**File**: `/home/hhh/repos/Firefly/src/Alex/Bindings/Registry.fs` (NEW)

Centralized binding registration and lookup:

```fsharp
module Alex.Bindings.Registry

/// Register all platform bindings (called once at startup)
let registerAll () : unit

/// Check if a binding exists for the given extern entry point
let hasBinding (entryPoint: string) : bool

/// Get all registered entry points (for diagnostics)
let registeredEntryPoints () : string list
```

### Task 2.2: Create Expression Parser Module
**File**: `/home/hhh/repos/Firefly/src/Alex/Emit/ExprEmitter.fs` (NEW)

Main expression dispatcher using XParsec:

```fsharp
module Alex.Emit.ExprEmitter

open Alex.Traversal.PSGXParsec
open Alex.Traversal.PSGZipper

/// Main expression emitter - dispatches based on node kind
let rec pExpr : PSGChildParser<ExprResult>

/// Emit a constant (Int32, String, Bool, Unit)
let pConst : PSGChildParser<ExprResult>

/// Emit an identifier reference (follows def-use edges)
let pIdent : PSGChildParser<ExprResult>

/// Emit a function application
let pApp : PSGChildParser<ExprResult>

/// Emit a sequential expression (expr1; expr2; ...)
let pSequential : PSGChildParser<ExprResult>

/// Entry point: emit MLIR for a PSG starting from entry point
let emitModule (graph: ProgramSemanticGraph) (entryPoint: PSGNode) : Result<string, string>
```

### Task 2.3: Update Project File
**File**: `/home/hhh/repos/Firefly/src/Firefly.fsproj`

Add new files in correct order:
```xml
<Compile Include="Alex/Bindings/Registry.fs" />
<Compile Include="Alex/Emit/ExprEmitter.fs" />
```

### Task 2.4: Implement Constant Emission
**Scope**: Handle `Const:Int32`, `Const:String`, `Const:Bool`, `Const:Unit`

Required MLIR patterns:
- Integer: `arith.constant <value> : i32`
- String: Global string constant + pointer
- Bool: `arith.constant 0/1 : i1`
- Unit: No emission needed (void)

### Task 2.5: Implement Identifier Emission
**Scope**: Handle `Ident` and `LongIdent` nodes

Strategy:
1. Look up node in `EmitContext.NodeSSA` map
2. If not found, follow `SymbolUse` edge to definition
3. Emit definition first if not yet emitted
4. Return SSA value from definition

### Validation Checkpoint: Sample 01
**Command**:
```bash
cd /home/hhh/repos/Firefly/samples/console/FidelityHelloWorld/01_HelloWorldDirect
../../../../src/bin/Debug/net9.0/Firefly compile HelloWorldDirect.fidproj -k --verbose
./target/hello_direct
```

**Expected**: Binary prints "Hello, World!" and exits with code 0

---

## Phase 3: Function Calls and I/O
**Estimated Scope**: Medium-Large (modifications to 3-4 files, ~200 lines)
**Checkpoint**: After completion
**Validation**: Sample 02 (HelloWorldSaturated) compiles

### Task 3.1: Implement Function Application Emission
**File**: `/home/hhh/repos/Firefly/src/Alex/Emit/ExprEmitter.fs`

Handle `App:FunctionCall` nodes:

```fsharp
let pApp : PSGChildParser<ExprResult> = parser {
    let! appNode = childKindPrefix "App"
    let! children = remainingChildren

    // First child is the function, rest are arguments
    match children with
    | [] -> return! fail "Empty application"
    | funcNode :: argNodes ->
        // Check if this is an extern primitive call
        match tryGetExternInfo funcNode with
        | Some externInfo ->
            // Dispatch to binding registry
            return! emitExternCall externInfo argNodes
        | None ->
            // Regular function call
            return! emitRegularCall funcNode argNodes
}
```

### Task 3.2: Integrate Extern Dispatch with XParsec
**File**: `/home/hhh/repos/Firefly/src/Alex/Emit/ExprEmitter.fs`

Bridge between XParsec parsing and ExternDispatch:

```fsharp
let emitExternCall (externInfo: ExternInfo) (args: PSGNode list) : PSGChildParser<ExprResult> = parser {
    // Emit each argument first
    let! argResults = emitArgs args

    // Build ExternPrimitive record
    let prim = {
        EntryPoint = externInfo.EntryPoint
        Library = externInfo.Library
        CallingConvention = externInfo.CallingConvention
        Args = argResults |> List.choose extractVal
        ReturnType = externInfo.ReturnType
    }

    // Dispatch to platform binding
    let! ctx = getEmitCtx
    match ExternDispatch.dispatch prim ctx.Platform with
    | Ok result -> return result
    | Error msg -> return EmitError msg
}
```

### Task 3.3: Extract DllImport Info from PSG Nodes
**File**: `/home/hhh/repos/Firefly/src/Alex/Emit/ExprEmitter.fs`

Helper to detect and extract extern declarations:

```fsharp
let tryGetExternInfo (node: PSGNode) : ExternInfo option =
    match node.Symbol with
    | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
        mfv.Attributes
        |> Seq.tryFind (fun attr ->
            attr.AttributeType.CompiledName = "DllImportAttribute")
        |> Option.map (fun attr ->
            let library = extractLibraryName attr
            let entryPoint = extractEntryPoint attr
            let callingConv = extractCallingConvention attr
            { Library = library; EntryPoint = entryPoint; ... })
    | _ -> None
```

### Task 3.4: Implement Let Binding Emission
**File**: `/home/hhh/repos/Firefly/src/Alex/Emit/ExprEmitter.fs`

Handle `LetOrUse` and `Binding` nodes:

```fsharp
let pLet : PSGChildParser<ExprResult> = parser {
    let! letNode = childKindPrefix "LetOrUse"
    // Navigate to binding and body
    // Emit binding value, record in NodeSSA
    // Emit body with binding in scope
    ...
}
```

### Task 3.5: Implement Sequential Expression Emission
**File**: `/home/hhh/repos/Firefly/src/Alex/Emit/ExprEmitter.fs`

Handle `Sequential` nodes:

```fsharp
let pSequential : PSGChildParser<ExprResult> = parser {
    let! seqNode = childSyntaxKind "Sequential"
    let! children = remainingChildren

    // Emit each child in order
    let! results = pMany pExpr

    // Return last result (or Void if empty)
    return results |> List.tryLast |> Option.defaultValue Void
}
```

### Task 3.6: Implement String Interpolation Support
**File**: `/home/hhh/repos/Firefly/src/Alex/Emit/ExprEmitter.fs`

Handle lowered interpolated strings (from `LowerInterpolatedStrings` nanopass):

```fsharp
let pInterpolatedString : PSGChildParser<ExprResult> = parser {
    let! node = childKindPrefix "App:LoweredInterpolatedString"
    // The nanopass has already lowered this to concat calls
    // Just emit as regular function application
    return! pApp
}
```

### Validation Checkpoint: Sample 02
**Command**:
```bash
cd /home/hhh/repos/Firefly/samples/console/FidelityHelloWorld/02_HelloWorldSaturated
../../../../src/bin/Debug/net9.0/Firefly compile HelloWorldSaturated.fidproj -k --verbose
./target/hello_saturated
# Enter name when prompted
```

**Expected**:
- Binary prompts "Enter your name: "
- User enters name
- Binary prints "Hello, {name}!"

---

## Phase 4: Pipe Operators and Function Values
**Estimated Scope**: Small-Medium (~100-150 lines)
**Checkpoint**: After completion
**Validation**: Sample 03 (HelloWorldHalfCurried) compiles

### Task 4.1: Verify Pipe Operator Reduction
**File**: `/home/hhh/repos/Firefly/src/Core/PSG/Nanopass/ReducePipeOperators.fs`

Confirm the nanopass transforms `x |> f` to `f x` in the PSG.

**Check**: After nanopass, pipe expressions should appear as regular `App:FunctionCall` nodes.

### Task 4.2: Handle Function Value References
**File**: `/home/hhh/repos/Firefly/src/Alex/Emit/ExprEmitter.fs`

When a function is referenced as a value (not called):

```fsharp
let pFunctionValue : PSGChildParser<ExprResult> = parser {
    let! node = childKindPrefix "Ident"
    match node.Symbol with
    | Some (:? FSharpMemberOrFunctionOrValue as mfv) when mfv.IsFunction ->
        // Emit function pointer/reference
        let funcName = getMangledName mfv
        return Value ($"@{funcName}", "ptr")
    | _ ->
        return! pzero
}
```

### Task 4.3: Test Pipe Semantics End-to-End
Verify that:
1. `Console.ReadLine() |> greet` in source
2. Becomes `greet (Console.ReadLine())` in PSG (after ReducePipeOperators)
3. Emits correct MLIR: call ReadLine, pass result to greet

### Validation Checkpoint: Sample 03
**Command**:
```bash
cd /home/hhh/repos/Firefly/samples/console/FidelityHelloWorld/03_HelloWorldHalfCurried
../../../../src/bin/Debug/net9.0/Firefly compile HelloWorldHalfCurried.fidproj -k --verbose
./target/hello_halfcurried
# Enter name when prompted
```

**Expected**: Same behavior as Sample 02 (different code structure, same output)

---

## Phase 5: Platform Organization Improvements
**Estimated Scope**: Medium (~150-200 lines, reorganization)
**Checkpoint**: After completion

### Task 5.1: Create Syscall Database
**File**: `/home/hhh/repos/Firefly/src/Alex/Bindings/Syscalls/SyscallDatabase.fs` (NEW)

Centralize all syscall numbers:

```fsharp
module Alex.Bindings.Syscalls.SyscallDatabase

type SyscallEntry = {
    Name: string
    Linux_x86_64: int64 option
    Linux_ARM64: int64 option
    MacOS_x86_64: int64 option
    MacOS_ARM64: int64 option
    FreeBSD_x86_64: int64 option
}

let database : SyscallEntry list = [
    { Name = "read"; Linux_x86_64 = Some 0L; Linux_ARM64 = Some 63L;
      MacOS_x86_64 = Some 0x2000003L; MacOS_ARM64 = Some 0x2000003L; FreeBSD_x86_64 = Some 3L }
    { Name = "write"; Linux_x86_64 = Some 1L; Linux_ARM64 = Some 64L;
      MacOS_x86_64 = Some 0x2000004L; MacOS_ARM64 = Some 0x2000004L; FreeBSD_x86_64 = Some 4L }
    { Name = "clock_gettime"; Linux_x86_64 = Some 228L; Linux_ARM64 = Some 113L;
      MacOS_x86_64 = None; MacOS_ARM64 = None; FreeBSD_x86_64 = Some 232L }
    { Name = "nanosleep"; Linux_x86_64 = Some 35L; Linux_ARM64 = Some 101L;
      MacOS_x86_64 = None; MacOS_ARM64 = None; FreeBSD_x86_64 = Some 240L }
    // ... more syscalls
]

let lookup (os: OSFamily) (arch: Architecture) (name: string) : int64 option
```

### Task 5.2: Refactor ConsoleBindings to Use Database
**File**: `/home/hhh/repos/Firefly/src/Alex/Bindings/Console/ConsoleBindings.fs`

Replace inline syscall maps with database lookup:

```fsharp
// BEFORE:
let linuxSyscalls = Map ["write", 1L; "read", 0L]

// AFTER:
let getSyscall name os arch = SyscallDatabase.lookup os arch name
```

### Task 5.3: Refactor TimeBindings to Use Database
**File**: `/home/hhh/repos/Firefly/src/Alex/Bindings/Time/TimeBindings.fs`

Same pattern as ConsoleBindings.

### Task 5.4: Implement macOS Time Bindings
**File**: `/home/hhh/repos/Firefly/src/Alex/Bindings/Time/TimeBindings.fs`

macOS uses `gettimeofday` (library call) instead of `clock_gettime`:

```fsharp
let emitMacOSGetTime () : MLIR<Val> = mlir {
    // macOS: use gettimeofday or mach_absolute_time
    // Different calling pattern than Linux syscall
    ...
}
```

### Task 5.5: Update Project File for Syscall Database
**File**: `/home/hhh/repos/Firefly/src/Firefly.fsproj`

Add syscall database before binding modules:
```xml
<Compile Include="Alex/Bindings/Syscalls/SyscallDatabase.fs" />
```

---

## Phase 6: Integration and Orchestration
**Estimated Scope**: Medium (~150 lines, wiring)
**Checkpoint**: After completion

### Task 6.1: Update CompilationOrchestrator
**File**: `/home/hhh/repos/Firefly/src/Alex/Pipeline/CompilationOrchestrator.fs`

Replace placeholder MLIR generation with XParsec-based emission:

```fsharp
let generateMLIRViaAlex (psg: ProgramSemanticGraph) (projectName: string) (targetTriple: string) : MLIRGenerationResult =
    // 1. Register all bindings
    Registry.registerAll()

    // 2. Set target platform
    let platform = TargetPlatform.parseTriple targetTriple |> Option.defaultValue TargetPlatform.detectHost()
    ExternDispatch.setTargetPlatform platform

    // 3. Find entry point
    let entryPoint = findEntryPoint psg

    // 4. Create emission context
    let emitCtx = EmitContext.create platform psg

    // 5. Run XParsec-based emission
    match ExprEmitter.emitModule psg entryPoint emitCtx with
    | Ok mlir -> { Success = true; MLIR = mlir; Errors = [] }
    | Error msg -> { Success = false; MLIR = ""; Errors = [msg] }
```

### Task 6.2: Wire Zipper + XParsec + Bindings
**File**: `/home/hhh/repos/Firefly/src/Alex/Emit/ExprEmitter.fs`

Ensure the traversal pattern:
1. Create PSGZipper from entry point
2. At each node, use XParsec to parse children
3. Dispatch to appropriate emitter based on node kind
4. Thread SSA state through emission
5. Collect MLIR output in EmitContext

### Task 6.3: Validate All Three Samples
**Commands**:
```bash
# Sample 01
cd /home/hhh/repos/Firefly/samples/console/FidelityHelloWorld/01_HelloWorldDirect
../../../../src/bin/Debug/net9.0/Firefly compile HelloWorldDirect.fidproj
./target/hello_direct  # Should print "Hello, World!"

# Sample 02
cd ../02_HelloWorldSaturated
../../../../src/bin/Debug/net9.0/Firefly compile HelloWorldSaturated.fidproj
./target/hello_saturated  # Should prompt, read name, print greeting

# Sample 03
cd ../03_HelloWorldHalfCurried
../../../../src/bin/Debug/net9.0/Firefly compile HelloWorldHalfCurried.fidproj
./target/hello_halfcurried  # Same behavior as 02
```

---

## Checkpoint Schedule

| Phase | Checkpoint After | Key Validation |
|-------|------------------|----------------|
| 0 | Alloy comment fix | None (documentation) |
| 1 | Combinator wrappers | Unit tests for combinators |
| 2 | Expression foundation | **Sample 01 compiles and runs** |
| 3 | Function calls + I/O | **Sample 02 compiles and runs** |
| 4 | Pipe operators | **Sample 03 compiles and runs** |
| 5 | Platform organization | Refactoring complete, samples still work |
| 6 | Full integration | All three samples work end-to-end |

---

## Risk Mitigation

### Risk: XParsec Combinator Mismatch
**Mitigation**: Verify against `~/repos/XParsec` source before implementing wrappers

### Risk: Extern Dispatch Doesn't Connect
**Mitigation**: Add diagnostic logging to ExternDispatch, verify registration happens

### Risk: SSA State Threading Breaks
**Mitigation**: Test incrementally, validate SSA counters after each phase

### Risk: Platform Differences Surface Late
**Mitigation**: Test on both Linux and macOS during Phase 5

---

## Success Criteria

1. **Alloy remains platform-agnostic** - No platform code in `~/repos/Alloy`
2. **MLIR generation is compositional** - Built from XParsec combinators, not ad-hoc pattern matching
3. **Samples 01-03 compile and run correctly** - End-to-end validation
4. **Platform bindings are data-driven** - Syscall numbers in database, not scattered
5. **Code is maintainable** - Adding new expressions = new parser + binding, no surgery

---

## File Summary

### New Files
| File | Phase | Purpose |
|------|-------|---------|
| `Alex/Bindings/Registry.fs` | 2 | Central binding registration |
| `Alex/Emit/ExprEmitter.fs` | 2 | XParsec-based expression emission |
| `Alex/Bindings/Syscalls/SyscallDatabase.fs` | 5 | Centralized syscall numbers |

### Modified Files
| File | Phase | Changes |
|------|-------|---------|
| `~/repos/Alloy/src/Primitives.fs` | 0 | Fix misleading comment |
| `Alex/Traversal/PSGXParsec.fs` | 1 | Add combinator wrappers |
| `Alex/Bindings/Console/ConsoleBindings.fs` | 5 | Use syscall database |
| `Alex/Bindings/Time/TimeBindings.fs` | 5 | Use syscall database, implement macOS |
| `Alex/Pipeline/CompilationOrchestrator.fs` | 6 | Wire XParsec emission |
| `Firefly.fsproj` | 2, 5 | Add new files |

---

## Next Steps

**Immediate**: Review this plan and approve Phase 0 + Phase 1 to begin implementation.

**Decision Points**:
- Phase 2: Confirm expression parser structure before implementing
- Phase 3: Validate extern dispatch integration approach
- Phase 5: Review syscall database schema before implementation
