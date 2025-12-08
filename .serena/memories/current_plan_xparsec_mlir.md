# Current Active Plan: XParsec MLIR Integration

**Plan Document**: `/home/hhh/repos/Firefly/docs/PLAN_XParsec_MLIR_Integration.md`

## Goal
Achieve fully compositional MLIR generation via XParsec where:
- Alloy provides platform-agnostic native types and primitives
- XParsec combinators traverse PSG and pattern-match structures  
- Alex Bindings produce platform-specific MLIR

## Validation Samples
- `01_HelloWorldDirect` - Static strings, basic calls
- `02_HelloWorldSaturated` - Let bindings, string interpolation, I/O
- `03_HelloWorldHalfCurried` - Pipe operators, function values

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Cleanup (Alloy comment fix) | Not Started |
| 1 | XParsec Combinator Wrappers | Not Started |
| 2 | Expression Dispatcher Foundation | Not Started |
| 3 | Function Calls and I/O | Not Started |
| 4 | Pipe Operators and Function Values | Not Started |
| 5 | Platform Organization Improvements | Not Started |
| 6 | Integration and Orchestration | Not Started |

## Key Checkpoints
- After Phase 2: **Sample 01 must compile and run**
- After Phase 3: **Sample 02 must compile and run** (I/O verified)
- After Phase 4: **Sample 03 must compile and run** (pipes verified)

## Key New Files
- `Alex/Bindings/Registry.fs` - Central binding registration
- `Alex/Emit/ExprEmitter.fs` - XParsec-based expression emission
- `Alex/Bindings/Syscalls/SyscallDatabase.fs` - Centralized syscall numbers

## Reference Resources
- XParsec library: `~/repos/XParsec`
- Alloy library: `~/repos/Alloy`
- FCS: `~/repos/fsharp`
- Samples: `/home/hhh/repos/Firefly/samples/console/FidelityHelloWorld/`
