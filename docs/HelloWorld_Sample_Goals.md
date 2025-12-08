# HelloWorld Sample Goals

This document outlines the FidelityHelloWorld samples, their purpose, and the compiler capabilities required to make each work.

## Sample Progression

The samples are ordered by increasing complexity in F# language features:

| Sample | Pattern | Key Features |
|--------|---------|--------------|
| 01_HelloWorldDirect | Direct module calls | `Console.Write`, `Console.readInto`, match expressions |
| 02_HelloWorldSaturated | Saturated function calls | `Prompt`, `readInto`, `WriteLine` - all args provided at once |
| 03_HelloWorldHalfCurried | Pipe operator | `\|>` operator, `String.format`, partial application |
| 04_HelloWorldFullCurried | Full currying | `Result.map`, `Result.defaultValue`, lambdas, HOFs |

## Expected Behavior (All Samples)

Every HelloWorld sample should:
1. Print a prompt: `"Enter your name: "`
2. **Wait for user input** (blocking read from stdin)
3. Read the input into a buffer
4. Handle the result (Ok/Error pattern)
5. Print greeting: `"Hello, {name}!"` or `"Hello, Unknown Person!"`

## Alloy Library Dependencies

### Critical Functions for HelloWorld

| Function | Module | Purpose |
|----------|--------|---------|
| `stackBuffer<byte>` | `Alloy.Memory` | Stack-allocated buffer for input |
| `readInto` | `Alloy.Console` | SRTP-based read into buffer |
| `Prompt` / `Write` | `Alloy.Console` | Output string to stdout |
| `WriteLine` | `Alloy.Console` | Output string + newline |
| `spanToString` | `Alloy.Text.UTF8` | Convert buffer to string |
| `AsReadOnlySpan` | `Alloy.Memory` | Get span view of buffer |

### Low-Level Primitives (Compiler Must Recognize)

| Function | Syscall | Description |
|----------|---------|-------------|
| `writeBytes` | `write(1, buf, len)` | Write to stdout |
| `readBytes` | `read(0, buf, len)` | Read from stdin |
| `stringToBytes` | N/A | Convert F# string to byte buffer |
| `bytesToString` | N/A | Convert byte buffer to F# string |

## Alloy Library Status

### Critical Finding: Library Mismatch

The formal `~/repos/Alloy` library has **placeholder implementations** while `FidelityHelloWorld/lib/Alloy` has **working implementations**.

| File | Formal Alloy | FidelityHelloWorld | Status |
|------|--------------|-------------------|--------|
| Console.fs | 147 lines (placeholders) | 297 lines (implementations) | **MISMATCH** |
| Text.fs | 75 lines | 522 lines | **MISMATCH** |
| Memory.fs | 257 lines | 357 lines | **MISMATCH** |
| Core.fs | 260 lines | 491 lines | **MISMATCH** |

### Key Missing Implementations in Formal Alloy

1. **Console.Write** - Returns `()` instead of actual write
2. **Console.ReadLine** - Missing entirely
3. **Console.sprintf** - Missing
4. **Text.UTF8.stringToBytes** - Unclear if present
5. **Text.UTF8.bytesToString** - Unclear if present

## Compiler Pipeline Requirements

For HelloWorld to work, the following must all be correct:

### 1. FCS Ingestion
- Alloy library files must be included in compilation
- Symbol resolution must find Alloy functions
- SRTP constraints (like `readInto`) must be resolved

### 2. PSG Construction
- `stackBuffer` allocation must be captured
- `readInto` call and its Result type must be captured
- Match expression (Ok/Error) must be captured
- Variable bindings (`buffer`, `name`) must be tracked
- Function calls with variable arguments must be captured

### 3. Reachability Analysis
- Entry point `main` must be found
- `hello()` function must be marked reachable
- All Alloy functions called must be marked reachable
- The entire call chain must be preserved

### 4. MLIR Generation
- `stackBuffer` → `llvm.alloca`
- `readInto` → `read` syscall with buffer
- Match expression → conditional branches
- `Write`/`WriteLine` → `write` syscall
- Variable references → SSA values from scope

## Testing Criteria

A HelloWorld sample is **working** when:

```bash
$ ./hello
Enter your name: Claude
Hello, Claude!
```

NOT working (current state):
```bash
$ ./hello
What is your name? Hello, !
```

The second output shows the program:
- Prints all strings immediately (no blocking read)
- Doesn't capture user input
- Doesn't use the input in output

## Next Steps

1. **Decide**: Use FidelityHelloWorld Alloy lib or fix formal Alloy lib
2. **Verify**: FCS correctly ingests chosen Alloy library
3. **Verify**: PSG captures complete program structure including reads
4. **Fix**: MLIR Generation to handle all required operations
5. **Test**: Interactive execution with actual user input
