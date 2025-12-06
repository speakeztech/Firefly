# HelloWorld: Lessons Learned

This document captures the complete journey of getting a minimal interactive HelloWorld program working in Firefly, from F# source code to a native freestanding executable with blocking I/O.

## Overview

The goal was to compile this F# program to a native Linux executable without libc:

```fsharp
module HelloWorld

open FSharp.NativeInterop
open Alloy

[<EntryPoint>]
let main argv =
    Console.writeLine "What is your name?"

    let mutable buffer = NativePtr.stackalloc<byte> 64
    let bytesRead = Console.readLine buffer 64

    Console.write "Hello, "
    Console.writeLine "!"

    0
```

This required solving numerous challenges across the entire compilation pipeline.

---

## 1. MLIR Comment Syntax

### Problem
Initial MLIR output used semicolons for comments:

```mlir
; This is a comment  <- WRONG
```

### Solution
MLIR uses C++-style comments:

```mlir
// This is a comment  <- CORRECT
```

### Code Change
In `Emitter.fs`, all comment generation was updated:
```fsharp
// Before
MLIRBuilder.line builder (sprintf "; Comment here")

// After
MLIRBuilder.line builder (sprintf "// Comment here")
```

---

## 2. Syscall Side Effects

### Problem
Syscalls were being optimized away by LLVM. The write syscall would simply disappear from the final binary, producing no output.

### Root Cause
The `llvm.inline_asm` operation wasn't marked with side effects, so LLVM's optimizer determined the result was unused and eliminated the call.

### Solution
Add `has_side_effects` attribute to all inline assembly:

```mlir
// Before - gets optimized away
%result = llvm.inline_asm "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %rax, %rdi, %rsi, %rdx : (i64, i64, !llvm.ptr, i64) -> i64

// After - preserved by optimizer
%result = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %rax, %rdi, %rsi, %rdx : (i64, i64, !llvm.ptr, i64) -> i64
```

### Code Change
In `Emitter.fs`:
```fsharp
let emitSyscallWrite builder ctx fd buf len =
    let result = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"=r,{rax},{rdi},{rsi},{rdx}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64"
        result sysNum fdVal bufPtr len)
    result
```

---

## 3. Expression Execution Order

### Problem
Output appeared in reverse order. If the source had:
```fsharp
Console.writeLine "First"
Console.writeLine "Second"
```

The output was:
```
Second
First
```

### Root Cause
PSG children were stored in reverse order (prepended with `::` during construction), but not reversed when retrieved.

### Solution
Reverse children when extracting from PSG:

```fsharp
let getChildren (psg: ProgramSemanticGraph) (node: PSGNode) : PSGNode list =
    match node.Children with
    | Parent childIds ->
        childIds
        |> List.rev  // <- Critical fix: reverse to get source order
        |> List.choose (fun id -> Map.tryFind id.Value psg.Nodes)
    | _ -> []
```

---

## 4. OutputKind for Scalable Target Handling

### Problem
Hardcoding freestanding binary behavior at the end of compilation wasn't scalable. Different targets (console apps, embedded, libraries) need different entry point handling.

### Solution
Added `OutputKind` type to `MLIRTypes.fs`:

```fsharp
type OutputKind =
    | Console       // Standard console app - uses libc, main is entry point
    | Freestanding  // No libc - generates _start wrapper, exit syscall
    | Embedded      // Microcontroller target - no OS, custom startup
    | Library       // Shared/static library - no entry point

module OutputKind =
    let parse (s: string) =
        match s.ToLowerInvariant() with
        | "console" -> Console
        | "freestanding" | "bare" | "nostdlib" -> Freestanding
        | "embedded" | "firmware" | "mcu" -> Embedded
        | "library" | "lib" -> Library
        | _ -> Console
```

Configuration in `HelloWorld.fidproj`:
```toml
output_kind = "freestanding"
```

For freestanding binaries, the emitter generates a `_start` wrapper:
```mlir
func.func @_start() attributes {llvm.emit_c_interface} {
    %argc = arith.constant 0 : i32
    %retval = func.call @main(%argc) : (i32) -> i32
    %retval64 = arith.extsi %retval : i32 to i64
    %sys_exit = arith.constant 60 : i64
    %unused = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi}" %sys_exit, %retval64 : (i64, i64) -> i64
    return
}
```

---

## 5. Stack Allocation (stackalloc)

### Problem
`NativePtr.stackalloc<byte> 64` needed to emit proper LLVM stack allocation.

### Initial Wrong Approach
Tried to use literal values inline:
```mlir
%ptr = llvm.alloca i8 x 64 : (i64) -> !llvm.ptr  // WRONG - syntax error
```

### Correct Solution
`llvm.alloca` requires an SSA value for the count, not a literal:

```mlir
%size = llvm.mlir.constant(64 : i64) : i64
%ptr = llvm.alloca %size x i8 : (i64) -> !llvm.ptr  // CORRECT
```

### Code Change
In `Emitter.fs`:
```fsharp
// Emit stack allocation
// llvm.alloca requires SSA value for count, not inline literal
let sizeVal = SSAContext.nextValue ctx
MLIRBuilder.line builder (sprintf "%s = llvm.mlir.constant(%d : i64) : i64" sizeVal size)
let ptr = SSAContext.nextValue ctx
MLIRBuilder.line builder (sprintf "%s = llvm.alloca %s x %s : (i64) -> !llvm.ptr" ptr sizeVal llvmElemType)
```

### PSG Detection
The stackalloc is detected by examining the PSG structure:
- `TypeApp:byte` node contains the element type
- `Const:Int` node contains the size
- Symbol `Microsoft.FSharp.NativeInterop.NativePtr.stackalloc` identifies the operation

---

## 6. Variable Binding and Lookup

### Problem
`Console.readLine buffer 64` needed to find the `buffer` variable that was allocated by stackalloc.

### Solution
Added local variable tracking to SSAContext:

```fsharp
type SSAContext = {
    mutable Counter: int
    mutable Locals: Map<string, string>  // F# name -> SSA name mapping
}

module SSAContext =
    let registerLocal ctx fsharpName ssaName =
        ctx.Locals <- Map.add fsharpName ssaName ctx.Locals

    let lookupLocal ctx name =
        Map.tryFind name ctx.Locals
```

Let bindings register variables:
```fsharp
| sk when sk = "LetOrUse" || sk = "Binding" || sk.StartsWith("Binding:") ->
    // Find pattern node (variable name) and expression
    let patternNode = children |> List.tryFind (fun c -> c.SyntaxKind.StartsWith("Pattern:"))

    // Get variable name from pattern
    let varName = match patternNode with
        | Some pat -> match pat.Symbol with
            | Some sym -> Some sym.DisplayName
            | None -> None
        | None -> None

    // Emit the expression
    let result = emitExpression builder ctx psg exprNode

    // Register the SSA name
    match varName, result with
    | Some name, Some ssaName -> SSAContext.registerLocal ctx name ssaName
    | _ -> ()
```

---

## 7. Function Arguments vs Call Target

### Problem
When processing `Console.readLine buffer 64`, the code was finding `readLine` as the "buffer argument" instead of the actual `buffer` variable.

### Root Cause
The children of an `App` node include both the call target and the arguments. `List.tryFind` was returning the first `Ident` node, which was the function reference.

### Solution
Filter out the call target before looking for arguments:

```fsharp
| "readLine" ->
    // Filter out the call target from children to get just the arguments
    let args = children |> List.filter (fun c ->
        match c.Symbol with
        | Some sym ->
            let fn = sym.FullName
            not (fn.Contains("Console.readLine") || fn.Contains("Console"))
        | None ->
            not (c.SyntaxKind.Contains("readLine") || c.SyntaxKind.Contains("Console")))

    // Now find buffer argument among filtered args
    let bufferArg = args |> List.tryFind (fun c ->
        c.SyntaxKind.StartsWith("Ident") || c.SyntaxKind.StartsWith("LongIdent"))
```

---

## 8. Linker Optimization Stripping Code

### Problem
The final executable had all syscall code stripped out. `main` was just:
```asm
xor %eax,%eax
ret
```

### Root Cause
Using `clang -nostdlib -static` was invoking optimizations that eliminated the "unused" syscall results.

### Solution
Use `ld` directly with `--no-gc-sections`:

```bash
# Before - code gets stripped
clang -nostdlib -static -o hello HelloWorld.o -e _mlir_ciface__start

# After - code preserved
ld -static --no-gc-sections -e _mlir_ciface__start -o hello HelloWorld.o
```

---

## 9. Complete Build Pipeline

The final working pipeline:

```bash
# 1. Firefly: F# -> MLIR
Firefly compile --emit-mlir

# 2. mlir-opt: Lower to LLVM dialect
mlir-opt --convert-to-llvm HelloWorld.mlir -o HelloWorld.llvm.mlir

# 3. mlir-translate: LLVM MLIR -> LLVM IR
mlir-translate --mlir-to-llvmir HelloWorld.llvm.mlir -o HelloWorld.ll

# 4. llc: LLVM IR -> Object file (with PIC for linking)
llc -filetype=obj -relocation-model=pic -o HelloWorld.o HelloWorld.ll

# 5. ld: Link to executable (preserving all code)
ld -static --no-gc-sections -e _mlir_ciface__start -o hello HelloWorld.o
```

---

## 10. Final Working MLIR

The complete generated MLIR for the HelloWorld program:

```mlir
// Firefly-generated MLIR for HelloWorld
// Target: x86_64-unknown-linux-gnu
// Output: freestanding

module {
  func.func @main(%arg0: i32) -> i32 {
    // Console.writeLine "What is your name?"
    %v0 = llvm.mlir.addressof @str0 : !llvm.ptr
    %v1 = arith.constant 19 : i64
    %v2 = arith.constant 1 : i64
    %v3 = arith.constant 1 : i64
    %v4 = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %v3, %v2, %v0, %v1 : (i64, i64, !llvm.ptr, i64) -> i64

    // let buffer = NativePtr.stackalloc<byte> 64
    %v5 = llvm.mlir.constant(64 : i64) : i64
    %v6 = llvm.alloca %v5 x i8 : (i64) -> !llvm.ptr

    // Console.readLine buffer 64 (blocking read from stdin)
    %v7 = arith.constant 64 : i64
    %v8 = arith.constant 0 : i64   // sys_read
    %v9 = arith.constant 0 : i64   // stdin
    %v10 = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %v9, %v8, %v6, %v7 : (i64, i64, !llvm.ptr, i64) -> i64
    %v11 = arith.trunci %v10 : i64 to i32

    // Console.write "Hello, "
    %v12 = llvm.mlir.addressof @str1 : !llvm.ptr
    %v13 = arith.constant 7 : i64
    %v14 = arith.constant 1 : i64
    %v15 = arith.constant 1 : i64
    %v16 = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %v15, %v14, %v12, %v13 : (i64, i64, !llvm.ptr, i64) -> i64

    // Console.writeLine "!"
    %v17 = llvm.mlir.addressof @str2 : !llvm.ptr
    %v18 = arith.constant 2 : i64
    %v19 = arith.constant 1 : i64
    %v20 = arith.constant 1 : i64
    %v21 = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" %v20, %v19, %v17, %v18 : (i64, i64, !llvm.ptr, i64) -> i64

    %v22 = arith.constant 0 : i32
    return %v22 : i32
  }

  // Entry point wrapper for freestanding binary
  func.func @_start() attributes {llvm.emit_c_interface} {
    %argc = arith.constant 0 : i32
    %retval = func.call @main(%argc) : (i32) -> i32
    %retval64 = arith.extsi %retval : i32 to i64
    %sys_exit = arith.constant 60 : i64
    %unused = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi}" %sys_exit, %retval64 : (i64, i64) -> i64
    return
  }

  // String constants
  llvm.mlir.global private constant @str0("What is your name?\0A\00") : !llvm.array<20 x i8>
  llvm.mlir.global private constant @str1("Hello, \00") : !llvm.array<8 x i8>
  llvm.mlir.global private constant @str2("!\0A\00") : !llvm.array<3 x i8>
}
```

---

## 11. Key Takeaways

1. **MLIR Syntax Matters**: Small syntax differences (comments, SSA values vs literals) cause hard-to-debug failures.

2. **Side Effects Must Be Explicit**: LLVM aggressively optimizes. If a syscall result is "unused", it will be eliminated unless marked with `has_side_effects`.

3. **PSG Order**: Data structures that prepend elements need reversal for source-order traversal.

4. **Linker Behavior**: Different linkers and flags have dramatically different optimization behaviors. `ld` with `--no-gc-sections` is safer for freestanding code than `clang -nostdlib`.

5. **Use PSG Semantics**: Don't hardcode buffer allocations or sizes. Extract them from the PSG structure where they're already captured with full type information.

6. **Separate Call Targets from Arguments**: In function application nodes, the first matching child might be the function itself, not an argument.

7. **Scalable Design**: Add configuration options (like `output_kind`) early rather than hardcoding behavior that will need to change later.

---

## 12. Files Modified

- `/src/Core/MLIR/Emitter.fs` - Main MLIR emission logic
- `/src/Core/Types/MLIRTypes.fs` - Added OutputKind type
- `/src/CLI/Configurations/FidprojLoader.fs` - Parse output_kind from fidproj
- `/samples/console/HelloWorld/HelloWorld.fidproj` - Added output_kind configuration

---

## 13. FidelityHelloWorld Patterns Integration

### Background

A new repository `/home/hhh/repos/FidelityHelloWorld` was created containing five canonical patterns for console I/O, each demonstrating different approaches to function application:

1. **00_HelloWorldBcl.fs** - BCL reference (System.Console)
2. **01_HelloWorldDirect.fs** - Direct Alloy pattern
3. **02_HelloWorldSaturated.fs** - Saturated calls (all args at once)
4. **03_HelloWorldHalfCurried.fs** - Mixed pipeline
5. **04_HelloWorldFullCurried.fs** - Full currying

### Saturated Pattern (Simplest for Compiler)

The saturated pattern provides all function arguments at once, avoiding currying complexity:

```fsharp
module HelloWorld
open Alloy
open Alloy.Console
open Alloy.Memory

[<EntryPoint>]
let main argv =
    use buffer = stackBuffer<byte> 64
    Prompt "What is your name? "
    let name =
        match readInto buffer with
        | Ok length -> "User"
        | Error _ -> "Unknown"
    Write "Hello, "
    Write name
    WriteLine "!"
    0
```

### Key API Differences

FidelityHelloWorld introduces a cleaner API:

| Original Alloy | FidelityHelloWorld | Notes |
|---------------|-------------------|-------|
| `Console.writeLine` | `WriteLine` | Module-level function |
| `Console.write` | `Write` | Module-level function |
| `Console.readLine buffer size` | `readInto buffer` | SRTP-based, infers size |
| `NativePtr.stackalloc<byte>` | `stackBuffer<byte>` | Returns struct with Pointer/Length |

### StackBuffer Type

```fsharp
[<Struct>]
type StackBuffer<'T when 'T : unmanaged> =
    val Pointer: nativeptr<'T>
    val Length: int
    member this.Dispose() = ()
```

The `readInto` function uses SRTP to work with any buffer type that has `Pointer` and `Length` members:

```fsharp
let inline readInto (buffer: ^T) : Result<int, int> =
    let ptr = (^T : (member Pointer: nativeptr<byte>) buffer)
    let len = (^T : (member Length: int) buffer)
    // ... syscall implementation
```

---

## 14. PSG Paren Node Handling

### Problem

Expressions like `Console.writeBuffer buffer (bytesRead - 1)` weren't being traversed correctly. The parenthesized expression `(bytesRead - 1)` was being skipped.

### Root Cause

The PSG builder didn't have a handler for `SynExpr.Paren`, so parenthesized expressions were not being captured as child nodes.

### Solution

Added `SynExpr.Paren` handling to `/src/Core/PSG/Builder.fs`:

```fsharp
| SynExpr.Paren(inner, _, _, _) ->
    let parenId = NodeId.create()
    let children = walkExpr inner (Some parenId)
    { Id = parenId
      SyntaxKind = "Paren"
      Symbol = None
      Children = Parent children
      // ... }
```

---

## 15. Reachability Stack Overflow Fix

### Problem

The PSG reachability analysis was causing a stack overflow with infinite recursion when processing certain graph structures.

### Root Cause

The `markDescendants` function in `Reachability.fs` could revisit the same nodes indefinitely if there were cycles or self-references in the PSG.

### Solution

Added a visited set to prevent infinite recursion:

```fsharp
let visited = System.Collections.Generic.HashSet<string>()
let rec markDescendants nId funcName =
    if visited.Add(nId) then
        result <- Map.add nId funcName result
        match Map.tryFind nId psg.Nodes with
        | Some node ->
            match node.Children with
            | Parent childIds ->
                for child in childIds do
                    markDescendants child.Value funcName
            | _ -> ()
        | None -> ()
```

---

## 16. LLVM Optimization Breaking Syscalls

### Problem

When compiled with LLVM optimizations, the read syscall would return 0 instead of the actual bytes read. The `sideeffect` attribute wasn't sufficient.

### Symptoms

```bash
# With optimizations - read returns 0
echo "Claude" | ./hello
# Output: What is your name? Hello, !

# Without optimizations - read works correctly
echo "Claude" | ./hello_O0
# Output: What is your name? Hello, Claude!
```

### Analysis

The inline assembly was marked with `has_side_effects`:

```mlir
%result = llvm.inline_asm has_side_effects "syscall", "=r,{rax},{rdi},{rsi},{rdx}" ...
```

However, LLVM's optimizer was still reordering or eliminating the syscall result. The `~{memory}` clobber alone wasn't sufficient for the aggressive optimization passes.

### Workaround

Use `llc -O0` to disable optimizations:

```bash
# Working build pipeline
llc -O0 -filetype=obj HelloWorld_main.ll -o HelloWorld_main.o
ld -nostdlib -static HelloWorld_main.o -o hello_O0
```

### Future Investigation

May need additional attributes or clobbers:
- `~{rcx}` and `~{r11}` (syscall clobbers these on Linux x86_64)
- `volatile` memory semantics
- Custom LLVM intrinsic instead of inline assembly

---

## 17. Updated Build Pipeline

The complete working pipeline for freestanding executables:

```bash
# 1. Firefly: F# -> MLIR
/home/hhh/repos/Firefly/src/bin/Debug/net9.0/Firefly compile --emit-mlir

# 2. mlir-opt: Lower to LLVM dialect
mlir-opt --convert-arith-to-llvm --convert-func-to-llvm --convert-cf-to-llvm \
    --reconcile-unrealized-casts HelloWorld_main.mlir -o HelloWorld_main.llvm.mlir

# 3. mlir-translate: LLVM MLIR -> LLVM IR
mlir-translate --mlir-to-llvmir HelloWorld_main.llvm.mlir -o HelloWorld_main.ll

# 4. llc: LLVM IR -> Object file (NO OPTIMIZATIONS for syscalls)
llc -O0 -filetype=obj HelloWorld_main.ll -o HelloWorld_main.o

# 5. ld: Link to executable
ld -nostdlib -static HelloWorld_main.o -o hello
```

**Critical**: The `-O0` flag on `llc` is required to prevent syscall optimization issues.

---

## 18. Emitter Updates for FidelityHelloWorld

Added support for FidelityHelloWorld Console operations in `/src/Core/MLIR/Emitter.fs`:

```fsharp
| "WriteLine" -> // FidelityHelloWorld pattern
    match strArg with
    | Some strNode ->
        match extractStringContent strNode.SyntaxKind with
        | Some content -> emitConsoleWriteLine builder ctx content
        | None -> None
    | None -> None

| "Write" | "Prompt" -> // FidelityHelloWorld patterns
    match strArg with
    | Some strNode ->
        match extractStringContent strNode.SyntaxKind with
        | Some content -> emitConsoleWrite builder ctx content
        | None -> None
    | None -> None

| "readInto" ->
    // Emit read syscall for stdin
    // Uses SRTP buffer with Pointer/Length members
    emitSyscallRead builder ctx bufferPtrSSA maxLenSSA
```

Also updated `isStackAlloc` to recognize `stackBuffer`:

```fsharp
let isStackAlloc (sym: FSharpSymbol option) =
    match sym with
    | Some s ->
        let fullName = s.FullName
        fullName.Contains("NativePtr.stackalloc") ||
        fullName.Contains("stackalloc") ||
        fullName.Contains("stackBuffer") ||  // FidelityHelloWorld pattern
        s.DisplayName = "stackBuffer"
    | None -> false
```

---

## 19. Future Work

1. **Support all FidelityHelloWorld patterns**: Currently only Saturated works. Need to implement:
   - HalfCurried (partial application with pipeline)
   - FullCurried (full currying)

2. **Fix LLVM optimization issue properly**: Investigate proper syscall clobbers or intrinsics to avoid needing `-O0`.

3. **Write buffer content to greeting**: Currently reads user input but greeting hardcodes "User" - need to emit the actual buffer content.

4. **Integrated build**: The 5-step build pipeline should be automated within Firefly.

5. **String handling**: Need proper string type with length tracking, not just null-terminated C strings.

6. **Error handling**: Read syscall can fail; need to handle error codes from Result type.

---

## 20. Key Takeaways (Updated)

1. **Pattern-based API design**: FidelityHelloWorld patterns show a progression from simplest (Saturated) to most complex (FullCurried). Start with Saturated for initial compiler support.

2. **SRTP for polymorphism**: Using Statically Resolved Type Parameters allows buffer abstraction without runtime overhead.

3. **Optimization flags matter**: `-O0` is currently required for syscall correctness. This is a workaround, not a solution.

4. **Visited tracking is essential**: Any recursive graph traversal needs cycle detection to avoid stack overflow.

5. **Paren nodes are structural**: Even "trivial" syntax like parentheses needs explicit handling in the PSG.
