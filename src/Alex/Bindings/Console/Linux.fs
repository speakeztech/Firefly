module Alex.Bindings.Console.Linux

open Core.PSG.Types
open Alex.CodeGeneration.EmissionContext
open Alex.Bindings.BindingTypes

// ===================================================================
// Linux Syscall Numbers (x86-64)
// ===================================================================

module Syscalls =
    let read = 0L
    let write = 1L

// ===================================================================
// MLIR Emission Helpers
// ===================================================================

/// Emit write syscall using inline assembly
/// write(fd, buf, count) -> returns bytes written
let emitWriteSyscall (builder: MLIRBuilder) (ctx: SSAContext)
                     (fd: string) (buf: string) (count: string) : string =
    // Extend fd from i32 to i64 for syscall
    let fdExt = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.extsi %s : i32 to i64" fdExt fd)

    // Syscall number for write
    let syscallNum = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant %d : i64" syscallNum Syscalls.write)

    // Execute syscall: write(fd, buf, count)
    // rax = syscall number, rdi = fd, rsi = buf, rdx = count
    let result = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64"
        result syscallNum fdExt buf count)

    result

/// Emit read syscall using inline assembly
/// read(fd, buf, count) -> returns bytes read
let emitReadSyscall (builder: MLIRBuilder) (ctx: SSAContext)
                    (fd: string) (buf: string) (count: string) : string =
    // Extend fd from i32 to i64 for syscall
    let fdExt = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.extsi %s : i32 to i64" fdExt fd)

    // Syscall number for read
    let syscallNum = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant %d : i64" syscallNum Syscalls.read)

    // Execute syscall: read(fd, buf, count)
    let result = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64"
        result syscallNum fdExt buf count)

    result

// ===================================================================
// Console Operation Emission
// ===================================================================

/// Emit Console.Write - write string to stdout (no newline)
let emitConsoleWrite (builder: MLIRBuilder) (ctx: SSAContext)
                     (ptr: string) (len: string) : EmissionResult =
    // fd = 1 (stdout)
    let fd = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i32" fd)

    let _ = emitWriteSyscall builder ctx fd ptr len
    EmittedVoid

/// Emit Console.WriteLine - write string to stdout followed by newline
let emitConsoleWriteLine (builder: MLIRBuilder) (ctx: SSAContext)
                         (ptr: string) (len: string) : EmissionResult =
    // fd = 1 (stdout)
    let fd = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i32" fd)

    // Write the string
    let _ = emitWriteSyscall builder ctx fd ptr len

    // Write newline character
    let nlByte = SSAContext.nextValue ctx
    let allocSize = SSAContext.nextValue ctx
    let nlPtr = SSAContext.nextValue ctx
    let one = SSAContext.nextValue ctx

    MLIRBuilder.line builder (sprintf "%s = arith.constant 10 : i8" nlByte)
    MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i64" allocSize)
    MLIRBuilder.line builder (sprintf "%s = llvm.alloca %s x i8 : (i64) -> !llvm.ptr" nlPtr allocSize)
    MLIRBuilder.line builder (sprintf "llvm.store %s, %s : i8, !llvm.ptr" nlByte nlPtr)
    MLIRBuilder.line builder (sprintf "%s = arith.constant 1 : i64" one)

    let _ = emitWriteSyscall builder ctx fd nlPtr one
    EmittedVoid

/// Emit Console.readInto - read from stdin into buffer
let emitConsoleReadInto (builder: MLIRBuilder) (ctx: SSAContext)
                        (ptr: string) (cap: string) : EmissionResult =
    // fd = 0 (stdin)
    let fd = SSAContext.nextValue ctx
    MLIRBuilder.line builder (sprintf "%s = arith.constant 0 : i32" fd)

    let result = emitReadSyscall builder ctx fd ptr cap
    Emitted result

/// Emit Console.writeBytes - write raw bytes to fd
let emitConsoleWriteBytes (builder: MLIRBuilder) (ctx: SSAContext)
                          (fd: string) (ptr: string) (count: string) : EmissionResult =
    let result = emitWriteSyscall builder ctx fd ptr count
    Emitted result

/// Emit Console.readBytes - read raw bytes from fd
let emitConsoleReadBytes (builder: MLIRBuilder) (ctx: SSAContext)
                         (fd: string) (ptr: string) (count: string) : EmissionResult =
    let result = emitReadSyscall builder ctx fd ptr count
    Emitted result

// ===================================================================
// Console Operation Pattern Matching
// ===================================================================

/// Check if a node matches a console operation pattern
let matchesConsolePattern (node: PSGNode) : string option =
    match node.Symbol with
    | Some sym ->
        let fullName = try sym.FullName with _ -> ""
        if fullName = "Alloy.Console.Write" || fullName.EndsWith(".Console.Write") then
            Some "Write"
        elif fullName = "Alloy.Console.WriteLine" || fullName.EndsWith(".Console.WriteLine") then
            Some "WriteLine"
        elif fullName = "Alloy.Console.writeBytes" || fullName.EndsWith(".Console.writeBytes") then
            Some "writeBytes"
        elif fullName = "Alloy.Console.readBytes" || fullName.EndsWith(".Console.readBytes") then
            Some "readBytes"
        elif fullName = "Alloy.Console.readInto" || fullName.EndsWith(".Console.readInto") then
            Some "readInto"
        elif fullName = "Alloy.Console.newLine" || fullName.EndsWith(".Console.newLine") then
            Some "newLine"
        elif fullName = "Alloy.Console.write" || fullName.EndsWith(".Console.write") then
            Some "write"
        elif fullName = "Alloy.Console.writeln" || fullName.EndsWith(".Console.writeln") then
            Some "writeln"
        else None
    | None -> None

// ===================================================================
// Console Binding Registration
// ===================================================================

/// Create the Linux console binding
let createBinding () : PlatformBinding =
    {
        Name = "Linux.Console"
        SymbolPatterns = [
            "Alloy.Console.Write"
            "Alloy.Console.WriteLine"
            "Alloy.Console.writeBytes"
            "Alloy.Console.readBytes"
            "Alloy.Console.readInto"
            "Alloy.Console.newLine"
            "Alloy.Console.write"
            "Alloy.Console.writeln"
        ]
        SupportedPlatforms = [
            TargetPlatform.linux_x86_64
            { TargetPlatform.linux_x86_64 with Arch = ARM64; Triple = "aarch64-unknown-linux-gnu" }
        ]
        Matches = fun node -> matchesConsolePattern node |> Option.isSome
        Emit = fun _platform builder ctx _psg node ->
            // The actual emission happens when ExpressionEmitter delegates to us
            // with the evaluated arguments. This Emit function is called from
            // BindingRegistry.tryEmit when we have a matching node.
            //
            // For now, we signal that we handle this pattern but defer actual
            // emission to when arguments are available (via ExpressionEmitter bridge).
            //
            // TODO: Once the zipper-based traversal is fully integrated, this
            // function should use XParsec to parse arguments from child nodes
            // and emit directly.
            match matchesConsolePattern node with
            | Some _ -> DeferToGeneric  // Handled by ExpressionEmitter bridge for now
            | None -> DeferToGeneric
        GetExternalDeclarations = fun _platform ->
            // Linux uses inline syscalls, no external declarations needed
            []
    }
