/// ExpressionEmitter - Classifier-driven expression emission
///
/// Uses PSGPatterns predicates/extractors and AlloyPatterns classifiers
/// for pattern matching, with EmissionMonad's tryEmissions for dispatch.
///
/// ARCHITECTURE:
/// 1. Classifiers (PSGPatterns, AlloyPatterns) extract typed data from PSG nodes
/// 2. Each tryEmit* function is self-contained: classify then handle, returning Option
/// 3. tryEmissions dispatches to the first matching handler
/// 4. EmissionMonad threads state and builds MLIR output
module Alex.Emission.ExpressionEmitter

open Core.PSG.Types
open Alex.CodeGeneration.EmissionContext
open Alex.CodeGeneration.TypeMapping
open Alex.CodeGeneration.EmissionMonad
open Alex.Traversal.PSGZipper
open Alex.Patterns.PSGPatterns
open Alex.Patterns.AlloyPatterns

// ===================================================================
// Expression Emission Result
// ===================================================================

/// Result of emitting an expression
type ExprResult =
    | Value of ssa: string * mlirType: string
    | Void
    | Error of message: string

// ===================================================================
// Constant Extraction (from SyntaxKind metadata)
// ===================================================================

/// Extract string content from Const:String syntax kind
let private extractStringFromKind (kind: string) : string option =
    if kind.StartsWith("Const:String") then
        let start = kind.IndexOf("(\"")
        if start >= 0 then
            let contentStart = start + 2
            let endQuote = kind.IndexOf("\",", contentStart)
            if endQuote >= contentStart then  // >= to handle empty strings
                Some (kind.Substring(contentStart, endQuote - contentStart))
            else None
        else None
    else None

/// Extract int32 from Const:Int32 syntax kind
let private extractInt32FromKind (kind: string) : int option =
    if kind.StartsWith("Const:Int32 ") then
        let afterPrefix = kind.Substring("Const:Int32 ".Length)
        let valueStr =
            match afterPrefix.IndexOf(' ') with
            | -1 -> afterPrefix
            | idx -> afterPrefix.Substring(0, idx)
        match System.Int32.TryParse(valueStr) with
        | true, v -> Some v
        | false, _ -> None
    else None

/// Extract int64 from Const:Int64 syntax kind
let private extractInt64FromKind (kind: string) : int64 option =
    if kind.StartsWith("Const:Int64:") then
        let valueStr = kind.Substring("Const:Int64:".Length).TrimEnd('L')
        match System.Int64.TryParse(valueStr) with
        | true, v -> Some v
        | false, _ -> None
    elif kind.StartsWith("Const:Int64 ") then
        let valueStr = kind.Substring("Const:Int64 ".Length).TrimEnd('L')
        match System.Int64.TryParse(valueStr) with
        | true, v -> Some v
        | false, _ -> None
    else None

/// Extract byte from Const:Byte syntax kind (e.g., "Const:Byte 0uy" or "Const:Byte 10uy")
let private extractByteFromKind (kind: string) : byte option =
    if kind.StartsWith("Const:Byte ") then
        let valueStr = kind.Substring("Const:Byte ".Length).TrimEnd('u', 'y')
        match System.Byte.TryParse(valueStr) with
        | true, v -> Some v
        | false, _ -> None
    else None

/// Extract byte array from Const:Bytes syntax kind
/// Format: "Const:Bytes\n  ([|69uy; 110uy; ...|], Regular, ...)"
/// Returns the bytes and the length
let private extractBytesFromKind (kind: string) : (byte[] * int) option =
    if kind.StartsWith("Const:Bytes") then
        // Find the byte array literal: [|...|]
        let start = kind.IndexOf("[|")
        let endPos = kind.IndexOf("|]")
        if start >= 0 && endPos > start then
            let content = kind.Substring(start + 2, endPos - start - 2)
            // Parse individual bytes like "69uy; 110uy; ..."
            let bytes =
                content.Split([|';'|], System.StringSplitOptions.RemoveEmptyEntries)
                |> Array.choose (fun s ->
                    let trimmed = s.Trim().TrimEnd('u', 'y')
                    match System.Byte.TryParse(trimmed) with
                    | true, v -> Some v
                    | false, _ -> None)
            Some (bytes, bytes.Length)
        else
            None
    else
        None

// ===================================================================
// Forward declaration for mutual recursion
// ===================================================================

// emitExpr will be defined at the end after all handlers
// Handlers reference it via mutable cell for mutual recursion
let mutable private emitExprImpl : Emit<ExprResult> =
    fun _env state -> (state, Error "emitExpr not initialized")

/// Call emitExpr (used by handlers before emitExpr is defined)
let private callEmitExpr : Emit<ExprResult> =
    fun env state -> emitExprImpl env state

// ===================================================================
// Console Emission Handlers
// ===================================================================

/// Emit write syscall (Linux x86-64: syscall 1 = write)
let private emitWriteSyscall (fd: string) (buf: string) (len: string) : Emit<string> =
    emitI64 1L >>= fun sysnumReg ->
    freshSSAWithType "i64" >>= fun result ->
    line (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64"
        result sysnumReg fd buf len) >>.
    emit result

/// Emit Console.Write with string content
let private emitWriteString (content: string) : Emit<ExprResult> =
    registerStringLiteral content >>= fun globalName ->
    emitAddressOf globalName >>= fun ptr ->
    emitI64 (int64 content.Length) >>= fun len ->
    emitI64 1L >>= fun fd ->
    emitWriteSyscall fd ptr len >>= fun _ ->
    emit Void

/// Emit Console.WriteLine with string content (adds newline)
let private emitWriteLineString (content: string) : Emit<ExprResult> =
    emitWriteString (content + "\n")

/// Emit write syscall for a dynamic string (NativeStr struct: {ptr, len})
/// Extracts the pointer and length from the struct and calls write syscall
let private emitWriteDynamicString (strStruct: string) (strType: string) : Emit<ExprResult> =
    // Check if this is a NativeStr struct or a plain pointer
    if strType = "!llvm.struct<(ptr, i64)>" then
        // Extract pointer and length from the struct
        freshSSAWithType "!llvm.ptr" >>= fun ptr ->
        line (sprintf "%s = llvm.extractvalue %s[0] : !llvm.struct<(ptr, i64)>" ptr strStruct) >>.
        freshSSAWithType "i64" >>= fun len ->
        line (sprintf "%s = llvm.extractvalue %s[1] : !llvm.struct<(ptr, i64)>" len strStruct) >>.

        // Write syscall with the actual length
        emitI64 1L >>= fun sysWrite ->
        emitI64 1L >>= fun fdStdout ->
        freshSSAWithType "i64" >>= fun result ->
        line (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64"
            result sysWrite fdStdout ptr len) >>.
        emit Void
    else
        // Fallback for plain pointer (shouldn't happen with proper typing)
        emitI64 1L >>= fun sysWrite ->
        emitI64 1L >>= fun fdStdout ->
        emitI64 256L >>= fun maxLen ->  // Write up to 256 bytes
        freshSSAWithType "i64" >>= fun result ->
        line (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64"
            result sysWrite fdStdout strStruct maxLen) >>.
        emit Void

/// Handler for Console operations
let private handleConsole (op: ConsoleOp) : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z
    match op with
    | Write ->
        match children with
        | _ :: argNode :: _ ->
            match extractStringFromKind argNode.SyntaxKind with
            | Some content -> emitWriteString content
            | None ->
                atNode argNode callEmitExpr >>= fun argResult ->
                match argResult with
                | Value(ssa, mlirType) ->
                    emitWriteDynamicString ssa mlirType
                | _ -> emit Void
        | _ ->
            emit (Error "Console.Write: missing argument")
    | WriteLine ->
        match children with
        | _ :: argNode :: _ ->
            match extractStringFromKind argNode.SyntaxKind with
            | Some content -> emitWriteLineString content
            | None ->
                atNode argNode callEmitExpr >>= fun argResult ->
                match argResult with
                | Value(ssa, mlirType) ->
                    emitWriteDynamicString ssa mlirType >>= fun _ ->
                    emitWriteString "\n"
                | _ -> emit Void
        | _ ->
            emit (Error "Console.WriteLine: missing argument")
    | ReadLine ->
        // Allocate a 1024-byte buffer on the stack
        emitI64 1024L >>= fun bufSize ->
        emitAlloca "i8" bufSize >>= fun buffer ->

        // Read from stdin (fd 0) into buffer
        // syscall 0 = read(fd, buf, count)
        emitI64 0L >>= fun sysRead ->
        emitI64 0L >>= fun fdStdin ->
        emitI64 1023L >>= fun maxRead ->  // Leave room for null terminator
        freshSSAWithType "i64" >>= fun bytesRead ->
        line (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64"
            bytesRead sysRead fdStdin buffer maxRead) >>.

        // Subtract 1 from bytesRead to exclude the trailing newline
        // (read() includes the newline in the count)
        emitI64 1L >>= fun one ->
        freshSSAWithType "i64" >>= fun strLen ->
        line (sprintf "%s = arith.subi %s, %s : i64" strLen bytesRead one) >>.

        // Build a NativeStr struct: { ptr: !llvm.ptr, len: i64 }
        // This is Firefly's native string representation
        freshSSAWithType "!llvm.struct<(ptr, i64)>" >>= fun strStruct ->
        line (sprintf "%s = llvm.mlir.undef : !llvm.struct<(ptr, i64)>" strStruct) >>.
        freshSSAWithType "!llvm.struct<(ptr, i64)>" >>= fun strStruct1 ->
        line (sprintf "%s = llvm.insertvalue %s, %s[0] : !llvm.struct<(ptr, i64)>" strStruct1 buffer strStruct) >>.
        freshSSAWithType "!llvm.struct<(ptr, i64)>" >>= fun strStruct2 ->
        line (sprintf "%s = llvm.insertvalue %s, %s[1] : !llvm.struct<(ptr, i64)>" strStruct2 strLen strStruct1) >>.

        emit (Value(strStruct2, "!llvm.struct<(ptr, i64)>"))

    | Read ->
        // Similar to ReadLine but for readInto pattern
        line "// Console.Read - not yet implemented" >>.
        emit Void

/// Try to emit as Console operation
let private tryEmitConsole : Emit<ExprResult option> =
    getZipper >>= fun z ->
    match extractConsoleOp z.Graph z.Focus with
    | Some op -> handleConsole op |>> Some
    | None -> emit None

// ===================================================================
// Time Emission Handlers
// ===================================================================

open Alex.Bindings.Time.Linux

/// Handler for Time operations
let private handleTime (op: TimeOp) : Emit<ExprResult> =
    match op with
    | CurrentTicks ->
        invokeBinding (fun builder ctx ->
            let ticks = emitClockGettime builder ctx ClockId.REALTIME
            let epochOffset = SSAContext.nextValue ctx
            let result = SSAContext.nextValue ctx
            MLIRBuilder.line builder (sprintf "%s = arith.constant 621355968000000000 : i64" epochOffset)
            MLIRBuilder.line builder (sprintf "%s = arith.addi %s, %s : i64" result ticks epochOffset)
            Value(result, "i64")
        )
    | CurrentUnixTimestamp ->
        invokeBinding (fun builder ctx ->
            let ticks = emitClockGettime builder ctx ClockId.REALTIME
            let ticksPerSec = SSAContext.nextValue ctx
            let result = SSAContext.nextValue ctx
            MLIRBuilder.line builder (sprintf "%s = arith.constant 10000000 : i64" ticksPerSec)
            MLIRBuilder.line builder (sprintf "%s = arith.divui %s, %s : i64" result ticks ticksPerSec)
            Value(result, "i64")
        )
    | CurrentDateTimeString ->
        invokeBinding (fun builder ctx ->
            let buf = emitCurrentDateTimeString builder ctx
            Value(buf, "!llvm.ptr")
        )
    | Sleep ->
        getZipper >>= fun z ->
        let children = PSGZipper.childNodes z
        match children with
        | _ :: argNode :: _ ->
            match extractInt32FromKind argNode.SyntaxKind with
            | Some ms ->
                emitI32 ms >>= fun msReg ->
                invokeBinding (fun builder ctx ->
                    emitNanosleep builder ctx msReg
                    Void
                )
            | None ->
                atNode argNode callEmitExpr >>= fun argResult ->
                match argResult with
                | Value(msReg, _) ->
                    invokeBinding (fun builder ctx ->
                        emitNanosleep builder ctx msReg
                        Void
                    )
                | _ -> emit (Error "Time.sleep: argument must produce value")
        | _ ->
            emit (Error "Time.sleep: missing argument")
    | _ ->
        line (sprintf "// Time.%A - not yet implemented" op) >>.
        emit Void

/// Try to emit as Time operation
let private tryEmitTime : Emit<ExprResult option> =
    getZipper >>= fun z ->
    match extractTimeOp z.Graph z.Focus with
    | Some op -> handleTime op |>> Some
    | None -> emit None

// ===================================================================
// Memory Emission Handlers
// ===================================================================

/// Handler for Memory operations
let private handleMemory (op: MemoryOp) : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z
    match op with
    | StackBuffer _ ->
        // stackBuffer<T>(size) - allocate stack memory
        // Children: TypeApp:byte [stackBuffer], Const:Int32 256
        match children with
        | _ :: sizeNode :: _ ->
            // Get size from constant
            match extractInt32FromKind sizeNode.SyntaxKind with
            | Some size ->
                // Emit stack allocation: %ptr = llvm.alloca %size x i8 : (i64) -> !llvm.ptr
                emitI64 (int64 size) >>= fun sizeReg ->
                emitAlloca "i8" sizeReg >>= fun ptr ->
                emit (Value(ptr, "!llvm.ptr"))
            | None ->
                // Dynamic size - evaluate the size expression
                atNode sizeNode callEmitExpr >>= fun sizeResult ->
                match sizeResult with
                | Value(sizeReg, "i32") ->
                    // Convert i32 to i64 for alloca
                    emitExtsi sizeReg "i32" "i64" >>= fun sizeReg64 ->
                    emitAlloca "i8" sizeReg64 >>= fun ptr ->
                    emit (Value(ptr, "!llvm.ptr"))
                | Value(sizeReg, "i64") ->
                    emitAlloca "i8" sizeReg >>= fun ptr ->
                    emit (Value(ptr, "!llvm.ptr"))
                | _ -> emit (Error "stackBuffer size must be an integer")
        | _ -> emit (Error "stackBuffer requires size argument")

    | SpanToString ->
        // spanToString(span) - convert bytes to string (for now, just return pointer)
        // Debug: log children count and kinds
        let childInfo = children |> List.mapi (fun i c -> sprintf "child[%d]=%s" i c.SyntaxKind) |> String.concat ", "
        line (sprintf "// spanToString: %d children [%s]" (List.length children) childInfo) >>.
        match children with
        | _ :: spanNode :: _ ->
            atNode spanNode callEmitExpr >>= fun spanResult ->
            match spanResult with
            | Value(ptr, _) -> emit (Value(ptr, "!llvm.ptr"))
            | _ -> emit (Error "spanToString requires span argument")
        | [singleChild] ->
            // Single child - might be the span argument directly
            atNode singleChild callEmitExpr >>= fun spanResult ->
            match spanResult with
            | Value(ptr, _) -> emit (Value(ptr, "!llvm.ptr"))
            | _ -> emit (Error "spanToString single child is not a value")
        | [] -> emit (Error "spanToString has no children")

    | AsReadOnlySpan ->
        // buffer.AsReadOnlySpan(start, length) - get span from buffer
        // For method calls like buffer.AsReadOnlySpan(...), the receiver (buffer) is in the
        // App node's symbol, not in the children. The children are [MethodName, Arguments].
        // We need to look up the buffer variable from the current scope.
        getFocus >>= fun node ->
        match node.Symbol with
        | Some sym ->
            // The symbol is the receiver (buffer)
            let bufferName = sym.DisplayName
            lookupLocal bufferName >>= fun localOpt ->
            match localOpt with
            | Some bufferSsa ->
                // Return the buffer pointer as the span (simplified - just pass through)
                emit (Value(bufferSsa, "!llvm.ptr"))
            | None ->
                emit (Error (sprintf "AsReadOnlySpan: buffer '%s' not found in locals" bufferName))
        | None ->
            emit (Error "AsReadOnlySpan: no symbol for receiver")

/// Try to emit as Memory operation
let private tryEmitMemory : Emit<ExprResult option> =
    getZipper >>= fun z ->
    match extractMemoryOp z.Graph z.Focus with
    | Some op -> handleMemory op |>> Some
    | None -> emit None

// ===================================================================
// Result DU Constructor Handlers
// ===================================================================

/// Handler for Result DU constructor operations (Ok, Error)
/// Constructs the Result struct inline: !llvm.struct<(i32, i64, i64)>
/// - Field 0: Tag (0 for Ok, 1 for Error)
/// - Field 1: Payload slot 1 (value for Ok, error for Error)
/// - Field 2: Payload slot 2 (unused, for alignment)
let private handleResult (op: ResultOp) : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z

    // Get the value argument (second child after the constructor ident)
    match children with
    | _ :: valueNode :: _ ->
        atNode valueNode callEmitExpr >>= fun valueResult ->
        match valueResult with
        | Value(valueSsa, valueType) ->
            // Determine the tag based on constructor
            let tag = match op with OkCtor -> 0 | ErrorCtor -> 1

            // Create the Result struct
            // First, create an undef struct
            freshSSAWithType "!llvm.struct<(i32, i64, i64)>" >>= fun structUndef ->
            line (sprintf "%s = llvm.mlir.undef : !llvm.struct<(i32, i64, i64)>" structUndef) >>.

            // Insert tag (field 0)
            emitI32 tag >>= fun tagReg ->
            freshSSAWithType "!llvm.struct<(i32, i64, i64)>" >>= fun struct1 ->
            line (sprintf "%s = llvm.insertvalue %s, %s[0] : !llvm.struct<(i32, i64, i64)>" struct1 tagReg structUndef) >>.

            // Insert payload (field 1) - extend to i64 if needed
            (if valueType = "i64" || valueType = "!llvm.ptr" then
                // Already i64 or pointer (which is i64-sized)
                if valueType = "!llvm.ptr" then
                    // Cast pointer to i64
                    freshSSAWithType "i64" >>= fun ptrAsInt ->
                    line (sprintf "%s = llvm.ptrtoint %s : !llvm.ptr to i64" ptrAsInt valueSsa) >>.
                    emit ptrAsInt
                else
                    emit valueSsa
            elif valueType = "i32" then
                emitExtsi valueSsa "i32" "i64"
            else
                // For other types, try sign extension
                emitExtsi valueSsa valueType "i64") >>= fun payload64 ->

            freshSSAWithType "!llvm.struct<(i32, i64, i64)>" >>= fun struct2 ->
            line (sprintf "%s = llvm.insertvalue %s, %s[1] : !llvm.struct<(i32, i64, i64)>" struct2 payload64 struct1) >>.

            // Insert zero for field 2 (unused)
            emitI64 0L >>= fun zero64 ->
            freshSSAWithType "!llvm.struct<(i32, i64, i64)>" >>= fun structFinal ->
            line (sprintf "%s = llvm.insertvalue %s, %s[2] : !llvm.struct<(i32, i64, i64)>" structFinal zero64 struct2) >>.

            emit (Value(structFinal, "!llvm.struct<(i32, i64, i64)>"))
        | Void ->
            // Void payload - use 0
            let tag = match op with OkCtor -> 0 | ErrorCtor -> 1
            freshSSAWithType "!llvm.struct<(i32, i64, i64)>" >>= fun structUndef ->
            line (sprintf "%s = llvm.mlir.undef : !llvm.struct<(i32, i64, i64)>" structUndef) >>.
            emitI32 tag >>= fun tagReg ->
            freshSSAWithType "!llvm.struct<(i32, i64, i64)>" >>= fun struct1 ->
            line (sprintf "%s = llvm.insertvalue %s, %s[0] : !llvm.struct<(i32, i64, i64)>" struct1 tagReg structUndef) >>.
            emitI64 0L >>= fun zero64 ->
            freshSSAWithType "!llvm.struct<(i32, i64, i64)>" >>= fun struct2 ->
            line (sprintf "%s = llvm.insertvalue %s, %s[1] : !llvm.struct<(i32, i64, i64)>" struct2 zero64 struct1) >>.
            freshSSAWithType "!llvm.struct<(i32, i64, i64)>" >>= fun structFinal ->
            line (sprintf "%s = llvm.insertvalue %s, %s[2] : !llvm.struct<(i32, i64, i64)>" structFinal zero64 struct2) >>.
            emit (Value(structFinal, "!llvm.struct<(i32, i64, i64)>"))
        | Error msg ->
            emit (Error (sprintf "Result constructor value error: %s" msg))
    | [singleChild] ->
        // Single child - might be just the value
        atNode singleChild callEmitExpr >>= fun valueResult ->
        match valueResult with
        | Value(valueSsa, valueType) ->
            let tag = match op with OkCtor -> 0 | ErrorCtor -> 1
            freshSSAWithType "!llvm.struct<(i32, i64, i64)>" >>= fun structUndef ->
            line (sprintf "%s = llvm.mlir.undef : !llvm.struct<(i32, i64, i64)>" structUndef) >>.
            emitI32 tag >>= fun tagReg ->
            freshSSAWithType "!llvm.struct<(i32, i64, i64)>" >>= fun struct1 ->
            line (sprintf "%s = llvm.insertvalue %s, %s[0] : !llvm.struct<(i32, i64, i64)>" struct1 tagReg structUndef) >>.
            (if valueType = "i32" then emitExtsi valueSsa "i32" "i64" else emit valueSsa) >>= fun payload64 ->
            freshSSAWithType "!llvm.struct<(i32, i64, i64)>" >>= fun struct2 ->
            line (sprintf "%s = llvm.insertvalue %s, %s[1] : !llvm.struct<(i32, i64, i64)>" struct2 payload64 struct1) >>.
            emitI64 0L >>= fun zero64 ->
            freshSSAWithType "!llvm.struct<(i32, i64, i64)>" >>= fun structFinal ->
            line (sprintf "%s = llvm.insertvalue %s, %s[2] : !llvm.struct<(i32, i64, i64)>" structFinal zero64 struct2) >>.
            emit (Value(structFinal, "!llvm.struct<(i32, i64, i64)>"))
        | _ ->
            emit (Error "Result constructor requires a value")
    | [] ->
        emit (Error "Result constructor has no arguments")

/// Try to emit as Result operation
let private tryEmitResult : Emit<ExprResult option> =
    getZipper >>= fun z ->
    match extractResultOp z.Graph z.Focus with
    | Some op -> handleResult op |>> Some
    | None -> emit None

// ===================================================================
// Core Operation Handlers (Alloy.Core)
// ===================================================================

/// Handler for Core operations like ignore
let private handleCore (op: CoreOp) : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z
    match op with
    | Ignore ->
        // ignore: emit the argument for side effects, then return Void
        // Structure: App [Ident:ignore, argNode]
        match children with
        | _ :: argNode :: _ ->
            // Emit the argument (for its side effects)
            atNode argNode callEmitExpr >>= fun _ ->
            // Discard the result and return Void
            emit Void
        | [_ignoreIdent] ->
            // No argument - this shouldn't happen but handle gracefully
            emit Void
        | [] ->
            emit Void

/// Try to emit as Core operation
let private tryEmitCore : Emit<ExprResult option> =
    getZipper >>= fun z ->
    match extractCoreOp z.Graph z.Focus with
    | Some op -> handleCore op |>> Some
    | None -> emit None

// ===================================================================
// NativePtr Operation Handlers
// ===================================================================

/// Helper to extract all arguments from a curried NativePtr call
/// NativePtr.set ptr idx value -> [ptr, idx, value]
/// Uses the same curried collection algorithm as extractFunctionCall
/// PSG children order may be [arg, func] due to construction order
let private extractNativePtrArgs (graph: ProgramSemanticGraph) (node: PSGNode) : PSGNode list =
    let getChildNodes (n: PSGNode) : PSGNode list =
        match n.Children with
        | ChildrenState.Parent ids -> ids |> List.choose (fun id -> Map.tryFind id.Value graph.Nodes)
        | _ -> []

    // Helper to check if a node looks like a value/argument (not a function ref)
    let isLikelyArg (n: PSGNode) =
        n.SyntaxKind.StartsWith("Const") ||
        n.SyntaxKind.StartsWith("PropertyAccess") ||
        n.SyntaxKind.StartsWith("AddressOf") ||
        n.SyntaxKind.StartsWith("DotIndexedGet") ||
        n.SyntaxKind.StartsWith("Ident:")  // Variable references are args in NativePtr context

    // Given two children, determine which is the func/partial and which is the arg
    let classifyChildren (c1: PSGNode) (c2: PSGNode) : (PSGNode * PSGNode) =
        // If one is clearly a value, it's the arg
        if isLikelyArg c1 && not (isLikelyArg c2) then (c2, c1)
        elif isLikelyArg c2 && not (isLikelyArg c1) then (c1, c2)
        // TypeApp is the function (stackalloc<T>)
        elif c1.SyntaxKind.StartsWith("TypeApp") then (c1, c2)
        elif c2.SyntaxKind.StartsWith("TypeApp") then (c2, c1)
        // LongIdent is the function
        elif c1.SyntaxKind.StartsWith("LongIdent:") && c2.SyntaxKind.StartsWith("App") then (c2, c1)
        elif c2.SyntaxKind.StartsWith("LongIdent:") && c1.SyntaxKind.StartsWith("App") then (c1, c2)
        // Both Apps - first is usually arg in PSG
        elif c1.SyntaxKind.StartsWith("App") && c2.SyntaxKind.StartsWith("App") then (c2, c1)
        // c2 is function ref
        elif c2.SyntaxKind.StartsWith("LongIdent:") then (c2, c1)
        elif c1.SyntaxKind.StartsWith("LongIdent:") then (c1, c2)
        else (c1, c2)

    // Recursively traverse nested App nodes to collect all arguments
    let rec collectCurriedArgs (appNode: PSGNode) (accArgs: PSGNode list) : PSGNode list =
        let children = getChildNodes appNode
        match children with
        | [c1; c2] ->
            let (funcOrApp, arg) = classifyChildren c1 c2
            let newArgs = arg :: accArgs
            if funcOrApp.SyntaxKind.StartsWith("App") then
                collectCurriedArgs funcOrApp newArgs
            elif funcOrApp.SyntaxKind.StartsWith("TypeApp") then
                newArgs
            else
                newArgs
        | [single] ->
            if single.SyntaxKind.StartsWith("App") then
                collectCurriedArgs single accArgs
            else
                accArgs
        | first :: rest ->
            let (funcOrApp, arg) = classifyChildren first (List.head rest)
            let remainingArgs = List.tail rest
            let newArgs = arg :: remainingArgs @ accArgs
            if funcOrApp.SyntaxKind.StartsWith("App") then
                collectCurriedArgs funcOrApp newArgs
            elif funcOrApp.SyntaxKind.StartsWith("TypeApp") then
                newArgs
            else
                newArgs
        | [] -> accArgs

    if node.SyntaxKind.StartsWith("App") then
        collectCurriedArgs node []
    else
        []

/// Handler for NativePtr operations
let private handleNativePtr (op: NativePtrOp) : Emit<ExprResult> =
    getZipper >>= fun z ->
    let args = extractNativePtrArgs z.Graph z.Focus

    match op with
    | PtrSet ->
        // NativePtr.set ptr idx value -> llvm.store
        match args with
        | ptrNode :: idxNode :: valueNode :: _ ->
            atNode ptrNode callEmitExpr >>= fun ptrResult ->
            atNode idxNode callEmitExpr >>= fun idxResult ->
            atNode valueNode callEmitExpr >>= fun valueResult ->
            match ptrResult, idxResult, valueResult with
            | Value(ptrReg, _), Value(idxReg, idxType), Value(valueReg, valueType) ->
                // Compute element pointer: ptr + idx * sizeof(element)
                // MLIR LLVM dialect syntax: llvm.getelementptr %ptr[%idx] : (ptr_type, idx_type) -> result_type, element_type
                freshSSAWithType "!llvm.ptr" >>= fun elemPtr ->
                line (sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, %s) -> !llvm.ptr, %s" elemPtr ptrReg idxReg idxType valueType) >>.
                // Store value at pointer
                line (sprintf "llvm.store %s, %s : %s, !llvm.ptr" valueReg elemPtr valueType) >>.
                emit Void
            | _ -> emit (Error "NativePtr.set: invalid arguments")
        | _ -> emit (Error "NativePtr.set requires ptr, idx, value")

    | PtrGet ->
        // NativePtr.get ptr idx -> llvm.load
        match args with
        | ptrNode :: idxNode :: _ ->
            atNode ptrNode callEmitExpr >>= fun ptrResult ->
            atNode idxNode callEmitExpr >>= fun idxResult ->
            match ptrResult, idxResult with
            | Value(ptrReg, _), Value(idxReg, idxType) ->
                // Compute element pointer
                // MLIR LLVM dialect syntax: llvm.getelementptr %ptr[%idx] : (ptr_type, idx_type) -> result_type, element_type
                freshSSAWithType "!llvm.ptr" >>= fun elemPtr ->
                line (sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, %s) -> !llvm.ptr, i8" elemPtr ptrReg idxReg idxType) >>.
                // Load value - default to i8 for byte arrays
                freshSSAWithType "i8" >>= fun result ->
                line (sprintf "%s = llvm.load %s : !llvm.ptr -> i8" result elemPtr) >>.
                emit (Value(result, "i8"))
            | _ -> emit (Error "NativePtr.get: invalid arguments")
        | _ -> emit (Error "NativePtr.get requires ptr, idx")

    | PtrRead ->
        // NativePtr.read ptr -> llvm.load at offset 0
        match args with
        | ptrNode :: _ ->
            atNode ptrNode callEmitExpr >>= fun ptrResult ->
            match ptrResult with
            | Value(ptrReg, _) ->
                freshSSAWithType "i8" >>= fun result ->
                line (sprintf "%s = llvm.load %s : !llvm.ptr -> i8" result ptrReg) >>.
                emit (Value(result, "i8"))
            | _ -> emit (Error "NativePtr.read: invalid pointer")
        | _ -> emit (Error "NativePtr.read requires ptr")

    | PtrWrite ->
        // NativePtr.write ptr value -> llvm.store at offset 0
        match args with
        | ptrNode :: valueNode :: _ ->
            atNode ptrNode callEmitExpr >>= fun ptrResult ->
            atNode valueNode callEmitExpr >>= fun valueResult ->
            match ptrResult, valueResult with
            | Value(ptrReg, _), Value(valueReg, valueType) ->
                line (sprintf "llvm.store %s, %s : %s, !llvm.ptr" valueReg ptrReg valueType) >>.
                emit Void
            | _ -> emit (Error "NativePtr.write: invalid arguments")
        | _ -> emit (Error "NativePtr.write requires ptr, value")

    | PtrAdd ->
        // NativePtr.add ptr offset -> getelementptr
        match args with
        | ptrNode :: offsetNode :: _ ->
            atNode ptrNode callEmitExpr >>= fun ptrResult ->
            atNode offsetNode callEmitExpr >>= fun offsetResult ->
            match ptrResult, offsetResult with
            | Value(ptrReg, _), Value(offsetReg, offsetType) ->
                freshSSAWithType "!llvm.ptr" >>= fun result ->
                // MLIR LLVM dialect syntax: llvm.getelementptr %ptr[%offset] : (ptr_type, offset_type) -> result_type, element_type
                line (sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, %s) -> !llvm.ptr, i8" result ptrReg offsetReg offsetType) >>.
                emit (Value(result, "!llvm.ptr"))
            | _ -> emit (Error "NativePtr.add: invalid arguments")
        | _ -> emit (Error "NativePtr.add requires ptr, offset")

    | PtrToInt ->
        // NativePtr.toNativeInt ptr -> llvm.ptrtoint
        match args with
        | ptrNode :: _ ->
            atNode ptrNode callEmitExpr >>= fun ptrResult ->
            match ptrResult with
            | Value(ptrReg, _) ->
                freshSSAWithType "i64" >>= fun result ->
                line (sprintf "%s = llvm.ptrtoint %s : !llvm.ptr to i64" result ptrReg) >>.
                emit (Value(result, "i64"))
            | _ -> emit (Error "NativePtr.toNativeInt: invalid pointer")
        | _ -> emit (Error "NativePtr.toNativeInt requires ptr")

    | PtrNull ->
        // NativePtr.nullPtr<T> -> llvm null pointer
        freshSSAWithType "!llvm.ptr" >>= fun result ->
        line (sprintf "%s = llvm.mlir.zero : !llvm.ptr" result) >>.
        emit (Value(result, "!llvm.ptr"))

    | PtrToVoid | PtrOfVoid ->
        // NativePtr.toVoidPtr / ofVoidPtr - just pass through the pointer
        // In MLIR/LLVM all pointers are opaque, so no actual conversion needed
        match args with
        | ptrNode :: _ ->
            atNode ptrNode callEmitExpr >>= fun ptrResult ->
            match ptrResult with
            | Value(ptrReg, _) -> emit (Value(ptrReg, "!llvm.ptr"))
            | _ -> emit (Error "NativePtr.toVoidPtr/ofVoidPtr: invalid pointer")
        | _ -> emit (Error "NativePtr.toVoidPtr/ofVoidPtr requires ptr")

/// Try to emit as NativePtr operation
let private tryEmitNativePtr : Emit<ExprResult option> =
    getZipper >>= fun z ->
    match extractNativePtrOp z.Graph z.Focus with
    | Some op -> handleNativePtr op |>> Some
    | None -> emit None

// ===================================================================
// Arithmetic and Comparison Handlers
// ===================================================================

/// Helper to extract operands from curried binary operator application
/// For ((op) left) right, structure is App[App[op, left], right]
/// PSG children order may be [arg, App] or [App, arg]
/// The inner App contains [left_operand, operator] where operator starts with LongIdent:op_ or Ident:op_
let private extractBinaryOperands (graph: ProgramSemanticGraph) (children: PSGNode list) : (PSGNode * PSGNode) option =
    // Find the inner App among children
    let findInnerApp (children: PSGNode list) =
        children |> List.tryFind (fun c -> c.SyntaxKind.StartsWith("App"))

    // Check if a node is an operator (starts with op_ pattern)
    let isOperatorNode (n: PSGNode) =
        n.SyntaxKind.StartsWith("LongIdent:op_") ||
        n.SyntaxKind.StartsWith("Ident:op_")

    // Find the operand (non-operator) child in inner App
    let findOperandChild (innerChildren: PSGNode list) =
        innerChildren |> List.tryFind (fun c -> not (isOperatorNode c))

    match children with
    | [c1; c2] ->
        // Two children - find which is the inner App and which is the right operand
        let innerAppOpt = findInnerApp [c1; c2]
        match innerAppOpt with
        | Some innerApp ->
            let rightNode = if c1.Id = innerApp.Id then c2 else c1
            // Get inner children to find left operand
            let innerChildren =
                match innerApp.Children with
                | ChildrenState.Parent ids -> ids |> List.choose (fun id -> Map.tryFind id.Value graph.Nodes)
                | _ -> []
            // Left operand is the non-operator child of inner App
            let leftNodeOpt = findOperandChild innerChildren
            match leftNodeOpt with
            | Some leftNode -> Some (leftNode, rightNode)
            | None -> None
        | None -> None
    | _ :: leftNode :: rightNode :: _ ->
        // Non-curried form: [op, left, right]
        Some (leftNode, rightNode)
    | _ -> None

/// Handler for arithmetic operations (binary and unary)
let private handleArith (op: ArithOp) : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z

    // Check for unary operations first
    match op with
    | Negate ->
        // Unary negation: -value
        // Structure is App[op_UnaryNegation, operand]
        match children with
        | [_opNode; operandNode] ->
            atNode operandNode callEmitExpr >>= fun operandResult ->
            match operandResult with
            | Value(operandReg, operandType) ->
                // Emit: result = arith.constant 0 : type; result2 = arith.subi 0, operand : type
                // Use type-aware zero to match operand type
                emitZero operandType >>= fun zeroReg ->
                freshSSAWithType operandType >>= fun result ->
                line (sprintf "%s = arith.subi %s, %s : %s" result zeroReg operandReg operandType) >>.
                emit (Value(result, operandType))
            | Error msg -> emit (Error msg)
            | Void -> emit (Error "Unary negation operand must produce value")
        | [operandNode] ->
            // Direct application form
            atNode operandNode callEmitExpr >>= fun operandResult ->
            match operandResult with
            | Value(operandReg, operandType) ->
                emitZero operandType >>= fun zeroReg ->
                freshSSAWithType operandType >>= fun result ->
                line (sprintf "%s = arith.subi %s, %s : %s" result zeroReg operandReg operandType) >>.
                emit (Value(result, operandType))
            | Error msg -> emit (Error msg)
            | Void -> emit (Error "Unary negation operand must produce value")
        | _ -> emit (Error "Unary negation requires 1 operand")
    | Not ->
        // Logical not: not value
        let operandNodeOpt =
            match children with
            | [_opNode; operandNode] -> Some operandNode
            | [operandNode] -> Some operandNode
            | _ -> None
        match operandNodeOpt with
        | Some operandNode ->
            atNode operandNode callEmitExpr >>= fun operandResult ->
            match operandResult with
            | Value(operandReg, _) ->
                // Emit: result = arith.constant 1 : i1; result2 = arith.xori operand, 1 : i1
                emitI1 true >>= fun oneReg ->
                freshSSAWithType "i1" >>= fun result ->
                line (sprintf "%s = arith.xori %s, %s : i1" result operandReg oneReg) >>.
                emit (Value(result, "i1"))
            | Error msg -> emit (Error msg)
            | Void -> emit (Error "Logical not operand must produce value")
        | None -> emit (Error "Logical not requires 1 operand")
    | _ ->
        // Binary operations
        match extractBinaryOperands z.Graph children with
        | Some (leftNode, rightNode) ->
            atNode leftNode callEmitExpr >>= fun leftResult ->
            match leftResult with
            | Value(leftReg, leftType) ->
                atNode rightNode callEmitExpr >>= fun rightResult ->
                match rightResult with
                | Value(rightReg, _) ->
                    let mlirOp = arithOpToMLIR op
                    freshSSAWithType leftType >>= fun result ->
                    line (sprintf "%s = %s %s, %s : %s" result mlirOp leftReg rightReg leftType) >>.
                    emit (Value(result, leftType))
                | Error msg -> emit (Error msg)
                | Void -> emit (Error "Right operand must produce value")
            | Error msg -> emit (Error msg)
            | Void -> emit (Error "Left operand must produce value")
        | None ->
            emit (Error "Binary arithmetic operation requires 2 operands")

/// Try to emit as arithmetic operation
let private tryEmitArith : Emit<ExprResult option> =
    getZipper >>= fun z ->
    match extractArithOp z.Graph z.Focus with
    | Some op -> handleArith op |>> Some
    | None -> emit None

/// Handler for comparison operations
let private handleCompare (op: CompareOp) : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z
    match extractBinaryOperands z.Graph children with
    | Some (leftNode, rightNode) ->
        atNode leftNode callEmitExpr >>= fun leftResult ->
        match leftResult with
        | Value(leftReg, leftType) ->
            atNode rightNode callEmitExpr >>= fun rightResult ->
            match rightResult with
            | Value(rightReg, _) ->
                let pred = compareOpToPredicate op
                freshSSAWithType "i1" >>= fun result ->
                line (sprintf "%s = arith.cmpi %s, %s, %s : %s" result pred leftReg rightReg leftType) >>.
                emit (Value(result, "i1"))
            | Error msg -> emit (Error msg)
            | Void -> emit (Error "Right operand must produce value")
        | Error msg -> emit (Error msg)
        | Void -> emit (Error "Left operand must produce value")
    | None ->
        emit (Error "Comparison operation requires 2 operands")

/// Try to emit as comparison operation
let private tryEmitCompare : Emit<ExprResult option> =
    getZipper >>= fun z ->
    match extractCompareOp z.Graph z.Focus with
    | Some op -> handleCompare op |>> Some
    | None -> emit None

// ===================================================================
// Type Conversion Handler
// ===================================================================

/// Handler for type conversion operations (byte, int, int64, etc.)
let private handleConversion (op: ConversionOp) : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z

    // Get the operand (skip the operator node)
    let operandNode =
        match children with
        | [_opNode; operand] -> Some operand
        | [operand] -> Some operand  // Sometimes just the operand
        | _ -> None

    match operandNode with
    | Some node ->
        atNode node callEmitExpr >>= fun operandResult ->
        match operandResult with
        | Value(operandReg, operandType) ->
            // Determine target type based on conversion op
            let targetType =
                match op with
                | ToByte | ToSByte -> "i8"
                | ToInt16 | ToUInt16 -> "i16"
                | ToInt32 | ToUInt32 -> "i32"
                | ToInt64 | ToUInt64 -> "i64"
                | ToFloat -> "f32"
                | ToDouble -> "f64"

            // If types match, no conversion needed
            if operandType = targetType then
                emit (Value(operandReg, targetType))
            else
                // Emit appropriate conversion
                freshSSAWithType targetType >>= fun result ->
                let convOp =
                    match operandType, targetType with
                    // Integer truncation
                    | "i64", "i32" | "i64", "i16" | "i64", "i8"
                    | "i32", "i16" | "i32", "i8"
                    | "i16", "i8" -> sprintf "%s = arith.trunci %s : %s to %s" result operandReg operandType targetType
                    // Integer extension (signed)
                    | "i8", "i16" | "i8", "i32" | "i8", "i64"
                    | "i16", "i32" | "i16", "i64"
                    | "i32", "i64" -> sprintf "%s = arith.extsi %s : %s to %s" result operandReg operandType targetType
                    // Float to integer
                    | "f32", _ | "f64", _ when targetType.StartsWith("i") ->
                        sprintf "%s = arith.fptosi %s : %s to %s" result operandReg operandType targetType
                    // Integer to float
                    | _, "f32" | _, "f64" when operandType.StartsWith("i") ->
                        sprintf "%s = arith.sitofp %s : %s to %s" result operandReg operandType targetType
                    // Float to float
                    | "f32", "f64" -> sprintf "%s = arith.extf %s : f32 to f64" result operandReg
                    | "f64", "f32" -> sprintf "%s = arith.truncf %s : f64 to f32" result operandReg
                    // Default: try bitcast (same size types)
                    | _ -> sprintf "%s = arith.trunci %s : %s to %s" result operandReg operandType targetType

                line convOp >>.
                emit (Value(result, targetType))
        | Void -> emit (Error "Conversion operand must produce value")
        | Error msg -> emit (Error msg)
    | None ->
        emit (Error "Conversion operation requires 1 operand")

/// Try to emit as type conversion
let private tryEmitConversion : Emit<ExprResult option> =
    getZipper >>= fun z ->
    match extractConversionOp z.Graph z.Focus with
    | Some op -> handleConversion op |>> Some
    | None -> emit None

// ===================================================================
// Constant Emission Handler
// ===================================================================

/// Handler for constant values
let private handleConst (kind: string) : Emit<ExprResult> =
    if kind.StartsWith("Const:Int32") then
        match extractInt32FromKind kind with
        | Some v -> emitI32 v |>> fun ssa -> Value(ssa, "i32")
        | None -> emit (Error "Invalid i32 constant")
    elif kind.StartsWith("Const:Int64") then
        match extractInt64FromKind kind with
        | Some v -> emitI64 v |>> fun ssa -> Value(ssa, "i64")
        | None -> emit (Error "Invalid i64 constant")
    // IMPORTANT: Check Const:Bytes BEFORE Const:Byte (prefix match issue)
    elif kind.StartsWith("Const:Bytes") then
        match extractBytesFromKind kind with
        | Some (bytes, len) ->
            // Create a NativeStr/ByteArray struct: { ptr: !llvm.ptr, len: i64 }
            // Register bytes as a global constant
            registerByteLiteral bytes >>= fun globalName ->
            emitAddressOf globalName >>= fun ptr ->
            emitI64 (int64 len) >>= fun lenReg ->
            // Build the struct
            freshSSAWithType "!llvm.struct<(ptr, i64)>" >>= fun undef ->
            line (sprintf "%s = llvm.mlir.undef : !llvm.struct<(ptr, i64)>" undef) >>.
            freshSSAWithType "!llvm.struct<(ptr, i64)>" >>= fun withPtr ->
            line (sprintf "%s = llvm.insertvalue %s, %s[0] : !llvm.struct<(ptr, i64)>" withPtr ptr undef) >>.
            freshSSAWithType "!llvm.struct<(ptr, i64)>" >>= fun withLen ->
            line (sprintf "%s = llvm.insertvalue %s, %s[1] : !llvm.struct<(ptr, i64)>" withLen lenReg withPtr) >>.
            emit (Value(withLen, "!llvm.struct<(ptr, i64)>"))
        | None -> emit (Error "Invalid byte array constant")
    elif kind.StartsWith("Const:Byte ") then
        match extractByteFromKind kind with
        | Some v -> emitI8 v |>> fun ssa -> Value(ssa, "i8")
        | None -> emit (Error "Invalid byte constant")
    elif kind.StartsWith("Const:String") then
        match extractStringFromKind kind with
        | Some content ->
            registerStringLiteral content >>= fun globalName ->
            emitAddressOf globalName |>> fun ptr ->
            Value(ptr, "!llvm.ptr")
        | None -> emit (Error "Invalid string constant")
    elif kind = "Const:Bool true" then
        emitI1 true |>> fun ssa -> Value(ssa, "i1")
    elif kind = "Const:Bool false" then
        emitI1 false |>> fun ssa -> Value(ssa, "i1")
    elif kind = "Const:Unit" || kind.StartsWith("Const:()") then
        emit Void
    else
        emit (Error (sprintf "Unknown constant: %s" kind))

/// Try to emit as constant
let private tryEmitConst : Emit<ExprResult option> =
    getFocus >>= fun node ->
    if isConst node then
        handleConst node.SyntaxKind |>> Some
    else
        emit None

// ===================================================================
// Variable Reference Handler
// ===================================================================

/// Handler for identifier/variable references using SYMBOL-BASED lookup
/// For mutable variables, load from stack slot to get current value
/// For immutable locals, look up by symbol in SSA context
let private handleIdentBySymbol (sym: FSharp.Compiler.Symbols.FSharpSymbol) (node: PSGNode) : Emit<ExprResult> =
    // First check if this is a mutable variable with a stack slot (still string-based for now)
    lookupMutableSlot sym.DisplayName >>= fun mutableOpt ->
    match mutableOpt with
    | Some (slotPtr, elemType) ->
        // Mutable variable: load current value from stack slot
        freshSSAWithType elemType >>= fun loadedValue ->
        line (sprintf "%s = llvm.load %s : !llvm.ptr -> %s" loadedValue slotPtr elemType) >>.
        emit (Value(loadedValue, elemType))
    | None ->
        // Try symbol-based lookup (new, preferred)
        lookupSymbolSSA sym >>= fun symbolOpt ->
        match symbolOpt with
        | Some ssaInfo ->
            emit (Value(ssaInfo.Value, ssaInfo.Type))
        | None ->
            // Fallback to string-based lookup (deprecated, for backwards compatibility)
            lookupLocal sym.DisplayName >>= fun localOpt ->
            match localOpt with
            | Some ssaName ->
                lookupLocalType sym.DisplayName >>= fun typeOpt ->
                emit (Value(ssaName, typeOpt |> Option.defaultValue "i32"))
            | None ->
                emit (Error (sprintf "Variable not found: %s" sym.DisplayName))

/// Handler for identifier/variable references (DEPRECATED - use handleIdentBySymbol)
/// For mutable variables, load from stack slot to get current value
let private handleIdent (name: string) : Emit<ExprResult> =
    // First check if this is a mutable variable with a stack slot
    lookupMutableSlot name >>= fun mutableOpt ->
    match mutableOpt with
    | Some (slotPtr, elemType) ->
        // Mutable variable: load current value from stack slot
        freshSSAWithType elemType >>= fun loadedValue ->
        line (sprintf "%s = llvm.load %s : !llvm.ptr -> %s" loadedValue slotPtr elemType) >>.
        emit (Value(loadedValue, elemType))
    | None ->
        // Regular (immutable) local variable
        lookupLocal name >>= fun localOpt ->
        match localOpt with
        | Some ssaName ->
            lookupLocalType name >>= fun typeOpt ->
            emit (Value(ssaName, typeOpt |> Option.defaultValue "i32"))
        | None ->
            emit (Error (sprintf "Variable not found: %s" name))

/// Try to emit as identifier
/// For local variables, looks up in scope using SYMBOL-BASED lookup.
/// For [<Literal>] constants, inlines the value directly.
/// For module-level values (like NativeStr.empty), emits a zero-arg function call.
let private tryEmitIdent : Emit<ExprResult option> =
    getFocus >>= fun node ->
    if isIdent node then
        match node.Symbol with
        | Some sym ->
            // Use symbol-based lookup (new, preferred)
            handleIdentBySymbol sym node >>= fun result ->
            match result with
            | Error errMsg ->
                // Not found in local scope - check if it's a [<Literal>] constant or module-level value
                match sym with
                | :? FSharp.Compiler.Symbols.FSharpMemberOrFunctionOrValue as mfv when mfv.LiteralValue.IsSome ->
                    // This is a [<Literal>] constant - inline the value
                    match mfv.LiteralValue.Value with
                    | :? int as i -> emitI32 i |>> fun ssa -> Some (Value(ssa, "i32"))
                    | :? int64 as i -> emitI64 i |>> fun ssa -> Some (Value(ssa, "i64"))
                    | :? byte as b -> emitI8 b |>> fun ssa -> Some (Value(ssa, "i8"))
                    | :? bool as b -> emitI1 b |>> fun ssa -> Some (Value(ssa, "i1"))
                    | other ->
                        // Unsupported literal type
                        emit (Some (Error (sprintf "Unsupported literal type: %A" (other.GetType()))))
                | :? FSharp.Compiler.Symbols.FSharpMemberOrFunctionOrValue as mfv
                    when mfv.IsModuleValueOrMember &&
                         not (mfv.DisplayName.Contains(".ctor") || mfv.DisplayName.Contains("ctor")) ->
                    // Module-level value (not a constructor) - emit as zero-arg function call
                    // Get the result type from the PSG node's Type field
                    let resultType =
                        match node.Type with
                        | Some ftype ->
                            if ftype.IsFunctionType then
                                // This is a function value - return it as a function pointer
                                "!llvm.ptr"
                            else
                                fsharpTypeToMLIR ftype
                        | None -> "!llvm.ptr"  // Default for unknown types

                    // Get sanitized function name
                    let funcName =
                        sym.DisplayName
                            .Replace(".", "_")
                            .Replace("``", "")

                    // Emit as zero-arg function call
                    freshSSAWithType resultType >>= fun resultReg ->
                    line (sprintf "%s = func.call @%s() : () -> %s" resultReg funcName resultType) >>.
                    emit (Some (Value(resultReg, resultType)))
                | _ ->
                    // Local variable that should have been bound but wasn't found
                    // This is an error - don't fall back to function call
                    emit (Some (Error errMsg))
            | success -> emit (Some success)
        | None ->
            // No symbol - fall back to string-based lookup
            let kind = node.SyntaxKind
            let name =
                if kind.StartsWith("Ident:") then kind.Substring("Ident:".Length)
                elif kind.StartsWith("Value:") then kind.Substring("Value:".Length)
                elif kind.StartsWith("LongIdent:") then kind.Substring("LongIdent:".Length)
                else kind
            handleIdent name |>> Some
    else
        emit None

// ===================================================================
// Sequential Expression Handler
// ===================================================================

/// Handler for sequential expressions - emit all children, return last
let private handleSequential () : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z
    match children with
    | [] -> emit Void
    | [single] -> atNode single callEmitExpr
    | _ ->
        let allButLast = children |> List.take (List.length children - 1)
        let lastNode = children |> List.last
        forEach (fun child ->
            atNode child callEmitExpr >>= fun _ -> emit ()
        ) allButLast >>.
        atNode lastNode callEmitExpr

/// Try to emit as sequential
let private tryEmitSequential : Emit<ExprResult option> =
    getFocus >>= fun node ->
    if isSequential node then
        handleSequential () |>> Some
    else
        emit None

// ===================================================================
// Let Binding Handler
// ===================================================================

/// Emit a single Binding node
/// For mutable bindings (Binding:Mutable), allocate stack space and store initial value
/// Uses SYMBOL-BASED binding for proper variable resolution
let private emitBindingNode (bindingNode: PSGNode) : Emit<ExprResult> =
    let isMutable = bindingNode.SyntaxKind = "Binding:Mutable"

    atNode bindingNode (
        getZipper >>= fun bz ->
        let bindingChildren = PSGZipper.childNodes bz
        // Binding children are typically [valueExpr, pattern] or [valueExpr, pattern, ...]
        // The value expression is the FIRST child (before the pattern)
        match bindingChildren with
        | valueNode :: _ when not (valueNode.SyntaxKind.StartsWith("Pattern:")) ->
            atNode valueNode callEmitExpr
        | [_patternOnly] ->
            emit Void
        | [] ->
            emit Void
        | _ ->
            emit Void  // Edge case: only patterns, no value expression
    ) >>= fun valueResult ->
    match valueResult with
    | Value(ssa, typ) ->
        match bindingNode.Symbol with
        | Some symbol ->
            if isMutable then
                // Mutable binding: allocate stack slot and store initial value
                // This avoids SSA complications with control flow
                freshSSAWithType "!llvm.ptr" >>= fun slotPtr ->
                emitI64 1L >>= fun oneReg ->
                line (sprintf "%s = llvm.alloca %s x %s : (i64) -> !llvm.ptr" slotPtr oneReg typ) >>.
                line (sprintf "llvm.store %s, %s : %s, !llvm.ptr" ssa slotPtr typ) >>.
                registerMutableSlot symbol.DisplayName slotPtr typ >>.
                // Bind using both symbol-based (new) and string-based (deprecated) for compatibility
                bindSymbolSSA symbol bindingNode.Id ssa typ >>.
                bindLocal symbol.DisplayName ssa typ >>.
                emit (Value(ssa, typ))
            else
                // Immutable binding: use symbol-based binding (primary) + string fallback
                bindSymbolSSA symbol bindingNode.Id ssa typ >>.
                bindLocal symbol.DisplayName ssa typ >>.
                emit (Value(ssa, typ))
        | None ->
            emit (Value(ssa, typ))
    | other -> emit other

/// Handler for let bindings
let private handleLet () : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z

    let rec processChildren (nodes: PSGNode list) (lastResult: ExprResult) : Emit<ExprResult> =
        match nodes with
        | [] -> emit lastResult
        | node :: rest ->
            if node.SyntaxKind.StartsWith("Binding") then
                emitBindingNode node >>= fun result ->
                processChildren rest result
            elif node.SyntaxKind.StartsWith("Pattern:") then
                processChildren rest lastResult
            else
                atNode node callEmitExpr >>= fun result ->
                processChildren rest result

    processChildren children Void

/// Try to emit as let binding
let private tryEmitLet : Emit<ExprResult option> =
    getFocus >>= fun node ->
    if isLet node then
        handleLet () |>> Some
    else
        emit None

// ===================================================================
// Mutable Variable Collection Helper
// ===================================================================

/// Collect all MutableSet variable names in a PSG subtree
/// Used by both while loops and if-expressions to properly handle SSA form
let private collectMutableSets (graph: ProgramSemanticGraph) (node: PSGNode) : string list =
    let rec collect (n: PSGNode) (acc: string Set) : string Set =
        // Check if this node is a MutableSet
        let acc' =
            match extractMutableSetName n with
            | Some varName -> Set.add varName acc
            | None -> acc
        // Recurse into children
        match n.Children with
        | ChildrenState.Parent ids ->
            let children = ids |> List.choose (fun id -> Map.tryFind id.Value graph.Nodes)
            children |> List.fold (fun a child -> collect child a) acc'
        | _ -> acc'
    collect node Set.empty |> Set.toList

// ===================================================================
// If Expression Handler
// ===================================================================

/// Handler for if expressions
/// Uses MLIR cf dialect block arguments for proper SSA when both branches produce values
/// CORRECT BY CONSTRUCTION: Uses PSG node's Type field to determine if expression produces a value
let private handleIf () : Emit<ExprResult> =
    getFocus >>= fun ifNode ->

    // Get the expected result type from the PSG node's Type field
    // This is set by FCS type integration and tells us if this if-expression produces a value
    let expectedType =
        match ifNode.Type with
        | Some ftype ->
            let mlirType = fsharpTypeToMLIR ftype
            if mlirType = "()" then None else Some mlirType
        | None -> None

    getState >>= fun state ->
    let labelNum = state.SSACounter
    let thenLabel = sprintf "then_%d" labelNum
    let elseLabel = sprintf "else_%d" labelNum
    let mergeLabel = sprintf "merge_%d" labelNum
    modifyState (fun s -> { s with SSACounter = s.SSACounter + 3 }) >>.

    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z
    match children with
    | condNode :: thenNode :: rest ->
        atNode condNode callEmitExpr >>= fun condResult ->
        match condResult with
        | Value(condReg, _) ->
            match rest with
            | [elseNode] ->
                // Both branches exist
                // CRITICAL: Save state before then-branch to restore for else-branch
                // This ensures the else-branch sees the original bindings, not the then-branch's mutations
                getState >>= fun stateBeforeThen ->

                emitCondBr condReg thenLabel elseLabel >>.
                emitBlockLabel thenLabel >>.
                atNode thenNode callEmitExpr >>= fun thenResult ->

                // Branch with value if we have one
                (match thenResult with
                 | Value(thenSsa, thenType) ->
                     line (sprintf "cf.br ^%s(%s : %s)" mergeLabel thenSsa thenType) >>.
                     emit (Some (thenSsa, thenType))
                 | _ ->
                     line (sprintf "cf.br ^%s" mergeLabel) >>.
                     emit None) >>= fun thenInfo ->

                // Get state after then-branch (for possible merge reconciliation later)
                getState >>= fun stateAfterThen ->

                // CRITICAL: Restore locals to pre-then state for the else-branch
                // Keep SSA counter to avoid name collisions, but reset local bindings
                modifyState (fun s -> { s with Locals = stateBeforeThen.Locals; LocalTypes = stateBeforeThen.LocalTypes }) >>.

                emitBlockLabel elseLabel >>.
                atNode elseNode callEmitExpr >>= fun elseResult ->

                // Branch with value if we have one
                (match elseResult with
                 | Value(elseSsa, elseType) ->
                     line (sprintf "cf.br ^%s(%s : %s)" mergeLabel elseSsa elseType) >>.
                     emit (Some elseType)
                 | _ ->
                     line (sprintf "cf.br ^%s" mergeLabel) >>.
                     emit None) >>= fun elseTypeInfo ->

                // Emit merge block - use expected type from PSG, fall back to branch inference
                let mergeType =
                    match expectedType with
                    | Some t -> Some t
                    | None ->
                        match thenInfo, elseTypeInfo with
                        | Some (_, t), _ -> Some t
                        | None, Some t -> Some t
                        | None, None -> None

                (match mergeType with
                 | Some typ ->
                     freshSSAWithType typ >>= fun mergeArg ->
                     line (sprintf "^%s(%s: %s):" mergeLabel mergeArg typ) >>.
                     emit (Value(mergeArg, typ))
                 | None ->
                     line (sprintf "^%s:" mergeLabel) >>.
                     emit Void)

            | [] ->
                // No else branch - simple conditional jump
                // With stack-based mutable variables, no SSA complications
                emitCondBr condReg thenLabel mergeLabel >>.
                emitBlockLabel thenLabel >>.
                atNode thenNode callEmitExpr >>= fun _ ->
                emitBr mergeLabel >>.
                emitBlockLabel mergeLabel >>.
                emit Void
            | _ -> emit (Error "If: unexpected structure")
        | _ -> emit (Error "If condition must produce value")
    | _ -> emit (Error "If: wrong number of children")

/// Try to emit as if expression
let private tryEmitIf : Emit<ExprResult option> =
    getFocus >>= fun node ->
    if isIf node then
        handleIf () |>> Some
    else
        emit None

// ===================================================================
// While Loop Handler
// ===================================================================

/// Handler for while loops
/// With stack-based mutable variables, this is now a simple loop without SSA complications
let private handleWhile () : Emit<ExprResult> =
    getState >>= fun state ->
    let labelNum = state.SSACounter
    let condLabel = sprintf "while_cond_%d" labelNum
    let bodyLabel = sprintf "while_body_%d" labelNum
    let exitLabel = sprintf "while_exit_%d" labelNum
    modifyState (fun s -> { s with SSACounter = s.SSACounter + 3 }) >>.

    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z

    match children with
    | condNode :: bodyNode :: _ ->
        // Simple loop structure - mutable variables are on the stack
        emitBr condLabel >>.
        emitBlockLabel condLabel >>.
        atNode condNode callEmitExpr >>= fun condResult ->
        match condResult with
        | Value(condReg, _) ->
            emitCondBr condReg bodyLabel exitLabel >>.
            emitBlockLabel bodyLabel >>.
            atNode bodyNode callEmitExpr >>= fun _ ->
            emitBr condLabel >>.
            emitBlockLabel exitLabel >>.
            emit Void
        | Void ->
            line "// ERROR: While condition returned Void" >>.
            emit Void
        | Error msg ->
            emit (Error msg)
    | _ ->
        emit (Error "While: expected condition and body")

/// Try to emit as while loop
let private tryEmitWhile : Emit<ExprResult option> =
    getFocus >>= fun node ->
    if isWhile node then
        handleWhile () |>> Some
    else
        emit None

// ===================================================================
// Match Expression Handler
// ===================================================================

/// Extract pattern variable name from a Pattern:UnionCase node's children
/// Note: Children are now in correct source order after PSG finalization
let private extractPatternVariableName (graph: ProgramSemanticGraph) (patternNode: PSGNode) : string option =
    match patternNode.Children with
    | ChildrenState.Parent ids ->
        let children = ids |> List.choose (fun id -> Map.tryFind id.Value graph.Nodes)
        // Look for a Pattern:Named node in the children
        children |> List.tryPick (fun child ->
            if child.SyntaxKind.StartsWith("Pattern:Named:") then
                // Extract name from "Pattern:Named:length" -> "length"
                let parts = child.SyntaxKind.Split(':')
                if parts.Length >= 3 then Some parts.[2] else None
            elif child.SyntaxKind.StartsWith("Pattern:") then
                // Might be a nested pattern or wildcard, extract from symbol
                child.Symbol |> Option.map (fun s -> s.DisplayName)
            else None
        )
    | _ -> None

/// Determine if a pattern is for the Ok case of Result
let private isOkPattern (node: PSGNode) : bool =
    node.SyntaxKind.Contains("UnionCase:Ok") ||
    (node.Symbol |> Option.map (fun s -> s.DisplayName = "Ok") |> Option.defaultValue false)

/// Determine if a pattern is for the Error case of Result
let private isErrorPattern (node: PSGNode) : bool =
    node.SyntaxKind.Contains("UnionCase:Error") ||
    (node.Symbol |> Option.map (fun s -> s.DisplayName = "Error") |> Option.defaultValue false)

/// Handler for match expressions with Result type support
/// Uses MLIR cf dialect block arguments for proper SSA phi-like behavior
let private handleMatch () : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z

    match children with
    | scrutineeNode :: clauses when clauses.Length > 0 ->
        atNode scrutineeNode callEmitExpr >>= fun scrutineeResult ->
        match scrutineeResult with
        | Value(scrutineeReg, scrutineeType) when scrutineeType = "!llvm.struct<(i32, i64, i64)>" ->
            // This is a Result type match
            line (sprintf "// Match on Result: %s" scrutineeReg) >>.

            // Extract the tag from the Result struct (field 0)
            freshSSAWithType "i32" >>= fun tagReg ->
            line (sprintf "%s = llvm.extractvalue %s[0] : !llvm.struct<(i32, i64, i64)>" tagReg scrutineeReg) >>.

            // Find Ok and Error clauses
            // Note: Children are now in correct source order after PSG finalization
            let findClause pred =
                clauses |> List.tryFind (fun clause ->
                    match clause.Children with
                    | ChildrenState.Parent ids ->
                        let clauseChildren = ids |> List.choose (fun id -> Map.tryFind id.Value z.Graph.Nodes)
                        clauseChildren |> List.exists (fun n -> n.SyntaxKind.StartsWith("Pattern:") && pred n)
                    | _ -> false)

            let okClause = findClause isOkPattern
            let errorClause = findClause isErrorPattern

            match okClause, errorClause with
            | Some okC, Some errC ->
                // Extract clause children helper
                let getClauseChildren (clause: PSGNode) =
                    match clause.Children with
                    | ChildrenState.Parent ids -> ids |> List.choose (fun id -> Map.tryFind id.Value z.Graph.Nodes)
                    | _ -> []

                let okChildren = getClauseChildren okC
                let errChildren = getClauseChildren errC

                // Find pattern and body nodes
                let okPatternOpt = okChildren |> List.tryFind (fun n -> n.SyntaxKind.StartsWith("Pattern:"))
                let okBodyOpt = okChildren |> List.tryFind (fun n -> not (n.SyntaxKind.StartsWith("Pattern:")))
                let errBodyOpt = errChildren |> List.tryFind (fun n -> not (n.SyntaxKind.StartsWith("Pattern:")))

                // Generate labels for branches
                freshLabel >>= fun okLabel ->
                freshLabel >>= fun errLabel ->
                freshLabel >>= fun endLabel ->

                // Generate the block argument name for the merge block
                // This will receive the result from whichever branch is taken
                freshSSAWithType "!llvm.ptr" >>= fun mergeArg ->

                // Compare tag to 0 (Ok case)
                freshSSAWithType "i1" >>= fun cmpReg ->
                emitI32 0 >>= fun zeroReg ->
                line (sprintf "%s = arith.cmpi eq, %s, %s : i32" cmpReg tagReg zeroReg) >>.
                line (sprintf "cf.cond_br %s, ^%s, ^%s" cmpReg okLabel errLabel) >>.

                // Ok branch
                line (sprintf "^%s:" okLabel) >>.

                // Extract payload (field 1 contains the int value for Ok case)
                freshSSAWithType "i64" >>= fun payloadReg ->
                line (sprintf "%s = llvm.extractvalue %s[1] : !llvm.struct<(i32, i64, i64)>" payloadReg scrutineeReg) >>.

                // Truncate to i32 for the length value
                freshSSAWithType "i32" >>= fun lengthReg ->
                line (sprintf "%s = arith.trunci %s : i64 to i32" lengthReg payloadReg) >>.

                // Bind the pattern variable (e.g., "length") if present
                (match okPatternOpt with
                 | Some patternNode ->
                     match extractPatternVariableName z.Graph patternNode with
                     | Some varName ->
                         bindLocal varName lengthReg "i32"
                     | None -> emit ()
                 | None -> emit ()) >>.

                // Emit Ok branch body and get result value
                (match okBodyOpt with
                 | Some bodyNode ->
                     atNode bodyNode callEmitExpr >>= fun okResult ->
                     match okResult with
                     | Value(okVal, _) ->
                         // Branch to merge block with Ok result as block argument
                         line (sprintf "cf.br ^%s(%s : !llvm.ptr)" endLabel okVal) >>.
                         emit okVal
                     | Void ->
                         // Void result - need a placeholder value
                         emitI64 0L >>= fun placeholder ->
                         line (sprintf "cf.br ^%s(%s : !llvm.ptr)" endLabel placeholder) >>.
                         emit placeholder
                     | Error msg ->
                         line (sprintf "// ERROR in Ok branch: %s" msg) >>.
                         emitI64 0L >>= fun placeholder ->
                         line (sprintf "cf.br ^%s(%s : !llvm.ptr)" endLabel placeholder) >>.
                         emit placeholder
                 | None ->
                     emitI64 0L >>= fun placeholder ->
                     line (sprintf "cf.br ^%s(%s : !llvm.ptr)" endLabel placeholder) >>.
                     emit placeholder) >>= fun okVal ->

                // Error branch
                line (sprintf "^%s:" errLabel) >>.

                // Emit Error branch body and get result value
                (match errBodyOpt with
                 | Some bodyNode ->
                     atNode bodyNode callEmitExpr >>= fun errResult ->
                     match errResult with
                     | Value(errVal, _) ->
                         // Branch to merge block with Error result as block argument
                         line (sprintf "cf.br ^%s(%s : !llvm.ptr)" endLabel errVal) >>.
                         emit errVal
                     | Void ->
                         emitI64 0L >>= fun placeholder ->
                         line (sprintf "cf.br ^%s(%s : !llvm.ptr)" endLabel placeholder) >>.
                         emit placeholder
                     | Error msg ->
                         line (sprintf "// ERROR in Error branch: %s" msg) >>.
                         emitI64 0L >>= fun placeholder ->
                         line (sprintf "cf.br ^%s(%s : !llvm.ptr)" endLabel placeholder) >>.
                         emit placeholder
                 | None ->
                     emitI64 0L >>= fun placeholder ->
                     line (sprintf "cf.br ^%s(%s : !llvm.ptr)" endLabel placeholder) >>.
                     emit placeholder) >>= fun _ ->

                // Merge block with block argument to receive the result
                line (sprintf "^%s(%s: !llvm.ptr):" endLabel mergeArg) >>.

                // Return the merge block argument as the match result
                emit (Value(mergeArg, "!llvm.ptr"))

            | _ ->
                // Fallback: just emit first clause body
                line (sprintf "// Match on %s : %s (non-Result fallback)" scrutineeReg scrutineeType) >>.
                match clauses with
                | firstClause :: _ ->
                    let clauseChildren =
                        match firstClause.Children with
                        | ChildrenState.Parent ids ->
                            ids |> List.choose (fun id -> Map.tryFind id.Value z.Graph.Nodes)
                        | _ -> []
                    match clauseChildren |> List.filter (fun n -> not (n.SyntaxKind.StartsWith("Pattern:"))) with
                    | bodyNode :: _ -> atNode bodyNode callEmitExpr
                    | [] -> emit (Value(scrutineeReg, scrutineeType))
                | [] -> emit (Error "Match with no clauses")

        | Value(scrutineeReg, scrutineeType) ->
            // Non-Result match - use simple fallback
            line (sprintf "// Match on %s : %s" scrutineeReg scrutineeType) >>.
            match clauses with
            | firstClause :: _ ->
                let clauseChildren =
                    match firstClause.Children with
                    | ChildrenState.Parent ids ->
                        ids |> List.choose (fun id -> Map.tryFind id.Value z.Graph.Nodes)
                    | _ -> []
                match clauseChildren |> List.filter (fun n -> not (n.SyntaxKind.StartsWith("Pattern:"))) with
                | bodyNode :: _ -> atNode bodyNode callEmitExpr
                | [] -> emit (Value(scrutineeReg, scrutineeType))
            | [] -> emit (Error "Match with no clauses")

        | Error msg -> emit (Error msg)
        | Void -> emit (Error "Match scrutinee must produce a value")
    | _ ->
        emit (Error "Match: expected scrutinee and clauses")

/// Try to emit as match expression
let private tryEmitMatch : Emit<ExprResult option> =
    getFocus >>= fun node ->
    if isMatch node then
        handleMatch () |>> Some
    else
        emit None

// ===================================================================
// Mutable Set Handler
// ===================================================================

/// Handler for mutable variable assignment
/// Stores the new value to the stack slot for the mutable variable
let private handleMutableSet (varName: string) : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z
    match children |> List.tryLast with
    | Some valueNode ->
        atNode valueNode callEmitExpr >>= fun valueResult ->
        match valueResult with
        | Value(ssa, typ) ->
            // Check if this is a mutable variable with a stack slot
            lookupMutableSlot varName >>= fun mutableOpt ->
            match mutableOpt with
            | Some (slotPtr, elemType) ->
                // Store new value to the stack slot
                line (sprintf "llvm.store %s, %s : %s, !llvm.ptr" ssa slotPtr elemType) >>.
                emit Void
            | None ->
                // Fallback: update the binding (for backwards compatibility)
                bindLocal varName ssa typ >>.
                emit Void
        | _ -> emit Void
    | None -> emit Void

/// Try to emit as mutable set
let private tryEmitMutableSet : Emit<ExprResult option> =
    getFocus >>= fun node ->
    match extractMutableSetName node with
    | Some varName -> handleMutableSet varName |>> Some
    | None -> emit None

// ===================================================================
// AddressOf Handler (&&var for mutable variables)
// ===================================================================

/// Handler for AddressOf expression (&&var or &var)
/// For mutable variables, returns the stack slot pointer directly
let private handleAddressOf () : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z
    match children with
    | [innerNode] ->
        // The inner node should be an identifier (the mutable variable)
        match innerNode.Symbol with
        | Some sym ->
            // Look up the mutable slot for this variable
            lookupMutableSlot sym.DisplayName >>= fun slotOpt ->
            match slotOpt with
            | Some (slotPtr, _elemType) ->
                // Return the pointer to the mutable slot
                emit (Value(slotPtr, "!llvm.ptr"))
            | None ->
                // Not a mutable variable - try to emit and take address
                // For now, just return an error for unsupported cases
                emit (Error (sprintf "AddressOf: '%s' is not a mutable variable with a stack slot" sym.DisplayName))
        | None ->
            emit (Error "AddressOf: inner expression has no symbol")
    | _ ->
        emit (Error "AddressOf: expected single child node")

/// Try to emit as address-of
let private tryEmitAddressOf : Emit<ExprResult option> =
    getFocus >>= fun node ->
    if isAddressOf node then
        handleAddressOf () |>> Some
    else
        emit None

// ===================================================================
// Struct Construction Handler (NativeStr etc.)
// ===================================================================

/// Handler for NativeStr struct construction: NativeStr(ptr, len)
/// This is fundamental lowering - struct construction becomes inline MLIR, not a function call
///
/// Per F# spec (expressions.md:1619-1652), NativeStr(ptr, len) is a standard
/// object construction expression that gets resolved to a constructor call.
/// We inline the struct construction here instead of emitting a function call.
let private tryEmitNativeStrConstruction : Emit<ExprResult option> =
    getZipper >>= fun z ->
    let node = z.Focus
    // Check if this is a function call that returns NativeStr
    if not (node.SyntaxKind.StartsWith("App")) then
        emit None
    else
        // Safely check if this is a NativeStr type
        let isNativeStr =
            match node.Type with
            | Some ftype when ftype.HasTypeDefinition ->
                try
                    ftype.TypeDefinition.FullName = "Alloy.NativeTypes.NativeString.NativeStr"
                with _ -> false
            | _ -> false
        if not isNativeStr then
            emit None
        else
            // Get children - PSG may store them as [Tuple; Ident:NativeStr] or [Ident; Tuple]
            let children = PSGZipper.childNodes z

            // Helper to check if a node is a NativeStr constructor ident
            let isNativeStrCtor (n: PSGNode) : bool =
                if not (n.SyntaxKind.StartsWith("Ident")) then false
                else
                    // Check 1: SyntaxKind contains "NativeStr" (e.g., "Ident:NativeStr")
                    let syntaxHasNativeStr = n.SyntaxKind.Contains("NativeStr")
                    // Check 2: Symbol is a constructor
                    let hasCtorSymbol =
                        match n.Symbol with
                        | Some sym -> sym.DisplayName = ".ctor" || sym.DisplayName.Contains("ctor")
                        | None -> false
                    // Either the syntax indicates NativeStr OR it's a ctor symbol
                    syntaxHasNativeStr || hasCtorSymbol

            // Helper to identify the constructor ident and arguments
            // Children are in source order: [funcExpr, argExpr] for App nodes
            let findCtorAndArgs (nodes: PSGNode list) : (PSGNode * PSGNode) option =
                match nodes with
                | [n1; n2] ->
                    // n1 should be the constructor ident, n2 should be the args tuple
                    if isNativeStrCtor n1 then Some (n1, n2)
                    // Handle curried case where ctor might be deeper in nested App
                    elif isNativeStrCtor n2 then Some (n2, n1)
                    else None
                | _ -> None

            match findCtorAndArgs children with
            | Some (_ctorNode, argNode) ->
                // Navigate to argNode and get its children (the tuple elements)
                atNode argNode (
                    getZipper >>= fun argZ ->
                    let argChildren = PSGZipper.childNodes argZ
                    match argChildren with
                    | [ptrNode; lenNode] ->
                        // Emit the pointer argument
                        atNode ptrNode callEmitExpr >>= fun ptrResult ->
                        match ptrResult with
                        | Value(ptrSsa, _) ->
                            // Emit the length argument
                            atNode lenNode callEmitExpr >>= fun lenResult ->
                            match lenResult with
                            | Value(lenSsa, lenType) ->
                                // Convert length to i64 if needed
                                let emitLen64 =
                                    if lenType = "i64" then emit lenSsa
                                    else
                                        freshSSAWithType "i64" >>= fun len64 ->
                                        line (sprintf "%s = arith.extsi %s : %s to i64" len64 lenSsa lenType) >>.
                                        emit len64
                                emitLen64 >>= fun len64Ssa ->
                                // Build the NativeStr struct
                                freshSSAWithType "!llvm.struct<(ptr, i64)>" >>= fun strStruct ->
                                line (sprintf "%s = llvm.mlir.undef : !llvm.struct<(ptr, i64)>" strStruct) >>.
                                freshSSAWithType "!llvm.struct<(ptr, i64)>" >>= fun strStruct1 ->
                                line (sprintf "%s = llvm.insertvalue %s, %s[0] : !llvm.struct<(ptr, i64)>" strStruct1 ptrSsa strStruct) >>.
                                freshSSAWithType "!llvm.struct<(ptr, i64)>" >>= fun strStruct2 ->
                                line (sprintf "%s = llvm.insertvalue %s, %s[1] : !llvm.struct<(ptr, i64)>" strStruct2 len64Ssa strStruct1) >>.
                                emit (Some (Value(strStruct2, "!llvm.struct<(ptr, i64)>")))
                            | _ -> emit None
                        | _ -> emit None
                    | _ -> emit None  // Wrong number of arguments
                )
            | None -> emit None  // Not a NativeStr constructor call

// ===================================================================
// Function Call Handler
// ===================================================================

/// Handler for user-defined function calls (not Alloy, not operators)
let private handleFunctionCall (info: FunctionCallInfo) : Emit<ExprResult> =
    // Get the function's MLIR name - use last two segments (Module_Function) to avoid collisions
    let funcName =
        let segments = info.FunctionName.Split('.')
        let shortName =
            if segments.Length >= 2 then
                sprintf "%s_%s" segments.[segments.Length - 2] segments.[segments.Length - 1]
            else
                segments.[segments.Length - 1]
        shortName
            .Replace(" ", "_")
            .Replace("``", "")  // Remove F# double-backtick escaping

    // Get the return type from the App node's Type (the result of the application)
    // NOT from the function identifier's type (which is a function type)
    let returnType =
        match info.AppNode.Type with
        | Some ftype -> fsharpTypeToMLIR ftype
        | None -> "i32"

    // Emit arguments
    let args = info.Arguments

    // For unit argument, emit no actual arguments
    let emitArgs : Emit<(string * string) list> =
        args
        |> List.filter (fun arg -> not (arg.SyntaxKind.StartsWith("Const:Unit") || arg.SyntaxKind = "Const:()"))
        |> traverse (fun argNode ->
            atNode argNode callEmitExpr >>= fun result ->
            match result with
            | Value(ssa, typ) -> emit (ssa, typ)
            | Void -> emit ("", "()")  // Unit value
            | Error msg -> emit ("", msg)  // Will be filtered out
        )
        |>> List.filter (fun (ssa, _) -> ssa <> "")

    emitArgs >>= fun argPairs ->

    // Build the function call
    if returnType = "()" then
        // Void return - no result SSA
        let argStr = argPairs |> List.map fst |> String.concat ", "
        let argTypeStr = argPairs |> List.map snd |> String.concat ", "
        if argPairs.IsEmpty then
            line (sprintf "func.call @%s() : () -> ()" funcName) >>.
            emit Void
        else
            line (sprintf "func.call @%s(%s) : (%s) -> ()" funcName argStr argTypeStr) >>.
            emit Void
    else
        // Has return value
        freshSSAWithType returnType >>= fun result ->
        let argStr = argPairs |> List.map fst |> String.concat ", "
        let argTypeStr = argPairs |> List.map snd |> String.concat ", "
        if argPairs.IsEmpty then
            line (sprintf "%s = func.call @%s() : () -> %s" result funcName returnType) >>.
            emit (Value(result, returnType))
        else
            line (sprintf "%s = func.call @%s(%s) : (%s) -> %s" result funcName argStr argTypeStr returnType) >>.
            emit (Value(result, returnType))

// ===================================================================
// Pipe Operator Handler
// ===================================================================

/// Handler for pipe expressions (value |> func)
/// Transforms value |> func into func(value) during emission
let private handlePipe (info: PipeInfo) : Emit<ExprResult> =
    // Check if the function is 'ignore' - if so, just emit the value for side effects
    let isIgnoreFunction =
        match info.Function.Symbol with
        | Some sym ->
            let displayName = sym.DisplayName
            let fullName = try sym.FullName with _ -> displayName
            displayName = "ignore" && (fullName.Contains("Alloy") || fullName.Contains("Core") || fullName.Contains("FSharp"))
        | None -> false

    if isIgnoreFunction then
        // For 'x |> ignore', just emit x for its side effects and return Void
        atNode info.Value callEmitExpr >>= fun _ ->
        emit Void
    else
        // First, emit the value expression
        atNode info.Value callEmitExpr >>= fun valueResult ->
        match valueResult with
        | Value(valueSsa, valueType) ->
            // Now emit the function application with the value as argument
            // The function node might be a partial application like String.format $"..."
            // or a simple function like Console.WriteLine
            atNode info.Function callEmitExpr >>= fun funcResult ->
            match funcResult with
            | Value(funcSsa, funcType) ->
                // If the function emitted a value, we need to apply the piped value to it
                // This handles partially applied functions like (String.format $"...")
                // For now, emit a comment and return the function result
                // TODO: Handle partial application properly
                emit funcResult
            | Void ->
                // The function returned void (like Console.WriteLine)
                // This is expected for functions that consume the value and produce side effects
                emit Void
            | Error msg -> emit (Error msg)
        | Void ->
            // The value is void (unit) - still apply to function
            atNode info.Function callEmitExpr
        | Error msg -> emit (Error msg)

/// Try to emit as pipe expression
let private tryEmitPipe : Emit<ExprResult option> =
    getZipper >>= fun z ->
    let node = z.Focus
    match extractPipe z.Graph node with
    | Some info -> handlePipe info |>> Some
    | None -> emit None

/// Check if a function call info represents a constructor call
let private isConstructorCall (info: FunctionCallInfo) : bool =
    info.FunctionName.Contains(".ctor") ||
    info.FunctionName.Contains("ctor") ||
    // Also check the symbol on the function node if available
    (info.FunctionNode.Symbol
     |> Option.map (fun sym -> sym.DisplayName = ".ctor" || sym.DisplayName.Contains("ctor"))
     |> Option.defaultValue false)

/// Try to emit as function call (for user-defined functions, not Alloy ops or operators)
let private tryEmitFunctionCall : Emit<ExprResult option> =
    getZipper >>= fun z ->
    let node = z.Focus

    // Only handle App nodes that aren't handled by other classifiers
    if not (node.SyntaxKind.StartsWith("App")) then
        emit None
    else
        // Check if this is a pipe expression - if so, let the pipe handler deal with it
        match extractPipe z.Graph node with
        | Some _ -> emit None  // Let pipe handler deal with it
        | None ->
            // Check if this is an Alloy operation - if so, let the Alloy handler deal with it
            match extractAlloyOp z.Graph node with
            | Some _ -> emit None  // Let Alloy handler deal with it
            | None ->
                // Check if this is an arithmetic or comparison operator
                match extractArithOp z.Graph node with
                | Some _ -> emit None  // Let arith handler deal with it
                | None ->
                    match extractCompareOp z.Graph node with
                    | Some _ -> emit None  // Let compare handler deal with it
                    | None ->
                        // This is a user-defined function call
                        match extractFunctionCall z.Graph node with
                        | Some info ->
                            // FAIL-FAST: Constructor calls should have been handled by struct construction handlers.
                            // If we reach here with a constructor, it means the struct lowering failed.
                            // Do NOT silently generate a broken function call - error out immediately.
                            if isConstructorCall info then
                                let typeName =
                                    match node.Type with
                                    | Some t when t.HasTypeDefinition -> t.TypeDefinition.FullName
                                    | _ -> "unknown type"
                                emit (Some (Error (sprintf "Unhandled struct constructor for '%s'. Constructor calls must be lowered to inline struct construction, not function calls. This indicates a missing pattern in tryEmitNativeStrConstruction or similar handler." typeName)))
                            else
                                handleFunctionCall info |>> Some
                        | None -> emit None

// ===================================================================
// Type Application Handler
// ===================================================================

/// Handler for TypeApp (generic instantiation like stackBuffer<byte>)
let private handleTypeApp () : Emit<ExprResult> =
    getFocus >>= fun node ->
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z
    match children with
    | [innerNode] ->
        atNode innerNode callEmitExpr
    | [] ->
        match node.Symbol with
        | Some symbol ->
            let name = symbol.DisplayName
            lookupLocal name >>= fun localOpt ->
            match localOpt with
            | Some ssaName ->
                lookupLocalType name |>> fun typeOpt ->
                Value(ssaName, typeOpt |> Option.defaultValue "i32")
            | None ->
                emit (Value(sprintf "@%s" name, "func"))
        | None ->
            emit (Error "TypeApp without symbol or children")
    | _ ->
        atNode (List.head children) callEmitExpr

/// Try to emit as type application
let private tryEmitTypeApp : Emit<ExprResult option> =
    getFocus >>= fun node ->
    if isTypeApp node then
        handleTypeApp () |>> Some
    else
        emit None

// ===================================================================
// Pattern Node Handler
// ===================================================================

/// Try to emit as pattern (structural, returns Void)
let private tryEmitPattern : Emit<ExprResult option> =
    getFocus >>= fun node ->
    if isPattern node then
        emit (Some Void)
    else
        emit None

// ===================================================================
// PropertyAccess Handler (receiver.Property)
// ===================================================================

/// Handler for PropertyAccess nodes like receiver.Length
let private handlePropertyAccess (propertyName: string) : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z

    // The child is the receiver (e.g., the string variable)
    match children with
    | receiverNode :: _ ->
        atNode receiverNode callEmitExpr >>= fun receiverResult ->
        match receiverResult with
        | Value(receiverSsa, receiverType) ->
            match propertyName with
            | "Length" ->
                // For NativeStr (struct with ptr and length), extract the length field
                if receiverType = "!llvm.struct<(ptr, i64)>" then
                    // Extract i64 length from struct field [1]
                    freshSSAWithType "i64" >>= fun lenI64 ->
                    line (sprintf "%s = llvm.extractvalue %s[1] : !llvm.struct<(ptr, i64)>" lenI64 receiverSsa) >>.
                    // NativeStr.Length is int (i32), so truncate
                    freshSSAWithType "i32" >>= fun lenI32 ->
                    line (sprintf "%s = arith.trunci %s : i64 to i32" lenI32 lenI64) >>.
                    emit (Value(lenI32, "i32"))
                else
                    // For other types (plain pointers, etc.), emit a placeholder
                    line (sprintf "// PropertyAccess:Length on %s (%s) - unsupported type" receiverSsa receiverType) >>.
                    emitI32 0 >>= fun lenReg ->
                    emit (Value(lenReg, "i32"))

            | "Pointer" ->
                // For NativeStr, extract the pointer field
                if receiverType = "!llvm.struct<(ptr, i64)>" then
                    freshSSAWithType "!llvm.ptr" >>= fun ptrReg ->
                    line (sprintf "%s = llvm.extractvalue %s[0] : !llvm.struct<(ptr, i64)>" ptrReg receiverSsa) >>.
                    emit (Value(ptrReg, "!llvm.ptr"))
                else
                    // Already a pointer, return as-is
                    emit (Value(receiverSsa, "!llvm.ptr"))

            | _ ->
                line (sprintf "// Unhandled PropertyAccess:%s on %s" propertyName receiverSsa) >>.
                emit (Error (sprintf "Unhandled property: %s" propertyName))
        | Error msg ->
            emit (Error (sprintf "PropertyAccess receiver error: %s" msg))
        | Void ->
            emit (Error "PropertyAccess receiver returned Void")
    | [] ->
        emit (Error "PropertyAccess has no receiver")

/// Try to emit as PropertyAccess
let private tryEmitPropertyAccess : Emit<ExprResult option> =
    getFocus >>= fun node ->
    if isPropertyAccess node then
        match extractPropertyName node with
        | Some propName -> handlePropertyAccess propName |>> Some
        | None -> emit (Some (Error "PropertyAccess without property name"))
    else
        emit None

// ===================================================================
// TraitCall Handler (SRTP resolution)
// ===================================================================

/// Handler for SRTP TraitCall nodes
/// These resolve statically to concrete member implementations based on the type
let private handleTraitCall (memberName: string) : Emit<ExprResult> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z

    // The child is the argument (e.g., the buffer variable)
    match children with
    | argNode :: _ ->
        atNode argNode callEmitExpr >>= fun argResult ->
        match argResult with
        | Value(argSsa, argType) ->
            match memberName with
            | "Pointer" ->
                // TraitCall:Pointer on StackBuffer returns the buffer pointer directly
                // StackBuffer IS the pointer (stack-allocated array)
                emit (Value(argSsa, "!llvm.ptr"))

            | "Length" ->
                // TraitCall:Length on StackBuffer returns the allocated size
                // For stack buffers, we need to track this - for now, use a constant
                // TODO: Track buffer sizes in emission state
                // The length should come from the stackBuffer<T> size argument
                // For now, emit a comment and return a placeholder
                line (sprintf "// TraitCall:Length on %s - size lookup needed" argSsa) >>.
                // Try to look up the size from the local type info
                // If argType contains size info, extract it; otherwise default
                emitI32 256 >>= fun sizeReg ->  // Default for StackBuffer
                emit (Value(sizeReg, "i32"))

            | "ToString" ->
                // TraitCall:ToString - convert to string representation
                emit (Value(argSsa, "!llvm.ptr"))

            | _ ->
                line (sprintf "// Unhandled TraitCall:%s on %s" memberName argSsa) >>.
                emit (Value(argSsa, argType))
        | Error msg ->
            emit (Error (sprintf "TraitCall argument error: %s" msg))
        | Void ->
            emit (Error "TraitCall argument returned Void")
    | [] ->
        emit (Error "TraitCall has no arguments")

/// Try to emit as TraitCall
let private tryEmitTraitCall : Emit<ExprResult option> =
    getFocus >>= fun node ->
    if node.SyntaxKind.StartsWith("TraitCall:") then
        // Extract member name from "TraitCall:MemberName"
        let memberName = node.SyntaxKind.Substring(10)  // Skip "TraitCall:"
        handleTraitCall memberName |>> Some
    elif node.SyntaxKind = "TraitCall" then
        // Generic TraitCall without member name - just pass through argument
        handleTraitCall "Unknown" |>> Some
    else
        emit None

// ===================================================================
// Main Expression Dispatcher
// ===================================================================

/// Main expression emission using tryEmissions for dispatch
let emitExpr : Emit<ExprResult> =
    tryEmissions [
        // Pipe operator (highest priority - transforms value |> func to func(value))
        tryEmitPipe

        // Console operations (highest priority for Alloy calls)
        tryEmitConsole

        // Time operations
        tryEmitTime

        // Memory operations (stackBuffer, spanToString, etc.)
        tryEmitMemory

        // Result DU constructors (Ok, Error)
        tryEmitResult

        // Core operations (ignore, etc.)
        tryEmitCore

        // NativePtr operations (get, set, read, write, add)
        tryEmitNativePtr

        // SRTP TraitCall resolution (before function calls)
        tryEmitTraitCall

        // PropertyAccess (receiver.Property like string.Length)
        tryEmitPropertyAccess

        // Arithmetic operations
        tryEmitArith

        // Comparison operations
        tryEmitCompare

        // Type conversion operations (byte, int, int64, etc.)
        tryEmitConversion

        // NativeStr struct construction (before function calls)
        tryEmitNativeStrConstruction

        // User-defined function calls (after Alloy and operators)
        tryEmitFunctionCall

        // Structural patterns
        tryEmitConst
        tryEmitIdent
        tryEmitSequential
        tryEmitLet
        tryEmitIf
        tryEmitWhile
        tryEmitMatch
        tryEmitTypeApp
        tryEmitMutableSet
        tryEmitAddressOf
        tryEmitPattern
    ] >>= fun resultOpt ->
    match resultOpt with
    | Some result -> emit result
    | None ->
        // No handler matched - this is a compiler error
        getFocus >>= fun node ->
        emit (Error (sprintf "Unhandled PSG node type: %s" node.SyntaxKind))

// Initialize the mutable reference for recursive calls
do emitExprImpl <- emitExpr

// ===================================================================
// Public API
// ===================================================================

/// Emit expression at current zipper focus (public alias)
let emitCurrentExpr : Emit<ExprResult> = emitExpr

/// Legacy emitExpr that takes explicit PSG and node
let emitExprLegacy (psg: ProgramSemanticGraph) (node: PSGNode) : Emit<ExprResult> =
    atNode node emitExpr
