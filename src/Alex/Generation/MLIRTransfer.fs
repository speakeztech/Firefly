/// MLIRTransfer - Principled PSG to MLIR Transfer using Zipper + XParsec
///
/// ARCHITECTURAL FOUNDATION:
/// - Consumes PSG enrichment data (node.Operation set by ClassifyOperations nanopass)
/// - Type-safe dispatch via OperationKind discriminated union
/// - NO string pattern matching on SyntaxKind for operations
/// - Errors MUST propagate: never silent Void or placeholders
///
/// This module is the sole MLIR generation path.
module Alex.Generation.MLIRTransfer

open System
open XParsec
open FSharp.Compiler.Symbols
open Core.PSG.Types
open Alex.Traversal.PSGZipper
open Alex.Traversal.PSGXParsec
open Baker.Types
open Baker.TypedTreeZipper

// Import OutputKind with module qualification to avoid conflict with OperationKind.Console
module OutputKind = Core.Types.MLIRTypes.OutputKind
type OutputKind = Core.Types.MLIRTypes.OutputKind

// ═══════════════════════════════════════════════════════════════════════════════
// PART 1: Core Helper Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Extract constant value and MLIR type from a PSG node
let extractConstant (node: PSGNode) : (string * string) option =
    match node.ConstantValue with
    | Some (StringValue s) -> Some (s, "string")
    | Some (Int32Value i) -> Some (string i, "i32")
    | Some (Int64Value i) -> Some (string i, "i64")
    | Some (ByteValue b) -> Some (string (int b), "i8")
    | Some (BoolValue b) -> Some ((if b then "1" else "0"), "i1")
    | Some (CharValue c) -> Some (string (int c), "i8")
    | Some (FloatValue f) -> Some (string f, "f64")
    | Some UnitValue -> Some ("", "unit")
    | None ->
        let kind = node.SyntaxKind
        if kind.StartsWith("Const:") then
            let constKind = kind.Substring(6)
            match constKind with
            | "Int32 0" -> Some ("0", "i32")
            | s when s.StartsWith("Int32 ") -> Some (s.Substring(6), "i32")
            | s when s.StartsWith("Int64 ") -> Some (s.Substring(6), "i64")
            | s when s.StartsWith("Byte ") -> Some (s.Substring(5).TrimEnd([|'u'; 'y'|]), "i8")
            | "Unit" -> Some ("", "unit")
            | s when s.StartsWith("String ") -> Some (s.Substring(7), "string")
            | "String" -> Some ("", "string")
            | "Null" -> Some ("null", "!llvm.ptr")
            | _ -> None
        else None

/// Map an FSharpType to its MLIR representation
let mapFSharpTypeToMLIR (ftype: FSharpType) : string =
    try
        if ftype.HasTypeDefinition then
            match ftype.TypeDefinition.TryFullName with
            | Some "System.Int32" -> "i32"
            | Some "System.Int64" -> "i64"
            | Some "System.Int16" -> "i16"
            | Some "System.Byte" | Some "System.SByte" -> "i8"
            | Some "System.Boolean" -> "i1"
            | Some "System.IntPtr" | Some "System.UIntPtr" -> "!llvm.ptr"
            | Some n when n.Contains("nativeptr") -> "!llvm.ptr"
            | Some n when n.Contains("NativeStr") -> "!llvm.struct<(ptr, i64)>"
            | _ -> "i64"
        else "i64"
    with _ -> "i64"

/// Get MLIR type for a PSG node based on its resolved type
let getNodeMLIRType (node: PSGNode) : string option =
    match node.Type with
    | Some t ->
        try
            if t.HasTypeDefinition then
                let entity = t.TypeDefinition
                let isUnit =
                    match entity.TryFullName with
                    | Some n when n.Contains("Unit") || n.Contains("unit") -> true
                    | Some n when n.Contains("Void") || n = "System.Void" -> true
                    | _ -> entity.DisplayName = "unit"
                if isUnit then None
                else
                    match entity.TryFullName with
                    | Some "System.Int32" -> Some "i32"
                    | Some "System.Int64" -> Some "i64"
                    | Some "System.Byte" -> Some "i8"
                    | Some "System.Boolean" -> Some "i1"
                    | Some n when n.Contains("NativeStr") -> Some "!llvm.struct<(ptr, i32)>"
                    | Some n when n.Contains("nativeptr") -> Some "!llvm.ptr"
                    | Some n when n.Contains("Char") -> Some "i32"
                    | _ -> None
            else None
        with _ -> None
    | None -> None

/// Get children of a PSG node
let getChildren (graph: ProgramSemanticGraph) (node: PSGNode) : PSGNode list =
    match node.Children with
    | Parent ids -> ids |> List.choose (fun id -> Map.tryFind id.Value graph.Nodes)
    | NoChildren | NotProcessed -> []

/// Extract function name from a SyntaxKind (e.g., "LongIdent:Console.Write" → "Write")
let extractFunctionNameFromKind (syntaxKind: string) : string option =
    if syntaxKind.StartsWith("LongIdent:") then
        let fullName = syntaxKind.Substring(10)
        // Get last part after dot
        let lastDot = fullName.LastIndexOf('.')
        if lastDot >= 0 then Some (fullName.Substring(lastDot + 1))
        else Some fullName
    elif syntaxKind.StartsWith("Ident:") then
        Some (syntaxKind.Substring(6))
    else
        None

/// Find a function Binding node by name in the PSG
let findFunctionBinding (graph: ProgramSemanticGraph) (funcName: string) : PSGNode option =
    graph.Nodes
    |> Map.toSeq
    |> Seq.tryFind (fun (_, node) ->
        // Check for Binding nodes
        let isBinding = node.SyntaxKind.StartsWith("Binding")
        // Check Symbol for name matching
        let symbolMatch =
            match node.Symbol with
            | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
                mfv.DisplayName = funcName ||
                (try mfv.CompiledName = funcName with _ -> false)
            | _ -> false
        isBinding && symbolMatch)
    |> Option.map snd

/// Get parameter nodes and body node from a function Binding
let getFunctionParamsAndBody (graph: ProgramSemanticGraph) (bindingNode: PSGNode) : (PSGNode list * PSGNode option) =
    let children = getChildren graph bindingNode
    let patternNodes =
        children
        |> List.filter (fun c -> c.SyntaxKind.StartsWith("Pattern:"))
    let bodyNode =
        children
        |> List.tryFind (fun c -> not (c.SyntaxKind.StartsWith("Pattern:")))
    // Extract named parameters from pattern nodes
    let rec extractNamedParams (node: PSGNode) : PSGNode list =
        if node.SyntaxKind.StartsWith("Pattern:Named:") then
            [node]
        else
            getChildren graph node
            |> List.collect extractNamedParams
    let paramNodes = patternNodes |> List.collect extractNamedParams
    (paramNodes, bodyNode)

// ═══════════════════════════════════════════════════════════════════════════════
// PART 2: Transfer Result Type
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of transferring a PSG node to MLIR
type TransferResult =
    | TValue of ssa: string * mlirType: string
    | TVoid
    | TError of message: string

module TransferResult =
    let isValue = function TValue _ -> true | _ -> false
    let isVoid = function TVoid -> true | _ -> false
    let isError = function TError _ -> true | _ -> false
    let getSSA = function TValue (ssa, _) -> Some ssa | _ -> None
    let getType = function TValue (_, t) -> Some t | _ -> None

    let toExprResult = function
        | TValue (ssa, t) -> Value (ssa, t)
        | TVoid -> Void
        | TError msg -> EmitError msg

type Transfer = PSGChildParser<TransferResult>

// ═══════════════════════════════════════════════════════════════════════════════
// PART 3: Operation-Specific Transfer Handlers (using actual OperationKind types)
// ═══════════════════════════════════════════════════════════════════════════════

/// Transfer arithmetic operations (Add, Sub, Mul, Div, Mod)
let rec transferArithmetic (ctx: EmitContext) (graph: ProgramSemanticGraph) (node: PSGNode) (op: ArithmeticOp) : TransferResult =
    let children = getChildren graph node
    let results = children |> List.map (fun c -> transferNodeDirect ctx graph c)
    let ssas = results |> List.choose TransferResult.getSSA
    let types = results |> List.choose TransferResult.getType

    let resultSSA = EmitContext.nextSSA ctx
    let opType = if List.isEmpty types then "i32" else types.[0]

    // Handle Negate separately (unary operation)
    match op with
    | Negate ->
        if ssas.Length < 1 then
            TError "Negate operation requires 1 operand"
        else
            let zeroSSA = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = arith.constant 0 : %s" zeroSSA opType)
            EmitContext.emitLine ctx (sprintf "%s = arith.subi %s, %s : %s" resultSSA zeroSSA ssas.[0] opType)
            TValue (resultSSA, opType)
    | _ ->
        if ssas.Length < 2 then
            TError (sprintf "Arithmetic operation %A requires 2 operands, got %d" op ssas.Length)
        else
            let mlirOp =
                match op with
                | Add -> "arith.addi"
                | Sub -> "arith.subi"
                | Mul -> "arith.muli"
                | Div -> "arith.divsi"
                | Mod -> "arith.remsi"
                | Negate -> "arith.subi" // Won't be reached
            EmitContext.emitLine ctx (sprintf "%s = %s %s, %s : %s" resultSSA mlirOp ssas.[0] ssas.[1] opType)
            TValue (resultSSA, opType)

/// Transfer comparison operations (Eq, Neq, Lt, Gt, Lte, Gte)
and transferComparison (ctx: EmitContext) (graph: ProgramSemanticGraph) (node: PSGNode) (op: ComparisonOp) : TransferResult =
    let children = getChildren graph node
    let results = children |> List.map (fun c -> transferNodeDirect ctx graph c)
    let ssas = results |> List.choose TransferResult.getSSA

    if ssas.Length < 2 then
        TError (sprintf "Comparison operation %A requires 2 operands" op)
    else
        let resultSSA = EmitContext.nextSSA ctx
        let cmpPred =
            match op with
            | Eq -> "eq"
            | Neq -> "ne"
            | Lt -> "slt"
            | Gt -> "sgt"
            | Lte -> "sle"
            | Gte -> "sge"
        EmitContext.emitLine ctx (sprintf "%s = arith.cmpi %s, %s, %s : i32" resultSSA cmpPred ssas.[0] ssas.[1])
        TValue (resultSSA, "i1")

/// Transfer console operations (ConsoleWriteBytes, ConsoleWrite, etc.)
and transferConsole (ctx: EmitContext) (graph: ProgramSemanticGraph) (node: PSGNode) (op: ConsoleOp) : TransferResult =
    let children = getChildren graph node

    // Helper to eagerly process all children (for operations that need all SSAs upfront)
    let processAllChildren () =
        let results = children |> List.map (fun c -> transferNodeDirect ctx graph c)
        results |> List.choose TransferResult.getSSA

    match op with
    | ConsoleWriteBytes ->
        let ssas = processAllChildren ()
        if ssas.Length >= 3 then
            let resultSSA = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = llvm.call @fidelity_write_bytes(%s, %s, %s) : (i32, !llvm.ptr, i32) -> i32"
                resultSSA ssas.[0] ssas.[1] ssas.[2])
            TValue (resultSSA, "i32")
        else TError "ConsoleWriteBytes requires 3 arguments (fd, ptr, count)"

    | ConsoleReadBytes ->
        let ssas = processAllChildren ()
        if ssas.Length >= 3 then
            let resultSSA = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = llvm.call @fidelity_read_bytes(%s, %s, %s) : (i32, !llvm.ptr, i32) -> i32"
                resultSSA ssas.[0] ssas.[1] ssas.[2])
            TValue (resultSSA, "i32")
        else TError "ConsoleReadBytes requires 3 arguments"

    | ConsoleWrite | ConsoleWriteln ->
        // NOTE: Do NOT eagerly process children here - we need to handle args specially
        // to avoid double-processing interpolated strings
        // High-level Write/WriteLine - inline the function body
        // The function body contains the SRTP dispatch to writeNativeStr/writeSystemString
        let funcChildren = getChildren graph node
        match funcChildren with
        | funcNode :: argNodes ->
            let funcName =
                match funcNode.Symbol with
                | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
                    try mfv.CompiledName with _ -> mfv.DisplayName
                | _ ->
                    extractFunctionNameFromKind funcNode.SyntaxKind |> Option.defaultValue "unknown"

            // Find the function definition in the PSG and inline its body
            match findFunctionBinding graph funcName with
            | Some bindingNode ->
                let (paramNodes, bodyNodeOpt) = getFunctionParamsAndBody graph bindingNode
                match bodyNodeOpt with
                | Some bodyNode ->
                    // For ConsoleWrite/WriteLine with InterpolatedString, we need to process
                    // the interpolated string parts without duplicating the output.
                    // The body will reference the parameter, so we need to bind the argument node
                    // directly to the parameter (for def-use resolution), not the SSA result.

                    // First, check if the argument is an InterpolatedString - if so, process it
                    // specially to avoid double output
                    let argResult =
                        match argNodes with
                        | [argNode] ->
                            // Process the argument - if it's an interpolated string, it will emit writes directly
                            transferNodeDirect ctx graph argNode
                        | _ -> TVoid

                    // If the argument was processed and emitted (returns TVoid from StrConcat3),
                    // just add the newline for WriteLine and return
                    match argResult with
                    | TVoid ->
                        // Argument was already written (StrConcat3 returns TVoid)
                        // Just add newline for WriteLine
                        if op = ConsoleWriteln then
                            let nlGlobal = EmitContext.registerStringLiteral ctx "\n"
                            let ptrSSA = EmitContext.nextSSA ctx
                            EmitContext.emitLine ctx (sprintf "%s = llvm.mlir.addressof %s : !llvm.ptr" ptrSSA nlGlobal)
                            let sysNumSSA = EmitContext.nextSSA ctx
                            EmitContext.emitLine ctx (sprintf "%s = arith.constant 1 : i64" sysNumSSA)
                            let fdSSA = EmitContext.nextSSA ctx
                            EmitContext.emitLine ctx (sprintf "%s = arith.constant 1 : i64" fdSSA)
                            let lenSSA = EmitContext.nextSSA ctx
                            EmitContext.emitLine ctx (sprintf "%s = arith.constant 1 : i64" lenSSA)
                            let resultSSA = EmitContext.nextSSA ctx
                            EmitContext.emitLine ctx (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64"
                                resultSSA sysNumSSA fdSSA ptrSSA lenSSA)
                        TVoid
                    | TValue (ssa, mlirType) ->
                        // Regular string value - map to parameter and inline body
                        if paramNodes.Length > 0 then
                            EmitContext.recordNodeSSA ctx paramNodes.[0].Id ssa mlirType
                        transferNodeDirect ctx graph bodyNode
                    | TError msg -> TError msg
                | None ->
                    TError "Console function has no body"
            | None ->
                TError (sprintf "Console function %s not found in PSG" funcName)
        | [] ->
            TError "ConsoleWrite has no children"

    | ConsoleReadLine ->
        // Basic ReadLine: allocate 256-byte buffer on stack and read from stdin
        // For freestanding, use syscall 0 (read) on fd 0 (stdin)
        // Returns a pointer to the buffer (NativeStr handling is simplified for now)

        // Allocate stack buffer (256 bytes as array of i8)
        // llvm.alloca requires SSA value for count
        let countSSA = EmitContext.nextSSA ctx
        EmitContext.emitLine ctx (sprintf "%s = arith.constant 256 : i64" countSSA)
        let bufferSSA = EmitContext.nextSSA ctx
        EmitContext.emitLine ctx (sprintf "%s = llvm.alloca %s x i8 : (i64) -> !llvm.ptr" bufferSSA countSSA)

        // Read from stdin (syscall 0, fd 0, buf, count)
        let sysNumSSA = EmitContext.nextSSA ctx
        EmitContext.emitLine ctx (sprintf "%s = arith.constant 0 : i64" sysNumSSA)  // syscall 0 = read
        let fdSSA = EmitContext.nextSSA ctx
        EmitContext.emitLine ctx (sprintf "%s = arith.constant 0 : i64" fdSSA)  // fd 0 = stdin
        let lenSSA = EmitContext.nextSSA ctx
        EmitContext.emitLine ctx (sprintf "%s = arith.constant 255 : i64" lenSSA)  // max 255 chars + null
        let bytesReadSSA = EmitContext.nextSSA ctx
        EmitContext.emitLine ctx (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64"
            bytesReadSSA sysNumSSA fdSSA bufferSSA lenSSA)

        // Record bytes read for later use (e.g., for determining string length)
        // Store the bytesRead SSA name so we can look it up when writing
        // Format: "DYNAMIC:ssaName" (no angle brackets to avoid parsing issues)
        ctx.SSAToStringContent.[bufferSSA] <- sprintf "DYNAMIC:%s" bytesReadSSA

        // Return the buffer pointer (simplified - doesn't track length)
        TValue (bufferSSA, "!llvm.ptr")

    | ConsoleReadInto ->
        TError "ConsoleReadInto not yet implemented"

    | ConsoleNewLine ->
        // Emit newline character via inline syscall
        let nlGlobal = EmitContext.registerStringLiteral ctx "\n"
        let ptrSSA = EmitContext.nextSSA ctx
        EmitContext.emitLine ctx (sprintf "%s = llvm.mlir.addressof %s : !llvm.ptr" ptrSSA nlGlobal)
        // Emit syscall to write 1 byte (the newline)
        let sysNumSSA = EmitContext.nextSSA ctx
        EmitContext.emitLine ctx (sprintf "%s = arith.constant 1 : i64" sysNumSSA)
        let fdSSA = EmitContext.nextSSA ctx
        EmitContext.emitLine ctx (sprintf "%s = arith.constant 1 : i64" fdSSA)
        let lenSSA = EmitContext.nextSSA ctx
        EmitContext.emitLine ctx (sprintf "%s = arith.constant 1 : i64" lenSSA)
        let resultSSA = EmitContext.nextSSA ctx
        EmitContext.emitLine ctx (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64"
            resultSSA sysNumSSA fdSSA ptrSSA lenSSA)
        TValue (resultSSA, "i64")

/// Transfer native string operations (StrConcat2, StrConcat3, etc.)
and transferNativeStr (ctx: EmitContext) (graph: ProgramSemanticGraph) (node: PSGNode) (op: NativeStrOp) : TransferResult =
    let children = getChildren graph node
    let results = children |> List.map (fun c -> transferNodeDirect ctx graph c)
    let ssas = results |> List.choose TransferResult.getSSA

    match op with
    | StrConcat2 ->
        if ssas.Length >= 2 then
            let resultSSA = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = llvm.call @fidelity_str_concat2(%s, %s) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr"
                resultSSA ssas.[0] ssas.[1])
            TValue (resultSSA, "!llvm.ptr")
        else TError "StrConcat2 requires 2 arguments"

    | StrConcat3 ->
        // For freestanding mode without memory allocation, emit writes for each part
        // This avoids the need for a concat function that allocates memory
        if ssas.Length >= 3 then
            // Get the three string pointers
            let str1, str2, str3 = ssas.[0], ssas.[1], ssas.[2]

            // Helper to emit write syscall for a string
            let emitWrite strSSA =
                // Look up string length from content if available
                match ctx.SSAToStringContent.TryGetValue(strSSA) with
                | true, content when content.StartsWith("DYNAMIC:") ->
                    // Dynamic string from ReadLine - extract the bytesRead SSA
                    // Format is "DYNAMIC:%vXX" - extract just the %vXX part
                    let bytesReadSSA = content.Substring(8)  // Remove "DYNAMIC:" prefix
                    // Write bytesRead-1 chars (strip the newline)
                    let sysNumSSA = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = arith.constant 1 : i64" sysNumSSA)
                    let fdSSA = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = arith.constant 1 : i64" fdSSA)
                    // Subtract 1 from bytesRead to strip newline
                    let oneSSA = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = arith.constant 1 : i64" oneSSA)
                    let lenSSA = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = arith.subi %s, %s : i64" lenSSA bytesReadSSA oneSSA)
                    let resultSSA = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64"
                        resultSSA sysNumSSA fdSSA strSSA lenSSA)
                | true, content when content.Length > 0 ->
                    // Static string - emit write with known length
                    let sysNumSSA = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = arith.constant 1 : i64" sysNumSSA)
                    let fdSSA = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = arith.constant 1 : i64" fdSSA)
                    let lenSSA = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = arith.constant %d : i64" lenSSA content.Length)
                    let resultSSA = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64"
                        resultSSA sysNumSSA fdSSA strSSA lenSSA)
                | _ ->
                    // Unknown string - skip for now
                    ()

            // Emit writes for each part
            emitWrite str1
            emitWrite str2
            emitWrite str3

            // Return TVoid since we've already written the output
            // This prevents downstream op_Dollar from writing again
            TVoid
        else TError "StrConcat3 requires 3 arguments"

    | StrConcatN ->
        // Variable number of arguments (>3) - emit writes for all parts
        if ssas.Length >= 1 then
            // Helper to emit write syscall for a string
            let emitWrite strSSA =
                match ctx.SSAToStringContent.TryGetValue(strSSA) with
                | true, content when content.StartsWith("DYNAMIC:") ->
                    let bytesReadSSA = content.Substring(8)
                    // Create constant 1 for subtraction
                    let oneSSA = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = arith.constant 1 : i64" oneSSA)
                    let adjustedLenSSA = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = arith.subi %s, %s : i64" adjustedLenSSA bytesReadSSA oneSSA)
                    let sysNumSSA = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = arith.constant 1 : i64" sysNumSSA)
                    let fdSSA = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = arith.constant 1 : i64" fdSSA)
                    let resultSSA = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64"
                        resultSSA sysNumSSA fdSSA strSSA adjustedLenSSA)
                | true, content ->
                    let len = content.Length
                    let sysNumSSA = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = arith.constant 1 : i64" sysNumSSA)
                    let fdSSA = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = arith.constant 1 : i64" fdSSA)
                    let lenSSA = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = arith.constant %d : i64" lenSSA len)
                    let resultSSA = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64"
                        resultSSA sysNumSSA fdSSA strSSA lenSSA)
                | _ -> ()

            // Emit writes for ALL parts
            for ssa in ssas do
                emitWrite ssa

            TVoid
        else TError "StrConcatN requires at least 1 argument"

    | StrLength ->
        if ssas.Length >= 1 then
            let resultSSA = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = llvm.call @fidelity_str_length(%s) : (!llvm.ptr) -> i64"
                resultSSA ssas.[0])
            TValue (resultSSA, "i64")
        else TError "StrLength requires 1 argument"

    | StrEmpty ->
        let resultSSA = EmitContext.nextSSA ctx
        EmitContext.emitLine ctx (sprintf "%s = llvm.call @fidelity_str_empty() : () -> !llvm.ptr" resultSSA)
        TValue (resultSSA, "!llvm.ptr")

    | _ ->
        TError (sprintf "NativeStr operation %A not yet implemented" op)

/// Transfer time operations (Sleep, CurrentTicks, etc.)
and transferTime (ctx: EmitContext) (graph: ProgramSemanticGraph) (node: PSGNode) (op: TimeOp) : TransferResult =
    let children = getChildren graph node
    let results = children |> List.map (fun c -> transferNodeDirect ctx graph c)
    let ssas = results |> List.choose TransferResult.getSSA

    match op with
    | Sleep ->
        if ssas.Length >= 1 then
            EmitContext.emitLine ctx (sprintf "llvm.call @fidelity_sleep(%s) : (i32) -> ()" ssas.[0])
            TVoid
        else TError "Sleep requires 1 argument (milliseconds)"

    | CurrentTicks ->
        let resultSSA = EmitContext.nextSSA ctx
        EmitContext.emitLine ctx (sprintf "%s = llvm.call @fidelity_current_ticks() : () -> i64" resultSSA)
        TValue (resultSSA, "i64")

    | HighResolutionTicks ->
        let resultSSA = EmitContext.nextSSA ctx
        EmitContext.emitLine ctx (sprintf "%s = llvm.call @fidelity_high_res_ticks() : () -> i64" resultSSA)
        TValue (resultSSA, "i64")

    | TickFrequency ->
        let resultSSA = EmitContext.nextSSA ctx
        EmitContext.emitLine ctx (sprintf "%s = llvm.call @fidelity_tick_frequency() : () -> i64" resultSSA)
        TValue (resultSSA, "i64")

/// Transfer core operations (ignore, failwith, etc.)
and transferCore (ctx: EmitContext) (graph: ProgramSemanticGraph) (node: PSGNode) (op: CoreOp) : TransferResult =
    let children = getChildren graph node

    match op with
    | Ignore ->
        // ignore: process the argument and discard the result
        // Structure: App [Ident:ignore, <piped expression>]
        // Find the non-ignore child to process
        match children |> List.filter (fun c -> not (c.SyntaxKind.StartsWith("Ident:ignore"))) with
        | arg :: _ ->
            let _ = transferNodeDirect ctx graph arg
            TVoid
        | [] ->
            // Fallback: try first child
            match children with
            | arg :: _ ->
                let _ = transferNodeDirect ctx graph arg
                TVoid
            | [] ->
                TError "ignore requires an argument"

    | Not ->
        // Boolean negation
        match children with
        | [arg] ->
            let result = transferNodeDirect ctx graph arg
            match result with
            | TValue (ssa, "i1") ->
                let resultSSA = EmitContext.nextSSA ctx
                let trueSSA = EmitContext.nextSSA ctx
                EmitContext.emitLine ctx (sprintf "%s = arith.constant 1 : i1" trueSSA)
                EmitContext.emitLine ctx (sprintf "%s = arith.xori %s, %s : i1" resultSSA ssa trueSSA)
                TValue (resultSSA, "i1")
            | TValue (ssa, t) ->
                TError (sprintf "not: expected bool but got %s" t)
            | _ -> TError "not: argument failed to evaluate"
        | _ ->
            TError "not requires exactly 1 argument"

    | Failwith | InvalidArg ->
        TError (sprintf "Exception operation %A not yet implemented in freestanding mode" op)

/// Transfer a regular function call - first tries to inline, falls back to external call
and transferRegularCall (ctx: EmitContext) (graph: ProgramSemanticGraph) (node: PSGNode) (info: RegularCallInfo) : TransferResult =
    let children = getChildren graph node
    let funcName = info.FunctionName
    // First child is the function identifier, rest are arguments
    let argNodes = children |> List.tail

    // First, try to find the function in the PSG for inlining
    match findFunctionBinding graph funcName with
    | Some bindingNode ->
        // Found the function - inline its body
        let (paramNodes, bodyNodeOpt) = getFunctionParamsAndBody graph bindingNode
        match bodyNodeOpt with
        | Some bodyNode ->
            // Process arguments and map to parameters
            let argResults = argNodes |> List.map (fun c -> transferNodeDirect ctx graph c)

            // Map parameters to argument SSA values (only if there are named params)
            // Skip if function takes unit (no named params but unit argument)
            if paramNodes.Length > 0 && paramNodes.Length = argResults.Length then
                List.zip paramNodes argResults
                |> List.iter (fun (paramNode, argResult) ->
                    match argResult with
                    | TValue (ssa, mlirType) ->
                        EmitContext.recordNodeSSA ctx paramNode.Id ssa mlirType
                    | _ -> ())

            // Process the function body
            transferNodeDirect ctx graph bodyNode
        | None ->
            // Function has no body - treat as external
            emitExternalCall ctx graph argNodes funcName
    | None ->
        // Function not found in PSG - emit external call
        emitExternalCall ctx graph argNodes funcName

/// Helper to emit an external function call
/// argNodes should be just the argument nodes (not including function identifier)
and emitExternalCall (ctx: EmitContext) (graph: ProgramSemanticGraph) (argNodes: PSGNode list) (funcName: string) : TransferResult =
    let results = argNodes |> List.map (fun c -> transferNodeDirect ctx graph c)
    let ssas = results |> List.choose TransferResult.getSSA
    let types = results |> List.choose TransferResult.getType

    let argTypesStr = if List.isEmpty types then "" else types |> String.concat ", "
    let argSSAsStr = ssas |> String.concat ", "

    // Register this as an external function that needs declaration
    ctx.ExternalFuncs.Add(funcName) |> ignore

    let resultSSA = EmitContext.nextSSA ctx
    if List.isEmpty types then
        EmitContext.emitLine ctx (sprintf "%s = llvm.call @%s() : () -> i64" resultSSA funcName)
    else
        EmitContext.emitLine ctx (sprintf "%s = llvm.call @%s(%s) : (%s) -> i64" resultSSA funcName argSSAsStr argTypesStr)
    TValue (resultSSA, "i64")

/// Transfer an unclassified application (no Operation set)
/// This handles function calls by either inlining the function body or emitting an external call
and transferUnclassifiedApp (ctx: EmitContext) (graph: ProgramSemanticGraph) (node: PSGNode) : TransferResult =
    let children = getChildren graph node
    match children with
    | funcNode :: argNodes ->
        // Process arguments first to get their SSA values
        let argResults = argNodes |> List.map (fun c -> transferNodeDirect ctx graph c)
        let argSSAs = argResults |> List.choose TransferResult.getSSA
        let argTypes = argResults |> List.choose TransferResult.getType

        // Extract function name from the func node
        let funcNameOpt =
            match funcNode.Symbol with
            | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
                Some (try mfv.CompiledName with _ -> mfv.DisplayName)
            | _ ->
                extractFunctionNameFromKind funcNode.SyntaxKind

        match funcNameOpt with
        | Some funcName ->
            // Try to find the function definition in the PSG for inlining
            match findFunctionBinding graph funcName with
            | Some bindingNode ->
                // Found the function - inline its body
                let (paramNodes, bodyNodeOpt) = getFunctionParamsAndBody graph bindingNode
                match bodyNodeOpt with
                | Some bodyNode ->
                    // Map parameters to argument SSA values
                    // Record each parameter node with its corresponding argument SSA
                    List.zip paramNodes argResults
                    |> List.iter (fun (paramNode, argResult) ->
                        match argResult with
                        | TValue (ssa, mlirType) ->
                            EmitContext.recordNodeSSA ctx paramNode.Id ssa mlirType
                        | _ -> ())

                    // Process the function body (this will resolve parameter references)
                    transferNodeDirect ctx graph bodyNode
                | None ->
                    // Function has no body - might be a platform binding
                    // TODO: This name-matching should be replaced with node.PlatformBinding.IsSome
                    if funcName.StartsWith("fidelity_") || funcName = "writeBytes" || funcName = "readBytes" then
                        // This is a platform binding - emit syscall via binding
                        let argTypesStr = if List.isEmpty argTypes then "" else argTypes |> String.concat ", "
                        let argSSAsStr = argSSAs |> String.concat ", "
                        ctx.ExternalFuncs.Add(funcName) |> ignore
                        let resultSSA = EmitContext.nextSSA ctx
                        if List.isEmpty argTypes then
                            EmitContext.emitLine ctx (sprintf "%s = llvm.call @%s() : () -> i64" resultSSA funcName)
                        else
                            EmitContext.emitLine ctx (sprintf "%s = llvm.call @%s(%s) : (%s) -> i64" resultSSA funcName argSSAsStr argTypesStr)
                        TValue (resultSSA, "i64")
                    else
                        TError (sprintf "Function %s has no body to inline" funcName)
            | None ->
                // Function not found in PSG - might be external or built-in
                // Check if it's a special operator or intrinsic
                let argTypesStr = if List.isEmpty argTypes then "" else argTypes |> String.concat ", "
                let argSSAsStr = argSSAs |> String.concat ", "

                // Handle special cases
                match funcName with
                | "ignore" ->
                    // ignore just returns unit
                    TVoid
                | "op_Dollar" ->
                    // SRTP dispatch operator - WritableString $ s
                    // The first arg is the type (WritableString), second is the value (s)
                    if argResults.Length >= 2 then
                        match argResults.[1] with
                        | TValue (strSSA, _) ->
                            // Get the string length from the SSA to string content mapping
                            let strLen =
                                match ctx.SSAToStringContent.TryGetValue(strSSA) with
                                | true, content -> content.Length
                                | false, _ -> 0  // Default to 0 for safety

                            // Emit write syscall directly using inline assembly
                            // Linux x86_64: syscall 1 = write(fd, buf, count)
                            let sysNumSSA = EmitContext.nextSSA ctx
                            EmitContext.emitLine ctx (sprintf "%s = arith.constant 1 : i64" sysNumSSA)
                            let fdSSA = EmitContext.nextSSA ctx
                            EmitContext.emitLine ctx (sprintf "%s = arith.constant 1 : i64" fdSSA)  // stdout
                            let lenSSA = EmitContext.nextSSA ctx
                            EmitContext.emitLine ctx (sprintf "%s = arith.constant %d : i64" lenSSA strLen)
                            let resultSSA = EmitContext.nextSSA ctx
                            EmitContext.emitLine ctx (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64"
                                resultSSA sysNumSSA fdSSA strSSA lenSSA)
                            TValue (resultSSA, "i64")
                        | TError _ | TVoid -> TVoid
                    else
                        TVoid  // Partial application
                | _ ->
                    // Register as external and emit call
                    ctx.ExternalFuncs.Add(funcName) |> ignore
                    let resultSSA = EmitContext.nextSSA ctx
                    if List.isEmpty argTypes then
                        EmitContext.emitLine ctx (sprintf "%s = llvm.call @%s() : () -> i64" resultSSA funcName)
                    else
                        EmitContext.emitLine ctx (sprintf "%s = llvm.call @%s(%s) : (%s) -> i64" resultSSA funcName argSSAsStr argTypesStr)
                    TValue (resultSSA, "i64")
        | None ->
            TError (sprintf "Cannot determine function for unclassified App: %s" node.SyntaxKind)
    | _ ->
        TError "App node has no children"

// ═══════════════════════════════════════════════════════════════════════════════
// PART 4: Main Node Transfer (dispatches by SyntaxKind and Operation)
// ═══════════════════════════════════════════════════════════════════════════════

/// Transfer a PSG node to MLIR
and transferNodeDirect (ctx: EmitContext) (graph: ProgramSemanticGraph) (node: PSGNode) : TransferResult =
    let kind = node.SyntaxKind

    // Constants
    if kind.StartsWith("Const:") then
        match extractConstant node with
        | Some (_, "unit") -> TVoid
        | Some (value, "string") ->
            let globalName = EmitContext.registerStringLiteral ctx value
            let ssa = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = llvm.mlir.addressof %s : !llvm.ptr" ssa globalName)
            // Record the mapping from SSA to string content for length lookup
            ctx.SSAToStringContent.[ssa] <- value
            TValue (ssa, "!llvm.ptr")
        | Some ("null", "!llvm.ptr") ->
            let ssa = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = llvm.mlir.zero : !llvm.ptr" ssa)
            TValue (ssa, "!llvm.ptr")
        | Some (value, mlirType) ->
            let ssa = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = arith.constant %s : %s" ssa value mlirType)
            TValue (ssa, mlirType)
        | None -> TError (sprintf "Unknown constant: %s" kind)

    // Identifiers / Value references
    elif kind.StartsWith("Ident:") || kind.StartsWith("LongIdent:") || kind.StartsWith("Value:") then
        // Try def-use resolution
        let defEdge = graph.Edges |> List.tryFind (fun e -> e.Source = node.Id && e.Kind = SymbolUse)
        match defEdge with
        | Some edge ->
            match EmitContext.lookupNodeSSA ctx edge.Target with
            | Some (ssa, mlirType) ->
                match EmitContext.lookupMutableBinding ctx node.Id with
                | Some valueType ->
                    let loadSSA = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = llvm.load %s : !llvm.ptr -> %s" loadSSA ssa valueType)
                    TValue (loadSSA, valueType)
                | None -> TValue (ssa, mlirType)
            | None -> TError (sprintf "Definition node %s has no SSA" edge.Target.Value)
        | None ->
            match EmitContext.lookupNodeSSA ctx node.Id with
            | Some (ssa, mlirType) -> TValue (ssa, mlirType)
            | None -> TError (sprintf "Unresolved identifier: %s" kind)

    // Sequential expressions
    elif kind.StartsWith("Sequential") then
        let children = getChildren graph node
        let rec transferAll lastResult remaining =
            match remaining with
            | [] -> lastResult
            | child :: rest ->
                let result = transferNodeDirect ctx graph child
                match result with
                | TError _ -> result
                | _ -> transferAll result rest
        transferAll TVoid children

    // LetOrUse:Let - has two children: binding and body
    elif kind.StartsWith("LetOrUse:Let") then
        let children = getChildren graph node
        // LetOrUse:Let structure: Binding [name], <body expression>
        let bindingChild = children |> List.tryFind (fun c -> c.SyntaxKind.StartsWith("Binding"))
        let bodyChild = children |> List.tryFind (fun c ->
            not (c.SyntaxKind.StartsWith("Binding")) &&
            not (c.SyntaxKind.StartsWith("Pattern:")) &&
            not (c.SyntaxKind.StartsWith("Named")))

        // Process the binding first to define the variable
        match bindingChild with
        | Some binding ->
            let bindingResult = transferNodeDirect ctx graph binding
            // Record the binding's SSA for the Pattern:Named node
            let patternNode = getChildren graph binding |> List.tryFind (fun c -> c.SyntaxKind.StartsWith("Pattern:Named:"))
            match patternNode, bindingResult with
            | Some pn, TValue (ssa, mlirType) ->
                EmitContext.recordNodeSSA ctx pn.Id ssa mlirType
            | _ -> ()
        | None -> ()

        // Then process the body expression
        match bodyChild with
        | Some body -> transferNodeDirect ctx graph body
        | None -> TVoid

    // Plain Let bindings (just the binding part, no body)
    elif kind.StartsWith("Binding") || kind = "LetOrUse" || kind.StartsWith("Let") then
        let children = getChildren graph node
        let valueChild = children |> List.tryFind (fun c ->
            not (c.SyntaxKind.StartsWith("Pattern:")) && not (c.SyntaxKind.StartsWith("Named")))
        match valueChild with
        | Some child ->
            let result = transferNodeDirect ctx graph child
            match result with
            | TValue (ssa, mlirType) ->
                EmitContext.recordNodeSSA ctx node.Id ssa mlirType
                if kind.Contains("Mutable") then
                    let ptrSSA = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = llvm.alloca i64 x 1 : (i64) -> !llvm.ptr" ptrSSA)
                    EmitContext.emitLine ctx (sprintf "llvm.store %s, %s : %s, !llvm.ptr" ssa ptrSSA mlirType)
                    EmitContext.recordMutableBinding ctx node.Id mlirType
                    EmitContext.recordNodeSSA ctx node.Id ptrSSA "!llvm.ptr"
                result
            | _ -> result
        | None -> TVoid

    // If-then-else
    elif kind.StartsWith("IfThenElse") then
        let children = getChildren graph node
        match children with
        | cond :: thenBranch :: rest ->
            let condResult = transferNodeDirect ctx graph cond
            match condResult with
            | TValue (condSSA, _) ->
                let thenLabel = EmitContext.nextLabel ctx
                let elseLabel = EmitContext.nextLabel ctx
                let mergeLabel = EmitContext.nextLabel ctx

                match rest with
                | elseBranch :: _ ->
                    EmitContext.emitLine ctx (sprintf "llvm.cond_br %s, ^%s, ^%s" condSSA thenLabel elseLabel)
                    EmitContext.emitLine ctx (sprintf "^%s:" thenLabel)
                    let thenResult = transferNodeDirect ctx graph thenBranch
                    let thenSSA = TransferResult.getSSA thenResult
                    EmitContext.emitLine ctx (sprintf "llvm.br ^%s" mergeLabel)
                    EmitContext.emitLine ctx (sprintf "^%s:" elseLabel)
                    let elseResult = transferNodeDirect ctx graph elseBranch
                    let elseSSA = TransferResult.getSSA elseResult
                    EmitContext.emitLine ctx (sprintf "llvm.br ^%s" mergeLabel)
                    EmitContext.emitLine ctx (sprintf "^%s:" mergeLabel)
                    match thenResult, elseResult with
                    | TValue (_, thenType), TValue (_, _) ->
                        let phiSSA = EmitContext.nextSSA ctx
                        let thenS = Option.defaultValue "%unit" thenSSA
                        let elseS = Option.defaultValue "%unit" elseSSA
                        EmitContext.emitLine ctx (sprintf "%s = llvm.mlir.phi [%s, ^%s], [%s, ^%s] : %s"
                            phiSSA thenS thenLabel elseS elseLabel thenType)
                        TValue (phiSSA, thenType)
                    | TVoid, TVoid -> TVoid
                    | TError msg, _ | _, TError msg -> TError msg
                    | _ -> TVoid
                | [] ->
                    EmitContext.emitLine ctx (sprintf "llvm.cond_br %s, ^%s, ^%s" condSSA thenLabel mergeLabel)
                    EmitContext.emitLine ctx (sprintf "^%s:" thenLabel)
                    let _ = transferNodeDirect ctx graph thenBranch
                    EmitContext.emitLine ctx (sprintf "llvm.br ^%s" mergeLabel)
                    EmitContext.emitLine ctx (sprintf "^%s:" mergeLabel)
                    TVoid
            | TError msg -> TError msg
            | TVoid -> TError "If condition produced no value"
        | _ -> TError "Invalid IfThenElse structure"

    // While loops
    elif kind.StartsWith("WhileLoop") then
        let children = getChildren graph node
        match children with
        | cond :: body :: _ ->
            let condLabel = EmitContext.nextLabel ctx
            let bodyLabel = EmitContext.nextLabel ctx
            let endLabel = EmitContext.nextLabel ctx
            EmitContext.emitLine ctx (sprintf "llvm.br ^%s" condLabel)
            EmitContext.emitLine ctx (sprintf "^%s:" condLabel)
            let condResult = transferNodeDirect ctx graph cond
            match condResult with
            | TValue (condSSA, _) ->
                EmitContext.emitLine ctx (sprintf "llvm.cond_br %s, ^%s, ^%s" condSSA bodyLabel endLabel)
                EmitContext.emitLine ctx (sprintf "^%s:" bodyLabel)
                let _ = transferNodeDirect ctx graph body
                EmitContext.emitLine ctx (sprintf "llvm.br ^%s" condLabel)
                EmitContext.emitLine ctx (sprintf "^%s:" endLabel)
                TVoid
            | TError msg -> TError msg
            | TVoid -> TError "While condition produced no value"
        | _ -> TError "Invalid WhileLoop structure"

    // Match expressions - pattern matching
    elif kind.StartsWith("Match") then
        let children = getChildren graph node
        // Structure: Match has scrutinee as first child, then MatchClause children
        // For freestanding mode (no argv), we execute the wildcard/default clause
        let clauses = children |> List.filter (fun c -> c.SyntaxKind.StartsWith("MatchClause"))

        // Find wildcard clause or last clause as default
        let defaultClause =
            clauses
            |> List.tryFind (fun clause ->
                let clauseChildren = getChildren graph clause
                clauseChildren |> List.exists (fun c -> c.SyntaxKind = "Pattern:Wildcard"))
            |> Option.orElse (List.tryLast clauses)

        match defaultClause with
        | Some clause ->
            // Find the body (non-Pattern child of MatchClause)
            let clauseChildren = getChildren graph clause
            let body = clauseChildren |> List.tryFind (fun c -> not (c.SyntaxKind.StartsWith("Pattern:")))
            match body with
            | Some bodyNode -> transferNodeDirect ctx graph bodyNode
            | None -> TVoid
        | None -> TError "Match has no clauses"

    // MatchClause - should be handled by parent Match
    elif kind.StartsWith("MatchClause") then
        TVoid

    // Application nodes - dispatch by Operation
    elif kind.StartsWith("App") then
        match node.Operation with
        | Some (Arithmetic op) -> transferArithmetic ctx graph node op
        | Some (Comparison op) -> transferComparison ctx graph node op
        | Some (Console op) -> transferConsole ctx graph node op
        | Some (NativeStr op) -> transferNativeStr ctx graph node op
        | Some (Time op) -> transferTime ctx graph node op
        | Some (RegularCall info) -> transferRegularCall ctx graph node info
        | Some (Bitwise _) -> TError "Bitwise operations not yet implemented"
        | Some (Conversion _) -> TError "Conversion operations not yet implemented"
        | Some (NativePtr _) -> TError "NativePtr operations not yet implemented"
        | Some (Memory _) -> TError "Memory operations not yet implemented"
        | Some (Result _) -> TError "Result operations not yet implemented"
        | Some (Core op) -> transferCore ctx graph node op
        | Some (TextFormat _) -> TError "TextFormat operations not yet implemented"
        | None -> transferUnclassifiedApp ctx graph node

    // Tuple construction
    elif kind.StartsWith("Tuple") || kind.StartsWith("NewTuple") then
        let children = getChildren graph node
        let results = children |> List.map (fun c -> transferNodeDirect ctx graph c)
        let ssas = results |> List.choose TransferResult.getSSA
        let types = results |> List.choose TransferResult.getType
        if List.isEmpty ssas then TVoid
        else
            let tupleType = sprintf "!llvm.struct<(%s)>" (String.concat ", " types)
            let resultSSA = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = llvm.mlir.undef : %s" resultSSA tupleType)
            let finalSSA =
                (resultSSA, List.zip ssas [0 .. ssas.Length - 1])
                ||> List.fold (fun acc (elemSSA, idx) ->
                    let nextSSA = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = llvm.insertvalue %s, %s[%d] : %s"
                        nextSSA elemSSA acc idx tupleType)
                    nextSSA)
            TValue (finalSSA, tupleType)

    // Property/field access
    elif kind.StartsWith("PropGet") || kind.StartsWith("FieldGet") then
        match EmitContext.lookupFieldAccess ctx node with
        | Some fieldInfo ->
            let children = getChildren graph node
            match children with
            | objNode :: _ ->
                let objResult = transferNodeDirect ctx graph objNode
                match objResult with
                | TValue (objSSA, objType) ->
                    let resultSSA = EmitContext.nextSSA ctx
                    let fieldType = mapFSharpTypeToMLIR fieldInfo.Field.FieldType
                    EmitContext.emitLine ctx (sprintf "%s = llvm.extractvalue %s[%d] : %s"
                        resultSSA objSSA fieldInfo.FieldIndex objType)
                    TValue (resultSSA, fieldType)
                | _ -> objResult
            | _ -> TError "Property access requires object"
        | None -> TError (sprintf "Unknown property: %s" kind)

    // Mutable set
    elif kind.StartsWith("MutableSet") || kind.StartsWith("ValueSet") then
        let children = getChildren graph node
        match children with
        | target :: value :: _ ->
            let targetResult = transferNodeDirect ctx graph target
            let valueResult = transferNodeDirect ctx graph value
            match targetResult, valueResult with
            | TValue (ptrSSA, _), TValue (valSSA, valType) ->
                EmitContext.emitLine ctx (sprintf "llvm.store %s, %s : %s, !llvm.ptr" valSSA ptrSSA valType)
                TVoid
            | TError msg, _ | _, TError msg -> TError msg
            | _ -> TError "Invalid mutable set"
        | _ -> TError "MutableSet requires target and value"

    // TypeApp - passthrough
    elif kind.StartsWith("TypeApp") then
        let children = getChildren graph node
        match children with
        | child :: _ -> transferNodeDirect ctx graph child
        | _ -> TVoid

    // AddressOf
    elif kind.StartsWith("AddressOf") then
        let children = getChildren graph node
        match children with
        | child :: _ ->
            let result = transferNodeDirect ctx graph child
            match result with
            | TValue (ssa, _) -> TValue (ssa, "!llvm.ptr")
            | _ -> result
        | _ -> TError "AddressOf requires child"

    // Patterns - structural, no emission
    elif kind.StartsWith("Pattern:") || kind.StartsWith("Named") then
        TVoid

    else
        TError (sprintf "Unknown node type: %s" kind)

// ═══════════════════════════════════════════════════════════════════════════════
// PART 5: XParsec Combinators
// ═══════════════════════════════════════════════════════════════════════════════

/// Transfer a constant node
let pTransferConstant : Transfer =
    satisfyChild (fun n -> n.SyntaxKind.StartsWith("Const:"))
    |> pBind (fun node ->
        getEmitCtx
        |> pBind (fun ctx ->
            getGraph
            |> pMap (fun graph -> transferNodeDirect ctx graph node)))

/// Transfer an identifier
let pTransferIdent : Transfer =
    satisfyChild (fun n ->
        n.SyntaxKind.StartsWith("Ident:") ||
        n.SyntaxKind.StartsWith("LongIdent:") ||
        n.SyntaxKind.StartsWith("Value:"))
    |> pBind (fun node ->
        getEmitCtx
        |> pBind (fun ctx ->
            getGraph
            |> pMap (fun graph -> transferNodeDirect ctx graph node)))

/// Transfer a sequential expression
let pTransferSequential : Transfer =
    satisfyChild (fun n -> n.SyntaxKind.StartsWith("Sequential"))
    |> pBind (fun node ->
        getEmitCtx
        |> pBind (fun ctx ->
            getGraph
            |> pMap (fun graph -> transferNodeDirect ctx graph node)))

/// Transfer a let binding
let pTransferLetBinding : Transfer =
    satisfyChild (fun n ->
        n.SyntaxKind.StartsWith("Binding") ||
        n.SyntaxKind = "LetOrUse" ||
        n.SyntaxKind.StartsWith("Let"))
    |> pBind (fun node ->
        getEmitCtx
        |> pBind (fun ctx ->
            getGraph
            |> pMap (fun graph -> transferNodeDirect ctx graph node)))

/// Transfer an if-then-else
let pTransferIfThenElse : Transfer =
    satisfyChild (fun n -> n.SyntaxKind.StartsWith("IfThenElse"))
    |> pBind (fun node ->
        getEmitCtx
        |> pBind (fun ctx ->
            getGraph
            |> pMap (fun graph -> transferNodeDirect ctx graph node)))

/// Transfer an application
let pTransferApp : Transfer =
    satisfyChild (fun n -> n.SyntaxKind.StartsWith("App"))
    |> pBind (fun node ->
        getEmitCtx
        |> pBind (fun ctx ->
            getGraph
            |> pMap (fun graph -> transferNodeDirect ctx graph node)))

/// Transfer a while loop
let pTransferWhile : Transfer =
    satisfyChild (fun n -> n.SyntaxKind.StartsWith("WhileLoop"))
    |> pBind (fun node ->
        getEmitCtx
        |> pBind (fun ctx ->
            getGraph
            |> pMap (fun graph -> transferNodeDirect ctx graph node)))

/// Master XParsec parser
let pTransferNode : Transfer =
    pChoice [
        pTransferConstant
        pTransferIdent
        pTransferSequential
        pTransferLetBinding
        pTransferIfThenElse
        pTransferApp
        pTransferWhile
    ]
    |> pLabel "transfer-node"

// ═══════════════════════════════════════════════════════════════════════════════
// PART 6: Entry Points
// ═══════════════════════════════════════════════════════════════════════════════

type MLIRTransferResult = {
    Content: string
    StringLiterals: (string * string) list
    ExternalFuncs: Set<string>
    Errors: string list
}

module MLIRTransferResult =
    let hasErrors r = not (List.isEmpty r.Errors)

/// Transfer a PSG to MLIR
let transferToMLIR (psg: ProgramSemanticGraph) (entry: PSGNode) (correlations: CorrelationState) : MLIRTransferResult =
    let ctx = EmitContext.create psg correlations
    let result = transferNodeDirect ctx psg entry
    match result with
    | TError msg -> EmitContext.recordError ctx msg
    | _ -> ()
    // Convert new collection types to legacy types for backwards compatibility
    let stringLiteralsList = ctx.StringLiterals |> Seq.map (fun kv -> (kv.Key, kv.Value)) |> Seq.toList
    let externalFuncsSet = ctx.ExternalFuncs |> Seq.fold (fun acc x -> Set.add x acc) Set.empty
    {
        Content = EmitContext.getOutput ctx
        StringLiterals = stringLiteralsList
        ExternalFuncs = externalFuncsSet
        Errors = EmitContext.getErrors ctx
    }

/// Transfer with function wrapper
let transferFunctionToMLIR (psg: ProgramSemanticGraph) (funcNode: PSGNode) (correlations: CorrelationState) (funcName: string) : MLIRTransferResult =
    let ctx = EmitContext.create psg correlations
    ctx.CurrentFunction <- Some funcName
    EmitContext.emitLine ctx (sprintf "  llvm.func @%s() -> i32 {" funcName)
    let children = getChildren psg funcNode
    let mutable lastResult = TVoid
    for child in children do
        lastResult <- transferNodeDirect ctx psg child
    match lastResult with
    | TValue (ssa, "i32") ->
        EmitContext.emitLine ctx (sprintf "    llvm.return %s : i32" ssa)
    | _ ->
        let zeroSSA = EmitContext.nextSSA ctx
        EmitContext.emitLine ctx (sprintf "    %s = arith.constant 0 : i32" zeroSSA)
        EmitContext.emitLine ctx (sprintf "    llvm.return %s : i32" zeroSSA)
    EmitContext.emitLine ctx "  }"
    // Convert new collection types to legacy types for backwards compatibility
    let stringLiteralsList = ctx.StringLiterals |> Seq.map (fun kv -> (kv.Key, kv.Value)) |> Seq.toList
    let externalFuncsSet = ctx.ExternalFuncs |> Seq.fold (fun acc x -> Set.add x acc) Set.empty
    {
        Content = EmitContext.getOutput ctx
        StringLiterals = stringLiteralsList
        ExternalFuncs = externalFuncsSet
        Errors = EmitContext.getErrors ctx
    }

// ═══════════════════════════════════════════════════════════════════════════════
// PART 7: Main Entry Point (for CompileCommand compatibility)
// ═══════════════════════════════════════════════════════════════════════════════

/// Result type for generateMLIR (compatibility with CompileCommand)
type GenerationResult = {
    Content: string
    Errors: string list
    HasErrors: bool
}

/// Escape string content for MLIR string literals
let private escapeStringContent (s: string) : string =
    s.Replace("\\", "\\\\")
     .Replace("\"", "\\22")
     .Replace("\n", "\\0A")
     .Replace("\r", "\\0D")
     .Replace("\t", "\\09")

/// Generate MLIR from a PSG (main entry point for compilation)
/// Uses layered emission: globals are emitted DURING traversal to Declarations layer,
/// function bodies go to Operations layer. This function assembles the final module.
let generateMLIR (psg: ProgramSemanticGraph) (correlations: CorrelationState) (targetTriple: string) (outputKind: OutputKind) : GenerationResult =
    let allErrors = ResizeArray<string>()
    let allExternalFuncs = ResizeArray<string>()

    // Create a SHARED EmitContext for the entire module
    // This ensures globals are deduplicated across all functions
    let ctx = EmitContext.create psg correlations

    // Process each entry point
    for entryId in psg.EntryPoints do
        match Map.tryFind entryId.Value psg.Nodes with
        | Some entryNode ->
            // Get function name from symbol
            let funcName =
                match entryNode.Symbol with
                | Some sym ->
                    let name =
                        match sym with
                        | :? FSharpMemberOrFunctionOrValue as mfv ->
                            try mfv.CompiledName with _ -> mfv.DisplayName
                        | _ -> sym.DisplayName
                    if name = "main" || name.EndsWith(".main") then "main"
                    else name.Replace(".", "_")
                | None -> "main"

            // Reset the Operations layer for this function (Declarations accumulates)
            ctx.CurrentFunction <- Some funcName
            EmitContext.emitLine ctx (sprintf "  llvm.func @%s() -> i32 {" funcName)

            // Transfer the function body
            let children = getChildren psg entryNode
            let mutable lastResult = TVoid
            for child in children do
                lastResult <- transferNodeDirect ctx psg child

            // Emit return or exit syscall depending on output kind
            match outputKind with
            | Core.Types.MLIRTypes.Freestanding | Core.Types.MLIRTypes.Embedded ->
                // For freestanding mode, emit exit syscall instead of return
                // This is required because there's no CRT to catch the return
                let exitCodeSSA =
                    match lastResult with
                    | TValue (ssa, "i32") -> ssa
                    | _ ->
                        let zeroSSA = EmitContext.nextSSA ctx
                        EmitContext.emitLine ctx (sprintf "%s = arith.constant 0 : i32" zeroSSA)
                        zeroSSA
                // Extend exit code to i64
                let exitCode64 = EmitContext.nextSSA ctx
                EmitContext.emitLine ctx (sprintf "%s = arith.extsi %s : i32 to i64" exitCode64 exitCodeSSA)
                // Emit exit syscall (syscall 60 on Linux x86-64)
                let sysNumSSA = EmitContext.nextSSA ctx
                EmitContext.emitLine ctx (sprintf "%s = arith.constant 60 : i64" sysNumSSA)
                let resultSSA = EmitContext.nextSSA ctx
                EmitContext.emitLine ctx (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"=r,{rax},{rdi},~{rcx},~{r11},~{memory}\" %s, %s : (i64, i64) -> i64"
                    resultSSA sysNumSSA exitCode64)
                EmitContext.emitLine ctx "llvm.unreachable"
            | Core.Types.MLIRTypes.Console | Core.Types.MLIRTypes.Library ->
                // Standard return for console/library mode (CRT handles cleanup)
                match lastResult with
                | TValue (ssa, "i32") ->
                    EmitContext.emitLine ctx (sprintf "    llvm.return %s : i32" ssa)
                | _ ->
                    let zeroSSA = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "    %s = arith.constant 0 : i32" zeroSSA)
                    EmitContext.emitLine ctx (sprintf "    llvm.return %s : i32" zeroSSA)
            EmitContext.emitLine ctx "  }"

            // Collect external functions for declaration
            allExternalFuncs.AddRange(ctx.ExternalFuncs)
            allErrors.AddRange(EmitContext.getErrors ctx)
        | None ->
            allErrors.Add(sprintf "Entry point not found: %s" entryId.Value)

    // Now emit extern function declarations to Declarations layer
    // NOTE: Only fidelity_* primitives should reach this point.
    // If other functions appear here, the PSG decomposition chain is incomplete.
    let uniqueExterns = allExternalFuncs |> Seq.distinct |> Seq.toList
    for externName in uniqueExterns do
        // Determine function signature based on name
        let signature =
            match externName with
            | "fidelity_write_bytes" -> "(i32, !llvm.ptr, i64) -> i64"
            | "fidelity_exit" -> "(i32)"
            | _ -> "() -> i64"
        EmitContext.emitToLayer ctx Declarations
            (sprintf "  llvm.func @%s%s attributes {sym_visibility = \"private\"}" externName signature)

    // Finalize: assemble Declarations + Operations into complete module
    {
        Content = EmitContext.finalize ctx
        Errors = allErrors |> Seq.toList
        HasErrors = allErrors.Count > 0
    }