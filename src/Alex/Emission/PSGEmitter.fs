/// PSGEmitter - Alex-driven PSG to MLIR emission
///
/// This module drives the transformation from PSG to MLIR.
/// It traverses the PSG using the zipper and applies XParsec patterns
/// to recognize and transform nodes to MLIR.
///
/// ARCHITECTURAL PRINCIPLE:
/// - Alex DRIVES the traversal and transformation
/// - MLIRBuilder provides combinators for MLIR generation
/// - ExpressionEmitter.EmitResult is the communication type
/// - Alloy functions are INLINED at call sites (not emitted as calls)
/// - The zipper gives the fully flattened function call stack
module Alex.Emission.PSGEmitter

open Core.PSG.Types
open Alex.CodeGeneration.MLIRBuilder
open Alex.Emission.ExpressionEmitter

// ═══════════════════════════════════════════════════════════════════
// MLIR Monad Helpers
// ═══════════════════════════════════════════════════════════════════

/// Map a function over a list in the MLIR monad, collecting results
let rec private mlirMapM (f: 'a -> MLIR<'b>) (xs: 'a list) : MLIR<'b list> =
    match xs with
    | [] -> mlir { return [] }
    | x :: rest ->
        mlir {
            let! y = f x
            let! ys = mlirMapM f rest
            return y :: ys
        }

// ═══════════════════════════════════════════════════════════════════
// Helper: Get Children
// ═══════════════════════════════════════════════════════════════════

let private getChildNodes (psg: ProgramSemanticGraph) (node: PSGNode) : PSGNode list =
    match node.Children with
    | Parent ids -> ids |> List.choose (fun id -> Map.tryFind id.Value psg.Nodes)
    | _ -> []

let private getSymbolFullName (node: PSGNode) : string option =
    node.Symbol |> Option.bind (fun s ->
        try Some s.FullName with _ -> None)

// ═══════════════════════════════════════════════════════════════════
// Function Definition Index
// ═══════════════════════════════════════════════════════════════════

/// Build an index from function symbol full names to their binding nodes
/// This enables O(1) lookup when inlining Alloy functions
let buildFunctionIndex (psg: ProgramSemanticGraph) : Map<string, PSGNode> =
    psg.Nodes
    |> Map.toSeq
    |> Seq.choose (fun (_, node) ->
        match node.SyntaxKind with
        | sk when sk.StartsWith("Binding") ->
            match node.Symbol with
            | Some symbol ->
                // Only index functions (not value bindings)
                match symbol with
                | :? FSharp.Compiler.Symbols.FSharpMemberOrFunctionOrValue as mfv
                    when mfv.IsFunction || mfv.IsModuleValueOrMember ->
                    Some (symbol.FullName, node)
                | _ -> None
            | None -> None
        | _ -> None)
    |> Map.ofSeq

/// Get the body of a function binding (the expression after the pattern)
let getFunctionBody (psg: ProgramSemanticGraph) (bindingNode: PSGNode) : PSGNode option =
    let children = getChildNodes psg bindingNode
    // Binding has [Pattern; body] or [Pattern; Pattern...; body]
    // The body is the last child that is NOT a Pattern
    children
    |> List.tryFindBack (fun c -> not (c.SyntaxKind.StartsWith("Pattern:")))

/// Get the parameters of a function binding (the Pattern:Named nodes)
let getFunctionParams (psg: ProgramSemanticGraph) (bindingNode: PSGNode) : (string * PSGNode) list =
    let children = getChildNodes psg bindingNode
    // Find Pattern:LongIdent (function name pattern), then get its Pattern:Named children
    children
    |> List.tryFind (fun c -> c.SyntaxKind.StartsWith("Pattern:LongIdent"))
    |> Option.map (fun patternNode ->
        getChildNodes psg patternNode
        |> List.choose (fun p ->
            if p.SyntaxKind.StartsWith("Pattern:Named:") then
                // Extract parameter name from "Pattern:Named:paramName"
                let name = p.SyntaxKind.Substring("Pattern:Named:".Length)
                Some (name, p)
            else None))
    |> Option.defaultValue []

// ═══════════════════════════════════════════════════════════════════
// Emission Context - carries function index and local bindings
// ═══════════════════════════════════════════════════════════════════

/// Context for emission, carrying state that needs to be threaded through
type EmissionContext = {
    /// Index from function symbol names to their binding nodes
    FunctionIndex: Map<string, PSGNode>
    /// Current parameter bindings: parameter name -> emitted SSA value
    ParamBindings: Map<string, Val>
    /// Current local bindings: local name -> emitted SSA value
    LocalBindings: Map<string, Val>
    /// Recursion depth (for debugging/limiting inlining)
    InlineDepth: int
    /// Maximum inline depth
    MaxInlineDepth: int
}

module EmissionContext =
    let create funcIndex = {
        FunctionIndex = funcIndex
        ParamBindings = Map.empty
        LocalBindings = Map.empty
        InlineDepth = 0
        MaxInlineDepth = 20  // Limit recursion
    }

    let withParam name value ctx =
        { ctx with ParamBindings = Map.add name value ctx.ParamBindings }

    let withLocal name value ctx =
        { ctx with LocalBindings = Map.add name value ctx.LocalBindings }

    let lookupBinding name ctx =
        // Try locals first, then params
        match Map.tryFind name ctx.LocalBindings with
        | Some v -> Some v
        | None -> Map.tryFind name ctx.ParamBindings

    let enterInline ctx =
        { ctx with InlineDepth = ctx.InlineDepth + 1 }

    let canInline ctx =
        ctx.InlineDepth < ctx.MaxInlineDepth

// ═══════════════════════════════════════════════════════════════════
// Constant Emission
// ═══════════════════════════════════════════════════════════════════

let private extractStringFromKind (kind: string) : string option =
    if kind.StartsWith("Const:String") then
        let start = kind.IndexOf("(\"")
        if start >= 0 then
            let contentStart = start + 2
            let endQuote = kind.IndexOf("\",", contentStart)
            if endQuote >= contentStart then
                Some (kind.Substring(contentStart, endQuote - contentStart))
            else None
        else None
    else None

let private extractInt32FromKind (kind: string) : int option =
    if kind.StartsWith("Const:Int32 ") then
        let numStr = kind.Substring(12).Trim()
        match System.Int32.TryParse(numStr) with
        | true, n -> Some n
        | _ -> None
    else None

let private extractByteFromKind (kind: string) : byte option =
    if kind.StartsWith("Const:Byte ") then
        let numStr = kind.Substring(11).Trim().Replace("uy", "")
        match System.Byte.TryParse(numStr) with
        | true, b -> Some b
        | _ -> None
    else None

let private extractBoolFromKind (kind: string) : bool option =
    if kind.StartsWith("Const:Bool ") then
        let valStr = kind.Substring(11).Trim().ToLower()
        if valStr = "true" then Some true
        elif valStr = "false" then Some false
        else None
    else None

let emitConst (node: PSGNode) : MLIR<EmitResult> =
    let kind = node.SyntaxKind
    match extractStringFromKind kind with
    | Some str ->
        mlir {
            let! nstr = buildNativeStr str
            return Emitted nstr
        }
    | None ->
    match extractInt32FromKind kind with
    | Some n ->
        mlir {
            let! v = arith.constant (int64 n) I32
            return Emitted v
        }
    | None ->
    match extractByteFromKind kind with
    | Some b ->
        mlir {
            let! v = arith.constant (int64 b) I8
            return Emitted v
        }
    | None ->
    match extractBoolFromKind kind with
    | Some b ->
        mlir {
            let! v = arith.constant (if b then 1L else 0L) I1
            return Emitted v
        }
    | None ->
    if kind = "Const:Unit" || kind.Contains("Unit") then
        mlir { return Void }
    else
        mlir { return Error ("Unknown constant: " + kind) }

// ═══════════════════════════════════════════════════════════════════
// User Function Call Helpers
// ═══════════════════════════════════════════════════════════════════

/// Convert F# symbol name to MLIR function name
let private toMLIRFunctionName (fullName: string) : string =
    let segments = fullName.Split('.')
    let shortName =
        if segments.Length >= 2 then
            sprintf "%s_%s" segments.[segments.Length - 2] segments.[segments.Length - 1]
        else
            segments.[segments.Length - 1]
    shortName
        .Replace(".", "_")
        .Replace("``", "")
        .Replace("'", "_p")

/// Check if a symbol is a library function (Alloy, FSharp, System)
let private isLibraryFunction (fullName: string) : bool =
    fullName.StartsWith("Alloy.") ||
    fullName.StartsWith("Microsoft.FSharp.") ||
    fullName.StartsWith("System.")

/// Check if a symbol represents a user-defined function
let private isUserDefinedFunction (node: PSGNode) : bool =
    match node.Symbol with
    | Some sym ->
        match sym with
        | :? FSharp.Compiler.Symbols.FSharpMemberOrFunctionOrValue as mfv ->
            mfv.IsModuleValueOrMember && mfv.IsFunction
        | _ -> false
    | None -> false

// NOTE: emitUserFunctionCall is defined in the mutual recursion group below

// ═══════════════════════════════════════════════════════════════════
// Main Expression Emission - recursive traversal with pattern matching
// ═══════════════════════════════════════════════════════════════════

/// Extract identifier name from "Ident:name" or "LongIdent:Module.name"
let private extractIdentName (kind: string) : string option =
    if kind.StartsWith("Ident:") then
        Some (kind.Substring(6))
    elif kind.StartsWith("LongIdent:") then
        let full = kind.Substring(10)
        // Get the last segment (the actual name)
        let segments = full.Split('.')
        Some segments.[segments.Length - 1]
    else
        None

/// Check if this is a primitive operation that should be emitted directly
let private isPrimitiveOp (name: string) : bool =
    name.StartsWith("Microsoft.FSharp.Core.Operators.") ||
    name = "Microsoft.FSharp.Core.LanguagePrimitives.GenericZero"
    // Note: NativePtr operations are handled separately by emitNativePtrOp

/// Emit an expression node to MLIR with context
/// This is the main traversal function that dispatches based on node kind
let rec emitExprWithContext (psg: ProgramSemanticGraph) (ctx: EmissionContext) (node: PSGNode) : MLIR<EmitResult> =
    let kind = node.SyntaxKind

    // Constants
    if kind.StartsWith("Const:") then
        emitConst node

    // Sequential - emit all children, return last result
    elif kind.StartsWith("Sequential") then
        emitSequentialWithContext psg ctx node

    // Let bindings - emit value, then body
    elif kind.StartsWith("LetOrUse") || kind.StartsWith("Let") then
        emitLetWithContext psg ctx node

    // Patterns - structural, no emission
    elif kind.StartsWith("Pattern:") then
        mlir { return Void }

    // Bindings - structural, no emission
    elif kind.StartsWith("Binding") then
        mlir { return Void }

    // Module - structural
    elif kind.StartsWith("Module:") then
        mlir { return Void }

    // Function application - the main dispatch point for operations
    elif kind.StartsWith("App") then
        emitAppWithContext psg ctx node

    // Identifiers - resolve from context bindings
    elif kind.StartsWith("Ident:") || kind.StartsWith("LongIdent:") then
        emitIdentWithContext ctx node

    // TypeApp - type application, delegate to child
    elif kind.StartsWith("TypeApp") then
        let children = getChildNodes psg node
        match children with
        | [child] -> emitExprWithContext psg ctx child
        | _ -> mlir { return Error "TypeApp: unexpected children" }

    // Tuple - emit all elements
    elif kind.StartsWith("Tuple") then
        let children = getChildNodes psg node
        mlir {
            // For now, just emit all children for side effects
            for child in children do
                let! _ = emitExprWithContext psg ctx child
                ()
            return Void
        }

    // WhileLoop - emit as MLIR scf.while or cf blocks
    elif kind.StartsWith("WhileLoop") then
        emitWhileLoopWithContext psg ctx node

    // IfThenElse - emit as MLIR scf.if or cf.cond_br
    elif kind.StartsWith("IfThenElse") then
        emitIfThenElseWithContext psg ctx node

    // MutableSet - assignment to mutable variable
    elif kind.StartsWith("MutableSet:") then
        emitMutableSetWithContext psg ctx node

    // AddressOf - get address of a variable
    elif kind.StartsWith("AddressOf") then
        emitAddressOfWithContext psg ctx node

    // InterpolatedString - for now emit as comment/error
    elif kind.StartsWith("InterpolatedString") then
        mlir { return Error "InterpolatedString: not yet implemented" }

    else
        mlir { return Error $"Unknown node kind: {kind}" }

/// Emit an Ident node by resolving it from context bindings
and private emitIdentWithContext (ctx: EmissionContext) (node: PSGNode) : MLIR<EmitResult> =
    match extractIdentName node.SyntaxKind with
    | Some name ->
        match EmissionContext.lookupBinding name ctx with
        | Some value -> mlir { return Emitted value }
        | None ->
            // Check if it's a known constant like STDIN_FILENO
            if name = "STDIN_FILENO" then
                mlir {
                    let! v = arith.constant 0L I32
                    return Emitted v
                }
            elif name = "STDOUT_FILENO" then
                mlir {
                    let! v = arith.constant 1L I32
                    return Emitted v
                }
            elif name = "STDERR_FILENO" then
                mlir {
                    let! v = arith.constant 2L I32
                    return Emitted v
                }
            else
                mlir { return Error $"Ident not bound: {name}" }
    | None ->
        mlir { return Error $"Cannot extract ident name from: {node.SyntaxKind}" }

/// Emit a Sequential node with context
and private emitSequentialWithContext (psg: ProgramSemanticGraph) (ctx: EmissionContext) (node: PSGNode) : MLIR<EmitResult> =
    let children = getChildNodes psg node
    match children with
    | [] -> mlir { return Void }
    | [single] -> emitExprWithContext psg ctx single
    | _ ->
        mlir {
            let mutable lastResult = Void
            for child in children do
                let! result = emitExprWithContext psg ctx child
                lastResult <- result
            return lastResult
        }

/// Emit a Let binding with context - creates new bindings
and private emitLetWithContext (psg: ProgramSemanticGraph) (ctx: EmissionContext) (node: PSGNode) : MLIR<EmitResult> =
    let children = getChildNodes psg node
    // LetOrUse typically has: [Binding; body] or [Binding1; Binding2; ...; body]
    // The body is the last non-Binding child
    let bindings, body =
        let rec split acc = function
            | [] -> (List.rev acc, None)
            | [last] ->
                if last.SyntaxKind.StartsWith("Binding") then
                    (List.rev (last :: acc), None)
                else
                    (List.rev acc, Some last)
            | h :: t ->
                split (h :: acc) t
        split [] children

    // Process bindings, building up new context
    let rec processBindings currentCtx = function
        | [] -> mlir { return currentCtx }
        | binding :: rest ->
            mlir {
                let bindingChildren = getChildNodes psg binding
                match bindingChildren with
                | patternNode :: valueNode :: _ ->
                    // Extract binding name from Pattern:Named:name
                    let bindingName =
                        if patternNode.SyntaxKind.StartsWith("Pattern:Named:") then
                            Some (patternNode.SyntaxKind.Substring(14))
                        elif patternNode.SyntaxKind.StartsWith("Pattern:LongIdent:") then
                            // For function definitions like "let buffer = ..."
                            Some (patternNode.SyntaxKind.Substring(18))
                        else None

                    // Emit the value expression
                    let! valueResult = emitExprWithContext psg currentCtx valueNode

                    // Add to context if we got a value
                    let newCtx =
                        match bindingName, valueResult with
                        | Some name, Emitted value ->
                            EmissionContext.withLocal name value currentCtx
                        | _ -> currentCtx

                    return! processBindings newCtx rest
                | _ ->
                    return! processBindings currentCtx rest
            }

    mlir {
        let! finalCtx = processBindings ctx bindings
        // Emit body with accumulated bindings
        match body with
        | Some b -> return! emitExprWithContext psg finalCtx b
        | None -> return Void
    }

/// Emit a WhileLoop using cf.br/cf.cond_br control flow
/// WhileLoop has structure: [condition; body]
/// Translates to:
///   ^header:
///     %cond = <condition>
///     cf.cond_br %cond, ^body, ^exit
///   ^body:
///     <body>
///     cf.br ^header
///   ^exit:
and private emitWhileLoopWithContext (psg: ProgramSemanticGraph) (ctx: EmissionContext) (node: PSGNode) : MLIR<EmitResult> =
    let children = getChildNodes psg node
    match children with
    | [condNode; bodyNode] ->
        mlir {
            // Generate unique labels for this loop
            let! headerLabel = freshLabel ()
            let! bodyLabel = freshLabel ()
            let! exitLabel = freshLabel ()

            // Jump to header to start the loop
            do! cf.br headerLabel []

            // Header block: evaluate condition
            do! emitBlockLabel headerLabel []
            let! condResult = emitExprWithContext psg ctx condNode
            match condResult with
            | Emitted condVal ->
                // Branch based on condition
                do! cf.condBr condVal bodyLabel [] exitLabel []

                // Body block: execute body, then jump back to header
                do! emitBlockLabel bodyLabel []
                let! _ = emitExprWithContext psg ctx bodyNode
                do! cf.br headerLabel []

                // Exit block: loop is done
                do! emitBlockLabel exitLabel []
                return Void

            | Error msg ->
                return Error $"WhileLoop condition failed: {msg}"
            | Void ->
                return Error "WhileLoop condition produced no value"
        }
    | _ ->
        mlir { return Error $"WhileLoop: expected 2 children (condition, body), got {children.Length}" }

/// Emit an IfThenElse - emit as scf.if or just emit branches
and private emitIfThenElseWithContext (psg: ProgramSemanticGraph) (ctx: EmissionContext) (node: PSGNode) : MLIR<EmitResult> =
    let children = getChildNodes psg node
    match children with
    | [condNode; thenNode; elseNode] ->
        mlir {
            // For now, just emit the condition and both branches
            // TODO: implement proper control flow
            let! condResult = emitExprWithContext psg ctx condNode
            match condResult with
            | Emitted _condVal ->
                // For now, just emit both branches for testing
                let! _thenResult = emitExprWithContext psg ctx thenNode
                let! elseResult = emitExprWithContext psg ctx elseNode
                return elseResult
            | _ ->
                return Error "IfThenElse: condition emission failed"
        }
    | _ ->
        mlir { return Error "IfThenElse: expected 3 children" }

/// Emit a MutableSet (assignment to mutable variable)
and private emitMutableSetWithContext (psg: ProgramSemanticGraph) (ctx: EmissionContext) (node: PSGNode) : MLIR<EmitResult> =
    // MutableSet has [value] child, and the variable name is in SyntaxKind
    // For now, emit a placeholder
    mlir { return Void }

/// Emit AddressOf - get pointer to a variable
and private emitAddressOfWithContext (psg: ProgramSemanticGraph) (ctx: EmissionContext) (node: PSGNode) : MLIR<EmitResult> =
    // For now, emit as error/placeholder
    mlir { return Error "AddressOf: not yet implemented" }

/// Emit a NativeStr operation based on the classified Operation field
/// This handles lowered interpolated strings and direct NativeStr calls
and private emitNativeStrOp (psg: ProgramSemanticGraph) (ctx: EmissionContext) (node: PSGNode) (op: NativeStrOp) : MLIR<EmitResult> =
    let children = getChildNodes psg node
    match op with
    | StrConcat3 ->
        // Lowered interpolated string: concat3 dest a b c
        // The children are the parts of the interpolated string
        // Structure: [InterpolatedStringPart:String:<hash>; InterpolatedStringPart:Fill; ...]
        mlir {
            match children with
            | [] ->
                return Error "StrConcat3: no children"
            | parts ->
                // Allocate result buffer (256 bytes max for now)
                let! bufSize = arith.constant 256L I64
                let! resultBuf = llvm.alloca bufSize (Int I8)

                // Track current position in buffer
                let! initialPos = arith.constant 0L I64

                // Process each part and accumulate into buffer
                let rec emitParts (pos: Val) (remaining: PSGNode list) : MLIR<Val> =
                    mlir {
                        match remaining with
                        | [] -> return pos
                        | part :: rest ->
                            let partKind = part.SyntaxKind

                            // Check if this is a string literal part
                            if partKind.StartsWith("InterpolatedStringPart:String:") then
                                // Extract hash from "InterpolatedStringPart:String:<hash>"
                                let hashStr = partKind.Substring("InterpolatedStringPart:String:".Length)
                                match System.UInt32.TryParse(hashStr) with
                                | true, hash ->
                                    // Look up string content in PSG's StringLiterals
                                    match Map.tryFind hash psg.StringLiterals with
                                    | Some content ->
                                        // Build NativeStr from the string literal
                                        let! strVal = buildNativeStr content
                                        let! (srcPtr, srcLen) = extractNativeStr strVal

                                        // Copy this part to the buffer at current position
                                        // GEP to get pointer at current offset
                                        let! destPtr = llvm.getelementptr resultBuf (Int I8) [pos]

                                        // Emit a simple byte-by-byte copy loop
                                        // For efficiency, we'd use llvm.memcpy, but let's do inline loop
                                        // Actually, for simple strings, just copy inline (compiler will optimize)
                                        let contentLen = int64 content.Length
                                        let! lenVal = arith.constant contentLen I64

                                        // Emit inline copy using llvm.memcpy intrinsic pattern
                                        // For now, do an unrolled store of each character
                                        for i = 0 to content.Length - 1 do
                                            let! offset = arith.constant (int64 i) I64
                                            let! byteVal = arith.constant (int64 (byte content.[i])) I8
                                            let! destI = llvm.getelementptr destPtr (Int I8) [offset]
                                            do! llvm.store byteVal destI

                                        // Advance position
                                        let! newPos = arith.addi pos lenVal
                                        return! emitParts newPos rest
                                    | None ->
                                        do! errorComment $"String literal not found for hash {hash}"
                                        return! emitParts pos rest
                                | false, _ ->
                                    do! errorComment $"Invalid hash in: {partKind}"
                                    return! emitParts pos rest

                            // Check if this is a fill expression (variable interpolation)
                            elif partKind.StartsWith("InterpolatedStringPart:Fill") then
                                // The fill part contains child expression(s)
                                let fillChildren = getChildNodes psg part
                                match fillChildren with
                                | [exprNode] ->
                                    // Emit the expression - should return a NativeStr
                                    let! exprResult = emitExprWithContext psg ctx exprNode
                                    match exprResult with
                                    | Emitted exprVal ->
                                        let! (srcPtr, srcLen) = extractNativeStr exprVal

                                        // GEP to get dest pointer at current offset
                                        let! destPtr = llvm.getelementptr resultBuf (Int I8) [pos]

                                        // Use inline assembly for memcpy (rep movsb on x86-64)
                                        // This copies srcLen bytes from srcPtr to destPtr
                                        // rep movsb: RCX=count, RSI=src, RDI=dest
                                        let! _ = llvm.inlineAsm "rep movsb" "={rcx},{rcx},{rsi},{rdi},~{memory},~{dirflag}" [srcLen; srcPtr; destPtr] (Int I64)

                                        // Advance position by srcLen
                                        let! newPos = arith.addi pos srcLen
                                        return! emitParts newPos rest
                                    | Void ->
                                        do! errorComment "Fill expression produced no value"
                                        return! emitParts pos rest
                                    | Error msg ->
                                        do! errorComment $"Fill expression error: {msg}"
                                        return! emitParts pos rest
                                | _ ->
                                    do! errorComment $"Fill part has {fillChildren.Length} children, expected 1"
                                    return! emitParts pos rest
                            else
                                do! errorComment $"Unknown part kind: {partKind}"
                                return! emitParts pos rest
                    }

                // Process all parts
                let! finalPos = emitParts initialPos parts

                // Build final NativeStr from result buffer and total length
                let! result = buildNativeStrFromValues resultBuf finalPos
                return Emitted result
        }
    | StrConcat2 ->
        mlir { return Error "StrConcat2: not yet implemented" }
    | StrCreate ->
        // NativeStr constructor - delegate to existing implementation
        let args = match children with | _ :: rest -> rest | [] -> []
        emitNativeStrConstructor psg ctx args
    | StrEmpty ->
        mlir {
            // Empty string: ptr=zero, len=0
            let! nullPtr = llvm.zero Ptr
            let! zero = arith.constant 0L I64
            let! nstr = buildNativeStrFromValues nullPtr zero
            return Emitted nstr
        }
    | StrLength ->
        match children with
        | [_func; strArg] ->
            mlir {
                let! strResult = emitExprWithContext psg ctx strArg
                match strResult with
                | Emitted strVal ->
                    let! (_, len) = extractNativeStr strVal
                    return Emitted len
                | _ -> return Error "StrLength: argument emission failed"
            }
        | _ -> mlir { return Error "StrLength: expected 1 argument" }
    | _ ->
        mlir { return Error $"NativeStr op not implemented: {op}" }

/// Emit a Console operation based on the classified Operation field
and private emitConsoleOp (psg: ProgramSemanticGraph) (ctx: EmissionContext) (node: PSGNode) (op: ConsoleOp) : MLIR<EmitResult> =
    match op with
    | ConsoleWrite ->
        emitConsoleWriteWithContext psg ctx node
    | ConsoleWriteln ->
        emitConsoleWriteLineWithContext psg ctx node
    | ConsoleReadBytes ->
        let children = getChildNodes psg node
        let args = match children with | _ :: rest -> rest | [] -> []
        emitReadBytesSyscall psg ctx args
    | ConsoleWriteBytes ->
        let children = getChildNodes psg node
        let args = match children with | _ :: rest -> rest | [] -> []
        emitWriteBytesSyscall psg ctx args
    | ConsoleReadLine ->
        // ReadLine/readln/readLine - inline the correct function based on symbol
        let children = getChildNodes psg node
        let funcNode = children |> List.tryHead
        let funcName = funcNode |> Option.bind getSymbolFullName |> Option.defaultValue "Alloy.Console.ReadLine"
        let args = match children with | _ :: rest -> rest | [] -> []
        emitInlinedAlloyFunction psg ctx funcName args
    | ConsoleReadInto ->
        // readInto/readLineInto - use specialized emission with proper mutable state handling
        let children = getChildNodes psg node
        let args = match children with | _ :: rest -> rest | [] -> []
        emitReadLineInto psg ctx args
    | ConsoleNewLine ->
        // newLine - emit a single newline character
        mlir {
            let! nl = arith.constant 10L I8
            let! one = arith.constant 1L I64
            let! buf = llvm.alloca one (Int I8)
            do! llvm.store nl buf
            let! fd = arith.constant 1L I32
            let! fdExt = arith.extsi fd I64
            let! syscallNum = arith.constant 1L I64
            let! _ = llvm.inlineAsm "syscall" "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}" [syscallNum; fdExt; buf; one] (Int I64)
            return Void
        }
    | _ ->
        mlir { return Error $"Console op not implemented: {op}" }

/// Emit a function application with context - handles inlining
and private emitAppWithContext (psg: ProgramSemanticGraph) (ctx: EmissionContext) (node: PSGNode) : MLIR<EmitResult> =
    // First, check if this node has a classified Operation (set by ClassifyOperations nanopass)
    // This takes precedence over symbol-based dispatch
    match node.Operation with
    | Some (NativeStr op) ->
        emitNativeStrOp psg ctx node op
    | Some (Console op) ->
        emitConsoleOp psg ctx node op
    | _ ->
        // Fall back to symbol-based dispatch
        let children = getChildNodes psg node
        match children with
        | funcNode :: args ->
            let symbolName = getSymbolFullName funcNode
            match symbolName with
            // Console.Write - inline directly as syscall
            | Some name when name = "Alloy.Console.Write" || name.EndsWith(".Console.Write") ->
                emitConsoleWriteWithContext psg ctx node

            // Console.WriteLine - inline directly as syscall
            | Some name when name = "Alloy.Console.WriteLine" || name.EndsWith(".Console.WriteLine") ->
                emitConsoleWriteLineWithContext psg ctx node

            // Console.readLineInto - specialized emission with proper mutable state handling
            | Some name when name = "Alloy.Console.readLineInto" || name.EndsWith(".Console.readLineInto") ->
                emitReadLineInto psg ctx args

            // Console.readBytes - primitive syscall (read from fd)
            | Some name when name = "Alloy.Console.readBytes" || name.EndsWith(".Console.readBytes") ->
                emitReadBytesSyscall psg ctx args

            // Console.writeBytes - primitive syscall (write to fd)
            | Some name when name = "Alloy.Console.writeBytes" || name.EndsWith(".Console.writeBytes") ->
                emitWriteBytesSyscall psg ctx args

            // NativeStr constructor - build struct {ptr, len}
            | Some name when name = "Alloy.NativeTypes.NativeString.NativeStr" || name.EndsWith(".NativeStr") ->
                emitNativeStrConstructor psg ctx args

            // Primitive operations (operators, NativePtr, etc.) - emit directly
            | Some name when isPrimitiveOp name ->
                emitPrimitiveOp psg ctx name args

            // NativePtr operations
            | Some name when name.Contains("NativePtr.") ->
                emitNativePtrOp psg ctx name args

            // Alloy library functions - INLINE from PSG
            | Some name when name.StartsWith("Alloy.") && EmissionContext.canInline ctx ->
                emitInlinedAlloyFunction psg ctx name args

            // Alloy library functions - depth exceeded
            | Some name when name.StartsWith("Alloy.") ->
                mlir { return Error $"Alloy function inline depth exceeded: {name}" }

            // User-defined functions - emit as call
            | Some name when not (isLibraryFunction name) && isUserDefinedFunction funcNode ->
                emitUserFunctionCall psg ctx node

            // Unhandled
            | Some name ->
                mlir { return Error $"Unhandled function: {name}" }
            | None ->
                mlir { return Error "App: no symbol on function node" }
        | [] ->
            mlir { return Error "App: no children" }

/// Emit Console.Write with context
and private emitConsoleWriteWithContext (psg: ProgramSemanticGraph) (ctx: EmissionContext) (node: PSGNode) : MLIR<EmitResult> =
    let children = getChildNodes psg node
    let args = match children with | _ :: rest -> rest | [] -> []

    match args with
    | [strArg] ->
        mlir {
            let! argResult = emitExprWithContext psg ctx strArg
            match argResult with
            | Emitted strVal ->
                let! (ptr, len) = extractNativeStr strVal
                let! fd = arith.constant 1L I32
                let! fdExt = arith.extsi fd I64
                let! syscallNum = arith.constant 1L I64
                let! _ = llvm.inlineAsm "syscall" "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}" [syscallNum; fdExt; ptr; len] (Int I64)
                return Void
            | Void -> return Error "Console.Write: argument produced no value"
            | Error msg -> return Error msg
        }
    | _ ->
        mlir { return Error "Console.Write: expected 1 argument" }

/// Emit Console.WriteLine with context
and private emitConsoleWriteLineWithContext (psg: ProgramSemanticGraph) (ctx: EmissionContext) (node: PSGNode) : MLIR<EmitResult> =
    let children = getChildNodes psg node
    let args = match children with | _ :: rest -> rest | [] -> []

    match args with
    | [strArg] ->
        mlir {
            let! argResult = emitExprWithContext psg ctx strArg
            match argResult with
            | Emitted strVal ->
                let! (ptr, len) = extractNativeStr strVal
                let! fd = arith.constant 1L I32
                let! fdExt = arith.extsi fd I64
                let! syscallNum = arith.constant 1L I64
                let! _ = llvm.inlineAsm "syscall" "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}" [syscallNum; fdExt; ptr; len] (Int I64)
                // Write newline
                let! nlByte = arith.constant 10L I8
                let! allocSize = arith.constant 1L I64
                let! nlPtr = llvm.alloca allocSize (Int I8)
                do! llvm.store nlByte nlPtr
                let! one = arith.constant 1L I64
                let! _ = llvm.inlineAsm "syscall" "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}" [syscallNum; fdExt; nlPtr; one] (Int I64)
                return Void
            | Void -> return Error "Console.WriteLine: argument produced no value"
            | Error msg -> return Error msg
        }
    | _ ->
        mlir { return Error "Console.WriteLine: expected 1 argument" }

/// Emit readBytes syscall - read(fd, buffer, count)
/// Signature: readBytes (fd: int) (buffer: nativeptr<byte>) (maxCount: int) : int
and private emitReadBytesSyscall (psg: ProgramSemanticGraph) (ctx: EmissionContext) (args: PSGNode list) : MLIR<EmitResult> =
    match args with
    | [fdArg; bufferArg; countArg] ->
        mlir {
            let! fdResult = emitExprWithContext psg ctx fdArg
            let! bufferResult = emitExprWithContext psg ctx bufferArg
            let! countResult = emitExprWithContext psg ctx countArg

            match fdResult, bufferResult, countResult with
            | Emitted fdVal, Emitted bufferPtr, Emitted countVal ->
                // Extend fd and count to i64 for syscall
                let! fdI64 = arith.extsi fdVal I64
                let! countI64 = arith.extsi countVal I64
                // syscall 0 = read on x86-64 Linux
                let! syscallNum = arith.constant 0L I64
                let! result = llvm.inlineAsm "syscall" "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}" [syscallNum; fdI64; bufferPtr; countI64] (Int I64)
                // Truncate result to i32 (bytes read)
                let! resultI32 = arith.trunci result I32
                return Emitted resultI32
            | _ ->
                return Error "readBytes: argument emission failed"
        }
    | _ ->
        mlir { return Error "readBytes: expected 3 arguments (fd, buffer, count)" }

/// Emit writeBytes syscall - write(fd, buffer, count)
/// Signature: writeBytes (fd: int) (buffer: nativeptr<byte>) (count: int) : int
and private emitWriteBytesSyscall (psg: ProgramSemanticGraph) (ctx: EmissionContext) (args: PSGNode list) : MLIR<EmitResult> =
    match args with
    | [fdArg; bufferArg; countArg] ->
        mlir {
            let! fdResult = emitExprWithContext psg ctx fdArg
            let! bufferResult = emitExprWithContext psg ctx bufferArg
            let! countResult = emitExprWithContext psg ctx countArg

            match fdResult, bufferResult, countResult with
            | Emitted fdVal, Emitted bufferPtr, Emitted countVal ->
                // Extend fd and count to i64 for syscall
                let! fdI64 = arith.extsi fdVal I64
                let! countI64 = arith.extsi countVal I64
                // syscall 1 = write on x86-64 Linux
                let! syscallNum = arith.constant 1L I64
                let! result = llvm.inlineAsm "syscall" "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}" [syscallNum; fdI64; bufferPtr; countI64] (Int I64)
                // Truncate result to i32 (bytes written)
                let! resultI32 = arith.trunci result I32
                return Emitted resultI32
            | _ ->
                return Error "writeBytes: argument emission failed"
        }
    | _ ->
        mlir { return Error "writeBytes: expected 3 arguments (fd, buffer, count)" }

/// Emit readLineInto - specialized emission with proper mutable state handling
/// This implements the loop that reads one byte at a time until newline or EOF
/// Signature: readLineInto (fd: int) (buffer: nativeptr<byte>) (maxLength: int) : int
and private emitReadLineInto (psg: ProgramSemanticGraph) (ctx: EmissionContext) (args: PSGNode list) : MLIR<EmitResult> =
    match args with
    | [fdArg; bufferArg; maxLengthArg] ->
        mlir {
            let! fdResult = emitExprWithContext psg ctx fdArg
            let! bufferResult = emitExprWithContext psg ctx bufferArg
            let! maxLengthResult = emitExprWithContext psg ctx maxLengthArg

            match fdResult, bufferResult, maxLengthResult with
            | Emitted fdVal, Emitted bufferPtr, Emitted maxLenVal ->
                // Allocate mutable state on stack
                let! one64 = arith.constant 1L I64

                // count: i32 initialized to 0
                let! countAlloca = llvm.alloca one64 (Int I32)
                let! zeroI32 = arith.constant 0L I32
                do! llvm.store zeroI32 countAlloca

                // done_: i1 initialized to false
                let! doneAlloca = llvm.alloca one64 (Int I1)
                let! falseBool = arith.constBool false
                do! llvm.store falseBool doneAlloca

                // Allocate single byte buffer for reading one character at a time
                let! byteBuf = llvm.alloca one64 (Int I8)

                // Generate labels
                let! headerLabel = freshLabel ()
                let! bodyLabel = freshLabel ()
                let! exitLabel = freshLabel ()

                // Jump to loop header
                do! cf.br headerLabel []

                // === Header block: check loop condition ===
                do! emitBlockLabel headerLabel []
                // Load done_ and count
                let! doneVal = llvm.load (Int I1) doneAlloca
                let! countVal = llvm.load (Int I32) countAlloca

                // Check: not done_ && count < (maxLength - 1)
                let! trueBool = arith.constBool true
                let! notDone = arith.xori doneVal trueBool
                let! oneI32 = arith.constant 1L I32
                let! maxMinusOne = arith.subi maxLenVal oneI32
                let! countLtMax = arith.cmpi "slt" countVal maxMinusOne
                let! loopCond = arith.andi notDone countLtMax

                // Branch: continue or exit
                do! cf.condBr loopCond bodyLabel [] exitLabel []

                // === Body block: read one byte ===
                do! emitBlockLabel bodyLabel []

                // syscall read(fd, byteBuf, 1)
                let! fdI64 = arith.extsi fdVal I64
                let! syscallRead = arith.constant 0L I64  // read = 0
                let! bytesRead = llvm.inlineAsm "syscall" "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}" [syscallRead; fdI64; byteBuf; one64] (Int I64)
                let! bytesReadI32 = arith.trunci bytesRead I32

                // Check if bytesRead <= 0 -> EOF or error -> done
                let! zeroI32Check = arith.constant 0L I32
                let! isEofOrError = arith.cmpi "sle" bytesReadI32 zeroI32Check

                // Load the byte we read
                let! byteVal = llvm.load (Int I8) byteBuf
                let! newlineChar = arith.constant 10L I8  // '\n'
                let! isNewline = arith.cmpi "eq" byteVal newlineChar

                // done_ = isEofOrError || isNewline
                let! newDone = arith.ori isEofOrError isNewline
                do! llvm.store newDone doneAlloca

                // If not done, store byte in buffer and increment count
                // Only store the byte if it's not a newline and we got data
                let! trueBool2 = arith.constBool true
                let! notNewDone = arith.xori newDone trueBool2
                let! shouldStore = arith.andi notNewDone trueBool2

                // Get current count and compute buffer offset
                let! curCount = llvm.load (Int I32) countAlloca
                let! curCountI64 = arith.extsi curCount I64
                let! destPtr = llvm.getelementptr bufferPtr (Int I8) [curCountI64]

                // Store the byte (will only matter if we didn't hit newline/EOF)
                do! llvm.store byteVal destPtr

                // Increment count (only if we actually stored)
                let! newCount = arith.addi curCount oneI32
                // Use select to conditionally update count: if shouldStore then newCount else curCount
                let! actualNewCount = llvm.select shouldStore newCount curCount
                do! llvm.store actualNewCount countAlloca

                // Jump back to header
                do! cf.br headerLabel []

                // === Exit block ===
                do! emitBlockLabel exitLabel []

                // Null-terminate the buffer
                let! finalCount = llvm.load (Int I32) countAlloca
                let! finalCountI64 = arith.extsi finalCount I64
                let! nullPtr = llvm.getelementptr bufferPtr (Int I8) [finalCountI64]
                let! zeroI8 = arith.constant 0L I8
                do! llvm.store zeroI8 nullPtr

                // Return the count
                return Emitted finalCount
            | _ ->
                return Error "readLineInto: argument emission failed"
        }
    | _ ->
        mlir { return Error $"readLineInto: expected 3 arguments (fd, buffer, maxLength), got {args.Length}" }

/// Emit NativeStr constructor - builds struct {ptr, len}
/// Signature: NativeStr(ptr: nativeptr<byte>, len: int) : NativeStr
/// The PSG structure is: App [ Ident:NativeStr ; Tuple [ ptrArg ; lenArg ] ]
and private emitNativeStrConstructor (psg: ProgramSemanticGraph) (ctx: EmissionContext) (args: PSGNode list) : MLIR<EmitResult> =
    // Arguments may come as [ptrArg; lenArg] or as [Tuple [ptrArg; lenArg]]
    let actualArgs =
        match args with
        | [tupleArg] when tupleArg.SyntaxKind.StartsWith("Tuple") ->
            // Extract children from Tuple
            getChildNodes psg tupleArg
        | _ -> args

    match actualArgs with
    | [ptrArg; lenArg] ->
        mlir {
            let! ptrResult = emitExprWithContext psg ctx ptrArg
            let! lenResult = emitExprWithContext psg ctx lenArg

            match ptrResult, lenResult with
            | Emitted ptrVal, Emitted lenVal ->
                // Extend length to i64
                let! lenI64 = arith.extsi lenVal I64
                // Build the NativeStr struct {ptr, i64}
                let! nstr = buildNativeStrFromValues ptrVal lenI64
                return Emitted nstr
            | _ ->
                return Error $"NativeStr: argument emission failed (ptr={ptrResult}, len={lenResult})"
        }
    | _ ->
        mlir { return Error $"NativeStr: expected 2 arguments (ptr, len), got {actualArgs.Length}" }

/// Emit a primitive operation (operators, etc.)
and private emitPrimitiveOp (psg: ProgramSemanticGraph) (ctx: EmissionContext) (name: string) (args: PSGNode list) : MLIR<EmitResult> =
    match name with
    | "Microsoft.FSharp.Core.Operators.(<)" ->
        match args with
        | [a; b] ->
            mlir {
                let! aResult = emitExprWithContext psg ctx a
                let! bResult = emitExprWithContext psg ctx b
                match aResult, bResult with
                | Emitted aVal, Emitted bVal ->
                    let! result = arith.cmpi "slt" aVal bVal
                    return Emitted result
                | _ -> return Error "(<): operand emission failed"
            }
        | _ -> mlir { return Error "(<): expected 2 arguments" }

    | "Microsoft.FSharp.Core.Operators.(<=)" ->
        match args with
        | [a; b] ->
            mlir {
                let! aResult = emitExprWithContext psg ctx a
                let! bResult = emitExprWithContext psg ctx b
                match aResult, bResult with
                | Emitted aVal, Emitted bVal ->
                    let! result = arith.cmpi "sle" aVal bVal
                    return Emitted result
                | _ -> return Error "(<=): operand emission failed"
            }
        | _ -> mlir { return Error "(<=): expected 2 arguments" }

    | "Microsoft.FSharp.Core.Operators.(=)" ->
        match args with
        | [a; b] ->
            mlir {
                let! aResult = emitExprWithContext psg ctx a
                let! bResult = emitExprWithContext psg ctx b
                match aResult, bResult with
                | Emitted aVal, Emitted bVal ->
                    let! result = arith.cmpi "eq" aVal bVal
                    return Emitted result
                | _ -> return Error "(=): operand emission failed"
            }
        | _ -> mlir { return Error "(=): expected 2 arguments" }

    | "Microsoft.FSharp.Core.Operators.(+)" ->
        match args with
        | [a; b] ->
            mlir {
                let! aResult = emitExprWithContext psg ctx a
                let! bResult = emitExprWithContext psg ctx b
                match aResult, bResult with
                | Emitted aVal, Emitted bVal ->
                    let! result = arith.addi aVal bVal
                    return Emitted result
                | _ -> return Error "(+): operand emission failed"
            }
        | _ -> mlir { return Error "(+): expected 2 arguments" }

    | "Microsoft.FSharp.Core.Operators.(-)" ->
        match args with
        | [a; b] ->
            mlir {
                let! aResult = emitExprWithContext psg ctx a
                let! bResult = emitExprWithContext psg ctx b
                match aResult, bResult with
                | Emitted aVal, Emitted bVal ->
                    let! result = arith.subi aVal bVal
                    return Emitted result
                | _ -> return Error "(-): operand emission failed"
            }
        | _ -> mlir { return Error "(-): expected 2 arguments" }

    | "Microsoft.FSharp.Core.Operators.``not``" ->
        match args with
        | [a] ->
            mlir {
                let! aResult = emitExprWithContext psg ctx a
                match aResult with
                | Emitted aVal ->
                    // not = xor with 1
                    let! one = arith.constant 1L I1
                    let! result = arith.xori aVal one
                    return Emitted result
                | _ -> return Error "not: operand emission failed"
            }
        | _ -> mlir { return Error "not: expected 1 argument" }

    | _ ->
        mlir { return Error $"Unhandled primitive op: {name}" }

/// Emit a NativePtr operation
and private emitNativePtrOp (psg: ProgramSemanticGraph) (ctx: EmissionContext) (name: string) (args: PSGNode list) : MLIR<EmitResult> =
    if name.EndsWith(".stackalloc") then
        match args with
        | [countArg] ->
            mlir {
                let! countResult = emitExprWithContext psg ctx countArg
                match countResult with
                | Emitted countVal ->
                    let! countI64 = arith.extsi countVal I64
                    let! ptr = llvm.alloca countI64 (Int I8)
                    return Emitted ptr
                | _ -> return Error "stackalloc: count emission failed"
            }
        | _ -> mlir { return Error "stackalloc: expected 1 argument" }

    elif name.EndsWith(".set") then
        match args with
        | [ptrArg; indexArg; valueArg] ->
            mlir {
                let! ptrResult = emitExprWithContext psg ctx ptrArg
                let! indexResult = emitExprWithContext psg ctx indexArg
                let! valueResult = emitExprWithContext psg ctx valueArg
                match ptrResult, indexResult, valueResult with
                | Emitted ptr, Emitted index, Emitted value ->
                    // ptr + index * elemSize, then store
                    let! indexI64 = arith.extsi index I64
                    let! offsetPtr = llvm.getelementptr ptr (Int I8) [indexI64]
                    do! llvm.store value offsetPtr
                    return Void
                | _ -> return Error "NativePtr.set: operand emission failed"
            }
        | _ -> mlir { return Error "NativePtr.set: expected 3 arguments" }

    elif name.EndsWith(".get") then
        match args with
        | [ptrArg; indexArg] ->
            mlir {
                let! ptrResult = emitExprWithContext psg ctx ptrArg
                let! indexResult = emitExprWithContext psg ctx indexArg
                match ptrResult, indexResult with
                | Emitted ptr, Emitted index ->
                    let! indexI64 = arith.extsi index I64
                    let! offsetPtr = llvm.getelementptr ptr (Int I8) [indexI64]
                    let! value = llvm.load (Int I8) offsetPtr
                    return Emitted value
                | _ -> return Error "NativePtr.get: operand emission failed"
            }
        | _ -> mlir { return Error "NativePtr.get: expected 2 arguments" }

    elif name.EndsWith(".toVoidPtr") then
        match args with
        | [ptrArg] ->
            mlir {
                let! result = emitExprWithContext psg ctx ptrArg
                return result  // ptr is already void-compatible in LLVM dialect
            }
        | _ -> mlir { return Error "NativePtr.toVoidPtr: expected 1 argument" }

    elif name.EndsWith(".ofVoidPtr") then
        match args with
        | [ptrArg] ->
            mlir {
                let! result = emitExprWithContext psg ctx ptrArg
                return result  // ptr is already the right type in LLVM dialect
            }
        | _ -> mlir { return Error "NativePtr.ofVoidPtr: expected 1 argument" }

    else
        mlir { return Error $"Unhandled NativePtr op: {name}" }

/// Emit an inlined Alloy library function
and private emitInlinedAlloyFunction (psg: ProgramSemanticGraph) (ctx: EmissionContext) (funcName: string) (args: PSGNode list) : MLIR<EmitResult> =
    // Look up the function in our index
    match Map.tryFind funcName ctx.FunctionIndex with
    | Some bindingNode ->
        // Get the function body
        match getFunctionBody psg bindingNode with
        | Some body ->
            // Get parameters and bind arguments
            let params' = getFunctionParams psg bindingNode

            // Build new context with parameter bindings
            let rec bindParams paramCtx paramList argList =
                match paramList, argList with
                | [], _ -> mlir { return paramCtx }
                | (paramName, _) :: pRest, argNode :: aRest ->
                    mlir {
                        let! argResult = emitExprWithContext psg paramCtx argNode
                        match argResult with
                        | Emitted argVal ->
                            let newCtx = EmissionContext.withParam paramName argVal paramCtx
                            return! bindParams newCtx pRest aRest
                        | _ ->
                            // Skip unbound params
                            return! bindParams paramCtx pRest aRest
                    }
                | _, [] ->
                    // More params than args - skip remaining
                    mlir { return paramCtx }

            mlir {
                let inlineCtx = EmissionContext.enterInline ctx
                let! boundCtx = bindParams inlineCtx params' args
                // Emit the function body with bound parameters
                let! result = emitExprWithContext psg boundCtx body
                return result
            }
        | None ->
            mlir { return Error $"Alloy function has no body: {funcName}" }
    | None ->
        mlir { return Error $"Alloy function not in index: {funcName}" }

/// Emit a user function call
and private emitUserFunctionCall (psg: ProgramSemanticGraph) (ctx: EmissionContext) (node: PSGNode) : MLIR<EmitResult> =
    let children = getChildNodes psg node
    match children with
    | funcNode :: args ->
        match getSymbolFullName funcNode with
        | Some fullName ->
            let mlirFuncName = toMLIRFunctionName fullName
            mlir {
                // Emit all arguments
                let! argResults = mlirMapM (emitExprWithContext psg ctx) args
                let emittedArgs =
                    argResults |> List.choose (function
                        | Emitted v -> Some v
                        | _ -> None)
                let argTypes = emittedArgs |> List.map (fun v -> v.Type)

                // If we have args, pass them; otherwise call with no args
                if List.isEmpty emittedArgs then
                    do! func.callVoid mlirFuncName [] []
                    return Void
                else
                    do! func.callVoid mlirFuncName emittedArgs argTypes
                    return Void
            }
        | None ->
            mlir { return Error "User function call: no symbol name" }
    | [] ->
        mlir { return Error "User function call: empty children" }

/// Legacy wrapper for backward compatibility
let emitExpr (psg: ProgramSemanticGraph) (node: PSGNode) : MLIR<EmitResult> =
    let funcIndex = buildFunctionIndex psg
    let ctx = EmissionContext.create funcIndex
    emitExprWithContext psg ctx node

/// Emit an expression with initial parameter bindings
/// Used when emitting function bodies where parameters are already bound
let emitExprWithParams (psg: ProgramSemanticGraph) (paramBindings: Map<string, Val>) (node: PSGNode) : MLIR<EmitResult> =
    let funcIndex = buildFunctionIndex psg
    let ctx = { EmissionContext.create funcIndex with ParamBindings = paramBindings }
    emitExprWithContext psg ctx node
