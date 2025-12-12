/// PSGGeneration - MLIR generation from PSG using EmitContext
///
/// ARCHITECTURAL PRINCIPLE:
/// Uses EmitContext from PSGXParsec for all state management.
/// Variable resolution uses def-use edges, NOT name-based parameter bindings.
/// The PSG structure drives generation - no central dispatch.
module Alex.Traversal.PSGGeneration

open System
open System.Collections.Immutable
open FSharp.Compiler.Symbols
open Core.PSG.Types
open Alex.Traversal.PSGZipper
open Alex.Traversal.PSGXParsec
open Alex.Patterns.PSGPatterns
open Alex.CodeGeneration.MLIRBuilder
open Alex.Bindings.BindingTypes
open Core.PSG.Nanopass.DefUseEdges

// ═══════════════════════════════════════════════════════════════════
// MLIR Type Mapping
// ═══════════════════════════════════════════════════════════════════

let mapType (ftype: FSharpType option) : string =
    match ftype with
    | None -> "i32"
    | Some t ->
        try
            if t.HasTypeDefinition then
                match t.TypeDefinition.TryFullName with
                | Some "System.Int32" -> "i32"
                | Some "System.Int64" -> "i64"
                | Some "System.Int16" -> "i16"
                | Some "System.Byte" | Some "System.SByte" -> "i8"
                | Some "System.UInt32" -> "i32"
                | Some "System.UInt64" -> "i64"
                | Some "System.Boolean" -> "i1"
                | Some "System.Single" -> "f32"
                | Some "System.Double" -> "f64"
                | Some "System.Void" | Some "Microsoft.FSharp.Core.unit" -> "unit"
                | Some "System.IntPtr" | Some "System.UIntPtr" -> "!llvm.ptr"
                | Some n when n.Contains("nativeptr") -> "!llvm.ptr"
                | Some "Microsoft.FSharp.Core.string" -> "!llvm.ptr"
                | _ -> "i32"
            elif t.IsGenericParameter then
                "!llvm.ptr"
            else
                "i32"
        with _ -> "i32"

// ═══════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════

/// Get children of a node from the PSG
let getChildren (ctx: EmitContext) (node: PSGNode) : PSGNode list =
    ChildrenStateHelpers.getChildrenList node
    |> List.choose (fun id -> Map.tryFind id.Value ctx.Graph.Nodes)

/// Extract constant value from a Const node
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

/// Extract identifier name from an Ident node
let extractIdentName (node: PSGNode) : string option =
    let kind = node.SyntaxKind
    if kind.StartsWith("Ident:") then Some (kind.Substring(6))
    elif kind.StartsWith("LongIdent:") then
        let fullName = kind.Substring(10)
        match fullName.LastIndexOf('.') with
        | -1 -> Some fullName
        | i -> Some (fullName.Substring(i + 1))
    else None

/// Flatten tuple arguments
let rec flattenTupleArgs (ctx: EmitContext) (argNodes: PSGNode list) : PSGNode list =
    argNodes
    |> List.collect (fun node ->
        if node.SyntaxKind = "Tuple" then getChildren ctx node
        else [node])

/// Extract parameter names from a pattern node
let rec extractParameterNames (ctx: EmitContext) (pattern: PSGNode) : string list =
    if pattern.SyntaxKind.StartsWith("Pattern:Named:") then
        [pattern.SyntaxKind.Substring(14)]
    elif pattern.SyntaxKind.StartsWith("Pattern:LongIdent:") then
        getChildren ctx pattern |> List.collect (extractParameterNames ctx)
    elif pattern.SyntaxKind = "Pattern:Paren" || pattern.SyntaxKind = "Pattern:Tuple" then
        getChildren ctx pattern |> List.collect (extractParameterNames ctx)
    else
        getChildren ctx pattern |> List.collect (extractParameterNames ctx)

// ═══════════════════════════════════════════════════════════════════
// Generation Functions - Use EmitContext, return ExprResult
// ═══════════════════════════════════════════════════════════════════

/// Generate MLIR for a constant
let genConstant (ctx: EmitContext) (node: PSGNode) : ExprResult =
    match extractConstant node with
    | Some (_, "unit") -> Void
    | Some (value, "string") ->
        let globalName = EmitContext.registerStringLiteral ctx value
        let ssa = EmitContext.nextSSA ctx
        EmitContext.emitLine ctx (sprintf "%s = llvm.mlir.addressof %s : !llvm.ptr" ssa globalName)
        Value (ssa, "!llvm.ptr")
    | Some ("null", "!llvm.ptr") ->
        let ssa = EmitContext.nextSSA ctx
        EmitContext.emitLine ctx (sprintf "%s = llvm.mlir.zero : !llvm.ptr" ssa)
        Value (ssa, "!llvm.ptr")
    | Some (value, mlirType) ->
        let ssa = EmitContext.nextSSA ctx
        EmitContext.emitLine ctx (sprintf "%s = arith.constant %s : %s" ssa value mlirType)
        Value (ssa, mlirType)
    | None -> EmitError (sprintf "Unknown constant: %s" node.SyntaxKind)

/// Generate MLIR for an identifier (variable reference via def-use edges)
let genIdent (ctx: EmitContext) (node: PSGNode) : ExprResult =
    match findDefiningNode ctx.Graph node with
    | Some defNode ->
        match EmitContext.lookupNodeSSA ctx defNode.Id with
        | Some (ssa, ty) -> Value (ssa, ty)
        | None ->
            printfn "[GEN] Warning: Definition node %s has no SSA value yet" defNode.Id.Value
            let ty = mapType node.Type
            EmitError (sprintf "Definition %s not yet emitted" defNode.Id.Value)
    | None ->
        printfn "[GEN] No def-use edge for: %s (%s)" node.SyntaxKind (extractIdentName node |> Option.defaultValue "?")
        let ty = mapType node.Type
        EmitError "No defining node found"

// Forward declaration for mutual recursion
let rec genNode (ctx: EmitContext) (node: PSGNode) : ExprResult =
    let kind = node.SyntaxKind

    if kind.StartsWith("Const:") then
        genConstant ctx node
    elif kind.StartsWith("Ident:") || kind.StartsWith("LongIdent:") then
        genIdent ctx node
    elif kind = "Sequential" then
        genSequential ctx node
    elif kind.StartsWith("LetOrUse:") || kind.StartsWith("Binding") then
        genLetBinding ctx node
    elif kind = "IfThenElse" || kind.StartsWith("IfThenElse:") then
        genIfThenElse ctx node
    elif kind.StartsWith("App:") then
        genApp ctx node
    elif kind = "Tuple" then
        genTuple ctx node
    elif kind = "AddressOf" then
        genAddressOf ctx node
    elif kind.StartsWith("TypeApp:") then
        genTypeApp ctx node
    elif kind.StartsWith("PropertyAccess:") then
        genPropertyAccess ctx node
    elif kind.StartsWith("SemanticPrimitive:") then
        genSemanticPrimitive ctx node
    elif kind.StartsWith("Pattern:") then
        let children = getChildren ctx node
        match children with
        | [child] -> genNode ctx child
        | _ -> Void
    else
        let children = getChildren ctx node
        match children with
        | [child] -> genNode ctx child
        | _ ->
            printfn "[GEN] Unknown node kind: %s" kind
            Void

/// Generate MLIR for property access (e.g., string.Length)
/// NOTE: For compile-time string length, the ConstantPropagation nanopass
/// should have already set ConstantValue on the target node.
and genPropertyAccess (ctx: EmitContext) (node: PSGNode) : ExprResult =
    let kind = node.SyntaxKind
    let propName = if kind.StartsWith("PropertyAccess:") then kind.Substring(15) else ""

    // First check if the PropertyAccess node itself has a constant value
    // (set by ConstantPropagation nanopass for compile-time folding)
    match node.ConstantValue, propName with
    | Some (Int32Value length), "Length" ->
        // Compile-time constant from nanopass
        let ssa = EmitContext.nextSSA ctx
        EmitContext.emitLine ctx (sprintf "%s = arith.constant %d : i32" ssa length)
        Value (ssa, "i32")
    | _ ->
        let children = getChildren ctx node
        match children, propName with
        | [target], "Length" ->
            // Check if target has a constant string value (direct literal)
            match target.ConstantValue with
            | Some (StringValue s) ->
                // String literal - length is known at compile time
                let length = s.Length
                let ssa = EmitContext.nextSSA ctx
                EmitContext.emitLine ctx (sprintf "%s = arith.constant %d : i32" ssa length)
                Value (ssa, "i32")
            | _ ->
                // Runtime string/array - delegate to SemanticPrimitive if transformed by nanopass
                // Otherwise fall through (should not happen after LowerStringLength nanopass)
                printfn "[GEN] Warning: PropertyAccess:Length reached code gen without transformation"
                let ssa = EmitContext.nextSSA ctx
                EmitContext.emitLine ctx (sprintf "%s = arith.constant 0 : i32" ssa)
                Value (ssa, "i32")
        | _ ->
            printfn "[GEN] Unknown property access: %s" propName
            Void

/// Generate MLIR for semantic primitives (set by LowerStringLength nanopass)
/// These are property accesses that have been lowered to fidelity primitive calls
and genSemanticPrimitive (ctx: EmitContext) (node: PSGNode) : ExprResult =
    let kind = node.SyntaxKind
    let primName = if kind.StartsWith("SemanticPrimitive:") then kind.Substring(18) else ""
    let children = getChildren ctx node

    match primName, children with
    | "fidelity_strlen", [target] ->
        // Generate the target (string pointer)
        let targetResult = genNode ctx target
        match targetResult with
        | Value (targetSsa, "!llvm.ptr") ->
            // Inline strlen implementation - counts bytes until null terminator
            let result = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = llvm.inline_asm \"xor %%rcx, %%rcx\\0A1: cmpb $$0, (%%rdi,%%rcx)\\0Aje 2f\\0Ainc %%rcx\\0Ajmp 1b\\0A2:\", \"={rcx},{rdi}\" %s : (!llvm.ptr) -> i64" result targetSsa)
            let truncResult = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = arith.trunci %s : i64 to i32" truncResult result)
            Value (truncResult, "i32")
        | _ ->
            printfn "[GEN] SemanticPrimitive:fidelity_strlen - target is not !llvm.ptr"
            Void
    | _ ->
        printfn "[GEN] Unknown semantic primitive: %s" primName
        Void

/// Generate MLIR for sequential expression
and genSequential (ctx: EmitContext) (node: PSGNode) : ExprResult =
    let children = getChildren ctx node
    children |> List.fold (fun _ child -> genNode ctx child) Void

/// Generate MLIR for let binding
and genLetBinding (ctx: EmitContext) (node: PSGNode) : ExprResult =
    let children = getChildren ctx node

    match children with
    | [bindingNode; continuation] when bindingNode.SyntaxKind.StartsWith("Binding") ->
        let isMutable = bindingNode.SyntaxKind.Contains("Mutable")
        let bindingChildren = getChildren ctx bindingNode

        match bindingChildren with
        | [pattern; value] ->
            if isMutable then
                let valueResult = genNode ctx value
                match valueResult with
                | Value (valueSsa, valueTy) ->
                    let sizeSsa = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = arith.constant 1 : i64" sizeSsa)
                    let allocSsa = EmitContext.nextSSA ctx
                    EmitContext.emitLine ctx (sprintf "%s = llvm.alloca %s x %s : (i64) -> !llvm.ptr" allocSsa sizeSsa valueTy)
                    EmitContext.emitLine ctx (sprintf "llvm.store %s, %s : %s, !llvm.ptr" valueSsa allocSsa valueTy)
                    EmitContext.recordNodeSSA ctx bindingNode.Id allocSsa "!llvm.ptr"
                    genNode ctx continuation
                | _ -> genNode ctx continuation
            else
                let valueResult = genNode ctx value
                match valueResult with
                | Value (ssa, ty) ->
                    EmitContext.recordNodeSSA ctx bindingNode.Id ssa ty
                    genNode ctx continuation
                | _ -> genNode ctx continuation
        | _ -> genNode ctx continuation

    | [pattern; value] when node.SyntaxKind.StartsWith("Binding") ->
        genNode ctx value

    | [pattern; value; continuation] ->
        let valueResult = genNode ctx value
        match valueResult with
        | Value (ssa, ty) ->
            EmitContext.recordNodeSSA ctx node.Id ssa ty
            genNode ctx continuation
        | _ -> genNode ctx continuation

    | child :: _ -> genNode ctx child
    | [] -> Void

/// Generate MLIR for if-then-else
and genIfThenElse (ctx: EmitContext) (node: PSGNode) : ExprResult =
    let children = getChildren ctx node

    match children with
    | [condition; thenBranch; elseBranch] ->
        genIfThenElseFull ctx condition thenBranch (Some elseBranch)
    | [condition; thenBranch] ->
        genIfThenElseFull ctx condition thenBranch None
    | _ -> EmitError "Malformed IfThenElse node"

/// Generate full if-then-else with basic blocks
and genIfThenElseFull (ctx: EmitContext) (condition: PSGNode) (thenBranch: PSGNode) (elseBranch: PSGNode option) : ExprResult =
    let condResult = genNode ctx condition

    match condResult with
    | Value (condSsa, _) ->
        let thenLabel = EmitContext.nextLabel ctx
        let elseLabel = EmitContext.nextLabel ctx
        let mergeLabel = EmitContext.nextLabel ctx

        let branchLine =
            match elseBranch with
            | Some _ -> sprintf "llvm.cond_br %s, %s, %s" condSsa thenLabel elseLabel
            | None -> sprintf "llvm.cond_br %s, %s, %s" condSsa thenLabel mergeLabel
        EmitContext.emitLine ctx branchLine

        EmitContext.emitLine ctx (sprintf "%s:" thenLabel)
        let thenResult = genNode ctx thenBranch
        EmitContext.emitLine ctx (sprintf "llvm.br %s" mergeLabel)

        let elseResult =
            match elseBranch with
            | Some eb ->
                EmitContext.emitLine ctx (sprintf "%s:" elseLabel)
                let r = genNode ctx eb
                EmitContext.emitLine ctx (sprintf "llvm.br %s" mergeLabel)
                r
            | None -> Void

        EmitContext.emitLine ctx (sprintf "%s:" mergeLabel)

        match thenResult, elseResult with
        | Value (thenSsa, thenTy), Value (_, elseTy) when thenTy = elseTy ->
            Value (thenSsa, thenTy)
        | _ -> Void
    | _ -> EmitError "IfThenElse condition did not produce a value"

/// Generate MLIR for function application
and genApp (ctx: EmitContext) (node: PSGNode) : ExprResult =
    let children = getChildren ctx node

    match children with
    | funcNode :: argNodes ->
        match node.SRTPResolution with
        | Some (FSMethodByName methodFullName) ->
            printfn "[GEN] SRTP resolved to: %s" methodFullName
            genInlinedCallByName ctx methodFullName argNodes
        | Some (FSMethod (_, mfv, _)) ->
            printfn "[GEN] SRTP resolved to FSMethod: %s" (try mfv.FullName with _ -> "?")
            genInlinedCallByName ctx (try mfv.FullName with _ -> "") argNodes
        | Some BuiltIn ->
            genBuiltInOperator ctx funcNode argNodes
        | Some (Unresolved reason) ->
            printfn "[GEN] SRTP unresolved: %s" reason
            genRegularApp ctx funcNode argNodes
        | _ -> genRegularApp ctx funcNode argNodes
    | [] -> Void

/// Generate for regular (non-SRTP) function application
and genRegularApp (ctx: EmitContext) (funcNode: PSGNode) (argNodes: PSGNode list) : ExprResult =
    if isNativeInteropIntrinsic funcNode then
        genNativeInteropIntrinsic ctx funcNode argNodes
    elif isExternPrimitive funcNode then
        genExternPrimitive ctx funcNode argNodes
    else
        match isFSharpCoreOperator funcNode with
        | Some (mfv, opKind) ->
            genFSharpCoreOperator ctx mfv opKind argNodes
        | None ->
            match tryExtractCurriedOperator ctx funcNode with
            | Some (mfv, opKind, firstArg) ->
                genFSharpCoreOperator ctx mfv opKind (firstArg :: argNodes)
            | None ->
                genInlinedCall ctx funcNode argNodes

/// Try to extract a curried FSharp.Core operator
and tryExtractCurriedOperator (ctx: EmitContext) (funcNode: PSGNode) : (FSharpMemberOrFunctionOrValue * string * PSGNode) option =
    if funcNode.SyntaxKind.StartsWith("App") then
        let children = getChildren ctx funcNode
        match children with
        | opNode :: [firstArg] ->
            match isFSharpCoreOperator opNode with
            | Some (mfv, opKind) -> Some (mfv, opKind, firstArg)
            | None -> None
        | _ -> None
    else None

/// Generate MLIR for FSharp.Core built-in operators
and genFSharpCoreOperator (ctx: EmitContext) (mfv: FSharpMemberOrFunctionOrValue) (opKind: string) (argNodes: PSGNode list) : ExprResult =
    let flatArgs = flattenTupleArgs ctx argNodes
    let argResults = flatArgs |> List.map (genNode ctx)
    let argSsas = argResults |> List.choose (function Value (s, t) -> Some (s, t) | _ -> None)

    printfn "[GEN] FSharp.Core operator: %s (%s) with %d args" mfv.CompiledName opKind argSsas.Length

    match opKind, argSsas with
    | "arith", [(left, leftTy); (right, _)] ->
        match arithmeticOp mfv with
        | Some mlirOp ->
            let ssa = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = %s %s, %s : %s" ssa mlirOp left right leftTy)
            Value (ssa, leftTy)
        | None ->
            printfn "[GEN] Unknown arithmetic operator: %s" mfv.CompiledName
            Void

    | "compare", [(left, leftTy); (right, _)] ->
        match comparisonOp mfv with
        | Some predicate ->
            let ssa = EmitContext.nextSSA ctx
            let line =
                if leftTy = "!llvm.ptr" then
                    let llvmPred = match predicate with
                                   | "eq" -> "eq" | "ne" -> "ne" | "slt" -> "slt"
                                   | "sle" -> "sle" | "sgt" -> "sgt" | "sge" -> "sge"
                                   | _ -> predicate
                    sprintf "%s = llvm.icmp \"%s\" %s, %s : !llvm.ptr" ssa llvmPred left right
                else
                    sprintf "%s = arith.cmpi \"%s\", %s, %s : %s" ssa predicate left right leftTy
            EmitContext.emitLine ctx line
            Value (ssa, "i1")
        | None ->
            printfn "[GEN] Unknown comparison operator: %s" mfv.CompiledName
            Void
    | _ ->
        printfn "[GEN] FSharp.Core operator with wrong arity: %s has %d args" mfv.CompiledName argSsas.Length
        Void

/// Generate for built-in SRTP operator
and genBuiltInOperator (ctx: EmitContext) (funcNode: PSGNode) (argNodes: PSGNode list) : ExprResult =
    genRegularApp ctx funcNode argNodes

/// Find all parameter definition nodes from a function pattern
and findParameterNodes (ctx: EmitContext) (pattern: PSGNode) : PSGNode list =
    if pattern.SyntaxKind.StartsWith("Pattern:Named:") then
        [pattern]
    elif pattern.SyntaxKind.StartsWith("Pattern:LongIdent:") then
        // Parameters are children of LongIdent pattern
        getChildren ctx pattern |> List.collect (findParameterNodes ctx)
    elif pattern.SyntaxKind = "Pattern:Paren" || pattern.SyntaxKind = "Pattern:Tuple" then
        getChildren ctx pattern |> List.collect (findParameterNodes ctx)
    else
        getChildren ctx pattern |> List.collect (findParameterNodes ctx)

/// Bind argument SSA values to parameter definition nodes
and bindArgumentsToParameters (ctx: EmitContext) (paramNodes: PSGNode list) (argSsas: (string * string) list) : unit =
    List.zip paramNodes argSsas
    |> List.iter (fun (paramNode, (ssa, ty)) ->
        printfn "[GEN] Binding parameter %s -> %s : %s" paramNode.SyntaxKind ssa ty
        EmitContext.recordNodeSSA ctx paramNode.Id ssa ty)

/// Inline a function call by finding it by name in the PSG
and genInlinedCallByName (ctx: EmitContext) (methodFullName: string) (argNodes: PSGNode list) : ExprResult =
    let flatArgs = flattenTupleArgs ctx argNodes
    let argResults = flatArgs |> List.map (genNode ctx)
    let argSsas = argResults |> List.choose (function Value (s, t) -> Some (s, t) | _ -> None)

    let funcBinding =
        ctx.Graph.Nodes
        |> Map.toSeq
        |> Seq.tryFind (fun (_, n) ->
            n.SyntaxKind.StartsWith("Binding") &&
            match n.Symbol with
            | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
                try mfv.FullName = methodFullName with _ -> false
            | _ -> false)
        |> Option.map snd

    match funcBinding with
    | Some binding ->
        let bindingChildren = getChildren ctx binding
        match bindingChildren with
        | [pattern; body] ->
            printfn "[GEN] Inlining SRTP-resolved function %s" methodFullName
            // Find parameter nodes and bind argument values to them
            let paramNodes = findParameterNodes ctx pattern
            if paramNodes.Length = argSsas.Length then
                bindArgumentsToParameters ctx paramNodes argSsas
            else
                printfn "[GEN] Warning: parameter count mismatch: %d params, %d args" paramNodes.Length argSsas.Length
            genNode ctx body
        | _ ->
            printfn "[GEN] Malformed SRTP function binding for: %s" methodFullName
            Void
    | None ->
        printfn "[GEN] SRTP function not found in PSG: %s" methodFullName
        Void

/// Generate for NativeInterop intrinsics
and genNativeInteropIntrinsic (ctx: EmitContext) (funcNode: PSGNode) (argNodes: PSGNode list) : ExprResult =
    let intrinsicName = getNativeInteropIntrinsicName funcNode |> Option.defaultValue "unknown"
    let flatArgs = flattenTupleArgs ctx argNodes
    let argResults = flatArgs |> List.map (genNode ctx)
    let argSsas = argResults |> List.choose (function Value (s, t) -> Some (s, t) | _ -> None)

    match intrinsicName with
    | "toNativeInt" | "toVoidPtr" | "ofVoidPtr" | "ofNativeInt" ->
        match argSsas with
        | [(ssa, _)] when ssa <> "" -> Value (ssa, "!llvm.ptr")
        | _ ->
            let ssa = EmitContext.nextSSA ctx
            Value (ssa, "!llvm.ptr")
    | _ ->
        printfn "[GEN] Unknown NativeInterop intrinsic: %s" intrinsicName
        Void

/// Generate for extern primitive (DllImport)
and genExternPrimitive (ctx: EmitContext) (funcNode: PSGNode) (argNodes: PSGNode list) : ExprResult =
    match tryExtractExternPrimitiveInfo funcNode with
    | Some info when info.Library = "__fidelity" ->
        genFidelityExtern ctx info argNodes
    | Some info ->
        printfn "[GEN] Non-fidelity extern: %s/%s" info.Library info.EntryPoint
        Void
    | None -> Void

/// Generate Fidelity extern call (e.g., fidelity_write_bytes -> syscall)
and genFidelityExtern (ctx: EmitContext) (info: ExternPrimitiveInfo) (argNodes: PSGNode list) : ExprResult =
    let flatArgs = flattenTupleArgs ctx argNodes
    let argResults = flatArgs |> List.map (genNode ctx)
    let argSsas = argResults |> List.choose (function Value (s, t) -> Some (s, t) | _ -> None)

    match info.EntryPoint with
    | "fidelity_write_bytes" ->
        match argSsas with
        | [(fdSsa, _); (bufSsa, _); (countSsa, _)] ->
            let fdExt = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = arith.extsi %s : i32 to i64" fdExt fdSsa)
            let countExt = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = arith.extsi %s : i32 to i64" countExt countSsa)
            let sysNum = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = arith.constant 1 : i64" sysNum)
            let result = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64" result sysNum fdExt bufSsa countExt)
            let truncResult = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = arith.trunci %s : i64 to i32" truncResult result)
            Value (truncResult, "i32")
        | _ ->
            printfn "[GEN] fidelity_write_bytes: expected 3 args, got %d" (List.length argSsas)
            Void

    | "fidelity_read_bytes" ->
        match argSsas with
        | [(fdSsa, _); (bufSsa, _); (countSsa, _)] ->
            let fdExt = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = arith.extsi %s : i32 to i64" fdExt fdSsa)
            let countExt = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = arith.extsi %s : i32 to i64" countExt countSsa)
            let sysNum = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = arith.constant 0 : i64" sysNum)
            let result = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}\" %s, %s, %s, %s : (i64, i64, !llvm.ptr, i64) -> i64" result sysNum fdExt bufSsa countExt)
            let truncResult = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = arith.trunci %s : i64 to i32" truncResult result)
            Value (truncResult, "i32")
        | _ -> Void

    | "fidelity_strlen" ->
        // Inline strlen: count bytes until null terminator
        // Uses x86_64 assembly: xor rcx,rcx; loop: cmpb $0,(rdi,rcx); je done; inc rcx; jmp loop; done:
        match argSsas with
        | [(strSsa, _)] ->
            let result = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = llvm.inline_asm \"xor %%rcx, %%rcx\\0A1: cmpb $$0, (%%rdi,%%rcx)\\0Aje 2f\\0Ainc %%rcx\\0Ajmp 1b\\0A2:\", \"={rcx},{rdi}\" %s : (!llvm.ptr) -> i64" result strSsa)
            let truncResult = EmitContext.nextSSA ctx
            EmitContext.emitLine ctx (sprintf "%s = arith.trunci %s : i64 to i32" truncResult result)
            Value (truncResult, "i32")
        | _ ->
            printfn "[GEN] fidelity_strlen: expected 1 arg, got %d" (List.length argSsas)
            Void

    | _ ->
        printfn "[GEN] Unknown fidelity extern: %s" info.EntryPoint
        Void

/// Generate inlined function call
and genInlinedCall (ctx: EmitContext) (funcNode: PSGNode) (argNodes: PSGNode list) : ExprResult =
    let flatArgs = flattenTupleArgs ctx argNodes
    let argResults = flatArgs |> List.map (genNode ctx)
    let argSsas = argResults |> List.choose (function Value (s, t) -> Some (s, t) | _ -> None)

    let funcName =
        match funcNode.Symbol with
        | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
            try Some mfv.FullName with _ -> None
        | _ -> None

    match funcName with
    | Some name ->
        let funcBinding =
            ctx.Graph.Nodes
            |> Map.toSeq
            |> Seq.tryFind (fun (_, n) ->
                n.SyntaxKind.StartsWith("Binding") &&
                match n.Symbol with
                | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
                    try mfv.FullName = name with _ -> false
                | _ -> false)
            |> Option.map snd

        match funcBinding with
        | Some binding ->
            let bindingChildren = getChildren ctx binding
            match bindingChildren with
            | [pattern; body] ->
                printfn "[GEN] Inlining function %s" name
                // Bind argument values to parameter nodes
                let paramNodes = findParameterNodes ctx pattern
                if paramNodes.Length = argSsas.Length then
                    bindArgumentsToParameters ctx paramNodes argSsas
                else
                    printfn "[GEN] Warning: parameter count mismatch: %d params, %d args" paramNodes.Length argSsas.Length
                genNode ctx body
            | _ ->
                printfn "[GEN] Malformed function binding for: %s" name
                Void
        | None ->
            printfn "[GEN] Function not found in PSG: %s" name
            Void
    | None -> Void

/// Generate MLIR for tuple
and genTuple (ctx: EmitContext) (node: PSGNode) : ExprResult =
    let children = getChildren ctx node
    match children with
    | [] -> Void
    | _ -> children |> List.fold (fun _ child -> genNode ctx child) Void

/// Generate MLIR for address-of
and genAddressOf (ctx: EmitContext) (node: PSGNode) : ExprResult =
    let children = getChildren ctx node
    match children with
    | [child] ->
        let result = genNode ctx child
        match result with
        | Value (ssa, _) -> Value (ssa, "!llvm.ptr")
        | _ -> EmitError "AddressOf child did not produce value"
    | _ -> Void

/// Generate MLIR for type application (passthrough)
and genTypeApp (ctx: EmitContext) (node: PSGNode) : ExprResult =
    let children = getChildren ctx node
    match children with
    | [child] -> genNode ctx child
    | _ -> Void

// ═══════════════════════════════════════════════════════════════════
// Entry Point Generation
// ═══════════════════════════════════════════════════════════════════

/// Generate MLIR for an entry point function
let genEntryPoint (ctx: EmitContext) (entryNode: PSGNode) : unit =
    EmitContext.emitLine ctx "llvm.func @main() -> i32 {"
    EmitContext.emitLine ctx "  ^entry:"

    let children = getChildren ctx entryNode
    let body =
        match children with
        | [_pattern; body] -> Some body
        | _ -> None

    match body with
    | Some b -> genNode ctx b |> ignore
    | None -> ()

    let retVal = EmitContext.nextSSA ctx
    EmitContext.emitLine ctx (sprintf "  %s = arith.constant 0 : i32" retVal)
    EmitContext.emitLine ctx (sprintf "  llvm.return %s : i32" retVal)
    EmitContext.emitLine ctx "}"

/// Generate complete MLIR module from PSG
let generateMLIR (psg: ProgramSemanticGraph) (target: string) : string =
    let ctx = EmitContext.create psg
    let header = sprintf "// Firefly-generated MLIR\n// Target: %s\n\nmodule {\n" target

    let entryPoints = psg.EntryPoints |> List.choose (fun id -> Map.tryFind id.Value psg.Nodes)
    entryPoints |> List.iter (genEntryPoint ctx)

    let body = EmitContext.getOutput ctx

    let externalFuncDecls =
        ctx.ExternalFuncs
        |> Set.toList
        |> List.map (fun name ->
            match name with
            | "strlen" -> "llvm.func @strlen(!llvm.ptr) -> i64"
            | _ -> sprintf "// Unknown external: %s" name)
        |> String.concat "\n"

    let stringLiterals =
        ctx.StringLiterals
        |> List.map (fun (content, name) ->
            let escaped = content.Replace("\\", "\\\\").Replace("\"", "\\\"")
            let len = content.Length + 1
            sprintf "llvm.mlir.global internal constant %s(\"%s\\00\") : !llvm.array<%d x i8>" name escaped len)
        |> String.concat "\n"

    let footer = "\n}\n"
    header + body + "\n" + externalFuncDecls + "\n" + stringLiterals + footer
