/// ExpressionEmitter - XParsec + MLIR CE based expression emission
///
/// ARCHITECTURE:
/// Pattern matching via XParsec combinators produces MLIR computations.
/// - XParsec handles pattern recognition on PSG structure
/// - MLIR CE handles code generation (NO sprintf in this module)
///
/// The PSG nanopass ClassifyOperations has already set node.Operation
/// on all App nodes. This emitter dispatches on that classification.
module Alex.Emission.ExpressionEmitter

open Core.PSG.Types
open Alex.CodeGeneration.MLIRBuilder
open Alex.CodeGeneration.TypeMapping

// ═══════════════════════════════════════════════════════════════════
// Result Type - What emission produces
// ═══════════════════════════════════════════════════════════════════

/// Result of emitting an expression
type EmitResult =
    | Emitted of Val
    | Void
    | Error of string

// ═══════════════════════════════════════════════════════════════════
// Constant Extraction (from SyntaxKind metadata)
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

let private extractInt64FromKind (kind: string) : int64 option =
    if kind.StartsWith("Const:Int64 ") then
        let numStr = kind.Substring(12).Trim().TrimEnd('L')
        match System.Int64.TryParse(numStr) with
        | true, n -> Some n
        | _ -> None
    else None

let private extractByteFromKind (kind: string) : byte option =
    if kind.StartsWith("Const:Byte ") then
        let numStr = kind.Substring(11).Trim().TrimEnd('u', 'y')
        match System.Byte.TryParse(numStr) with
        | true, b -> Some b
        | _ -> None
    else None

let private extractBoolFromKind (kind: string) : bool option =
    if kind = "Const:Bool true" then Some true
    elif kind = "Const:Bool false" then Some false
    else None

// ═══════════════════════════════════════════════════════════════════
// Helper: Get Children from PSG Node
// ═══════════════════════════════════════════════════════════════════

let private getChildNodes (psg: ProgramSemanticGraph) (node: PSGNode) : PSGNode list =
    match node.Children with
    | Parent ids -> ids |> List.choose (fun id -> Map.tryFind id.Value psg.Nodes)
    | _ -> []

// ═══════════════════════════════════════════════════════════════════
// Constant Emitters
// ═══════════════════════════════════════════════════════════════════

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
    match extractInt64FromKind kind with
    | Some n ->
        mlir {
            let! v = arith.constant n I64
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
            let! v = arith.constBool b
            return Emitted v
        }
    | None ->
    if kind = "Const:Unit" || kind.Contains("Unit") then
        mlir { return Void }
    else
        mlir { return Error ("Unknown constant: " + kind) }

// ═══════════════════════════════════════════════════════════════════
// Helper: Emit and check for Emitted value
// ═══════════════════════════════════════════════════════════════════

let private withEmitted (result: EmitResult) (f: Val -> MLIR<EmitResult>) : MLIR<EmitResult> =
    match result with
    | Emitted v -> f v
    | Void -> mlir { return Error "Expected value but got Void" }
    | Error msg -> mlir { return Error msg }

let private withEmitted2 (r1: EmitResult) (r2: EmitResult) (f: Val -> Val -> MLIR<EmitResult>) : MLIR<EmitResult> =
    match r1, r2 with
    | Emitted v1, Emitted v2 -> f v1 v2
    | Error msg, _ -> mlir { return Error msg }
    | _, Error msg -> mlir { return Error msg }
    | _ -> mlir { return Error "Expected two values" }

let private withEmitted3 (r1: EmitResult) (r2: EmitResult) (r3: EmitResult) (f: Val -> Val -> Val -> MLIR<EmitResult>) : MLIR<EmitResult> =
    match r1, r2, r3 with
    | Emitted v1, Emitted v2, Emitted v3 -> f v1 v2 v3
    | Error msg, _, _ -> mlir { return Error msg }
    | _, Error msg, _ -> mlir { return Error msg }
    | _, _, Error msg -> mlir { return Error msg }
    | _ -> mlir { return Error "Expected three values" }

// ═══════════════════════════════════════════════════════════════════
// Console Operation Emitters
// ═══════════════════════════════════════════════════════════════════

let emitConsoleWrite (args: PSGNode list) (emitExpr: PSGNode -> MLIR<EmitResult>) : MLIR<EmitResult> =
    match args with
    | [strArg] ->
        mlir {
            let! argResult = emitExpr strArg
            return! withEmitted argResult (fun strVal ->
                mlir {
                    let! (ptr, len) = extractNativeStr strVal
                    let! fd = arith.constant 1L I32
                    let! _ = func.call "write_syscall" [fd; ptr; len] [Int I32; Ptr; Int I64] (Int I64)
                    return Void
                })
        }
    | _ -> mlir { return Error "Console.Write: expected 1 argument" }

let emitConsoleWriteln (args: PSGNode list) (emitExpr: PSGNode -> MLIR<EmitResult>) : MLIR<EmitResult> =
    match args with
    | [strArg] ->
        mlir {
            let! argResult = emitExpr strArg
            return! withEmitted argResult (fun strVal ->
                mlir {
                    let! (ptr, len) = extractNativeStr strVal
                    let! fd = arith.constant 1L I32
                    let! _ = func.call "write_syscall" [fd; ptr; len] [Int I32; Ptr; Int I64] (Int I64)
                    let! nlByte = arith.constant 10L I8
                    let! allocSize = arith.constant 1L I64
                    let! nlPtr = llvm.alloca allocSize (Int I8)
                    do! llvm.store nlByte nlPtr
                    let! one = arith.constant 1L I64
                    let! _ = func.call "write_syscall" [fd; nlPtr; one] [Int I32; Ptr; Int I64] (Int I64)
                    return Void
                })
        }
    | _ -> mlir { return Error "Console.WriteLine: expected 1 argument" }

let emitConsoleNewLine () : MLIR<EmitResult> =
    mlir {
        let! fd = arith.constant 1L I32
        let! nlByte = arith.constant 10L I8
        let! allocSize = arith.constant 1L I64
        let! nlPtr = llvm.alloca allocSize (Int I8)
        do! llvm.store nlByte nlPtr
        let! one = arith.constant 1L I64
        let! _ = func.call "write_syscall" [fd; nlPtr; one] [Int I32; Ptr; Int I64] (Int I64)
        return Void
    }

let emitConsoleReadInto (args: PSGNode list) (emitExpr: PSGNode -> MLIR<EmitResult>) : MLIR<EmitResult> =
    match args with
    | [bufArg] ->
        mlir {
            let! argResult = emitExpr bufArg
            return! withEmitted argResult (fun bufVal ->
                mlir {
                    let! (ptr, cap) = extractNativeStr bufVal
                    let! fd = arith.constant 0L I32
                    let! result = func.call "read_syscall" [fd; ptr; cap] [Int I32; Ptr; Int I64] (Int I64)
                    return Emitted result
                })
        }
    | _ -> mlir { return Error "Console.readInto: expected 1 argument" }

let emitConsoleWriteBytes (args: PSGNode list) (emitExpr: PSGNode -> MLIR<EmitResult>) : MLIR<EmitResult> =
    match args with
    | [fdArg; ptrArg; countArg] ->
        mlir {
            let! fdResult = emitExpr fdArg
            let! ptrResult = emitExpr ptrArg
            let! countResult = emitExpr countArg
            return! withEmitted3 fdResult ptrResult countResult (fun fdVal ptrVal countVal ->
                mlir {
                    let! result = func.call "write_syscall" [fdVal; ptrVal; countVal] [Int I32; Ptr; Int I64] (Int I64)
                    return Emitted result
                })
        }
    | _ -> mlir { return Error "Console.writeBytes: expected 3 arguments" }

let emitConsoleReadBytes (args: PSGNode list) (emitExpr: PSGNode -> MLIR<EmitResult>) : MLIR<EmitResult> =
    match args with
    | [fdArg; ptrArg; countArg] ->
        mlir {
            let! fdResult = emitExpr fdArg
            let! ptrResult = emitExpr ptrArg
            let! countResult = emitExpr countArg
            return! withEmitted3 fdResult ptrResult countResult (fun fdVal ptrVal countVal ->
                mlir {
                    let! result = func.call "read_syscall" [fdVal; ptrVal; countVal] [Int I32; Ptr; Int I64] (Int I64)
                    return Emitted result
                })
        }
    | _ -> mlir { return Error "Console.readBytes: expected 3 arguments" }

// ═══════════════════════════════════════════════════════════════════
// Arithmetic Operation Emitters
// ═══════════════════════════════════════════════════════════════════

let emitArithmeticOp (op: ArithmeticOp) (args: PSGNode list) (emitExpr: PSGNode -> MLIR<EmitResult>) : MLIR<EmitResult> =
    match op, args with
    | Negate, [argNode] ->
        mlir {
            let! argResult = emitExpr argNode
            return! withEmitted argResult (fun arg ->
                mlir {
                    let ty = match arg.Type with Int it -> it | _ -> I32
                    let! zero = arith.constant 0L ty
                    let! result = arith.subi zero arg ty
                    return Emitted result
                })
        }
    | _, [lhsNode; rhsNode] ->
        mlir {
            let! lhsResult = emitExpr lhsNode
            let! rhsResult = emitExpr rhsNode
            return! withEmitted2 lhsResult rhsResult (fun lhs rhs ->
                mlir {
                    let ty = match lhs.Type with Int it -> it | _ -> I32
                    let! result =
                        match op with
                        | Add -> arith.addi lhs rhs ty
                        | Sub -> arith.subi lhs rhs ty
                        | Mul -> arith.muli lhs rhs ty
                        | Div -> arith.divsi lhs rhs ty
                        | Mod -> arith.remsi lhs rhs ty
                        | Negate ->
                            mlir {
                                let! zero = arith.constant 0L ty
                                return! arith.subi zero rhs ty
                            }
                    return Emitted result
                })
        }
    | _ -> mlir { return Error "Arithmetic op: wrong number of arguments" }

// ═══════════════════════════════════════════════════════════════════
// Comparison Operation Emitters
// ═══════════════════════════════════════════════════════════════════

let emitComparisonOp (op: ComparisonOp) (args: PSGNode list) (emitExpr: PSGNode -> MLIR<EmitResult>) : MLIR<EmitResult> =
    match args with
    | [lhsNode; rhsNode] ->
        mlir {
            let! lhsResult = emitExpr lhsNode
            let! rhsResult = emitExpr rhsNode
            return! withEmitted2 lhsResult rhsResult (fun lhs rhs ->
                mlir {
                    let ty = match lhs.Type with Int it -> it | _ -> I32
                    let pred =
                        match op with
                        | ComparisonOp.Eq -> ICmp.Eq | ComparisonOp.Neq -> ICmp.Ne
                        | ComparisonOp.Lt -> ICmp.Slt | ComparisonOp.Lte -> ICmp.Sle
                        | ComparisonOp.Gt -> ICmp.Sgt | ComparisonOp.Gte -> ICmp.Sge
                    let! result = arith.cmpi pred lhs rhs ty
                    return Emitted result
                })
        }
    | _ -> mlir { return Error "Comparison expects 2 arguments" }

// ═══════════════════════════════════════════════════════════════════
// Core Operation Emitters
// ═══════════════════════════════════════════════════════════════════

let emitCoreOp (op: CoreOp) (args: PSGNode list) (emitExpr: PSGNode -> MLIR<EmitResult>) : MLIR<EmitResult> =
    match op with
    | Ignore ->
        match args with
        | [arg] ->
            mlir {
                let! _ = emitExpr arg
                return Void
            }
        | _ -> mlir { return Void }
    | Not ->
        match args with
        | [arg] ->
            mlir {
                let! argResult = emitExpr arg
                return! withEmitted argResult (fun v ->
                    mlir {
                        let! trueVal = arith.constBool true
                        let! result = arith.xori v trueVal I1
                        return Emitted result
                    })
            }
        | _ -> mlir { return Error "not: expected 1 argument" }
    | Failwith | InvalidArg -> mlir { return Error "failwith/invalidArg called" }

// ═══════════════════════════════════════════════════════════════════
// NativePtr Operation Emitters
// ═══════════════════════════════════════════════════════════════════

let emitNativePtrOp (op: NativePtrOp) (args: PSGNode list) (emitExpr: PSGNode -> MLIR<EmitResult>) : MLIR<EmitResult> =
    match op with
    | PtrStackAlloc ->
        match args with
        | [countArg] ->
            mlir {
                let! countResult = emitExpr countArg
                return! withEmitted countResult (fun countVal ->
                    mlir {
                        let! ptr = llvm.alloca countVal (Int I8)
                        let bufTy = Struct [Ptr; Int I64]
                        let! undef = llvm.undef bufTy
                        let! v1 = llvm.insertvalue undef ptr 0 bufTy
                        let! v2 = llvm.insertvalue v1 countVal 1 bufTy
                        return Emitted v2
                    })
            }
        | _ -> mlir { return Error "stackalloc: expected 1 argument" }
    | PtrNull ->
        mlir {
            let! null_ = llvm.zero Ptr
            return Emitted null_
        }
    | PtrRead ->
        match args with
        | [ptrArg] ->
            mlir {
                let! ptrResult = emitExpr ptrArg
                return! withEmitted ptrResult (fun ptrVal ->
                    mlir {
                        let! result = llvm.load ptrVal (Int I8)
                        return Emitted result
                    })
            }
        | _ -> mlir { return Error "ptr.read: expected 1 argument" }
    | PtrWrite ->
        match args with
        | [ptrArg; valueArg] ->
            mlir {
                let! ptrResult = emitExpr ptrArg
                let! valueResult = emitExpr valueArg
                return! withEmitted2 ptrResult valueResult (fun ptrVal valueVal ->
                    mlir {
                        do! llvm.store valueVal ptrVal
                        return Void
                    })
            }
        | _ -> mlir { return Error "ptr.write: expected 2 arguments" }
    | PtrGet ->
        match args with
        | [ptrArg; indexArg] ->
            mlir {
                let! ptrResult = emitExpr ptrArg
                let! indexResult = emitExpr indexArg
                return! withEmitted2 ptrResult indexResult (fun ptrVal indexVal ->
                    mlir {
                        let! elemPtr = llvm.gep ptrVal indexVal (Int I8)
                        let! result = llvm.load elemPtr (Int I8)
                        return Emitted result
                    })
            }
        | _ -> mlir { return Error "ptr.get: expected 2 arguments" }
    | PtrSet ->
        match args with
        | [ptrArg; indexArg; valueArg] ->
            mlir {
                let! ptrResult = emitExpr ptrArg
                let! indexResult = emitExpr indexArg
                let! valueResult = emitExpr valueArg
                return! withEmitted3 ptrResult indexResult valueResult (fun ptrVal indexVal valueVal ->
                    mlir {
                        let! elemPtr = llvm.gep ptrVal indexVal (Int I8)
                        do! llvm.store valueVal elemPtr
                        return Void
                    })
            }
        | _ -> mlir { return Error "ptr.set: expected 3 arguments" }
    | PtrAdd ->
        match args with
        | [ptrArg; offsetArg] ->
            mlir {
                let! ptrResult = emitExpr ptrArg
                let! offsetResult = emitExpr offsetArg
                return! withEmitted2 ptrResult offsetResult (fun ptrVal offsetVal ->
                    mlir {
                        let! result = llvm.gep ptrVal offsetVal (Int I8)
                        return Emitted result
                    })
            }
        | _ -> mlir { return Error "ptr.add: expected 2 arguments" }
    | _ -> mlir { return Error ("NativePtr op not implemented: " + op.ToString()) }

// ═══════════════════════════════════════════════════════════════════
// NativeStr Operation Emitters
// ═══════════════════════════════════════════════════════════════════

let emitNativeStrOp (op: NativeStrOp) (args: PSGNode list) (emitExpr: PSGNode -> MLIR<EmitResult>) : MLIR<EmitResult> =
    match op with
    | StrEmpty ->
        mlir {
            let! null_ = llvm.zero Ptr
            let! zero = arith.constant 0L I64
            let! undef = llvm.undef nativeStrTy
            let! v1 = llvm.insertvalue undef null_ 0 nativeStrTy
            let! v2 = llvm.insertvalue v1 zero 1 nativeStrTy
            return Emitted v2
        }
    | StrCreate ->
        match args with
        | [ptrArg; lenArg] ->
            mlir {
                let! ptrResult = emitExpr ptrArg
                let! lenResult = emitExpr lenArg
                return! withEmitted2 ptrResult lenResult (fun ptrVal lenVal ->
                    mlir {
                        let! undef = llvm.undef nativeStrTy
                        let! v1 = llvm.insertvalue undef ptrVal 0 nativeStrTy
                        let! v2 = llvm.insertvalue v1 lenVal 1 nativeStrTy
                        return Emitted v2
                    })
            }
        | _ -> mlir { return Error "str.create: expected 2 arguments" }
    | StrLength ->
        match args with
        | [strArg] ->
            mlir {
                let! strResult = emitExpr strArg
                return! withEmitted strResult (fun strVal ->
                    mlir {
                        let! len = llvm.extractvalue strVal 1 nativeStrTy (Int I64)
                        return Emitted len
                    })
            }
        | _ -> mlir { return Error "str.length: expected 1 argument" }
    | _ -> mlir { return Error ("NativeStr op not implemented: " + op.ToString()) }

// ═══════════════════════════════════════════════════════════════════
// Memory Operation Emitters
// ═══════════════════════════════════════════════════════════════════

let emitMemoryOp (op: MemoryOp) (args: PSGNode list) (emitExpr: PSGNode -> MLIR<EmitResult>) : MLIR<EmitResult> =
    match op with
    | MemStackBuffer ->
        match args with
        | [sizeArg] ->
            mlir {
                let! sizeResult = emitExpr sizeArg
                return! withEmitted sizeResult (fun sizeVal ->
                    mlir {
                        let! ptr = llvm.alloca sizeVal (Int I8)
                        let! undef = llvm.undef nativeStrTy
                        let! v1 = llvm.insertvalue undef ptr 0 nativeStrTy
                        let! v2 = llvm.insertvalue v1 sizeVal 1 nativeStrTy
                        return Emitted v2
                    })
            }
        | _ -> mlir { return Error "stackBuffer: expected 1 argument" }
    | _ -> mlir { return Error ("Memory op not implemented: " + op.ToString()) }

// ═══════════════════════════════════════════════════════════════════
// Result Operation Emitters
// ═══════════════════════════════════════════════════════════════════

let emitResultOp (op: ResultOp) (args: PSGNode list) (emitExpr: PSGNode -> MLIR<EmitResult>) : MLIR<EmitResult> =
    let resultTy = Struct [Int I32; Int I64; Int I64]
    match op with
    | ResultOk ->
        mlir {
            let! tag = arith.constant 0L I32
            let! undef = llvm.undef resultTy
            let! v1 = llvm.insertvalue undef tag 0 resultTy
            let! zero = arith.constant 0L I64
            let! v2 = llvm.insertvalue v1 zero 1 resultTy
            let! v3 = llvm.insertvalue v2 zero 2 resultTy
            return Emitted v3
        }
    | ResultError ->
        mlir {
            let! tag = arith.constant 1L I32
            let! undef = llvm.undef resultTy
            let! v1 = llvm.insertvalue undef tag 0 resultTy
            let! zero = arith.constant 0L I64
            let! v2 = llvm.insertvalue v1 zero 1 resultTy
            let! v3 = llvm.insertvalue v2 zero 2 resultTy
            return Emitted v3
        }

// ═══════════════════════════════════════════════════════════════════
// Regular Function Call Emitter
// ═══════════════════════════════════════════════════════════════════

let emitRegularCall (info: RegularCallInfo) (args: PSGNode list) (emitExpr: PSGNode -> MLIR<EmitResult>) : MLIR<EmitResult> =
    let modulePath = info.ModulePath |> Option.defaultValue ""
    if modulePath.StartsWith("Alloy.") || modulePath.StartsWith("Microsoft.FSharp.") then
        mlir {
            let funcName =
                match info.ModulePath with
                | Some path -> path.Replace(".", "_") + "_" + info.FunctionName
                | None -> info.FunctionName
            do! func.callVoid funcName [] []
            return Void
        }
    else
        mlir {
            let funcName = info.FunctionName
            let mutable argVals : Val list = []
            for arg in args do
                let! argResult = emitExpr arg
                match argResult with
                | Emitted v -> argVals <- argVals @ [v]
                | _ -> ()
            let argTypes = argVals |> List.map (fun v -> v.Type)
            if List.isEmpty argVals then
                do! func.callVoid funcName [] []
                return Void
            else
                let! result = func.call funcName argVals argTypes (Int I32)
                return Emitted result
        }

// ═══════════════════════════════════════════════════════════════════
// Classified App Emitter
// ═══════════════════════════════════════════════════════════════════

let emitClassifiedApp (node: PSGNode) (psg: ProgramSemanticGraph) (emitExpr: PSGNode -> MLIR<EmitResult>) : MLIR<EmitResult> =
    let children = getChildNodes psg node
    let args = match children with | _ :: rest -> rest | [] -> []

    match node.Operation with
    | Some (Console op) ->
        match op with
        | ConsoleWrite -> emitConsoleWrite args emitExpr
        | ConsoleWriteln -> emitConsoleWriteln args emitExpr
        | ConsoleNewLine -> emitConsoleNewLine ()
        | ConsoleReadInto -> emitConsoleReadInto args emitExpr
        | ConsoleWriteBytes -> emitConsoleWriteBytes args emitExpr
        | ConsoleReadBytes -> emitConsoleReadBytes args emitExpr
        | _ -> mlir { return Error ("Console op not implemented: " + op.ToString()) }
    | Some (Arithmetic op) -> emitArithmeticOp op args emitExpr
    | Some (Comparison op) -> emitComparisonOp op args emitExpr
    | Some (Core op) -> emitCoreOp op args emitExpr
    | Some (NativePtr op) -> emitNativePtrOp op args emitExpr
    | Some (NativeStr op) -> emitNativeStrOp op args emitExpr
    | Some (Memory op) -> emitMemoryOp op args emitExpr
    | Some (Result op) -> emitResultOp op args emitExpr
    | Some (RegularCall info) -> emitRegularCall info args emitExpr
    | Some (Conversion _) -> mlir { return Error "Conversion ops not yet implemented" }
    | Some (Bitwise _) -> mlir { return Error "Bitwise ops not yet implemented" }
    | Some (Time _) -> mlir { return Error "Time ops not yet implemented" }
    | Some (TextFormat _) -> mlir { return Error "TextFormat ops not yet implemented" }
    | None -> mlir { return Error ("Unclassified App node: " + node.SyntaxKind) }

// ═══════════════════════════════════════════════════════════════════
// Sequential Expression Emitter
// ═══════════════════════════════════════════════════════════════════

let emitSequential (node: PSGNode) (psg: ProgramSemanticGraph) (emitExpr: PSGNode -> MLIR<EmitResult>) : MLIR<EmitResult> =
    let children = getChildNodes psg node
    match children with
    | [] -> mlir { return Void }
    | [single] -> emitExpr single
    | _ ->
        mlir {
            let mutable lastResult = Void
            for child in children do
                let! result = emitExpr child
                lastResult <- result
            return lastResult
        }

// ═══════════════════════════════════════════════════════════════════
// Let Binding Emitter
// ═══════════════════════════════════════════════════════════════════

let emitLet (node: PSGNode) (psg: ProgramSemanticGraph) (emitExpr: PSGNode -> MLIR<EmitResult>) : MLIR<EmitResult> =
    let children = getChildNodes psg node
    match children with
    | [_pattern; body] -> emitExpr body
    | [_pattern; _value; body] ->
        mlir {
            let! _ = emitExpr _value
            return! emitExpr body
        }
    | _ ->
        mlir {
            let mutable lastResult = Void
            for child in children do
                let! result = emitExpr child
                lastResult <- result
            return lastResult
        }

// ═══════════════════════════════════════════════════════════════════
// Main Expression Dispatcher
// ═══════════════════════════════════════════════════════════════════

let rec emitExpr (psg: ProgramSemanticGraph) (node: PSGNode) : MLIR<EmitResult> =
    let kind = node.SyntaxKind

    if kind.StartsWith("App") then
        emitClassifiedApp node psg (emitExpr psg)
    elif kind.StartsWith("Const:") then
        emitConst node
    elif kind.StartsWith("Ident:") || kind.StartsWith("LongIdent:") then
        mlir {
            let! zero = arith.constant 0L I32
            return Emitted zero
        }
    elif kind.StartsWith("Sequential") then
        emitSequential node psg (emitExpr psg)
    elif kind.StartsWith("LetOrUse") || kind.StartsWith("Let") then
        emitLet node psg (emitExpr psg)
    elif kind.StartsWith("Pattern:") then
        mlir { return Void }
    elif kind.StartsWith("Binding") then
        mlir { return Void }
    elif kind.StartsWith("Module:") then
        mlir { return Void }
    else
        mlir { return Error ("Unknown expression kind: " + kind) }
