/// ExpressionEmitter - Thin transport layer for MLIR emission
///
/// This module is a TRANSPORT LAYER ONLY. It serializes PSG nodes to MLIR text.
/// NO transformation logic belongs here. Transformations are done by Alex.
module Alex.Emission.ExpressionEmitter

open Core.PSG.Types
open Alex.CodeGeneration.MLIRBuilder
open Alex.Pipeline.CompilationTypes

// ═══════════════════════════════════════════════════════════════════
// Result Type
// ═══════════════════════════════════════════════════════════════════

type EmitResult =
    | Emitted of Val
    | Void
    | Error of string

// ═══════════════════════════════════════════════════════════════════
// Constant Emission (pure data extraction, no transformation)
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
    if kind = "Const:Unit" || kind.Contains("Unit") then
        mlir { return Void }
    else
        mlir { return Error ("Unknown constant: " + kind) }

// ═══════════════════════════════════════════════════════════════════
// Helper: Get Children
// ═══════════════════════════════════════════════════════════════════

let private getChildNodes (psg: ProgramSemanticGraph) (node: PSGNode) : PSGNode list =
    match node.Children with
    | Parent ids -> ids |> List.choose (fun id -> Map.tryFind id.Value psg.Nodes)
    | _ -> []

// ═══════════════════════════════════════════════════════════════════
// Sequential and Let - structural traversal only
// ═══════════════════════════════════════════════════════════════════

let rec emitSequential (node: PSGNode) (psg: ProgramSemanticGraph) : MLIR<EmitResult> =
    let children = getChildNodes psg node
    match children with
    | [] -> mlir { return Void }
    | [single] -> emitExpr psg single
    | _ ->
        mlir {
            let mutable lastResult = Void
            for child in children do
                let! result = emitExpr psg child
                lastResult <- result
            return lastResult
        }

and emitLet (node: PSGNode) (psg: ProgramSemanticGraph) : MLIR<EmitResult> =
    let children = getChildNodes psg node
    match children with
    | [_pattern; body] -> emitExpr psg body
    | [_pattern; _value; body] ->
        mlir {
            let! _ = emitExpr psg _value
            return! emitExpr psg body
        }
    | _ ->
        mlir {
            let mutable lastResult = Void
            for child in children do
                let! result = emitExpr psg child
                lastResult <- result
            return lastResult
        }

// ═══════════════════════════════════════════════════════════════════
// Main Dispatcher - routes to appropriate handler
// ═══════════════════════════════════════════════════════════════════

and emitExpr (psg: ProgramSemanticGraph) (node: PSGNode) : MLIR<EmitResult> =
    let kind = node.SyntaxKind

    if kind.StartsWith("Const:") then
        emitConst node
    elif kind.StartsWith("Sequential") then
        emitSequential node psg
    elif kind.StartsWith("LetOrUse") || kind.StartsWith("Let") then
        emitLet node psg
    elif kind.StartsWith("Pattern:") then
        mlir { return Void }
    elif kind.StartsWith("Binding") then
        mlir { return Void }
    elif kind.StartsWith("Module:") then
        mlir { return Void }
    elif kind.StartsWith("App") then
        // App nodes should have been transformed by Alex before reaching here
        // If we see one, Alex didn't handle it - report error and continue
        let symbol = node.Symbol |> Option.map (fun s -> try s.FullName with _ -> s.DisplayName)
        EmissionErrors.add kind symbol "App node not transformed by Alex - missing binding"
        mlir { return Error ("App node not transformed by Alex: " + kind) }
    elif kind.StartsWith("Ident:") || kind.StartsWith("LongIdent:") then
        // Variable reference - should be resolved by Alex
        let symbol = node.Symbol |> Option.map (fun s -> try s.FullName with _ -> s.DisplayName)
        EmissionErrors.add kind symbol "Identifier not resolved - missing variable binding"
        mlir { return Error ("Ident not resolved: " + kind) }
    else
        EmissionErrors.add kind None ("Unknown node kind - not handled by emitter")
        mlir { return Error ("Unknown node kind: " + kind) }
