/// LowerStructConstructors Nanopass
/// Transforms App nodes that are struct constructor calls to Record nodes.
///
/// F# struct types with explicit constructors (like `NativeStr(buffer, len)`) get parsed
/// as App (function application). This is correct syntactically, but semantically these
/// are structural constructions, not method calls.
///
/// The Fidelity memory model requires treating all construction as structural assembly
/// (not BCL-style `.ctor` invocation). This nanopass transforms:
///   Kind = SKExpr EApp (where symbol is struct constructor)
/// to:
///   Kind = SKExpr ERecord
///
/// This enables downstream code generation to use the proper `undef + insertvalue` pattern
/// rather than attempting to call a constructor method.
module Core.PSG.Nanopass.LowerStructConstructors

open FSharp.Native.Compiler.Symbols
open Core.PSG.Types

/// Check if a node's symbol is a struct constructor
let private isStructConstructor (node: PSGNode) : bool =
    match node.Symbol with
    | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
        try
            mfv.IsConstructor &&
            match mfv.DeclaringEntity with
            | Some entity -> entity.IsValueType
            | None -> false
        with _ -> false
    | _ -> false

/// Lower App nodes that are struct constructor calls to Record nodes
let private lowerStructConstructorNodes (psg: ProgramSemanticGraph) : ProgramSemanticGraph =
    let updatedNodes =
        psg.Nodes
        |> Map.map (fun _ node ->
            match node.Kind with
            | SKExpr EApp when isStructConstructor node ->
                { node with Kind = SKExpr ERecord }
            | _ -> node)
    { psg with Nodes = updatedNodes }

/// Main entry point for the LowerStructConstructors nanopass
let lowerStructConstructors (psg: ProgramSemanticGraph) : ProgramSemanticGraph =
    lowerStructConstructorNodes psg
