/// MLIRBuilder - Compositional MLIR generation via computation expression
///
/// This is the core Alex MLIR generation infrastructure.
/// ALL MLIR generation happens through this module - no sprintf anywhere else.
///
/// Usage:
///   mlir {
///       let! x = arith.constant 42 I32
///       let! y = arith.constant 10 I32
///       let! z = arith.addi x y I32
///       return z
///   }
module Alex.CodeGeneration.MLIRBuilder

open System.Text

// ═══════════════════════════════════════════════════════════════════
// Core Types
// ═══════════════════════════════════════════════════════════════════

/// MLIR integer types
type IntTy = I1 | I8 | I16 | I32 | I64

/// MLIR float types
type FloatTy = F32 | F64

/// MLIR types
type Ty =
    | Int of IntTy
    | Float of FloatTy
    | Ptr
    | Struct of Ty list
    | Array of int * Ty
    | Func of Ty list * Ty
    | Unit
    | Index

/// An SSA value - the primary currency of MLIR combinators
[<Struct>]
type SSA =
    | V of int
    | Arg of int
    member this.Name =
        match this with
        | V n -> sprintf "%%v%d" n
        | Arg n -> sprintf "%%arg%d" n

/// A typed SSA value
type Val = { SSA: SSA; Type: Ty }

/// Global symbol reference
type Global =
    | GFunc of string
    | GStr of uint32
    | GBytes of int

/// Integer comparison predicates
type ICmp = Eq | Ne | Slt | Sle | Sgt | Sge | Ult | Ule | Ugt | Uge

// ═══════════════════════════════════════════════════════════════════
// Builder State
// ═══════════════════════════════════════════════════════════════════

/// Builder state - accumulates MLIR output
type BuilderState = {
    Output: StringBuilder
    mutable SSACounter: int
    mutable Indent: int
    mutable Globals: (Global * string) list  // symbol -> content
}

module BuilderState =
    let create () = {
        Output = StringBuilder()
        SSACounter = 0
        Indent = 0
        Globals = []
    }

    let freshSSA (st: BuilderState) : SSA =
        let n = st.SSACounter
        st.SSACounter <- st.SSACounter + 1
        V n

    let indent (st: BuilderState) : string =
        String.replicate st.Indent "  "

    let emit (st: BuilderState) (line: string) : unit =
        st.Output.AppendLine(indent st + line) |> ignore

    let emitRaw (st: BuilderState) (line: string) : unit =
        st.Output.AppendLine(line) |> ignore

    let pushIndent (st: BuilderState) = st.Indent <- st.Indent + 1
    let popIndent (st: BuilderState) = st.Indent <- max 0 (st.Indent - 1)

    let registerGlobal (st: BuilderState) (g: Global) (content: string) =
        st.Globals <- (g, content) :: st.Globals

// ═══════════════════════════════════════════════════════════════════
// Type Serialization (internal only)
// ═══════════════════════════════════════════════════════════════════

module private Serialize =
    let intTy = function
        | I1 -> "i1" | I8 -> "i8" | I16 -> "i16" | I32 -> "i32" | I64 -> "i64"

    let floatTy = function F32 -> "f32" | F64 -> "f64"

    let rec ty = function
        | Int it -> intTy it
        | Float ft -> floatTy ft
        | Ptr -> "!llvm.ptr"
        | Struct fields ->
            let fs = fields |> List.map ty |> String.concat ", "
            sprintf "!llvm.struct<(%s)>" fs
        | Array (n, elem) -> sprintf "!llvm.array<%d x %s>" n (ty elem)
        | Func (args, ret) ->
            let argStr = args |> List.map ty |> String.concat ", "
            sprintf "(%s) -> %s" argStr (ty ret)
        | Unit -> "()"
        | Index -> "index"

    let ssa (s: SSA) = s.Name

    let global_ = function
        | GFunc name -> sprintf "@%s" name
        | GStr hash -> sprintf "@str_%u" hash
        | GBytes idx -> sprintf "@bytes_%d" idx

    let icmp = function
        | Eq -> "eq" | Ne -> "ne"
        | Slt -> "slt" | Sle -> "sle" | Sgt -> "sgt" | Sge -> "sge"
        | Ult -> "ult" | Ule -> "ule" | Ugt -> "ugt" | Uge -> "uge"

    let escape (s: string) =
        s.Replace("\\", "\\\\")
         .Replace("\"", "\\\"")
         .Replace("\n", "\\0A")
         .Replace("\r", "\\0D")
         .Replace("\t", "\\09")

// ═══════════════════════════════════════════════════════════════════
// MLIR Computation Expression
// ═══════════════════════════════════════════════════════════════════

/// The MLIR builder monad - builds MLIR through composition
type MLIR<'T> = BuilderState -> 'T

/// Computation expression builder for MLIR generation
type MLIRBuilderCE() =
    member _.Return(x: 'T) : MLIR<'T> = fun _ -> x
    member _.ReturnFrom(m: MLIR<'T>) : MLIR<'T> = m
    member _.Bind(m: MLIR<'T>, f: 'T -> MLIR<'U>) : MLIR<'U> =
        fun st -> f (m st) st
    member _.Zero() : MLIR<unit> = fun _ -> ()
    member _.Combine(m1: MLIR<unit>, m2: MLIR<'T>) : MLIR<'T> =
        fun st -> m1 st; m2 st
    member _.Delay(f: unit -> MLIR<'T>) : MLIR<'T> = fun st -> f () st
    member _.For(seq: seq<'T>, body: 'T -> MLIR<unit>) : MLIR<unit> =
        fun st -> for item in seq do body item st

/// The mlir computation expression
let mlir = MLIRBuilderCE()

// ═══════════════════════════════════════════════════════════════════
// Primitive Operations
// ═══════════════════════════════════════════════════════════════════

/// Get a fresh SSA value
let freshSSA : MLIR<SSA> = fun st -> BuilderState.freshSSA st

/// Emit a raw line (internal use)
let private emitLine (line: string) : MLIR<unit> = fun st ->
    BuilderState.emit st line

/// Register a string global, returns the global symbol
let registerString (content: string) : MLIR<Global> = fun st ->
    let hash = uint32 (hash content)
    let g = GStr hash
    BuilderState.registerGlobal st g content
    g

// ═══════════════════════════════════════════════════════════════════
// Arith Dialect Combinators
// ═══════════════════════════════════════════════════════════════════

module arith =
    /// arith.constant with integer value
    let constant (value: int64) (ty: IntTy) : MLIR<Val> = mlir {
        let! ssa = freshSSA
        let tyStr = Serialize.intTy ty
        do! emitLine (sprintf "%s = arith.constant %d : %s" ssa.Name value tyStr)
        return { SSA = ssa; Type = Int ty }
    }

    /// arith.constant with bool
    let constBool (value: bool) : MLIR<Val> = mlir {
        let! ssa = freshSSA
        let v = if value then "true" else "false"
        do! emitLine (sprintf "%s = arith.constant %s" ssa.Name v)
        return { SSA = ssa; Type = Int I1 }
    }

    /// arith.addi
    let addi (lhs: Val) (rhs: Val) (ty: IntTy) : MLIR<Val> = mlir {
        let! ssa = freshSSA
        let tyStr = Serialize.intTy ty
        do! emitLine (sprintf "%s = arith.addi %s, %s : %s" ssa.Name lhs.SSA.Name rhs.SSA.Name tyStr)
        return { SSA = ssa; Type = Int ty }
    }

    /// arith.subi
    let subi (lhs: Val) (rhs: Val) (ty: IntTy) : MLIR<Val> = mlir {
        let! ssa = freshSSA
        let tyStr = Serialize.intTy ty
        do! emitLine (sprintf "%s = arith.subi %s, %s : %s" ssa.Name lhs.SSA.Name rhs.SSA.Name tyStr)
        return { SSA = ssa; Type = Int ty }
    }

    /// arith.muli
    let muli (lhs: Val) (rhs: Val) (ty: IntTy) : MLIR<Val> = mlir {
        let! ssa = freshSSA
        let tyStr = Serialize.intTy ty
        do! emitLine (sprintf "%s = arith.muli %s, %s : %s" ssa.Name lhs.SSA.Name rhs.SSA.Name tyStr)
        return { SSA = ssa; Type = Int ty }
    }

    /// arith.divsi (signed division)
    let divsi (lhs: Val) (rhs: Val) (ty: IntTy) : MLIR<Val> = mlir {
        let! ssa = freshSSA
        let tyStr = Serialize.intTy ty
        do! emitLine (sprintf "%s = arith.divsi %s, %s : %s" ssa.Name lhs.SSA.Name rhs.SSA.Name tyStr)
        return { SSA = ssa; Type = Int ty }
    }

    /// arith.remsi (signed remainder)
    let remsi (lhs: Val) (rhs: Val) (ty: IntTy) : MLIR<Val> = mlir {
        let! ssa = freshSSA
        let tyStr = Serialize.intTy ty
        do! emitLine (sprintf "%s = arith.remsi %s, %s : %s" ssa.Name lhs.SSA.Name rhs.SSA.Name tyStr)
        return { SSA = ssa; Type = Int ty }
    }

    /// arith.cmpi
    let cmpi (pred: ICmp) (lhs: Val) (rhs: Val) (ty: IntTy) : MLIR<Val> = mlir {
        let! ssa = freshSSA
        let tyStr = Serialize.intTy ty
        do! emitLine (sprintf "%s = arith.cmpi %s, %s, %s : %s" ssa.Name (Serialize.icmp pred) lhs.SSA.Name rhs.SSA.Name tyStr)
        return { SSA = ssa; Type = Int I1 }
    }

    /// arith.xori
    let xori (lhs: Val) (rhs: Val) (ty: IntTy) : MLIR<Val> = mlir {
        let! ssa = freshSSA
        let tyStr = Serialize.intTy ty
        do! emitLine (sprintf "%s = arith.xori %s, %s : %s" ssa.Name lhs.SSA.Name rhs.SSA.Name tyStr)
        return { SSA = ssa; Type = Int ty }
    }

    /// arith.extsi (sign extend)
    let extsi (value: Val) (toTy: IntTy) : MLIR<Val> = mlir {
        let! ssa = freshSSA
        let fromTyStr = Serialize.ty value.Type
        let toTyStr = Serialize.intTy toTy
        do! emitLine (sprintf "%s = arith.extsi %s : %s to %s" ssa.Name value.SSA.Name fromTyStr toTyStr)
        return { SSA = ssa; Type = Int toTy }
    }

// ═══════════════════════════════════════════════════════════════════
// LLVM Dialect Combinators
// ═══════════════════════════════════════════════════════════════════

module llvm =
    /// llvm.mlir.addressof
    let addressof (sym: Global) : MLIR<Val> = mlir {
        let! ssa = freshSSA
        do! emitLine (sprintf "%s = llvm.mlir.addressof %s : !llvm.ptr" ssa.Name (Serialize.global_ sym))
        return { SSA = ssa; Type = Ptr }
    }

    /// llvm.alloca
    let alloca (size: Val) (elemTy: Ty) : MLIR<Val> = mlir {
        let! ssa = freshSSA
        do! emitLine (sprintf "%s = llvm.alloca %s x %s : (i64) -> !llvm.ptr" ssa.Name size.SSA.Name (Serialize.ty elemTy))
        return { SSA = ssa; Type = Ptr }
    }

    /// llvm.load
    let load (ptr: Val) (resultTy: Ty) : MLIR<Val> = mlir {
        let! ssa = freshSSA
        do! emitLine (sprintf "%s = llvm.load %s : !llvm.ptr -> %s" ssa.Name ptr.SSA.Name (Serialize.ty resultTy))
        return { SSA = ssa; Type = resultTy }
    }

    /// llvm.store
    let store (value: Val) (ptr: Val) : MLIR<unit> = mlir {
        do! emitLine (sprintf "llvm.store %s, %s : %s, !llvm.ptr" value.SSA.Name ptr.SSA.Name (Serialize.ty value.Type))
    }

    /// llvm.getelementptr
    let gep (base_: Val) (index: Val) (elemTy: Ty) : MLIR<Val> = mlir {
        let! ssa = freshSSA
        do! emitLine (sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i64) -> !llvm.ptr, %s"
            ssa.Name base_.SSA.Name index.SSA.Name (Serialize.ty elemTy))
        return { SSA = ssa; Type = Ptr }
    }

    /// llvm.insertvalue
    let insertvalue (container: Val) (value: Val) (index: int) (containerTy: Ty) : MLIR<Val> = mlir {
        let! ssa = freshSSA
        do! emitLine (sprintf "%s = llvm.insertvalue %s, %s[%d] : %s"
            ssa.Name value.SSA.Name container.SSA.Name index (Serialize.ty containerTy))
        return { SSA = ssa; Type = containerTy }
    }

    /// llvm.extractvalue
    let extractvalue (container: Val) (index: int) (containerTy: Ty) (resultTy: Ty) : MLIR<Val> = mlir {
        let! ssa = freshSSA
        do! emitLine (sprintf "%s = llvm.extractvalue %s[%d] : %s"
            ssa.Name container.SSA.Name index (Serialize.ty containerTy))
        return { SSA = ssa; Type = resultTy }
    }

    /// llvm.mlir.undef
    let undef (ty: Ty) : MLIR<Val> = mlir {
        let! ssa = freshSSA
        do! emitLine (sprintf "%s = llvm.mlir.undef : %s" ssa.Name (Serialize.ty ty))
        return { SSA = ssa; Type = ty }
    }

    /// llvm.mlir.zero
    let zero (ty: Ty) : MLIR<Val> = mlir {
        let! ssa = freshSSA
        do! emitLine (sprintf "%s = llvm.mlir.zero : %s" ssa.Name (Serialize.ty ty))
        return { SSA = ssa; Type = ty }
    }

    /// llvm.inline_asm
    let inlineAsm (asm: string) (constraints: string) (args: Val list) (resultTy: Ty) : MLIR<Val> = mlir {
        let! ssa = freshSSA
        let argNames = args |> List.map (fun a -> a.SSA.Name) |> String.concat ", "
        let argTypes = args |> List.map (fun a -> Serialize.ty a.Type) |> String.concat ", "
        do! emitLine (sprintf "%s = llvm.inline_asm has_side_effects \"%s\", \"%s\" %s : (%s) -> %s"
            ssa.Name asm constraints argNames argTypes (Serialize.ty resultTy))
        return { SSA = ssa; Type = resultTy }
    }

// ═══════════════════════════════════════════════════════════════════
// Func Dialect Combinators
// ═══════════════════════════════════════════════════════════════════

module func =
    /// func.call with result
    let call (callee: string) (args: Val list) (argTypes: Ty list) (resultTy: Ty) : MLIR<Val> = mlir {
        let! ssa = freshSSA
        let argNames = args |> List.map (fun a -> a.SSA.Name) |> String.concat ", "
        let typeStrs = argTypes |> List.map Serialize.ty |> String.concat ", "
        do! emitLine (sprintf "%s = func.call @%s(%s) : (%s) -> %s"
            ssa.Name callee argNames typeStrs (Serialize.ty resultTy))
        return { SSA = ssa; Type = resultTy }
    }

    /// func.call void (no result)
    let callVoid (callee: string) (args: Val list) (argTypes: Ty list) : MLIR<unit> = mlir {
        let argNames = args |> List.map (fun a -> a.SSA.Name) |> String.concat ", "
        let typeStrs = argTypes |> List.map Serialize.ty |> String.concat ", "
        do! emitLine (sprintf "func.call @%s(%s) : (%s) -> ()" callee argNames typeStrs)
    }

    /// func.return with value
    let ret (value: Val) : MLIR<unit> = mlir {
        do! emitLine (sprintf "func.return %s : %s" value.SSA.Name (Serialize.ty value.Type))
    }

    /// func.return void
    let retVoid : MLIR<unit> = mlir {
        do! emitLine "func.return"
    }

// ═══════════════════════════════════════════════════════════════════
// NativeStr Helper (ptr + i64 struct)
// ═══════════════════════════════════════════════════════════════════

let nativeStrTy = Struct [Ptr; Int I64]

/// Build a NativeStr from a string constant
let buildNativeStr (content: string) : MLIR<Val> = mlir {
    let! sym = registerString content
    let! ptr = llvm.addressof sym
    let! len = arith.constant (int64 content.Length) I64
    let! undef = llvm.undef nativeStrTy
    let! v1 = llvm.insertvalue undef ptr 0 nativeStrTy
    let! v2 = llvm.insertvalue v1 len 1 nativeStrTy
    return v2
}

/// Extract pointer and length from NativeStr
let extractNativeStr (str: Val) : MLIR<Val * Val> = mlir {
    let! ptr = llvm.extractvalue str 0 nativeStrTy Ptr
    let! len = llvm.extractvalue str 1 nativeStrTy (Int I64)
    return (ptr, len)
}

// ═══════════════════════════════════════════════════════════════════
// Module Building
// ═══════════════════════════════════════════════════════════════════

/// Run MLIR builder and get output string
let run (m: MLIR<'T>) : string * 'T =
    let st = BuilderState.create ()
    let result = m st

    // Build globals section
    let globalsStr = StringBuilder()
    for (g, content) in st.Globals |> List.rev do
        match g with
        | GStr hash ->
            let escaped = Serialize.escape content
            globalsStr.AppendLine(sprintf "  llvm.mlir.global private constant @str_%u(\"%s\") : !llvm.array<%d x i8>"
                hash escaped content.Length) |> ignore
        | GBytes idx ->
            globalsStr.AppendLine(sprintf "  llvm.mlir.global private constant @bytes_%d(...) : !llvm.array<...>" idx) |> ignore
        | GFunc _ -> ()

    (st.Output.ToString() + globalsStr.ToString(), result)

/// Run and get only the MLIR text
let runText (m: MLIR<'T>) : string = run m |> fst
