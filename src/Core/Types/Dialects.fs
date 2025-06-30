module Core.Types.Dialects

type MLIRDialect =
    | Standard | LLVM | Func | Arith | SCF
    | MemRef | Index | Affine | Builtin