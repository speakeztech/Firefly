module Dabbit.CodeGeneration.MLIRSyntax

open Core.XParsec.Foundation

/// MLIR Dialect operations
type MLIRDialect = 
    | Func | Arith | LLVM | Memref | Scf | Cf | Math | Vector | Tensor
    
/// MLIR operation components
type MLIROpComponent =
    | Dialect of MLIRDialect
    | Operation of string
    | SSAValue of string
    | Result of string
    | Type of string
    | Attribute of string * string
    | Region of MLIROpComponent list
    | Block of MLIROpComponent list

/// Basic MLIR syntax combinators
let pSSA name = pString "%" >>. pString name

let pAt = pChar '@'

let pDialect = function
    | Func -> pString "func"
    | Arith -> pString "arith"  
    | LLVM -> pString "llvm"
    | Memref -> pString "memref"
    | Scf -> pString "scf"
    | Cf -> pString "cf"
    | Math -> pString "math"
    | Vector -> pString "vector"
    | Tensor -> pString "tensor"

let pDot = pChar '.'

let pEquals = pSpaces >>. pChar '=' <+> pSpaces

let pColon = pSpaces >>. pChar ':' <+> pSpaces

let pComma = pSpaces >>. pChar ',' <+> pSpaces

let pExclaim = pChar '!'

/// Type combinators
let pMLIRType = 
    choice [
        pString "i1" >>% "i1"
        pString "i8" >>% "i8"
        pString "i16" >>% "i16"
        pString "i32" >>% "i32"
        pString "i64" >>% "i64"
        pString "f32" >>% "f32"
        pString "f64" >>% "f64"
        pString "index" >>% "index"
        // Composite types
        pExclaim >>. pString "llvm.ptr" >>% "!llvm.ptr"
        pExclaim >>. pString "llvm.array" >>. pBetween (pChar '<') (pChar '>') 
            (pInt <+> pSpaces <+> pString "x" <+> pSpaces <+> pMLIRType) 
            |>> (fun (count, _, _, _, elemType) -> sprintf "!llvm.array<%d x %s>" count elemType)
        // Memref types
        pString "memref" >>. pBetween (pChar '<') (pChar '>') pMLIRType
            |>> (fun elemType -> sprintf "memref<%s>" elemType)
    ]

/// Function type: (type1, type2) -> returnType
let pFuncType = 
    pBetween (pChar '(') (pChar ')') (pSepBy pMLIRType pComma) <+>
    pSpaces >>. pString "->" >>. pSpaces >>. pMLIRType
    |>> (fun (parameterTypes, returnType) -> sprintf "(%s) -> %s" (String.concat ", " parameterTypes) returnType)

/// MLIR operation builders
let buildFuncCall target funcName args resultType =
    let argList = String.concat ", " args
    sprintf "%s = func.call @%s(%s) : %s" target funcName argList resultType

let buildArithOp op result left right typ =
    sprintf "%s = arith.%s %s, %s : %s" result op left right typ

let buildSelect result condition trueVal falseVal typ =
    sprintf "%s = arith.select %s, %s, %s : %s" result condition trueVal falseVal typ

let buildAlloca result elemType size =
    match size with
    | Some s -> sprintf "%s = memref.alloca(%s) : memref<%s>" result s elemType
    | None -> sprintf "%s = memref.alloca() : memref<1x%s>" result elemType

let buildLoad result memref indices =
    let indexList = String.concat ", " indices
    sprintf "%s = memref.load %s[%s] : %s" result memref indexList "memref<?xi32>"  // Type would need to be tracked

let buildStore value memref indices =
    let indexList = String.concat ", " indices
    sprintf "memref.store %s, %s[%s] : %s" value memref indexList "memref<?xi32>"

/// Global variable declarations
let buildGlobalConstant name value typ =
    sprintf "llvm.mlir.global internal constant @%s(%s) : %s" name value typ

/// Block and region builders
let buildBlock label ops =
    sprintf "^%s:\n%s" label (String.concat "\n" ops |> String.indent 2)

let buildRegion ops =
    sprintf "{\n%s\n}" (String.concat "\n" ops |> String.indent 2)

/// Higher-level operation patterns
let funcCallPattern = 
    pipe3
        (pSSA "result")
        (pEquals >>. pDialect Func >>. pDot >>. pString "call" >>. pSpaces >>. pAt >>. pIdentifier)
        (pBetween (pChar '(') (pChar ')') (pSepBy (pSSA "arg") pComma) <+> pColon <+> pFuncType)
        (fun result funcName (args, funcType) -> 
            buildFuncCall result funcName args funcType)

/// Combinator to generate and validate MLIR operations
let generateMLIROp opBuilder =
    // This would integrate with the MLIR emitter to ensure valid syntax
    opBuilder |> Result.Ok

/// Helper to indent strings
module String =
    let indent spaces (str: string) =
        let lines = str.Split('\n')
        lines |> Array.map (fun line -> String.replicate spaces " " + line) |> String.concat "\n"