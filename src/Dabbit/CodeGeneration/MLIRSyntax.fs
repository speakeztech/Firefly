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

let pEquals = pSpaces >>. pChar '=' >>. pSpaces

let pColon = pSpaces >>. pChar ':' >>. pSpaces

let pComma = pSpaces >>. pChar ',' >>. pSpaces

let pExclaim = pChar '!'

/// Type combinators
let rec pMLIRType = 
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
        pExclaim >>. pString "llvm.array" >>. pBetween '<' '>' 
            (pInt >>= fun count ->
             pSpaces >>. pString "x" >>. pSpaces >>. pMLIRType >>= fun elemType ->
             preturn (sprintf "!llvm.array<%d x %s>" count elemType))
        // Memref types
        pString "memref" >>. pBetween '<' '>' pMLIRType
            |>> (fun elemType -> sprintf "memref<%s>" elemType)
    ]

/// Function type: (type1, type2) -> returnType
let pFuncType = 
    pBetween '(' ')' (pSepBy pMLIRType pComma) >>= fun paramTypes ->
    pSpaces >>. pString "->" >>. pSpaces >>. pMLIRType >>= fun returnType ->
    let paramTypeStr = String.concat ", " paramTypes
    let funcTypeStr = sprintf "(%s) -> %s" paramTypeStr returnType
    preturn funcTypeStr

/// Attribute parser
let pAttribute =
    pString "#" >>. many1 (pChar (fun c -> c <> '=' && c <> ' ' && c <> ',')) >>= fun attrName ->
    pEquals >>. pMLIRType >>= fun attrValue ->
    let attrNameStr = new String(Array.ofList attrName)
    preturn (attrNameStr, attrValue)

/// Operation parser for dialect.op
let pOperation =
    pDialect >>= fun dialectStr ->
    pDot >>. many1 (pChar (fun c -> c <> ' ' && c <> '(' && c <> '<')) >>= fun opChars ->
    let opStr = new String(Array.ofList opChars)
    preturn (dialectStr, opStr)

/// Result assignment
let pResult =
    pSSA >>= fun name ->
    pEquals >>= fun _ ->
    preturn name

/// Full operation
let pFullOperation =
    opt pResult >>= fun resultOpt ->
    pOperation >>= fun (dialect, op) ->
    opt (pBetween '(' ')' (pSepBy pAttribute pComma)) >>= fun attrsOpt ->
    opt (pBetween '<' '>' (pSepBy pAttribute pComma)) >>= fun genAttrsOpt ->
    pColon >>. pFuncType >>= fun funcType ->
    let attrs = defaultArg attrsOpt []
    let genAttrs = defaultArg genAttrsOpt []
    let result = defaultArg resultOpt ""
    preturn (result, dialect, op, attrs, genAttrs, funcType)

/// Format MLIR operation as string
let formatOperation (indentLevel: int) (result: string) (dialect: MLIRDialect) (op: string) (attrs: (string * string) list) (funcType: string) =
    let dialectStr = 
        match dialect with
        | Func -> "func"
        | Arith -> "arith"
        | LLVM -> "llvm"
        | Memref -> "memref"
        | Scf -> "scf"
        | Cf -> "cf"
        | Math -> "math"
        | Vector -> "vector"
        | Tensor -> "tensor"
    
    let resultPrefix = 
        if String.IsNullOrEmpty(result) then "" 
        else sprintf "%s = " result
    
    let attrsStr = 
        if List.isEmpty attrs then ""
        else 
            let attrItems = attrs |> List.map (fun (name, value) -> sprintf "#%s = %s" name value)
            sprintf "(%s)" (String.concat ", " attrItems)
    
    indent indentLevel (sprintf "%s%s.%s %s : %s" resultPrefix dialectStr op attrsStr funcType)

/// Format a basic block
let formatBlock (indentLevel: int) (label: string) (operations: string list) =
    let labelLine = indent indentLevel (sprintf "%s:" label)
    let blockBody = operations |> List.map (indent (indentLevel + 1))
    String.concat "\n" (labelLine :: blockBody)

/// Format a function
let formatFunction (name: string) (args: (string * string) list) (returnType: string) (body: string) =
    let argStr = args |> List.map (fun (name, typ) -> sprintf "%s: %s" name typ) |> String.concat ", "
    let header = sprintf "func.func @%s(%s) -> %s {" name argStr returnType
    let footer = "}"
    String.concat "\n" [header; body; footer]

/// Format a module
let formatModule (name: string) (body: string) =
    let header = sprintf "module @%s {" name
    let footer = "}"
    String.concat "\n" [header; body; footer]