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

/// Basic MLIR syntax parsers
let pSSA (name: string) = pCharLiteral '%' >>. pString name

let pDialectName = function
    | Func -> pString "func"
    | Arith -> pString "arith"  
    | LLVM -> pString "llvm"
    | Memref -> pString "memref"
    | Scf -> pString "scf"
    | Cf -> pString "cf"
    | Math -> pString "math"
    | Vector -> pString "vector"
    | Tensor -> pString "tensor"

/// Simple MLIR type parsers
let pBasicType () = 
    choice [
        pString "i1" >>% "i1"
        pString "i8" >>% "i8"  
        pString "i16" >>% "i16"
        pString "i32" >>% "i32"
        pString "i64" >>% "i64"
        pString "f32" >>% "f32"
        pString "f64" >>% "f64"
        pString "index" >>% "index"
    ]

let pLLVMPtrType () = 
    pCharLiteral '!' >>. pString "llvm.ptr" >>% "!llvm.ptr"

/// Type combinators for MLIR types  
let rec pMLIRType () = 
    choice [
        pBasicType ()
        pLLVMPtrType ()
    ]

/// Function type: (type1, type2) -> returnType
let pFuncType () = 
    pCharLiteral '(' >>. 
    pSepBy (pMLIRType ()) (pCommaSpaced ()) >>= fun paramTypes ->
    pCharLiteral ')' >>. 
    pSpaces >>. 
    pString "->" >>. 
    pSpaces >>. 
    pMLIRType () >>= fun returnType ->
    let struct (paramTypeList, _) = paramTypes
    let paramTypeStr = String.concat ", " (paramTypeList |> Seq.toList)
    let funcTypeStr = sprintf "(%s) -> %s" paramTypeStr returnType
    preturn funcTypeStr

/// Attribute parser
let pAttribute () =
    pString "#" >>. 
    pIdentifier () >>= fun attrName ->
    pEqualsSpaced () >>. 
    (pQuotedString () <|> pIdentifier ()) >>= fun attrValue ->
    preturn (attrName, attrValue)

/// Operation parser for dialect.op
let pOperation () =
    pIdentifier () >>= fun dialectName ->
    pCharLiteral '.' >>. 
    pIdentifier () >>= fun opName ->
    preturn (dialectName, opName)

/// Parse MLIR operation arguments
let pOperationArgs () = 
    pCharLiteral '(' >>. 
    pSepBy (pSSAName () <|> pQuotedString () <|> pIdentifier ()) (pCommaSpaced ()) >>= fun args ->
    pCharLiteral ')' >>. 
    preturn args

/// Parse MLIR operation attributes
let pOperationAttrs () = 
    pCharLiteral '{' >>. 
    pSepBy (pAttribute ()) (pCommaSpaced ()) >>= fun attrs ->
    pCharLiteral '}' >>. 
    preturn attrs

/// Parse MLIR operation results
let pOperationResults () = 
    pSepBy1 (pSSAName ()) (pCommaSpaced ()) >>= fun results ->
    pEqualsSpaced () >>. 
    preturn results

/// Full MLIR operation parser
let pFullOperation () =
    opt (pOperationResults ()) >>= fun resultsOpt ->
    pOperation () >>= fun (dialect, op) ->
    opt (pOperationArgs ()) >>= fun argsOpt ->
    opt (pOperationAttrs ()) >>= fun attrsOpt ->
    pColonSpaced () >>. 
    (pFuncType () <|> pMLIRType ()) >>= fun opType ->
    let results = resultsOpt |> ValueOption.toOption |> Option.map (fun r -> let struct (data, _) = r in data |> Seq.toList) |> Option.defaultValue []
    let args = argsOpt |> ValueOption.toOption |> Option.map (fun a -> let struct (data, _) = a in data |> Seq.toList) |> Option.defaultValue []
    let attrs = attrsOpt |> ValueOption.toOption |> Option.map (fun a -> let struct (data, _) = a in data |> Seq.toList) |> Option.defaultValue []
    preturn {|
        Results = results
        Dialect = dialect
        Operation = op
        Arguments = args
        Attributes = attrs
        Type = opType
    |}

/// MLIR value representation
type MLIRValue = {
    SSA: string
    Type: string
    IsConstant: bool
}

/// MLIR instruction representation  
type MLIRInstruction = {
    Results: string list
    Operation: string
    Arguments: string list
    Attributes: (string * string) list
    Type: string
}

/// MLIR block representation
type MLIRBlock = {
    Label: string option
    Arguments: (string * string) list
    Instructions: MLIRInstruction list
}

/// MLIR function representation
type MLIRFunction = {
    Name: string
    Parameters: (string * string) list
    ReturnType: string
    Attributes: (string * string) list
    Blocks: MLIRBlock list
    IsPrivate: bool
}

/// MLIR module representation
type MLIRModule = {
    Name: string option
    Functions: MLIRFunction list
    GlobalConstants: (string * string * string) list
    ExternalDeclarations: string list
}

/// Parse MLIR function signature
let pFunctionSignature () =
    opt (pString "private" >>. pSpaces1) >>= fun isPrivateOpt ->
    pString "func" >>. 
    pSpaces1 >>. 
    pCharLiteral '@' >>. 
    pIdentifier () >>= fun funcName ->
    pCharLiteral '(' >>. 
    pSepBy (pSSAName () >>= fun name -> 
            pColonSpaced () >>. 
            pMLIRType () >>= fun typ ->
            preturn (name, typ)) (pCommaSpaced ()) >>= fun parameters ->
    pCharLiteral ')' >>. 
    pSpaces >>. 
    pString "->" >>. 
    pSpaces >>. 
    pMLIRType () >>= fun returnType ->
    opt (pOperationAttrs ()) >>= fun attrsOpt ->
    let attrs = attrsOpt |> ValueOption.toOption |> Option.map (fun a -> let struct (data, _) = a in data |> Seq.toList) |> Option.defaultValue []
    let isPrivate = isPrivateOpt |> ValueOption.toOption |> Option.isSome
    let struct (paramList, _) = parameters
    preturn (funcName, paramList |> Seq.toList, returnType, attrs, isPrivate)

/// Parse MLIR block
let pBlock () =
    opt (pString "^" >>. 
         pIdentifier () >>= fun label ->
         opt (pCharLiteral '(' >>. 
              pSepBy (pSSAName () >>= fun name ->
                      pColonSpaced () >>. 
                      pMLIRType () >>= fun typ ->
                      preturn (name, typ)) (pCommaSpaced ())) >>= fun argsOpt ->
         opt (pCharLiteral ')') >>. 
         pColonSpaced () >>. 
         preturn (Some label, argsOpt |> ValueOption.toOption |> Option.map (fun a -> let struct (data, _) = a in data |> Seq.toList) |> Option.defaultValue [])) >>= fun blockHeaderOpt ->
    many (pSpaces >>. 
          pFullOperation () >>= fun op ->
          preturn {
              Results = op.Results
              Operation = op.Dialect + "." + op.Operation
              Arguments = op.Arguments
              Attributes = op.Attributes
              Type = op.Type
          }) >>= fun instructions ->
    let (label, args) = 
        match blockHeaderOpt |> ValueOption.toOption with
        | Some (lbl, a) -> (lbl, a)
        | None -> (None, [])
    let instructionList = instructions |> Seq.toList
    preturn {
        Label = label
        Arguments = args
        Instructions = instructionList
    }

/// Parse complete MLIR function
let pMLIRFunction () =
    pFunctionSignature () >>= fun (name, parameters, returnType, attrs, isPrivate) ->
    pSpaces >>. 
    pCharLiteral '{' >>. 
    pSpaces >>. 
    many1 (pBlock ()) >>= fun blocks ->
    pSpaces >>. 
    pCharLiteral '}' >>. 
    let blockList = blocks |> Seq.toList
    preturn {
        Name = name
        Parameters = parameters
        ReturnType = returnType
        Attributes = attrs
        Blocks = blockList
        IsPrivate = isPrivate
    }

/// Parse global constant declaration
let pGlobalConstant () =
    pString "llvm.mlir.global" >>. 
    pSpaces1 >>. 
    opt (pString "constant" >>. pSpaces1) >>. 
    pCharLiteral '@' >>. 
    pIdentifier () >>= fun name ->
    pCharLiteral '(' >>. 
    pQuotedString () >>= fun value ->
    pCharLiteral ')' >>. 
    pSpaces >>. 
    pColonSpaced () >>. 
    pMLIRType () >>= fun typ ->
    preturn (name, value, typ)

/// Parse external function declaration
let pExternalDeclaration () = 
    pString "func.func" >>. 
    pSpaces1 >>. 
    opt (pString "private" >>. pSpaces1) >>. 
    pCharLiteral '@' >>. 
    pIdentifier () >>= fun name ->
    pFuncType () >>= fun signature ->
    preturn (name + " : " + signature)

/// Parse complete MLIR module
let pMLIRModule () =
    opt (pString "module" >>. 
         pSpaces1 >>. 
         pCharLiteral '@' >>. 
         pIdentifier () >>= fun name ->
         pSpaces >>. 
         pCharLiteral '{' >>. 
         preturn name) >>= fun moduleNameOpt ->
    many (pGlobalConstant ()) >>= fun globals ->
    many (pExternalDeclaration ()) >>= fun externals ->
    many (pMLIRFunction ()) >>= fun functions ->
    opt (pCharLiteral '}' >>. pSpaces) >>. 
    let moduleNameOption = moduleNameOpt |> ValueOption.toOption
    let functionList = functions |> Seq.toList
    let globalList = globals |> Seq.toList
    let externalList = externals |> Seq.toList
    preturn {
        Name = moduleNameOption
        Functions = functionList
        GlobalConstants = globalList
        ExternalDeclarations = externalList
    }

/// Formatting functions for MLIR output
module MLIRFormatting =
    
    /// Format MLIR instruction with proper indentation
    let formatInstruction (indentLevel: int) (instr: MLIRInstruction) : string =
        let indentStr = indent indentLevel
        let results = 
            if instr.Results.IsEmpty then "" 
            else String.concat ", " instr.Results + " = "
        let args = 
            if instr.Arguments.IsEmpty then ""
            else "(" + String.concat ", " instr.Arguments + ")"
        let attrs = 
            if instr.Attributes.IsEmpty then ""
            else " {" + (instr.Attributes |> List.map (fun (k,v) -> k + " = " + v) |> String.concat ", ") + "}"
        sprintf "%s%s%s%s%s : %s" indentStr results instr.Operation args attrs instr.Type

    /// Format MLIR block with proper indentation  
    let formatBlock (indentLevel: int) (block: MLIRBlock) : string =
        let indentStr = indent indentLevel
        let header = 
            match block.Label with
            | Some label -> 
                let args = 
                    if block.Arguments.IsEmpty then ""
                    else "(" + (block.Arguments |> List.map (fun (n,t) -> n + ": " + t) |> String.concat ", ") + ")"
                sprintf "%s^%s%s:\n" indentStr label args
            | None -> ""
        let instructions = 
            block.Instructions 
            |> List.map (formatInstruction (indentLevel + 1))
            |> String.concat "\n"
        header + instructions

    /// Format complete MLIR function
    let formatFunction (functionDef: MLIRFunction) : string =
        let privacy = if functionDef.IsPrivate then "private " else ""
        let parameters = functionDef.Parameters |> List.map (fun (n,t) -> n + ": " + t) |> String.concat ", "
        let attrs = 
            if functionDef.Attributes.IsEmpty then ""
            else " attributes {" + (functionDef.Attributes |> List.map (fun (k,v) -> k + " = " + v) |> String.concat ", ") + "}"
        let signature = sprintf "%sfunc @%s(%s) -> %s%s {" privacy functionDef.Name parameters functionDef.ReturnType attrs
        let blocks = functionDef.Blocks |> List.map (formatBlock 1) |> String.concat "\n\n"
        sprintf "%s\n%s\n}" signature blocks

    /// Format complete MLIR module
    let formatModule (moduleData: MLIRModule) : string =
        let header = 
            match moduleData.Name with
            | Some name -> sprintf "module @%s {\n" name
            | None -> ""
        
        let globals = 
            moduleData.GlobalConstants 
            |> List.map (fun (name, value, typ) -> 
                sprintf "llvm.mlir.global constant @%s(%s) : %s" name value typ)
            |> String.concat "\n"
        
        let externals = 
            moduleData.ExternalDeclarations
            |> String.concat "\n"
        
        let functions = 
            moduleData.Functions 
            |> List.map formatFunction
            |> String.concat "\n\n"
        
        let content = 
            [globals; externals; functions] 
            |> List.filter (not << isNullOrWhiteSpace)
            |> String.concat "\n\n"
        
        let footer = if Option.isSome moduleData.Name then "\n}" else ""
        
        header + content + footer

/// MLIR syntax validation helpers
module MLIRValidation =
    
    /// Validate SSA name format
    let isValidSSAName (name: string) : bool =
        not (isNullOrEmpty name) && 
        name.StartsWith("%") && 
        name.Length > 1 &&
        System.Char.IsLetter(name.[1]) || name.[1] = '_'
    
    /// Validate global name format  
    let isValidGlobalName (name: string) : bool =
        not (isNullOrEmpty name) &&
        name.StartsWith("@") &&
        name.Length > 1 &&
        System.Char.IsLetter(name.[1]) || name.[1] = '_'
    
    /// Validate MLIR type name
    let isValidTypeName (typeName: string) : bool =
        not (isNullOrEmpty typeName) &&
        (typeName.StartsWith("!") || 
         List.contains typeName ["i1"; "i8"; "i16"; "i32"; "i64"; "f32"; "f64"; "index"])
    
    /// Validate complete MLIR instruction
    let validateInstruction (instr: MLIRInstruction) : CompilerResult<unit> =
        if isNullOrEmpty instr.Operation then
            CompilerFailure [SyntaxError({ Line = 0; Column = 0; File = ""; Offset = 0 }, 
                                        "Empty operation name", [])]
        elif not (isValidTypeName instr.Type) then
            CompilerFailure [SyntaxError({ Line = 0; Column = 0; File = ""; Offset = 0 }, 
                                        sprintf "Invalid type: %s" instr.Type, [])]
        else
            Success ()
    
    /// Validate MLIR function
    let validateFunction (functionDef: MLIRFunction) : CompilerResult<unit> =
        if isNullOrEmpty functionDef.Name then
            CompilerFailure [SyntaxError({ Line = 0; Column = 0; File = ""; Offset = 0 }, 
                                        "Empty function name", [])]
        elif functionDef.Blocks.IsEmpty then
            CompilerFailure [SyntaxError({ Line = 0; Column = 0; File = ""; Offset = 0 }, 
                                        "Function must have at least one block", [])]
        else
            // Validate all instructions in all blocks
            let allInstructions = functionDef.Blocks |> List.collect (fun b -> b.Instructions)
            allInstructions 
            |> List.fold (fun acc instr ->
                match acc with
                | CompilerFailure _ -> acc
                | Success _ -> validateInstruction instr
            ) (Success ())