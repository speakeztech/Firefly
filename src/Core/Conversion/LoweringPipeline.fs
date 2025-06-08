module Core.Conversion.LoweringPipeline

open System
open XParsec
open Core.XParsec.Foundation
open Core.XParsec.Foundation.Combinators
open Core.XParsec.Foundation.CharParsers
open Core.XParsec.Foundation.StringParsers
open Core.XParsec.Foundation.ErrorHandling
open Core.MLIRGeneration.Dialect

/// Lowering transformation state
type LoweringState = {
    CurrentDialect: MLIRDialect
    TargetDialect: MLIRDialect
    TransformedOperations: string list
    SymbolTable: Map<string, string>
    PassName: string
    ErrorContext: string list
}

/// Lowering pass definition with XParsec-based transformation
type LoweringPass = {
    Name: string
    SourceDialect: MLIRDialect
    TargetDialect: MLIRDialect
    Transform: Parser<string, LoweringState>
    Validate: string -> bool
}

/// MLIR operation parsing using XParsec combinators
module MLIRParsers =
    
    /// Parses an SSA value name
    let ssaValue : Parser<string, LoweringState> =
        pchar '%' >>= fun _ ->
        identifier >>= fun name ->
        succeed ("%" + name)
        |> withErrorContext "SSA value"
    
    /// Parses a function name
    let functionName : Parser<string, LoweringState> =
        pchar '@' >>= fun _ ->
        identifier >>= fun name ->
        succeed ("@" + name)
        |> withErrorContext "function name"
    
    /// Parses a type annotation
    let typeAnnotation : Parser<string, LoweringState> =
        let primitiveType = 
            (pstring "i1" <|> pstring "i8" <|> pstring "i16" <|> pstring "i32" <|> pstring "i64" <|>
             pstring "f32" <|> pstring "f64" <|> pstring "void")
        
        let pointerType = 
            pstring "!llvm.ptr<" >>= fun _ ->
            typeAnnotation >>= fun innerType ->
            pchar '>' >>= fun _ ->
            succeed ("!llvm.ptr<" + innerType + ">")
        
        let memrefType =
            pstring "memref<" >>= fun _ ->
            many (satisfy (fun c -> c <> '>')) >>= fun chars ->
            pchar '>' >>= fun _ ->
            succeed ("memref<" + String(Array.ofList chars) + ">")
        
        pointerType <|> memrefType <|> primitiveType
        |> withErrorContext "type annotation"
    
    /// Parses operation attributes
    let operationAttributes : Parser<Map<string, string>, LoweringState> =
        let attribute = 
            identifier >>= fun key ->
            ws >>= fun _ ->
            pchar '=' >>= fun _ ->
            ws >>= fun _ ->
            (between (pchar '"') (pchar '"') (many (satisfy (fun c -> c <> '"'))) |>> fun chars -> String(Array.ofList chars)) >>= fun value ->
            succeed (key, value)
        
        let attributeList = sepBy attribute (pchar ',')
        
        between (pchar '{') (pchar '}') attributeList |>> Map.ofList <|>
        succeed Map.empty
        |> withErrorContext "operation attributes"
    
    /// Parses an MLIR operation with dialect prefix
    let mlirOperation : Parser<(string * string list * string list * Map<string, string> * string), LoweringState> =
        let resultList = sepBy ssaValue (pchar ',') .>> ws .>> pchar '=' .>> ws <|> succeed []
        let operandList = between (pchar '(') (pchar ')') (sepBy ssaValue (pchar ',')) <|> succeed []
        
        resultList >>= fun results ->
        identifier >>= fun dialectName ->
        pchar '.' >>= fun _ ->
        identifier >>= fun opName ->
        ws >>= fun _ ->
        operandList >>= fun operands ->
        ws >>= fun _ ->
        operationAttributes >>= fun attrs ->
        ws >>= fun _ ->
        opt (pchar ':' >>= fun _ -> ws >>= fun _ -> typeAnnotation) >>= fun typeOpt ->
        let fullOpName = dialectName + "." + opName
        let typeStr = typeOpt |> Option.defaultValue ""
        succeed (fullOpName, results, operands, attrs, typeStr)
        |> withErrorContext "MLIR operation"

/// Dialect-specific transformation combinators
module DialectTransformers =
    
    /// Transforms func dialect operations to LLVM dialect
    let funcToLLVMTransformer : Parser<string, LoweringState> =
        let transformFuncOp = 
            pstring "func.func" >>= fun _ ->
            ws >>= fun _ ->
            functionName >>= fun fname ->
            between (pchar '(') (pchar ')') (sepBy typeAnnotation (pchar ',')) >>= fun paramTypes ->
            ws >>= fun _ ->
            pstring "->" >>= fun _ ->
            ws >>= fun _ ->
            typeAnnotation >>= fun returnType ->
            ws >>= fun _ ->
            pchar '{' >>= fun _ ->
            let llvmFunc = sprintf "llvm.func %s(%s) -> %s {" 
                                  fname 
                                  (String.concat ", " paramTypes) 
                                  returnType
            succeed llvmFunc
        
        let transformFuncReturn =
            pstring "func.return" >>= fun _ ->
            ws >>= fun _ ->
            opt ssaValue >>= fun valueOpt ->
            opt (pchar ':' >>= fun _ -> ws >>= fun _ -> typeAnnotation) >>= fun typeOpt ->
            match valueOpt, typeOpt with
            | Some value, Some typeStr -> succeed (sprintf "llvm.return %s : %s" value typeStr)
            | None, _ -> succeed "llvm.return"
            | Some value, None -> succeed (sprintf "llvm.return %s" value)
        
        let transformFuncCall =
            mlirOperation >>= fun (opName, results, operands, attrs, typeStr) ->
            if opName = "func.call" then
                match Map.tryFind "callee" attrs with
                | Some callee ->
                    let resultStr = if results.IsEmpty then "" else String.concat ", " results + " = "
                    let operandStr = String.concat ", " operands
                    succeed (sprintf "%sllvm.call %s(%s) : %s" resultStr callee operandStr typeStr)
                | None ->
                    compilerFail (TransformError("func-to-llvm", "func.call", "llvm.call", "Missing callee attribute"))
            else
                compilerFail (TransformError("func-to-llvm", opName, "llvm", "Unsupported func operation"))
        
        transformFuncOp <|> transformFuncReturn <|> transformFuncCall
        |> withErrorContext "func to LLVM transformation"
    
    /// Transforms arith dialect operations to LLVM dialect
    let arithToLLVMTransformer : Parser<string, LoweringState> =
        let transformBinaryOp (arithOp: string) (llvmOp: string) =
            mlirOperation >>= fun (opName, results, operands, attrs, typeStr) ->
            if opName = ("arith." + arithOp) && operands.Length = 2 then
                let resultStr = if results.IsEmpty then "" else String.concat ", " results + " = "
                succeed (sprintf "%sllvm.%s %s, %s : %s" resultStr llvmOp operands.[0] operands.[1] typeStr)
            else
                compilerFail (TransformError("arith-to-llvm", opName, "llvm", "Invalid binary operation format"))
        
        let transformConstant =
            mlirOperation >>= fun (opName, results, operands, attrs, typeStr) ->
            if opName = "arith.constant" then
                match Map.tryFind "value" attrs with
                | Some value ->
                    let resultStr = if results.IsEmpty then "" else String.concat ", " results + " = "
                    succeed (sprintf "%sllvm.mlir.constant(%s) : %s" resultStr value typeStr)
                | None ->
                    compilerFail (TransformError("arith-to-llvm", "arith.constant", "llvm.mlir.constant", "Missing value attribute"))
            else
                compilerFail (TransformError("arith-to-llvm", opName, "llvm", "Not a constant operation"))
        
        transformBinaryOp "addi" "add" <|>
        transformBinaryOp "subi" "sub" <|>
        transformBinaryOp "muli" "mul" <|>
        transformBinaryOp "divsi" "sdiv" <|>
        transformConstant
        |> withErrorContext "arith to LLVM transformation"
    
    /// Transforms standard dialect operations to LLVM dialect
    let stdToLLVMTransformer : Parser<string, LoweringState> =
        mlirOperation >>= fun (opName, results, operands, attrs, typeStr) ->
        match opName with
        | "std.alloc" ->
            let resultStr = if results.IsEmpty then "" else String.concat ", " results + " = "
            succeed (sprintf "%sllvm.alloca %s : %s" resultStr (String.concat ", " operands) typeStr)
        | "std.load" ->
            let resultStr = if results.IsEmpty then "" else String.concat ", " results + " = "
            succeed (sprintf "%sllvm.load %s : %s" resultStr (String.concat ", " operands) typeStr)
        | "std.store" ->
            if operands.Length >= 2 then
                succeed (sprintf "llvm.store %s, %s : %s" operands.[0] operands.[1] typeStr)
            else
                compilerFail (TransformError("std-to-llvm", "std.store", "llvm.store", "Insufficient operands for store"))
        | _ ->
            compilerFail (TransformError("std-to-llvm", opName, "llvm", "Unsupported std operation"))
        |> withErrorContext "std to LLVM transformation"

/// Pass management using XParsec combinators
module PassManagement =
    
    /// Creates a lowering pass with validation
    let createLoweringPass (name: string) (source: MLIRDialect) (target: MLIRDialect) 
                          (transformer: Parser<string, LoweringState>) : LoweringPass =
        {
            Name = name
            SourceDialect = source
            TargetDialect = target
            Transform = transformer
            Validate = fun _ -> true  // Would implement proper validation
        }
    
    /// Applies a single lowering pass to MLIR text
    let applyPass (pass: LoweringPass) (mlirText: string) : CompilerResult<string> =
        let lines = mlirText.Split('\n') |> Array.toList
        let initialState = {
            CurrentDialect = pass.SourceDialect
            TargetDialect = pass.TargetDialect
            TransformedOperations = []
            SymbolTable = Map.empty
            PassName = pass.Name
            ErrorContext = []
        }
        
        let transformLine (line: string) : CompilerResult<string> =
            let trimmedLine = line.Trim()
            if String.IsNullOrEmpty(trimmedLine) || trimmedLine.StartsWith("//") then
                Success line
            else
                match pass.Transform trimmedLine initialState with
                | Reply(Ok transformedLine, _) -> Success transformedLine
                | Reply(Error, error) -> 
                    CompilerFailure [TransformError(pass.Name, trimmedLine, pass.TargetDialect.ToString(), error)]
        
        let transformAllLines (lines: string list) : CompilerResult<string list> =
            let rec processLines acc remaining =
                match remaining with
                | [] -> Success (List.rev acc)
                | line :: rest ->
                    match transformLine line with
                    | Success transformedLine -> processLines (transformedLine :: acc) rest
                    | CompilerFailure errors -> CompilerFailure errors
            processLines [] lines
        
        transformAllLines lines >>= fun transformedLines ->
        Success (String.concat "\n" transformedLines)
    
    /// Validates that transformed MLIR is in target dialect
    let validateDialectPurity (targetDialect: MLIRDialect) (mlirText: string) : CompilerResult<unit> =
        let lines = mlirText.Split('\n')
        let dialectPrefix = dialectToString targetDialect + "."
        
        let invalidLines = 
            lines 
            |> Array.mapi (fun i line -> (i + 1, line.Trim()))
            |> Array.filter (fun (_, line) -> 
                not (String.IsNullOrEmpty(line)) && 
                not (line.StartsWith("//")) &&
                not (line.StartsWith("module")) &&
                not (line.StartsWith("func")) &&
                line.Contains(".") &&
                not (line.Contains(dialectPrefix)))
        
        if invalidLines.Length > 0 then
            let errorDetails = 
                invalidLines 
                |> Array.map (fun (lineNum, line) -> sprintf "Line %d: %s" lineNum line)
                |> String.concat "\n"
            CompilerFailure [TransformError(
                "dialect validation", 
                "mixed dialects", 
                dialectToString targetDialect, 
                sprintf "Found non-%s operations:\n%s" (dialectToString targetDialect) errorDetails)]
        else
            Success ()

/// Standard lowering pipeline creation
let createStandardLoweringPipeline() : LoweringPass list = [
    PassManagement.createLoweringPass "convert-func-to-llvm" Func LLVM DialectTransformers.funcToLLVMTransformer
    PassManagement.createLoweringPass "convert-arith-to-llvm" Arith LLVM DialectTransformers.arithToLLVMTransformer  
    PassManagement.createLoweringPass "convert-std-to-llvm" Standard LLVM DialectTransformers.stdToLLVMTransformer
]

/// Applies the complete lowering pipeline - NO FALLBACKS
let applyLoweringPipeline (mlirModule: string) : CompilerResult<string> =
    if String.IsNullOrWhiteSpace(mlirModule) then
        CompilerFailure [TransformError("lowering pipeline", "empty input", "LLVM dialect", "Input MLIR module is empty")]
    else
        let passes = createStandardLoweringPipeline()
        
        let applyPassSequence (currentText: string) (remainingPasses: LoweringPass list) : CompilerResult<string> =
            match remainingPasses with
            | [] -> Success currentText
            | pass :: rest ->
                PassManagement.applyPass pass currentText >>= fun transformedText ->
                applyPassSequence transformedText rest
        
        applyPassSequence mlirModule passes >>= fun loweredMLIR ->
        PassManagement.validateDialectPurity LLVM loweredMLIR >>= fun _ ->
        Success loweredMLIR

/// Validates that lowered MLIR contains only LLVM dialect operations
let validateLLVMDialectOnly (llvmDialectModule: string) : CompilerResult<unit> =
    PassManagement.validateDialectPurity LLVM llvmDialectModule

/// Creates a custom lowering pipeline for specific dialect combinations
let createCustomLoweringPipeline (sourceDialects: MLIRDialect list) (targetDialect: MLIRDialect) : CompilerResult<LoweringPass list> =
    let availableTransformers = [
        (Func, LLVM, DialectTransformers.funcToLLVMTransformer)
        (Arith, LLVM, DialectTransformers.arithToLLVMTransformer)
        (Standard, LLVM, DialectTransformers.stdToLLVMTransformer)
    ]
    
    let createPassForDialect (source: MLIRDialect) : CompilerResult<LoweringPass> =
        match availableTransformers |> List.tryFind (fun (s, t, _) -> s = source && t = targetDialect) with
        | Some (s, t, transformer) ->
            let passName = sprintf "convert-%s-to-%s" (dialectToString s) (dialectToString t)
            Success (PassManagement.createLoweringPass passName s t transformer)
        | None ->
            CompilerFailure [TransformError(
                "pipeline creation", 
                dialectToString source, 
                dialectToString targetDialect, 
                sprintf "No transformer available for %s -> %s" (dialectToString source) (dialectToString targetDialect))]
    
    sourceDialects
    |> List.map createPassForDialect
    |> List.fold (fun acc result ->
        match acc, result with
        | Success passes, Success pass -> Success (pass :: passes)
        | CompilerFailure errors, Success _ -> CompilerFailure errors
        | Success _, CompilerFailure errors -> CompilerFailure errors
        | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
    ) (Success [])
    |>> List.rev