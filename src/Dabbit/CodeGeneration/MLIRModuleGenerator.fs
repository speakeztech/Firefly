module Dabbit.CodeGeneration.MLIRModuleGenerator

open FSharp.Compiler.Syntax
open Core.Types.TypeSystem
open Core.XParsec.Foundation
open TypeMapping
open MLIREmitter
open MLIRControlFlow
open MLIRTypeOperations

// ===================================================================
// AST Pattern Combinators - Using XParsec for Pattern Matching
// ===================================================================

/// AST parser type - parses F# AST nodes to produce MLIR values
type ASTParser<'AST, 'Result> = 'AST -> MLIRCombinator<'Result option>

/// Lift a pattern match function into an AST parser
let astPattern (pattern: 'AST -> 'Result option) : ASTParser<'AST, 'Result> =
    fun ast -> MLIRCombinators.lift (pattern ast)

/// Choice combinator for AST parsers
let (<|>) (p1: ASTParser<'AST, 'Result>) (p2: ASTParser<'AST, 'Result>) : ASTParser<'AST, 'Result> =
    fun ast ->
        mlir {
            let! result1 = p1 ast
            match result1 with
            | Some r -> return Some r
            | None -> return! p2 ast
        }

/// Sequence combinator for AST parsers
let (>>>) (p1: ASTParser<'AST, 'A>) (f: 'A -> MLIRCombinator<'B>) : ASTParser<'AST, 'B> =
    fun ast ->
        mlir {
            let! result1 = p1 ast
            match result1 with
            | Some a -> 
                let! b = f a
                return Some b
            | None -> return None
        }

/// Map combinator for AST parsers
let (|>>) (p: ASTParser<'AST, 'A>) (f: 'A -> 'B) : ASTParser<'AST, 'B> =
    fun ast ->
        mlir {
            let! result = p ast
            return Option.map f result
        }

/// Try multiple AST parsers until one succeeds
let choice (parsers: ASTParser<'AST, 'Result> list) : ASTParser<'AST, 'Result> =
    List.reduce (<|>) parsers

/// Always succeed with a value
let preturn (value: 'Result) : ASTParser<'AST, 'Result> =
    fun _ -> MLIRCombinators.lift (Some value)

/// Always fail
let pfail : ASTParser<'AST, 'Result> =
    fun _ -> MLIRCombinators.lift None

// ===================================================================
// Expression Pattern Parsers - Composable Expression Matching
// ===================================================================

/// Parse integer constants with type preservation
let pInt32Const : ASTParser<SynExpr, int * MLIRType> =
    astPattern (function
        | SynExpr.Const(SynConst.Int32 n, _) -> Some (n, MLIRTypes.i32)
        | _ -> None)

let pInt64Const : ASTParser<SynExpr, int64 * MLIRType> =
    astPattern (function
        | SynExpr.Const(SynConst.Int64 n, _) -> Some (n, MLIRTypes.i64)
        | _ -> None)

let pFloatConst : ASTParser<SynExpr, float * MLIRType> =
    astPattern (function
        | SynExpr.Const(SynConst.Double f, _) -> Some (f, MLIRTypes.f64)
        | _ -> None)

let pFloat32Const : ASTParser<SynExpr, float32 * MLIRType> =
    astPattern (function
        | SynExpr.Const(SynConst.Single f, _) -> Some (f, MLIRTypes.f32)
        | _ -> None)

let pBoolConst : ASTParser<SynExpr, bool * MLIRType> =
    astPattern (function
        | SynExpr.Const(SynConst.Bool b, _) -> Some (b, MLIRTypes.i1)
        | _ -> None)

let pStringConst : ASTParser<SynExpr, string * MLIRType> =
    astPattern (function
        | SynExpr.Const(SynConst.String(s, _, _), _) -> Some (s, MLIRTypes.memref MLIRTypes.i8)
        | _ -> None)

let pUnitConst : ASTParser<SynExpr, unit * MLIRType> =
    astPattern (function
        | SynExpr.Const(SynConst.Unit, _) -> Some ((), MLIRTypes.void_)
        | _ -> None)

/// Parse identifiers with type lookup
let pIdent : ASTParser<SynExpr, string> =
    astPattern (function
        | SynExpr.Ident(ident) -> Some ident.idText
        | _ -> None)

/// Parse function application
let pApp : ASTParser<SynExpr, (SynExpr * SynExpr)> =
    astPattern (function
        | SynExpr.App(_, _, funcExpr, argExpr, _) -> Some (funcExpr, argExpr)
        | _ -> None)

/// Parse if-then-else
let pIfThenElse : ASTParser<SynExpr, (SynExpr * SynExpr * SynExpr option)> =
    astPattern (function
        | SynExpr.IfThenElse(cond, thenExpr, elseExpr, _, _, _, _) -> Some (cond, thenExpr, elseExpr)
        | _ -> None)

/// Parse let binding
let pLet : ASTParser<SynExpr, (SynBinding list * SynExpr)> =
    astPattern (function
        | SynExpr.LetOrUse(_, _, bindings, body, _, _) -> Some (bindings, body)
        | _ -> None)

// ===================================================================
// MLIR Generation Combinators - Type-Preserving Transformations
// ===================================================================

/// Generate MLIR constant from parsed value
let generateConstant (value: 'T) (typ: MLIRType) : MLIRCombinator<MLIRValue> =
    mlir {
        let valueStr = 
            match box value with
            | :? bool as b -> if b then "1" else "0"
            | :? string as s -> sprintf "\"%s\"" s
            | v -> string v
        return! mlirConstant valueStr typ
    }

/// Generate identifier reference with type checking
let generateIdentRef (name: string) : MLIRCombinator<MLIRValue> =
    mlir {
        let! state = getState
        match Map.tryFind name state.LocalVars with
        | Some (ssa, typeStr) ->
            let mlirType = parseTypeFromString typeStr
            return mlirValue ssa mlirType false
        | None ->
            // Check if it's a function
            match Map.tryFind name state.SymbolRegistry.State.SymbolsByShort with
            | Some symbol ->
                return mlirValue ("@" + name) symbol.ReturnType false
            | None ->
                do! emitComment (sprintf "Undefined identifier: %s" name)
                return mlirValue "%undefined" MLIRTypes.i32 false
    }

/// Generate function call with type checking
let generateCall (funcName: string) (arg: MLIRValue) : MLIRCombinator<MLIRValue> =
    mlir {
        let! state = getState
        match Map.tryFind funcName state.SymbolRegistry.State.SymbolsByShort with
        | Some symbol ->
            let! result = nextSSA "call"
            let returnType = symbol.ReturnType
            do! emitLine (sprintf "%s = func.call @%s(%s) : (%s) -> %s" 
                            result funcName arg.SSA arg.Type (mlirTypeToString returnType))
            return mlirValue result returnType false
        | None ->
            // Unknown function - default to i32 return
            let! result = nextSSA "call"
            do! emitLine (sprintf "%s = func.call @%s(%s) : (%s) -> i32" 
                            result funcName arg.SSA arg.Type)
            return mlirValue result MLIRTypes.i32 false
    }

// ===================================================================
// Main Expression Parser - Composing All Patterns
// ===================================================================

/// Parse any constant expression
let pAnyConst : ASTParser<SynExpr, MLIRValue> =
    choice [
        pInt32Const >>> fun (n, typ) -> generateConstant n typ
        pInt64Const >>> fun (n, typ) -> generateConstant n typ
        pFloatConst >>> fun (f, typ) -> generateConstant f typ
        pFloat32Const >>> fun (f, typ) -> generateConstant f typ
        pBoolConst >>> fun (b, typ) -> generateConstant b typ
        pStringConst >>> fun (s, typ) -> generateConstant s typ
        pUnitConst >>> fun (_, typ) -> generateConstant () typ
    ]

/// Parse and generate expressions recursively
let rec pExpression : ASTParser<SynExpr, MLIRValue> =
    let pSimpleExpr = 
        pAnyConst
        <|> (pIdent >>> generateIdentRef)
        <|> pApplicationExpr
        <|> pIfThenElseExpr
        <|> pLetExpr
        <|> pDefault
    
    pSimpleExpr

and pApplicationExpr : ASTParser<SynExpr, MLIRValue> =
    pApp >>> fun (funcExpr, argExpr) ->
        mlir {
            // Parse function name
            let! funcOpt = (pIdent |>> Some) funcExpr
            let! argOpt = pExpression argExpr
            
            match funcOpt, argOpt with
            | Some funcName, Some arg ->
                return! generateCall funcName arg
            | _ ->
                do! emitComment "Complex function application not supported"
                return mlirValue "%error" MLIRTypes.i32 false
        }

and pIfThenElseExpr : ASTParser<SynExpr, MLIRValue> =
    pIfThenElse >>> fun (cond, thenExpr, elseOpt) ->
        mlir {
            let! condOpt = pExpression cond
            let! thenOpt = pExpression thenExpr
            
            match condOpt, thenOpt with
            | Some condVal, Some thenVal ->
                match elseOpt with
                | Some elseExpr ->
                    let! elseOpt = pExpression elseExpr
                    match elseOpt with
                    | Some elseVal ->
                        // Generate select instruction
                        let! result = nextSSA "select"
                        let resultType = thenVal.Type  // Should unify types
                        do! emitLine (sprintf "%s = arith.select %s, %s, %s : %s" 
                                        result condVal.SSA thenVal.SSA elseVal.SSA resultType)
                        return mlirValue result (parseTypeFromString resultType) false
                    | None ->
                        return thenVal
                | None ->
                    return thenVal
            | _ ->
                return mlirValue "%error" MLIRTypes.i32 false
        }

and pLetExpr : ASTParser<SynExpr, MLIRValue> =
    pLet >>> fun (bindings, body) ->
        mlir {
            // Process bindings
            for binding in bindings do
                do! processBinding binding
            
            // Process body
            let! bodyOpt = pExpression body
            match bodyOpt with
            | Some value -> return value
            | None -> return mlirValue "%error" MLIRTypes.i32 false
        }

and pDefault : ASTParser<SynExpr, MLIRValue> =
    fun expr ->
        mlir {
            do! emitComment (sprintf "Unsupported expression: %A" expr)
            return Some (mlirValue "%unsupported" MLIRTypes.i32 false)
        }

and processBinding (binding: SynBinding) : MLIRCombinator<unit> =
    mlir {
        let (SynBinding(_, _, _, _, _, _, _, headPat, _, expr, _, _, _)) = binding
        
        match headPat with
        | SynPat.Named(SynIdent(ident, _), _, _, _) ->
            let! exprOpt = pExpression expr
            match exprOpt with
            | Some value ->
                do! bindLocal ident.idText value.SSA value.Type
            | None ->
                do! emitComment (sprintf "Failed to parse binding expression for %s" ident.idText)
        | _ ->
            do! emitComment "Complex binding pattern not supported"
    }

// ===================================================================
// Type Definition Parsers - Pattern Matching on Type AST
// ===================================================================

/// Parse record type fields
let pRecordField : ASTParser<SynField, (string * MLIRType)> =
    fun field ->
        mlir {
            let (SynField(_, _, idOpt, type', _, _, _, _, _)) = field
            let fieldName = 
                match idOpt with
                | Some ident -> ident.idText
                | None -> "_anonymous_"
            let mlirType = TypeConversion.synTypeToMLIRType type'
            return Some (fieldName, mlirType)
        }

/// Parse union case
let pUnionCase : ASTParser<SynUnionCase, (string * MLIRType list)> =
    fun unionCase ->
        mlir {
            let (SynUnionCase(_, ident, caseType, _, _, _, _)) = unionCase
            let caseName = ident.idText
            let fieldTypes =
                match caseType with
                | SynUnionCase.Fields fields ->
                    fields |> List.map (fun field ->
                        let (SynField(_, _, _, type', _, _, _, _, _)) = field
                        TypeConversion.synTypeToMLIRType type')
                | SynUnionCase.FullType _ -> []
            return Some (caseName, fieldTypes)
        }

/// Parse type definition representation
let pTypeDefnRepr : ASTParser<SynTypeDefnRepr, unit> =
    astPattern (function
        | SynTypeDefnRepr.Simple(simpleRepr, _) -> Some simpleRepr
        | _ -> None) >>> fun simpleRepr ->
            mlir {
                match simpleRepr with
                | SynTypeDefnSimpleRepr.Record(_, fields, _) ->
                    let! fieldParsers = fields |> List.map pRecordField |> sequence
                    let fields = fieldParsers |> List.choose id
                    // Generate record type
                    return ()
                | _ ->
                    return ()
            }

// ===================================================================
// Module-Level Parsers
// ===================================================================

/// Parse function binding
let pFunctionBinding : ASTParser<SynBinding, unit> =
    fun binding ->
        mlir {
            let (SynBinding(_, _, _, _, _, _, valData, headPat, returnInfo, expr, _, _, _)) = binding
            
            match headPat with
            | SynPat.Named(SynIdent(ident, _), _, _, _) ->
                let functionName = ident.idText
                
                // Extract parameters
                let parameters = 
                    match valData with
                    | SynValData(_, _, Some(SynValInfo(paramGroups, _))) ->
                        paramGroups 
                        |> List.concat
                        |> List.mapi (fun i _ -> (sprintf "arg%d" i, MLIRTypes.i32))
                    | _ -> []
                
                // Extract return type
                let returnType =
                    match returnInfo with
                    | Some(SynBindingReturnInfo(synReturnType, _, _, _)) ->
                        TypeConversion.synTypeToMLIRType synReturnType
                    | None -> MLIRTypes.void_
                
                // Generate function
                let returnTypes = if returnType = MLIRTypes.void_ then [] else [returnType]
                do! mlirFuncDecl functionName parameters returnTypes
                
                // Generate body
                do! indentScope <|
                    mlir {
                        // Bind parameters
                        for (name, typ) in parameters do
                            do! bindLocal name ("%" + name) (mlirTypeToString typ)
                        
                        // Parse body expression
                        let! bodyOpt = pExpression expr
                        match bodyOpt with
                        | Some result ->
                            if returnType = MLIRTypes.void_ then
                                do! mlirReturn []
                            else
                                do! mlirReturn [result]
                        | None ->
                            do! emitComment "Failed to parse function body"
                            do! mlirReturn []
                    }
                    
                do! emitLine "}"
                return Some ()
            | _ ->
                do! emitComment "Unsupported binding pattern"
                return None
        }

/// Parse module declaration
let rec pModuleDecl : ASTParser<SynModuleDecl, unit> =
    astPattern (fun decl -> Some decl) >>> fun decl ->
        mlir {
            match decl with
            | SynModuleDecl.Let(_, bindings, _) ->
                for binding in bindings do
                    let! _ = pFunctionBinding binding
                    ()
                return ()
                
            | SynModuleDecl.Types(typeDefns, _) ->
                do! emitComment "Type definitions"
                return ()
                
            | SynModuleDecl.NestedModule(componentInfo, _, decls, _, _, _) ->
                let (SynComponentInfo(_, _, _, longId, _, _, _, _)) = componentInfo
                let moduleName = String.concat "." (longId |> List.map (fun id -> id.idText))
                do! emitComment (sprintf "Nested module: %s" moduleName)
                for decl in decls do
                    let! _ = pModuleDecl decl
                    ()
                return ()
                
            | _ ->
                do! emitComment "Unsupported module declaration"
                return ()
        }

// ===================================================================
// Helper Functions for List Processing with Parsers
// ===================================================================

/// Sequence a list of parsers
let sequence (parsers: MLIRCombinator<'T option> list) : MLIRCombinator<'T option list> =
    mlir {
        let rec loop acc = function
            | [] -> return List.rev acc
            | p::ps ->
                let! result = p
                return! loop (result::acc) ps
        return! loop [] parsers
    }

// ===================================================================
// Main Entry Point - Parse Entire AST
// ===================================================================

/// Generate MLIR from parsed F# input using XParsec patterns
let generateMLIR (parsedInput: ParsedInput) : MLIRBuilder<unit> =
    mlir {
        do! emitLine "module {"
        do! indentScope <|
            mlir {
                match parsedInput with
                | ParsedInput.ImplFile(ParsedImplFileInput(fileName, _, _, _, _, modules, _, _)) ->
                    do! emitComment (sprintf "File: %s" fileName)
                    
                    for (SynModuleOrNamespace(longId, _, _, decls, _, _, _, _, _)) in modules do
                        let namespaceName = String.concat "." (longId |> List.map (fun id -> id.idText))
                        do! emitComment (sprintf "Module/namespace: %s" namespaceName)
                        
                        for decl in decls do
                            let! _ = pModuleDecl decl
                            ()
                            
                | ParsedInput.SigFile _ ->
                    do! emitComment "Signature file processing not implemented"
            }
        do! emitLine "}"
    }

/// Run the MLIR generator
let generateModuleFromAST (ast: ParsedInput) : string =
    let builderState = MLIREmitter.createInitialState()
    
    match MLIREmitter.runBuilder (generateMLIR ast) builderState with
    | Success(_, state) -> state.Output.ToString()
    | Failure(error) -> sprintf "// Error generating MLIR: %s" error