module Dabbit.Parsing.AstConverter

open System
open FSharp.Compiler.Syntax
open FSharp.Compiler.Text
open Fantomas.Core
open Fantomas.FCS
open Dabbit.Parsing.OakAst

/// Converts F# type to Oak type (simplified)
let rec private convertTypeToOak (synType: SynType option) : OakType =
    match synType with
    | Some(SynType.LongIdent(longIdent)) ->
        match longIdent.LongIdent with
        | [ident] ->
            match ident.idText with
            | "int" -> IntType
            | "float" -> FloatType
            | "bool" -> BoolType
            | "string" -> StringType
            | "unit" -> UnitType
            | _ -> UnitType
        | _ -> UnitType
    | Some(SynType.Fun(argType, returnType, _, _)) ->
        let argOakType = convertTypeToOak (Some argType)
        let returnOakType = convertTypeToOak (Some returnType)
        FunctionType([argOakType], returnOakType)
    | _ -> UnitType

/// Converts F# expression to Oak expression (simplified)
and private convertExpressionToOak (expr: SynExpr) : OakExpression =
    match expr with
    | SynExpr.Const(constant, range) ->
        match constant with
        | SynConst.Int32(value) -> Literal(IntLiteral(value))
        | SynConst.Double(value) -> Literal(FloatLiteral(value))
        | SynConst.String(value, range, _) -> Literal(StringLiteral(value))
        | SynConst.Bool(value) -> Literal(BoolLiteral(value))
        | SynConst.Unit -> Literal(UnitLiteral)
        | _ -> Literal(UnitLiteral)
    
    | SynExpr.Ident(ident) ->
        Variable(ident.idText)
    
    | SynExpr.App(_, _, funcExpr, argExpr, _) ->
        let func = convertExpressionToOak funcExpr
        let arg = convertExpressionToOak argExpr
        Application(func, [arg])
    
    | SynExpr.Lambda(_, _, args, body, _, _, _) ->
        let oakArgs = [("x", UnitType)] // Simplified argument handling
        let oakBody = convertExpressionToOak body
        Lambda(oakArgs, oakBody)
    
    | SynExpr.LetOrUse(_, _, bindings, body, _, _) ->
        let oakBody = convertExpressionToOak body
        Let("temp", Literal(UnitLiteral), oakBody) // Simplified binding handling
    
    | SynExpr.IfThenElse(cond, thenExpr, elseExpr, _, _, _, _) ->
        let oakCond = convertExpressionToOak cond
        let oakThen = convertExpressionToOak thenExpr
        let oakElse = 
            match elseExpr with
            | Some expr -> convertExpressionToOak expr
            | None -> Literal(UnitLiteral)
        IfThenElse(oakCond, oakThen, oakElse)
    
    | SynExpr.Sequential(_, _, expr1, expr2, _, _) ->
        let oak1 = convertExpressionToOak expr1
        let oak2 = convertExpressionToOak expr2
        Sequential(oak1, oak2)
    
    | _ -> Literal(UnitLiteral)

/// Extracts name from SynBinding (simplified)
let private getBindingName (binding: SynBinding) : string =
    match binding with
    | SynBinding(_, _, _, _, _, _, SynValData(_, _, _), pat, _, _, _, _, _) ->
        match pat with
        | SynPat.Named(synIdent, _, _, _) -> 
            match synIdent with
            | SynIdent(ident, _) -> ident.idText
        | SynPat.LongIdent(longDotId, _, _, _, _, _) ->
            match longDotId.LongIdent with
            | [ident] -> ident.idText
            | _ -> "unknown"
        | _ -> "unknown"

/// Extracts expression from SynBinding (simplified)
let private getBindingExpr (binding: SynBinding) : SynExpr =
    match binding with
    | SynBinding(_, _, _, _, _, _, _, _, _, expr, _, _, _) -> expr

/// Converts F# module declaration to Oak declaration
let private convertDeclarationToOak (decl: SynModuleDecl) : OakDeclaration option =
    match decl with
    | SynModuleDecl.Let(_, bindings, _) ->
        match bindings with
        | [binding] ->
            let name = getBindingName binding
            let expr = getBindingExpr binding
            let body = convertExpressionToOak expr
            Some(FunctionDecl(name, [], UnitType, body))
        | _ -> None
    
    | SynModuleDecl.Types(typeDefs, _) ->
        match typeDefs with
        | [SynTypeDefn(typeInfo, _, _, _, _, _)] ->
            match typeInfo with
            | SynComponentInfo(_, _, _, [ident], _, _, _, _) ->
                Some(TypeDecl(ident.idText, UnitType))
            | _ -> Some(TypeDecl("UnknownType", UnitType))
        | _ -> None
    
    | SynModuleDecl.Expr(expr, _) ->
        let oakExpr = convertExpressionToOak expr
        Some(EntryPoint(oakExpr))
    
    | _ -> None

/// Converts F# AST to Oak AST
let convertToOakAst (parsedInput: ParsedInput) : OakProgram =
    let modules = 
        match parsedInput with
        | ParsedInput.ImplFile(ParsedImplFileInput(fileName, isScript, qualName, scopedPragmas, hashDirectives, modules, _, _, _)) ->
            modules |> List.map (fun parsedModule ->
                match parsedModule with
                | SynModuleOrNamespace(longIdent, isRecursive, moduleKind, declarations, preXmlDoc, attribs, access, range, _) ->
                    let moduleName = 
                        longIdent 
                        |> List.map (fun ident -> ident.idText) 
                        |> String.concat "."
                        |> fun name -> if String.IsNullOrEmpty(name) then "Main" else name
                    
                    let oakDecls = 
                        declarations 
                        |> List.choose convertDeclarationToOak
                    
                    { Name = moduleName; Declarations = oakDecls }
            )
        | ParsedInput.SigFile(_) -> []

    { Modules = modules }

/// Simple F# parser using basic string analysis as fallback
let private parseSimpleFs (sourceCode: string) : OakProgram =
    // Very basic parsing - create a simple main function
    let hasMain = sourceCode.Contains("main") || sourceCode.Contains("printfn")
    
    if hasMain then
        let mainDecl = EntryPoint(Literal(IntLiteral(0)))
        { Modules = [{ Name = "Main"; Declarations = [mainDecl] }] }
    else
        { Modules = [{ Name = "Main"; Declarations = [] }] }

/// Parses F# code and converts to Oak AST
let parseAndConvertToOakAst (sourceCode: string) : OakProgram =
    try
        // Create a basic parser using FSharp.Compiler.Service
        let sourceText = SourceText.ofString sourceCode
        let fileName = "temp.fs"
        
        // For now, use simplified parsing since Fantomas API is unstable
        parseSimpleFs sourceCode
    with
    | ex ->
        printfn "Parse error: %s" ex.Message
        // Return minimal program on parse failure
        { Modules = [{ Name = "Main"; Declarations = [EntryPoint(Literal(UnitLiteral))] }] }