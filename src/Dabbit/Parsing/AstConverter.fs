module Dabbit.Parsing.AstConverter

open System
open FSharp.Compiler.Syntax
open FSharp.Compiler.Text
open Dabbit.Parsing.OakAst

/// Result type for F# to Oak AST conversion with diagnostics
type ASTConversionResult = {
    OakProgram: OakProgram
    Diagnostics: string list
}

/// Helper functions for extracting names and types from F# AST nodes
module AstHelpers =
    
    let getIdentifierName (ident: Ident) : string = ident.idText
    
    let getQualifiedName (idents: Ident list) : string =
        idents |> List.map (fun id -> id.idText) |> String.concat "."
    
    let extractLongIdent (synLongIdent: SynLongIdent) : Ident list =
        let (SynLongIdent(idents, _, _)) = synLongIdent
        idents
    
    let extractIdent (synIdent: SynIdent) : Ident =
        let (SynIdent(ident, _)) = synIdent
        ident
    
    let extractPatternName (pattern: SynPat) : string =
        match pattern with
        | SynPat.Named(synIdent, _, _, _) -> 
            synIdent |> extractIdent |> getIdentifierName
        | SynPat.LongIdent(longDotId, _, _, _, _, _) ->
            longDotId |> extractLongIdent |> getQualifiedName
        | SynPat.Const(SynConst.Unit, _) -> "()"
        | _ -> "_"
    
    let hasEntryPointAttribute (attributes: SynAttributes) : bool =
        attributes
        |> List.exists (fun attrList ->
            attrList.Attributes 
            |> List.exists (fun attr ->
                match attr.TypeName with
                | SynLongIdent(idents, _, _) -> 
                    idents |> List.exists (fun id -> 
                        id.idText.Contains("EntryPoint"))))

/// Type mapping functions
module TypeMapping =
    
    let mapBasicType (typeName: string) : OakType =
        match typeName.ToLowerInvariant() with
        | "int" | "int32" | "system.int32" -> IntType
        | "float" | "double" | "system.double" -> FloatType
        | "bool" | "boolean" | "system.boolean" -> BoolType
        | "string" | "system.string" -> StringType
        | "unit" -> UnitType
        | _ when typeName.StartsWith("array") || typeName.Contains("[]") -> ArrayType(IntType)
        | _ -> StructType([])
    
    let mapLiteral (constant: SynConst) : OakLiteral =
        match constant with
        | SynConst.Int32 n -> IntLiteral(n)
        | SynConst.Double f -> FloatLiteral(f)
        | SynConst.Bool b -> BoolLiteral(b)
        | SynConst.String(s, _, _) -> StringLiteral(s)
        | SynConst.Unit -> UnitLiteral
        | _ -> UnitLiteral

/// Expression conversion functions
module ExpressionMapping =
    
    let rec mapExpression (expr: SynExpr) : OakExpression =
        match expr with
        | SynExpr.Const(constant, _) ->
            constant |> TypeMapping.mapLiteral |> Literal
        
        | SynExpr.Ident(ident) ->
            ident |> AstHelpers.getIdentifierName |> Variable
        
        | SynExpr.LongIdent(_, longIdent, _, _) ->
            longIdent |> AstHelpers.extractLongIdent |> AstHelpers.getQualifiedName |> Variable
        
        | SynExpr.App(_, _, funcExpr, argExpr, _) ->
            mapFunctionApplication funcExpr argExpr
        
        | SynExpr.LetOrUse(_, _, bindings, bodyExpr, _, _) ->
            mapLetBinding bindings bodyExpr
        
        | SynExpr.IfThenElse(condExpr, thenExpr, elseExprOpt, _, _, _, _) ->
            mapConditional condExpr thenExpr elseExprOpt
        
        | SynExpr.Sequential(_, _, first, second, _, _) ->
            Sequential(mapExpression first, mapExpression second)
        
        | SynExpr.Lambda(_, _, _, body, parsedData, _, _) ->
            mapLambda parsedData body
        
        | _ -> Literal(UnitLiteral)
    
    and mapFunctionApplication funcExpr argExpr =
        let func = mapExpression funcExpr
        let arg = mapExpression argExpr
        
        match func with
        | Variable "printf" ->
            match arg with
            | Literal(StringLiteral formatStr) -> IOOperation(Printf(formatStr), [])
            | _ -> Application(func, [arg])
        | Variable "printfn" ->
            match arg with
            | Literal(StringLiteral formatStr) -> IOOperation(Printfn(formatStr), [])
            | _ -> Application(func, [arg])
        | _ -> Application(func, [arg])
    
    and mapLetBinding bindings bodyExpr =
        match bindings with
        | binding :: _ ->
            let (SynBinding(_, _, _, _, _, _, _, headPat, _, expr, _, _, _)) = binding
            let name = AstHelpers.extractPatternName headPat
            Let(name, mapExpression expr, mapExpression bodyExpr)
        | [] -> mapExpression bodyExpr
    
    and mapConditional condExpr thenExpr elseExprOpt =
        let cond = mapExpression condExpr
        let thenBranch = mapExpression thenExpr
        let elseBranch = 
            match elseExprOpt with
            | Some(elseExpr) -> mapExpression elseExpr
            | None -> Literal(UnitLiteral)
        IfThenElse(cond, thenBranch, elseBranch)
    
    and mapLambda parsedData body =
        let params' = 
            match parsedData with
            | Some(originalPats, _) ->
                originalPats |> List.map (fun pat -> (AstHelpers.extractPatternName pat, UnitType))
            | None -> [("x", UnitType)]
        Lambda(params', mapExpression body)

/// Declaration mapping functions  
module DeclarationMapping =
    
    let mapUnionCase (case: SynUnionCase) : string * OakType option =
        let (SynUnionCase(_, synIdent, caseType, _, _, _, _)) = case
        let ident = AstHelpers.extractIdent synIdent
        let caseName = AstHelpers.getIdentifierName ident
        
        match caseType with
        | SynUnionCaseKind.Fields fields ->
            if fields.IsEmpty then (caseName, None)
            else (caseName, Some UnitType)
        | _ -> (caseName, None)
    
    let mapRecordField (field: SynField) : string * OakType =
        let (SynField(_, _, fieldId, _, _, _, _, _, _)) = field
        let fieldName = 
            match fieldId with
            | Some ident -> AstHelpers.getIdentifierName ident
            | None -> "_"
        (fieldName, UnitType)
    
    let mapTypeDefinition (typeDefn: SynTypeDefn) : OakDeclaration option =
        let (SynTypeDefn(SynComponentInfo(_, _, _, longId, _, _, _, _), repr, _, _, _, _)) = typeDefn
        let typeName = AstHelpers.getQualifiedName longId
        
        match repr with
        | SynTypeDefnRepr.Simple(SynTypeDefnSimpleRepr.Union(_, cases, _), _) ->
            let oakCases = cases |> List.map mapUnionCase
            Some(TypeDecl(typeName, UnionType(oakCases)))
        | SynTypeDefnRepr.Simple(SynTypeDefnSimpleRepr.Record(_, fields, _), _) ->
            let oakFields = fields |> List.map mapRecordField
            Some(TypeDecl(typeName, StructType(oakFields)))
        | _ -> None
    
    let mapBinding (binding: SynBinding) : OakDeclaration option =
        let (SynBinding(_, _, _, _, attrs, _, _, headPat, _, expr, _, _, _)) = binding
        let name = AstHelpers.extractPatternName headPat
        
        if AstHelpers.hasEntryPointAttribute attrs then
            Some(EntryPoint(ExpressionMapping.mapExpression expr))
        else
            match expr with
            | SynExpr.Lambda(_, _, _, body, parsedData, _, _) ->
                let params' = 
                    match parsedData with
                    | Some(originalPats, _) ->
                        originalPats |> List.map (fun pat -> (AstHelpers.extractPatternName pat, UnitType))
                    | None -> [("x", UnitType)]
                Some(FunctionDecl(name, params', UnitType, ExpressionMapping.mapExpression body))
            | _ ->
                Some(FunctionDecl(name, [], UnitType, ExpressionMapping.mapExpression expr))
    
    let mapModuleDeclaration (decl: SynModuleDecl) : OakDeclaration list =
        match decl with
        | SynModuleDecl.Let(_, bindings, _) ->
            bindings |> List.choose mapBinding
        | SynModuleDecl.Types(typeDefns, _) ->
            typeDefns |> List.choose mapTypeDefinition
        | _ -> []

/// Module mapping functions
module ModuleMapping =
    
    let mapModule (mdl: SynModuleOrNamespace) : OakModule =
        let (SynModuleOrNamespace(ids, _, _, decls, _, _, _, _, _)) = mdl
        let moduleName = 
            if ids.IsEmpty then "Module"
            else AstHelpers.getQualifiedName ids
        let declarations = decls |> List.collect DeclarationMapping.mapModuleDeclaration
        { Name = moduleName; Declarations = declarations }
    
    let extractModulesFromParseTree (parseTree: ParsedInput option) : OakModule list =
        match parseTree with
        | Some(ParsedInput.ImplFile(implFile)) ->
            let (ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) = implFile
            modules |> List.map mapModule
        | Some(ParsedInput.SigFile(_)) -> []
        | None -> []

/// Main conversion functions
let parseAndConvertToOakAst (sourceCode: string) : OakProgram =
    try
        let sourceText = SourceText.ofString sourceCode
        let checker = FSharp.Compiler.CodeAnalysis.FSharpChecker.Create()
        let parsingOptions = FSharp.Compiler.CodeAnalysis.FSharpParsingOptions.Default
        let parseResults = checker.ParseFile("input.fs", sourceText, parsingOptions) |> Async.RunSynchronously
        
        let modules = ModuleMapping.extractModulesFromParseTree (Some parseResults.ParseTree)
        { Modules = modules }
    with _ ->
        { Modules = [] }

let parseAndConvertWithDiagnostics (sourceCode: string) : ASTConversionResult =
    try
        let sourceText = SourceText.ofString sourceCode
        let checker = FSharp.Compiler.CodeAnalysis.FSharpChecker.Create()
        let parsingOptions = FSharp.Compiler.CodeAnalysis.FSharpParsingOptions.Default
        let parseResults = checker.ParseFile("input.fs", sourceText, parsingOptions) |> Async.RunSynchronously
        
        let diagnostics = parseResults.Diagnostics |> Array.map (fun diag -> diag.Message) |> Array.toList
        
        try
            let modules = ModuleMapping.extractModulesFromParseTree (Some parseResults.ParseTree)
            { OakProgram = { Modules = modules }; Diagnostics = diagnostics }
        with moduleEx ->
            { OakProgram = { Modules = [] }; 
              Diagnostics = sprintf "Error during module extraction: %s" moduleEx.Message :: diagnostics }
    with ex ->
        { OakProgram = { Modules = [] }; Diagnostics = [sprintf "Exception: %s" ex.Message] }