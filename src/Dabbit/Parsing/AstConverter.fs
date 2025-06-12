module Dabbit.Parsing.AstConverter

open System
open System.IO
open FSharp.Compiler.Syntax
open FSharp.Compiler.Text
open Dabbit.Parsing.OakAst

/// Result type for F# to Oak AST conversion with diagnostics and F# AST
type ASTConversionResult = {
    OakProgram: OakProgram
    Diagnostics: string list
    FSharpASTText: string  // Add F# AST text representation
}

/// Helper functions for extracting names and types from F# AST nodes
module AstHelpers =
    
    let getIdentifierName (ident: Ident) : string = ident.idText
    
    let getQualifiedName (idents: Ident list) : string =
        match idents with
        | [] -> "_empty_"
        | _ -> idents |> List.map (fun id -> id.idText) |> String.concat "."
    
    let extractLongIdent (synLongIdent: SynLongIdent) : Ident list =
        let (SynLongIdent(idents, _, _)) = synLongIdent
        idents
    
    let extractIdent (synIdent: SynIdent) : Ident =
        let (SynIdent(ident, _)) = synIdent
        ident
    
    let extractPatternName (pattern: SynPat) : string =
        match pattern with
        | SynPat.Named(synIdent, _, _, _) -> 
            try
                synIdent |> extractIdent |> getIdentifierName
            with
            | _ -> "_pattern_"
        | SynPat.LongIdent(longDotId, _, _, _, _, _) ->
            try
                let idents = longDotId |> extractLongIdent
                if idents.IsEmpty then "_empty_pattern_" else getQualifiedName idents
            with
            | _ -> "_pattern_"
        | SynPat.Const(SynConst.Unit, _) -> "()"
        | _ -> "_"
    
    let hasEntryPointAttribute (attributes: SynAttributes) : bool =
        try
            attributes
            |> List.exists (fun attrList ->
                if attrList.Attributes.IsEmpty then false
                else
                    attrList.Attributes 
                    |> List.exists (fun attr ->
                        match attr.TypeName with
                        | SynLongIdent(idents, _, _) -> 
                            if idents.IsEmpty then false
                            else idents |> List.exists (fun id -> 
                                id.idText.Contains("EntryPoint"))))
        with
        | _ -> false

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
        try
            match expr with
            | SynExpr.Const(constant, _) ->
                constant |> TypeMapping.mapLiteral |> Literal
            
            | SynExpr.Ident(ident) ->
                ident |> AstHelpers.getIdentifierName |> Variable
            
            | SynExpr.LongIdent(_, longIdent, _, _) ->
                try
                    let idents = longIdent |> AstHelpers.extractLongIdent
                    let qualifiedName = AstHelpers.getQualifiedName idents
                    Variable qualifiedName
                with
                | _ -> Variable "_unknown_"
            
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
        with
        | _ -> Literal(UnitLiteral)
    
    and mapFunctionApplication funcExpr argExpr =
        try
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
        with
        | _ -> Literal(UnitLiteral)
    
    and mapLetBinding bindings bodyExpr =
        try
            match bindings with
            | binding :: _ ->
                let (SynBinding(_, _, _, _, _, _, _, headPat, _, expr, _, _, _)) = binding
                let name = AstHelpers.extractPatternName headPat
                Let(name, mapExpression expr, mapExpression bodyExpr)
            | [] -> mapExpression bodyExpr
        with
        | _ -> mapExpression bodyExpr
    
    and mapConditional condExpr thenExpr elseExprOpt =
        try
            let cond = mapExpression condExpr
            let thenBranch = mapExpression thenExpr
            let elseBranch = 
                match elseExprOpt with
                | Some(elseExpr) -> mapExpression elseExpr
                | None -> Literal(UnitLiteral)
            IfThenElse(cond, thenBranch, elseBranch)
        with
        | _ -> Literal(UnitLiteral)
    
    and mapLambda parsedData body =
        try
            let params' = 
                match parsedData with
                | Some(originalPats, _) ->
                    if originalPats.IsEmpty then [("x", UnitType)]
                    else originalPats |> List.map (fun pat -> (AstHelpers.extractPatternName pat, UnitType))
                | None -> [("x", UnitType)]
            Lambda(params', mapExpression body)
        with
        | _ -> Lambda([("x", UnitType)], Literal(UnitLiteral))

/// Declaration mapping functions  
module DeclarationMapping =
    
    let mapUnionCase (case: SynUnionCase) : string * OakType option =
        try
            let (SynUnionCase(_, synIdent, caseType, _, _, _, _)) = case
            let ident = AstHelpers.extractIdent synIdent
            let caseName = AstHelpers.getIdentifierName ident
            
            match caseType with
            | SynUnionCaseKind.Fields fields ->
                if fields.IsEmpty then (caseName, None)
                else (caseName, Some UnitType)
            | _ -> (caseName, None)
        with
        | _ -> ("_case_", None)
    
    let mapRecordField (field: SynField) : string * OakType =
        try
            let (SynField(_, _, fieldId, _, _, _, _, _, _)) = field
            let fieldName = 
                match fieldId with
                | Some ident -> AstHelpers.getIdentifierName ident
                | None -> "_"
            (fieldName, UnitType)
        with
        | _ -> ("_field_", UnitType)
    
    let mapTypeDefinition (typeDefn: SynTypeDefn) : OakDeclaration option =
        try
            let (SynTypeDefn(SynComponentInfo(_, _, _, longId, _, _, _, _), repr, _, _, _, _)) = typeDefn
            let idents = longId
            let typeName = if idents.IsEmpty then "_type_" else AstHelpers.getQualifiedName idents
            
            match repr with
            | SynTypeDefnRepr.Simple(SynTypeDefnSimpleRepr.Union(_, cases, _), _) ->
                let oakCases = cases |> List.map mapUnionCase
                Some(TypeDecl(typeName, UnionType(oakCases)))
            | SynTypeDefnRepr.Simple(SynTypeDefnSimpleRepr.Record(_, fields, _), _) ->
                let oakFields = fields |> List.map mapRecordField
                Some(TypeDecl(typeName, StructType(oakFields)))
            | _ -> None
        with
        | _ -> None
    
    let mapBinding (binding: SynBinding) : OakDeclaration option =
        try
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
                            if originalPats.IsEmpty then [("x", UnitType)]
                            else originalPats |> List.map (fun pat -> (AstHelpers.extractPatternName pat, UnitType))
                        | None -> [("x", UnitType)]
                    Some(FunctionDecl(name, params', UnitType, ExpressionMapping.mapExpression body))
                | _ ->
                    Some(FunctionDecl(name, [], UnitType, ExpressionMapping.mapExpression expr))
        with
        | _ -> None
    
    let mapModuleDeclaration (decl: SynModuleDecl) : OakDeclaration list =
        try
            match decl with
            | SynModuleDecl.Let(_, bindings, _) ->
                if bindings.IsEmpty then []
                else bindings |> List.choose mapBinding
            | SynModuleDecl.Types(typeDefns, _) ->
                if typeDefns.IsEmpty then []
                else typeDefns |> List.choose mapTypeDefinition
            | _ -> []
        with
        | _ -> []

/// Module mapping functions
module ModuleMapping =
    
    let mapModule (mdl: SynModuleOrNamespace) : OakModule =
        try
            let (SynModuleOrNamespace(ids, _, _, decls, _, _, _, _, _)) = mdl
            let moduleName = 
                if ids.IsEmpty then "Module"
                else AstHelpers.getQualifiedName ids
            let declarations = 
                if decls.IsEmpty then []
                else decls |> List.collect DeclarationMapping.mapModuleDeclaration
            { Name = moduleName; Declarations = declarations }
        with
        | _ -> { Name = "_module_"; Declarations = [] }
    
    let extractModulesFromParseTree (parseTree: ParsedInput option) : OakModule list =
        try
            match parseTree with
            | Some(ParsedInput.ImplFile(implFile)) ->
                try
                    let (ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) = implFile
                    if modules.IsEmpty then []
                    else modules |> List.map mapModule
                with
                | _ -> []
            | Some(ParsedInput.SigFile(_)) -> []
            | None -> []
        with
        | _ -> []

/// Enhanced unified conversion function including F# AST capture
let parseAndConvertToOakAst (inputPath: string) (sourceCode: string) : ASTConversionResult =
    try
        let sourceText = SourceText.ofString sourceCode
        
        let checker = FSharp.Compiler.CodeAnalysis.FSharpChecker.Create(keepAssemblyContents = true)
        let parsingOptions = { 
            FSharp.Compiler.CodeAnalysis.FSharpParsingOptions.Default with
                SourceFiles = [|Path.GetFileName(inputPath)|]
                ConditionalDefines = []
                ApplyLineDirectives = false
        }
        
        let parseResults = checker.ParseFile(inputPath, sourceText, parsingOptions) |> Async.RunSynchronously
        
        // Capture F# AST as text
        let fsharpASTText = sprintf "F# AST for %s:\n%A" (Path.GetFileName(inputPath)) parseResults.ParseTree
        
        // Process diagnostics quietly
        let diagnostics = 
            if parseResults.Diagnostics.Length = 0 then []
            else parseResults.Diagnostics |> Array.map (fun diag -> diag.Message) |> Array.toList
        
        // Extract modules and create Oak AST
        let modules = ModuleMapping.extractModulesFromParseTree (Some parseResults.ParseTree)
        let oakProgram = { Modules = modules }
        
        { OakProgram = oakProgram; Diagnostics = diagnostics; FSharpASTText = fsharpASTText }
        
    with parseEx ->
        let parseError = sprintf "F# parsing failed: %s" parseEx.Message
        { OakProgram = { Modules = [] }; Diagnostics = [parseError]; FSharpASTText = parseError }