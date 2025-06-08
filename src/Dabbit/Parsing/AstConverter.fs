module Dabbit.Parsing.AstConverter

open System
open System.IO
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Syntax
open FSharp.Compiler.Text
open Dabbit.Parsing.OakAst

/// Creates a checker instance for parsing F# code
let private createChecker() =
    FSharpChecker.Create(keepAssemblyContents=true)

/// Converts F# Compiler Services SynType to Oak type
let rec private synTypeToOakType (synType: SynType) : OakType =
    match synType with
    | SynType.LongIdent(SynLongIdent(id, _, _)) ->
        match id with
        | [ident] when ident.idText = "int" -> IntType
        | [ident] when ident.idText = "float" -> FloatType
        | [ident] when ident.idText = "bool" -> BoolType
        | [ident] when ident.idText = "string" -> StringType
        | [ident] when ident.idText = "unit" -> UnitType
        | _ -> UnitType // Default fallback
    | SynType.App(typeName, _, typeArgs, _, _, _, _) ->
        match typeArgs with
        | [elemType] -> ArrayType(synTypeToOakType elemType)
        | _ -> UnitType
    | SynType.Fun(argType, returnType, _, _) ->
        FunctionType([synTypeToOakType argType], synTypeToOakType returnType)
    | SynType.Tuple(_, elementTypes, _) ->
        let fieldTypes = elementTypes |> List.mapi (fun i elemType -> (sprintf "Item%d" (i+1), synTypeToOakType elemType))
        StructType fieldTypes
    | _ -> UnitType

/// Converts F# Compiler Services SynConst to Oak literal
let private synConstToOakLiteral (synConst: SynConst) : OakLiteral =
    match synConst with
    | SynConst.Int32(value) -> IntLiteral(value)
    | SynConst.Double(value) -> FloatLiteral(value)
    | SynConst.String(text, _, _) -> StringLiteral(text)
    | SynConst.Bool(value) -> BoolLiteral(value)
    | SynConst.Unit -> UnitLiteral
    | _ -> UnitLiteral

/// Converts F# Compiler Services SynExpr to Oak expression
let rec private synExprToOakExpr (synExpr: SynExpr) : OakExpression =
    match synExpr with
    | SynExpr.Const(synConst, _) -> 
        Literal(synConstToOakLiteral synConst)
    
    | SynExpr.Ident(ident) -> 
        Variable(ident.idText)
    
    | SynExpr.LongIdent(_, SynLongIdent(id, _, _), _, _) ->
        let name = id |> List.map (fun i -> i.idText) |> String.concat "."
        Variable(name)
    
    | SynExpr.App(_, _, funcExpr, argExpr, _) ->
        let func = synExprToOakExpr funcExpr
        let arg = synExprToOakExpr argExpr
        Application(func, [arg])
    
    | SynExpr.Lambda(_, _, args, bodyExpr, _, _, _) ->
        let parameters = 
            match args with
            | SynSimplePats.SimplePats(patterns, _) ->
                patterns |> List.choose (function
                    | SynSimplePat.Id(ident, _, _, _, _, _) -> Some (ident.idText, UnitType)
                    | SynSimplePat.Typed(SynSimplePat.Id(ident, _, _, _, _, _), synType, _) -> 
                        Some (ident.idText, synTypeToOakType synType)
                    | _ -> None)
            | _ -> []
        
        let body = synExprToOakExpr bodyExpr
        Lambda(parameters, body)
    
    | SynExpr.LetOrUse(_, _, bindings, bodyExpr, _, _) ->
        // Handle let bindings
        let rec buildLetChain bindings body =
            match bindings with
            | [] -> synExprToOakExpr body
            | binding :: rest ->
                match binding with
                | SynBinding(_, _, _, _, _, _, _, SynPat.Named(SynIdent(ident, _), _, _, _), _, rhsExpr, _, _, _) ->
                    let value = synExprToOakExpr rhsExpr
                    let innerBody = buildLetChain rest body
                    Let(ident.idText, value, innerBody)
                | _ -> synExprToOakExpr body
        
        buildLetChain bindings bodyExpr
    
    | SynExpr.IfThenElse(condExpr, thenExpr, elseExprOpt, _, _, _, _) ->
        let cond = synExprToOakExpr condExpr
        let thenBranch = synExprToOakExpr thenExpr
        let elseBranch = 
            match elseExprOpt with
            | Some elseExpr -> synExprToOakExpr elseExpr
            | None -> Literal(UnitLiteral)
        IfThenElse(cond, thenBranch, elseBranch)
    
    | SynExpr.Sequential(_, _, expr1, expr2, _) ->
        let first = synExprToOakExpr expr1
        let second = synExprToOakExpr expr2
        Sequential(first, second)
    
    | SynExpr.DotGet(targetExpr, _, SynLongIdent(id, _, _), _) ->
        let target = synExprToOakExpr targetExpr
        let fieldName = id |> List.map (fun i -> i.idText) |> String.concat "."
        FieldAccess(target, fieldName)
    
    | SynExpr.DotIndexedGet(targetExpr, indexExprs, _, _) ->
        let target = synExprToOakExpr targetExpr
        let indices = indexExprs |> List.map synExprToOakExpr
        // Convert array access to method call for now
        MethodCall(target, "get_Item", indices)
    
    | SynExpr.ArrayOrListComputed(_, exprs, _) ->
        let elements = exprs |> List.map synExprToOakExpr
        Literal(ArrayLiteral(elements))
    
    | SynExpr.Paren(innerExpr, _, _, _) ->
        synExprToOakExpr innerExpr
    
    | _ -> 
        // For unsupported expressions, create a placeholder
        Literal(UnitLiteral)

/// Converts F# Compiler Services SynBinding to Oak declaration
let private synBindingToOakDecl (binding: SynBinding) : OakDeclaration option =
    match binding with
    | SynBinding(_, _, _, _, _, _, _, SynPat.Named(SynIdent(ident, _), _, _, _), returnInfo, rhsExpr, _, _, _) ->
        let name = ident.idText
        let body = synExprToOakExpr rhsExpr
        
        // Determine if this is a function or value binding
        match rhsExpr with
        | SynExpr.Lambda(_, _, args, lambdaBody, _, _, _) ->
            // This is a function
            let parameters = 
                match args with
                | SynSimplePats.SimplePats(patterns, _) ->
                    patterns |> List.choose (function
                        | SynSimplePat.Id(ident, _, _, _, _, _) -> Some (ident.idText, UnitType)
                        | SynSimplePat.Typed(SynSimplePat.Id(ident, _, _, _, _, _), synType, _) -> 
                            Some (ident.idText, synTypeToOakType synType)
                        | _ -> None)
                | _ -> []
            
            let returnType = 
                match returnInfo with
                | Some (SynBindingReturnInfo(synType, _, _, _)) -> synTypeToOakType synType
                | None -> UnitType
            
            let functionBody = synExprToOakExpr lambdaBody
            Some (FunctionDecl(name, parameters, returnType, functionBody))
        
        | _ when name = "main" ->
            // This could be an entry point
            Some (EntryPoint(body))
        
        | _ ->
            // This is a simple function with no parameters
            Some (FunctionDecl(name, [], UnitType, body))
    
    | SynBinding(_, _, _, _, _, _, _, SynPat.LongIdent(SynLongIdent(id, _, _), _, _, SynArgPats.Pats(patterns), _, _), returnInfo, rhsExpr, _, _, _) ->
        // Function with parameters
        let name = id |> List.map (fun i -> i.idText) |> String.concat "."
        let parameters = 
            patterns |> List.choose (function
                | SynPat.Named(SynIdent(ident, _), _, _, _) -> Some (ident.idText, UnitType)
                | SynPat.Typed(SynPat.Named(SynIdent(ident, _), _, _, _), synType, _) -> 
                    Some (ident.idText, synTypeToOakType synType)
                | _ -> None)
        
        let returnType = 
            match returnInfo with
            | Some (SynBindingReturnInfo(synType, _, _, _)) -> synTypeToOakType synType
            | None -> UnitType
        
        let body = synExprToOakExpr rhsExpr
        Some (FunctionDecl(name, parameters, returnType, body))
    
    | _ -> None

/// Converts F# Compiler Services SynModuleDecl to Oak declarations
let private synModuleDeclToOakDecls (moduleDecl: SynModuleDecl) : OakDeclaration list =
    match moduleDecl with
    | SynModuleDecl.Let(_, bindings, _) ->
        bindings |> List.choose synBindingToOakDecl
    
    | SynModuleDecl.Types(typeDefns, _) ->
        typeDefns |> List.choose (function
            | SynTypeDefn(SynComponentInfo(_, _, _, [ident], _, _, _, _), synTypeDefnRepr, _, _, _, _) ->
                let name = ident.idText
                match synTypeDefnRepr with
                | SynTypeDefnRepr.Simple(SynTypeDefnSimpleRepr.Union(_, unionCases, _), _) ->
                    let cases = 
                        unionCases |> List.map (function
                            | SynUnionCase(_, SynIdent(caseIdent, _), caseType, _, _, _, _) ->
                                let caseName = caseIdent.idText
                                match caseType with
                                | SynUnionCaseKind.Fields(fields) when not fields.IsEmpty ->
                                    // For simplicity, take the first field type
                                    match fields.[0] with
                                    | SynField(_, _, _, synType, _, _, _, _) ->
                                        (caseName, Some(synTypeToOakType synType))
                                | _ -> (caseName, None))
                    Some (TypeDecl(name, UnionType(cases)))
                | _ -> None
            | _ -> None)
    
    | SynModuleDecl.Expr(expr, _) ->
        // Top-level expression - treat as entry point
        [EntryPoint(synExprToOakExpr expr)]
    
    | _ -> []

/// Parses F# source code using F# Compiler Services and converts to Oak AST
let parseAndConvertToOakAst (sourceCode: string) : OakProgram =
    try
        let checker = createChecker()
        let fileName = "input.fs"
        let sourceText = SourceText.ofString sourceCode
        
        // Parse the source code
        let parseFileResults = 
            checker.ParseFile(fileName, sourceText, FSharpParsingOptions.Default)
            |> Async.RunSynchronously
        
        match parseFileResults.ParseTree with
        | ParsedInput.ImplFile(ParsedImplFileInput(fileName, isScript, qualifiedNameOfFile, scopedPragmas, hashDirectives, modules, _)) ->
            let oakModules = 
                modules |> List.map (function
                    | SynModuleOrNamespace(longId, _, _, moduleDecls, _, _, _, _, _) ->
                        let moduleName = 
                            if longId.IsEmpty then "Main" 
                            else longId |> List.map (fun i -> i.idText) |> String.concat "."
                        
                        let declarations = moduleDecls |> List.collect synModuleDeclToOakDecls
                        
                        // If we have declarations but no explicit entry point, and there's a function call at module level,
                        // create an entry point that calls the first function
                        let finalDeclarations = 
                            if declarations |> List.exists (function | EntryPoint(_) -> true | _ -> false) then
                                declarations
                            else
                                // Look for functions and create a simple entry point
                                let functionNames = 
                                    declarations |> List.choose (function 
                                        | FunctionDecl(name, [], _, _) when name <> "main" -> Some name 
                                        | _ -> None)
                                
                                match functionNames with
                                | firstFunc :: _ ->
                                    declarations @ [EntryPoint(Application(Variable(firstFunc), []))]
                                | [] ->
                                    if declarations.IsEmpty then
                                        [EntryPoint(Literal(IntLiteral(0)))]
                                    else
                                        declarations
                        
                        { Name = moduleName; Declarations = finalDeclarations })
            
            { Modules = oakModules }
        
        | ParsedInput.SigFile(_) ->
            // Signature files not supported yet
            { Modules = [{ Name = "Main"; Declarations = [EntryPoint(Literal(IntLiteral(0)))] }] }
    
    with
    | ex ->
        // On parse error, create a minimal program
        printfn "Parse error: %s" ex.Message
        { Modules = [{ Name = "Main"; Declarations = [EntryPoint(Literal(IntLiteral(0)))] }] }