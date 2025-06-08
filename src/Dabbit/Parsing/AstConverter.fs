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

/// Extracts the innermost function name from a potentially complex expression
let rec private extractFunctionName (expr: SynExpr) : string option =
    match expr with
    | SynExpr.Ident(ident) -> Some ident.idText
    | SynExpr.LongIdent(_, SynLongIdent(ids, _, _), _, _) ->
        // For qualified names like Printf.printf, take the last part
        ids |> List.tryLast |> Option.map (fun id -> id.idText)
    | SynExpr.TypeApp(funcExpr, _, _, _, _, _, _) ->
        // Handle generic function calls
        extractFunctionName funcExpr
    | SynExpr.App(_, _, funcExpr, _, _) ->
        // Recursively extract from nested applications
        extractFunctionName funcExpr
    | _ -> None

/// Collects all arguments from a curried function application
let rec private collectApplicationArgs (expr: SynExpr) : (SynExpr * SynExpr list) =
    match expr with
    | SynExpr.App(_, _, funcExpr, argExpr, _) ->
        let (baseFunc, existingArgs) = collectApplicationArgs funcExpr
        (baseFunc, existingArgs @ [argExpr])
    | _ -> (expr, [])

/// Recognizes F# I/O operations and converts to Oak I/O operations
let private recognizeIOOperation (funcExpr: SynExpr) (args: SynExpr list) : OakExpression option =
    let funcNameOpt = extractFunctionName funcExpr
    
    match funcNameOpt with
    | Some "printf" ->
        match args with
        | SynExpr.Const(SynConst.String(format, _, _), _) :: valueExprs ->
            let oakArgs = valueExprs |> List.map synExprToOakExpr
            Some (IOOperation(Printf(format), oakArgs))
        | _ -> None
    
    | Some "printfn" ->
        match args with
        | SynExpr.Const(SynConst.String(format, _, _), _) :: valueExprs ->
            let oakArgs = valueExprs |> List.map synExprToOakExpr
            Some (IOOperation(Printfn(format), oakArgs))
        | _ -> None
    
    | _ -> None

/// Converts F# Compiler Services SynExpr to Oak expression
and synExprToOakExpr (synExpr: SynExpr) : OakExpression =
    match synExpr with
    | SynExpr.Const(synConst, _) -> 
        Literal(synConstToOakLiteral synConst)
    
    | SynExpr.Ident(ident) -> 
        Variable(ident.idText)
    
    | SynExpr.LongIdent(_, SynLongIdent(id, _, _), _, _) ->
        let name = id |> List.map (fun i -> i.idText) |> String.concat "."
        Variable(name)
    
    | SynExpr.App(_, _, _, _, _) ->
        // Collect all arguments for curried applications
        let (baseFunc, allArgs) = collectApplicationArgs synExpr
        
        // First check if this is an I/O operation
        match recognizeIOOperation baseFunc allArgs with
        | Some ioOp -> ioOp
        | None ->
            // Handle as regular function application
            let func = synExprToOakExpr baseFunc
            let args = allArgs |> List.map synExprToOakExpr
            if args.IsEmpty then func
            else Application(func, args)
    
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
                | SynBinding(_, _, _, _, _, _, _, pat, _, rhsExpr, _, _, _) ->
                    match pat with
                    | SynPat.Named(SynIdent(ident, _), _, _, _) ->
                        let value = synExprToOakExpr rhsExpr
                        let innerBody = buildLetChain rest body
                        Let(ident.idText, value, innerBody)
                    | SynPat.Wild(_) ->
                        // Handle wildcard pattern (let _ = expr)
                        let value = synExprToOakExpr rhsExpr
                        let innerBody = buildLetChain rest body
                        Sequential(value, innerBody)
                    | _ -> buildLetChain rest body
        
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
        // Properly handle sequential expressions
        let rec flattenSequential expr acc =
            match expr with
            | SynExpr.Sequential(_, _, e1, e2, _) ->
                flattenSequential e2 (synExprToOakExpr e1 :: acc)
            | e -> List.rev (synExprToOakExpr e :: acc)
        
        let expressions = flattenSequential synExpr []
        
        // Build right-associative Sequential chain
        let rec buildSequential exprs =
            match exprs with
            | [] -> Literal(UnitLiteral)
            | [single] -> single
            | first :: rest -> Sequential(first, buildSequential rest)
        
        buildSequential expressions
    
    | SynExpr.DotGet(targetExpr, _, SynLongIdent(id, _, _), _) ->
        match targetExpr, id with
        | SynExpr.Ident(obj), [method] when obj.idText = "stdin" && method.idText = "ReadLine" ->
            // Special case for stdin.ReadLine()
            IOOperation(ReadLine, [])
        | _ ->
            let target = synExprToOakExpr targetExpr
            let fieldName = id |> List.map (fun i -> i.idText) |> String.concat "."
            FieldAccess(target, fieldName)
    
    | SynExpr.DotIndexedGet(targetExpr, indexExprs, _, _) ->
        let target = synExprToOakExpr targetExpr
        let indices = indexExprs |> List.map synExprToOakExpr
        MethodCall(target, "get_Item", indices)
    
    | SynExpr.ArrayOrListComputed(_, exprs, _) ->
        let elements = exprs |> List.map synExprToOakExpr
        Literal(ArrayLiteral(elements))
    
    | SynExpr.Paren(innerExpr, _, _, _) ->
        synExprToOakExpr innerExpr
    
    | SynExpr.TypeApp(expr, _, _, _, _, _, _) ->
        // Handle type applications by processing the inner expression
        synExprToOakExpr expr
    
    | _ -> 
        // For unsupported expressions, create a placeholder
        Literal(UnitLiteral)

/// Converts function arguments pattern to parameters
let private extractFunctionParameters (argPats: SynArgPats) : (string * OakType) list =
    match argPats with
    | SynArgPats.Pats(patterns) ->
        patterns |> List.choose (function
            | SynPat.Named(SynIdent(ident, _), _, _, _) -> 
                Some (ident.idText, UnitType)
            | SynPat.Typed(SynPat.Named(SynIdent(ident, _), _, _, _), synType, _) -> 
                Some (ident.idText, synTypeToOakType synType)
            | SynPat.Paren(SynPat.Typed(SynPat.Named(SynIdent(ident, _), _, _, _), synType, _), _) ->
                Some (ident.idText, synTypeToOakType synType)
            | _ -> None)
    | _ -> []

/// Converts F# Compiler Services SynBinding to Oak declaration
let private synBindingToOakDecl (binding: SynBinding) : OakDeclaration option =
    match binding with
    | SynBinding(_, _, _, _, attrs, _, _, pat, returnInfo, rhsExpr, _, _, _) ->
        // Check if this is an entry point
        let isEntryPoint = 
            attrs |> List.exists (fun attr ->
                match attr with
                | { TypeName = SynLongIdent(ids, _, _) } ->
                    ids |> List.exists (fun id -> id.idText = "EntryPoint")
            )
        
        match pat with
        | SynPat.LongIdent(SynLongIdent(id, _, _), _, _, argPats, _, _) ->
            let name = id |> List.map (fun i -> i.idText) |> String.concat "."
            
            if isEntryPoint then
                // Handle entry point - convert the body but wrap it properly
                let body = synExprToOakExpr rhsExpr
                Some (EntryPoint(body))
            else
                // Regular function declaration
                let parameters = extractFunctionParameters argPats
                let returnType = 
                    match returnInfo with
                    | Some (SynBindingReturnInfo(synType, _, _, _)) -> synTypeToOakType synType
                    | None -> UnitType
                
                let body = synExprToOakExpr rhsExpr
                Some (FunctionDecl(name, parameters, returnType, body))
        
        | SynPat.Named(SynIdent(ident, _), _, _, _) ->
            // Simple binding (no parameters)
            let name = ident.idText
            let body = synExprToOakExpr rhsExpr
            
            if isEntryPoint then
                Some (EntryPoint(body))
            else
                Some (FunctionDecl(name, [], UnitType, body))
        
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
                                    match fields.[0] with
                                    | SynField(_, _, _, synType, _, _, _, _) ->
                                        (caseName, Some(synTypeToOakType synType))
                                | _ -> (caseName, None))
                    Some (TypeDecl(name, UnionType(cases)))
                | _ -> None
            | _ -> None)
    
    | SynModuleDecl.Expr(expr, _) ->
        // Top-level expression - could be part of script
        []
    
    | _ -> []

/// Checks if an expression contains I/O operations
let rec private containsIOOperations (expr: OakExpression) : bool =
    match expr with
    | IOOperation(_, _) -> true
    | Application(func, args) -> containsIOOperations func || List.exists containsIOOperations args
    | Let(_, value, body) -> containsIOOperations value || containsIOOperations body
    | IfThenElse(cond, thenExpr, elseExpr) -> 
        containsIOOperations cond || containsIOOperations thenExpr || containsIOOperations elseExpr
    | Sequential(first, second) -> containsIOOperations first || containsIOOperations second
    | FieldAccess(target, _) -> containsIOOperations target
    | MethodCall(target, _, args) -> containsIOOperations target || List.exists containsIOOperations args
    | Lambda(_, body) -> containsIOOperations body
    | _ -> false

/// Adds external declarations for standard I/O functions
let private addStandardIODeclarations (declarations: OakDeclaration list) : OakDeclaration list =
    let hasIO = 
        declarations 
        |> List.exists (function
            | FunctionDecl(_, _, _, body) -> containsIOOperations body
            | EntryPoint(expr) -> containsIOOperations expr
            | _ -> false)
    
    if hasIO then
        // For Windows command-line linking, we need the correct signatures
        let externalDecls = [
            ExternalDecl("printf", [StringType], IntType, "msvcrt")
            ExternalDecl("scanf", [StringType], IntType, "msvcrt")
            ExternalDecl("gets", [StringType], StringType, "msvcrt")
            ExternalDecl("puts", [StringType], IntType, "msvcrt")
            ExternalDecl("getchar", [], IntType, "msvcrt")
        ]
        externalDecls @ declarations
    else
        declarations

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
                        
                        // Add external declarations if needed
                        let declarationsWithExternals = addStandardIODeclarations declarations
                        
                        { Name = moduleName; Declarations = declarationsWithExternals })
            
            { Modules = oakModules }
        
        | ParsedInput.SigFile(_) ->
            // Signature files not supported yet
            { Modules = [{ Name = "Main"; Declarations = [EntryPoint(Literal(IntLiteral(0)))] }] }
    
    with
    | ex ->
        // On parse error, create a minimal program
        printfn "Parse error: %s" ex.Message
        { Modules = [{ Name = "Main"; Declarations = [EntryPoint(Literal(IntLiteral(0)))] }] }