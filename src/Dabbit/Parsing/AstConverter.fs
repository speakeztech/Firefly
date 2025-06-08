module Dabbit.Parsing.AstConverter

open System
open System.IO
open System.Text
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Syntax
open FSharp.Compiler.Text
open Dabbit.Parsing.OakAst

/// Represents the parsing result with intermediate representations
type ParsingResult = {
    OakProgram: OakProgram
    FCSAstText: string
    OakAstText: string
}

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

/// Enhanced function name extraction with debugging
let rec private extractFunctionName (expr: SynExpr) : string option =
    match expr with
    | SynExpr.Ident(ident) -> 
        let name = ident.idText
        printfn "Debug: Extracted function name: %s" name
        Some name
    | SynExpr.LongIdent(_, SynLongIdent(ids, _, _), _, _) ->
        let qualifiedName = ids |> List.map (fun id -> id.idText) |> String.concat "."
        let name = ids |> List.tryLast |> Option.map (fun id -> id.idText)
        printfn "Debug: Extracted qualified function name: %s -> %s" qualifiedName (name |> Option.defaultValue "None")
        name
    | SynExpr.TypeApp(funcExpr, _, _, _, _, _, _) ->
        printfn "Debug: Processing type application for function name"
        extractFunctionName funcExpr
    | SynExpr.App(_, _, funcExpr, _, _) ->
        printfn "Debug: Processing application for function name"
        extractFunctionName funcExpr
    | _ -> 
        printfn "Debug: Could not extract function name from expression type: %s" (expr.GetType().Name)
        None

/// Enhanced application argument collection with debugging
let rec private collectApplicationArgs (expr: SynExpr) : (SynExpr * SynExpr list) =
    match expr with
    | SynExpr.App(_, _, funcExpr, argExpr, _) ->
        let (baseFunc, existingArgs) = collectApplicationArgs funcExpr
        printfn "Debug: Collected argument in application chain"
        (baseFunc, existingArgs @ [argExpr])
    | _ -> 
        printfn "Debug: Found base function expression"
        (expr, [])

/// Enhanced I/O operation recognition with comprehensive pattern matching
let private recognizeIOOperation (funcExpr: SynExpr) (args: SynExpr list) : OakExpression option =
    let funcNameOpt = extractFunctionName funcExpr
    
    printfn "Debug: Attempting to recognize I/O operation for function: %s with %d args" 
            (funcNameOpt |> Option.defaultValue "None") args.Length
    
    match funcNameOpt with
    | Some "printf" ->
        printfn "Debug: Recognized printf function call"
        match args with
        | SynExpr.Const(SynConst.String(format, _, _), _) :: valueExprs ->
            printfn "Debug: Printf with string literal format: %s" format
            let oakArgs = valueExprs |> List.map synExprToOakExpr
            Some (IOOperation(Printf(format), oakArgs))
        | formatExpr :: valueExprs ->
            printfn "Debug: Printf with dynamic format string"
            // For dynamic format strings, convert the format expression and use a generic format
            let formatArg = synExprToOakExpr formatExpr
            let oakArgs = formatArg :: (valueExprs |> List.map synExprToOakExpr)
            Some (IOOperation(Printf("%s"), oakArgs))
        | [] ->
            printfn "Debug: Printf with no arguments - invalid"
            None
        | _ -> 
            printfn "Debug: Printf pattern not matched"
            None
    
    | Some "printfn" ->
        printfn "Debug: Recognized printfn function call"
        match args with
        | SynExpr.Const(SynConst.String(format, _, _), _) :: valueExprs ->
            printfn "Debug: Printfn with string literal format: %s" format
            let oakArgs = valueExprs |> List.map synExprToOakExpr
            Some (IOOperation(Printfn(format), oakArgs))
        | formatExpr :: valueExprs ->
            printfn "Debug: Printfn with dynamic format string"
            let formatArg = synExprToOakExpr formatExpr
            let oakArgs = formatArg :: (valueExprs |> List.map synExprToOakExpr)
            Some (IOOperation(Printfn("%s"), oakArgs))
        | [] ->
            printfn "Debug: Printfn with no arguments - invalid"
            None
        | _ ->
            printfn "Debug: Printfn pattern not matched"
            None
    
    | Some "scanf" ->
        printfn "Debug: Recognized scanf function call"
        match args with
        | SynExpr.Const(SynConst.String(format, _, _), _) :: valueExprs ->
            let oakArgs = valueExprs |> List.map synExprToOakExpr
            Some (IOOperation(Scanf(format), oakArgs))
        | _ -> None
    
    | Some name ->
        printfn "Debug: Function '%s' not recognized as I/O operation" name
        None
    
    | None ->
        printfn "Debug: Could not extract function name for I/O recognition"
        None

/// Enhanced expression conversion with comprehensive I/O support
let rec synExprToOakExpr (synExpr: SynExpr) : OakExpression =
    match synExpr with
    | SynExpr.Const(synConst, _) -> 
        Literal(synConstToOakLiteral synConst)
    
    | SynExpr.Ident(ident) -> 
        Variable(ident.idText)
    
    | SynExpr.LongIdent(_, SynLongIdent(id, _, _), _, _) ->
        let name = id |> List.map (fun i -> i.idText) |> String.concat "."
        Variable(name)
    
    | SynExpr.App(_, _, _, _, _) ->
        printfn "Debug: Processing function application"
        let (baseFunc, allArgs) = collectApplicationArgs synExpr
        
        printfn "Debug: Function application with %d arguments" allArgs.Length
        
        // Enhanced I/O operation detection
        match recognizeIOOperation baseFunc allArgs with
        | Some ioOp -> 
            printfn "Debug: Converted to I/O operation: %A" ioOp
            ioOp
        | None ->
            printfn "Debug: Processing as regular function application"
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
        let rec flattenSequential expr acc =
            match expr with
            | SynExpr.Sequential(_, _, e1, e2, _) ->
                flattenSequential e2 (synExprToOakExpr e1 :: acc)
            | e -> List.rev (synExprToOakExpr e :: acc)
        
        let expressions = flattenSequential synExpr []
        
        let rec buildSequential exprs =
            match exprs with
            | [] -> Literal(UnitLiteral)
            | [single] -> single
            | first :: rest -> Sequential(first, buildSequential rest)
        
        buildSequential expressions
    
    | SynExpr.DotGet(targetExpr, _, SynLongIdent(id, _, _), _) ->
        printfn "Debug: Processing property/method access: %A" id
        match targetExpr, id with
        | SynExpr.Ident(obj), [method] when obj.idText = "stdin" && method.idText = "ReadLine" ->
            printfn "Debug: Recognized stdin.ReadLine() as I/O operation"
            IOOperation(ReadLine, [])
        | SynExpr.Ident(obj), [method] when obj.idText = "Console" && method.idText = "ReadLine" ->
            printfn "Debug: Recognized Console.ReadLine() as I/O operation"
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
        synExprToOakExpr expr
    
    | SynExpr.Match(_, clauses, _) ->
        // Handle pattern matching - simplified conversion
        // For now, convert to nested if-then-else
        match clauses with
        | [] -> Literal(UnitLiteral)
        | [clause] ->
            match clause with
            | SynMatchClause(_, _, _, resultExpr, _, _, _) -> synExprToOakExpr resultExpr
        | _ ->
            // Multiple clauses - would need more sophisticated handling
            Literal(UnitLiteral)
    
    | _ -> 
        printfn "Debug: Unsupported expression type: %s, creating unit literal" (synExpr.GetType().Name)
        Literal(UnitLiteral)

/// Enhanced function parameter extraction with debugging
let private extractFunctionParameters (argPats: SynArgPats) : (string * OakType) list =
    match argPats with
    | SynArgPats.Pats(patterns) ->
        printfn "Debug: Extracting parameters from %d patterns" patterns.Length
        patterns |> List.choose (function
            | SynPat.Named(SynIdent(ident, _), _, _, _) -> 
                printfn "Debug: Found parameter: %s" ident.idText
                Some (ident.idText, UnitType)
            | SynPat.Typed(SynPat.Named(SynIdent(ident, _), _, _, _), synType, _) -> 
                let oakType = synTypeToOakType synType
                printfn "Debug: Found typed parameter: %s : %A" ident.idText oakType
                Some (ident.idText, oakType)
            | SynPat.Paren(SynPat.Typed(SynPat.Named(SynIdent(ident, _), _, _, _), synType, _), _) ->
                let oakType = synTypeToOakType synType
                printfn "Debug: Found parenthesized typed parameter: %s : %A" ident.idText oakType
                Some (ident.idText, oakType)
            | _ -> 
                printfn "Debug: Unsupported parameter pattern"
                None)
    | _ -> 
        printfn "Debug: No patterns found in function arguments"
        []

/// Enhanced binding conversion with comprehensive entry point detection
let private synBindingToOakDecl (binding: SynBinding) : OakDeclaration option =
    match binding with
    | SynBinding(_, _, _, _, attrs, _, _, pat, returnInfo, rhsExpr, _, _, _) ->
        // Enhanced entry point detection
        let isEntryPoint = 
            attrs |> List.exists (fun attr ->
                match attr with
                | { TypeName = SynLongIdent(ids, _, _) } ->
                    ids |> List.exists (fun id -> id.idText = "EntryPoint")
            )
        
        printfn "Debug: Processing binding, isEntryPoint: %b" isEntryPoint
        
        match pat with
        | SynPat.LongIdent(SynLongIdent(id, _, _), _, _, argPats, _, _) ->
            let name = id |> List.map (fun i -> i.idText) |> String.concat "."
            printfn "Debug: Processing function declaration: %s" name
            
            if isEntryPoint then
                printfn "Debug: Converting entry point function to EntryPoint declaration"
                let body = synExprToOakExpr rhsExpr
                Some (EntryPoint(body))
            else
                let parameters = extractFunctionParameters argPats
                let returnType = 
                    match returnInfo with
                    | Some (SynBindingReturnInfo(synType, _, _, _)) -> synTypeToOakType synType
                    | None -> IntType  // Default to int for functions
                
                printfn "Debug: Function %s has %d parameters, return type: %A" name parameters.Length returnType
                let body = synExprToOakExpr rhsExpr
                Some (FunctionDecl(name, parameters, returnType, body))
        
        | SynPat.Named(SynIdent(ident, _), _, _, _) ->
            let name = ident.idText
            printfn "Debug: Processing simple binding: %s" name
            let body = synExprToOakExpr rhsExpr
            
            if isEntryPoint then
                printfn "Debug: Converting simple entry point to EntryPoint declaration"
                Some (EntryPoint(body))
            else
                printfn "Debug: Converting simple binding to function declaration"
                Some (FunctionDecl(name, [], UnitType, body))
        
        | _ -> 
            printfn "Debug: Unsupported pattern in binding"
            None

/// Enhanced module declaration conversion
let private synModuleDeclToOakDecls (moduleDecl: SynModuleDecl) : OakDeclaration list =
    match moduleDecl with
    | SynModuleDecl.Let(_, bindings, _) ->
        printfn "Debug: Processing %d let bindings" bindings.Length
        bindings |> List.choose synBindingToOakDecl
    
    | SynModuleDecl.Types(typeDefns, _) ->
        printfn "Debug: Processing %d type definitions" typeDefns.Length
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
        printfn "Debug: Processing top-level expression"
        []
    
    | _ -> 
        printfn "Debug: Unsupported module declaration type"
        []

/// Enhanced I/O operation detection in expressions
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

/// Enhanced external declaration generation for I/O operations
let private addStandardIODeclarations (declarations: OakDeclaration list) : OakDeclaration list =
    let hasIO = 
        declarations 
        |> List.exists (function
            | FunctionDecl(_, _, _, body) -> containsIOOperations body
            | EntryPoint(expr) -> containsIOOperations expr
            | _ -> false)
    
    printfn "Debug: Module contains I/O operations: %b" hasIO
    
    if hasIO then
        let externalDecls = [
            ExternalDecl("printf", [StringType], IntType, "msvcrt")
            ExternalDecl("scanf", [StringType], IntType, "msvcrt")
            ExternalDecl("gets", [StringType], StringType, "msvcrt")
            ExternalDecl("puts", [StringType], IntType, "msvcrt")
            ExternalDecl("getchar", [], IntType, "msvcrt")
        ]
        printfn "Debug: Added %d external I/O declarations" externalDecls.Length
        externalDecls @ declarations
    else
        declarations

/// Formats F# Compiler Services AST as readable text
let private formatFCSAst (parseTree: ParsedInput) : string =
    let sb = StringBuilder()
    
    let rec formatSynExpr (expr: SynExpr) (indent: int) =
        let indentStr = String.replicate indent "  "
        match expr with
        | SynExpr.Const(constant, _) ->
            sb.AppendLine(sprintf "%sSynExpr.Const(%A)" indentStr constant) |> ignore
        | SynExpr.Ident(ident) ->
            sb.AppendLine(sprintf "%sSynExpr.Ident(%s)" indentStr ident.idText) |> ignore
        | SynExpr.LongIdent(_, longIdent, _, _) ->
            let ids = match longIdent with SynLongIdent(ids, _, _) -> ids |> List.map (fun i -> i.idText) |> String.concat "."
            sb.AppendLine(sprintf "%sSynExpr.LongIdent(%s)" indentStr ids) |> ignore
        | SynExpr.App(_, _, func, arg, _) ->
            sb.AppendLine(sprintf "%sSynExpr.App(" indentStr) |> ignore
            formatSynExpr func (indent + 1)
            formatSynExpr arg (indent + 1)
            sb.AppendLine(sprintf "%s)" indentStr) |> ignore
        | SynExpr.LetOrUse(_, _, bindings, body, _, _) ->
            sb.AppendLine(sprintf "%sSynExpr.LetOrUse(" indentStr) |> ignore
            bindings |> List.iter (fun binding -> 
                sb.AppendLine(sprintf "%s  Binding(...)" indentStr) |> ignore)
            formatSynExpr body (indent + 1)
            sb.AppendLine(sprintf "%s)" indentStr) |> ignore
        | SynExpr.Sequential(_, _, expr1, expr2, _) ->
            sb.AppendLine(sprintf "%sSynExpr.Sequential(" indentStr) |> ignore
            formatSynExpr expr1 (indent + 1)
            formatSynExpr expr2 (indent + 1)
            sb.AppendLine(sprintf "%s)" indentStr) |> ignore
        | _ ->
            sb.AppendLine(sprintf "%sSynExpr.%s(...)" indentStr (expr.GetType().Name.Replace("SynExpr", ""))) |> ignore
    
    let rec formatDeclaration (decl: SynModuleDecl) (indent: int) =
        let indentStr = String.replicate indent "  "
        match decl with
        | SynModuleDecl.Let(_, bindings, _) ->
            sb.AppendLine(sprintf "%sSynModuleDecl.Let(" indentStr) |> ignore
            bindings |> List.iter (fun binding ->
                match binding with
                | SynBinding(_, _, _, _, _, _, _, pat, _, rhsExpr, _, _, _) ->
                    sb.AppendLine(sprintf "%s  Binding:" indentStr) |> ignore
                    formatSynExpr rhsExpr (indent + 2))
            sb.AppendLine(sprintf "%s)" indentStr) |> ignore
        | SynModuleDecl.Types(typeDefns, _) ->
            sb.AppendLine(sprintf "%sSynModuleDecl.Types(%d type definitions)" indentStr typeDefns.Length) |> ignore
        | SynModuleDecl.Expr(expr, _) ->
            sb.AppendLine(sprintf "%sSynModuleDecl.Expr(" indentStr) |> ignore
            formatSynExpr expr (indent + 1)
            sb.AppendLine(sprintf "%s)" indentStr) |> ignore
        | _ ->
            sb.AppendLine(sprintf "%s%s" indentStr (decl.GetType().Name)) |> ignore
    
    match parseTree with
    | ParsedInput.ImplFile(ParsedImplFileInput(fileName, isScript, qualifiedNameOfFile, scopedPragmas, hashDirectives, modules, _)) ->
        sb.AppendLine("F# Compiler Services AST:") |> ignore
        sb.AppendLine("========================") |> ignore
        sb.AppendLine(sprintf "File: %s" fileName) |> ignore
        sb.AppendLine(sprintf "Is Script: %b" isScript) |> ignore
        sb.AppendLine("") |> ignore
        
        modules |> List.iteri (fun i moduleOrNamespace ->
            match moduleOrNamespace with
            | SynModuleOrNamespace(longId, _, _, moduleDecls, _, _, _, _, _) ->
                let moduleName = 
                    if longId.IsEmpty then sprintf "Module%d" i
                    else longId |> List.map (fun i -> i.idText) |> String.concat "."
                sb.AppendLine(sprintf "Module: %s" moduleName) |> ignore
                sb.AppendLine("Declarations:") |> ignore
                moduleDecls |> List.iter (fun decl -> formatDeclaration decl 1))
    
    | ParsedInput.SigFile(_) ->
        sb.AppendLine("F# Signature File (not detailed)") |> ignore
    
    sb.ToString()

/// Formats Oak AST as readable text
let private formatOakAst (program: OakProgram) : string =
    let sb = StringBuilder()
    
    let rec formatOakExpr (expr: OakExpression) (indent: int) =
        let indentStr = String.replicate indent "  "
        match expr with
        | Literal lit ->
            sb.AppendLine(sprintf "%sLiteral(%A)" indentStr lit) |> ignore
        | Variable name ->
            sb.AppendLine(sprintf "%sVariable(%s)" indentStr name) |> ignore
        | Application(func, args) ->
            sb.AppendLine(sprintf "%sApplication(" indentStr) |> ignore
            formatOakExpr func (indent + 1)
            args |> List.iter (fun arg -> formatOakExpr arg (indent + 1))
            sb.AppendLine(sprintf "%s)" indentStr) |> ignore
        | Let(name, value, body) ->
            sb.AppendLine(sprintf "%sLet(%s," indentStr name) |> ignore
            formatOakExpr value (indent + 1)
            formatOakExpr body (indent + 1)
            sb.AppendLine(sprintf "%s)" indentStr) |> ignore
        | Sequential(first, second) ->
            sb.AppendLine(sprintf "%sSequential(" indentStr) |> ignore
            formatOakExpr first (indent + 1)
            formatOakExpr second (indent + 1)
            sb.AppendLine(sprintf "%s)" indentStr) |> ignore
        | IOOperation(ioType, args) ->
            sb.AppendLine(sprintf "%sIOOperation(%A," indentStr ioType) |> ignore
            args |> List.iter (fun arg -> formatOakExpr arg (indent + 1))
            sb.AppendLine(sprintf "%s)" indentStr) |> ignore
        | _ ->
            sb.AppendLine(sprintf "%s%s(...)" indentStr (expr.GetType().Name)) |> ignore
    
    let formatDeclaration (decl: OakDeclaration) (indent: int) =
        let indentStr = String.replicate indent "  "
        match decl with
        | FunctionDecl(name, params', returnType, body) ->
            sb.AppendLine(sprintf "%sFunctionDecl(%s, %A, %A," indentStr name params' returnType) |> ignore
            formatOakExpr body (indent + 1)
            sb.AppendLine(sprintf "%s)" indentStr) |> ignore
        | EntryPoint(expr) ->
            sb.AppendLine(sprintf "%sEntryPoint(" indentStr) |> ignore
            formatOakExpr expr (indent + 1)
            sb.AppendLine(sprintf "%s)" indentStr) |> ignore
        | TypeDecl(name, oakType) ->
            sb.AppendLine(sprintf "%sTypeDecl(%s, %A)" indentStr name oakType) |> ignore
        | ExternalDecl(name, paramTypes, returnType, libraryName) ->
            sb.AppendLine(sprintf "%sExternalDecl(%s, %A, %A, %s)" indentStr name paramTypes returnType libraryName) |> ignore
    
    sb.AppendLine("Oak AST:") |> ignore
    sb.AppendLine("========") |> ignore
    sb.AppendLine("") |> ignore
    
    program.Modules |> List.iter (fun module' ->
        sb.AppendLine(sprintf "Module: %s" module'.Name) |> ignore
        sb.AppendLine("Declarations:") |> ignore
        module'.Declarations |> List.iter (fun decl -> formatDeclaration decl 1)
        sb.AppendLine("") |> ignore)
    
    sb.ToString()

/// Cross-platform text file writing with UTF-8 encoding and Unix line endings
let private writeTextFile (filePath: string) (content: string) : unit =
    let encoding = UTF8Encoding(false)
    let normalizedContent = content.Replace("\r\n", "\n").Replace("\r", "\n")
    File.WriteAllText(filePath, normalizedContent, encoding)

/// Enhanced F# source parsing with comprehensive error handling and intermediate output
let parseAndConvertToOakAstWithIntermediate (sourceCode: string) : ParsingResult =
    try
        printfn "Debug: Starting F# source code parsing with intermediate output generation"
        printfn "Debug: Source code length: %d characters" sourceCode.Length
        
        let checker = createChecker()
        let fileName = "input.fs"
        let sourceText = SourceText.ofString sourceCode
        
        // Parse the source code
        let parseFileResults = 
            checker.ParseFile(fileName, sourceText, FSharpParsingOptions.Default)
            |> Async.RunSynchronously
        
        printfn "Debug: F# parsing completed, has errors: %b" (parseFileResults.Diagnostics.Length > 0)
        
        if parseFileResults.Diagnostics.Length > 0 then
            printfn "Debug: Parse diagnostics:"
            parseFileResults.Diagnostics |> Array.iter (fun diag ->
                printfn "  %s" (diag.ToString()))
        
        // Generate FCS AST text representation
        let fcsAstText = 
            match parseFileResults.ParseTree with
            | Some parseTree -> formatFCSAst parseTree
            | None -> "No parse tree available (parsing failed)"
        
        match parseFileResults.ParseTree with
        | Some parseTree ->
            match parseTree with
            | ParsedInput.ImplFile(ParsedImplFileInput(fileName, isScript, qualifiedNameOfFile, scopedPragmas, hashDirectives, modules, _)) ->
                printfn "Debug: Successfully parsed implementation file with %d modules" modules.Length
                
                let oakModules = 
                    modules |> List.mapi (fun i module' ->
                        match module' with
                        | SynModuleOrNamespace(longId, _, _, moduleDecls, _, _, _, _, _) ->
                            let moduleName = 
                                if longId.IsEmpty then 
                                    sprintf "Module%d" i
                                else 
                                    longId |> List.map (fun i -> i.idText) |> String.concat "."
                            
                            printfn "Debug: Processing module: %s with %d declarations" moduleName moduleDecls.Length
                            
                            let declarations = moduleDecls |> List.collect synModuleDeclToOakDecls
                            printfn "Debug: Module %s converted to %d Oak declarations" moduleName declarations.Length
                            
                            // Add external declarations if needed
                            let declarationsWithExternals = addStandardIODeclarations declarations
                            printfn "Debug: Module %s final declaration count: %d" moduleName declarationsWithExternals.Length
                            
                            { Name = moduleName; Declarations = declarationsWithExternals })
                
                let program = { Modules = oakModules }
                let oakAstText = formatOakAst program
                
                printfn "Debug: Created Oak program with %d modules" program.Modules.Length
                
                {
                    OakProgram = program
                    FCSAstText = fcsAstText
                    OakAstText = oakAstText
                }
            
            | ParsedInput.SigFile(_) ->
                printfn "Debug: Signature file not supported, creating default program"
                let defaultProgram = { Modules = [{ Name = "Main"; Declarations = [EntryPoint(Literal(IntLiteral(0)))] }] }
                {
                    OakProgram = defaultProgram
                    FCSAstText = fcsAstText
                    OakAstText = formatOakAst defaultProgram
                }
        
        | None ->
            printfn "Debug: No parse tree available, creating fallback program"
            let fallbackProgram = { Modules = [{ Name = "Main"; Declarations = [EntryPoint(Literal(IntLiteral(0)))] }] }
            {
                OakProgram = fallbackProgram
                FCSAstText = fcsAstText
                OakAstText = formatOakAst fallbackProgram
            }
    
    with
    | ex ->
        printfn "Debug: Parse error: %s" ex.Message
        printfn "Debug: Creating minimal fallback program"
        let minimalProgram = { Modules = [{ Name = "Main"; Declarations = [EntryPoint(Literal(IntLiteral(0)))] }] }
        {
            OakProgram = minimalProgram
            FCSAstText = sprintf "Parse error: %s" ex.Message
            OakAstText = formatOakAst minimalProgram
        }

/// Enhanced parsing function that saves intermediate files
let parseAndConvertToOakAstWithFiles (sourceCode: string) (basePath: string) (baseName: string) (keepIntermediates: bool) : OakProgram =
    let result = parseAndConvertToOakAstWithIntermediate sourceCode
    
    if keepIntermediates then
        try
            let intermediatesDir = Path.Combine(basePath, "intermediates")
            if not (Directory.Exists(intermediatesDir)) then
                Directory.CreateDirectory(intermediatesDir) |> ignore
            
            // Save FCS AST file
            let fcsPath = Path.Combine(intermediatesDir, baseName + ".fcs")
            writeTextFile fcsPath result.FCSAstText
            printfn "Saved F# Compiler Services AST to: %s" fcsPath
            
            // Save Oak AST file  
            let oakPath = Path.Combine(intermediatesDir, baseName + ".oak")
            writeTextFile oakPath result.OakAstText
            printfn "Saved Oak AST to: %s" oakPath
            
        with
        | ex ->
            printfn "Warning: Failed to save intermediate AST files: %s" ex.Message
    
    result.OakProgram

/// Backward compatibility function - maintains existing interface
let parseAndConvertToOakAst (sourceCode: string) : OakProgram =
    let result = parseAndConvertToOakAstWithIntermediate sourceCode
    result.OakProgram