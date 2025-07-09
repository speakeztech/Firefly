module Core.IngestionPipeline

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization
open FSharp.Compiler.Syntax
open FSharp.Compiler.Text
open Core.FCS.ProjectContext
open Core.FCS.SymbolAnalysis
open Core.Analysis.CouplingCohesion
open Core.Analysis.Reachability
open Core.Analysis.MemoryLayout
open Core.Utilities.IntermediateWriter

/// Configure JSON serialization with F# support
let private createJsonOptions() =
    let options = JsonSerializerOptions(WriteIndented = true)
    options.Converters.Add(JsonFSharpConverter())
    options

/// Global JSON options for the pipeline
let private jsonOptions = createJsonOptions()

/// Extract the string value from QualifiedNameOfFile
let private extractQualifiedName (qualName: QualifiedNameOfFile) =
    match qualName with
    | QualifiedNameOfFile ident -> ident.idText

/// Pipeline configuration
type PipelineConfig = {
    CacheStrategy: CacheStrategy
    TemplateName: string option
    CustomTemplateDir: string option
    EnableCouplingAnalysis: bool
    EnableMemoryOptimization: bool
    OutputIntermediates: bool
    IntermediatesDir: string option
}

/// Pipeline result
type PipelineResult = {
    Success: bool
    ProjectResults: ProjectResults option
    CouplingAnalysis: CouplingAnalysisResult option
    ReachabilityAnalysis: EnhancedReachability option
    MemoryLayout: LayoutStrategy option
    Diagnostics: Diagnostic list
}

and CouplingAnalysisResult = {
    Components: CodeComponent list
    Couplings: Coupling list
    Report: {| TotalUnits: int; ComponentCount: int; AverageCohesion: float |}
}

and Diagnostic = {
    Severity: DiagnosticSeverity
    Message: string
    Location: string option
}

and DiagnosticSeverity = Info | Warning | Error

/// Default pipeline configuration
let defaultConfig = {
    CacheStrategy = Balanced
    TemplateName = None
    CustomTemplateDir = None
    EnableCouplingAnalysis = true
    EnableMemoryOptimization = true
    OutputIntermediates = false
    IntermediatesDir = None
}

/// Flexible JSON representation for AST nodes
type JsonNode = System.Collections.Generic.Dictionary<string, obj>

/// Create a JSON node with a kind field
let private jsonNode (kind: string) =
    let node = JsonNode()
    node["kind"] <- kind
    node

/// Convert range to JSON representation
let private rangeToJson (range: range) =
    {| 
        fileName = range.FileName
        start = {| line = range.Start.Line; column = range.Start.Column |}
        ``end`` = {| line = range.End.Line; column = range.End.Column |}
    |}

/// Convert range to parenthetical format
let private rangeToString (range: range) =
    sprintf "[%d:%d-%d:%d]" 
        range.Start.Line range.Start.Column 
        range.End.Line range.End.Column

/// Extract constant value to JSON
let private extractConstant (c: SynConst) =
    let node = jsonNode "Const"
    match c with
    | SynConst.Unit -> 
        node["constType"] <- "Unit"
        node["value"] <- null
    | SynConst.Bool b -> 
        node["constType"] <- "Bool"
        node["value"] <- b
    | SynConst.Byte b -> 
        node["constType"] <- "Byte"
        node["value"] <- b
    | SynConst.Int16 i -> 
        node["constType"] <- "Int16"
        node["value"] <- i
    | SynConst.Int32 i -> 
        node["constType"] <- "Int32"
        node["value"] <- i
    | SynConst.Int64 i -> 
        node["constType"] <- "Int64"
        node["value"] <- i
    | SynConst.UInt16 i -> 
        node["constType"] <- "UInt16"
        node["value"] <- i
    | SynConst.UInt32 i -> 
        node["constType"] <- "UInt32"
        node["value"] <- i
    | SynConst.UInt64 i -> 
        node["constType"] <- "UInt64"
        node["value"] <- i
    | SynConst.Single f -> 
        node["constType"] <- "Single"
        node["value"] <- f
    | SynConst.Double d -> 
        node["constType"] <- "Double"
        node["value"] <- d
    | SynConst.Char c -> 
        node["constType"] <- "Char"
        node["value"] <- c
    | SynConst.String(s, _, _) -> 
        node["constType"] <- "String"
        node["value"] <- s
    | SynConst.Decimal d -> 
        node["constType"] <- "Decimal"
        node["value"] <- d
    | _ -> 
        node["constType"] <- "Other"
        node["value"] <- c.ToString()
    node

/// Extract pattern to JSON
let rec private extractPattern (pat: SynPat) =
    let node = jsonNode "Pattern"
    let range = 
        match pat with
        | SynPat.Const(_, r) -> r
        | SynPat.Wild r -> r
        | SynPat.Named(_, _, _, r) -> r
        | SynPat.LongIdent(_, _, _, _, _, r) -> r
        | SynPat.Tuple(_, _, _, r) -> r
        | SynPat.ArrayOrList(_, _, r) -> r
        | SynPat.Or(_, _, r, _) -> r
        | SynPat.As(_, _, r) -> r
        | _ -> range.Zero
    
    node["range"] <- rangeToJson range
    
    match pat with
    | SynPat.Const(c, _) ->
        node["patternType"] <- "Const"
        node["constant"] <- extractConstant c
    | SynPat.Wild _ ->
        node["patternType"] <- "Wild"
    | SynPat.Named(SynIdent(ident, _), _, _, _) ->
        node["patternType"] <- "Named"
        node["name"] <- ident.idText
    | SynPat.LongIdent(SynLongIdent(ids, _, _), _, _, args, _, _) ->
        node["patternType"] <- "LongIdent"
        node["name"] <- ids |> List.map (fun id -> id.idText) |> String.concat "."
        match args with
        | SynArgPats.Pats pats -> 
            node["args"] <- (pats |> List.map extractPattern |> List.toArray)
        | _ -> ()
    | SynPat.Tuple(_, pats, _, _) ->
        node["patternType"] <- "Tuple"
        node["patterns"] <- (pats |> List.map extractPattern |> List.toArray)
    | SynPat.ArrayOrList(isList, pats, _) ->
        node["patternType"] <- if isList then "List" else "Array"
        node["patterns"] <- (pats |> List.map extractPattern |> List.toArray)
    | SynPat.Or(lhs, rhs, _, _) ->
        node["patternType"] <- "Or"
        node["left"] <- extractPattern lhs
        node["right"] <- extractPattern rhs
    | SynPat.As(lhs, rhs, _) ->
        node["patternType"] <- "As"
        node["pattern"] <- extractPattern lhs
        node["name"] <- extractPattern rhs
    | _ ->
        node["patternType"] <- pat.GetType().Name
    node

/// Extract expression tree to JSON
let rec private extractExpression (expr: SynExpr) =
    match expr with
    | SynExpr.Paren(expr, _, _, _) ->
        // For parentheses, just return the inner expression
        extractExpression expr
        
    | _ ->
        // For all other cases, create a node and populate it
        let node = jsonNode "Expr"
        let range = expr.Range
        node["range"] <- rangeToJson range
        
        match expr with
        | SynExpr.Const(c, _) ->
            node["exprType"] <- "Const"
            node["constant"] <- extractConstant c
            
        | SynExpr.Ident ident ->
            node["exprType"] <- "Ident"
            node["name"] <- ident.idText
            
        | SynExpr.LongIdent(_, SynLongIdent(ids, _, _), _, _) ->
            node["exprType"] <- "LongIdent"
            node["name"] <- ids |> List.map (fun id -> id.idText) |> String.concat "."
            
        | SynExpr.App(_, isInfix, funcExpr, argExpr, _) ->
            node["exprType"] <- "App"
            node["isInfix"] <- isInfix
            node["funcExpr"] <- extractExpression funcExpr
            node["argExpr"] <- extractExpression argExpr
               
        | SynExpr.Lambda(_, _, args, body, _, _, _) ->
            node["exprType"] <- "Lambda"
            node["args"] <- jsonNode "SimplePats"  // TODO: extract args properly
            node["body"] <- extractExpression body
               
        | SynExpr.LetOrUse(isRec, isUse, bindings, body, _, _) ->
            node["exprType"] <- if isUse then "Use" else "Let"
            node["isRecursive"] <- isRec
            let bindingNodes = 
                bindings |> List.map (fun (SynBinding(_, _, _, _, _, _, _, pat, _, expr, _, _, _)) ->
                    let bindNode = jsonNode "Binding"
                    bindNode["pattern"] <- extractPattern pat
                    bindNode["expression"] <- extractExpression expr
                    bindNode)
            node["bindings"] <- bindingNodes |> List.toArray
            node["body"] <- extractExpression body
               
        | SynExpr.Match(_, expr, clauses, _, _) ->
            node["exprType"] <- "Match"
            node["expression"] <- extractExpression expr
            let clauseNodes = 
                clauses |> List.map (fun (SynMatchClause(pat, whenExpr, result, _, _, _)) ->
                    let clauseNode = jsonNode "MatchClause"
                    clauseNode["pattern"] <- extractPattern pat
                    whenExpr |> Option.iter (fun e -> clauseNode["guard"] <- extractExpression e)
                    clauseNode["result"] <- extractExpression result
                    clauseNode)
            node["clauses"] <- clauseNodes |> List.toArray
                      
        | SynExpr.Sequential(_, _, expr1, expr2, _, _) ->
            node["exprType"] <- "Sequential"
            node["expr1"] <- extractExpression expr1
            node["expr2"] <- extractExpression expr2
               
        | SynExpr.IfThenElse(ifExpr, thenExpr, elseExpr, _, _, _, _) ->
            node["exprType"] <- "IfThenElse"
            node["condition"] <- extractExpression ifExpr
            node["thenBranch"] <- extractExpression thenExpr
            elseExpr |> Option.iter (fun e -> node["elseBranch"] <- extractExpression e)
               
        | SynExpr.Tuple(_, exprs, _, _) ->
            node["exprType"] <- "Tuple"
            node["elements"] <- (exprs |> List.map extractExpression |> List.toArray)
               
        | SynExpr.ArrayOrList(isList, exprs, _) ->
            node["exprType"] <- if isList then "List" else "Array"
            node["elements"] <- (exprs |> List.map extractExpression |> List.toArray)
               
        | SynExpr.Record(_, copyInfo, fields, _) ->
            node["exprType"] <- "Record"
            copyInfo |> Option.iter (fun (expr, _) -> node["copyFrom"] <- extractExpression expr)
            let fieldNodes =
                fields |> List.choose (fun (SynExprRecordField((SynLongIdent(ids, _, _), _), _, expr, _)) ->
                    expr |> Option.map (fun e ->
                        let fieldNode = jsonNode "RecordField"
                        fieldNode["name"] <- ids |> List.map (fun id -> id.idText) |> String.concat "."
                        fieldNode["value"] <- extractExpression e
                        fieldNode))
            node["fields"] <- fieldNodes |> List.toArray
                          
        | SynExpr.New(_, typeName, expr, _) ->
            node["exprType"] <- "New"
            node["targetType"] <- jsonNode "Type"  // TODO: extract type properly
            node["argExpr"] <- extractExpression expr
               
        | SynExpr.TypeApp(expr, _, types, _, _, _, _) ->
            node["exprType"] <- "TypeApp"
            node["expression"] <- extractExpression expr
            node["typeArgCount"] <- types.Length
               
        | _ ->
            node["exprType"] <- expr.GetType().Name
        
        node

/// Convert pattern to parenthetical notation with ranges
let rec private patternToParenthetical (pat: SynPat) : string =
    let range = 
        match pat with
        | SynPat.Const(_, r) -> r
        | SynPat.Wild r -> r
        | SynPat.Named(_, _, _, r) -> r
        | SynPat.LongIdent(_, _, _, _, _, r) -> r
        | SynPat.Tuple(_, _, _, r) -> r
        | SynPat.ArrayOrList(_, _, r) -> r
        | SynPat.Or(_, _, r, _) -> r
        | SynPat.As(_, _, r) -> r
        | _ -> range.Zero
        
    let rangeStr = rangeToString range
    
    match pat with
    | SynPat.Const(c, _) ->
        sprintf "(Pat.Const %A :range %s)" c rangeStr
    | SynPat.Wild _ ->
        sprintf "(Pat.Wild :range %s)" rangeStr
    | SynPat.Named(SynIdent(ident, _), _, _, _) ->
        sprintf "(Pat.Named \"%s\" :range %s)" ident.idText rangeStr
    | SynPat.LongIdent(SynLongIdent(ids, _, _), _, _, args, _, _) ->
        let name = ids |> List.map (fun id -> id.idText) |> String.concat "."
        match args with
        | SynArgPats.Pats pats when not (List.isEmpty pats) ->
            let argsStr = pats |> List.map patternToParenthetical |> String.concat " "
            sprintf "(Pat.LongIdent \"%s\" %s :range %s)" name argsStr rangeStr
        | _ ->
            sprintf "(Pat.LongIdent \"%s\" :range %s)" name rangeStr
    | SynPat.Tuple(_, pats, _, _) ->
        let patsStr = pats |> List.map patternToParenthetical |> String.concat " "
        sprintf "(Pat.Tuple %s :range %s)" patsStr rangeStr
    | _ ->
        sprintf "(Pat.%s :range %s)" (pat.GetType().Name) rangeStr

/// Convert expression to parenthetical (LISP-like) notation with ranges
let rec private exprToParenthetical (expr: SynExpr) : string =
    match expr with
    | SynExpr.Paren(expr, _, _, _) ->
        // Unwrap parens
        exprToParenthetical expr
        
    | _ ->
        let rangeStr = rangeToString expr.Range
        match expr with
        | SynExpr.Const(c, _) ->
            match c with
            | SynConst.Unit -> sprintf "(Const.Unit :range %s)" rangeStr
            | SynConst.Bool b -> sprintf "(Const.Bool %b :range %s)" b rangeStr
            | SynConst.Int32 i -> sprintf "(Const.Int32 %d :range %s)" i rangeStr
            | SynConst.String(s, _, _) -> sprintf "(Const.String \"%s\" :range %s)" (s.Replace("\"", "\\\"")) rangeStr
            | _ -> sprintf "(Const %A :range %s)" c rangeStr
            
        | SynExpr.Ident ident ->
            sprintf "(Ident \"%s\" :range %s)" ident.idText rangeStr
            
        | SynExpr.LongIdent(_, SynLongIdent(ids, _, _), _, _) ->
            let name = ids |> List.map (fun id -> id.idText) |> String.concat "."
            sprintf "(LongIdent \"%s\" :range %s)" name rangeStr
            
        | SynExpr.App(_, isInfix, funcExpr, argExpr, _) ->
            if isInfix then
                sprintf "(InfixApp %s %s :range %s)" (exprToParenthetical funcExpr) (exprToParenthetical argExpr) rangeStr
            else
                sprintf "(App %s %s :range %s)" (exprToParenthetical funcExpr) (exprToParenthetical argExpr) rangeStr
                
        | SynExpr.Lambda(_, _, _, body, _, _, _) ->
            sprintf "(Lambda <args> %s :range %s)" (exprToParenthetical body) rangeStr
            
        | SynExpr.LetOrUse(_, isUse, bindings, body, _, _) ->
            let bindStrs = 
                bindings 
                |> List.map (fun (SynBinding(_, _, _, _, _, _, _, pat, _, expr, _, _, _)) ->
                    sprintf "(Binding %s %s)" (patternToParenthetical pat) (exprToParenthetical expr))
                |> String.concat " "
            sprintf "(%s %s %s :range %s)" (if isUse then "Use" else "Let") bindStrs (exprToParenthetical body) rangeStr
            
        | SynExpr.Match(_, expr, clauses, _, _) ->
            let clauseStrs = 
                clauses 
                |> List.map (fun (SynMatchClause(pat, whenExpr, result, _, _, _)) ->
                    let whenStr = whenExpr |> Option.map (fun e -> sprintf " :when %s" (exprToParenthetical e)) |> Option.defaultValue ""
                    sprintf "(Clause %s%s %s)" (patternToParenthetical pat) whenStr (exprToParenthetical result))
                |> String.concat " "
            sprintf "(Match %s %s :range %s)" (exprToParenthetical expr) clauseStrs rangeStr
            
        | SynExpr.Sequential(_, _, expr1, expr2, _, _) ->
            sprintf "(Seq %s %s :range %s)" (exprToParenthetical expr1) (exprToParenthetical expr2) rangeStr
            
        | SynExpr.IfThenElse(ifExpr, thenExpr, elseExpr, _, _, _, _) ->
            match elseExpr with
            | Some e -> sprintf "(If %s %s %s :range %s)" (exprToParenthetical ifExpr) (exprToParenthetical thenExpr) (exprToParenthetical e) rangeStr
            | None -> sprintf "(If %s %s :range %s)" (exprToParenthetical ifExpr) (exprToParenthetical thenExpr) rangeStr
            
        | SynExpr.Tuple(_, exprs, _, _) ->
            let elemsStr = exprs |> List.map exprToParenthetical |> String.concat " "
            sprintf "(Tuple %s :range %s)" elemsStr rangeStr
            
        | SynExpr.ArrayOrList(isList, exprs, _) ->
            let elemsStr = exprs |> List.map exprToParenthetical |> String.concat " "
            sprintf "(%s %s :range %s)" (if isList then "List" else "Array") elemsStr rangeStr
            
        | SynExpr.TypeApp(expr, _, types, _, _, _, _) ->
            sprintf "(TypeApp %s <%d types> :range %s)" (exprToParenthetical expr) types.Length rangeStr
            
        | _ ->
            sprintf "(%s ... :range %s)" (expr.GetType().Name) rangeStr

/// Run the complete compilation pipeline
let runPipeline (projectPath: string) (config: PipelineConfig) = async {
    let diagnostics = ResizeArray<Diagnostic>()
    
    try
        // Ensure intermediates directory exists if needed
        if config.OutputIntermediates && config.IntermediatesDir.IsSome then
            Directory.CreateDirectory(config.IntermediatesDir.Value) |> ignore
        
        // Step 1: Load project with FCS
        printfn "[Pipeline] Loading project: %s" projectPath
        let! ctx = loadProject projectPath config.CacheStrategy
        
        // Step 2: Get complete project results
        printfn "[Pipeline] Analyzing project with FCS..."
        let! projectResults = getProjectResults ctx
        
        // Write parsed AST immediately with full details
        if config.OutputIntermediates && config.IntermediatesDir.IsSome then
            // Write summary
            let parseData = {|
                files = projectResults.ParseResults |> Array.map (fun pr -> {|
                    fileName = pr.FileName
                    hasErrors = pr.Diagnostics.Length > 0
                    diagnosticCount = pr.Diagnostics.Length
                |})
                parseCount = projectResults.ParseResults.Length
            |}
            let parseJson = JsonSerializer.Serialize(parseData, jsonOptions)
            let parsePath = Path.Combine(config.IntermediatesDir.Value, "parse.ast.json")
            writeFileToPath parsePath parseJson
            
            // Write a combined initial AST file for quick overview
            let allInitialAst = 
                projectResults.ParseResults
                |> Array.map (fun pr ->
                    match pr.ParseTree with
                    | ParsedInput.ImplFile(ParsedImplFileInput(fileName, _, qualName, _, _, modules, _, _, _)) ->
                        let moduleStrs = 
                            modules |> List.map (fun m ->
                                match m with
                                | SynModuleOrNamespace(longId, _, _, _, _, _, _, _, _) ->
                                    let name = longId |> List.map (fun id -> id.idText) |> String.concat "."
                                    sprintf "(Module \"%s\" :range %s)" name (rangeToString m.Range))
                            |> String.concat "\n  "
                        sprintf ";; File: %s\n;; Qualified: %s\n%s" fileName (extractQualifiedName qualName) moduleStrs
                    | ParsedInput.SigFile(_) -> ";; Signature file")
                |> String.concat "\n\n"
            let allInitialPath = Path.Combine(config.IntermediatesDir.Value, "all.initial.ast")
            writeFileToPath allInitialPath (sprintf ";; Firefly Initial AST - %d files\n\n%s" projectResults.ParseResults.Length allInitialAst)
            
            // Write detailed AST for each file
            projectResults.ParseResults
            |> Array.iter (fun pr ->
                match pr.ParseTree with
                | ParsedInput.ImplFile(ParsedImplFileInput(fileName, isScript, qualName, _, _, modules, _, _, _)) ->
                    let detailedAst = {|
                        fileName = fileName
                        isScript = isScript
                        qualifiedName = extractQualifiedName qualName
                        modules = 
                            modules |> List.map (fun m ->
                                match m with
                                | SynModuleOrNamespace(longId, isRecursive, kind, decls, _, attrs, _, _, trivia) ->
                                    {|
                                        name = longId |> List.map (fun id -> id.idText) |> String.concat "."
                                        isRecursive = isRecursive
                                        isModule = match kind with SynModuleOrNamespaceKind.NamedModule -> true | _ -> false
                                        range = rangeToJson m.Range
                                        declarationCount = decls.Length
                                        declarations = 
                                            decls |> List.map (fun decl ->
                                                match decl with
                                                | SynModuleDecl.Let(isRec, bindings, range) ->
                                                    let bindingDetails = 
                                                        bindings |> List.map (fun binding ->
                                                            match binding with
                                                            | SynBinding(_, _, _, _, _, _, _, pat, returnInfo, expr, _, _, _) ->
                                                                let name, pattern = 
                                                                    match pat with
                                                                    | SynPat.Named(synIdent, _, _, _) -> 
                                                                        match synIdent with
                                                                        | SynIdent(ident, _) -> ident.idText, "Named"
                                                                    | SynPat.LongIdent(longIdentPat, _, _, _, _, _) ->
                                                                        let ids = longIdentPat.LongIdent  
                                                                        (ids |> List.map (fun id -> id.idText) |> String.concat "."), "LongIdent"
                                                                    | _ -> 
                                                                        "<complex>", pat.GetType().Name
                                                                
                                                                {| 
                                                                    name = name
                                                                    pattern = pattern
                                                                    expression = box (extractExpression expr)
                                                                |})
                                                    {| 
                                                        kind = "Let"
                                                        isRecursive = isRec
                                                        range = rangeToJson range
                                                        bindingCount = bindings.Length
                                                        bindings = bindingDetails
                                                    |}
                                                | SynModuleDecl.Types(typeDefns, range) ->
                                                    {| 
                                                        kind = "Types"
                                                        isRecursive = false
                                                        range = rangeToJson range
                                                        bindingCount = typeDefns.Length
                                                        bindings = typeDefns |> List.map (fun td ->
                                                            match td with
                                                            | SynTypeDefn(componentInfo, _, _, _, _, _) ->
                                                                match componentInfo with
                                                                | SynComponentInfo(_, _, _, longId, _, _, _, _) ->
                                                                    {| name = longId |> List.map (fun id -> id.idText) |> String.concat "."
                                                                       pattern = "TypeDefn"
                                                                       expression = box (jsonNode "TypeDefinition") |})
                                                    |}
                                                | SynModuleDecl.NestedModule(componentInfo, _, _, _, range, _) ->
                                                    match componentInfo with
                                                    | SynComponentInfo(_, _, _, longId, _, _, _, _) ->
                                                        {| 
                                                            kind = "NestedModule"
                                                            isRecursive = false
                                                            range = rangeToJson range
                                                            bindingCount = 0
                                                            bindings = [{| name = longId |> List.map (fun id -> id.idText) |> String.concat "."
                                                                           pattern = "Module"
                                                                           expression = box (jsonNode "ModuleDefinition") |}]
                                                        |}
                                                | SynModuleDecl.Open(target, range) ->
                                                    let openName = 
                                                        match target with
                                                        | SynOpenDeclTarget.ModuleOrNamespace(SynLongIdent(identList, _, _), _) ->
                                                            identList |> List.map (fun id -> id.idText) |> String.concat "."
                                                        | _ -> "<type>"
                                                    {| 
                                                        kind = "Open"
                                                        isRecursive = false
                                                        range = rangeToJson range
                                                        bindingCount = 0
                                                        bindings = [{| name = openName
                                                                       pattern = "Open"
                                                                       expression = box (jsonNode "OpenDeclaration") |}]
                                                    |}
                                                | _ ->
                                                    {| 
                                                        kind = decl.GetType().Name
                                                        isRecursive = false
                                                        range = rangeToJson decl.Range
                                                        bindingCount = 0
                                                        bindings = []
                                                    |})
                                    |})
                    |}
                    let fileName = Path.GetFileNameWithoutExtension(pr.FileName)
                    let detailPath = Path.Combine(config.IntermediatesDir.Value, $"{fileName}.parsed.ast.json")
                    let detailJson = JsonSerializer.Serialize(detailedAst, jsonOptions)
                    writeFileToPath detailPath detailJson
                    
                    // Also write initial parenthetical AST with full structure
                    let initialAstPath = Path.Combine(config.IntermediatesDir.Value, $"{fileName}.initial.ast")
                    let initialAst = 
                        modules 
                        |> List.map (fun m ->
                            match m with
                            | SynModuleOrNamespace(longId, _, kind, decls, _, _, _, _, _) ->
                                let moduleName = longId |> List.map (fun id -> id.idText) |> String.concat "."
                                let moduleRange = rangeToString m.Range
                                let declsStr = 
                                    decls 
                                    |> List.map (fun decl ->
                                        match decl with
                                        | SynModuleDecl.Let(_, bindings, range) ->
                                            let bindingsStr =
                                                bindings
                                                |> List.map (fun (SynBinding(_, _, _, _, _, _, _, pat, _, expr, _, _, _)) ->
                                                    let name = 
                                                        match pat with
                                                        | SynPat.Named(SynIdent(ident, _), _, _, _) -> ident.idText
                                                        | SynPat.LongIdent(SynLongIdent(ids, _, _), _, _, _, _, _) ->
                                                            ids |> List.map (fun id -> id.idText) |> String.concat "."
                                                        | _ -> "<pattern>"
                                                    sprintf "(Let \"%s\" %s :range %s)" name (exprToParenthetical expr) (rangeToString range))
                                                |> String.concat "\n  "
                                            bindingsStr
                                        | SynModuleDecl.Open(SynOpenDeclTarget.ModuleOrNamespace(SynLongIdent(ids, _, _), _), range) ->
                                            sprintf "(Open \"%s\" :range %s)" (ids |> List.map (fun id -> id.idText) |> String.concat ".") (rangeToString range)
                                        | SynModuleDecl.Types(types, range) ->
                                            sprintf "(Types %d :range %s)" types.Length (rangeToString range)
                                        | SynModuleDecl.NestedModule(SynComponentInfo(_, _, _, longId, _, _, _, _), _, _, _, range, _) ->
                                            sprintf "(Module \"%s\" :range %s)" (longId |> List.map (fun id -> id.idText) |> String.concat ".") (rangeToString range)
                                        | _ ->
                                            sprintf "(%s :range %s)" (decl.GetType().Name) (rangeToString decl.Range))
                                    |> String.concat "\n  "
                                sprintf "(Module \"%s\" :range %s\n  %s)" moduleName moduleRange declsStr)
                        |> String.concat "\n\n"
                    writeFileToPath initialAstPath initialAst
                    
                    // Write AST size report
                    printfn "  Wrote %s (%d bytes)" (Path.GetFileName detailPath) (detailJson.Length)
                    printfn "  Wrote %s (%d bytes)" (Path.GetFileName initialAstPath) (initialAst.Length)
                | ParsedInput.SigFile(_) -> ()
            )
        
        // Write type-checked results 
        if config.OutputIntermediates && config.IntermediatesDir.IsSome then
            let typeCheckData = {|
                Files = projectResults.CompilationOrder
                SymbolCount = projectResults.SymbolUses.Length
                Timestamp = DateTime.UtcNow
            |}
            let typeCheckJson = JsonSerializer.Serialize(typeCheckData, jsonOptions)
            let typeCheckPath = Path.Combine(config.IntermediatesDir.Value, "typeChecked.json")
            writeFileToPath typeCheckPath typeCheckJson
        
        // Continue with rest of pipeline...
        if projectResults.CheckResults.HasCriticalErrors then
            diagnostics.Add {
                Severity = Error
                Message = "Project has critical errors"
                Location = Some projectPath
            }
            return {
                Success = false
                ProjectResults = Some projectResults
                CouplingAnalysis = None
                ReachabilityAnalysis = None
                MemoryLayout = None
                Diagnostics = List.ofSeq diagnostics
            }
        else
            // Rest of your pipeline implementation...
            printfn "[Pipeline] Extracting symbol relationships..."
            let relationships = extractRelationships projectResults.SymbolUses
            
            // Continue with analysis phases...
            return {
                Success = true
                ProjectResults = Some projectResults
                CouplingAnalysis = None
                ReachabilityAnalysis = None
                MemoryLayout = None
                Diagnostics = List.ofSeq diagnostics
            }
            
    with ex ->
        diagnostics.Add {
            Severity = Error
            Message = sprintf "Pipeline failed: %s" ex.Message
            Location = Some projectPath
        }
        return {
            Success = false
            ProjectResults = None
            CouplingAnalysis = None
            ReachabilityAnalysis = None
            MemoryLayout = None
            Diagnostics = List.ofSeq diagnostics
        }
}