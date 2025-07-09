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
    CacheStrategy = Conservative
    TemplateName = None
    CustomTemplateDir = None
    EnableCouplingAnalysis = true
    EnableMemoryOptimization = true
    OutputIntermediates = true
    IntermediatesDir = None
}

/// Convert range to JSON-friendly format
let private rangeToJson (range: range) =
    {| 
        fileName = range.FileName
        start = {| line = range.Start.Line; column = range.Start.Column |}
        ``end`` = {| line = range.End.Line; column = range.End.Column |}
    |}

/// Convert range to string for parenthetical AST
let private rangeToString (range: range) =
    sprintf "[%d:%d-%d:%d]" range.Start.Line range.Start.Column range.End.Line range.End.Column

/// Create a simple JSON node placeholder
let private jsonNode (nodeType: string) =
    {| kind = nodeType |}

/// Run the ingestion pipeline
let runPipeline (projectPath: string) (config: PipelineConfig) : Async<PipelineResult> = async {
    let diagnostics = ResizeArray<Diagnostic>()
    
    try
        // Step 1: Load project
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
                                | SynModuleOrNamespace(longId, isRecursive, kind, decls, _, attrs, _, _, _) ->
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
                                                            | SynBinding(_, _, _, _, _, _, _, pat, _, expr, _, _, _) ->
                                                                let name, pattern = 
                                                                    match pat with
                                                                    | SynPat.Named(SynIdent(ident, _), _, _, _) -> 
                                                                        ident.idText, "Named"
                                                                    | SynPat.LongIdent(SynLongIdent(ids, _, _), _, _, _, _, _) ->
                                                                        (ids |> List.map (fun id -> id.idText) |> String.concat "."), "LongIdent"
                                                                    | _ -> 
                                                                        "<pattern>", pat.GetType().Name
                                                                {| name = name
                                                                   pattern = pattern
                                                                   expression = jsonNode "Expression" |})
                                                    {| 
                                                        kind = "Let"
                                                        isRecursive = isRec
                                                        range = rangeToJson range
                                                        bindingCount = bindings.Length
                                                        bindings = bindingDetails
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
                                                                       expression = jsonNode "OpenDeclaration" |}]
                                                    |}
                                                
                                                | SynModuleDecl.Types(typeDefns, range) ->
                                                    {| 
                                                        kind = "Types"
                                                        isRecursive = false
                                                        range = rangeToJson range
                                                        bindingCount = 0
                                                        bindings = 
                                                            typeDefns |> List.map (fun td ->
                                                                match td with
                                                                | SynTypeDefn(SynComponentInfo(_, _, _, longId, _, _, _, _), _, _, _, _, _) ->
                                                                    {| name = longId |> List.map (fun id -> id.idText) |> String.concat "."
                                                                       pattern = "TypeDefn"
                                                                       expression = jsonNode "TypeDefinition" |})
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
                                                                           expression = jsonNode "ModuleDefinition" |}]
                                                        |}
                                                
                                                | SynModuleDecl.Exception(exnDefn, range) ->
                                                    match exnDefn with
                                                    | SynExceptionDefn(SynExceptionDefnRepr(_, SynUnionCase(_, SynIdent(ident, _), _, _, _, _, _), _, _, _, _), _, _, _) ->
                                                        {| 
                                                            kind = "Exception"
                                                            isRecursive = false
                                                            range = rangeToJson range
                                                            bindingCount = 0
                                                            bindings = [{| name = ident.idText
                                                                           pattern = "Exception"
                                                                           expression = jsonNode "ExceptionDefinition" |}]
                                                        |}
                                                
                                                | SynModuleDecl.Expr(expr, range) ->
                                                    {| 
                                                        kind = "Expr"
                                                        isRecursive = false
                                                        range = rangeToJson range
                                                        bindingCount = 0
                                                        bindings = []
                                                    |}
                                                
                                                | SynModuleDecl.Attributes(attrs, range) ->
                                                    {| 
                                                        kind = "Attributes"
                                                        isRecursive = false
                                                        range = rangeToJson range
                                                        bindingCount = 0
                                                        bindings = []
                                                    |}
                                                
                                                | SynModuleDecl.HashDirective(ParsedHashDirective(directive, _, _), range) ->
                                                    {| 
                                                        kind = "HashDirective"
                                                        isRecursive = false
                                                        range = rangeToJson range
                                                        bindingCount = 0
                                                        bindings = [{| name = directive
                                                                       pattern = "Directive"
                                                                       expression = jsonNode directive |}]
                                                    |}
                                                
                                                | SynModuleDecl.ModuleAbbrev(ident, longId, range) ->
                                                    {| 
                                                        kind = "ModuleAbbrev"
                                                        isRecursive = false
                                                        range = rangeToJson range
                                                        bindingCount = 0
                                                        bindings = [{| name = ident.idText
                                                                       pattern = "Abbrev"
                                                                       expression = jsonNode (longId |> List.map (fun id -> id.idText) |> String.concat ".") |}]
                                                    |}
                                                
                                                | SynModuleDecl.NamespaceFragment _ ->
                                                    {| 
                                                        kind = "NamespaceFragment"
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
                    let rec exprToParenthetical expr = 
                        match expr with
                        | SynExpr.Const(constant, _) ->
                            match constant with
                            | SynConst.Unit -> "()"
                            | SynConst.Bool b -> sprintf "%b" b
                            | SynConst.Int32 n -> sprintf "%d" n
                            | SynConst.String(s, _, _) -> sprintf "\"%s\"" s
                            | _ -> sprintf "<%A>" constant
                        | SynExpr.Ident ident ->
                            sprintf "(Ident \"%s\")" ident.idText
                        | SynExpr.LongIdent(_, SynLongIdent(ids, _, _), _, _) ->
                            sprintf "(LongIdent \"%s\")" (ids |> List.map (fun id -> id.idText) |> String.concat ".")
                        | SynExpr.App(_, _, funcExpr, argExpr, _) ->
                            sprintf "(App %s %s)" (exprToParenthetical funcExpr) (exprToParenthetical argExpr)
                        | SynExpr.LetOrUse(_, isUse, bindings, body, _, _) ->
                            let kind = if isUse then "Use" else "Let"
                            sprintf "(%s ... %s)" kind (exprToParenthetical body)
                        | SynExpr.Match(_, expr, clauses, _, _) ->
                            sprintf "(Match %s %d_clauses)" (exprToParenthetical expr) clauses.Length
                        | SynExpr.Sequential(_, _, expr1, expr2, _, _) ->
                            sprintf "(Seq %s %s)" (exprToParenthetical expr1) (exprToParenthetical expr2)
                        | SynExpr.Tuple(_, exprs, _, _) ->
                            sprintf "(Tuple %d_elements)" exprs.Length
                        | _ ->
                            sprintf "(%s ...)" (expr.GetType().Name.Replace("SynExpr.", ""))
                    
                    let rec moduleToParenthetical (SynModuleOrNamespace(longId, _, kind, decls, _, _, _, range, _)) indent =
                        let moduleName = longId |> List.map (fun id -> id.idText) |> String.concat "."
                        let moduleRange = rangeToString range
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
                                
                                | SynModuleDecl.NestedModule(SynComponentInfo(_, _, _, longId, _, _, _, _), _, nestedDecls, _, range, _) ->
                                    let nestedName = longId |> List.map (fun id -> id.idText) |> String.concat "."
                                    sprintf "(Module \"%s\" :range %s)" nestedName (rangeToString range)
                                
                                | SynModuleDecl.Exception(SynExceptionDefn(SynExceptionDefnRepr(_, SynUnionCase(_, SynIdent(ident, _), _, _, _, _, _), _, _, _, _), _, _, _), range) ->
                                    sprintf "(Exception \"%s\" :range %s)" ident.idText (rangeToString range)
                                
                                | SynModuleDecl.Expr(expr, range) ->
                                    sprintf "(Expr %s :range %s)" (exprToParenthetical expr) (rangeToString range)
                                
                                | SynModuleDecl.Attributes(attrs, range) ->
                                    sprintf "(Attributes %d :range %s)" attrs.Length (rangeToString range)
                                
                                | SynModuleDecl.HashDirective(ParsedHashDirective(directive, _, _), range) ->
                                    sprintf "(HashDirective #%s :range %s)" directive (rangeToString range)
                                
                                | SynModuleDecl.ModuleAbbrev(ident, longId, range) ->
                                    sprintf "(ModuleAbbrev %s = %s :range %s)" ident.idText (longId |> List.map (fun id -> id.idText) |> String.concat ".") (rangeToString range)
                                
                                | SynModuleDecl.NamespaceFragment _ ->
                                    sprintf "(NamespaceFragment)"
                                | _ ->
                                    sprintf "(%s :range %s)" (decl.GetType().Name) (rangeToString decl.Range))
                            |> String.concat "\n  "
                        sprintf "(Module \"%s\" :range %s\n  %s)" moduleName moduleRange declsStr
                    
                    let initialAstPath = Path.Combine(config.IntermediatesDir.Value, $"{fileName}.initial.ast")
                    let initialAst = 
                        modules 
                        |> List.map (fun m -> moduleToParenthetical m 0)
                        |> String.concat "\n\n"
                    writeFileToPath initialAstPath initialAst
                    
                    // Write AST size report
                    printfn "  Wrote %s (%d bytes)" (Path.GetFileName detailPath) detailJson.Length
                    printfn "  Wrote %s (%d bytes)" (Path.GetFileName initialAstPath) initialAst.Length
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
            // Step 3: Extract symbol relationships
            printfn "[Pipeline] Extracting symbol relationships..."
            let relationships = extractRelationships projectResults.SymbolUses
            
            // Step 4: Build PSG (placeholder for now)
            printfn "[ASTTransformation] Building Program Semantic Graph..."
            if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                let psgData = {|
                    entryPoints = []
                    functions = 100  // Placeholder
                    modules = 6      // Placeholder
                    types = 50       // Placeholder
                |}
                let psgPath = Path.Combine(config.IntermediatesDir.Value, "semantic.psg.json")
                writeFileToPath psgPath (JsonSerializer.Serialize(psgData, jsonOptions))
            
            // Step 5: Generate MLIR (placeholder)
            printfn "[MLIRGeneration] Generating MLIR dialects..."
            if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                let mlirContent = "// Placeholder MLIR content\nmodule @main {\n  func.func @main() -> i32 {\n    %0 = arith.constant 0 : i32\n    return %0 : i32\n  }\n}"
                let mlirPath = Path.Combine(config.IntermediatesDir.Value, "output.mlir")
                writeFileToPath mlirPath mlirContent
            
            // Step 6: Generate LLVM IR (placeholder)
            printfn "[LLVMGeneration] Lowering MLIR to LLVM IR..."
            if config.OutputIntermediates && config.IntermediatesDir.IsSome then
                let llContent = "; Placeholder LLVM IR\ndefine i32 @main() {\n  ret i32 0\n}"
                let llPath = Path.Combine(config.IntermediatesDir.Value, "output.ll")
                writeFileToPath llPath llContent
            
            printfn "[Finalization] Compilation completed successfully"
            
            // Return results
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