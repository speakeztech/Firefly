module Dabbit.Analysis.ReachabilityAnalyzer

open FSharp.Compiler.Syntax
open FSharp.Compiler.Symbols

/// Enhanced dependency tracking
type Dependency =
    | FunctionCall of string
    | TypeReference of string  
    | ModuleOpen of string
    | ValueBinding of string

/// Analyze expression to extract ALL dependencies
let rec analyzeDependencies (expr: SynExpr) : Set<Dependency> =
    match expr with
    | SynExpr.Ident ident ->
        Set.singleton (FunctionCall ident.idText)
    
    | SynExpr.LongIdent(_, SynLongIdent(ids, _, _), _, _) ->
        let fullName = ids |> List.map (fun id -> id.idText) |> String.concat "."
        Set.singleton (FunctionCall fullName)
    
    | SynExpr.App(_, _, func, arg, _) ->
        Set.union (analyzeDependencies func) (analyzeDependencies arg)
    
    | SynExpr.LetOrUse(_, _, bindings, body, _, _) ->
        let bindingDeps = 
            bindings 
            |> List.map (fun (SynBinding(_, _, _, _, _, _, _, _, _, expr, _, _, _)) -> 
                analyzeDependencies expr)
            |> Set.unionMany
        Set.union bindingDeps (analyzeDependencies body)
    
    | SynExpr.Sequential(_, _, e1, e2, _, _) ->
        Set.union (analyzeDependencies e1) (analyzeDependencies e2)
    
    | SynExpr.IfThenElse(cond, thenExpr, elseOpt, _, _, _, _) ->
        [Some cond; Some thenExpr; elseOpt]
        |> List.choose id
        |> List.map analyzeDependencies
        |> Set.unionMany
    
    | SynExpr.Match(_, matchExpr, clauses, _, _) ->
        let exprDeps = analyzeDependencies matchExpr
        let clauseDeps = 
            clauses 
            |> List.map (fun (SynMatchClause(_, whenOpt, result, _, _, _)) ->
                let whenDeps = whenOpt |> Option.map analyzeDependencies |> Option.defaultValue Set.empty
                Set.union whenDeps (analyzeDependencies result))
            |> Set.unionMany
        Set.union exprDeps clauseDeps
    
    | SynExpr.TypeApp(expr, _, _, _, _, _, _) ->
        analyzeDependencies expr
        
    | SynExpr.DotGet(expr, _, _, _) ->
        analyzeDependencies expr
        
    | SynExpr.Lambda(_, _, _, body, _, _, _) ->
        analyzeDependencies body
        
    | SynExpr.TryWith(tryExpr, withClauses, _, _, _, _) ->
        let tryDeps = analyzeDependencies tryExpr
        let withDeps = 
            withClauses
            |> List.map (fun (SynMatchClause(_, _, expr, _, _, _)) -> analyzeDependencies expr)
            |> Set.unionMany
        Set.union tryDeps withDeps
    
    | _ -> Set.empty

/// Extract function body from parsed AST
let extractFunctionBody (allInputs: (string * ParsedInput) list) (functionName: string) : SynExpr option =
    let rec findInModule (moduleName: string) (decls: SynModuleDecl list) =
        decls |> List.tryPick (function
            | SynModuleDecl.Let(_, bindings, _) ->
                bindings |> List.tryPick (fun (SynBinding(_, _, _, _, _, _, _, pat, _, expr, _, _, _)) ->
                    match pat with
                    | SynPat.Named(SynIdent(ident, _), _, _, _) when 
                        sprintf "%s.%s" moduleName ident.idText = functionName ||
                        ident.idText = functionName ->
                        Some expr
                    | SynPat.LongIdent(SynLongIdent(ids, _, _), _, _, _, _, _) ->
                        let name = ids |> List.map (fun id -> id.idText) |> String.concat "."
                        if sprintf "%s.%s" moduleName name = functionName || name = functionName then
                            Some expr
                        else None
                    | _ -> None)
            | _ -> None)
    
    allInputs |> List.tryPick (fun (_, input) ->
        match input with
        | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
            modules |> List.tryPick (fun (SynModuleOrNamespace(longId, _, _, decls, _, _, _, _, _)) ->
                let moduleName = longId |> List.map (fun id -> id.idText) |> String.concat "."
                findInModule moduleName decls)
        | _ -> None)

/// Perform deep reachability analysis
let analyzeReachability (allInputs: (string * ParsedInput) list) (entryPoints: Set<string>) =
    let mutable reachable = Set.empty
    let mutable analyzed = Set.empty
    let mutable worklist = entryPoints |> Set.toList
    
    printfn "Starting deep reachability analysis with %d entry points" (List.length worklist)
    
    while not worklist.IsEmpty do
        match worklist with
        | current :: rest ->
            worklist <- rest
            
            if not (Set.contains current analyzed) then
                analyzed <- Set.add current analyzed
                reachable <- Set.add current reachable
                
                // Extract and analyze function body
                match extractFunctionBody allInputs current with
                | Some body ->
                    printfn "  Analyzing body of %s" current
                    let deps = analyzeDependencies body
                    
                    for dep in deps do
                        match dep with
                        | FunctionCall callee ->
                            // Try to resolve qualified name
                            let qualifiedNames = [
                                callee
                                "Alloy.Memory." + callee
                                "Alloy.IO." + callee
                                "Alloy.IO.Console." + callee
                                "Alloy.IO.String." + callee
                            ]
                            
                            for qname in qualifiedNames do
                                if not (Set.contains qname analyzed) then
                                    printfn "    Found dependency: %s" qname
                                    worklist <- qname :: worklist
                                    
                        | _ -> ()
                | None ->
                    if not (current.StartsWith("FSharp.") || current.StartsWith("System.")) then
                        printfn "  WARNING: No body found for %s" current
        | [] -> ()
    
    printfn "Reachability complete: %d symbols reachable" (Set.count reachable)
    reachable

/// Resolve module opens to find used symbols
let resolveModuleOpens (opens: string list) (usedSymbols: Set<string>) (moduleSymbols: Map<string, string list>) =
    let mutable resolved = Set.empty
    
    for openModule in opens do
        match Map.tryFind openModule moduleSymbols with
        | Some symbols ->
            for symbol in symbols do
                // Check if any used symbol might be from this module
                if usedSymbols |> Set.exists (fun used -> 
                    used = symbol || 
                    used.EndsWith("." + symbol) ||
                    symbol.EndsWith("." + used)) then
                    resolved <- Set.add (openModule + "." + symbol) resolved
        | None -> ()
    
    resolved

/// Build complete dependency graph with bodies
let buildDependencyGraph (allInputs: (string * ParsedInput) list) =
    let mutable allSymbols = Map.empty<string, SynExpr option>
    let mutable dependencies = Map.empty<string, Set<string>>
    let mutable entryPoints = Set.empty
    
    // First pass: collect all symbols and their bodies
    for (filePath, input) in allInputs do
        match input with
        | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
            for (SynModuleOrNamespace(longId, _, _, decls, _, _, _, _, _)) in modules do
                let moduleName = longId |> List.map (fun id -> id.idText) |> String.concat "."
                
                for decl in decls do
                    match decl with
                    | SynModuleDecl.Let(_, bindings, _) ->
                        for (SynBinding(_, _, _, _, attrs, _, _, pat, _, expr, _, _, _)) in bindings do
                            let symbolName = 
                                match pat with
                                | SynPat.Named(SynIdent(ident, _), _, _, _) ->
                                    Some (sprintf "%s.%s" moduleName ident.idText)
                                | _ -> None
                            
                            match symbolName with
                            | Some name ->
                                allSymbols <- Map.add name (Some expr) allSymbols
                                
                                // Check for entry point
                                let isEntryPoint = attrs |> List.exists (fun attrList ->
                                    attrList.Attributes |> List.exists (fun attr ->
                                        match attr.TypeName with
                                        | SynLongIdent([ident], _, _) -> ident.idText = "EntryPoint"
                                        | _ -> false))
                                
                                if isEntryPoint then
                                    entryPoints <- Set.add name entryPoints
                            | None -> ()
                    | _ -> ()
        | _ -> ()
    
    // Second pass: build dependency edges
    for KeyValue(symbolName, bodyOpt) in allSymbols do
        match bodyOpt with
        | Some body ->
            let deps = analyzeDependencies body
            let depNames = 
                deps 
                |> Set.toList
                |> List.choose (function FunctionCall name -> Some name | _ -> None)
                |> Set.ofList
            dependencies <- Map.add symbolName depNames dependencies
        | None -> ()
    
    (allSymbols, dependencies, entryPoints)

/// Validate reachability results
let validateReachability (reachable: Set<string>) (allCalls: Set<string>) =
    let undefined = Set.difference allCalls reachable
    
    if not (Set.isEmpty undefined) then
        printfn "ERROR: Reachability validation failed!"
        printfn "Undefined symbols that are called but not reachable:"
        for symbol in undefined do
            printfn "  - %s" symbol
        false
    else
        true