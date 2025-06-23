module Dabbit.Analysis.ReachabilityAnalyzer

open FSharp.Compiler.Syntax
open FSharp.Compiler.Symbols

/// Statistics for a single module
type ModuleStats = {
    Module: string
    Total: int
    Retained: int
    Eliminated: int
}

/// Overall reachability statistics
type ReachabilityStats = {
    TotalSymbols: int
    ReachableSymbols: int
    EliminatedSymbols: int
    ModuleBreakdown: Map<string, ModuleStats>
}

/// Result of reachability analysis
type ReachabilityResult = {
    Reachable: Set<string>
    UnionCases: Map<string, Set<string>>
    Statistics: ReachabilityStats
}

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
    
    // SPECIFIC: Method call pattern (e.g., buffer.AsSpan(...))
    // MUST come BEFORE general App pattern
    | SynExpr.App(_, _, SynExpr.DotGet(target, _, SynLongIdent(ids, _, _), _), arg, _) ->
        let methodName = ids |> List.map (fun id -> id.idText) |> String.concat "."
        Set.unionMany [
            analyzeDependencies target
            analyzeDependencies arg
            Set.singleton (FunctionCall methodName)
        ]
    
    // GENERAL: Any other function application
    | SynExpr.App(_, _, func, arg, _) ->
        Set.union (analyzeDependencies func) (analyzeDependencies arg)
    
    | SynExpr.TypeApp(expr, _, _, _, _, _, _) ->
        analyzeDependencies expr
    
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
    
    | SynExpr.Lambda(_, _, _, body, _, _, _) ->
        analyzeDependencies body
    
    | SynExpr.Tuple(_, exprs, _, _) ->
        exprs |> List.map analyzeDependencies |> Set.unionMany
    
    | SynExpr.ArrayOrList(_, exprs, _) ->
        exprs |> List.map analyzeDependencies |> Set.unionMany
    
    | SynExpr.ArrayOrListComputed(_, expr, _) ->
        analyzeDependencies expr
    
    | SynExpr.Record(_, _, fields, _) ->
        fields 
        |> List.choose (fun (SynExprRecordField(_, _, expr, _)) -> expr)
        |> List.map analyzeDependencies
        |> Set.unionMany
    
    | SynExpr.New(_, _, expr, _) ->
        analyzeDependencies expr
    
    | SynExpr.ObjExpr(_, argOpt, _, bindings, members, _, _, _) ->
        let argDeps = argOpt |> Option.map (snd >> analyzeDependencies) |> Option.defaultValue Set.empty
        let bindingDeps = 
            bindings 
            |> List.map (fun (SynBinding(_, _, _, _, _, _, _, _, _, expr, _, _, _)) -> 
                analyzeDependencies expr)
            |> Set.unionMany
        let memberDeps =
            members 
            |> List.map (function
                | SynMemberDefn.Member(SynBinding(_, _, _, _, _, _, _, _, _, expr, _, _, _), _) -> 
                    analyzeDependencies expr
                | _ -> Set.empty)
            |> Set.unionMany
        Set.unionMany [argDeps; bindingDeps; memberDeps]
    
    | SynExpr.While(_, whileExpr, doExpr, _) ->
        Set.union (analyzeDependencies whileExpr) (analyzeDependencies doExpr)
    
    | SynExpr.For(_, _, _, _, toExpr, doExpr, _) ->
        Set.union (analyzeDependencies toExpr) (analyzeDependencies doExpr)
    
    | SynExpr.ForEach(_, _, _, _, enumExpr, bodyExpr, _) ->
        Set.union (analyzeDependencies enumExpr) (analyzeDependencies bodyExpr)
    
    | SynExpr.TryWith(tryExpr, withClauses, _, _, _, _) ->
        let tryDeps = analyzeDependencies tryExpr
        let withDeps = 
            withClauses
            |> List.map (fun (SynMatchClause(_, _, expr, _, _, _)) -> analyzeDependencies expr)
            |> Set.unionMany
        Set.union tryDeps withDeps
    
    | SynExpr.TryFinally(tryExpr, finallyExpr, _, _, _, _) ->
        Set.union (analyzeDependencies tryExpr) (analyzeDependencies finallyExpr)
    
    | SynExpr.TypeTest(expr, _, _) ->
        analyzeDependencies expr
    
    | SynExpr.Downcast(expr, _, _) ->
        analyzeDependencies expr
    
    | SynExpr.Upcast(expr, _, _) ->
        analyzeDependencies expr
    
    | SynExpr.AddressOf(_, expr, _, _) ->
        analyzeDependencies expr
    
    | SynExpr.InferredDowncast(expr, _) ->
        analyzeDependencies expr
    
    | SynExpr.InferredUpcast(expr, _) ->
        analyzeDependencies expr
    
    | SynExpr.DotGet(target, _, SynLongIdent(ids, _, _), _) ->
        let memberName = ids |> List.map (fun id -> id.idText) |> String.concat "."
        let targetDeps = analyzeDependencies target
        Set.add (FunctionCall memberName) targetDeps
    
    | SynExpr.DotSet(expr, _, rhsExpr, _) ->
        Set.union (analyzeDependencies expr) (analyzeDependencies rhsExpr)
    
    | SynExpr.DotIndexedGet(expr, indexArgs, _, _) ->
        let exprDeps = analyzeDependencies expr
        let indexDeps = indexArgs |> List.map analyzeDependencies |> Set.unionMany
        Set.union exprDeps indexDeps
    
    | SynExpr.DotIndexedSet(expr, indexArgs, valueExpr, _, _, _) ->
        let exprDeps = analyzeDependencies expr
        let indexDeps = indexArgs |> List.map analyzeDependencies |> Set.unionMany
        let valueDeps = analyzeDependencies valueExpr
        Set.unionMany [exprDeps; indexDeps; valueDeps]
    
    | SynExpr.NamedIndexedPropertySet(_, expr1, expr2, _) ->
        Set.union (analyzeDependencies expr1) (analyzeDependencies expr2)
    
    | SynExpr.DotNamedIndexedPropertySet(expr1, _, expr2, expr3, _) ->
        Set.unionMany [analyzeDependencies expr1; analyzeDependencies expr2; analyzeDependencies expr3]
    
    | SynExpr.JoinIn(expr1, _, expr2, _) ->
        Set.union (analyzeDependencies expr1) (analyzeDependencies expr2)
    
    | SynExpr.LetOrUseBang(_, _, _, _, rhsExpr, andBangs, bodyExpr, _, _) ->
        let rhsDeps = analyzeDependencies rhsExpr
        let andBangDeps = 
            andBangs 
            |> List.map (fun (SynExprAndBang(_, _, _, _, rhsExpr, _, _, _)) -> 
                analyzeDependencies rhsExpr)
            |> Set.unionMany
        let bodyDeps = analyzeDependencies bodyExpr
        Set.unionMany [rhsDeps; andBangDeps; bodyDeps]
    
    | SynExpr.DoBang(expr, _) ->
        analyzeDependencies expr
    
    | SynExpr.CompExpr(_, _, expr, _) ->
        analyzeDependencies expr
    
    | SynExpr.MatchBang(_, expr, clauses, _, _) ->
        let exprDeps = analyzeDependencies expr
        let clauseDeps = 
            clauses 
            |> List.map (fun (SynMatchClause(_, whenOpt, result, _, _, _)) ->
                let whenDeps = whenOpt |> Option.map analyzeDependencies |> Option.defaultValue Set.empty
                Set.union whenDeps (analyzeDependencies result))
            |> Set.unionMany
        Set.union exprDeps clauseDeps
    
    | SynExpr.YieldOrReturn(_, expr, _) ->
        analyzeDependencies expr
    
    | SynExpr.YieldOrReturnFrom(_, expr, _) ->
        analyzeDependencies expr
    
    | SynExpr.Paren(expr, _, _, _) ->
        analyzeDependencies expr
    
    | SynExpr.AnonRecd(_, copyInfo, fields, _, _) ->
        let copyDeps = copyInfo |> Option.map (snd >> analyzeDependencies) |> Option.defaultValue Set.empty
        let fieldDeps = fields |> List.map (snd >> snd >> analyzeDependencies) |> Set.unionMany
        Set.union copyDeps fieldDeps
    
    | SynExpr.Typed(expr, _, _) ->
        analyzeDependencies expr
    
    | SynExpr.Set(expr1, expr2, _) ->
        Set.union (analyzeDependencies expr1) (analyzeDependencies expr2)
    
    | SynExpr.Null _ 
    | SynExpr.Const _ 
    | SynExpr.Ident _ 
    | SynExpr.ImplicitZero _ 
    | SynExpr.MatchLambda _ 
    | SynExpr.Quote _ 
    | SynExpr.TypeApp _ 
    | SynExpr.ArbitraryAfterError _ 
    | SynExpr.FromParseError _ 
    | SynExpr.LibraryOnlyILAssembly _ 
    | SynExpr.LibraryOnlyStaticOptimization _ 
    | SynExpr.LibraryOnlyUnionCaseFieldGet _ 
    | SynExpr.LibraryOnlyUnionCaseFieldSet _ 
    | SynExpr.Lazy _ 
    | SynExpr.TraitCall _ 
    | SynExpr.Fixed _ 
    | SynExpr.Assert _ 
    | SynExpr.Do _ 
    | SynExpr.DiscardAfterMissingQualificationAfterDot _ ->
        Set.empty

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

/// Perform reachability analysis using worklist algorithm
let analyze (symbols: Map<string, FSharpSymbol>) (dependencies: Map<string, Set<string>>) (entryPoints: Set<string>) : ReachabilityResult =
    printfn "Starting reachability analysis with %d entry points" (Set.count entryPoints)
    
    let mutable reachable = Set.empty<string>
    let mutable worklist = entryPoints |> Set.toList
    let mutable reachableUnionCases = Map.empty<string, Set<string>>
    
    // Make sure we have at least one entry point
    if worklist.IsEmpty && not (Map.isEmpty symbols) then
        printfn "WARNING: No entry points found, using first declaration as seed"
        let firstDecl = symbols |> Map.toSeq |> Seq.head |> fst
        worklist <- [firstDecl]
    
    // Process worklist until empty
    let mutable iterations = 0
    while not worklist.IsEmpty do
        iterations <- iterations + 1
        
        match worklist with
        | current :: rest ->
            worklist <- rest
            
            if not (Set.contains current reachable) then
                reachable <- Set.add current reachable
                
                // Get dependencies of current item
                match Map.tryFind current dependencies with
                | Some deps ->
                    for dep in deps do
                        // Add all possible qualified versions
                        let possibleNames = [
                            dep
                            // Try common Alloy namespaces
                            "Alloy.Memory." + dep
                            "Alloy.IO." + dep
                            "Alloy.IO.Console." + dep
                            "Alloy.IO.String." + dep
                            // Try to find by suffix match
                            yield! symbols 
                                   |> Map.toSeq 
                                   |> Seq.choose (fun (k, _) -> 
                                       if k.EndsWith("." + dep) || k = dep then Some k 
                                       else None)
                        ]
                        
                        for name in possibleNames do
                            if Map.containsKey name symbols && 
                               not (Set.contains name reachable) && 
                               not (List.contains name worklist) then
                                worklist <- name :: worklist
                | None -> ()
        | [] -> ()
    
    printfn "Reachability analysis completed after %d iterations" iterations
    printfn "Found %d reachable symbols out of %d total" (Set.count reachable) (Map.count symbols)
    
    // Calculate statistics
    let stats = {
        TotalSymbols = Map.count symbols
        ReachableSymbols = Set.count reachable
        EliminatedSymbols = Map.count symbols - Set.count reachable
        ModuleBreakdown = Map.empty  // Could calculate if needed
    }
    
    {
        Reachable = reachable
        UnionCases = reachableUnionCases
        Statistics = stats
    }