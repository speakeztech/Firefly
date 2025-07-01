module Dabbit.Analysis.ReachabilityAnalyzer

open FSharp.Compiler.Syntax
open FSharp.Compiler.Symbols
open FSharp.Compiler.Text

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

/// Symbol information extracted from AST
type SymbolInfo = {
    FullName: string
    ModulePath: string list
    UnqualifiedName: string
    Kind: SymbolKind
}

and SymbolKind =
    | Function
    | Type
    | Method of parentType: string
    | Value

/// Resolution context built from AST
type ResolutionContext = {
    /// All symbols by their full names
    SymbolsByFullName: Map<string, SymbolInfo>
    /// Symbols grouped by module
    SymbolsByModule: Map<string list, Set<SymbolInfo>>
    /// Unqualified name to possible full names
    UnqualifiedToQualified: Map<string, Set<string>>
    /// Type members (for method resolution)
    TypeMembers: Map<string, Set<string>>
}

/// Module-level resolution state
type ModuleResolutionState = {
    CurrentModule: string list
    OpenedModules: string list list
}

/// Build symbol info from pattern in a binding
let extractSymbolFromPattern (modulePath: string list) (pat: SynPat) : SymbolInfo option =
    match pat with
    | SynPat.Named(SynIdent(ident, _), _, _, _) ->
        let fullName = (modulePath @ [ident.idText]) |> String.concat "."
        Some {
            FullName = fullName
            ModulePath = modulePath
            UnqualifiedName = ident.idText
            Kind = Function
        }
    | SynPat.LongIdent(SynLongIdent(ids, _, _), _, _, _, _, _) ->
        let names = ids |> List.map (fun id -> id.idText)
        let unqualified = List.last names
        let fullName = (modulePath @ names) |> String.concat "."
        Some {
            FullName = fullName
            ModulePath = modulePath
            UnqualifiedName = unqualified
            Kind = Function
        }
    | _ -> None

/// Extract all symbols from module declarations
let rec extractSymbolsFromDecl (modulePath: string list) (decl: SynModuleDecl) : SymbolInfo list =
    match decl with
    | SynModuleDecl.Let(_, bindings, _) ->
        bindings |> List.choose (fun (SynBinding(_, _, _, _, _, _, _, pat, _, _, _, _, _)) ->
            extractSymbolFromPattern modulePath pat)
    
    | SynModuleDecl.Types(typeDefs, _) ->
        typeDefs |> List.collect (fun (SynTypeDefn(componentInfo, typeRepr, members, _, _, _)) ->
            let (SynComponentInfo(_, _, _, longId, _, _, _, _)) = componentInfo
            let typeName = longId |> List.map (fun id -> id.idText) |> String.concat "."
            let typeFullName = (modulePath @ [typeName]) |> String.concat "."
            
            // Add the type itself
            let typeSymbol = {
                FullName = typeFullName
                ModulePath = modulePath
                UnqualifiedName = typeName
                Kind = Type
            }
            
            // Add type members
            let memberSymbols = 
                members |> List.choose (function
                    | SynMemberDefn.Member(SynBinding(_, _, _, _, _, _, _, pat, _, _, _, _, _), _) ->
                        match pat with
                        | SynPat.LongIdent(SynLongIdent(ids, _, _), _, _, _, _, _) ->
                            let memberName = ids |> List.map (fun id -> id.idText) |> List.last
                            Some {
                                FullName = typeFullName + "." + memberName
                                ModulePath = modulePath
                                UnqualifiedName = memberName
                                Kind = Method typeFullName
                            }
                        | _ -> None
                    | _ -> None)
            
            typeSymbol :: memberSymbols)
    
    | SynModuleDecl.NestedModule(componentInfo, _, nestedDecls, _, _, _) ->
        let (SynComponentInfo(_, _, _, longId, _, _, _, _)) = componentInfo
        let nestedName = longId |> List.map (fun id -> id.idText)
        let nestedPath = modulePath @ nestedName
        nestedDecls |> List.collect (extractSymbolsFromDecl nestedPath)
    
    | _ -> []

/// Extract symbols from a parsed input file
let extractSymbolsFromInput (input: ParsedInput) : SymbolInfo list =
    match input with
    | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
        modules |> List.collect (fun (SynModuleOrNamespace(longId, _, _, decls, _, _, _, _, _)) ->
            let modulePath = longId |> List.map (fun id -> id.idText)
            decls |> List.collect (extractSymbolsFromDecl modulePath))
    | _ -> []

/// Build resolution context from all parsed inputs
let buildResolutionContext (parsedInputs: (string * ParsedInput) list) : ResolutionContext =
    // Extract all symbols
    let allSymbols = 
        parsedInputs 
        |> List.collect (fun (_, input) -> extractSymbolsFromInput input)
    
    // Build maps
    let symbolsByFullName = 
        allSymbols 
        |> List.map (fun sym -> (sym.FullName, sym))
        |> Map.ofList
    
    let symbolsByModule =
        allSymbols
        |> List.groupBy (fun sym -> sym.ModulePath)
        |> List.map (fun (path, syms) -> (path, Set.ofList syms))
        |> Map.ofList
    
    let unqualifiedToQualified =
        allSymbols
        |> List.groupBy (fun sym -> sym.UnqualifiedName)
        |> List.map (fun (unqual, syms) -> 
            (unqual, syms |> List.map (fun s -> s.FullName) |> Set.ofList))
        |> Map.ofList
    
    let typeMembers =
        allSymbols
        |> List.choose (fun sym ->
            match sym.Kind with
            | Method parentType -> Some (parentType, sym.UnqualifiedName)
            | _ -> None)
        |> List.groupBy fst
        |> List.map (fun (typeName, members) -> 
            (typeName, members |> List.map snd |> Set.ofList))
        |> Map.ofList
    
    {
        SymbolsByFullName = symbolsByFullName
        SymbolsByModule = symbolsByModule
        UnqualifiedToQualified = unqualifiedToQualified
        TypeMembers = typeMembers
    }

/// Extract opened modules from declarations
let rec extractOpenedModules (decls: SynModuleDecl list) : string list list =
    decls |> List.choose (function
        | SynModuleDecl.Open(target, _) ->
            match target with
            | SynOpenDeclTarget.ModuleOrNamespace(SynLongIdent(ids, _, _), _) ->
                Some (ids |> List.map (fun id -> id.idText))
            | _ -> None
        | _ -> None)

/// Resolve a symbol name given current context
let resolveSymbol (ctx: ResolutionContext) (state: ModuleResolutionState) (name: string) : string option =
    // First try direct lookup
    if Map.containsKey name ctx.SymbolsByFullName then
        Some name
    else
        // Try with current module prefix
        let withCurrentModule = (state.CurrentModule @ [name]) |> String.concat "."
        if Map.containsKey withCurrentModule ctx.SymbolsByFullName then
            Some withCurrentModule
        else
            // Try with each opened module
            state.OpenedModules
            |> List.tryPick (fun openModule ->
                let qualified = (openModule @ [name]) |> String.concat "."
                if Map.containsKey qualified ctx.SymbolsByFullName then
                    Some qualified
                else None)

/// Analyze dependencies in an expression
let rec analyzeDependencies (ctx: ResolutionContext) (state: ModuleResolutionState) (expr: SynExpr) : Set<string> =
    match expr with
    // Simple identifiers - need resolution through opened modules
    | SynExpr.Ident ident ->
        match resolveSymbol ctx state ident.idText with
        | Some fullName -> Set.singleton fullName
        | None -> Set.empty
    
    // Qualified identifiers
    | SynExpr.LongIdent(_, SynLongIdent(ids, _, _), _, _) ->
        let name = ids |> List.map (fun id -> id.idText) |> String.concat "."
        match resolveSymbol ctx state name with
        | Some fullName -> Set.singleton fullName
        | None -> Set.empty
    
    // Generic type application: func<T>(args)
    | SynExpr.TypeApp(funcExpr, _, typeArgs, _, _, _, _) ->
        // The function itself needs to be resolved
        analyzeDependencies ctx state funcExpr
    
    // Property/method access: obj.Property or obj.Method
    | SynExpr.DotGet(targetExpr, _, SynLongIdent(memberIds, _, _), _) ->
        let targetDeps = analyzeDependencies ctx state targetExpr
        let memberName = memberIds |> List.map (fun id -> id.idText) |> String.concat "."
        
        // For now, just track the target
        // A more sophisticated version would resolve the member based on target type
        targetDeps
    
    // Method call with dot notation: obj.Method(arg)
    | SynExpr.App(_, _, SynExpr.DotGet(target, _, SynLongIdent(ids, _, _), _), arg, _) ->
        let targetDeps = analyzeDependencies ctx state target
        let methodName = ids |> List.map (fun id -> id.idText) |> String.concat "."
        let argDeps = analyzeDependencies ctx state arg
        
        Set.unionMany [targetDeps; argDeps]
    
    // Regular function application: f x
    | SynExpr.App(_, _, func, arg, _) ->
        Set.union (analyzeDependencies ctx state func) (analyzeDependencies ctx state arg)
    
    // Let bindings
    | SynExpr.LetOrUse(_, _, bindings, body, _, _) ->
        let bindingDeps = 
            bindings 
            |> List.map (fun (SynBinding(_, _, _, _, _, _, _, _, _, expr, _, _, _)) -> 
                analyzeDependencies ctx state expr)
            |> Set.unionMany
        let bodyDeps = analyzeDependencies ctx state body
        Set.union bindingDeps bodyDeps
    
    // Sequential expressions
    | SynExpr.Sequential(_, _, e1, e2, _, _) ->
        Set.union (analyzeDependencies ctx state e1) (analyzeDependencies ctx state e2)
    
    // If-then-else
    | SynExpr.IfThenElse(cond, thenExpr, elseOpt, _, _, _, _) ->
        [Some cond; Some thenExpr; elseOpt]
        |> List.choose id
        |> List.map (analyzeDependencies ctx state)
        |> Set.unionMany
    
    // Pattern matching
    | SynExpr.Match(_, matchExpr, clauses, _, _) ->
        let exprDeps = analyzeDependencies ctx state matchExpr
        let clauseDeps = 
            clauses 
            |> List.map (fun (SynMatchClause(_, whenOpt, result, _, _, _)) ->
                let whenDeps = whenOpt |> Option.map (analyzeDependencies ctx state) |> Option.defaultValue Set.empty
                Set.union whenDeps (analyzeDependencies ctx state result))
            |> Set.unionMany
        Set.union exprDeps clauseDeps
    
    // Lambda expressions
    | SynExpr.Lambda(_, _, _, body, _, _, _) ->
        analyzeDependencies ctx state body
    
    // Tuples
    | SynExpr.Tuple(_, exprs, _, _) ->
        exprs |> List.map (analyzeDependencies ctx state) |> Set.unionMany
    
    // Lists and arrays
    | SynExpr.ArrayOrList(_, exprs, _) ->
        exprs |> List.map (analyzeDependencies ctx state) |> Set.unionMany
    
    // Record construction
    | SynExpr.Record(_, _, fields, _) ->
        fields 
        |> List.choose (fun (SynExprRecordField(_, _, expr, _)) -> expr)
        |> List.map (analyzeDependencies ctx state)
        |> Set.unionMany
    
    // Object construction
    | SynExpr.New(_, _, expr, _) ->
        analyzeDependencies ctx state expr
    
    // Object expressions
    | SynExpr.ObjExpr(objType, argOptions, withKeyword, bindings, members, extraImpls, newExprRange, range) ->
        // Process bindings
        let bindingDeps = 
            bindings 
            |> List.map (fun (SynBinding(_, _, _, _, _, _, _, _, _, expr, _, _, _)) ->
                analyzeDependencies ctx state expr)
            |> Set.unionMany
        
        // Process members
        let memberDeps =
            members
            |> List.collect (fun member' ->
                match member' with
                | SynMemberDefn.Member(binding, _) ->
                    let (SynBinding(_, _, _, _, _, _, _, _, _, expr, _, _, _)) = binding
                    [analyzeDependencies ctx state expr]
                | _ -> [])
            |> Set.unionMany
        
        Set.union bindingDeps memberDeps
    
    // Try-with
    | SynExpr.TryWith(tryExpr, clauses, _, _, _, _) ->
        let tryDeps = analyzeDependencies ctx state tryExpr
        let clauseDeps = 
            clauses 
            |> List.map (fun (SynMatchClause(_, _, result, _, _, _)) ->
                analyzeDependencies ctx state result)
            |> Set.unionMany
        Set.union tryDeps clauseDeps
    
    // Try-finally
    | SynExpr.TryFinally(tryExpr, finallyExpr, _, _, _, _) ->
        Set.union (analyzeDependencies ctx state tryExpr) 
                  (analyzeDependencies ctx state finallyExpr)
    
    // Parentheses
    | SynExpr.Paren(expr, _, _, _) ->
        analyzeDependencies ctx state expr
    
    // Type test
    | SynExpr.TypeTest(expr, _, _) ->
        analyzeDependencies ctx state expr
    
    // Type cast
    | SynExpr.Downcast(expr, _, _)
    | SynExpr.Upcast(expr, _, _) ->
        analyzeDependencies ctx state expr
    
    // While loops
    | SynExpr.While(_, whileExpr, doExpr, _) ->
        Set.union (analyzeDependencies ctx state whileExpr) 
                  (analyzeDependencies ctx state doExpr)
    
    // For loops
    | SynExpr.For(_, _, _, _, _, _, toExpr, doExpr, _) ->
        let toDeps = analyzeDependencies ctx state toExpr
        let doDeps = analyzeDependencies ctx state doExpr
        Set.union toDeps doDeps

    // For-each loops
    | SynExpr.ForEach(spFor, spIn, seqExprOnly, isFromSource, pat, enumExpr, bodyExpr, range) ->
        Set.union (analyzeDependencies ctx state enumExpr) 
                (analyzeDependencies ctx state bodyExpr)
    
    // Lazy expressions
    | SynExpr.Lazy(expr, _) ->
        analyzeDependencies ctx state expr
    
    // Do expressions
    | SynExpr.Do(expr, _) ->
        analyzeDependencies ctx state expr
    
    // Assert expressions
    | SynExpr.Assert(expr, _) ->
        analyzeDependencies ctx state expr
    
    // Fixed expressions
    | SynExpr.Fixed(expr, _) ->
        analyzeDependencies ctx state expr
    
    // Address-of
    | SynExpr.AddressOf(_, expr, _, _) ->
        analyzeDependencies ctx state expr
    
    // Computation expressions
    | SynExpr.ComputationExpr(_, expr, _) ->
        analyzeDependencies ctx state expr
    
    // Yield/return
    | SynExpr.YieldOrReturn(_, expr, _, _)
    | SynExpr.YieldOrReturnFrom(_, expr, _, _) ->
        analyzeDependencies ctx state expr
    
    // Let/use bang (computation expressions)
    | SynExpr.LetOrUseBang(_, _, _, _, rhsExpr, andBangs, bodyExpr, _, _) ->
        let rhsDeps = analyzeDependencies ctx state rhsExpr
        let andBangDeps = 
            andBangs 
            |> List.map (fun (SynExprAndBang(_, _, _, _, expr, _, _)) -> 
                analyzeDependencies ctx state expr)
            |> Set.unionMany
        let bodyDeps = analyzeDependencies ctx state bodyExpr
        Set.unionMany [rhsDeps; andBangDeps; bodyDeps]
    
    // Match bang
    | SynExpr.MatchBang(_, expr, clauses, _, _) ->
        let exprDeps = analyzeDependencies ctx state expr
        let clauseDeps = 
            clauses 
            |> List.map (fun (SynMatchClause(_, _, result, _, _, _)) -> 
                analyzeDependencies ctx state result)
            |> Set.unionMany
        Set.union exprDeps clauseDeps
    
    // Do bang
    | SynExpr.DoBang(expr, _, _) ->
        analyzeDependencies ctx state expr
    
    // Set expressions
    | SynExpr.Set(targetExpr, rhsExpr, _) ->
        Set.union (analyzeDependencies ctx state targetExpr) 
                  (analyzeDependencies ctx state rhsExpr)
    
    // Dot-indexed get/set
    | SynExpr.DotIndexedGet(objectExpr, indexArgs, _, _) ->
        Set.union (analyzeDependencies ctx state objectExpr) 
                  (analyzeDependencies ctx state indexArgs)
    
    | SynExpr.DotIndexedSet(objectExpr, indexArgs, valueExpr, _, _, _) ->
        Set.unionMany [
            analyzeDependencies ctx state objectExpr
            analyzeDependencies ctx state indexArgs
            analyzeDependencies ctx state valueExpr
        ]
    
    // Dot set
    | SynExpr.DotSet(expr, _, rhsExpr, _) ->
        Set.union (analyzeDependencies ctx state expr) 
                  (analyzeDependencies ctx state rhsExpr)
    
    // Interpolated strings
    | SynExpr.InterpolatedString(parts, _, _) ->
        parts 
        |> List.choose (function 
            | SynInterpolatedStringPart.FillExpr(expr, _) -> Some expr
            | _ -> None)
        |> List.map (analyzeDependencies ctx state)
        |> Set.unionMany
    
    // Constants and terminals - no dependencies
    | SynExpr.Const _ 
    | SynExpr.Null _ 
    | SynExpr.ImplicitZero _
    | SynExpr.MatchLambda _
    | SynExpr.Quote _
    | SynExpr.AnonRecd _
    | SynExpr.LibraryOnlyILAssembly _
    | SynExpr.LibraryOnlyStaticOptimization _
    | SynExpr.LibraryOnlyUnionCaseFieldGet _
    | SynExpr.LibraryOnlyUnionCaseFieldSet _
    | SynExpr.ArbitraryAfterError _
    | SynExpr.FromParseError _
    | SynExpr.DiscardAfterMissingQualificationAfterDot _
    | SynExpr.DebugPoint _
    | SynExpr.Dynamic _ -> 
        Set.empty
    
    // Catch-all for any new patterns
    | _ -> Set.empty

/// Build complete dependency graph from parsed inputs
let buildDependencyGraph (parsedInputs: (string * ParsedInput) list) : (ResolutionContext * Map<string, Set<string>> * Set<string>) =
    let ctx = buildResolutionContext parsedInputs
    let mutable dependencies = Map.empty
    let mutable entryPoints = Set.empty
    
    // Process each input file
    for (_, input) in parsedInputs do
        match input with
        | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
            for (SynModuleOrNamespace(longId, _, _, decls, _, _, _, _, _)) in modules do
                let modulePath = longId |> List.map (fun id -> id.idText)
                let openedModules = extractOpenedModules decls
                let state = {
                    CurrentModule = modulePath
                    OpenedModules = openedModules
                }
                
                // Process declarations
                for decl in decls do
                    match decl with
                    | SynModuleDecl.Let(_, bindings, _) ->
                        for (SynBinding(_, _, _, _, attrs, _, _, pat, _, expr, _, _, _)) in bindings do
                            match extractSymbolFromPattern modulePath pat with
                            | Some symbol ->
                                // Check if entry point
                                let isEntryPoint = attrs |> List.exists (fun attrList ->
                                    attrList.Attributes |> List.exists (fun attr ->
                                        match attr.TypeName with
                                        | SynLongIdent([ident], _, _) -> ident.idText = "EntryPoint"
                                        | _ -> false))
                                
                                if isEntryPoint then
                                    entryPoints <- Set.add symbol.FullName entryPoints
                                
                                // Analyze dependencies
                                let deps = analyzeDependencies ctx state expr
                                dependencies <- Map.add symbol.FullName deps dependencies
                            | None -> ()
                    | _ -> ()
        | _ -> ()
    
    (ctx, dependencies, entryPoints)

/// Perform reachability analysis
let analyze (symbols: Map<string, obj>) (dependencies: Map<string, Set<string>>) (entryPoints: Set<string>) : ReachabilityResult =
    // Note: This function signature is kept for compatibility, but we'll use our own graph
    let mutable reachable = Set.empty
    let mutable worklist = entryPoints |> Set.toList
    let mutable iterations = 0
    
    while not worklist.IsEmpty && iterations < 1000 do
        iterations <- iterations + 1
        match worklist with
        | current :: rest ->
            worklist <- rest
            
            if not (Set.contains current reachable) then
                reachable <- Set.add current reachable
                
                match Map.tryFind current dependencies with
                | Some deps ->
                    for dep in deps do
                        if not (Set.contains dep reachable) && not (List.contains dep worklist) then
                            worklist <- dep :: worklist
                | None -> ()
        | [] -> ()
    
    let stats = {
        TotalSymbols = Map.count dependencies
        ReachableSymbols = Set.count reachable
        EliminatedSymbols = Map.count dependencies - Set.count reachable
        ModuleBreakdown = Map.empty
    }
    
    {
        Reachable = reachable
        UnionCases = Map.empty
        Statistics = stats
    }

/// Perform complete reachability analysis on parsed inputs
let analyzeFromParsedInputs (parsedInputs: (string * ParsedInput) list) : ReachabilityResult =
    let (ctx, dependencies, entryPoints) = buildDependencyGraph parsedInputs
    
    // Use our own reachability algorithm
    let mutable reachable = Set.empty
    let mutable worklist = entryPoints |> Set.toList
    
    printfn "Starting reachability analysis with %d entry points" (Set.count entryPoints)
    printfn "Total symbols found: %d" (Map.count ctx.SymbolsByFullName)
    
    while not worklist.IsEmpty do
        match worklist with
        | current :: rest ->
            worklist <- rest
            
            if not (Set.contains current reachable) then
                reachable <- Set.add current reachable
                
                match Map.tryFind current dependencies with
                | Some deps ->
                    for dep in deps do
                        if not (Set.contains dep reachable) && not (List.contains dep worklist) then
                            worklist <- dep :: worklist
                | None -> ()
        | [] -> ()
    
    printfn "Reachability complete: %d symbols reachable" (Set.count reachable)
    
    let stats = {
        TotalSymbols = Map.count ctx.SymbolsByFullName
        ReachableSymbols = Set.count reachable
        EliminatedSymbols = Map.count ctx.SymbolsByFullName - Set.count reachable
        ModuleBreakdown = Map.empty
    }
    
    {
        Reachable = reachable
        UnionCases = Map.empty
        Statistics = stats
    }