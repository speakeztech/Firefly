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

    | SynModuleDecl.Expr(expr, _) ->
        // Extract symbols from module initialization
        extractCompilerGeneratedSymbols modulePath expr
    
    | _ -> []

and extractCompilerGeneratedSymbols (modulePath: string list) expr =
    match expr with
    | SynExpr.App(_, _, funcExpr, _, _) ->
        match funcExpr with
        | SynExpr.Ident ident when ident.idText.StartsWith("op_") ->
            [{ FullName = (modulePath @ [ident.idText]) |> String.concat "."
               ModulePath = modulePath
               UnqualifiedName = ident.idText
               Kind = Function }]
        | _ -> []
    | SynExpr.Sequential(_, _, expr1, expr2, _, _) ->
        extractCompilerGeneratedSymbols modulePath expr1 @
        extractCompilerGeneratedSymbols modulePath expr2
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
    // Try direct lookup first
    if Map.containsKey name ctx.SymbolsByFullName then
        Some name
    // Handle dotted names (e.g., Array.zeroCreate)
    elif name.Contains(".") then
        let parts = name.Split('.')
        let typeName = parts.[0]
        let memberName = parts.[1]
        // Look in type members
        match Map.tryFind typeName ctx.TypeMembers with
        | Some members when Set.contains memberName members ->
            Some name
        | _ ->
            // Try with opened modules for the type
            state.OpenedModules
            |> List.tryPick (fun openModule ->
                let qualifiedType = (openModule @ [typeName]) |> String.concat "."
                match Map.tryFind qualifiedType ctx.TypeMembers with
                | Some members when Set.contains memberName members ->
                    Some (qualifiedType + "." + memberName)
                | _ -> None)
    else
        // Try current module, then opened modules
        [state.CurrentModule] @ state.OpenedModules
        |> List.tryPick (fun modulePath ->
            let qualified = (modulePath @ [name]) |> String.concat "."
            if Map.containsKey qualified ctx.SymbolsByFullName then
                Some qualified
            else None)

let resolveSymbolWithLogging ctx state name =
    let result = resolveSymbol ctx state name
    match result with
    | None -> 
        printfn "[RESOLVE] Failed to resolve '%s' in module %s with opens: %A" 
                name 
                (String.concat "." state.CurrentModule)
                state.OpenedModules
    | Some resolved ->
        printfn "[RESOLVE] '%s' -> '%s'" name resolved
    result

/// Analyze dependencies in an expression
let rec analyzeDependencies (ctx: ResolutionContext) (state: ModuleResolutionState) (expr: SynExpr) : Set<string> =
    match expr with
    | SynExpr.TypeApp(expr, _, types, _, _, _, _) ->
        // Handle generic instantiation
        analyzeDependencies ctx state expr

    // Function/method calls
    | SynExpr.App(_, _, funcExpr, argExpr, _) ->
        let funcDeps = analyzeDependencies ctx state funcExpr
        let argDeps = analyzeDependencies ctx state argExpr
        Set.union funcDeps argDeps
    
    // Identifier references (function names, variables, etc.)
    | SynExpr.Ident ident ->
        match resolveSymbolWithLogging ctx state ident.idText with  // <- Use here
        | Some resolved -> Set.singleton resolved
        | None -> Set.empty

    | SynExpr.LongIdent(_, longIdent, _, _) ->
        let (SynLongIdent(ids, _, _)) = longIdent
        let fullName = ids |> List.map (fun id -> id.idText) |> String.concat "."
        match resolveSymbolWithLogging ctx state fullName with  // <- And here
        | Some resolved -> Set.singleton resolved
        | None -> Set.empty
    
    // Long identifier with set (e.g., Module.value <- expr)
    | SynExpr.LongIdentSet(longIdent, expr, _) ->
        let (SynLongIdent(ids, _, _)) = longIdent
        let fullName = ids |> List.map (fun id -> id.idText) |> String.concat "."
        let identDeps = 
            match resolveSymbolWithLogging ctx state fullName with
            | Some resolved -> Set.singleton resolved
            | None -> Set.empty
        let exprDeps = analyzeDependencies ctx state expr
        Set.union identDeps exprDeps
    
    // Let bindings
    | SynExpr.LetOrUse(_, _, bindings, bodyExpr, _, _) ->
        let bindingDeps = 
            bindings 
            |> List.map (fun binding ->
                match binding with
                | SynBinding(_, _, _, _, _, _, _, _, _, expr, _, _, _) ->
                    analyzeDependencies ctx state expr)
            |> Set.unionMany
        let bodyDeps = analyzeDependencies ctx state bodyExpr
        Set.union bindingDeps bodyDeps
    
    // Sequential expressions
    | SynExpr.Sequential(_, _, expr1, expr2, _, _) ->
        let deps1 = analyzeDependencies ctx state expr1
        let deps2 = analyzeDependencies ctx state expr2
        Set.union deps1 deps2
    
    | SynExpr.DotGet(expr, _, longIdent, _) ->
    // Handle expr.Member patterns
        let exprDeps = analyzeDependencies ctx state expr
        let memberName = 
            match longIdent with
            | SynLongIdent(ids, _, _) -> 
                ids |> List.map (fun id -> id.idText) |> String.concat "."
        Set.add memberName exprDeps

    | SynExpr.New(_, typeExpr, argExpr, _) ->
        // Handle constructor calls
        let typeDeps = 
            match typeExpr with
            | SynType.LongIdent(SynLongIdent(ids, _, _)) ->
                let typeName = ids |> List.map (fun id -> id.idText) |> String.concat "."
                Set.singleton typeName
            | _ -> Set.empty
        let argDeps = analyzeDependencies ctx state argExpr
        Set.union typeDeps argDeps
    
    // If-then-else
    | SynExpr.IfThenElse(condExpr, thenExpr, elseExprOpt, _, _, _, _) ->
        let condDeps = analyzeDependencies ctx state condExpr
        let thenDeps = analyzeDependencies ctx state thenExpr
        let elseDeps = 
            match elseExprOpt with
            | Some elseExpr -> analyzeDependencies ctx state elseExpr
            | None -> Set.empty
        Set.unionMany [condDeps; thenDeps; elseDeps]
    
    // Match expressions
    | SynExpr.Match(_, expr, clauses, _, _) ->
        let exprDeps = analyzeDependencies ctx state expr
        let clauseDeps = 
            clauses 
            |> List.map (fun (SynMatchClause(_, _, expr, _, _, _)) ->
                analyzeDependencies ctx state expr)
            |> Set.unionMany
        Set.union exprDeps clauseDeps
    
    // For loops - FCS 43.9.300 signature
    | SynExpr.For(_, _, _, ident, fromExpr, _, toExpr, doExpr, _) ->
        let fromDeps = analyzeDependencies ctx state fromExpr
        let toDeps = analyzeDependencies ctx state toExpr
        let doDeps = analyzeDependencies ctx state doExpr
        Set.unionMany [fromDeps; toDeps; doDeps]
    
    // While loops
    | SynExpr.While(_, condExpr, doExpr, _) ->
        let condDeps = analyzeDependencies ctx state condExpr
        let doDeps = analyzeDependencies ctx state doExpr
        Set.union condDeps doDeps
    
    // Tuple expressions
    | SynExpr.Tuple(_, exprs, _, _) ->
        exprs 
        |> List.map (analyzeDependencies ctx state)
        |> Set.unionMany
    
    // Array/List expressions
    | SynExpr.ArrayOrList(_, exprs, _) ->
        exprs 
        |> List.map (analyzeDependencies ctx state)
        |> Set.unionMany
    
    // Array/List comprehensions
    | SynExpr.ArrayOrListComputed(_, expr, _) ->
        analyzeDependencies ctx state expr
    
    // Record construction
    | SynExpr.Record(_, _, fields, _) ->
        fields 
        |> List.choose (fun field ->
            match field with
            | SynExprRecordField(_, _, exprOpt, _) -> exprOpt)
        |> List.map (analyzeDependencies ctx state)
        |> Set.unionMany
    
    // Object expressions - FCS 43.9.300
    | SynExpr.ObjExpr(_, _, _, bindings, _, _, _, _) ->
        bindings 
        |> List.map (fun binding ->
            match binding with
            | SynBinding(_, _, _, _, _, _, _, _, _, expr, _, _, _) ->
                analyzeDependencies ctx state expr)
        |> Set.unionMany
    
    // Do expressions
    | SynExpr.Do(expr, _) ->
        analyzeDependencies ctx state expr
    
    // Parentheses
    | SynExpr.Paren(expr, _, _, _) ->
        analyzeDependencies ctx state expr
    
    // Typed expressions
    | SynExpr.Typed(expr, _, _) ->
        analyzeDependencies ctx state expr
    
    // Dot-indexed get (e.g., arr.[i])
    | SynExpr.DotIndexedGet(expr, indexExpr, _, _) ->
        let objDeps = analyzeDependencies ctx state expr
        let indexDeps = analyzeDependencies ctx state indexExpr
        Set.union objDeps indexDeps
    
    // Dot-indexed set (e.g., arr.[i] <- value)
    | SynExpr.DotIndexedSet(objExpr, indexExpr, valueExpr, _, _, _) ->
        let objDeps = analyzeDependencies ctx state objExpr
        let indexDeps = analyzeDependencies ctx state indexExpr
        let valueDeps = analyzeDependencies ctx state valueExpr
        Set.unionMany [objDeps; indexDeps; valueDeps]
    
    // Lambda expressions
    | SynExpr.Lambda(_, _, _, body, _, _, _) ->
        analyzeDependencies ctx state body
    
    // Try-with expressions - FCS 43.9.300
    | SynExpr.TryWith(tryExpr, withClauses, _, _, _, _) ->
        let tryDeps = analyzeDependencies ctx state tryExpr
        let withDeps = 
            withClauses 
            |> List.map (fun clause ->
                match clause with
                | SynMatchClause(_, _, expr, _, _, _) ->
                    analyzeDependencies ctx state expr)
            |> Set.unionMany
        Set.union tryDeps withDeps
    
    // Try-finally expressions
    | SynExpr.TryFinally(tryExpr, finallyExpr, _, _, _, _) ->
        let tryDeps = analyzeDependencies ctx state tryExpr
        let finallyDeps = analyzeDependencies ctx state finallyExpr
        Set.union tryDeps finallyDeps
    
    // Constants and literals
    | SynExpr.Const(_, _) ->
        Set.empty
    
    // Lazy expressions
    | SynExpr.Lazy(expr, _) ->
        analyzeDependencies ctx state expr
    
    // Quote expressions
    | SynExpr.Quote(_, _, expr, _, _) ->
        analyzeDependencies ctx state expr
    
    // Downcast
    | SynExpr.Downcast(expr, _, _) ->
        analyzeDependencies ctx state expr
    
    // Upcast
    | SynExpr.Upcast(expr, _, _) ->
        analyzeDependencies ctx state expr
    
    // AddressOf
    | SynExpr.AddressOf(_, expr, _, _) ->
        analyzeDependencies ctx state expr
    
    // InferredDowncast
    | SynExpr.InferredDowncast(expr, _) ->
        analyzeDependencies ctx state expr
    
    // InferredUpcast
    | SynExpr.InferredUpcast(expr, _) ->
        analyzeDependencies ctx state expr
    
    // Other cases we don't need to analyze
    | _ ->
        Set.empty

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