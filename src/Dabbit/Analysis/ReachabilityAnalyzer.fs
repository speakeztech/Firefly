module Dabbit.Analysis.ReachabilityAnalyzer

open FSharp.Compiler.Syntax
open FSharp.Compiler.Text
open Core.XParsec.Foundation
open Dabbit.Analysis.ScopeManager
open Dabbit.Analysis.ReachabilityDiagnostics
open Dabbit.Bindings.PatternLibrary

/// Symbol name utilities
module SymbolNames =
    /// Combine module path with symbol name
    let qualify (modulePath: string list) (name: string) =
        if List.isEmpty modulePath then name
        else (modulePath @ [name]) |> String.concat "."
    
    /// Get qualified name from longIdent  
    let fromLongIdent (longIdent: SynLongIdent) =
        let (SynLongIdent(ids, _, _)) = longIdent
        ids |> List.map (fun id -> id.idText) |> String.concat "."

/// Module-level resolution state
type ModuleResolutionState = {
    CurrentModule: string list
    OpenedModules: string list list
}

/// Resolution context containing all symbols
type ResolutionContext = {
    SymbolsByFullName: Map<string, Set<string>>
    TypeMembers: Map<string, Set<string>>
}

/// Reachability analysis result
type ReachabilityResult = {
    Reachable: Set<string>
    Dependencies: Map<string, Set<string>>
    EntryPoints: Set<string>
    Statistics: ReachabilityStatistics
}

and ReachabilityStatistics = {
    TotalSymbols: int
    ReachableSymbols: int
    EliminatedSymbols: int
    DependencyEdges: int
}

/// Extract opened modules from module declarations
let extractOpenedModules (decls: SynModuleDecl list) : string list list =
    decls |> List.choose (fun decl ->
        match decl with
        | SynModuleDecl.Open(target, _) ->
            match target with
            | SynOpenDeclTarget.ModuleOrNamespace(longIdent, _) ->
                let (SynLongIdent(ids, _, _)) = longIdent
                Some (ids |> List.map (fun id -> id.idText))
            | _ -> None
        | _ -> None)

/// Extract module path from parsed input
let extractModulePath (input: ParsedInput) : string list =
    match input with
    | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
        match modules with
        | SynModuleOrNamespace(longId, _, _, _, _, _, _, _, _) :: _ ->
            longId |> List.map (fun id -> id.idText)
        | [] -> []
    | _ -> []

/// Extract symbol from pattern
let rec extractSymbolFromPattern (modulePath: string list) (pat: SynPat) : string option =
    match pat with
    | SynPat.Named(SynIdent(ident, _), _, _, _) ->
        Some (SymbolNames.qualify modulePath ident.idText)
    | SynPat.LongIdent(longIdent, _, _, _, _, _) ->
        let (SynLongIdent(ids, _, _)) = longIdent
        Some (SymbolNames.qualify modulePath (ids |> List.map (fun id -> id.idText) |> String.concat "."))
    | SynPat.Paren(pat, _) ->
        extractSymbolFromPattern modulePath pat
    | SynPat.Typed(pat, _, _) ->
        extractSymbolFromPattern modulePath pat
    | _ -> None

/// Extract member name from pattern
let extractMemberName (pat: SynPat) : string option =
    match pat with
    | SynPat.LongIdent(SynLongIdent([_; memberName], _, _), _, _, _, _, _) ->
        Some memberName.idText
    | _ -> None

/// Extract all symbols from a module declaration
let rec extractSymbolsFromDecl (modulePath: string list) (decl: SynModuleDecl) : string list =
    match decl with
    | SynModuleDecl.Let(_, bindings, _) ->
        bindings |> List.choose (fun (SynBinding(_, _, _, _, _, _, _, pat, _, _, _, _, _)) ->
            extractSymbolFromPattern modulePath pat)
    
    | SynModuleDecl.Types(types, _) ->
        types |> List.collect (fun (SynTypeDefn(SynComponentInfo(_, _, _, longId, _, _, _, _), _, memberDefns, _, _, _)) ->
            let typeName = (modulePath @ [longId.Head.idText]) |> String.concat "."
            typeName :: 
            (memberDefns |> List.choose (fun memberDefn ->
                match memberDefn with
                | SynMemberDefn.Member(SynBinding(_, _, _, _, _, _, _, pat, _, _, _, _, _), _) ->
                    match extractMemberName pat with
                    | Some memberName -> Some (typeName + "." + memberName)
                    | None -> None
                | _ -> None)))
    
    | SynModuleDecl.NestedModule(SynComponentInfo(_, _, _, longId, _, _, _, _), _, nestedDecls, _, _, _) ->
        let nestedPath = modulePath @ (longId |> List.map (fun id -> id.idText))
        nestedDecls |> List.collect (extractSymbolsFromDecl nestedPath)
    
    | _ -> []

/// Extract simple pattern list from SynSimplePats
let extractSimplePatterns (pats: SynSimplePats) : SynSimplePat list =
    match pats with
    | SynSimplePats.SimplePats(patterns, _, _) -> patterns  // Handle the main case

/// Build resolution context from parsed inputs
let buildResolutionContext (parsedInputs: (string * ParsedInput) list) : ResolutionContext =
    let mutable symbols = Map.empty
    let mutable typeMembers = Map.empty
    
    // First, register all F# core and Alloy pattern library symbols
    let corePatterns = 
        allPatterns 
        |> List.groupBy (fun pattern -> 
            let parts = pattern.QualifiedName.Split('.')
            if parts.Length > 1 then
                parts.[0..parts.Length-2] |> String.concat "."
            else
                "FSharp.Core")
        |> List.iter (fun (namespace', patterns) ->
            let names = patterns |> List.map (fun p -> 
                let parts = p.QualifiedName.Split('.')
                parts.[parts.Length-1]) |> Set.ofList
            symbols <- Map.add namespace' names symbols)
    
    // Add F# intrinsic functions to FSharp.Core
    let intrinsics = Set.ofList ["not"; "failwith"; "sprintf"; "printf"; "printfn"]
    symbols <- 
        match Map.tryFind "Microsoft.FSharp.Core.Operators" symbols with
        | Some existing -> Map.add "Microsoft.FSharp.Core.Operators" (Set.union existing intrinsics) symbols
        | None -> Map.add "Microsoft.FSharp.Core.Operators" intrinsics symbols
    
    // Also add to ExtraTopLevelOperators
    symbols <- Map.add "Microsoft.FSharp.Core.ExtraTopLevelOperators" (Set.ofList ["sprintf"; "printf"; "printfn"]) symbols
    
    for (_, input) in parsedInputs do
        match input with
        | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
            for (SynModuleOrNamespace(longId, _, _, decls, _, _, _, _, _)) in modules do
                let modulePath = longId |> List.map (fun id -> id.idText)
                let moduleName = String.concat "." modulePath
                let moduleSymbols = ref Set.empty
                
                for decl in decls do
                    match decl with
                    | SynModuleDecl.Let(_, bindings, _) ->
                        for binding in bindings do
                            match binding with
                            | SynBinding(_, _, _, _, _, _, _, pat, _, _, _, _, _) ->
                                match pat with
                                | SynPat.Named(SynIdent(ident, _), _, _, _) ->
                                    moduleSymbols := Set.add ident.idText !moduleSymbols
                                | SynPat.LongIdent(SynLongIdent(ids, _, _), _, _, _, _, _) ->
                                    let name = ids |> List.last |> fun id -> id.idText
                                    moduleSymbols := Set.add name !moduleSymbols
                                | _ -> ()
                    
                    | SynModuleDecl.Types(types, _) ->
                        for (SynTypeDefn(SynComponentInfo(_, _, _, longId, _, _, _, _), _, memberDefns, _, _, _)) in types do
                            let typeName = longId.Head.idText
                            moduleSymbols := Set.add typeName !moduleSymbols
                            
                            let fullTypeName = (modulePath @ [typeName]) |> String.concat "."
                            let memberSet = ref Set.empty
                            
                            for memberDefn in memberDefns do
                                match memberDefn with
                                | SynMemberDefn.Member(SynBinding(_, _, _, _, _, _, _, pat, _, _, _, _, _), _) ->
                                    match extractMemberName pat with
                                    | Some memberName ->
                                        memberSet := Set.add memberName !memberSet
                                    | None -> ()
                                | _ -> ()
                            
                            if not (Set.isEmpty !memberSet) then
                                typeMembers <- Map.add fullTypeName !memberSet typeMembers
                    
                    | SynModuleDecl.NestedModule(SynComponentInfo(_, _, _, longId, _, _, _, _), _, nestedDecls, _, _, _) ->
                        // Handle nested modules
                        let nestedPath = modulePath @ (longId |> List.map (fun id -> id.idText))
                        let nestedName = String.concat "." nestedPath
                        let nestedSymbols = ref Set.empty
                        
                        for nestedDecl in nestedDecls do
                            match nestedDecl with
                            | SynModuleDecl.Let(_, bindings, _) ->
                                for binding in bindings do
                                    match binding with
                                    | SynBinding(_, _, _, _, _, _, _, pat, _, _, _, _, _) ->
                                        match pat with
                                        | SynPat.Named(SynIdent(ident, _), _, _, _) ->
                                            nestedSymbols := Set.add ident.idText !nestedSymbols
                                        | _ -> ()
                            | _ -> ()
                        
                        if not (Set.isEmpty !nestedSymbols) then
                            symbols <- Map.add nestedName !nestedSymbols symbols
                    
                    | _ -> ()
                
                if not (Set.isEmpty !moduleSymbols) then
                    symbols <- Map.add moduleName !moduleSymbols symbols
        | _ -> ()
    
    // Add known type members for builtin types
    typeMembers <- Map.add "voption" (Set.ofList ["hasValue"; "value"]) typeMembers
    
    { SymbolsByFullName = symbols; TypeMembers = typeMembers }

/// Try to resolve a symbol name
let resolveSymbol (ctx: ResolutionContext) (state: ModuleResolutionState) (name: string) : string option =
    // First check if it's a fully qualified name already
    if name.Contains(".") && not (name.StartsWith("this.")) then
        let parts = name.Split('.')
        if parts.Length = 2 then
            // Could be Type.member or Module.function
            let typeName = parts.[0]
            let memberName = parts.[1]
            
            // Check if it's a type member
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
            // Multi-part name, check if it exists as-is
            ctx.SymbolsByFullName 
            |> Map.exists (fun _ symbols -> Set.contains name symbols)
            |> function true -> Some name | false -> None
    else
        // Simple name - try current module first, then opened modules
        let searchPaths = state.CurrentModule :: state.OpenedModules
        
        searchPaths
        |> List.tryPick (fun modulePath ->
            let modulePathStr = String.concat "." modulePath
            
            // Check if this module has the symbol
            match Map.tryFind modulePathStr ctx.SymbolsByFullName with
            | Some symbols when Set.contains name symbols ->
                // Found as unqualified in this module
                Some name
            | Some symbols ->
                // Check if symbol exists with full qualification
                let qualified = 
                    if modulePathStr = "" then name
                    else modulePathStr + "." + name
                if Set.contains qualified symbols then
                    Some qualified
                else
                    None
            | None ->
                // Module not found, try constructing the qualified name anyway
                let qualified = 
                    if modulePath = [] then name
                    else (modulePath @ [name]) |> String.concat "."
                
                // Check if this qualified name exists in any module
                ctx.SymbolsByFullName 
                |> Map.exists (fun _ symbols -> Set.contains qualified symbols)
                |> function true -> Some qualified | false -> None
        )

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

/// Analyze dependencies in an expression with scope awareness
let rec analyzeDependencies (ctx: ResolutionContext) (state: ModuleResolutionState) (scopeMgr: ScopeManager.ScopeManager) (expr: SynExpr) : Set<string> * ScopeManager.ScopeManager =
    match expr with
    | SynExpr.TypeApp(expr, _, types, _, _, _, _) ->
        // Handle generic instantiation
        analyzeDependencies ctx state scopeMgr expr

    // Function/method calls
    | SynExpr.App(_, _, funcExpr, argExpr, _) ->
        let funcDeps, scopeMgr1 = analyzeDependencies ctx state scopeMgr funcExpr
        let argDeps, scopeMgr2 = analyzeDependencies ctx state scopeMgr1 argExpr
        Set.union funcDeps argDeps, scopeMgr2
    
    // Identifier references (function names, variables, etc.)
    | SynExpr.Ident ident ->
        // First check if it's a local binding
        if ScopeManager.isLocalBinding ident.idText scopeMgr then
            // Log this as a local binding resolution
            match ReachabilityDiagnostics.getDiagnostics() with
            | diag -> 
                diag.LogSymbolLookup(ident.idText, state.OpenedModules, 
                    ResolutionResult.ResolvedLocal (ScopeManager.BindingKind.LocalBinding), 
                    Some ident.idRange)
            Set.empty, scopeMgr
        else
            // Try to resolve as a module-level symbol
            match resolveSymbolWithLogging ctx state ident.idText with
            | Some resolved -> Set.singleton resolved, scopeMgr
            | None -> Set.empty, scopeMgr

    | SynExpr.LongIdent(_, longIdent, _, _) ->
        let (SynLongIdent(ids, _, _)) = longIdent
        let fullName = ids |> List.map (fun id -> id.idText) |> String.concat "."
        
        // Check if it's a member access on a local variable
        if fullName.Contains(".") then
            let parts = fullName.Split('.')
            let objName = parts.[0]
            
            if ScopeManager.isLocalBinding objName scopeMgr then
                // This is a member access on a local variable
                match ReachabilityDiagnostics.getDiagnostics() with
                | diag -> 
                    diag.LogSymbolLookup(fullName, state.OpenedModules, 
                        ResolutionResult.ResolvedLocal (ScopeManager.BindingKind.LocalBinding), 
                        None)
                // Return the member function as a dependency if we can resolve it
                let memberName = parts.[1]
                // Try to resolve the member based on type information
                match Map.tryFind objName scopeMgr.TypeMembers with
                | Some members when Set.contains memberName members ->
                    Set.singleton (objName + "." + memberName), scopeMgr
                | _ -> Set.empty, scopeMgr
            else
                match resolveSymbolWithLogging ctx state fullName with
                | Some resolved -> Set.singleton resolved, scopeMgr
                | None -> Set.empty, scopeMgr
        else
            match resolveSymbolWithLogging ctx state fullName with
            | Some resolved -> Set.singleton resolved, scopeMgr
            | None -> Set.empty, scopeMgr
    
    // Long identifier with set (e.g., Module.value <- expr)
    | SynExpr.LongIdentSet(longIdent, expr, _) ->
        let (SynLongIdent(ids, _, _)) = longIdent
        let fullName = ids |> List.map (fun id -> id.idText) |> String.concat "."
        let identDeps = 
            match resolveSymbolWithLogging ctx state fullName with
            | Some resolved -> Set.singleton resolved
            | None -> Set.empty
        let exprDeps, scopeMgr' = analyzeDependencies ctx state scopeMgr expr
        Set.union identDeps exprDeps, scopeMgr'
    
    // Let bindings - add bindings to scope
    | SynExpr.LetOrUse(_, _, bindings, bodyExpr, _, _) ->
        // Enter a new let scope
        let scopeMgr1 = ScopeManager.pushScope ScopeManager.ScopeKind.LetScope scopeMgr
        
        // Process bindings and add them to scope
        let bindingDeps, scopeMgr2 = 
            bindings 
            |> List.fold (fun (deps, mgr) binding ->
                match binding with
                | SynBinding(_, _, _, _, _, _, _, pat, _, expr, _, _, _) ->
                    // Extract bindings from pattern
                    let bindings = ScopeManager.extractPatternBindings pat
                    let mgr' = ScopeManager.addBindings bindings mgr
                    // Analyze the expression
                    let exprDeps, mgr'' = analyzeDependencies ctx state mgr' expr
                    Set.union deps exprDeps, mgr''
            ) (Set.empty, scopeMgr1)
        
        // Analyze body with bindings in scope
        let bodyDeps, scopeMgr3 = analyzeDependencies ctx state scopeMgr2 bodyExpr
        
        // Exit the let scope
        let scopeMgr4 = ScopeManager.popScope scopeMgr3
        
        Set.union bindingDeps bodyDeps, scopeMgr4
    
    // Sequential expressions - thread scope through
    | SynExpr.Sequential(_, _, expr1, expr2, _, _) ->
        let deps1, scopeMgr1 = analyzeDependencies ctx state scopeMgr expr1
        let deps2, scopeMgr2 = analyzeDependencies ctx state scopeMgr1 expr2
        Set.union deps1 deps2, scopeMgr2
    
    | SynExpr.DotGet(expr, _, longIdent, _) ->
        // Handle expr.Member patterns
        let exprDeps, scopeMgr' = analyzeDependencies ctx state scopeMgr expr
        let memberName = 
            match longIdent with
            | SynLongIdent(ids, _, _) -> 
                ids |> List.map (fun id -> id.idText) |> String.concat "."
        
        // Check if this is a member access on 'this' or a known type
        match expr with
        | SynExpr.Ident ident when ident.idText = "this" ->
            // Member access on 'this' is not an external dependency
            exprDeps, scopeMgr'
        | _ ->
            // Only add member name as dependency if it's a standalone member function
            // (not a member access on an object)
            if memberName.Contains(".") then
                Set.add memberName exprDeps, scopeMgr'
            else
                exprDeps, scopeMgr'

    | SynExpr.New(_, typeExpr, argExpr, _) ->
        // Handle constructor calls
        let typeDeps = 
            match typeExpr with
            | SynType.LongIdent(SynLongIdent(ids, _, _)) ->
                let typeName = ids |> List.map (fun id -> id.idText) |> String.concat "."
                Set.singleton typeName
            | _ -> Set.empty
        let argDeps, scopeMgr' = analyzeDependencies ctx state scopeMgr argExpr
        Set.union typeDeps argDeps, scopeMgr'
    
    // If-then-else
    | SynExpr.IfThenElse(condExpr, thenExpr, elseExprOpt, _, _, _, _) ->
        let condDeps, scopeMgr1 = analyzeDependencies ctx state scopeMgr condExpr
        let thenDeps, scopeMgr2 = analyzeDependencies ctx state scopeMgr1 thenExpr
        let elseDeps, scopeMgr3 = 
            match elseExprOpt with
            | Some elseExpr -> analyzeDependencies ctx state scopeMgr2 elseExpr
            | None -> Set.empty, scopeMgr2
        Set.unionMany [condDeps; thenDeps; elseDeps], scopeMgr3
    
    // Match expressions - add pattern bindings to each clause
    | SynExpr.Match(_, expr, clauses, _, _) ->
        let exprDeps, scopeMgr1 = analyzeDependencies ctx state scopeMgr expr
        let clauseDeps, scopeMgr2 = 
            clauses 
            |> List.fold (fun (deps, mgr) (SynMatchClause(pat, _, expr, _, _, _)) ->
                // Process match clause with pattern bindings
                let mgr' = ScopeManager.processMatchClause pat mgr
                let exprDeps, mgr'' = analyzeDependencies ctx state mgr' expr
                // Pop the match scope
                let mgr''' = ScopeManager.popScope mgr''
                Set.union deps exprDeps, mgr'''
            ) (Set.empty, scopeMgr1)
        Set.union exprDeps clauseDeps, scopeMgr2
    
    // The pattern appears to be: For(debugPoint, toBody, ident, identBody, equalsRange, fromExpr, toExpr, doBody, range)
    | SynExpr.For(_, _, ident, _, fromExpr, _, toExpr, doExpr, _) ->
        let fromDeps, scopeMgr1 = analyzeDependencies ctx state scopeMgr fromExpr
        let toDeps, scopeMgr2 = analyzeDependencies ctx state scopeMgr1 toExpr
        // Add loop variable to scope
        let scopeMgr3 = ScopeManager.processForLoop ident scopeMgr2
        let doDeps, scopeMgr4 = analyzeDependencies ctx state scopeMgr3 doExpr
        // Pop the for scope
        let scopeMgr5 = ScopeManager.popScope scopeMgr4
        Set.unionMany [fromDeps; toDeps; doDeps], scopeMgr5
    
    // While loops
    | SynExpr.While(_, condExpr, doExpr, _) ->
        let condDeps, scopeMgr1 = analyzeDependencies ctx state scopeMgr condExpr
        let doDeps, scopeMgr2 = analyzeDependencies ctx state scopeMgr1 doExpr
        Set.union condDeps doDeps, scopeMgr2
    
    // Tuple expressions
    | SynExpr.Tuple(_, exprs, _, _) ->
        exprs 
        |> List.fold (fun (deps, mgr) expr ->
            let exprDeps, mgr' = analyzeDependencies ctx state mgr expr
            Set.union deps exprDeps, mgr'
        ) (Set.empty, scopeMgr)
    
    // Array/List expressions
    | SynExpr.ArrayOrList(_, exprs, _) ->
        exprs 
        |> List.fold (fun (deps, mgr) expr ->
            let exprDeps, mgr' = analyzeDependencies ctx state mgr expr
            Set.union deps exprDeps, mgr'
        ) (Set.empty, scopeMgr)
    
    // Array/List comprehensions
    | SynExpr.ArrayOrListComputed(_, expr, _) ->
        analyzeDependencies ctx state scopeMgr expr
    
    // Record construction
    | SynExpr.Record(_, _, fields, _) ->
        fields 
        |> List.fold (fun (deps, mgr) field ->
            match field with
            | SynExprRecordField(_, _, exprOpt, _) ->
                match exprOpt with
                | Some expr ->
                    let exprDeps, mgr' = analyzeDependencies ctx state mgr expr
                    Set.union deps exprDeps, mgr'
                | None -> deps, mgr
        ) (Set.empty, scopeMgr)
    
    // Object expressions
    | SynExpr.ObjExpr(_, _, _, bindings, _, _, _, _) ->
        bindings 
        |> List.fold (fun (deps, mgr) binding ->
            match binding with
            | SynBinding(_, _, _, _, _, _, _, _, _, expr, _, _, _) ->
                let exprDeps, mgr' = analyzeDependencies ctx state mgr expr
                Set.union deps exprDeps, mgr'
        ) (Set.empty, scopeMgr)
    
    // Do expressions
    | SynExpr.Do(expr, _) ->
        analyzeDependencies ctx state scopeMgr expr
    
    // Parentheses
    | SynExpr.Paren(expr, _, _, _) ->
        analyzeDependencies ctx state scopeMgr expr
    
    // Typed expressions
    | SynExpr.Typed(expr, _, _) ->
        analyzeDependencies ctx state scopeMgr expr
    
    // Lambda expressions - add parameters to scope
    | SynExpr.Lambda(_, _, pats, body, _, _, _) ->
        // Extract the actual pattern list from SynSimplePats
        let patList = extractSimplePatterns pats
        let scopeMgr1 = ScopeManager.processLambda patList scopeMgr
        let deps, scopeMgr2 = analyzeDependencies ctx state scopeMgr1 body
        let scopeMgr3 = ScopeManager.popScope scopeMgr2
        deps, scopeMgr3
    
    // Try-with expressions
    | SynExpr.TryWith(tryExpr, withClauses, _, _, _, _) ->
        let tryDeps, scopeMgr1 = analyzeDependencies ctx state scopeMgr tryExpr
        let withDeps, scopeMgr2 = 
            withClauses 
            |> List.fold (fun (deps, mgr) clause ->
                match clause with
                | SynMatchClause(pat, _, expr, _, _, _) ->
                    // Process try-with clause with pattern bindings
                    let mgr' = ScopeManager.processMatchClause pat mgr
                    let exprDeps, mgr'' = analyzeDependencies ctx state mgr' expr
                    let mgr''' = ScopeManager.popScope mgr''
                    Set.union deps exprDeps, mgr'''
            ) (Set.empty, scopeMgr1)
        Set.union tryDeps withDeps, scopeMgr2
    
    // Try-finally expressions
    | SynExpr.TryFinally(tryExpr, finallyExpr, _, _, _, _) ->
        let tryDeps, scopeMgr1 = analyzeDependencies ctx state scopeMgr tryExpr
        let finallyDeps, scopeMgr2 = analyzeDependencies ctx state scopeMgr1 finallyExpr
        Set.union tryDeps finallyDeps, scopeMgr2
    
    // Constants and literals
    | SynExpr.Const(_, _) ->
        Set.empty, scopeMgr
    
    // Lazy expressions
    | SynExpr.Lazy(expr, _) ->
        analyzeDependencies ctx state scopeMgr expr
    
    // Quote expressions
    | SynExpr.Quote(_, _, expr, _, _) ->
        analyzeDependencies ctx state scopeMgr expr
    
    // Cast expressions
    | SynExpr.Downcast(expr, _, _) 
    | SynExpr.Upcast(expr, _, _)
    | SynExpr.InferredDowncast(expr, _)
    | SynExpr.InferredUpcast(expr, _) ->
        analyzeDependencies ctx state scopeMgr expr
    
    // AddressOf
    | SynExpr.AddressOf(_, expr, _, _) ->
        analyzeDependencies ctx state scopeMgr expr
    
    // Dot-indexed get/set
    | SynExpr.DotIndexedGet(expr, indexExpr, _, _) ->
        let objDeps, scopeMgr1 = analyzeDependencies ctx state scopeMgr expr
        let indexDeps, scopeMgr2 = analyzeDependencies ctx state scopeMgr1 indexExpr
        Set.union objDeps indexDeps, scopeMgr2
    
    | SynExpr.DotIndexedSet(objExpr, indexExpr, valueExpr, _, _, _) ->
        let objDeps, scopeMgr1 = analyzeDependencies ctx state scopeMgr objExpr
        let indexDeps, scopeMgr2 = analyzeDependencies ctx state scopeMgr1 indexExpr
        let valueDeps, scopeMgr3 = analyzeDependencies ctx state scopeMgr2 valueExpr
        Set.unionMany [objDeps; indexDeps; valueDeps], scopeMgr3
    
    // Other cases we don't need to analyze
    | _ ->
        Set.empty, scopeMgr

/// Build complete dependency graph from parsed inputs with scope awareness
let buildDependencyGraph (parsedInputs: (string * ParsedInput) list) : (ResolutionContext * Map<string, Set<string>> * Set<string>) =
    let ctx = buildResolutionContext parsedInputs
    let mutable dependencies = Map.empty
    let mutable entryPoints = Set.empty
    
    // Initialize diagnostics
    ReachabilityDiagnostics.initializeDiagnostics()
    let diagnostics = ReachabilityDiagnostics.getDiagnostics()
    
    // Process each input file
    for (filePath, input) in parsedInputs do
        match input with
        | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
            for (SynModuleOrNamespace(longId, _, _, decls, _, _, _, _, _)) in modules do
                let modulePath = longId |> List.map (fun id -> id.idText)
                let openedModules = extractOpenedModules decls
                let state = {
                    CurrentModule = modulePath
                    OpenedModules = openedModules
                }
                
                // Create scope manager for this module
                let scopeMgr = ScopeManager.create modulePath openedModules
                
                // Set current module in diagnostics
                diagnostics.SetCurrentModule(modulePath)
                
                // Process declarations
                for decl in decls do
                    match decl with
                    | SynModuleDecl.Let(_, bindings, _) ->
                        let mutable currentScopeMgr = scopeMgr
                        for (SynBinding(_, _, _, _, attrs, _, _, pat, _, expr, _, _, _)) in bindings do
                            // Extract bindings from this pattern and add to scope
                            let patternBindings = ScopeManager.extractPatternBindings pat
                            currentScopeMgr <- ScopeManager.addBindings patternBindings currentScopeMgr
                            
                            match extractSymbolFromPattern modulePath pat with
                            | Some symbol ->
                                // Check if entry point
                                let isEntryPoint = attrs |> List.exists (fun attrList ->
                                    attrList.Attributes |> List.exists (fun attr ->
                                        match attr.TypeName with
                                        | SynLongIdent(ids, _, _) ->
                                            ids |> List.exists (fun id -> id.idText = "EntryPoint")
                                    ))
                                
                                if isEntryPoint then
                                    entryPoints <- Set.add symbol entryPoints
                                
                                // Analyze dependencies with scope awareness
                                let deps, _ = analyzeDependencies ctx state currentScopeMgr expr
                                dependencies <- Map.add symbol deps dependencies
                                
                                // Log reachability marking for entry points
                                if isEntryPoint then
                                    diagnostics.LogReachabilityMark(symbol, "EntryPoint", "Marked as entry point")
                            | None -> ()
                    
                    | SynModuleDecl.Types(types, _) ->
                        for (SynTypeDefn(SynComponentInfo(_, _, _, longId, _, _, _, _), _, memberDefns, _, _, _)) in types do
                            let typeName = (modulePath @ [longId.Head.idText]) |> String.concat "."
                            
                            // Extract type members and update scope manager
                            let memberNames = 
                                memberDefns 
                                |> List.choose (fun memberDefn ->
                                    match memberDefn with
                                    | SynMemberDefn.Member(SynBinding(_, _, _, _, _, _, _, pat, _, _, _, _, _), _) ->
                                        extractMemberName pat
                                    | _ -> None)
                                |> Set.ofList
                            
                            let scopeMgr' = ScopeManager.updateTypeMembers typeName memberNames scopeMgr
                            
                            // Process member definitions
                            for memberDefn in memberDefns do
                                match memberDefn with
                                | SynMemberDefn.Member(SynBinding(_, _, _, _, _, _, _, pat, _, expr, _, _, _), _) ->
                                    match extractMemberName pat with
                                    | Some memberName ->
                                        let symbol = typeName + "." + memberName
                                        let deps, _ = analyzeDependencies ctx state scopeMgr' expr
                                        dependencies <- Map.add symbol deps dependencies
                                    | None -> ()
                                | _ -> ()
                    
                    | _ -> ()
        | _ -> ()
    
    (ctx, dependencies, entryPoints)

/// Compute transitive closure of dependencies
let computeTransitiveClosure (dependencies: Map<string, Set<string>>) (seeds: Set<string>) : Set<string> =
    let rec closure (visited: Set<string>) (toVisit: string list) =
        match toVisit with
        | [] -> visited
        | current :: rest ->
            if Set.contains current visited then
                closure visited rest
            else
                let newVisited = Set.add current visited
                let deps = Map.tryFind current dependencies |> Option.defaultValue Set.empty
                let newToVisit = Set.toList deps |> List.filter (fun d -> not (Set.contains d newVisited))
                closure newVisited (rest @ newToVisit)
    
    closure Set.empty (Set.toList seeds)

/// Extract all defined symbols from parsed inputs
let extractAllSymbols (parsedInputs: (string * ParsedInput) list) : Set<string> =
    let mutable allSymbols = Set.empty
    
    // Add F# core and pattern library symbols
    let coreSymbols = 
        allPatterns 
        |> List.map (fun pattern -> pattern.QualifiedName)
        |> Set.ofList
    
    // Add F# intrinsic functions
    let intrinsics = Set.ofList [
        "Microsoft.FSharp.Core.Operators.not"
        "Microsoft.FSharp.Core.Operators.failwith"
        "Microsoft.FSharp.Core.ExtraTopLevelOperators.sprintf"
        "Microsoft.FSharp.Core.ExtraTopLevelOperators.printf"
        "Microsoft.FSharp.Core.ExtraTopLevelOperators.printfn"
    ]
    
    allSymbols <- Set.union (Set.union coreSymbols intrinsics) allSymbols
    
    for (_, input) in parsedInputs do
        match input with
        | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
            for (SynModuleOrNamespace(longId, _, _, decls, _, _, _, _, _)) in modules do
                let modulePath = longId |> List.map (fun id -> id.idText)
                
                let moduleSymbols = 
                    decls 
                    |> List.collect (extractSymbolsFromDecl modulePath)
                    |> Set.ofList
                
                allSymbols <- Set.union allSymbols moduleSymbols
        | _ -> ()
    
    allSymbols

/// Main analysis function from parsed inputs
let analyzeFromParsedInputs (parsedInputs: (string * ParsedInput) list) : ReachabilityResult =
    printfn "Starting reachability analysis with %d files" (List.length parsedInputs)
    
    // Extract all symbols first
    let allSymbols = extractAllSymbols parsedInputs
    printfn "Total symbols found: %d" (Set.count allSymbols)
    
    // Build dependency graph
    let (ctx, dependencies, entryPoints) = buildDependencyGraph parsedInputs
    
    // If no explicit entry points, use the main module's bindings
    let actualEntryPoints =
        if Set.isEmpty entryPoints then
            // Find main module (last in compilation order)
            match List.tryLast parsedInputs with
            | Some (_, input) ->
                match input with
                | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
                    modules |> List.collect (fun (SynModuleOrNamespace(longId, _, _, decls, _, _, _, _, _)) ->
                        let modulePath = longId |> List.map (fun id -> id.idText)
                        decls |> List.choose (fun decl ->
                            match decl with
                            | SynModuleDecl.Let(_, bindings, _) ->
                                bindings |> List.tryPick (fun binding ->
                                    match binding with
                                    | SynBinding(_, _, _, _, _, _, _, pat, _, _, _, _, _) ->
                                        extractSymbolFromPattern modulePath pat)
                            | _ -> None))
                    |> Set.ofList
                | _ -> Set.empty
            | None -> Set.empty
        else entryPoints
    
    printfn "Starting reachability analysis with %d entry points" (Set.count actualEntryPoints)
    
    // Compute reachable symbols
    let reachable = computeTransitiveClosure dependencies actualEntryPoints
    printfn "Reachability complete: %d symbols reachable" (Set.count reachable)
    
    // Calculate statistics
    let stats = {
        TotalSymbols = Set.count allSymbols
        ReachableSymbols = Set.count reachable
        EliminatedSymbols = Set.count allSymbols - Set.count reachable
        DependencyEdges = dependencies |> Map.toList |> List.sumBy (fun (_, deps) -> Set.count deps)
    }
    
    {
        Reachable = reachable
        Dependencies = dependencies
        EntryPoints = actualEntryPoints
        Statistics = stats
    }

/// Analyze reachability from a single AST
let analyze (input: ParsedInput) : ReachabilityResult =
    analyzeFromParsedInputs [("main.fsx", input)]
