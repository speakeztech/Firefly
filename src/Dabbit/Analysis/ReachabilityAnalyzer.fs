module Dabbit.Analysis.ReachabilityAnalyzer

open FSharp.Compiler.Syntax
open Core.FCSIngestion.SymbolExtraction
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

/// F# core built-in functions that are implicitly available
let private coreBuiltIns = Map.ofList [
    // Basic functions
    ("not", "Microsoft.FSharp.Core.Operators.not")
    ("id", "Microsoft.FSharp.Core.Operators.id")
    ("ignore", "Microsoft.FSharp.Core.Operators.ignore")
    ("fst", "Microsoft.FSharp.Core.Operators.fst")
    ("snd", "Microsoft.FSharp.Core.Operators.snd")
    // Exception handling
    ("failwith", "Microsoft.FSharp.Core.Operators.failwith")
    ("failwithf", "Microsoft.FSharp.Core.Operators.failwithf")
    ("raise", "Microsoft.FSharp.Core.Operators.raise")
    ("reraise", "Microsoft.FSharp.Core.Operators.reraise")
    ("invalidArg", "Microsoft.FSharp.Core.Operators.invalidArg")
    ("invalidOp", "Microsoft.FSharp.Core.Operators.invalidOp")
    ("nullArg", "Microsoft.FSharp.Core.Operators.nullArg")
    // String formatting
    ("sprintf", "Microsoft.FSharp.Core.ExtraTopLevelOperators.sprintf")
    ("printf", "Microsoft.FSharp.Core.ExtraTopLevelOperators.printf")
    ("printfn", "Microsoft.FSharp.Core.ExtraTopLevelOperators.printfn")
    ("eprintf", "Microsoft.FSharp.Core.ExtraTopLevelOperators.eprintf")
    ("eprintfn", "Microsoft.FSharp.Core.ExtraTopLevelOperators.eprintfn")
    ("fprintf", "Microsoft.FSharp.Core.ExtraTopLevelOperators.fprintf")
    ("fprintfn", "Microsoft.FSharp.Core.ExtraTopLevelOperators.fprintfn")
    // Type operations
    ("typeof", "Microsoft.FSharp.Core.Operators.typeof")
    ("typedefof", "Microsoft.FSharp.Core.Operators.typedefof")
    ("sizeof", "Microsoft.FSharp.Core.Operators.sizeof")
    ("nameof", "Microsoft.FSharp.Core.Operators.nameof")
    ("box", "Microsoft.FSharp.Core.Operators.box")
    ("unbox", "Microsoft.FSharp.Core.Operators.unbox")
    // Comparison
    ("compare", "Microsoft.FSharp.Core.Operators.compare")
    ("hash", "Microsoft.FSharp.Core.Operators.hash")
    ("min", "Microsoft.FSharp.Core.Operators.min")
    ("max", "Microsoft.FSharp.Core.Operators.max")
    ("sign", "Microsoft.FSharp.Core.Operators.sign")
    ("abs", "Microsoft.FSharp.Core.Operators.abs")
    // Conversion functions
    ("int", "Microsoft.FSharp.Core.Operators.int")
    ("int32", "Microsoft.FSharp.Core.Operators.int32")
    ("int64", "Microsoft.FSharp.Core.Operators.int64")
    ("uint32", "Microsoft.FSharp.Core.Operators.uint32")
    ("uint64", "Microsoft.FSharp.Core.Operators.uint64")
    ("float", "Microsoft.FSharp.Core.Operators.float")
    ("float32", "Microsoft.FSharp.Core.Operators.float32")
    ("double", "Microsoft.FSharp.Core.Operators.double")
    ("decimal", "Microsoft.FSharp.Core.Operators.decimal")
    ("byte", "Microsoft.FSharp.Core.Operators.byte")
    ("sbyte", "Microsoft.FSharp.Core.Operators.sbyte")
    ("int16", "Microsoft.FSharp.Core.Operators.int16")
    ("uint16", "Microsoft.FSharp.Core.Operators.uint16")
    ("char", "Microsoft.FSharp.Core.Operators.char")
    ("string", "Microsoft.FSharp.Core.Operators.string")
]

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

/// Extract simple pattern list from SynSimplePats
let extractSimplePatterns (pats: SynSimplePats) : SynSimplePat list =
    match pats with
    | SynSimplePats.SimplePats(patterns, _, _) -> patterns  // Handle the main case

/// Extract type names from a SynType
let rec extractTypeReferences (synType: SynType) : Set<string> =
    match synType with
    | SynType.LongIdent(SynLongIdent(ids, _, _)) ->
        let typeName = ids |> List.map (fun id -> id.idText) |> String.concat "."
        Set.singleton typeName
    | SynType.App(typeName, _, typeArgs, _, _, _, _) ->
        let baseType = extractTypeReferences typeName
        let argTypes = typeArgs |> List.collect (extractTypeReferences >> Set.toList) |> Set.ofList
        Set.union baseType argTypes
    | SynType.Tuple(_, segments, _) ->
        segments 
        |> List.choose (fun segment ->
            match segment with
            | SynTupleTypeSegment.Type t -> Some (extractTypeReferences t)
            | SynTupleTypeSegment.Star _ -> None
            | SynTupleTypeSegment.Slash _ -> None)
        |> Set.unionMany
    | SynType.Array(rank, elementType, _) ->
        extractTypeReferences elementType
    | SynType.Fun(argType, returnType, _, _) ->
        Set.union (extractTypeReferences argType) (extractTypeReferences returnType)
    | SynType.Var(_, _) -> Set.empty  // Type variables
    | SynType.Anon _ -> Set.empty
    | _ -> Set.empty

/// Build resolution context from parsed inputs - CORRECTED VERSION
let private buildResolutionContext (parsedInputs: (string * ParsedInput) list) : ResolutionContext =
    let mutable symbols = Map.empty<string, Set<string>>
    let mutable typeMembers = Map.empty<string, Set<string>>
    
    // Add core built-in symbols first
    let coreSymbols = 
        coreBuiltIns 
        |> Map.toList 
        |> List.groupBy (fun (shortName, qualifiedName) ->
            // Extract module from qualified name
            match qualifiedName.LastIndexOf('.') with
            | -1 -> ""
            | idx -> qualifiedName.Substring(0, idx)
        )
        |> List.iter (fun (moduleName, syms) ->
            let shortNames = syms |> List.map fst |> Set.ofList
            symbols <- Map.add moduleName shortNames symbols
        )
    
    // Process each parsed input
    for (_, input) in parsedInputs do
        // Extract all symbols using unified extraction
        let allDeclSymbols = extractSymbolsFromParsedInput input
        
        // Debug logging
        if not (List.isEmpty allDeclSymbols) then
            let moduleName = 
                match input with
                | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
                    match modules with
                    | SynModuleOrNamespace(longId, _, _, _, _, _, _, _, _) :: _ ->
                        longId |> List.map (fun id -> id.idText) |> String.concat "."
                    | [] -> ""
                | _ -> ""
            
            let symbolNames = allDeclSymbols |> List.map (fun s -> s.QualifiedName)
            printfn "[DEBUG] Module %s has symbols: %A" moduleName symbolNames
        
        // Group symbols by their actual module path
        for symbol in allDeclSymbols do
            // Get the module path for this symbol
            let moduleName = 
                if List.isEmpty symbol.ModulePath then ""
                else String.concat "." symbol.ModulePath
            
            // For nested types/modules, we need to handle the full hierarchy
            match symbol.Kind with
            | Member typeName ->
                // Add to type members
                match Map.tryFind typeName typeMembers with
                | Some existing ->
                    typeMembers <- Map.add typeName (Set.add symbol.ShortName existing) typeMembers
                | None ->
                    typeMembers <- Map.add typeName (Set.singleton symbol.ShortName) typeMembers
                    
                // Also add the member to its module's symbols
                match Map.tryFind moduleName symbols with
                | Some existingSymbols ->
                    symbols <- Map.add moduleName (Set.add symbol.ShortName existingSymbols) symbols
                | None ->
                    symbols <- Map.add moduleName (Set.singleton symbol.ShortName) symbols
                    
            | _ ->
                // Regular symbols - add to module
                match Map.tryFind moduleName symbols with
                | Some existingSymbols ->
                    symbols <- Map.add moduleName (Set.add symbol.ShortName existingSymbols) symbols
                | None ->
                    symbols <- Map.add moduleName (Set.singleton symbol.ShortName) symbols
                
                // If this is a type, we might need to prepare for its members
                match symbol.Kind with
                | Type ->
                    if not (Map.containsKey symbol.QualifiedName typeMembers) then
                        typeMembers <- Map.add symbol.QualifiedName Set.empty typeMembers
                | _ -> ()
    
    // Add known type members for builtin types
    typeMembers <- Map.add "voption" (Set.ofList ["hasValue"; "value"]) typeMembers
    typeMembers <- Map.add "ValueOption" (Set.ofList ["IsSome"; "IsNone"; "Value"]) typeMembers
    
    // Debug: print all modules and their symbols
    printfn "[DEBUG] Final symbol context:"
    symbols |> Map.iter (fun modName syms ->
        if Set.count syms <= 10 then
            printfn "  Module '%s': %A" modName syms
        else
            printfn "  Module '%s': %d symbols" modName (Set.count syms)
    )
    
    { SymbolsByFullName = symbols; TypeMembers = typeMembers }

/// Try to resolve a symbol name
let resolveSymbol (ctx: ResolutionContext) (state: ModuleResolutionState) (name: string) : string option =
    // Check if it's a built-in F# function first
    match Map.tryFind name coreBuiltIns with
    | Some qualifiedName -> Some qualifiedName
    | None ->
        // Handle "this" member access specially
        if name.StartsWith("this.") then
            let memberName = name.Substring(5)
            // Return the member access pattern as-is for now
            Some name
        // First check if it's a fully qualified name already
        elif name.Contains(".") then
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
            
            // First, try direct resolution in each search path
            let directResult = 
                searchPaths
                |> List.tryPick (fun modulePath ->
                    let modulePathStr = 
                        if List.isEmpty modulePath then "" 
                        else String.concat "." modulePath
                    
                    // Debug log
                    printfn "[DEBUG] Checking module '%s' for symbol '%s'" modulePathStr name
                    
                    // Check if this module has the symbol
                    match Map.tryFind modulePathStr ctx.SymbolsByFullName with
                    | Some symbols -> 
                        printfn "[DEBUG] Module '%s' has symbols: %A" modulePathStr symbols
                        if Set.contains name symbols then
                            // Found! Return the fully qualified name
                            let qualified = 
                                if modulePathStr = "" then 
                                    name
                                else 
                                    modulePathStr + "." + name
                            printfn "[DEBUG] Resolved '%s' to '%s'" name qualified
                            Some qualified
                        else
                            None
                    | None -> 
                        printfn "[DEBUG] Module '%s' not found in context" modulePathStr
                        None
                )
            
            match directResult with
            | Some result -> Some result
            | None ->
                // If not found directly, check sub-modules of opened modules
                searchPaths
                |> List.tryPick (fun modulePath ->
                    let modulePathStr = 
                        if List.isEmpty modulePath then ""
                        else String.concat "." modulePath
                    
                    printfn "[DEBUG] Checking sub-modules of '%s' for symbol '%s'" modulePathStr name
                    
                    // Look for the symbol in any sub-module of this module
                    ctx.SymbolsByFullName
                    |> Map.tryPick (fun modName symbols ->
                        // Check if this module is a sub-module of our search path
                        let isSubModule = 
                            if modulePathStr = "" then
                                modName <> ""  // All non-empty modules are sub-modules of root
                            else
                                modName.StartsWith(modulePathStr + ".") || modName = modulePathStr
                        
                        if isSubModule && Set.contains name symbols then
                            // Found in a sub-module! Return fully qualified name
                            let qualified = modName + "." + name
                            printfn "[DEBUG] Found '%s' in sub-module '%s', resolved to '%s'" name modName qualified
                            Some qualified
                        else
                            None
                    )
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
            // Member access on 'this' - still try to resolve it
            let fullName = "this." + memberName
            match resolveSymbolWithLogging ctx state fullName with
            | Some resolved -> Set.singleton resolved, scopeMgr'
            | None -> exprDeps, scopeMgr'
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
                printfn "[DEBUG] New expression references type: %s" typeName
                Set.singleton typeName
            | SynType.App(typeName, _, _, _, _, _, _) ->
                // Handle generic type applications like StackBuffer<'T>
                let baseTypeDeps = extractTypeReferences typeName
                printfn "[DEBUG] New expression references generic type: %A" baseTypeDeps
                baseTypeDeps
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
        // Use the existing extractSimplePatterns function
        let patList = extractSimplePatterns pats
        
        // Use the existing processLambda function from ScopeManager
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
    
    // Constants and literals - check for string interpolation
    | SynExpr.Const(constant, _) ->
        match constant with
        | SynConst.String(value, _, _) when value.Contains("sprintf") || value.Contains("printf") ->
            // Handle format strings that might reference sprintf
            match resolveSymbolWithLogging ctx state "sprintf" with
            | Some resolved -> Set.singleton resolved, scopeMgr
            | None -> Set.empty, scopeMgr
        | _ -> Set.empty, scopeMgr
    
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
    
    // DotSet - property setters
    | SynExpr.DotSet(expr, longIdent, rhsExpr, _) ->
        let exprDeps, scopeMgr1 = analyzeDependencies ctx state scopeMgr expr
        let rhsDeps, scopeMgr2 = analyzeDependencies ctx state scopeMgr1 rhsExpr
        Set.union exprDeps rhsDeps, scopeMgr2
    
    // InterpolatedString - F# 5.0+
    | SynExpr.InterpolatedString(parts, _, _) ->
        parts 
        |> List.fold (fun (deps, mgr) part ->
            match part with
            | SynInterpolatedStringPart.FillExpr(expr, _) ->
                let exprDeps, mgr' = analyzeDependencies ctx state mgr expr
                Set.union deps exprDeps, mgr'
            | _ -> deps, mgr
        ) (Set.empty, scopeMgr)
    
    // Yield or return
    | SynExpr.YieldOrReturn(_, expr, _, _) ->
        analyzeDependencies ctx state scopeMgr expr
    
    // Yield or return from
    | SynExpr.YieldOrReturnFrom(_, expr, _, _) ->
        analyzeDependencies ctx state scopeMgr expr
    
    // Set (mutable assignment)
    | SynExpr.Set(expr, rhsExpr, _) ->
        let lhsDeps, scopeMgr1 = analyzeDependencies ctx state scopeMgr expr
        let rhsDeps, scopeMgr2 = analyzeDependencies ctx state scopeMgr1 rhsExpr
        Set.union lhsDeps rhsDeps, scopeMgr2
    
    // Fixed
    | SynExpr.Fixed(expr, _) ->
        analyzeDependencies ctx state scopeMgr expr
    
    // Null
    | SynExpr.Null _ ->
        Set.empty, scopeMgr
    
    // Type test
    | SynExpr.TypeTest(expr, _, _) ->
        analyzeDependencies ctx state scopeMgr expr
    
    // Assert
    | SynExpr.Assert(expr, _) ->
        analyzeDependencies ctx state scopeMgr expr
    
    // Computation expressions
    | SynExpr.ComputationExpr(_, expr, _) ->
        analyzeDependencies ctx state scopeMgr expr
    
    // ForEach
    | SynExpr.ForEach(debugPoint, forLoopKind, seqExprOnly, isFromSource, pat, enumExpr, bodyExpr, range) ->
        // Process the pattern to add loop variable to scope
        let scopeMgr1 = ScopeManager.processMatchClause pat scopeMgr
        let enumDeps, scopeMgr2 = analyzeDependencies ctx state scopeMgr1 enumExpr
        let bodyDeps, scopeMgr3 = analyzeDependencies ctx state scopeMgr2 bodyExpr
        // Pop the foreach scope
        let scopeMgr4 = ScopeManager.popScope scopeMgr3
        Set.union enumDeps bodyDeps, scopeMgr4
    
    // Other cases we don't need to analyze
    | _ ->
        Set.empty, scopeMgr


/// Build complete dependency graph from parsed inputs with scope awareness
let buildDependencyGraph (parsedInputs: (string * ParsedInput) list) : ResolutionContext * Map<string, Set<string>> * Set<string> =
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
                
                // Debug: print module and opens
                printfn "[DEBUG] Processing module: %s" (String.concat "." modulePath)
                printfn "[DEBUG] Opened modules: %A" openedModules
                
                let state = {
                    CurrentModule = modulePath
                    OpenedModules = openedModules
                }
                
                // Create scope manager for this module
                let scopeMgr = create modulePath openedModules
                
                // Set current module in diagnostics
                diagnostics.SetCurrentModule(modulePath)
                
                // Process declarations
                for decl in decls do
                    match decl with
                    | SynModuleDecl.Let(_, bindings, _) ->
                        let mutable currentScopeMgr = scopeMgr
                        for (SynBinding(_, _, _, _, attrs, _, valData, pat, returnInfo, expr, _, _, _)) in bindings do
                            // Extract bindings from this pattern and add to scope
                            let patternBindings = extractPatternBindings pat
                            currentScopeMgr <- addBindings patternBindings currentScopeMgr
                            
                            // Use unified extraction to get the symbol
                            let extractedSymbols = 
                                match pat with
                                | SynPat.Named(SynIdent(ident, _), _, _, _) ->
                                    let symbol : ExtractedSymbol = {
                                        QualifiedName = SymbolNames.qualify modulePath ident.idText
                                        ShortName = ident.idText
                                        ModulePath = modulePath
                                        Kind = Value 
                                    }
                                    [symbol]
                                | SynPat.LongIdent(longIdent, _, _, _, _, _) ->
                                    let (SynLongIdent(ids, _, _)) = longIdent
                                    let fullName = ids |> List.map (fun id -> id.idText) |> String.concat "."
                                    let shortName = match List.tryLast ids with 
                                                    | Some id -> id.idText 
                                                    | None -> fullName
                                    let symbol : ExtractedSymbol = {
                                        QualifiedName = SymbolNames.qualify modulePath fullName
                                        ShortName = shortName
                                        ModulePath = modulePath
                                        Kind = Value
                                    }
                                    [symbol]
                                | _ -> []
                            
                            for extractedSymbol in extractedSymbols do
                                let symbolName = extractedSymbol.QualifiedName  // Convert to string
                                
                                // Check if entry point
                                let isEntryPoint = attrs |> List.exists (fun attrList ->
                                    attrList.Attributes |> List.exists (fun attr ->
                                        match attr.TypeName with
                                        | SynLongIdent(ids, _, _) ->
                                            ids |> List.exists (fun id -> id.idText = "EntryPoint")))
                                
                                if isEntryPoint then
                                    entryPoints <- Set.add symbolName entryPoints
                                
                                // Extract type dependencies from the binding signature
                                let typeDeps = 
                                    let mutable deps = Set.empty
                                    
                                    // Extract from return type if present
                                    match returnInfo with
                                    | Some (SynBindingReturnInfo(returnType, _, _, _)) ->
                                        let extractedTypes = extractTypeReferences returnType
                                        printfn "[DEBUG] Return type deps for %s: %A" symbolName extractedTypes
                                        deps <- Set.union deps extractedTypes
                                    | None -> ()
                                    
                                    // Extract from pattern (for typed patterns like (x: int))
                                    let rec extractPatternTypes (p: SynPat) =
                                        match p with
                                        | SynPat.Typed(_, synType, _) ->
                                            let extractedTypes = extractTypeReferences synType
                                            printfn "[DEBUG] Pattern type deps: %A" extractedTypes
                                            extractedTypes
                                        | SynPat.Paren(innerPat, _) ->
                                            extractPatternTypes innerPat
                                        | SynPat.Tuple(_, pats, _, _) ->
                                            pats |> List.map extractPatternTypes |> Set.unionMany
                                        | _ -> Set.empty
                                    
                                    deps <- Set.union deps (extractPatternTypes pat)
                                    printfn "[DEBUG] Total type deps for %s: %A" symbolName deps
                                    deps
                                
                                // Analyze expression dependencies with scope awareness
                                let exprDeps, _ = analyzeDependencies ctx state currentScopeMgr expr
                                
                                // Combine type and expression dependencies
                                let allDeps = Set.union exprDeps typeDeps
                                dependencies <- Map.add symbolName allDeps dependencies
                                
                                // Log reachability marking for entry points
                                if isEntryPoint then
                                    diagnostics.LogReachabilityMark(symbolName, "EntryPoint", "Marked as entry point")
                    
                    | SynModuleDecl.Types(types, _) ->
                        for (SynTypeDefn(SynComponentInfo(_, _, _, longId, _, _, _, _), _, memberDefns, _, _, _)) in types do
                            let typeName = (modulePath @ [longId.Head.idText]) |> String.concat "."
                            dependencies <- Map.add typeName Set.empty dependencies
                            
                            // Process member definitions
                            for memberDefn in memberDefns do
                                match memberDefn with
                                | SynMemberDefn.Member(SynBinding(_, _, _, _, _, _, _, pat, _, expr, _, _, _), _) ->
                                    match extractMemberName pat with
                                    | Some memberName ->
                                        let memberQualifiedName = typeName + "." + memberName
                                        let deps, _ = analyzeDependencies ctx state scopeMgr expr
                                        dependencies <- Map.add memberQualifiedName deps dependencies
                                    | None -> ()
                                | _ -> ()
                    
                    | SynModuleDecl.NestedModule(SynComponentInfo(_, _, _, longId, _, _, _, _), _, nestedDecls, _, _, _) ->
                        let nestedModuleName = longId.Head.idText
                        let nestedPath = modulePath @ [nestedModuleName]
                        let nestedQualifiedName = SymbolNames.qualify modulePath nestedModuleName
                        dependencies <- Map.add nestedQualifiedName Set.empty dependencies
                    
                    | SynModuleDecl.Open(target, _) ->
                        // Already handled in extractOpenedModules
                        ()
                    
                    | _ -> ()
        | _ -> ()
    
    ctx, dependencies, entryPoints

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

/// Build symbol context from parsed inputs
let private buildSymbolContext (parsedInputs: (string * ParsedInput) list) : ResolutionContext =
    let mutable symbols = Map.empty<string, Set<string>>
    let mutable typeMembers = Map.empty<string, Set<string>>
    
    // Process each input file
    for (_, input) in parsedInputs do
        // Use unified extraction
        let extractedSymbols = extractSymbolsFromParsedInput input
        
        // Group by module using the unified function
        let symbolsByModule = groupSymbolsByModule extractedSymbols
        
        // Merge into our symbol map
        symbolsByModule |> Map.iter (fun moduleName moduleSymbols ->
            match Map.tryFind moduleName symbols with
            | Some existing -> 
                symbols <- Map.add moduleName (Set.union existing moduleSymbols) symbols
            | None -> 
                symbols <- Map.add moduleName moduleSymbols symbols
        )
        
        // Extract type members
        extractedSymbols
        |> List.iter (fun symbol ->
            match symbol.Kind with
            | Member typeName ->
                match Map.tryFind typeName typeMembers with
                | Some existing ->
                    typeMembers <- Map.add typeName (Set.add symbol.ShortName existing) typeMembers
                | None ->
                    typeMembers <- Map.add typeName (Set.singleton symbol.ShortName) typeMembers
            | _ -> ()
        )
    
    // Add known type members for builtin types
    typeMembers <- Map.add "voption" (Set.ofList ["hasValue"; "value"]) typeMembers
    typeMembers <- Map.add "ValueOption" (Set.ofList ["IsSome"; "IsNone"; "Value"]) typeMembers
    
    // Debug output
    printfn "[DEBUG] Final symbol context:"
    symbols |> Map.iter (fun modName syms ->
        if Set.count syms <= 10 then
            printfn "  Module '%s': %A" modName syms
        else
            printfn "  Module '%s': %d symbols" modName (Set.count syms)
    )
    
    { SymbolsByFullName = symbols; TypeMembers = typeMembers }

/// Extract all symbols from parsed inputs as qualified name strings
let private extractAllSymbols (parsedInputs: (string * ParsedInput) list) : Set<string> =
    let mutable allSymbols = Set.empty
    
    // Add core F# built-in symbols  
    let coreSymbols = coreBuiltIns |> Map.toList |> List.map snd |> Set.ofList
    
    // Add F# intrinsic functions
    let intrinsics = 
        [ "id"; "ignore"; "fst"; "snd"; "not"; "ref"; "incr"; "decr"
          "defaultArg"; "defaultValueArg"; "typeof"; "typedefof"; "sizeof"
          "nameof"; "reraise"; "rethrow" 
        ] |> List.map (fun name -> "Microsoft.FSharp.Core.Operators." + name) |> Set.ofList

    allSymbols <- Set.union (Set.union coreSymbols intrinsics) allSymbols
    
    // Extract from each file using unified extraction
    for (_, input) in parsedInputs do
        let extractedSymbols = Core.FCSIngestion.SymbolExtraction.extractSymbolsFromParsedInput input
        let qualifiedNames = extractedSymbols |> List.map (fun s -> s.QualifiedName) |> Set.ofList
        allSymbols <- Set.union allSymbols qualifiedNames
    
    allSymbols

/// Main analysis function from parsed inputs
let analyzeFromParsedInputs (parsedInputs: (string * ParsedInput) list) : ReachabilityResult =
    printfn "Starting reachability analysis with %d files" (List.length parsedInputs)
    
    // Extract all symbols first
    let allSymbols = extractAllSymbols parsedInputs
    printfn "Total symbols found: %d" (Set.count allSymbols)
    
    // Build dependency graph
    let ctx, dependencies, entryPoints = buildDependencyGraph parsedInputs
    
    // If no explicit entry points, use the main module's bindings
    let actualEntryPoints =
        if Set.isEmpty entryPoints then
            // Find main module (last in compilation order)
            match List.tryLast parsedInputs with
            | Some (_, input) ->
                match input with
                | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
                    modules 
                    |> List.collect (fun (SynModuleOrNamespace(longId, _, _, decls, _, _, _, _, _)) ->
                        let modulePath = longId |> List.map (fun id -> id.idText)
                        decls 
                        |> List.choose (fun decl ->
                            match decl with
                            | SynModuleDecl.Let(_, bindings, _) ->
                                bindings 
                                |> List.tryPick (fun binding ->
                                    match binding with
                                    | SynBinding(_, _, _, _, _, _, _, pat, _, _, _, _, _) ->
                                        // Extract qualified name string directly
                                        match pat with
                                        | SynPat.Named(SynIdent(ident, _), _, _, _) ->
                                            Some (SymbolNames.qualify modulePath ident.idText)
                                        | SynPat.LongIdent(longIdent, _, _, _, _, _) ->
                                            let (SynLongIdent(ids, _, _)) = longIdent
                                            let fullName = ids |> List.map (fun id -> id.idText) |> String.concat "."
                                            Some (SymbolNames.qualify modulePath fullName)
                                        | _ -> None)
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