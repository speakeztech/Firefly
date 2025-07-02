module Dabbit.Analysis.ScopeManager

open FSharp.Compiler.Syntax
open FSharp.Compiler.Text
open Core.Types.TypeSystem

/// Binding information for a symbol in scope
type BindingInfo = {
    Name: string
    Type: MLIRType option
    BindingKind: BindingKind
    SourceLocation: range option
}

and BindingKind =
    | LocalBinding        // let x = ...
    | Parameter          // function parameter
    | PatternBinding     // pattern match binding
    | ForLoopVariable    // for i in ...
    | TryBinding        // try ... with e ->

/// Lexical scope containing bindings visible at a point in the code
type LexicalScope = {
    /// Bindings defined in this scope
    LocalBindings: Map<string, BindingInfo>
    /// Parent scope (if any)
    ParentScope: LexicalScope option
    /// Modules opened in this scope
    OpenModules: string list list
    /// Type of this scope
    ScopeKind: ScopeKind
}

and ScopeKind =
    | ModuleScope of modulePath: string list
    | FunctionScope of functionName: string
    | LetScope
    | MatchScope
    | ForScope
    | TryScope

/// Scope manager tracks lexical scopes during AST traversal
type ScopeManager = {
    /// Current scope
    CurrentScope: LexicalScope
    /// All module-level symbols for cross-module resolution
    ModuleSymbols: Map<string, Set<string>>
    /// Type members for member resolution
    TypeMembers: Map<string, Set<string>>
}

/// Create an empty root scope
let createRootScope (modulePath: string list) (openModules: string list list) =
    {
        LocalBindings = Map.empty
        ParentScope = None
        OpenModules = openModules
        ScopeKind = ModuleScope modulePath
    }

/// Create a new scope manager
let create (modulePath: string list) (openModules: string list list) =
    {
        CurrentScope = createRootScope modulePath openModules
        ModuleSymbols = Map.empty
        TypeMembers = Map.empty
    }

/// Push a new scope
let pushScope (kind: ScopeKind) (manager: ScopeManager) =
    let newScope = {
        LocalBindings = Map.empty
        ParentScope = Some manager.CurrentScope
        OpenModules = manager.CurrentScope.OpenModules
        ScopeKind = kind
    }
    { manager with CurrentScope = newScope }

/// Pop to parent scope
let popScope (manager: ScopeManager) =
    match manager.CurrentScope.ParentScope with
    | Some parent -> { manager with CurrentScope = parent }
    | None -> manager // Already at root

/// Add a binding to the current scope
let addBinding (name: string) (info: BindingInfo) (manager: ScopeManager) =
    let updatedScope = {
        manager.CurrentScope with
            LocalBindings = Map.add name info manager.CurrentScope.LocalBindings
    }
    { manager with CurrentScope = updatedScope }

/// Add multiple bindings
let addBindings (bindings: (string * BindingInfo) list) (manager: ScopeManager) =
    bindings |> List.fold (fun mgr (name, info) -> addBinding name info mgr) manager

/// Try to find a binding in the scope chain
let rec tryFindBinding (name: string) (scope: LexicalScope) =
    match Map.tryFind name scope.LocalBindings with
    | Some binding -> Some binding
    | None ->
        match scope.ParentScope with
        | Some parent -> tryFindBinding name parent
        | None -> None

/// Check if a name is a local binding
let isLocalBinding (name: string) (manager: ScopeManager) =
    tryFindBinding name manager.CurrentScope |> Option.isSome

/// Extract bindings from a pattern
let rec extractPatternBindings (pat: SynPat) : (string * BindingInfo) list =
    match pat with
    | SynPat.Named(SynIdent(ident, _), _, _, _) ->
        [(ident.idText, {
            Name = ident.idText
            Type = None
            BindingKind = PatternBinding
            SourceLocation = Some ident.idRange
        })]
    
    | SynPat.Typed(pat, _, _) ->
        extractPatternBindings pat
    
    | SynPat.Tuple(_, pats, _, _) ->
        pats |> List.collect extractPatternBindings
    
    | SynPat.Paren(pat, _) ->
        extractPatternBindings pat
    
    | SynPat.Or(pat1, pat2, _, _) ->
        // For OR patterns, we need bindings that appear in both branches
        let bindings1 = extractPatternBindings pat1 |> Map.ofList
        let bindings2 = extractPatternBindings pat2 |> Map.ofList
        Map.toList bindings1 |> List.filter (fun (name, _) -> Map.containsKey name bindings2)
    
    | SynPat.Ands(pats, _) ->
        pats |> List.collect extractPatternBindings
    
    | SynPat.ArrayOrList(_, pats, _) ->
        pats |> List.collect extractPatternBindings
    
    | SynPat.Record(fieldPats, _) ->
        fieldPats |> List.collect (fun (_, _, pat) -> extractPatternBindings pat)
    
    | SynPat.LongIdent(_, _, _, ctorArgs, _, _) ->
        match ctorArgs with
        | SynArgPats.Pats pats -> pats |> List.collect extractPatternBindings
        | _ -> []
    
    | SynPat.As(pat1, pat2, _) ->
        extractPatternBindings pat1 @ extractPatternBindings pat2
    
    | _ -> []

/// Extract bindings from a let binding
let extractLetBindings (binding: SynBinding) =
    match binding with
    | SynBinding(_, _, _, _, _, _, _, pat, _, _, _, _, _) ->
        extractPatternBindings pat

/// Extract parameter bindings from simple patterns
let rec extractSimpleParameterBindings (pats: SynSimplePat list) : (string * BindingInfo) list =
    pats |> List.collect (fun pat ->
        match pat with
        | SynSimplePat.Id(ident, _, _, _, _, _) ->
            [(ident.idText, {
                Name = ident.idText
                Type = None
                BindingKind = Parameter
                SourceLocation = Some ident.idRange
            })]
        | SynSimplePat.Typed(pat, _, _) ->
            extractSimpleParameterBindings [pat]
        | _ -> []
    )

/// Process a lambda expression and extract parameter bindings
let processLambda (pats: SynSimplePat list) (manager: ScopeManager) =
    let bindings = extractSimpleParameterBindings pats
    manager
    |> pushScope (FunctionScope "lambda")
    |> addBindings bindings

/// Process a match clause
let processMatchClause (pat: SynPat) (manager: ScopeManager) =
    let bindings = extractPatternBindings pat
    manager
    |> pushScope MatchScope
    |> addBindings bindings

/// Process a for loop
let processForLoop (ident: Ident) (manager: ScopeManager) =
    let binding = {
        Name = ident.idText
        Type = None
        BindingKind = ForLoopVariable
        SourceLocation = Some ident.idRange
    }
    manager
    |> pushScope ForScope
    |> addBinding ident.idText binding

/// Update module symbols
let updateModuleSymbols (moduleName: string) (symbols: Set<string>) (manager: ScopeManager) =
    { manager with ModuleSymbols = Map.add moduleName symbols manager.ModuleSymbols }

/// Update type members
let updateTypeMembers (typeName: string) (members: Set<string>) (manager: ScopeManager) =
    { manager with TypeMembers = Map.add typeName members manager.TypeMembers }

/// Get all visible bindings in the current scope
let getVisibleBindings (manager: ScopeManager) =
    let rec collect (scope: LexicalScope) (acc: Map<string, BindingInfo>) =
        let merged = Map.fold (fun acc k v -> Map.add k v acc) acc scope.LocalBindings
        match scope.ParentScope with
        | Some parent -> collect parent merged
        | None -> merged
    collect manager.CurrentScope Map.empty

/// Debug helper to print scope information
let debugPrintScope (manager: ScopeManager) =
    let rec printScope indent (scope: LexicalScope) =
        printfn "%s[Scope: %A]" indent scope.ScopeKind
        scope.LocalBindings |> Map.iter (fun name info ->
            printfn "%s  %s (%A)" indent name info.BindingKind)
        match scope.ParentScope with
        | Some parent -> printScope (indent + "  ") parent
        | None -> ()
    printScope "" manager.CurrentScope