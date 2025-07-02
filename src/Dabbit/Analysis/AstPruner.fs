module Dabbit.Analysis.AstPruner

open FSharp.Compiler.Syntax
open FSharp.Compiler.Text

/// Check if a fully qualified name is in the reachable set
let private isReachable (reachable: Set<string>) (fullName: string) =
    Set.contains fullName reachable

/// Extract the full name from a pattern
let private getPatternName (moduleName: string) (pat: SynPat) : string option =
    match pat with
    | SynPat.Named(SynIdent(ident, _), _, _, _) -> 
        Some (sprintf "%s.%s" moduleName ident.idText)
    | SynPat.LongIdent(SynLongIdent(ids, _, _), _, _, _, _, _) ->
        let name = ids |> List.map (fun id -> id.idText) |> String.concat "."
        Some (sprintf "%s.%s" moduleName name)
    | _ -> None

/// Prune a single declaration based on reachability
let rec pruneDeclaration (reachable: Set<string>) (moduleName: string) (decl: SynModuleDecl) : SynModuleDecl option =
    match decl with
    | SynModuleDecl.Let(isRec, bindings, range) ->
        // Filter bindings to only include reachable ones
        let reachableBindings = 
            bindings 
            |> List.filter (fun (SynBinding(_, _, _, _, _, _, _, pat, _, _, _, _, _)) ->
                match getPatternName moduleName pat with
                | Some fullName -> isReachable reachable fullName
                | None -> false  // Can't determine name, exclude
            )
        
        // Only keep the declaration if there are reachable bindings
        if List.isEmpty reachableBindings then 
            None
        else 
            Some (SynModuleDecl.Let(isRec, reachableBindings, range))
    
    | SynModuleDecl.Types(typeDefs, range) ->
        // Filter type definitions to only include reachable ones
        let reachableTypeDefs = 
            typeDefs 
            |> List.filter (fun (SynTypeDefn(componentInfo, typeRepr, members, implicitCtor, typeRange, trivia)) ->
                let (SynComponentInfo(_, _, _, longId, _, _, _, _)) = componentInfo
                let typeName = longId |> List.map (fun id -> id.idText) |> String.concat "."
                let fullTypeName = sprintf "%s.%s" moduleName typeName
                
                if isReachable reachable fullTypeName then
                    true
                else
                    // Check if any members are reachable
                    members |> List.exists (function
                        | SynMemberDefn.Member(SynBinding(_, _, _, _, _, _, _, pat, _, _, _, _, _), _) ->
                            match pat with
                            | SynPat.LongIdent(SynLongIdent(ids, _, _), _, _, _, _, _) ->
                                let memberName = ids |> List.map (fun id -> id.idText) |> String.concat "."
                                // Check various possible qualified names
                                isReachable reachable memberName ||
                                isReachable reachable (sprintf "%s.%s" fullTypeName memberName) ||
                                isReachable reachable (sprintf "%s.%s.%s" moduleName typeName memberName)
                            | _ -> false
                        | _ -> false
                    )
            )
        
        if List.isEmpty reachableTypeDefs then 
            None
        else 
            Some (SynModuleDecl.Types(reachableTypeDefs, range))
    
    | SynModuleDecl.NestedModule(componentInfo, isRec, nestedDecls, range, trivia, moduleKeyword) ->
        let (SynComponentInfo(_, _, _, longId, _, _, _, _)) = componentInfo
        let nestedModuleName = longId |> List.map (fun id -> id.idText) |> String.concat "."
        let fullNestedName = sprintf "%s.%s" moduleName nestedModuleName
        
        // Recursively prune nested module declarations
        let prunedNestedDecls = 
            nestedDecls 
            |> List.choose (pruneDeclaration reachable fullNestedName)
        
        // Keep the module if it has any reachable content
        if List.isEmpty prunedNestedDecls then 
            None
        else 
            Some (SynModuleDecl.NestedModule(componentInfo, isRec, prunedNestedDecls, range, trivia, moduleKeyword))
    
    | SynModuleDecl.Open(target, range) ->
        // For now, keep all opens since they affect name resolution
        // TODO: Analyze which opens are actually used
        Some decl
    
    | _ -> 
        // Keep other declaration types for now
        Some decl

/// Prune a module based on reachability
let private pruneModule (reachable: Set<string>) (modul: SynModuleOrNamespace) : SynModuleOrNamespace =
    let (SynModuleOrNamespace(longId, isRec, kind, decls, xmlDoc, attrs, synAccess, range, moduleTrivia)) = modul
    let moduleName = longId |> List.map (fun id -> id.idText) |> String.concat "."
    
    // Prune each declaration
    let prunedDecls = 
        decls 
        |> List.choose (pruneDeclaration reachable moduleName)
    
    SynModuleOrNamespace(longId, isRec, kind, prunedDecls, xmlDoc, attrs, synAccess, range, moduleTrivia)

/// Prune entire ParsedInput based on reachability analysis
let prune (reachable: Set<string>) (input: ParsedInput) : ParsedInput =
    
    match input with
    | ParsedInput.ImplFile(implFile) ->
        let (ParsedImplFileInput(fileName, isScript, qualName, pragmas, directives, modules, isLast, trivia, ids)) = implFile
        
        // Prune each module
        let prunedModules = 
            modules 
            |> List.map (pruneModule reachable)
            |> List.filter (fun (SynModuleOrNamespace(_, _, _, decls, _, _, _, _, _)) ->
                // Only keep modules that have content after pruning
                not (List.isEmpty decls)
            )
        
        // Report pruning statistics
        let originalDeclCount = 
            modules |> List.sumBy (fun (SynModuleOrNamespace(_, _, _, decls, _, _, _, _, _)) -> 
                List.length decls)
        let prunedDeclCount = 
            prunedModules |> List.sumBy (fun (SynModuleOrNamespace(_, _, _, decls, _, _, _, _, _)) -> 
                List.length decls)
        
        ParsedInput.ImplFile(
            ParsedImplFileInput(fileName, isScript, qualName, pragmas, directives, prunedModules, isLast, trivia, ids))
    
    | ParsedInput.SigFile(sigFile) ->
        // For now, don't prune signature files
        input

/// Analyze which opens are actually used based on reachable symbols
let private analyzeOpenUsage (reachable: Set<string>) (opens: (string * range) list) : Set<string> =
    // This would require more sophisticated analysis to determine
    // which opens are actually providing symbols that are used
    // For now, we'll keep all opens
    opens |> List.map fst |> Set.ofList

/// More aggressive pruning that also removes unused opens
let pruneWithOpenAnalysis (reachable: Set<string>) (input: ParsedInput) : ParsedInput =
    // First do regular pruning
    let pruned = prune reachable input
    
    // Then analyze and remove unused opens
    // TODO: Implement open usage analysis
    pruned