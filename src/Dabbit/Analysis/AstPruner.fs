module Dabbit.Analysis.AstPruner

open FSharp.Compiler.Syntax
open FSharp.Compiler.Text
open Core.FCSIngestion.SymbolExtraction

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

/// Prune module declarations to only include reachable symbols
let rec pruneModule (reachableSymbols: Set<string>) (modulePath: string list) (decls: SynModuleDecl list) =
    decls |> List.choose (fun decl ->
        match decl with
        | SynModuleDecl.Let(isRec, bindings, range) ->
            let prunedBindings = 
                bindings |> List.filter (fun binding ->
                    let (SynBinding(_, _, _, _, _, _, _, pat, _, _, _, _, _)) = binding
                    match extractSymbolFromPattern modulePath pat with
                    | Some extractedSymbol -> Set.contains extractedSymbol.QualifiedName reachableSymbols
                    | None -> false)
            if List.isEmpty prunedBindings then None
            else Some (SynModuleDecl.Let(isRec, prunedBindings, range))
            
        | SynModuleDecl.NestedModule(compInfo, isRec, nestedDecls, isContinuing, range, trivia) ->
            let (SynComponentInfo(attrs, typeParams, constraints, longId, xmlDoc, preferPostfix, access, range2)) = compInfo
            let nestedPath = modulePath @ (longId |> List.map (fun id -> id.idText))
            let nestedModuleName = String.concat "." nestedPath
            
            // Check if ANY symbol in this nested module or its children is reachable
            let hasReachableContent = 
                reachableSymbols |> Set.exists (fun symbol ->
                    symbol.StartsWith(nestedModuleName + ".") || symbol = nestedModuleName)
            
            if hasReachableContent then
                let prunedNestedDecls = pruneModule reachableSymbols nestedPath nestedDecls
                Some (SynModuleDecl.NestedModule(compInfo, isRec, prunedNestedDecls, isContinuing, range, trivia))
            else 
                None
                
        | SynModuleDecl.Types(types, range) ->
            let prunedTypes = 
                types |> List.choose (fun typeDef ->
                    let (SynTypeDefn(compInfo, repr, members, implicitCtor, range2, trivia)) = typeDef
                    let (SynComponentInfo(_, _, _, longId, _, _, _, _)) = compInfo
                    let typeName = (modulePath @ [longId.Head.idText]) |> String.concat "."
                    
                    // First check if the type itself is reachable
                    if Set.contains typeName reachableSymbols then
                        // Keep the entire type if it's explicitly reachable
                        Some typeDef
                    else
                        // Check if any members are reachable
                        let reachableMembers = 
                            members |> List.filter (fun memberDefn ->
                                match memberDefn with
                                | SynMemberDefn.Member(binding, _) ->
                                    let (SynBinding(_, _, _, _, _, _, _, pat, _, _, _, _, _)) = binding
                                    match extractMemberName pat with
                                    | Some memberName ->
                                        let fullMemberName = typeName + "." + memberName
                                        Set.contains fullMemberName reachableSymbols
                                    | None -> false
                                | _ -> false)
                        
                        if not (List.isEmpty reachableMembers) then
                            // Only keep reachable members
                            Some (SynTypeDefn(compInfo, repr, reachableMembers, implicitCtor, range2, trivia))
                        else
                            // No members are reachable - remove the entire type
                            None)
            
            if List.isEmpty prunedTypes then None
            else Some (SynModuleDecl.Types(prunedTypes, range))
            
        | SynModuleDecl.Open _ -> 
            // Always keep open declarations - they're needed for name resolution
            Some decl
            
        | SynModuleDecl.Attributes _
        | SynModuleDecl.HashDirective _ ->
            // Keep these structural elements
            Some decl
            
        | SynModuleDecl.ModuleAbbrev _
        | SynModuleDecl.Exception _
        | SynModuleDecl.NamespaceFragment _ ->
            // For now, remove these
            None)

/// Prune declarations within a namespace (which typically contains modules)
let rec pruneNamespaceDecls (reachableSymbols: Set<string>) (namespacePath: string list) (decls: SynModuleDecl list) =
    decls |> List.choose (fun decl ->
        match decl with
        | SynModuleDecl.NestedModule(compInfo, isRec, nestedDecls, isContinuing, range, trivia) ->
            let (SynComponentInfo(attrs, typeParams, constraints, longId, xmlDoc, preferPostfix, access, range2)) = compInfo
            let modulePath = namespacePath @ (longId |> List.map (fun id -> id.idText))
            let moduleFullName = String.concat "." modulePath
            
            // Check if any symbol in this module is reachable
            let hasReachableSymbols = 
                reachableSymbols |> Set.exists (fun symbol ->
                    symbol.StartsWith(moduleFullName + ".") || symbol = moduleFullName)
            
            if hasReachableSymbols then
                // Recursively prune the module's content
                let prunedModuleDecls = pruneModule reachableSymbols modulePath nestedDecls
                // Keep the module structure even if empty - needed for compilation
                Some (SynModuleDecl.NestedModule(compInfo, isRec, prunedModuleDecls, isContinuing, range, trivia))
            else
                None
                
        | _ -> 
            // For other declarations in namespace, delegate to module pruning
            pruneModule reachableSymbols namespacePath [decl] |> List.tryHead)

/// Prune AST to only include reachable symbols
let prune (reachableSymbols: Set<string>) (input: ParsedInput) : ParsedInput =
    match input with
    | ParsedInput.ImplFile(ParsedImplFileInput(fileName, isScript, qualName, pragmas, hashDirectives, modules, isLastCompiled, isExe, trivia)) ->
        let prunedModules = 
            modules |> List.choose (fun modOrNs ->
                match modOrNs with
                | SynModuleOrNamespace(longId, isRec, kind, decls, xmlDoc, attrs, access, range, trivia2) ->
                    let path = longId |> List.map (fun id -> id.idText)
                    
                    match kind with
                    | SynModuleOrNamespaceKind.DeclaredNamespace
                    | SynModuleOrNamespaceKind.GlobalNamespace ->
                        // This is a namespace - need to check all modules within it
                        let namespaceName = String.concat "." path
                        
                        // Check if ANY symbol in this namespace is reachable
                        let hasReachableContent = 
                            reachableSymbols |> Set.exists (fun symbol ->
                                symbol.StartsWith(namespaceName + "."))
                        
                        if hasReachableContent then
                            // Prune the namespace contents but keep the namespace structure
                            let prunedDecls = pruneNamespaceDecls reachableSymbols path decls
                            Some (SynModuleOrNamespace(longId, isRec, kind, prunedDecls, xmlDoc, attrs, access, range, trivia2))
                        else
                            None
                            
                    | SynModuleOrNamespaceKind.NamedModule
                    | SynModuleOrNamespaceKind.AnonModule ->
                        // This is a module (not a namespace)
                        let prunedDecls = pruneModule reachableSymbols path decls
                        if List.isEmpty prunedDecls then None
                        else Some (SynModuleOrNamespace(longId, isRec, kind, prunedDecls, xmlDoc, attrs, access, range, trivia2)))
        
        ParsedInput.ImplFile(ParsedImplFileInput(fileName, isScript, qualName, pragmas, hashDirectives, prunedModules, isLastCompiled, isExe, trivia))
    
    | ParsedInput.SigFile _ -> input  // TODO: Implement signature pruning

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