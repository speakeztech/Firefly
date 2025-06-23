module Dabbit.Analysis.AstPruner

open FSharp.Compiler.Syntax
open FSharp.Compiler.Text

/// Prune unreachable declarations from module
let pruneModule (reachable: Set<string>) (moduleName: string) (decls: SynModuleDecl list) =
    let isReachable name = Set.contains (sprintf "%s.%s" moduleName name) reachable
    
    let rec pruneDecl = function
        | SynModuleDecl.Let(isRec, bindings, range) ->
            let kept = bindings |> List.filter (fun binding ->
                let (SynBinding(_, _, _, _, _, _, _, pat, _, _, _, _, _)) = binding
                match pat with
                | SynPat.Named(SynIdent(ident, _), _, _, _) -> isReachable ident.idText
                | SynPat.LongIdent(SynLongIdent(ids, _, _), _, _, _, _, _) ->
                    ids |> List.exists (fun id -> isReachable id.idText)
                | _ -> true)
            
            if List.isEmpty kept then None
            else Some (SynModuleDecl.Let(isRec, kept, range))
        
        | SynModuleDecl.Types(types, range) ->
            let kept = types |> List.filter (fun typeDef ->
                let (SynTypeDefn(SynComponentInfo(_, _, _, longId, _, _, _, _), _, _, _, _, _)) = typeDef
                longId |> List.exists (fun id -> isReachable id.idText))
            
            if List.isEmpty kept then None
            else Some (SynModuleDecl.Types(kept, range))
        
        | SynModuleDecl.NestedModule(componentInfo, isRec, nestedDecls, range, trivia, attrs) ->
            let (SynComponentInfo(_, _, _, longId, _, _, _, _)) = componentInfo
            let nestedName = longId |> List.map (fun id -> id.idText) |> String.concat "."
            let fullName = sprintf "%s.%s" moduleName nestedName
            let pruned = nestedDecls |> List.choose pruneDecl
            
            if List.isEmpty pruned then None
            else Some (SynModuleDecl.NestedModule(componentInfo, isRec, pruned, range, trivia, attrs))
        
        | decl -> Some decl  // Keep other declarations
    
    decls |> List.choose pruneDecl

/// Prune entire ParsedInput
let prune (reachable: Set<string>) (input: ParsedInput) =
    match input with
    | ParsedInput.ImplFile(ParsedImplFileInput(name, script, qname, attrs, mods, implFiles, binds, trivia, idents)) ->
        let prunedMods = mods |> List.map (fun modul ->
            let (SynModuleOrNamespace(longId, isRec, kind, decls, xmlDoc, attrs, synAccess, range, trivia)) = modul
            let moduleName = longId |> List.map (fun id -> id.idText) |> String.concat "."
            let prunedDecls = pruneModule reachable moduleName decls
            SynModuleOrNamespace(longId, isRec, kind, prunedDecls, xmlDoc, attrs, synAccess, range, trivia))
        ParsedInput.ImplFile(ParsedImplFileInput(name, script, qname, attrs, prunedMods, implFiles, binds, trivia, idents))
    
    | ParsedInput.SigFile _ -> input  // Don't prune signatures