module Dabbit.Analysis.DependencyGraphBuilder

open FSharp.Compiler.Symbols
open FSharp.Compiler.Syntax

/// Dependency types in FCS terms
type Dependency =
    | CallDep of FSharpMemberOrFunctionOrValue
    | TypeDep of FSharpEntity  
    | ModuleDep of FSharpEntity

/// Dependency graph for reachability
type DependencyGraph = {
    Nodes: Map<string, FSharpSymbol>
    Edges: Map<string, Set<string>>
    Roots: Set<string>
}

/// Build dependencies from expressions
let rec analyzeDeps expr =
    match expr with
    | SynExpr.Ident ident ->
        Set.singleton ident.idText
    
    | SynExpr.LongIdent(_, SynLongIdent(ids, _, _), _, _) ->
        Set.singleton (ids |> List.map (fun i -> i.idText) |> String.concat ".")
    
    | SynExpr.App(_, _, func, arg, _) ->
        Set.union (analyzeDeps func) (analyzeDeps arg)
    
    | SynExpr.TypeApp(expr, _, _, _, _, _, _) ->
        analyzeDeps expr
    
    | SynExpr.LetOrUse(_, _, bindings, body, _, _) ->
        let bindingDeps = 
            bindings 
            |> List.map (fun (SynBinding(_, _, _, _, _, _, _, _, _, expr, _, _, _)) -> analyzeDeps expr)
            |> Set.unionMany
        Set.union bindingDeps (analyzeDeps body)
    
    | SynExpr.IfThenElse(cond, thenExpr, elseOpt, _, _, _, _) ->
        [Some cond; Some thenExpr; elseOpt]
        |> List.choose id
        |> List.map analyzeDeps
        |> Set.unionMany
    
    | SynExpr.Match(_, expr, clauses, _, _) ->
        let exprDeps = analyzeDeps expr
        let clauseDeps = 
            clauses 
            |> List.map (fun (SynMatchClause(_, _, result, _, _, _)) -> analyzeDeps result)
            |> Set.unionMany
        Set.union exprDeps clauseDeps
    
    | SynExpr.Sequential(_, _, e1, e2, _, _) ->
        Set.union (analyzeDeps e1) (analyzeDeps e2)
    
    | SynExpr.Lambda(_, _, _, body, _, _, _) ->
        analyzeDeps body
    
    | SynExpr.Tuple(_, exprs, _, _) ->
        exprs |> List.map analyzeDeps |> Set.unionMany
    
    | SynExpr.ArrayOrList(_, exprs, _) ->
        exprs |> List.map analyzeDeps |> Set.unionMany
    
    | SynExpr.Record(_, _, fields, _) ->
        fields 
        |> List.choose (fun (SynExprRecordField(_, _, expr, _)) -> expr)
        |> List.map analyzeDeps
        |> Set.unionMany
    
    | SynExpr.TypeTest(expr, _, _) ->
        analyzeDeps expr
    
    | SynExpr.Downcast(expr, _, _) ->
        analyzeDeps expr
    
    | SynExpr.Upcast(expr, _, _) ->
        analyzeDeps expr
    
    | SynExpr.DotGet(expr, _, _, _) ->
        analyzeDeps expr
    
    | SynExpr.DotSet(expr, _, rhsExpr, _) ->
        Set.union (analyzeDeps expr) (analyzeDeps rhsExpr)
    
    | _ -> Set.empty

/// Build graph from module declarations
let buildFromModule (moduleName: string) (decls: SynModuleDecl list) =
    let rec processDecl graph = function
        | SynModuleDecl.Let(_, bindings, _) ->
            bindings |> List.fold (fun g binding ->
                let (SynBinding(_, _, _, _, _, _, _, pat, _, expr, _, _, _)) = binding
                match pat with
                | SynPat.Named(SynIdent(ident, _), _, _, _) ->
                    let name = sprintf "%s.%s" moduleName ident.idText
                    let deps = analyzeDeps expr
                    { g with Edges = Map.add name deps g.Edges }
                | SynPat.LongIdent(SynLongIdent(ids, _, _), _, _, _, _, _) ->
                    let name = sprintf "%s.%s" moduleName (ids |> List.map (fun i -> i.idText) |> String.concat ".")
                    let deps = analyzeDeps expr
                    { g with Edges = Map.add name deps g.Edges }
                | _ -> g) graph
        
        | SynModuleDecl.Types(typeDefs, _) ->
            typeDefs |> List.fold (fun g typeDef ->
                let (SynTypeDefn(componentInfo, _, memberDefns, _, _, _)) = typeDef
                let (SynComponentInfo(_, _, _, longId, _, _, _, _)) = componentInfo
                let typeName = sprintf "%s.%s" moduleName (longId |> List.map (fun i -> i.idText) |> String.concat ".")
                
                // Process member definitions
                memberDefns |> List.fold (fun g2 memberDefn ->
                    match memberDefn with
                    | SynMemberDefn.Member(binding, _) ->
                        let (SynBinding(_, _, _, _, _, _, _, pat, _, expr, _, _, _)) = binding
                        let memberName = 
                            match pat with
                            | SynPat.LongIdent(SynLongIdent(ids, _, _), _, _, _, _, _) ->
                                sprintf "%s.%s" typeName (ids |> List.map (fun i -> i.idText) |> String.concat ".")
                            | SynPat.Named(SynIdent(ident, _), _, _, _) ->
                                sprintf "%s.%s" typeName ident.idText
                            | _ -> typeName
                        let deps = analyzeDeps expr
                        { g2 with Edges = Map.add memberName deps g2.Edges }
                    | _ -> g2) g) graph
        
        | SynModuleDecl.NestedModule(componentInfo, _, nestedDecls, _, _, _) ->
            let (SynComponentInfo(_, _, _, longId, _, _, _, _)) = componentInfo
            let nestedModuleName = sprintf "%s.%s" moduleName (longId |> List.map (fun i -> i.idText) |> String.concat ".")
            nestedDecls |> List.fold processDecl graph
        
        | _ -> graph
    
    let emptyGraph = { Nodes = Map.empty; Edges = Map.empty; Roots = Set.empty }
    decls |> List.fold processDecl emptyGraph

/// Perform reachability analysis
let findReachable (graph: DependencyGraph) =
    let rec traverse visited toVisit =
        match toVisit with
        | [] -> visited
        | node :: rest ->
            if Set.contains node visited then
                traverse visited rest
            else
                let visited' = Set.add node visited
                let neighbors = 
                    Map.tryFind node graph.Edges
                    |> Option.defaultValue Set.empty
                    |> Set.toList
                traverse visited' (neighbors @ rest)
    
    traverse Set.empty (Set.toList graph.Roots)