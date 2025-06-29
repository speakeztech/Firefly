module Core.AST.Traversal

open FSharp.Compiler.Syntax
open FSharp.Compiler.Text

/// Traversal control flow
type TraversalFlow<'result> =
    | Continue of 'result
    | Skip of 'result      // Skip children, return result
    | Stop of 'result      // Stop entire traversal

/// Visitor function signature
type ExprVisitor<'state, 'result> = 'state -> SynExpr -> 'state * TraversalFlow<'result>

/// Result combiner function
type ResultCombiner<'result> = 'result -> 'result list -> 'result

/// Traversal order
type TraversalOrder = 
    | PreOrder   // Visit node before children
    | PostOrder  // Visit node after children

/// Core traversal engine for SynExpr
let traverseExpr (order: TraversalOrder) 
                 (visitor: ExprVisitor<'state, 'result>) 
                 (combine: ResultCombiner<'result>)
                 (initialState: 'state) 
                 (expr: SynExpr) : 'state * 'result =
    
    let rec traverse state expr =
        // Helper to traverse a list of expressions
        let traverseList state exprs =
            exprs |> List.fold (fun (accState, accResults) expr ->
                match traverse accState expr with
                | newState, Continue result -> (newState, result :: accResults)
                | newState, Skip result -> (newState, result :: accResults)
                | newState, Stop result -> (newState, result :: accResults)
            ) (state, [])
            |> fun (s, results) -> s, List.rev results

        // Helper to traverse optional expression
        let traverseOpt state = function
            | Some expr -> 
                let s, flow = traverse state expr
                s, Some flow
            | None -> state, None

        // Pre-order: visit node first
        let visitPre() = 
            let state', flow = visitor state expr
            match flow with
            | Stop result -> state', Stop result
            | Skip result -> state', Skip result
            | Continue result -> state', Continue result

        // Get children based on expression type
        let getChildren state =
            match expr with
            | SynExpr.Paren(expr, _, _) ->
                let s, r = traverse state expr
                s, [r]
                
            | SynExpr.Quote(_, _, quotedExpr, _, _) ->
                let s, r = traverse state quotedExpr
                s, [r]
                
            | SynExpr.Typed(expr, _, _) ->
                let s, r = traverse state expr
                s, [r]
                
            | SynExpr.Tuple(_, exprs, _, _) ->
                traverseList state exprs
                
            | SynExpr.ArrayOrList(_, exprs, _) ->
                traverseList state exprs
                
            | SynExpr.Record(_, _, fields, _) ->
                let exprs = fields |> List.choose (fun (SynExprRecordField(_, _, exprOpt, _)) -> exprOpt)
                traverseList state exprs
                
            | SynExpr.New(_, _, expr, _) ->
                let s, r = traverse state expr
                s, [r]
                
            | SynExpr.ObjExpr(_, _, bindings, _, _, _) ->
                let exprs = bindings |> List.map (fun (SynBinding(_, _, _, _, _, _, _, _, _, expr, _, _, _)) -> expr)
                traverseList state exprs
                
            | SynExpr.While(_, whileExpr, doExpr, _) ->
                let s1, r1 = traverse state whileExpr
                let s2, r2 = traverse s1 doExpr
                s2, [r1; r2]
                
            | SynExpr.For(_, _, _, _, _, toExpr, doExpr, _) ->
                let s1, r1 = traverse state toExpr
                let s2, r2 = traverse s1 doExpr
                s2, [r1; r2]
                
            | SynExpr.ForEach(_, _, _, _, _, enumExpr, bodyExpr, _) ->
                let s1, r1 = traverse state enumExpr
                let s2, r2 = traverse s1 bodyExpr
                s2, [r1; r2]
                
            | SynExpr.App(_, _, funcExpr, argExpr, _) ->
                let s1, r1 = traverse state funcExpr
                let s2, r2 = traverse s1 argExpr
                s2, [r1; r2]
                
            | SynExpr.LetOrUse(_, _, bindings, bodyExpr, _, _) ->
                let bindingExprs = bindings |> List.map (fun (SynBinding(_, _, _, _, _, _, _, _, _, expr, _, _, _)) -> expr)
                let s1, bindingResults = traverseList state bindingExprs
                let s2, bodyResult = traverse s1 bodyExpr
                s2, bindingResults @ [bodyResult]
                
            | SynExpr.TryWith(tryExpr, clauses, _, _, _, _) ->
                let s1, r1 = traverse state tryExpr
                let clauseExprs = clauses |> List.map (fun (SynMatchClause(_, _, expr, _, _, _)) -> expr)
                let s2, clauseResults = traverseList s1 clauseExprs
                s2, r1 :: clauseResults
                
            | SynExpr.TryFinally(tryExpr, finallyExpr, _, _, _, _) ->
                let s1, r1 = traverse state tryExpr
                let s2, r2 = traverse s1 finallyExpr
                s2, [r1; r2]
                
            | SynExpr.Lazy(expr, _) ->
                let s, r = traverse state expr
                s, [r]
                
            | SynExpr.Sequential(_, _, expr1, expr2, _, _) ->
                let s1, r1 = traverse state expr1
                let s2, r2 = traverse s1 expr2
                s2, [r1; r2]
                
            | SynExpr.IfThenElse(ifExpr, thenExpr, elseExprOpt, _, _, _, _) ->
                let s1, r1 = traverse state ifExpr
                let s2, r2 = traverse s1 thenExpr
                match elseExprOpt with
                | Some elseExpr ->
                    let s3, r3 = traverse s2 elseExpr
                    s3, [r1; r2; r3]
                | None ->
                    s2, [r1; r2]
                    
            | SynExpr.Lambda(_, _, _, bodyExpr, _, _, _) ->
                let s, r = traverse state bodyExpr
                s, [r]
                
            | SynExpr.Match(_, matchExpr, clauses, _, _) ->
                let s1, r1 = traverse state matchExpr
                let clauseExprs = clauses |> List.map (fun (SynMatchClause(_, whenOpt, resultExpr, _, _, _)) -> 
                    match whenOpt with
                    | Some whenExpr -> [whenExpr; resultExpr]
                    | None -> [resultExpr]
                ) |> List.concat
                let s2, clauseResults = traverseList s1 clauseExprs
                s2, r1 :: clauseResults
                
            | SynExpr.Do(expr, _) ->
                let s, r = traverse state expr
                s, [r]
                
            | SynExpr.Assert(expr, _) ->
                let s, r = traverse state expr
                s, [r]
                
            | SynExpr.TypeApp(expr, _, _, _, _, _, _) ->
                let s, r = traverse state expr
                s, [r]
                
            | SynExpr.Downcast(expr, _, _) ->
                let s, r = traverse state expr
                s, [r]
                
            | SynExpr.Upcast(expr, _, _) ->
                let s, r = traverse state expr
                s, [r]
                
            | SynExpr.DotGet(expr, _, _, _) ->
                let s, r = traverse state expr
                s, [r]
                
            | SynExpr.DotSet(expr1, _, expr2, _) ->
                let s1, r1 = traverse state expr1
                let s2, r2 = traverse s1 expr2
                s2, [r1; r2]
                
            | SynExpr.DotIndexedGet(objectExpr, indexArgs, _, _) ->
                let s1, r1 = traverse state objectExpr
                let argExprs = indexArgs |> List.map (fun arg -> arg.Expr)
                let s2, argResults = traverseList s1 argExprs
                s2, r1 :: argResults
                
            | SynExpr.DotIndexedSet(objectExpr, indexArgs, valueExpr, _, _, _) ->
                let s1, r1 = traverse state objectExpr
                let argExprs = indexArgs |> List.map (fun arg -> arg.Expr)
                let s2, argResults = traverseList s1 argExprs
                let s3, r3 = traverse s2 valueExpr
                s3, r1 :: argResults @ [r3]
                
            | SynExpr.TypeTest(expr, _, _) ->
                let s, r = traverse state expr
                s, [r]
                
            | SynExpr.AddressOf(_, expr, _, _) ->
                let s, r = traverse state expr
                s, [r]
                
            | SynExpr.TraitCall(_, _, expr, _) ->
                let s, r = traverse state expr
                s, [r]
                
            | SynExpr.YieldOrReturn(_, expr, _) ->
                let s, r = traverse state expr
                s, [r]
                
            | SynExpr.YieldOrReturnFrom(_, expr, _) ->
                let s, r = traverse state expr
                s, [r]
                
            | SynExpr.LetOrUseBang(_, _, _, _, _, rhsExpr, bodyExpr, _, _) ->
                let s1, r1 = traverse state rhsExpr
                let s2, r2 = traverse s1 bodyExpr
                s2, [r1; r2]
                
            | SynExpr.MatchBang(_, expr, clauses, _, _) ->
                let s1, r1 = traverse state expr
                let clauseExprs = clauses |> List.map (fun (SynMatchClause(_, _, expr, _, _, _)) -> expr)
                let s2, clauseResults = traverseList s1 clauseExprs
                s2, r1 :: clauseResults
                
            | SynExpr.DoBang(expr, _) ->
                let s, r = traverse state expr
                s, [r]
                
            | SynExpr.LibraryOnlyILAssembly _
            | SynExpr.LibraryOnlyStaticOptimization _
            | SynExpr.LibraryOnlyUnionCaseFieldGet _
            | SynExpr.LibraryOnlyUnionCaseFieldSet _ ->
                state, []
                
            // Leaf nodes (no children)
            | SynExpr.Const _
            | SynExpr.Ident _
            | SynExpr.LongIdent _
            | SynExpr.LongIdentSet _
            | SynExpr.Null _
            | SynExpr.ImplicitZero _
            | SynExpr.MatchLambda _
            | SynExpr.ArbitraryAfterError _
            | SynExpr.FromParseError _
            | SynExpr.DiscardAfterMissingQualificationAfterDot _
            | SynExpr.Fixed _
            | SynExpr.InterpolatedString _
            | SynExpr.Sequential _ ->  // Already handled above
                state, []

        // Execute traversal based on order
        match order with
        | PreOrder ->
            let state', nodeFlow = visitPre()
            match nodeFlow with
            | Stop result -> state', Stop result
            | Skip result -> state', Continue result  // Skip children but continue traversal
            | Continue nodeResult ->
                let childState, childFlows = getChildren state'
                let childResults = childFlows |> List.choose (function
                    | Continue r -> Some r
                    | Skip r -> Some r
                    | Stop r -> Some r)
                childState, Continue (combine nodeResult childResults)
                
        | PostOrder ->
            let childState, childFlows = getChildren state
            let childResults = childFlows |> List.choose (function
                | Continue r -> Some r
                | Skip r -> Some r
                | Stop r -> Some r)
            let state', nodeFlow = visitor childState expr
            match nodeFlow with
            | Stop result -> state', Stop result
            | Skip result -> state', Continue result
            | Continue nodeResult -> state', Continue (combine nodeResult childResults)

    let finalState, flow = traverse initialState expr
    match flow with
    | Continue result -> finalState, result
    | Skip result -> finalState, result
    | Stop result -> finalState, result

/// Traverse module declarations
let rec traverseModuleDecl (order: TraversalOrder)
                          (visitor: ExprVisitor<'state, 'result>)
                          (combine: ResultCombiner<'result>)
                          (state: 'state)
                          (decl: SynModuleDecl) : 'state * 'result list =
    match decl with
    | SynModuleDecl.Let(_, bindings, _) ->
        bindings |> List.fold (fun (accState, accResults) binding ->
            let (SynBinding(_, _, _, _, _, _, _, _, _, expr, _, _, _)) = binding
            let newState, result = traverseExpr order visitor combine accState expr
            (newState, result :: accResults)
        ) (state, [])
        |> fun (s, results) -> s, List.rev results
        
    | SynModuleDecl.Expr(expr, _) ->
        let s, r = traverseExpr order visitor combine state expr
        s, [r]
        
    | _ -> state, []

/// Common visitor patterns
module Patterns =
    /// Collect all identifiers
    let collectIdentifiers : ExprVisitor<unit, Set<string>> =
        fun () expr ->
            match expr with
            | SynExpr.Ident ident ->
                (), Continue (Set.singleton ident.idText)
            | SynExpr.LongIdent(_, SynLongIdent(ids, _, _), _, _) ->
                let names = ids |> List.map (fun id -> id.idText) |> Set.ofList
                (), Continue names
            | _ ->
                (), Continue Set.empty
                
    /// Transform expressions
    let transformer (f: SynExpr -> SynExpr option) : ExprVisitor<unit, SynExpr option> =
        fun () expr ->
            match f expr with
            | Some newExpr -> (), Skip (Some newExpr)  // Don't traverse into transformed nodes
            | None -> (), Continue None
            
    /// Dependency collector
    let collectDependencies : ExprVisitor<Set<string>, Set<string>> =
        fun knownSymbols expr ->
            match expr with
            | SynExpr.Ident ident when not (Set.contains ident.idText knownSymbols) ->
                knownSymbols, Continue (Set.singleton ident.idText)
            | SynExpr.App(_, _, SynExpr.Ident funcId, _, _) when not (Set.contains funcId.idText knownSymbols) ->
                knownSymbols, Continue (Set.singleton funcId.idText)
            | _ ->
                knownSymbols, Continue Set.empty

/// Combiner functions
module Combiners =
    let setUnion : ResultCombiner<Set<'a>> = 
        fun nodeResult childResults -> Set.unionMany (nodeResult :: childResults)
        
    let listConcat : ResultCombiner<'a list> =
        fun nodeResult childResults -> nodeResult @ (List.concat childResults)
        
    let optionOr : ResultCombiner<'a option> =
        fun nodeResult childResults ->
            match nodeResult with
            | Some _ -> nodeResult
            | None -> childResults |> List.tryFind Option.isSome |> Option.flatten