module Dabbit.Transformations.ClosureElimination

open FSharp.Compiler.Syntax
open FSharp.Compiler.Text

/// Closure elimination state
type ClosureState = {
    Counter: int
    Scope: Set<string>
    Lifted: (string * SynBinding) list
}

/// Free variable analysis
module FreeVars =
    let rec analyze scope = function
        | SynExpr.Ident ident -> 
            if Set.contains ident.idText scope then Set.empty 
            else Set.singleton ident.idText
        
        | SynExpr.App(_, _, func, arg, _) ->
            Set.union (analyze scope func) (analyze scope arg)
        
        | SynExpr.Lambda(_, _, args, body, _, _, _) ->
            let argNames = 
                match args with
                | SynSimplePats.SimplePats(pats, _) ->
                    pats |> List.choose (function
                        | SynSimplePat.Id(ident, _, _, _, _, _) -> Some ident.idText
                        | _ -> None)
                    |> Set.ofList
                | _ -> Set.empty
            analyze (Set.union scope argNames) body
        
        | SynExpr.LetOrUse(_, _, bindings, body, _, _) ->
            let bindingVars = 
                bindings |> List.choose (fun (SynBinding(_, _, _, _, _, _, _, pat, _, _, _, _, _)) ->
                    match pat with
                    | SynPat.Named(SynIdent(ident, _), _, _, _) -> Some ident.idText
                    | _ -> None)
                |> Set.ofList
            
            let bindingFrees = 
                bindings |> List.map (fun (SynBinding(_, _, _, _, _, _, _, _, _, expr, _, _, _)) ->
                    analyze scope expr)
                |> Set.unionMany
            
            Set.union bindingFrees (analyze (Set.union scope bindingVars) body)
        
        | SynExpr.IfThenElse(cond, thenExpr, elseOpt, _, _, _, _) ->
            let condFree = analyze scope cond
            let thenFree = analyze scope thenExpr
            let elseFree = elseOpt |> Option.map (analyze scope) |> Option.defaultValue Set.empty
            Set.unionMany [condFree; thenFree; elseFree]
        
        | SynExpr.Match(_, expr, clauses, _, _) ->
            let exprFree = analyze scope expr
            let clausesFree = 
                clauses |> List.map (fun (SynMatchClause(pat, whenOpt, result, _, _, _)) ->
                    let patVars = extractPatternVars pat
                    let whenFree = whenOpt |> Option.map (analyze scope) |> Option.defaultValue Set.empty
                    let resultFree = analyze (Set.union scope patVars) result
                    Set.union whenFree resultFree)
                |> Set.unionMany
            Set.union exprFree clausesFree
        
        | SynExpr.Sequential(_, _, e1, e2, _) ->
            Set.union (analyze scope e1) (analyze scope e2)
        
        | _ -> Set.empty
    
    and extractPatternVars = function
        | SynPat.Named(SynIdent(ident, _), _, _, _) -> Set.singleton ident.idText
        | SynPat.Paren(pat, _) -> extractPatternVars pat
        | SynPat.Tuple(_, pats, _) -> pats |> List.map extractPatternVars |> Set.unionMany
        | _ -> Set.empty

/// Transform expressions to eliminate closures
let rec transform (state: ClosureState) expr =
    match expr with
    | SynExpr.Lambda(fromMethod, inSeq, args, body, parsedData, range, trivia) ->
        let freeVars = FreeVars.analyze state.Scope body
        if Set.isEmpty freeVars then
            (expr, state)
        else
            // Lift lambda to top-level function
            let liftedName = sprintf "_closure_%d" state.Counter
            let liftedIdent = Ident(liftedName, range)
            
            // Create parameters for captured variables
            let capturedParams = 
                freeVars 
                |> Set.toList
                |> List.map (fun v -> 
                    let ident = Ident(v, range)
                    SynPat.Named(SynIdent(ident, None), false, None, range))
            
            // Create lifted function binding
            let allArgs = 
                match args with
                | SynSimplePats.SimplePats(pats, r) ->
                    let captured = capturedParams |> List.map (fun p ->
                        SynSimplePat.Id(Ident("_cap", range), None, false, false, false, range))
                    SynSimplePats.SimplePats(captured @ pats, r)
                | SynSimplePats.Typed(pats, ty, r) ->
                    SynSimplePats.SimplePats([], r) // Simplified
            
            let liftedBinding = 
                SynBinding(None, SynBindingKind.Normal, false, false, [], 
                          PreXmlDoc.Empty, SynValData(None, SynValInfo([], SynArgInfo([], false, None)), None),
                          SynPat.LongIdent(SynLongIdent([liftedIdent], [], [None]), None, None, 
                                         SynArgPats.Pats [], None, range),
                          None, body, range, DebugPointAtBinding.NoneAtInvisible, SynBindingTrivia.Zero)
            
            let newState = { state with 
                             Counter = state.Counter + 1
                             Lifted = (liftedName, liftedBinding) :: state.Lifted }
            
            // Replace with application
            let replacement = 
                freeVars 
                |> Set.toList
                |> List.fold (fun acc var ->
                    SynExpr.App(ExprAtomicFlag.NonAtomic, false, acc, 
                              SynExpr.Ident(Ident(var, range)), range))
                    (SynExpr.Ident liftedIdent)
            
            (replacement, newState)
    
    | SynExpr.App(flag, isInfix, func, arg, range) ->
        let (func', state1) = transform state func
        let (arg', state2) = transform state1 arg
        (SynExpr.App(flag, isInfix, func', arg', range), state2)
    
    | SynExpr.LetOrUse(isRec, isUse, bindings, body, range, trivia) ->
        let (bindings', state1) = 
            bindings |> List.fold (fun (accBindings, accState) binding ->
                let (SynBinding(access, kind, isInline, isMut, attrs, xmlDoc, valData, 
                              headPat, retInfo, expr, bindRange, debugPoint, bindTrivia)) = binding
                let (expr', newState) = transform accState expr
                let binding' = SynBinding(access, kind, isInline, isMut, attrs, xmlDoc, valData,
                                        headPat, retInfo, expr', bindRange, debugPoint, bindTrivia)
                (binding' :: accBindings, newState))
                ([], state)
        let (body', state2) = transform state1 body
        (SynExpr.LetOrUse(isRec, isUse, List.rev bindings', body', range, trivia), state2)
    
    | _ -> (expr, state)

/// Transform a module's bindings
let transformModule (bindings: SynBinding list) =
    let initialState = { Counter = 0; Scope = Set.empty; Lifted = [] }
    
    let (transformed, finalState) =
        bindings |> List.fold (fun (accBindings, accState) binding ->
            let (SynBinding(access, kind, isInline, isMut, attrs, xmlDoc, valData, 
                          headPat, retInfo, expr, range, debugPoint, trivia)) = binding
            let (expr', newState) = transform accState expr
            let binding' = SynBinding(access, kind, isInline, isMut, attrs, xmlDoc, valData,
                                    headPat, retInfo, expr', range, debugPoint, trivia)
            (binding' :: accBindings, newState))
            ([], initialState)
    
    // Add lifted functions to module
    let liftedBindings = finalState.Lifted |> List.map snd |> List.rev
    liftedBindings @ (List.rev transformed)