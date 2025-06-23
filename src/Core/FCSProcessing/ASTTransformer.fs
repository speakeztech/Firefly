module Core.FCSProcessing.ASTTransformer

open FSharp.Compiler.Syntax
open FSharp.Compiler.Symbols
open FSharp.Compiler.CodeAnalysis

/// Transformation context flowing through pipeline
type TransformContext = {
    TypeContext: Core.MLIRGeneration.TypeMapping.TypeContext
    SymbolRegistry: Dabbit.Bindings.SymbolRegistry.SymbolRegistry
    Reachability: Dabbit.Analysis.ReachabilityAnalyzer.ReachabilityResult
    ClosureState: Dabbit.Transformations.ClosureElimination.ClosureState
}

/// Apply all AST transformations in sequence
let transformAST (ctx: TransformContext) (input: ParsedInput) =
    // Extract declarations for transformation
    let getDecls = function
        | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, modules, _, _, _, _)) ->
            modules |> List.collect (fun (SynModuleOrNamespace(_, _, _, decls, _, _, _, _, _)) -> decls)
        | _ -> []
    
    // Extract bindings from declarations
    let getBindings decls =
        decls |> List.collect (function
            | SynModuleDecl.Let(_, bindings, _) -> bindings
            | _ -> [])
    
    // 1. Stack allocation transformation
    let stackTransformed = 
        match input with
        | ParsedInput.ImplFile impl ->
            let transformed = Dabbit.Transformations.StackAllocation.StackTransform.transform
            ParsedInput.ImplFile { impl with Contents = 
                impl.Contents |> List.map (fun modul ->
                    let (SynModuleOrNamespace(lid, isRec, kind, decls, xml, attrs, access, range, trivia)) = modul
                    let decls' = decls |> List.map (function
                        | SynModuleDecl.Let(isRec, bindings, range) ->
                            let bindings' = bindings |> List.map (fun binding ->
                                let (SynBinding(a, k, inl, mut, attrs, xml, valData, pat, ret, expr, r, sp, tr)) = binding
                                SynBinding(a, k, inl, mut, attrs, xml, valData, pat, ret, transformed expr, r, sp, tr))
                            SynModuleDecl.Let(isRec, bindings', range)
                        | decl -> decl)
                    SynModuleOrNamespace(lid, isRec, kind, decls', xml, attrs, access, range, trivia))}
        | sig_ -> sig_
    
    // 2. Closure elimination
    let closureEliminated =
        let bindings = getBindings (getDecls stackTransformed)
        let transformed = Dabbit.Transformations.ClosureElimination.transformModule bindings
        
        // Reconstruct the AST with transformed bindings
        match stackTransformed with
        | ParsedInput.ImplFile impl ->
            ParsedInput.ImplFile { impl with Contents = 
                impl.Contents |> List.map (fun modul ->
                    let (SynModuleOrNamespace(lid, isRec, kind, _, xml, attrs, access, range, trivia)) = modul
                    let decls' = [SynModuleDecl.Let(false, transformed, range)]
                    SynModuleOrNamespace(lid, isRec, kind, decls', xml, attrs, access, range, trivia))}
        | sig_ -> sig_
    
    // 3. Tree shaking (pruning)
    let pruned = Dabbit.Analysis.AstPruner.prune ctx.Reachability.Reachable closureEliminated
    
    pruned

/// Verify transformations maintain zero-allocation guarantees
let verifyTransformations (input: ParsedInput) =
    let getExprs = function
        | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, modules, _, _, _, _)) ->
            modules |> List.collect (fun (SynModuleOrNamespace(_, _, _, decls, _, _, _, _, _)) ->
                decls |> List.collect (function
                    | SynModuleDecl.Let(_, bindings, _) ->
                        bindings |> List.map (fun (SynBinding(_, _, _, _, _, _, _, _, _, expr, _, _, _)) -> expr)
                    | _ -> []))
        | _ -> []
    
    let exprs = getExprs input
    let results = exprs |> List.map Dabbit.Transformations.StackAllocation.StackSafety.verify
    
    match results |> List.tryFind Result.isError with
    | Some (Error msg) -> Error msg
    | _ -> Ok ()