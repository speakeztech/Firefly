module Core.FCSProcessing.ASTTransformer

open FSharp.Compiler.Syntax
open FSharp.Compiler.Symbols
open FSharp.Compiler.CodeAnalysis
open Core.XParsec.Foundation
open XParsec

/// Transformation context flowing through pipeline
type TransformContext = {
    TypeContext: Core.MLIRGeneration.TypeMapping.TypeContext
    SymbolRegistry: Dabbit.Bindings.SymbolRegistry.SymbolRegistry
    Reachability: Dabbit.Analysis.ReachabilityAnalyzer.ReachabilityResult
    ClosureState: Dabbit.Transformations.ClosureElimination.ClosureState
}

/// AST Transformer type - transforms one AST node to another
type ASTTransformer<'T> = 'T -> TransformContext -> CompilerResult<'T>

/// AST Transformer that can change types
type ASTTransformer<'TIn, 'TOut> = 'TIn -> TransformContext -> CompilerResult<'TOut>

/// Transformer combinators using XParsec patterns
module Transformers =
    /// Lifts a pure function into a transformer
    let lift (f: 'T -> 'T) : ASTTransformer<'T> =
        fun ast ctx -> Success (f ast)
    
    /// Identity transformer
    let id : ASTTransformer<'T> =
        fun ast ctx -> Success ast
    
    /// Sequential composition of transformers
    let compose (t1: ASTTransformer<'T>) (t2: ASTTransformer<'T>) : ASTTransformer<'T> =
        fun ast ctx ->
            match t1 ast ctx with
            | Success ast' -> t2 ast' ctx
            | CompilerFailure errors -> CompilerFailure errors
    
    /// Maps a transformer over a list
    let mapList (transformer: ASTTransformer<'T>) : ASTTransformer<'T list> =
        fun items ctx ->
            items |> List.fold (fun acc item ->
                match acc with
                | CompilerFailure errors -> CompilerFailure errors
                | Success results ->
                    match transformer item ctx with
                    | Success transformed -> Success (results @ [transformed])
                    | CompilerFailure errors -> CompilerFailure errors
            ) (Success [])
    
    /// Conditional transformer
    let when' (predicate: 'T -> bool) (transformer: ASTTransformer<'T>) : ASTTransformer<'T> =
        fun ast ctx ->
            if predicate ast then transformer ast ctx
            else Success ast

/// Expression transformers
module ExprTransformers =
    open Transformers
    
    /// Transform expressions with stack allocation
    let stackAllocTransform : ASTTransformer<SynExpr> =
        lift Dabbit.Transformations.StackAllocation.StackTransform.transform
    
    /// Transform a binding's expression
    let transformBinding : ASTTransformer<SynBinding> =
        fun (SynBinding(a, k, inl, mut, attrs, xml, valData, pat, ret, expr, r, sp, tr)) ctx ->
            match stackAllocTransform expr ctx with
            | Success expr' -> 
                Success (SynBinding(a, k, inl, mut, attrs, xml, valData, pat, ret, expr', r, sp, tr))
            | CompilerFailure errors -> CompilerFailure errors

/// Declaration transformers
module DeclTransformers =
    open Transformers
    open ExprTransformers
    
    /// Transform let declarations
    let transformLetDecl : ASTTransformer<SynModuleDecl> =
        fun decl ctx ->
            match decl with
            | SynModuleDecl.Let(isRec, bindings, range) ->
                match mapList transformBinding bindings ctx with
                | Success bindings' -> Success (SynModuleDecl.Let(isRec, bindings', range))
                | CompilerFailure errors -> CompilerFailure errors
            | _ -> Success decl
    
    /// Transform all declarations in a module
    let transformDeclarations : ASTTransformer<SynModuleDecl list> =
        mapList transformLetDecl

/// Module transformers
module ModuleTransformers =
    open Transformers
    open DeclTransformers
    
    /// Transform a single module
    let transformModule : ASTTransformer<SynModuleOrNamespace> =
        fun (SynModuleOrNamespace(lid, isRec, kind, decls, xml, attrs, access, range, trivia)) ctx ->
            match transformDeclarations decls ctx with
            | Success decls' -> 
                Success (SynModuleOrNamespace(lid, isRec, kind, decls', xml, attrs, access, range, trivia))
            | CompilerFailure errors -> CompilerFailure errors
    
    /// Apply closure elimination to a module
    let closureElimination : ASTTransformer<SynModuleOrNamespace> =
        fun (SynModuleOrNamespace(lid, isRec, kind, decls, xml, attrs, access, range, trivia)) ctx ->
            // Extract bindings
            let bindings = 
                decls |> List.collect (function
                    | SynModuleDecl.Let(_, bindings, _) -> bindings
                    | _ -> [])
            
            // Transform with closure elimination
            let transformed = Dabbit.Transformations.ClosureElimination.transformModule bindings
            
            // Reconstruct module
            let newDecls = 
                if transformed.IsEmpty then decls
                else [SynModuleDecl.Let(false, transformed, range)]
            
            Success (SynModuleOrNamespace(lid, isRec, kind, newDecls, xml, attrs, access, range, trivia))

/// File transformers
module FileTransformers =
    open Transformers
    open ModuleTransformers
    
    /// Transform implementation file
    let transformImplFile : ASTTransformer<ParsedInput> =
        fun input ctx ->
            match input with
            | ParsedInput.ImplFile(ParsedImplFileInput(fileName, isScript, qname, pragmas, directives, modules, isLast, trivia, ids)) ->
                match mapList transformModule modules ctx with
                | Success modules' ->
                    // Apply closure elimination
                    match mapList closureElimination modules' ctx with
                    | Success modules'' ->
                        let transformed = ParsedInput.ImplFile(
                            ParsedImplFileInput(fileName, isScript, qname, pragmas, directives, modules'', isLast, trivia, ids))
                        Success transformed
                    | CompilerFailure errors -> CompilerFailure errors
                | CompilerFailure errors -> CompilerFailure errors
            | sig_ -> Success sig_

/// Main transformation pipeline
let transformAST (ctx: TransformContext) (input: ParsedInput) : ParsedInput =
    // Compose the transformation pipeline
    let pipeline = FileTransformers.transformImplFile
    
    // Run the pipeline
    match pipeline input ctx with
    | Success result ->
        // Apply tree shaking
        Dabbit.Analysis.AstPruner.prune ctx.Reachability.Reachable result
    | CompilerFailure errors ->
        // Log errors and return original
        errors |> List.iter (fun e -> printfn "Transformation error: %A" e)
        input

/// Verify transformations maintain zero-allocation guarantees
let verifyTransformations (input: ParsedInput) =
    // Extract all expressions using a fold
    let rec extractExprs acc = function
        | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
            modules |> List.fold (fun acc (SynModuleOrNamespace(_, _, _, decls, _, _, _, _, _)) ->
                decls |> List.fold (fun acc decl ->
                    match decl with
                    | SynModuleDecl.Let(_, bindings, _) ->
                        bindings |> List.fold (fun acc (SynBinding(_, _, _, _, _, _, _, _, _, expr, _, _, _)) ->
                            expr :: acc) acc
                    | _ -> acc) acc) acc
        | _ -> acc
    
    let exprs = extractExprs [] input
    let results = exprs |> List.map Dabbit.Transformations.StackAllocation.StackSafety.verify
    
    match results |> List.tryFind Result.isError with
    | Some (Error msg) -> Error msg
    | _ -> Ok ()

/// Computation expression for AST transformation
type ASTBuilder() =
    member _.Bind(transformer: ASTTransformer<'T>, f: 'T -> ASTTransformer<'U>) : ASTTransformer<'U> =
        fun ast ctx ->
            match transformer ast ctx with
            | Success result -> (f result) result ctx
            | CompilerFailure errors -> CompilerFailure errors
    
    member _.Return(value: 'T) : ASTTransformer<'T> =
        fun _ _ -> Success value
    
    member _.ReturnFrom(transformer: ASTTransformer<'T>) : ASTTransformer<'T> =
        transformer
    
    member _.Zero() : ASTTransformer<unit> =
        fun _ _ -> Success ()

/// AST transformation builder
let ast = ASTBuilder()