module Dabbit.Pipeline.FCSPipeline

open FSharp.Compiler.Syntax
open FSharp.Compiler.CodeAnalysis
open Dabbit.CodeGeneration.TypeMapping
open Dabbit.Bindings.SymbolRegistry 
open Dabbit.Bindings.SymbolRegistry.Registry 
open Dabbit.Analysis.ReachabilityAnalyzer
open Dabbit.Analysis.DependencyGraphBuilder
open Dabbit.Analysis.AstPruner

/// Complete FCS processing pipeline
type ProcessingContext = {
    Checker: FSharpChecker
    Options: FSharpProjectOptions
    TypeCtx: TypeContext
    SymbolRegistry: SymbolRegistry 
}

/// Process complete compilation unit - apply transformations to already-typed AST
let processCompilationUnit (ctx: ProcessingContext) (input: ParsedInput) = async {
        // Apply transformations to the AST
        let transformed = 
            match input with
            | ParsedInput.ImplFile(ParsedImplFileInput(fileName, isScript, qualName, pragmas, hashDirectives, modules, isLastCompiled, isExe, _)) ->
                // Transform each module
                let transformedModules = 
                    modules |> List.map (fun m ->
                        // Extract bindings for closure elimination
                        let bindings = 
                            match m with
                            | SynModuleOrNamespace(_, _, _, decls, _, _, _, _, _) ->
                                decls |> List.collect (function
                                    | SynModuleDecl.Let(_, bindings, _) -> bindings
                                    | _ -> [])
                        
                        // Apply closure elimination to bindings
                        let transformedBindings = 
                            if List.isEmpty bindings then []
                            else Dabbit.Transformations.ClosureElimination.transformModule bindings
                        
                        // Reconstruct module with transformed bindings
                        match m with
                        | SynModuleOrNamespace(lid, isRec, kind, decls, xml, attrs, access, range, trivia) ->
                            let newDecls = 
                                if List.isEmpty transformedBindings then decls
                                else 
                                    // Replace let bindings with transformed ones
                                    decls |> List.map (function
                                        | SynModuleDecl.Let(isRec, _, range) -> 
                                            SynModuleDecl.Let(isRec, transformedBindings, range)
                                        | decl -> decl)
                            SynModuleOrNamespace(lid, isRec, kind, newDecls, xml, attrs, access, range, trivia))
                
                ParsedInput.ImplFile(ParsedImplFileInput(fileName, isScript, qualName, pragmas, hashDirectives, transformedModules, isLastCompiled, isExe, Set.empty))
            | sig_ -> sig_
        
        // Build dependency graph from transformed AST
        let deps = 
            match transformed with
            | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
                modules |> List.collect (fun (SynModuleOrNamespace(longId, _, _, decls, _, _, _, _, _)) ->
                    let modName = longId |> List.map (fun id -> id.idText) |> String.concat "."
                    let graph = buildFromModule modName decls
                    graph.Edges |> Map.toList)
                |> Map.ofList
            | _ -> Map.empty
        
        // Get entry points from the symbol registry
        let entries = getEntryPoints input
        
        // Get all symbols from the registry
        let symbols = getAllSymbols ctx.SymbolRegistry
        
        // Perform reachability analysis
        let reachability = analyze symbols deps entries
        
        // Prune unreachable code
        let pruned = prune reachability.Reachable transformed
        
        return {| 
            Input = pruned
            TypeContext = ctx.TypeCtx
            Reachability = reachability
            Symbols = symbols
        |}
}