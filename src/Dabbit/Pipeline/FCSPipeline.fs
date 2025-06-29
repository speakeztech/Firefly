module Dabbit.Pipeline.FCSPipeline

open FSharp.Compiler.Syntax
open FSharp.Compiler.Symbols
open FSharp.Compiler.CodeAnalysis

/// Complete FCS processing pipeline
type ProcessingContext = {
    Checker: FSharpChecker
    Options: FSharpProjectOptions
    TypeCtx: Dabbit.CodeGeneration.TypeMapping.TypeContext
    SymbolRegistry: Dabbit.Bindings.SymbolRegistry.SymbolRegistry
}

/// Process complete compilation unit
let processCompilationUnit (ctx: ProcessingContext) (input: ParsedInput) = async {
    // Type check
    let! checkResults = 
        match input with
        | ParsedInput.ImplFile impl ->
            ctx.Checker.CheckFileInProject(impl, impl.FileName, 0, SourceText.ofString "", ctx.Options)
        | _ -> failwith "Signature files not supported"
    
    // Build symbol table
    let symbols = 
        checkResults.PartialAssemblySignature.Entities
        |> Seq.collect (fun e -> e.MembersFunctionsAndValues)
        |> Seq.map (fun mfv -> (mfv.FullName, mfv :> FSharpSymbol))
        |> Map.ofSeq
    
    // Update type context
    let typeCtx = 
        symbols |> Map.fold (fun ctx name sym ->
            Dabbit.CodeGeneration.TypeMapping.TypeContextBuilder.addSymbol name sym ctx) ctx.TypeCtx
    
    // Apply transformations
    let transformed = 
        input
        |> Dabbit.Transformations.StackAllocation.StackTransform.transform
        |> Dabbit.Transformations.ClosureElimination.transformModule
    
    // Build dependency graph
    let deps = 
        match transformed with
        | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, modules, _, _, _, _)) ->
            modules |> List.collect (fun (SynModuleOrNamespace(longId, _, _, decls, _, _, _, _, _)) ->
                let modName = longId |> List.map (fun id -> id.idText) |> String.concat "."
                let graph = Dabbit.Analysis.DependencyGraphBuilder.buildFromModule modName decls
                graph.Edges |> Map.toList)
            |> Map.ofList
        | _ -> Map.empty
    
    // Find entry points
    let entries = 
        symbols |> Map.toList |> List.choose (fun (name, sym) ->
            match sym with
            | :? FSharpMemberOrFunctionOrValue as mfv when mfv.HasAttribute<EntryPointAttribute>() ->
                Some name
            | _ -> None)
        |> Set.ofList
    
    // Perform reachability analysis
    let reachability = Dabbit.Analysis.ReachabilityAnalyzer.analyze symbols deps entries
    
    // Prune unreachable code
    let pruned = Dabbit.Analysis.AstPruner.prune reachability.Reachable transformed
    
    return {| 
        Input = pruned
        TypeContext = typeCtx
        Reachability = reachability
        Symbols = symbols
    |}
}