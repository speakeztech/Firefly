module Core.FCSProcessing.DependencyResolver

open FSharp.Compiler.Symbols
open FSharp.Compiler.CodeAnalysis

/// Dependency information
type DependencyInfo = {
    DirectDeps: Map<string, Set<string>>
    TypeDeps: Map<string, Set<string>>
    ModuleDeps: Map<string, Set<string>>
}

/// Resolve dependencies from type-checked results
let resolveDependencies (checkResults: FSharpCheckFileResults) (entities: Map<string, FSharpEntity>) =
    // Use declaration list to find dependencies
    let rec scanUses acc = function
        | FSharpImplementationFileDeclaration.Entity(entity, decls) ->
            let deps = 
                entity.MembersFunctionsAndValues
                |> Seq.collect (fun mfv ->
                    // Scan for uses in implementation
                    let uses = checkResults.GetUsesOfSymbolInFile(mfv) |> Async.RunSynchronously
                    uses |> Array.map (fun u -> u.Symbol.FullName))
                |> Set.ofSeq
            
            let acc' = { acc with DirectDeps = Map.add entity.FullName deps acc.DirectDeps }
            decls |> List.fold scanUses acc'
        
        | FSharpImplementationFileDeclaration.MemberOrFunctionOrValue(mfv, _, _) ->
            let uses = checkResults.GetUsesOfSymbolInFile(mfv) |> Async.RunSynchronously
            let deps = uses |> Array.map (fun u -> u.Symbol.FullName) |> Set.ofArray
            { acc with DirectDeps = Map.add mfv.FullName deps acc.DirectDeps }
        
        | FSharpImplementationFileDeclaration.InitAction _ -> acc
    
    let empty = { DirectDeps = Map.empty; TypeDeps = Map.empty; ModuleDeps = Map.empty }
    
    match checkResults.ImplementationFile with
    | Some implFile -> implFile.Declarations |> List.fold scanUses empty
    | None -> empty

/// Find entry points
let findEntryPoints (members: Map<string, FSharpMemberOrFunctionOrValue>) =
    members 
    |> Map.toList
    |> List.choose (fun (name, mfv) ->
        if mfv.Attributes |> Seq.exists (fun a -> a.AttributeType.FullName = "Microsoft.FSharp.Core.EntryPointAttribute") then
            Some name
        else None)
    |> Set.ofList