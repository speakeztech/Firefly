module Core.FCSProcessing.DependencyResolver

open FSharp.Compiler.Symbols
open FSharp.Compiler.CodeAnalysis

/// Dependency information
type DependencyInfo = {
    DirectDeps: Map<string, Set<string>>
    TypeDeps: Map<string, Set<string>>
    ModuleDeps: Map<string, Set<string>>
}

/// Private helper to add a dependency to a map
let private addDependency (fromSymbol: string) (toSymbol: string) (deps: Map<string, Set<string>>) =
    let currentDeps = 
        Map.tryFind fromSymbol deps 
        |> Option.defaultValue Set.empty
    Map.add fromSymbol (Set.add toSymbol currentDeps) deps

/// Resolve dependencies from type-checked results
let resolveDependencies (checkResults: FSharpCheckFileResults) =
    try
        let allUses = checkResults.GetAllUsesOfAllSymbolsInFile()
        
        let mutable directDeps = Map.empty
        let mutable typeDeps = Map.empty
        let mutable moduleDeps = Map.empty
        
        for symbolUse in allUses do
            let fromSymbol = symbolUse.Symbol.FullName
            
            match symbolUse.Symbol with
            | :? FSharpMemberOrFunctionOrValue as mfv ->
                // Function/value dependencies
                for paramGroup in mfv.CurriedParameterGroups do
                    for param in paramGroup do
                        if param.Type.HasTypeDefinition then
                            let typeName = param.Type.TypeDefinition.FullName
                            typeDeps <- addDependency fromSymbol typeName typeDeps
                
                // Return type dependencies
                if mfv.ReturnParameter.Type.HasTypeDefinition then
                    let returnType = mfv.ReturnParameter.Type.TypeDefinition.FullName
                    typeDeps <- addDependency fromSymbol returnType typeDeps
                    
            | :? FSharpEntity as entity ->
                // Type/module dependencies
                if entity.IsFSharpModule then  // Correct property name
                    moduleDeps <- addDependency fromSymbol entity.FullName moduleDeps
                else
                    // Track type inheritance/implementation
                    for iface in entity.AllInterfaces do
                        if iface.HasTypeDefinition then
                            typeDeps <- addDependency entity.FullName iface.TypeDefinition.FullName typeDeps
                        
            | _ -> ()
            
            // Track direct call dependencies
            if symbolUse.IsFromUse then
                directDeps <- addDependency "current" fromSymbol directDeps
        
        { 
            DirectDeps = directDeps
            TypeDeps = typeDeps
            ModuleDeps = moduleDeps 
        }
    with ex ->
        // Return empty on failure but log the error
        printfn "Error resolving dependencies: %s" ex.Message
        { DirectDeps = Map.empty; TypeDeps = Map.empty; ModuleDeps = Map.empty }

/// Find entry points in the checked file
let findEntryPoints (checkResults: FSharpCheckFileResults) : Set<string> =
    try
        let assemSig = checkResults.PartialAssemblySignature
        let mutable entryPoints = Set.empty
        
        for entity in assemSig.Entities do
            for mfv in entity.MembersFunctionsAndValues do
                let hasEntryPoint = 
                    mfv.Attributes 
                    |> Seq.exists (fun attr -> 
                        attr.AttributeType.FullName = "Microsoft.FSharp.Core.EntryPointAttribute")
                
                if hasEntryPoint then
                    let fullName = entity.FullName + "." + mfv.DisplayName
                    entryPoints <- Set.add fullName entryPoints
        
        entryPoints
    with _ ->
        Set.empty

/// Get all symbols from check results
let getAllSymbols (checkResults: FSharpCheckFileResults) : Map<string, FSharpSymbol> =
    try
        let assemSig = checkResults.PartialAssemblySignature
        let mutable symbols = Map.empty
        
        let rec collectFromEntity (path: string list) (entity: FSharpEntity) =
            let fullName = (path @ [entity.DisplayName]) |> String.concat "."
            symbols <- Map.add fullName (entity :> FSharpSymbol) symbols
            
            // Add members
            for mfv in entity.MembersFunctionsAndValues do
                let memberName = fullName + "." + mfv.DisplayName
                symbols <- Map.add memberName (mfv :> FSharpSymbol) symbols
            
            // Process nested entities
            for nested in entity.NestedEntities do
                collectFromEntity (path @ [entity.DisplayName]) nested
        
        // Process all top-level entities
        for entity in assemSig.Entities do
            collectFromEntity [] entity
        
        symbols
    with _ ->
        Map.empty

/// Build a simple dependency map
let buildDependencyMap (checkResults: FSharpCheckFileResults) : Map<string, Set<string>> =
    try
        // Get all symbol uses
        let allUses = checkResults.GetAllUsesOfAllSymbolsInFile()
        let mutable deps = Map.empty
        
        // Process each symbol use
        for symbolUse in allUses do
            let symbolName = symbolUse.Symbol.FullName
            let currentDeps = Map.tryFind symbolName deps |> Option.defaultValue Set.empty
            let newDeps = Set.add symbolName currentDeps
            deps <- Map.add symbolName newDeps deps
        
        deps
    with _ ->
        Map.empty

/// Find transitive dependencies
let findTransitiveDependencies (deps: Map<string, Set<string>>) (roots: Set<string>) : Set<string> =
    let rec traverse visited remaining =
        match remaining with
        | [] -> visited
        | current :: rest ->
            if Set.contains current visited then
                traverse visited rest
            else
                let newVisited = Set.add current visited
                let neighbors = 
                    Map.tryFind current deps 
                    |> Option.defaultValue Set.empty
                    |> Set.toList
                traverse newVisited (neighbors @ rest)
    
    traverse Set.empty (Set.toList roots)

/// Simple extraction of dependencies
let extractDependencies (checkResults: FSharpCheckFileResults) : DependencyInfo =
    let deps = buildDependencyMap checkResults
    { 
        DirectDeps = deps
        TypeDeps = Map.empty
        ModuleDeps = Map.empty 
    }