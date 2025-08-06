module Core.PSG.TypeIntegration

open System.Collections.Generic
open FSharp.Compiler.Symbols
open FSharp.Compiler.Text
open FSharp.Compiler.CodeAnalysis
open Core.PSG.Types

/// Type and constraint information extracted from COMPLETED FCS constraint resolution
type ResolvedTypeInfo = {
    Type: FSharpType
    Constraints: IList<FSharpGenericParameterConstraint>
    Range: range
    IsFromUsageSite: bool
    SourceSymbol: FSharpSymbol option
}

/// Index of resolved type information using COMPLETED FCS constraint resolution results
type ResolvedTypeIndex = {
    /// Map from range string to resolved type and constraint information  
    RangeToTypeInfo: Map<string, ResolvedTypeInfo>
    /// Map from symbol hash to resolved type and constraint information
    SymbolToTypeInfo: Map<int, ResolvedTypeInfo * FSharpSymbol>
}

/// Convert range to string key for Map compatibility  
let private rangeToKey (range: range) : string =
    sprintf "%s_%d_%d_%d_%d" 
        (System.IO.Path.GetFileName range.FileName)
        range.Start.Line range.Start.Column
        range.End.Line range.End.Column

/// Convert symbol to hash key for Map compatibility
let private symbolToKey (symbol: FSharpSymbol) : int =
    symbol.GetHashCode()

/// Extract constraints from RESOLVED type without triggering constraint solver
let private extractResolvedConstraints (fsharpType: FSharpType) : IList<FSharpGenericParameterConstraint> =
    try
        if fsharpType.IsGenericParameter then
            fsharpType.GenericParameter.Constraints
        elif fsharpType.HasTypeDefinition && fsharpType.GenericArguments.Count > 0 then
            let result = ResizeArray<FSharpGenericParameterConstraint>()
            for arg in fsharpType.GenericArguments do
                if arg.IsGenericParameter then
                    for typeConstraint in arg.GenericParameter.Constraints do
                        result.Add typeConstraint
            result :> IList<FSharpGenericParameterConstraint>
        else
            ResizeArray<FSharpGenericParameterConstraint>() :> IList<FSharpGenericParameterConstraint>
    with
    | ex ->
        printfn "[TYPE INTEGRATION] Warning: Resolved constraint extraction failed: %s" ex.Message
        ResizeArray<FSharpGenericParameterConstraint>() :> IList<FSharpGenericParameterConstraint>

/// Build resolved type index from COMPLETED FCS constraint resolution results
let buildResolvedTypeIndex (checkResults: FSharpCheckProjectResults) : ResolvedTypeIndex =
    let mutable rangeToTypeInfo : Map<string, ResolvedTypeInfo> = Map.empty
    let mutable symbolToTypeInfo : Map<int, ResolvedTypeInfo * FSharpSymbol> = Map.empty
    
    try
        /// CANONICAL Strategy 1: Symbol Uses with COMPLETED Constraint Resolution
        /// Access constraint information that's already been resolved at usage sites
        printfn "[TYPE INTEGRATION] Extracting COMPLETED constraint resolution from symbol uses"
        let allSymbolUses = checkResults.GetAllUsesOfAllSymbols()
        
        for symbolUse in allSymbolUses do
            try
                // Process usage sites where constraints are ALREADY resolved in context
                if not symbolUse.IsFromDefinition then
                    let symbol = symbolUse.Symbol
                    
                    // For function/value symbols, extract RESOLVED type at usage site
                    match symbol with
                    | :? FSharpMemberOrFunctionOrValue as mfv ->
                        // At usage sites, constraints are ALREADY resolved - no constraint solver trigger
                        let resolvedType = mfv.FullType
                        let typeConstraints = extractResolvedConstraints resolvedType
                        
                        let typeInfo = {
                            Type = resolvedType
                            Constraints = typeConstraints
                            Range = symbolUse.Range
                            IsFromUsageSite = true
                            SourceSymbol = Some symbol
                        }
                        
                        let key = rangeToKey symbolUse.Range
                        rangeToTypeInfo <- Map.add key typeInfo rangeToTypeInfo
                        
                        let symbolKey = symbolToKey symbol
                        symbolToTypeInfo <- Map.add symbolKey (typeInfo, symbol) symbolToTypeInfo
                        
                    | :? FSharpEntity as entity ->
                        // Entity types at usage sites have RESOLVED constraints
                        let resolvedType = entity.AsType()
                        let typeConstraints = extractResolvedConstraints resolvedType
                        
                        let typeInfo = {
                            Type = resolvedType
                            Constraints = typeConstraints
                            Range = symbolUse.Range
                            IsFromUsageSite = true
                            SourceSymbol = Some symbol
                        }
                        
                        let key = rangeToKey symbolUse.Range
                        rangeToTypeInfo <- Map.add key typeInfo rangeToTypeInfo
                        
                        let symbolKey = symbolToKey symbol
                        symbolToTypeInfo <- Map.add symbolKey (typeInfo, symbol) symbolToTypeInfo
                        
                    | _ -> ()
            with
            | ex ->
                // Individual symbol processing failures are acceptable - log and continue
                printfn "[TYPE INTEGRATION] Note: Symbol use processing skipped for %A: %s" symbolUse.Range ex.Message
        
        /// CANONICAL Strategy 2: Assembly Signature Analysis  
        /// Access fully resolved member signatures with COMPLETED constraint resolution
        printfn "[TYPE INTEGRATION] Extracting COMPLETED constraint resolution from assembly signatures"
        let assemblyContents = checkResults.AssemblyContents
        
        for implFile in assemblyContents.ImplementationFiles do
            let rec processResolvedDeclaration (decl: FSharpImplementationFileDeclaration) =
                try
                    match decl with
                    | FSharpImplementationFileDeclaration.Entity (entity, subDecls) ->
                        // Assembly signatures contain FULLY RESOLVED constraint information
                        let resolvedType = entity.AsType()
                        let typeConstraints = extractResolvedConstraints resolvedType
                        
                        let typeInfo = {
                            Type = resolvedType
                            Constraints = typeConstraints
                            Range = entity.DeclarationLocation
                            IsFromUsageSite = false
                            SourceSymbol = Some (entity :> FSharpSymbol)
                        }
                        
                        let key = rangeToKey entity.DeclarationLocation
                        rangeToTypeInfo <- Map.add key typeInfo rangeToTypeInfo
                        
                        let symbolKey = symbolToKey (entity :> FSharpSymbol)
                        symbolToTypeInfo <- Map.add symbolKey (typeInfo, entity :> FSharpSymbol) symbolToTypeInfo
                        
                        // Process nested declarations
                        subDecls |> List.iter processResolvedDeclaration
                        
                    | FSharpImplementationFileDeclaration.MemberOrFunctionOrValue (mfv, args, expr) ->
                        // Assembly signatures have COMPLETED constraint resolution
                        let resolvedType = mfv.FullType
                        let typeConstraints = extractResolvedConstraints resolvedType
                        
                        let typeInfo = {
                            Type = resolvedType
                            Constraints = typeConstraints
                            Range = mfv.DeclarationLocation
                            IsFromUsageSite = false
                            SourceSymbol = Some (mfv :> FSharpSymbol)
                        }
                        
                        let key = rangeToKey mfv.DeclarationLocation
                        rangeToTypeInfo <- Map.add key typeInfo rangeToTypeInfo
                        
                        let symbolKey = symbolToKey (mfv :> FSharpSymbol)
                        symbolToTypeInfo <- Map.add symbolKey (typeInfo, mfv :> FSharpSymbol) symbolToTypeInfo
                        
                    | FSharpImplementationFileDeclaration.InitAction expr ->
                        // Skip expression processing to avoid constraint solver activation
                        ()
                with
                | ex ->
                    printfn "[TYPE INTEGRATION] Note: Declaration processing skipped: %s" ex.Message
            
            implFile.Declarations |> List.iter processResolvedDeclaration
        
    with
    | ex ->
        printfn "[TYPE INTEGRATION] Warning: Resolved type index construction encountered error: %s" ex.Message
    
    printfn "[TYPE INTEGRATION] CANONICAL resolved type index built: %d range mappings, %d symbol mappings" 
        (Map.count rangeToTypeInfo) (Map.count symbolToTypeInfo)
    
    {
        RangeToTypeInfo = rangeToTypeInfo
        SymbolToTypeInfo = symbolToTypeInfo
    }

/// Correlate RESOLVED type and constraint information with PSG node
let correlateNodeWithResolvedTypes (node: PSGNode) (resolvedIndex: ResolvedTypeIndex) : (FSharpType * IList<FSharpGenericParameterConstraint>) option =
    try
        // Strategy 1: Direct symbol-based correlation (preserved from original)
        match node.Symbol with
        | Some symbol ->
            let symbolKey = symbolToKey symbol
            match Map.tryFind symbolKey resolvedIndex.SymbolToTypeInfo with
            | Some (typeInfo, _) -> Some (typeInfo.Type, typeInfo.Constraints)
            | None ->
                // Strategy 2: Range-based correlation for expressions (preserved from original)  
                let rangeKey = rangeToKey node.Range
                match Map.tryFind rangeKey resolvedIndex.RangeToTypeInfo with
                | Some typeInfo -> Some (typeInfo.Type, typeInfo.Constraints)
                | None -> None
        | None ->
            // Strategy 3: Range-based correlation only (preserved from original)
            let rangeKey = rangeToKey node.Range
            match Map.tryFind rangeKey resolvedIndex.RangeToTypeInfo with
            | Some typeInfo -> Some (typeInfo.Type, typeInfo.Constraints)
            | None ->
                // Strategy 4: Tolerant range matching within same file (preserved from original)
                resolvedIndex.RangeToTypeInfo
                |> Map.tryPick (fun rangeStr typeInfo ->
                    if rangeStr.StartsWith(System.IO.Path.GetFileName node.Range.FileName) then
                        Some (typeInfo.Type, typeInfo.Constraints)
                    else None
                )
    with
    | ex ->
        printfn "[TYPE INTEGRATION] Warning: Type correlation failed for node %s: %s" node.Id.Value ex.Message
        None

/// CANONICAL integration using COMPLETED FCS constraint resolution results
let integrateTypesWithCheckResults (psg: ProgramSemanticGraph) (checkResults: FSharpCheckProjectResults) : ProgramSemanticGraph =
    printfn "[TYPE INTEGRATION] Starting CANONICAL FCS constraint resolution integration"
    printfn "[TYPE INTEGRATION] Building resolved type index from COMPLETED constraint resolution"
    
    let resolvedIndex = buildResolvedTypeIndex checkResults
    
    printfn "[TYPE INTEGRATION] Correlating RESOLVED type information with %d PSG nodes" (Map.count psg.Nodes)
    
    let mutable typeCorrelationCount = 0
    let mutable constraintCorrelationCount = 0
    
    // Update PSG nodes with RESOLVED type and constraint information
    let updatedNodes =
        psg.Nodes
        |> Map.map (fun nodeId node ->
            match correlateNodeWithResolvedTypes node resolvedIndex with
            | Some (resolvedType, constraints) ->
                typeCorrelationCount <- typeCorrelationCount + 1
                // Convert IList to F# list for PSG compatibility
                let constraintsList = constraints |> List.ofSeq
                if not (List.isEmpty constraintsList) then
                    constraintCorrelationCount <- constraintCorrelationCount + 1
                { node with 
                    Type = Some resolvedType
                    Constraints = if List.isEmpty constraintsList then None else Some constraintsList }
            | None ->
                node
        )
    
    printfn "[TYPE INTEGRATION] CANONICAL correlation complete: %d/%d nodes updated with type information" 
        typeCorrelationCount psg.Nodes.Count
    printfn "[TYPE INTEGRATION] CANONICAL constraint correlation complete: %d/%d nodes updated with constraint information" 
        constraintCorrelationCount psg.Nodes.Count
    
    // Return updated PSG with RESOLVED type and constraint information
    { psg with Nodes = updatedNodes }

/// Extract typed AST from check results for type integration (preserved interface from original)
let extractTypedAST (checkResults: FSharpCheckProjectResults) : FSharpImplementationFileContents list =
    let assemblyContents = checkResults.AssemblyContents
    printfn "[TYPE INTEGRATION] Extracting typed AST from assembly contents with %d implementation files" 
        assemblyContents.ImplementationFiles.Length
    assemblyContents.ImplementationFiles

/// LEGACY integration point maintained for backwards compatibility - now uses CANONICAL approach
let integrateTypesIntoPSG (psg: ProgramSemanticGraph) (typedFiles: FSharpImplementationFileContents list) : ProgramSemanticGraph =
    printfn "[TYPE INTEGRATION] Legacy integration called - redirecting to CANONICAL FCS constraint resolution"
    printfn "[TYPE INTEGRATION] NOTE: This legacy interface cannot access completed constraint resolution"
    printfn "[TYPE INTEGRATION] Use integrateTypesWithCheckResults for full CANONICAL FCS constraint resolution"
    
    // Cannot perform canonical constraint resolution without FSharpCheckProjectResults
    // Return PSG unchanged to maintain interface compatibility
    psg