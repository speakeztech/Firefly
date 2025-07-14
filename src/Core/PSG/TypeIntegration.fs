module Core.PSG.TypeIntegration

open FSharp.Compiler.Symbols
open FSharp.Compiler.Text
open FSharp.Compiler.CodeAnalysis
open Core.PSG.Types

/// Type and constraint information extracted directly from FCS
type ExtractedTypeInfo = {
    Type: FSharpType
    Constraints: FSharpGenericParameterConstraint list
    Range: range
    IsFromExpression: bool
    IsFromSymbol: bool
}

/// Direct type and constraint index avoiding constraint solver operations
type DirectTypeIndex = {
    /// Map from range string to extracted type and constraint information
    RangeToTypeInfo: Map<string, ExtractedTypeInfo>
    /// Map from symbol hash to extracted type and constraint information
    SymbolToTypeInfo: Map<int, ExtractedTypeInfo * FSharpSymbol>
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

/// Extract constraints directly from FSharpType without triggering constraint solver
let private extractConstraintsFromType (fsharpType: FSharpType) : FSharpGenericParameterConstraint list =
    try
        // Extract constraints from generic parameters without formatting
        if fsharpType.IsGenericParameter then
            fsharpType.GenericParameter.Constraints |> List.ofSeq
        elif fsharpType.HasTypeDefinition && fsharpType.GenericArguments.Count > 0 then
            // Extract constraints from generic arguments
            fsharpType.GenericArguments
            |> Seq.collect (fun arg -> 
                if arg.IsGenericParameter then 
                    arg.GenericParameter.Constraints 
                else 
                    System.Collections.Generic.List<FSharpGenericParameterConstraint>())
            |> List.ofSeq
        else
            []
    with
    | ex ->
        printfn "[TYPE INTEGRATION] Warning: Constraint extraction failed: %s" ex.Message
        []

/// Build direct type index from typed AST without constraint solver operations
let buildDirectTypeIndex (typedFiles: FSharpImplementationFileContents list) : DirectTypeIndex =
    let mutable rangeToTypeInfo : Map<string, ExtractedTypeInfo> = Map.empty
    let mutable symbolToTypeInfo : Map<int, ExtractedTypeInfo * FSharpSymbol> = Map.empty
    
    /// Process typed expression safely extracting type and constraint information
    let rec processTypedExpression (expr: FSharpExpr) =
        try
            // Extract type and constraints directly without formatting
            let constraints = extractConstraintsFromType expr.Type
            let typeInfo = {
                Type = expr.Type
                Constraints = constraints
                Range = expr.Range
                IsFromExpression = true
                IsFromSymbol = false
            }
            
            let rangeKey = rangeToKey expr.Range
            rangeToTypeInfo <- Map.add rangeKey typeInfo rangeToTypeInfo
            
            // Process sub-expressions recursively
            for subExpr in expr.ImmediateSubExpressions do
                processTypedExpression subExpr
        with
        | ex ->
            printfn "[TYPE INTEGRATION] Warning: Expression processing failed at %A: %s" expr.Range ex.Message
    
    /// Process typed declaration safely extracting symbol type information
    let rec processTypedDeclaration (decl: FSharpImplementationFileDeclaration) =
        try
            match decl with
            | FSharpImplementationFileDeclaration.Entity (entity, subDecls) ->
                // Extract entity type information without formatting
                try
                    let entityType = entity.AsType()
                    let constraints = extractConstraintsFromType entityType
                    let typeInfo = {
                        Type = entityType
                        Constraints = constraints
                        Range = entity.DeclarationLocation
                        IsFromExpression = false
                        IsFromSymbol = true
                    }
                    
                    let symbolKey = symbolToKey (entity :> FSharpSymbol)
                    symbolToTypeInfo <- Map.add symbolKey (typeInfo, entity :> FSharpSymbol) symbolToTypeInfo
                with
                | ex ->
                    printfn "[TYPE INTEGRATION] Warning: Entity type access failed for %s: %s" entity.DisplayName ex.Message
                
                // Process nested declarations
                subDecls |> List.iter processTypedDeclaration
                
            | FSharpImplementationFileDeclaration.MemberOrFunctionOrValue (mfv, args, expr) ->
                // Extract member function type information without formatting
                try
                    let memberType = mfv.FullType
                    let constraints = extractConstraintsFromType memberType
                    let typeInfo = {
                        Type = memberType
                        Constraints = constraints
                        Range = mfv.DeclarationLocation
                        IsFromExpression = false
                        IsFromSymbol = true
                    }
                    
                    let symbolKey = symbolToKey (mfv :> FSharpSymbol)
                    symbolToTypeInfo <- Map.add symbolKey (typeInfo, mfv :> FSharpSymbol) symbolToTypeInfo
                with
                | ex ->
                    printfn "[TYPE INTEGRATION] Warning: Member type access failed for %s: %s" mfv.DisplayName ex.Message
                
                // Process expression body
                processTypedExpression expr
                
            | FSharpImplementationFileDeclaration.InitAction expr ->
                processTypedExpression expr
        with
        | ex ->
            printfn "[TYPE INTEGRATION] Warning: Declaration processing failed: %s" ex.Message
    
    // Process all typed files
    try
        typedFiles |> List.iter (fun implFile ->
            implFile.Declarations |> List.iter processTypedDeclaration
        )
    with
    | ex ->
        printfn "[TYPE INTEGRATION] Warning: Type index construction failed: %s" ex.Message
    
    {
        RangeToTypeInfo = rangeToTypeInfo
        SymbolToTypeInfo = symbolToTypeInfo
    }

/// Correlate type and constraint information with PSG node
let correlateNodeTypeAndConstraints (node: PSGNode) (typeIndex: DirectTypeIndex) : (FSharpType * FSharpGenericParameterConstraint list) option =
    try
        // Strategy 1: Direct symbol-based correlation
        match node.Symbol with
        | Some symbol ->
            let symbolKey = symbolToKey symbol
            match Map.tryFind symbolKey typeIndex.SymbolToTypeInfo with
            | Some (typeInfo, _) -> Some (typeInfo.Type, typeInfo.Constraints)
            | None ->
                // Strategy 2: Range-based correlation for expressions
                let rangeKey = rangeToKey node.Range
                match Map.tryFind rangeKey typeIndex.RangeToTypeInfo with
                | Some typeInfo -> Some (typeInfo.Type, typeInfo.Constraints)
                | None -> None
        | None ->
            // Strategy 3: Range-based correlation only
            let rangeKey = rangeToKey node.Range
            match Map.tryFind rangeKey typeIndex.RangeToTypeInfo with
            | Some typeInfo -> Some (typeInfo.Type, typeInfo.Constraints)
            | None ->
                // Strategy 4: Tolerant range matching within same file
                typeIndex.RangeToTypeInfo
                |> Map.tryPick (fun rangeStr typeInfo ->
                    if rangeStr.StartsWith(System.IO.Path.GetFileName node.Range.FileName) then
                        Some (typeInfo.Type, typeInfo.Constraints)
                    else None
                )
    with
    | ex ->
        printfn "[TYPE INTEGRATION] Warning: Type correlation failed for node %s: %s" node.Id.Value ex.Message
        None

/// Integrate type and constraint information into existing PSG nodes
let integrateTypesIntoPSG (psg: ProgramSemanticGraph) (typedFiles: FSharpImplementationFileContents list) : ProgramSemanticGraph =
    printfn "[TYPE INTEGRATION] Building direct type index from %d typed files" typedFiles.Length
    
    let typeIndex = buildDirectTypeIndex typedFiles
    
    printfn "[TYPE INTEGRATION] Type index built: %d range-to-type mappings, %d symbol-to-type mappings" 
        typeIndex.RangeToTypeInfo.Count typeIndex.SymbolToTypeInfo.Count
    
    let mutable typeCorrelationCount = 0
    let mutable constraintCorrelationCount = 0
    
    let updatedNodes =
        psg.Nodes
        |> Map.map (fun nodeId node ->
            match correlateNodeTypeAndConstraints node typeIndex with
            | Some (fsharpType, constraints) ->
                typeCorrelationCount <- typeCorrelationCount + 1
                if not (List.isEmpty constraints) then
                    constraintCorrelationCount <- constraintCorrelationCount + 1
                { node with 
                    Type = Some fsharpType
                    Constraints = if List.isEmpty constraints then None else Some constraints }
            | None ->
                node
        )
    
    printfn "[TYPE INTEGRATION] Type correlation complete: %d/%d nodes updated with type information" 
        typeCorrelationCount psg.Nodes.Count
    printfn "[TYPE INTEGRATION] Constraint correlation complete: %d/%d nodes updated with constraint information" 
        constraintCorrelationCount psg.Nodes.Count
    
    { psg with Nodes = updatedNodes }

/// Extract typed AST from check results for type integration
let extractTypedAST (checkResults: FSharpCheckProjectResults) : FSharpImplementationFileContents list =
    let assemblyContents = checkResults.AssemblyContents
    printfn "[TYPE INTEGRATION] Extracting typed AST from assembly contents with %d implementation files" 
        assemblyContents.ImplementationFiles.Length
    assemblyContents.ImplementationFiles

/// Generate type integration statistics including constraint information
type TypeIntegrationStats = {
    TotalNodes: int
    NodesWithTypes: int
    NodesWithConstraints: int
    TypeCorrelationRate: float
    ConstraintCorrelationRate: float
    NodesWithSymbols: int
    SymbolBasedTypeCorrelations: int
    RangeBasedTypeCorrelations: int
    UnresolvedTypeNodes: int
}

/// Calculate comprehensive type and constraint integration statistics
let calculateTypeIntegrationStats (psg: ProgramSemanticGraph) : TypeIntegrationStats =
    let totalNodes = psg.Nodes.Count
    let nodesWithTypes = psg.Nodes |> Map.toSeq |> Seq.filter (fun (_, node) -> node.Type.IsSome) |> Seq.length
    let nodesWithConstraints = psg.Nodes |> Map.toSeq |> Seq.filter (fun (_, node) -> node.Constraints.IsSome) |> Seq.length
    let nodesWithSymbols = psg.Nodes |> Map.toSeq |> Seq.filter (fun (_, node) -> node.Symbol.IsSome) |> Seq.length
    
    let symbolBasedCorrelations = 
        psg.Nodes 
        |> Map.toSeq 
        |> Seq.filter (fun (_, node) -> node.Symbol.IsSome && node.Type.IsSome) 
        |> Seq.length
    
    let rangeBasedCorrelations = 
        psg.Nodes 
        |> Map.toSeq 
        |> Seq.filter (fun (_, node) -> node.Symbol.IsNone && node.Type.IsSome) 
        |> Seq.length
    
    let unresolvedTypeNodes = totalNodes - nodesWithTypes
    let typeCorrelationRate = 
        if totalNodes > 0 then 
            (float nodesWithTypes / float totalNodes) * 100.0 
        else 0.0
    
    let constraintCorrelationRate = 
        if totalNodes > 0 then 
            (float nodesWithConstraints / float totalNodes) * 100.0 
        else 0.0
    
    {
        TotalNodes = totalNodes
        NodesWithTypes = nodesWithTypes
        NodesWithConstraints = nodesWithConstraints
        TypeCorrelationRate = typeCorrelationRate
        ConstraintCorrelationRate = constraintCorrelationRate
        NodesWithSymbols = nodesWithSymbols
        SymbolBasedTypeCorrelations = symbolBasedCorrelations
        RangeBasedTypeCorrelations = rangeBasedCorrelations
        UnresolvedTypeNodes = unresolvedTypeNodes
    }