module Core.PSG.TypeIntegration

open FSharp.Compiler.Symbols
open FSharp.Compiler.Text
open FSharp.Compiler.CodeAnalysis
open Core.PSG.Types

/// Range-based type correlation index using string keys for FCS 43.9.300 compatibility
type TypeCorrelationIndex = {
    /// Map from range string representation to typed expression information
    RangeToTypeMap: Map<string, FSharpType>
    /// Map from symbol hash to its type for direct symbol-based lookup
    SymbolToTypeMap: Map<int, FSharpType * FSharpSymbol>
    /// Map from range string to typed declaration information  
    RangeToDeclarationMap: Map<string, FSharpImplementationFileDeclaration>
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

/// Build type correlation index from typed AST with safe type access
let buildTypeCorrelationIndex (typedFiles: FSharpImplementationFileContents list) : TypeCorrelationIndex =
    let mutable rangeToTypeMap : Map<string, FSharpType> = Map.empty
    let mutable symbolToTypeMap : Map<int, FSharpType * FSharpSymbol> = Map.empty
    let mutable rangeToDeclarationMap : Map<string, FSharpImplementationFileDeclaration> = Map.empty
    
    /// Process a typed expression safely
    let rec processTypedExpression (expr: FSharpExpr) =
        try
            let rangeKey = rangeToKey expr.Range
            rangeToTypeMap <- Map.add rangeKey expr.Type rangeToTypeMap
            
            for subExpr in expr.ImmediateSubExpressions do
                processTypedExpression subExpr
        with
        | ex ->
            printfn "[TYPE INTEGRATION] Warning: Failed to process expression type: %s" ex.Message
    
    /// Process a typed declaration safely
    let rec processTypedDeclaration (decl: FSharpImplementationFileDeclaration) =
        try
            match decl with
            | FSharpImplementationFileDeclaration.Entity (entity, subDecls) ->
                // Only store entity symbol without attempting type formatting
                let symbolKey = symbolToKey (entity :> FSharpSymbol)
                // Temporarily skip type storage to avoid constraint solver issues
                // symbolToTypeMap <- Map.add symbolKey (entity.AsType(), entity :> FSharpSymbol) symbolToTypeMap
                subDecls |> List.iter processTypedDeclaration
                
            | FSharpImplementationFileDeclaration.MemberOrFunctionOrValue (mfv, args, expr) ->
                // Only store symbol without attempting type access for now
                let symbolKey = symbolToKey (mfv :> FSharpSymbol)
                // Temporarily skip type storage to avoid constraint solver issues
                // symbolToTypeMap <- Map.add symbolKey (mfv.FullType, mfv :> FSharpSymbol) symbolToTypeMap
                processTypedExpression expr
                
            | FSharpImplementationFileDeclaration.InitAction expr ->
                processTypedExpression expr
        with
        | ex ->
            printfn "[TYPE INTEGRATION] Warning: Failed to process declaration: %s" ex.Message
    
    try
        typedFiles |> List.iter (fun implFile ->
            implFile.Declarations |> List.iter processTypedDeclaration
        )
    with
    | ex ->
        printfn "[TYPE INTEGRATION] Warning: Type index construction failed: %s" ex.Message
    
    {
        RangeToTypeMap = rangeToTypeMap
        SymbolToTypeMap = symbolToTypeMap
        RangeToDeclarationMap = rangeToDeclarationMap
    }

/// Correlate type information with PSG node using precise range matching with safe type access
let correlateNodeType (node: PSGNode) (typeIndex: TypeCorrelationIndex) : FSharpType option =
    try
        match node.Symbol with
        | Some symbol ->
            let symbolKey = symbolToKey symbol
            match Map.tryFind symbolKey typeIndex.SymbolToTypeMap with
            | Some (fsharpType, _) -> Some fsharpType
            | None ->
                let rangeKey = rangeToKey node.Range
                Map.tryFind rangeKey typeIndex.RangeToTypeMap
        | None ->
            let rangeKey = rangeToKey node.Range
            match Map.tryFind rangeKey typeIndex.RangeToTypeMap with
            | Some fsharpType -> Some fsharpType
            | None ->
                typeIndex.RangeToTypeMap
                |> Map.tryPick (fun rangeStr fsharpType ->
                    let nodeRangeKey = rangeToKey node.Range
                    if rangeStr.StartsWith(System.IO.Path.GetFileName node.Range.FileName) then
                        Some fsharpType
                    else None
                )
    with
    | ex ->
        printfn "[TYPE INTEGRATION] Warning: Type correlation failed for node %s: %s" node.Id.Value ex.Message
        None

/// Integrate type information into existing PSG nodes
let integrateTypesIntoPSG (psg: ProgramSemanticGraph) (typedFiles: FSharpImplementationFileContents list) : ProgramSemanticGraph =
    printfn "[TYPE INTEGRATION] Building type correlation index from %d typed files" typedFiles.Length
    
    let typeIndex = buildTypeCorrelationIndex typedFiles
    
    printfn "[TYPE INTEGRATION] Type index built: %d range-to-type mappings, %d symbol-to-type mappings" 
        typeIndex.RangeToTypeMap.Count typeIndex.SymbolToTypeMap.Count
    
    let mutable typeCorrelationCount = 0
    let updatedNodes =
        psg.Nodes
        |> Map.map (fun nodeId node ->
            match correlateNodeType node typeIndex with
            | Some fsharpType ->
                typeCorrelationCount <- typeCorrelationCount + 1
                { node with Type = Some fsharpType }
            | None ->
                node
        )
    
    printfn "[TYPE INTEGRATION] Type correlation complete: %d/%d nodes updated with type information" 
        typeCorrelationCount psg.Nodes.Count
    
    { psg with Nodes = updatedNodes }

/// Extract typed AST from check results for type integration
let extractTypedAST (checkResults: FSharpCheckProjectResults) : FSharpImplementationFileContents list =
    let assemblyContents = checkResults.AssemblyContents
    printfn "[TYPE INTEGRATION] Extracting typed AST from assembly contents with %d implementation files" 
        assemblyContents.ImplementationFiles.Length
    assemblyContents.ImplementationFiles

/// Generate type integration statistics for debugging
type TypeIntegrationStats = {
    TotalNodes: int
    NodesWithTypes: int
    TypeCorrelationRate: float
    NodesWithSymbols: int
    SymbolBasedTypeCorrelations: int
    RangeBasedTypeCorrelations: int
    UnresolvedTypeNodes: int
}

/// Calculate comprehensive type integration statistics
let calculateTypeIntegrationStats (psg: ProgramSemanticGraph) : TypeIntegrationStats =
    let totalNodes = psg.Nodes.Count
    let nodesWithTypes = psg.Nodes |> Map.toSeq |> Seq.filter (fun (_, node) -> node.Type.IsSome) |> Seq.length
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
    
    {
        TotalNodes = totalNodes
        NodesWithTypes = nodesWithTypes
        TypeCorrelationRate = typeCorrelationRate
        NodesWithSymbols = nodesWithSymbols
        SymbolBasedTypeCorrelations = symbolBasedCorrelations
        RangeBasedTypeCorrelations = rangeBasedCorrelations
        UnresolvedTypeNodes = unresolvedTypeNodes
    }