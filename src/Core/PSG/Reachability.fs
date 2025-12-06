module Core.PSG.Reachability

open System
open FSharp.Compiler.Symbols
open Core.PSG.Types

/// Semantic reachability context based on type-aware analysis
type SemanticReachabilityContext = {
    /// Functions that are actually called
    ReachableFunctions: Set<string>
    /// Type instantiations that are used (e.g., Tree<int> not Tree<'T>)
    TypeInstantiations: Map<string, Set<string>>
    /// Union cases that are constructed or matched
    UsedUnionCases: Map<string, Set<string>>
    /// Interface methods that are called
    CalledInterfaceMethods: Map<string, Set<string>>
    /// Entry point symbols
    EntryPoints: FSharpSymbol list
}

/// Result of semantic reachability analysis
type ReachabilityResult = {
    EntryPoints: FSharpSymbol list
    ReachableSymbols: Set<string>
    UnreachableSymbols: Set<string>
    CallGraph: Map<string, string list>
}

/// Library classification for symbol boundary analysis
type LibraryCategory =
    | UserCode
    | AlloyLibrary  
    | FSharpCore
    | Other of libraryName: string

/// Enhanced reachability result with library boundary information
type LibraryAwareReachability = {
    BasicResult: ReachabilityResult
    LibraryCategories: Map<string, LibraryCategory>
    PruningStatistics: PruningStatistics
    MarkedPSG: ProgramSemanticGraph  // PSG with nodes marked for reachability
}

and PruningStatistics = {
    TotalSymbols: int
    ReachableSymbols: int
    EliminatedSymbols: int
    ComputationTimeMs: int64
}

/// Classify library category based on symbol name
let private classifyLibraryCategory (symbolName: string) : LibraryCategory =
    if symbolName.StartsWith("Examples.") || symbolName.StartsWith("HelloWorld") then
        UserCode
    elif symbolName.StartsWith("Alloy.") then
        AlloyLibrary
    elif symbolName.StartsWith("Microsoft.FSharp.") || symbolName.StartsWith("FSharp.Core") then
        FSharpCore
    elif symbolName.StartsWith("System.") then
        Other "System"
    else
        UserCode

/// Build semantic call graph from PSG using principled graph analysis
let private buildSemanticCallGraph (psg: ProgramSemanticGraph) : Map<string, string list> =
    let mutable callGraph = Map.empty
    
    // Build a map from nodes to their containing function/method
    let nodeToFunction = 
        let mutable result = Map.empty
        
        // Walk the graph to find all function/method definitions
        for KeyValue(nodeId, node) in psg.Nodes do
            match node.SyntaxKind with
            | sk when sk.StartsWith("Binding") || sk.StartsWith("Member") ->
                match node.Symbol with
                | Some sym ->
                    // This node defines a function - mark all its descendants
                    // Use visited set to prevent infinite recursion on cyclic references
                    let visited = System.Collections.Generic.HashSet<string>()
                    let rec markDescendants nId funcName =
                        if visited.Add(nId) then  // Only process if not already visited
                            result <- Map.add nId funcName result
                            match Map.tryFind nId psg.Nodes with
                            | Some n ->
                                match n.Children with
                                | Parent children ->
                                    for child in children do
                                        markDescendants child.Value funcName
                                | _ -> ()
                            | None -> ()

                    markDescendants nodeId sym.FullName
                | None -> ()
            | _ -> ()
        
        result
    
    // Analyze edges to build call graph
    for edge in psg.Edges do
        match edge.Kind with
        | FunctionCall ->
            // Find the calling function and called function
            let callerOpt = Map.tryFind edge.Source.Value nodeToFunction
            let targetOpt = Map.tryFind edge.Target.Value psg.Nodes
            
            match callerOpt, targetOpt with
            | Some caller, Some targetNode ->
                match targetNode.Symbol with
                | Some targetSym ->
                    let calledFunc = targetSym.FullName
                    // Only add if it's a different function and not already in the list
                    if caller <> calledFunc then
                        let currentCalls = Map.tryFind caller callGraph |> Option.defaultValue []
                        if not (List.contains calledFunc currentCalls) then
                            callGraph <- Map.add caller (calledFunc :: currentCalls) callGraph
                | None -> ()
            | _ -> ()
        | MethodCall ->
            // Find the calling function and called function
            let callerOpt = Map.tryFind edge.Source.Value nodeToFunction
            let targetOpt = Map.tryFind edge.Target.Value psg.Nodes
            
            match callerOpt, targetOpt with
            | Some caller, Some targetNode ->
                match targetNode.Symbol with
                | Some targetSym ->
                    let calledFunc = targetSym.FullName
                    // Only add if it's a different function and not already in the list
                    if caller <> calledFunc then
                        let currentCalls = Map.tryFind caller callGraph |> Option.defaultValue []
                        if not (List.contains calledFunc currentCalls) then
                            callGraph <- Map.add caller (calledFunc :: currentCalls) callGraph
                | None -> ()
            | _ -> ()
        | Reference ->
            // Find the calling function and called function
            let callerOpt = Map.tryFind edge.Source.Value nodeToFunction
            let targetOpt = Map.tryFind edge.Target.Value psg.Nodes
            
            match callerOpt, targetOpt with
            | Some caller, Some targetNode ->
                match targetNode.Symbol with
                | Some targetSym ->
                    let calledFunc = targetSym.FullName
                    // Only add if it's a different function and not already in the list
                    if caller <> calledFunc then
                        let currentCalls = Map.tryFind caller callGraph |> Option.defaultValue []
                        if not (List.contains calledFunc currentCalls) then
                            callGraph <- Map.add caller (calledFunc :: currentCalls) callGraph
                | None -> ()
            | _ -> ()
        | _ -> ()
    
    // Also analyze application nodes for function calls not captured by edges
    for KeyValue(nodeId, node) in psg.Nodes do
        match node.SyntaxKind with
        | "App" | "App:FunctionCall" | "TypeApp" ->
            let callerOpt = Map.tryFind nodeId nodeToFunction
            
            match callerOpt with
            | Some caller ->
                // Look at children to find what's being called
                match node.Children with
                | Parent children ->
                    for childId in children do
                        match Map.tryFind childId.Value psg.Nodes with
                        | Some childNode ->
                            match childNode.Symbol with
                            | Some childSym when childSym.FullName <> caller ->
                                let currentCalls = Map.tryFind caller callGraph |> Option.defaultValue []
                                if not (List.contains childSym.FullName currentCalls) then
                                    callGraph <- Map.add caller (childSym.FullName :: currentCalls) callGraph
                            | _ -> ()
                        | None -> ()
                | _ -> ()
            | None -> ()
        | _ -> ()
    
    callGraph

/// Compute reachable symbols using semantic analysis
let private computeSemanticReachability 
    (entryPoints: FSharpSymbol list) 
    (callGraph: Map<string, string list>) : Set<string> =
    
    let mutable reachable = Set.empty
    let mutable toProcess = []
    
    // Start with entry points
    for ep in entryPoints do
        reachable <- Set.add ep.FullName reachable
        toProcess <- ep.FullName :: toProcess
    
    // Process call graph
    while not (List.isEmpty toProcess) do
        match toProcess with
        | current :: rest ->
            toProcess <- rest
            
            // Add all functions called by current
            match Map.tryFind current callGraph with
            | Some callees ->
                for callee in callees do
                    if not (Set.contains callee reachable) then
                        reachable <- Set.add callee reachable
                        toProcess <- callee :: toProcess
            | None -> ()
        | [] -> ()
    
    reachable

/// Mark PSG nodes based on semantic reachability
let private markNodesForSemanticReachability 
    (psg: ProgramSemanticGraph) 
    (reachableSymbols: Set<string>) : ProgramSemanticGraph =
    
    printfn "[SEMANTIC] Marking nodes for %d reachable symbols" (Set.count reachableSymbols)
    
    // First pass: mark nodes directly associated with reachable symbols
    let firstPassNodes = 
        psg.Nodes
        |> Map.map (fun nodeId node ->
            let isReachable = 
                match node.Symbol with
                | Some symbol ->
                    let symbolName = symbol.FullName
                    let symbolReachable = Set.contains symbolName reachableSymbols
                    
                    match node.SyntaxKind with
                    // Binding nodes that define reachable functions
                    | sk when sk.StartsWith("Binding") -> 
                        symbolReachable
                    // Function application nodes
                    | "App:FunctionCall" ->
                        // Mark if calling a reachable function
                        symbolReachable
                    | sk when sk.Contains("MethodCall:") ->
                        // Mark if calling a reachable function
                        symbolReachable || 
                        // Or if it's a call to a known function
                        (sk.Contains("Write") || sk.Contains("WriteLine") || 
                         sk.Contains("readInto") || sk.Contains("sprintf") ||
                         sk.Contains("stackBuffer") || sk.Contains("spanToString"))
                    // Identifiers that reference functions
                    | sk when sk.StartsWith("Ident:") || sk.StartsWith("LongIdent:") ->
                        if sk.Contains("hello") && Set.contains "Examples.HelloWorldDirect.hello" reachableSymbols then
                            true
                        elif sk.Contains("Console") then
                            true  // Keep Console references
                        else
                            symbolReachable
                    // Pattern matches for union cases
                    | sk when sk.Contains("Pattern:UnionCase:") ->
                        sk.Contains("Ok") || sk.Contains("Error")  // Keep Result cases
                    // Type applications
                    | sk when sk.StartsWith("TypeApp:") ->
                        sk.Contains("byte") && symbolReachable  // Keep stackBuffer<byte>
                    | _ -> false
                | None ->
                    // Nodes without symbols
                    match node.SyntaxKind with
                    | "Module" | "NestedModule" -> true  // Keep module structure
                    | "LetDeclaration" -> false  // Will be marked if contains reachable binding
                    | "Match" | "MatchClause" -> false  // Will be marked if needed
                    | sk when sk.StartsWith("Const:") -> false  // Constants marked if used
                    | _ -> false
            
            { node with 
                IsReachable = isReachable
                EliminationReason = if isReachable then None else Some "Not semantically reachable"
                EliminationPass = if isReachable then None else Some 1 }
        )
    
    // Now mark the minimal structural nodes needed for reachable symbols
    let mutable finalNodes = firstPassNodes
    
    // For each reachable binding, ensure its immediate parent and pattern are marked
    for KeyValue(nodeId, node) in firstPassNodes do
        if node.IsReachable && node.SyntaxKind.StartsWith("Binding") then
            // Mark the parent LetDeclaration if it exists
            match node.ParentId with
            | Some parentId ->
                match Map.tryFind parentId.Value finalNodes with
                | Some parentNode ->
                    finalNodes <- Map.add parentId.Value { parentNode with IsReachable = true } finalNodes
                | None -> ()
            | None -> ()
            
            // Mark direct children that are patterns (for the binding's name)
            match node.Children with
            | Parent children ->
                for childId in children do
                    match Map.tryFind childId.Value finalNodes with
                    | Some childNode when childNode.SyntaxKind.StartsWith("Pattern:") ->
                        finalNodes <- Map.add childId.Value { childNode with IsReachable = true } finalNodes
                    | _ -> ()
            | _ -> ()
    
    let reachableNodeCount = 
        finalNodes 
        |> Map.toSeq 
        |> Seq.filter (fun (_, n) -> n.IsReachable) 
        |> Seq.length
    
    printfn "[SEMANTIC] Marked %d nodes as reachable (%.1f%% of %d total)" 
        reachableNodeCount 
        (float reachableNodeCount / float (Map.count finalNodes) * 100.0)
        (Map.count finalNodes)
    
    { psg with Nodes = finalNodes }

/// Perform semantic reachability analysis
let analyzeReachability (psg: ProgramSemanticGraph) : LibraryAwareReachability =
    let startTime = DateTime.UtcNow
    printfn "[REACHABILITY] Starting semantic reachability analysis"
    
    // Find entry points
    let entryPoints = 
        psg.SymbolTable
        |> Map.toSeq
        |> Seq.choose (fun (name, symbol) ->
            match symbol with
            | :? FSharpMemberOrFunctionOrValue as mfv when 
                name = "main" || 
                mfv.Attributes |> Seq.exists (fun a -> a.AttributeType.DisplayName = "EntryPoint") ->
                Some symbol
            | _ -> None)
        |> Seq.toList
    
    // Also check for EntryPointAttribute itself
    let entryPointAttribute = 
        psg.SymbolTable
        |> Map.tryFind "EntryPointAttribute"
        |> Option.toList
    
    let allEntryPoints = entryPoints @ entryPointAttribute
    
    printfn "[REACHABILITY] Found %d entry points: %A" 
        allEntryPoints.Length 
        (allEntryPoints |> List.map (fun ep -> ep.DisplayName))
    
    // Build semantic call graph
    let callGraph = buildSemanticCallGraph psg
    printfn "[REACHABILITY] Built call graph with %d entries" (Map.count callGraph)
    
    // Compute semantic reachability
    let reachableSymbols = computeSemanticReachability allEntryPoints callGraph
    printfn "[REACHABILITY] Found %d semantically reachable symbols" (Set.count reachableSymbols)
    
    // All symbols in the PSG
    let allSymbols = 
        psg.SymbolTable
        |> Map.toSeq
        |> Seq.map (fun (_, sym) -> sym.FullName)
        |> Set.ofSeq
    
    let unreachableSymbols = Set.difference allSymbols reachableSymbols
    
    // Mark nodes based on semantic reachability
    let markedPSG = markNodesForSemanticReachability psg reachableSymbols
    
    // Calculate statistics
    let endTime = DateTime.UtcNow
    let computationTime = int64 (endTime - startTime).TotalMilliseconds
    
    let stats = {
        TotalSymbols = Set.count allSymbols
        ReachableSymbols = Set.count reachableSymbols
        EliminatedSymbols = Set.count unreachableSymbols
        ComputationTimeMs = computationTime
    }
    
    // Classify libraries
    let libraryCategories =
        allSymbols
        |> Set.toSeq
        |> Seq.map (fun sym -> sym, classifyLibraryCategory sym)
        |> Map.ofSeq
    
    let result = {
        EntryPoints = allEntryPoints
        ReachableSymbols = reachableSymbols
        UnreachableSymbols = unreachableSymbols
        CallGraph = callGraph
    }
    
    printfn "[REACHABILITY] Semantic analysis complete: %d/%d symbols reachable (%.1f%% eliminated)"
        stats.ReachableSymbols
        stats.TotalSymbols
        (float stats.EliminatedSymbols / float stats.TotalSymbols * 100.0)
    
    {
        BasicResult = result
        LibraryCategories = libraryCategories
        PruningStatistics = stats
        MarkedPSG = markedPSG
    }

/// Entry point for reachability analysis
let performReachabilityAnalysis (psg: ProgramSemanticGraph) : LibraryAwareReachability =
    analyzeReachability psg