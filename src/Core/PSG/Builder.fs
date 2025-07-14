module Core.PSG.Builder

open System
open System.IO
open FSharp.Compiler.Text
open FSharp.Compiler.Syntax
open FSharp.Compiler.CodeAnalysis
open Core.PSG.Types
open Core.PSG.Correlation
open Core.PSG.TypeIntegration

/// Build context for PSG construction
type BuildContext = {
    CheckResults: FSharpCheckProjectResults
    ParseResults: FSharpParseFileResults[]
    CorrelationContext: CorrelationContext
    SourceFiles: Map<string, string>
}

let private createNode syntaxKind range fileName symbol parentId =
    let cleanKind = (syntaxKind : string).Replace(":", "_").Replace(" ", "_")
    let uniqueFileName = sprintf "%s_%s.fs" (System.IO.Path.GetFileNameWithoutExtension(fileName : string)) cleanKind
    let nodeId = NodeId.FromRange(uniqueFileName, range)
    
    ChildrenStateHelpers.createWithNotProcessed nodeId syntaxKind symbol range fileName parentId

/// Add child to parent and return updated graph
let private addChildToParent (childId: NodeId) (parentId: NodeId option) (graph: ProgramSemanticGraph) =
    match parentId with
    | None -> graph
    | Some pid ->
        match Map.tryFind pid.Value graph.Nodes with
        | Some parentNode ->
            let updatedParent = ChildrenStateHelpers.addChild childId parentNode
            
            let childOfEdge = {
                Source = pid
                Target = childId
                Kind = ChildOf
            }
            
            { graph with 
                Nodes = Map.add pid.Value updatedParent graph.Nodes
                Edges = childOfEdge :: graph.Edges }
        | None -> graph

/// Process a binding (let/member) - Using existing FCS 43.9.300 patterns
let rec private processBinding binding parentId fileName context graph =
    let (SynBinding(accessibility, kind, isInline, isMutable, attributes, xmlDoc, valData, pat, returnInfo, expr, range, seqPoint, trivia)) = binding
    
    let symbol : FSharp.Compiler.Symbols.FSharpSymbol option = tryCorrelateSymbol range fileName context.CorrelationContext
    let bindingNode = createNode "Binding" range fileName symbol parentId
    
    let graph' = { graph with Nodes = Map.add bindingNode.Id.Value bindingNode graph.Nodes }
    let graph'' = addChildToParent bindingNode.Id parentId graph'
    
    let graph''' = 
        match symbol with
        | Some sym -> 
            { graph'' with SymbolTable = Map.add sym.DisplayName sym graph''.SymbolTable }
        | None -> graph''
    
    let graph'''' = processPattern pat (Some bindingNode.Id) fileName context graph'''
    processExpression expr (Some bindingNode.Id) fileName context graph''''

/// Process a pattern - Using existing FCS 43.9.300 patterns  
and private processPattern pat parentId fileName context graph =
    match pat with
    | SynPat.Named(synIdent, _, _, range) ->
        let (SynIdent(ident, _)) = synIdent
        
        let futureNodeId = NodeId.FromRange(fileName, range)
        
        match parentId with
        | Some pid when pid.Value = futureNodeId.Value ->
            failwith "Pattern node assigned itself as parent"
        | _ -> ()
        
        let symbol : FSharp.Compiler.Symbols.FSharpSymbol option = tryCorrelateSymbol range fileName context.CorrelationContext
        let patNode = createNode (sprintf "Pattern:Named:%s" ident.idText) range fileName symbol parentId
        
        let graph' = { graph with Nodes = Map.add patNode.Id.Value patNode graph.Nodes }
        let graph'' = addChildToParent patNode.Id parentId graph'
        
        match symbol with
        | Some sym -> 
            { graph'' with SymbolTable = Map.add sym.DisplayName sym graph''.SymbolTable }
        | None -> graph''
        
    | _ -> graph

/// Process expression nodes - Using existing FCS 43.9.300 patterns
and private processExpression (expr: SynExpr) (parentId: NodeId option) (fileName: string) 
                              (context: BuildContext) (graph: ProgramSemanticGraph) =
    match expr with
    | SynExpr.Ident ident ->
        let symbol = tryCorrelateSymbol ident.idRange fileName context.CorrelationContext
        let identNode = createNode (sprintf "Ident:%s" ident.idText) ident.idRange fileName symbol parentId
        
        let graph' = { graph with Nodes = Map.add identNode.Id.Value identNode graph.Nodes }
        let graph'' = addChildToParent identNode.Id parentId graph'
        
        match symbol with
        | Some sym -> 
            { graph'' with SymbolTable = Map.add sym.DisplayName sym graph''.SymbolTable }
        | None -> graph''
    
    | SynExpr.App(_, _, funcExpr, argExpr, range) ->
        let symbol = tryCorrelateSymbol range fileName context.CorrelationContext
        let appNode = createNode "App" range fileName symbol parentId
        
        let graph' = { graph with Nodes = Map.add appNode.Id.Value appNode graph.Nodes }
        let graph'' = addChildToParent appNode.Id parentId graph'
        
        let graph''' = 
            match symbol with
            | Some sym -> 
                { graph'' with SymbolTable = Map.add sym.DisplayName sym graph''.SymbolTable }
            | None -> graph''
        
        let graph4 = processExpression funcExpr (Some appNode.Id) fileName context graph'''
        let graph5 = processExpression argExpr (Some appNode.Id) fileName context graph4
        
        // Create FunctionCall edge from caller symbol to callee symbol
        let createCallEdgeFromSymbol (callerSymbol: FSharp.Compiler.Symbols.FSharpSymbol) (funcSymbol: FSharp.Compiler.Symbols.FSharpSymbol) =
            let callerNode = 
                graph5.Nodes
                |> Map.tryPick (fun _ node -> 
                    match node.Symbol with
                    | Some (sym: FSharp.Compiler.Symbols.FSharpSymbol) when sym.DisplayName = callerSymbol.DisplayName -> Some node
                    | _ -> None)
            
            let targetNode = 
                graph5.Nodes
                |> Map.tryPick (fun _ node -> 
                    match node.Symbol with
                    | Some (sym: FSharp.Compiler.Symbols.FSharpSymbol) when sym.DisplayName = funcSymbol.DisplayName -> Some node
                    | _ -> None)
                    
            match callerNode, targetNode with
            | Some caller, Some target ->
                let callEdge = { Source = caller.Id; Target = target.Id; Kind = FunctionCall }
                { graph5 with Edges = callEdge :: graph5.Edges }
            | _ -> graph5
        
        // Create FunctionCall edge if function expression has a symbol
        match funcExpr with
        | SynExpr.Ident ident ->
            let funcSymbol = tryCorrelateSymbol ident.idRange fileName context.CorrelationContext
            match funcSymbol, symbol with
            | Some funcSym, Some callerSym ->
                createCallEdgeFromSymbol callerSym funcSym
            | Some funcSym, None ->
                // If no caller symbol, try to find enclosing function
                let targetNode = 
                    graph5.Nodes
                    |> Map.tryPick (fun _ node -> 
                        match node.Symbol with
                        | Some sym when sym.DisplayName = funcSym.DisplayName -> Some node
                        | _ -> None)
                match targetNode with
                | Some target ->
                    let callEdge = { Source = appNode.Id; Target = target.Id; Kind = FunctionCall }
                    { graph5 with Edges = callEdge :: graph5.Edges }
                | None -> graph5
            | _ -> graph5
        | _ -> graph5
    
    | SynExpr.LetOrUse(_, _, bindings, body, range, _) ->
        let symbol = tryCorrelateSymbol range fileName context.CorrelationContext
        let letNode = createNode "LetOrUse" range fileName symbol parentId
        
        let graph' = { graph with Nodes = Map.add letNode.Id.Value letNode graph.Nodes }
        let graph'' = addChildToParent letNode.Id parentId graph'
        
        let graph''' = 
            match symbol with
            | Some sym -> 
                { graph'' with SymbolTable = Map.add sym.DisplayName sym graph''.SymbolTable }
            | None -> graph''
        
        let graph'''' = 
            bindings
            |> List.fold (fun acc binding -> 
                processBinding binding (Some letNode.Id) fileName context acc) graph'''
        
        processExpression body (Some letNode.Id) fileName context graph''''
        
    | _ -> graph

/// Process a module declaration - Using existing FCS 43.9.300 patterns
and private processModuleDecl decl parentId fileName context graph =
    match decl with
    | SynModuleDecl.Let(_, bindings, range) ->
        let letDeclNode = createNode "LetDeclaration" range fileName None parentId
        let graph' = { graph with Nodes = Map.add letDeclNode.Id.Value letDeclNode graph.Nodes }
        let graph'' = addChildToParent letDeclNode.Id parentId graph'
        
        bindings |> List.fold (fun g binding ->
            processBinding binding (Some letDeclNode.Id) fileName context g
        ) graph''
        
    | SynModuleDecl.Open(_, range) ->
        let openNode = createNode "Open" range fileName None parentId
        let graph' = { graph with Nodes = Map.add openNode.Id.Value openNode graph.Nodes }
        addChildToParent openNode.Id parentId graph'
        
    | SynModuleDecl.NestedModule(componentInfo, _, decls, _, range, _) ->
        let (SynComponentInfo(attributes, typeParams, constraints, longId, xmlDoc, preferPostfix, accessibility, range2)) = componentInfo
        let moduleName = longId |> List.map (fun id -> id.idText) |> String.concat "."
        
        let symbol : FSharp.Compiler.Symbols.FSharpSymbol option = tryCorrelateSymbol range fileName context.CorrelationContext
        let nestedModuleNode = createNode (sprintf "NestedModule:%s" moduleName) range fileName symbol parentId
        
        let graph' = { graph with Nodes = Map.add nestedModuleNode.Id.Value nestedModuleNode graph.Nodes }
        let graph'' = addChildToParent nestedModuleNode.Id parentId graph'
        
        let graph''' = 
            match symbol with
            | Some sym -> 
                { graph'' with SymbolTable = Map.add sym.DisplayName sym graph''.SymbolTable }
            | None -> graph''
        
        decls |> List.fold (fun g decl ->
            processModuleDecl decl (Some nestedModuleNode.Id) fileName context g
        ) graph'''
        
    | _ -> graph

/// Process implementation file - Using existing FCS 43.9.300 patterns
let rec private processImplFile (implFile: SynModuleOrNamespace) context graph =
    let (SynModuleOrNamespace(name, _, _, decls, _, _, _, range, _)) = implFile
    
    let moduleName = name |> List.map (fun i -> i.idText) |> String.concat "."
    let fileName = range.FileName
    
    let symbol : FSharp.Compiler.Symbols.FSharpSymbol option = tryCorrelateSymbol range fileName context.CorrelationContext
    let moduleNode = createNode (sprintf "Module:%s" moduleName) range fileName symbol None
    
    let graph' = { graph with Nodes = Map.add moduleNode.Id.Value moduleNode graph.Nodes }
    
    let graph'' = 
        match symbol with
        | Some sym -> 
            { graph' with SymbolTable = Map.add sym.DisplayName sym graph'.SymbolTable }
        | None -> graph'
    
    decls
    |> List.fold (fun acc decl -> 
        processModuleDecl decl (Some moduleNode.Id) fileName context acc) graph''

/// Build complete PSG from project results with CONSTRAINT-AWARE TYPE INTEGRATION
let buildProgramSemanticGraph 
    (checkResults: FSharpCheckProjectResults) 
    (parseResults: FSharpParseFileResults[]) : ProgramSemanticGraph =
    
    let correlationContext = createContext checkResults
    
    let sourceFiles =
        parseResults
        |> Array.map (fun pr ->
            let content = 
                if File.Exists pr.FileName then
                    File.ReadAllText pr.FileName
                else ""
            pr.FileName, content
        )
        |> Map.ofArray
    
    let context = {
        CheckResults = checkResults
        ParseResults = parseResults
        CorrelationContext = correlationContext
        SourceFiles = sourceFiles
    }
    
    // Process each file and merge results - Using existing working patterns
    let graphs = 
        parseResults
        |> Array.choose (fun pr ->
            match pr.ParseTree with
            | ParsedInput.ImplFile implFile ->
                let (ParsedImplFileInput(contents = modules)) = implFile
                let emptyGraph = {
                    Nodes = Map.empty
                    Edges = []
                    SymbolTable = Map.empty
                    EntryPoints = []
                    SourceFiles = sourceFiles
                    CompilationOrder = []
                }
                let processedGraph = 
                    modules |> List.fold (fun acc implFile -> 
                        processImplFile implFile context acc) emptyGraph
                Some processedGraph
            | _ -> None
        )
    
    // Merge all graphs - Using existing working patterns
    let mergedGraph =
        if Array.isEmpty graphs then
            {
                Nodes = Map.empty
                Edges = []
                SymbolTable = Map.empty
                EntryPoints = []
                SourceFiles = sourceFiles
                CompilationOrder = []
            }
        else
            graphs |> Array.reduce (fun g1 g2 ->
                {
                    Nodes = Map.fold (fun acc k v -> Map.add k v acc) g1.Nodes g2.Nodes
                    Edges = g1.Edges @ g2.Edges
                    SymbolTable = Map.fold (fun acc k v -> Map.add k v acc) g1.SymbolTable g2.SymbolTable
                    EntryPoints = g1.EntryPoints @ g2.EntryPoints
                    SourceFiles = Map.fold (fun acc k v -> Map.add k v acc) g1.SourceFiles g2.SourceFiles
                    CompilationOrder = g1.CompilationOrder @ g2.CompilationOrder
                }
            )
    
    // CRITICAL CHANGE: Use enhanced constraint-aware type integration
    printfn "[BUILDER] Applying enhanced constraint-aware type integration to PSG with %d nodes" mergedGraph.Nodes.Count
    let typeEnhancedGraph = integrateTypesWithCheckResults mergedGraph checkResults
    
    // Finalize all nodes after type integration - Using existing working patterns
    let finalNodes = 
        typeEnhancedGraph.Nodes
        |> Map.map (fun _ node -> ChildrenStateHelpers.finalizeChildren node)
    
    let finalGraph = 
        { typeEnhancedGraph with 
            Nodes = finalNodes
            CompilationOrder = parseResults |> Array.map (fun pr -> pr.FileName) |> List.ofArray 
        }
        
    finalGraph