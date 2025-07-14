module Core.PSG.Builder

open System
open System.IO
open FSharp.Compiler.Text
open FSharp.Compiler.Syntax
open FSharp.Compiler.CodeAnalysis
open Core.PSG.Types
open Core.PSG.Correlation
open Core.PSG.SymbolAnalysis

/// Build context for PSG construction
type BuildContext = {
    CheckResults: FSharpCheckProjectResults
    ParseResults: FSharpParseFileResults[]
    CorrelationContext: CorrelationContext
    SourceFiles: Map<string, string>
}

let private createNode syntaxKind range fileName symbol parentId =
    // Always use range-based IDs for syntax tree structure
    let cleanKind = (syntaxKind : string).Replace(":", "_").Replace(" ", "_")
    let uniqueFileName = sprintf "%s_%s.fs" (System.IO.Path.GetFileNameWithoutExtension(fileName : string)) cleanKind
    let nodeId = NodeId.FromRange(uniqueFileName, range)
    
    // Use helper function to create node with NotProcessed children state
    ChildrenStateHelpers.createWithNotProcessed nodeId syntaxKind symbol range fileName parentId

/// Add child to parent and return updated graph
let private addChildToParent (childId: NodeId) (parentId: NodeId option) (graph: ProgramSemanticGraph) =
    match parentId with
    | None -> graph
    | Some pid ->
        match Map.tryFind pid.Value graph.Nodes with
        | Some parentNode ->
            // Use helper function to add child to parent's children state
            let updatedParent = ChildrenStateHelpers.addChild childId parentNode
            
            // Create ChildOf edge: parent -> child
            let childOfEdge = {
                Source = pid
                Target = childId
                Kind = ChildOf
            }
            
            { graph with 
                Nodes = Map.add pid.Value updatedParent graph.Nodes
                Edges = childOfEdge :: graph.Edges }
        | None -> graph

/// Process a binding (let/member)
let rec private processBinding binding parentId fileName context graph =
    let (SynBinding(_, _, _, _, _, _, _, pat, _, expr, range, _, _)) = binding
    
    let symbol = tryCorrelateSymbol range fileName context.CorrelationContext
    let bindingNode = createNode "Binding" range fileName symbol parentId
    
    // Add node to graph
    let graph' = { graph with Nodes = Map.add bindingNode.Id.Value bindingNode graph.Nodes }
    
    // Update parent's children
    let graph'' = addChildToParent bindingNode.Id parentId graph'
    
    // Update symbol table if we have a symbol
    let graph''' = 
        match symbol with
        | Some sym -> 
            { graph'' with SymbolTable = Map.add sym.DisplayName sym graph''.SymbolTable }
        | None -> graph''
    
    // Process pattern and expression
    let graph'''' = processPattern pat (Some bindingNode.Id) fileName context graph'''
    processExpression expr (Some bindingNode.Id) fileName context graph''''

/// Process a pattern
and private processPattern pat parentId fileName context graph =
    match pat with
    | SynPat.Named(synIdent, _, _, range) ->
        let (SynIdent(ident, _)) = synIdent
        
        // DEBUG: Log the details before creating the node
        let futureNodeId = NodeId.FromRange(fileName, range)
        
        // Check for self-reference before creating
        match parentId with
        | Some pid when pid.Value = futureNodeId.Value ->
            failwith "Pattern node assigned itself as parent"
        | _ -> ()
        
        let symbol = tryCorrelateSymbol range fileName context.CorrelationContext
        let patNode = createNode (sprintf "Pattern:Named:%s" ident.idText) range fileName symbol parentId
        
        let graph' = { graph with Nodes = Map.add patNode.Id.Value patNode graph.Nodes }
        let graph'' = addChildToParent patNode.Id parentId graph'
        
        // Add to symbol table if correlated
        match symbol with
        | Some sym ->
            { graph'' with SymbolTable = Map.add ident.idText sym graph''.SymbolTable }
        | None -> graph''
        
    | SynPat.Wild range ->
        let wildNode = createNode "Pattern:Wild" range fileName None parentId
        let graph' = { graph with Nodes = Map.add wildNode.Id.Value wildNode graph.Nodes }
        addChildToParent wildNode.Id parentId graph'
        
    | SynPat.Typed(innerPat, _, range) ->
        let typedNode = createNode "Pattern:Typed" range fileName None parentId
        let graph' = { graph with Nodes = Map.add typedNode.Id.Value typedNode graph.Nodes }
        let graph'' = addChildToParent typedNode.Id parentId graph'
        processPattern innerPat (Some typedNode.Id) fileName context graph''
        
    | _ -> graph

/// Process an expression
and private processExpression expr parentId fileName context graph =
    match expr with
    | SynExpr.Ident ident ->
        let identNode = createNode (sprintf "Ident:%s" ident.idText) expr.Range fileName None parentId
        let graph' = { graph with Nodes = Map.add identNode.Id.Value identNode graph.Nodes }
        let graph'' = addChildToParent identNode.Id parentId graph'
        
        // Create edge to symbol definition if found
        match Map.tryFind ident.idText graph''.SymbolTable with
        | Some targetSymbol ->
            let edge = {
                Source = identNode.Id
                Target = NodeId.FromSymbol(targetSymbol)
                Kind = SymRef
            }
            { graph'' with Edges = edge :: graph''.Edges }
        | None -> graph''
        
    | SynExpr.App(_, _, funcExpr, argExpr, range) ->
        let appNode = createNode "Application" range fileName None parentId
        let graph' = { graph with Nodes = Map.add appNode.Id.Value appNode graph.Nodes }
        let graph'' = addChildToParent appNode.Id parentId graph'
        
        // Process function and argument
        let graph''' = processExpression funcExpr (Some appNode.Id) fileName context graph''
        let graph'''' = processExpression argExpr (Some appNode.Id) fileName context graph'''
        
        // Try to create call edge - handle different function expression types
        match funcExpr with
        | SynExpr.Ident funcIdent ->
            match Map.tryFind funcIdent.idText graph''''.SymbolTable with
            | Some funcSymbol ->
                // Find the actual node that has this symbol
                let targetNode = 
                    graph''''.Nodes
                    |> Map.tryPick (fun nodeId node -> 
                        match node.Symbol with
                        | Some sym when sym = funcSymbol -> Some node
                        | _ -> None)
                
                match targetNode with
                | Some target ->
                    let edge = {
                        Source = appNode.Id
                        Target = target.Id  // ✅ Use the actual node's ID
                        Kind = FunctionCall
                    }
                    { graph'''' with Edges = edge :: graph''''.Edges }
                | None -> graph''''
            | None -> graph''''
        | SynExpr.LongIdent(_, longIdent, _, _) ->
            // Handle qualified function calls like Module.func
            let (SynLongIdent(ids, _, _)) = longIdent
            let funcName = ids |> List.map (fun id -> id.idText) |> String.concat "."
            
            match Map.tryFind funcName graph''''.SymbolTable with
            | Some funcSymbol ->
                let targetNode = 
                    graph''''.Nodes
                    |> Map.tryPick (fun nodeId node -> 
                        match node.Symbol with
                        | Some sym when sym = funcSymbol -> Some node
                        | _ -> None)
                
                match targetNode with
                | Some target ->
                    let edge = {
                        Source = appNode.Id
                        Target = target.Id
                        Kind = FunctionCall
                    }
                    { graph'''' with Edges = edge :: graph''''.Edges }
                | None -> graph''''
            | None -> graph''''
        | _ -> graph''''
        
    | SynExpr.LetOrUse(_, _, bindings, body, range, _) ->
        let letNode = createNode "LocalLet" range fileName None parentId
        let graph' = { graph with Nodes = Map.add letNode.Id.Value letNode graph.Nodes }
        let graph'' = addChildToParent letNode.Id parentId graph'
        
        // Process bindings
        let graph''' = 
            bindings
            |> List.fold (fun acc binding -> 
                processBinding binding (Some letNode.Id) fileName context acc) graph''
        
        // Process body
        processExpression body (Some letNode.Id) fileName context graph'''
        
    | SynExpr.Paren(innerExpr, _, _, range) ->
        let parenNode = createNode "Parenthesized" range fileName None parentId
        let graph' = { graph with Nodes = Map.add parenNode.Id.Value parenNode graph.Nodes }
        let graph'' = addChildToParent parenNode.Id parentId graph'
        processExpression innerExpr (Some parenNode.Id) fileName context graph''
        
    | SynExpr.Const(constant, range) ->
        let constNode = createNode (sprintf "Constant:%A" constant) range fileName None parentId
        let graph' = { graph with Nodes = Map.add constNode.Id.Value constNode graph.Nodes }
        addChildToParent constNode.Id parentId graph'
        
    | _ -> graph

/// Process a module declaration
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
        let (SynComponentInfo(_, _, _, longId, _, _, _, _)) = componentInfo
        let moduleName = longId |> List.map (fun id -> id.idText) |> String.concat "."
        let symbol = tryCorrelateSymbol range fileName context.CorrelationContext
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

/// Process implementation file - using correct FCS 43.9.300 pattern
let rec private processImplFile (implFile: SynModuleOrNamespace) context graph =
    let (SynModuleOrNamespace(name, _, _, decls, _, _, _, range, _)) = implFile
    
    let moduleName = name |> List.map (fun i -> i.idText) |> String.concat "."
    let fileName = range.FileName
    
    let symbol = tryCorrelateSymbol range fileName context.CorrelationContext
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

/// Build complete PSG from project results
let buildProgramSemanticGraph 
    (checkResults: FSharpCheckProjectResults) 
    (parseResults: FSharpParseFileResults[]) : ProgramSemanticGraph =
    
    // Create enhanced correlation context
    let correlationContext = createContext checkResults
    
    // Load source files
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
    
    // Create build context
    let context = {
        CheckResults = checkResults
        ParseResults = parseResults
        CorrelationContext = correlationContext
        SourceFiles = sourceFiles
    }
    
    // Process each file and merge results
    let graphs = 
        parseResults
        |> Array.choose (fun pr ->
            match pr.ParseTree with
            | ParsedInput.ImplFile implFile ->
                // Correct FCS 43.9.300 pattern for ParsedImplFileInput
                let (ParsedImplFileInput(contents = modules)) = implFile
                let emptyGraph = {
                    Nodes = Map.empty
                    Edges = []
                    SymbolTable = Map.empty
                    EntryPoints = []
                    SourceFiles = sourceFiles
                    CompilationOrder = []
                }
                // Process each module in the file
                let processedGraph = 
                    modules |> List.fold (fun acc implFile -> 
                        processImplFile implFile context acc) emptyGraph
                Some processedGraph
            | _ -> None
        )
    
    // Merge all graphs
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
    
    // Set compilation order based on file order and finalize all nodes
    let finalNodes = 
        mergedGraph.Nodes
        |> Map.map (fun _ node -> ChildrenStateHelpers.finalizeChildren node)
    
    let finalGraph = 
        { mergedGraph with 
            Nodes = finalNodes
            CompilationOrder = parseResults |> Array.map (fun pr -> pr.FileName) |> List.ofArray 
        }
        
    finalGraph