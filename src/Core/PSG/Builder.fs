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
    
    {
        Id = nodeId
        SyntaxKind = syntaxKind
        Symbol = symbol  // Symbol correlation is separate from node identity
        Range = range
        SourceFile = fileName  // Keep original filename for the node
        ParentId = parentId
        Children = []
    }

/// Add child to parent and return updated graph
let private addChildToParent (childId: NodeId) (parentId: NodeId option) (graph: ProgramSemanticGraph) =
    match parentId with
    | None -> graph
    | Some pid ->
        match Map.tryFind pid.Value graph.Nodes with
        | Some parentNode ->
            let updatedParent = { parentNode with Children = childId :: parentNode.Children }
            
            // Create ChildOf edge: parent -> child
            let childOfEdge = {
                Source = pid
                Target = childId
                Kind = ChildOf
            }
            
            { graph with 
                Nodes = Map.add pid.Value updatedParent graph.Nodes
                Edges = childOfEdge :: graph.Edges  // ✅ ADD THE EDGE
            }
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
        printfn "[DEBUG-PATTERN] Processing pattern '%s' at %s:%d:%d" 
            ident.idText fileName range.Start.Line range.Start.Column
        printfn "[DEBUG-PATTERN] Future node ID: %s" futureNodeId.Value
        printfn "[DEBUG-PATTERN] Received parentId: %A" 
            (parentId |> Option.map (fun id -> id.Value) |> Option.defaultValue "None")
        
        // Check for self-reference before creating
        match parentId with
        | Some pid when pid.Value = futureNodeId.Value ->
            printfn "[ERROR] SELF-REFERENCE DETECTED!"
            printfn "[ERROR] Pattern '%s' would have itself as parent: %s" ident.idText pid.Value
            failwith "Pattern node assigned itself as parent"
        | _ -> ()
        
        let symbol = tryCorrelateSymbol range fileName context.CorrelationContext
        let patNode = createNode (sprintf "Pattern:Named:%s" ident.idText) range fileName symbol parentId
        
        printfn "[DEBUG-PATTERN] Created node: %s with parent: %A" 
            patNode.Id.Value (patNode.ParentId |> Option.map (fun id -> id.Value))
        
        let graph' = { graph with Nodes = Map.add patNode.Id.Value patNode graph.Nodes }
        let graph'' = addChildToParent patNode.Id parentId graph'
        
        // Add to symbol table if correlated
        match symbol with
        | Some sym ->
            { graph'' with SymbolTable = Map.add ident.idText sym graph''.SymbolTable }
        | None -> graph''
        
    | SynPat.Wild range ->
        let futureNodeId = NodeId.FromRange(fileName, range)
        printfn "[DEBUG-PATTERN] Processing Wild pattern at %s:%d:%d" 
            fileName range.Start.Line range.Start.Column
        printfn "[DEBUG-PATTERN] Future node ID: %s" futureNodeId.Value
        printfn "[DEBUG-PATTERN] Received parentId: %A" 
            (parentId |> Option.map (fun id -> id.Value) |> Option.defaultValue "None")
            
        let wildNode = createNode "Pattern:Wild" range fileName None parentId
        let graph' = { graph with Nodes = Map.add wildNode.Id.Value wildNode graph.Nodes }
        addChildToParent wildNode.Id parentId graph'
        
    | SynPat.Typed(innerPat, _, range) ->
        let futureNodeId = NodeId.FromRange(fileName, range)
        printfn "[DEBUG-PATTERN] Processing Typed pattern at %s:%d:%d" 
            fileName range.Start.Line range.Start.Column
        printfn "[DEBUG-PATTERN] Future node ID: %s" futureNodeId.Value
        printfn "[DEBUG-PATTERN] Received parentId: %A" 
            (parentId |> Option.map (fun id -> id.Value) |> Option.defaultValue "None")
            
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
                        Target = target.Id  // ✅ Use the actual node's ID
                        Kind = FunctionCall
                    }
                    { graph'''' with Edges = edge :: graph''''.Edges }
                | None -> graph''''
            | None -> graph''''
        | _ -> 
            // For other function expression types, just process without creating call edge
            graph''''
        
    | SynExpr.LetOrUse(_, _, bindings, body, range, _) ->
        let letNode = createNode "LetOrUse" range fileName None parentId
        let graph' = { graph with Nodes = Map.add letNode.Id.Value letNode graph.Nodes }
        let graph'' = addChildToParent letNode.Id parentId graph'
        
        // Process bindings
        let graph''' = 
            bindings |> List.fold (fun g binding ->
                processBinding binding (Some letNode.Id) fileName context g
            ) graph''
            
        // Process body
        processExpression body (Some letNode.Id) fileName context graph'''
        
    | SynExpr.Sequential(_, _, expr1, expr2, range, _) ->
        let seqNode = createNode "Sequential" range fileName None parentId
        let graph' = { graph with Nodes = Map.add seqNode.Id.Value seqNode graph.Nodes }
        let graph'' = addChildToParent seqNode.Id parentId graph'
        
        let graph''' = processExpression expr1 (Some seqNode.Id) fileName context graph''
        processExpression expr2 (Some seqNode.Id) fileName context graph'''
        
    | SynExpr.Lambda(_, _, args, body, _, range, _) ->
        let lambdaNode = createNode "Lambda" range fileName None parentId
        let graph' = { graph with Nodes = Map.add lambdaNode.Id.Value lambdaNode graph.Nodes }
        let graph'' = addChildToParent lambdaNode.Id parentId graph'
        
        let graph''' = processSimplePats args (Some lambdaNode.Id) fileName context graph''
        processExpression body (Some lambdaNode.Id) fileName context graph'''
        
    | SynExpr.Match(_, matchExpr, clauses, range, _) ->
        let matchNode = createNode "Match" range fileName None parentId
        let graph' = { graph with Nodes = Map.add matchNode.Id.Value matchNode graph.Nodes }
        let graph'' = addChildToParent matchNode.Id parentId graph'
        
        let graph''' = processExpression matchExpr (Some matchNode.Id) fileName context graph''
        
        clauses |> List.fold (fun g clause ->
            processMatchClause clause (Some matchNode.Id) fileName context g
        ) graph'''
        
    | _ -> graph

/// Process simple patterns (lambda args)
and private processSimplePats pats parentId fileName context graph =
    let (SynSimplePats.SimplePats(patterns, _, _)) = pats
    patterns |> List.fold (fun g pat ->
        processSimplePat pat parentId fileName context g
    ) graph

and private processSimplePat pat parentId fileName context graph =
    match pat with
    | SynSimplePat.Id(ident, _, _, _, _, range) ->
        let patNode = createNode (sprintf "SimplePat:%s" ident.idText) range fileName None parentId
        let graph' = { graph with Nodes = Map.add patNode.Id.Value patNode graph.Nodes }
        addChildToParent patNode.Id parentId graph'
    | _ -> graph

/// Process match clause
and private processMatchClause clause parentId fileName context graph =
    let (SynMatchClause(pat, whenExpr, resultExpr, range, _, _)) = clause
    let clauseNode = createNode "MatchClause" range fileName None parentId
    let graph' = { graph with Nodes = Map.add clauseNode.Id.Value clauseNode graph.Nodes }
    let graph'' = addChildToParent clauseNode.Id parentId graph'
    
    let graph''' = processPattern pat (Some clauseNode.Id) fileName context graph''
    
    let graph'''' = 
        match whenExpr with
        | Some expr -> processExpression expr (Some clauseNode.Id) fileName context graph'''
        | None -> graph'''
        
    processExpression resultExpr (Some clauseNode.Id) fileName context graph''''

/// Process module declaration
let rec private processModuleDecl decl parentId fileName context graph =
    match decl with
    | SynModuleDecl.Let(_, bindings, range) ->
        let letDeclNode = createNode "LetDeclaration" range fileName None parentId
        let graph' = { graph with Nodes = Map.add letDeclNode.Id.Value letDeclNode graph.Nodes }
        let graph'' = addChildToParent letDeclNode.Id parentId graph'
        
        bindings |> List.fold (fun g binding ->
            processBinding binding (Some letDeclNode.Id) fileName context g
        ) graph''
        
    | SynModuleDecl.Types(typeDefs, range) ->
        let typesDeclNode = createNode "TypesDeclaration" range fileName None parentId
        let graph' = { graph with Nodes = Map.add typesDeclNode.Id.Value typesDeclNode graph.Nodes }
        let graph'' = addChildToParent typesDeclNode.Id parentId graph'
        
        typeDefs |> List.fold (fun g typeDef ->
            processTypeDef typeDef (Some typesDeclNode.Id) fileName context g
        ) graph''
        
    | SynModuleDecl.NestedModule(componentInfo, _, decls, _, range, _) ->
        let (SynComponentInfo(_, _, _, longId, _, _, _, _)) = componentInfo
        let moduleName = longId |> List.map (fun id -> id.idText) |> String.concat "."
        let symbol = tryCorrelateSymbol range fileName context.CorrelationContext
        let moduleNode = createNode (sprintf "NestedModule:%s" moduleName) range fileName symbol parentId
        
        let graph' = { graph with Nodes = Map.add moduleNode.Id.Value moduleNode graph.Nodes }
        let graph'' = addChildToParent moduleNode.Id parentId graph'
        
        decls |> List.fold (fun g decl ->
            processModuleDecl decl (Some moduleNode.Id) fileName context g
        ) graph''
        
    | SynModuleDecl.Open(target, range) ->
        let openNode = createNode "Open" range fileName None parentId
        let graph' = { graph with Nodes = Map.add openNode.Id.Value openNode graph.Nodes }
        addChildToParent openNode.Id parentId graph'
        
    | _ -> graph

/// Process type definition
and private processTypeDef typeDef parentId fileName context graph =
    let (SynTypeDefn(componentInfo, _, members, _, range, _)) = typeDef
    let (SynComponentInfo(_, _, _, longId, _, _, _, _)) = componentInfo
    let typeName = longId |> List.map (fun id -> id.idText) |> String.concat "."
    let symbol = tryCorrelateSymbol range fileName context.CorrelationContext
    let typeNode = createNode (sprintf "TypeDef:%s" typeName) range fileName symbol parentId
    
    let graph' = { graph with Nodes = Map.add typeNode.Id.Value typeNode graph.Nodes }
    let graph'' = addChildToParent typeNode.Id parentId graph'
    
    members |> List.fold (fun g memberDefn ->
        processMemberDefn memberDefn (Some typeNode.Id) fileName context g
    ) graph''

/// Process member definition
and private processMemberDefn memberDefn parentId fileName context graph =
    match memberDefn with
    | SynMemberDefn.Member(binding, _) ->
        processBinding binding parentId fileName context graph
    | _ -> graph

/// Process implementation file
and private processImplFile implFile context =
    let (ParsedImplFileInput(fileName, _, _, _, _, modules, _, _, _)) = implFile
    
    let initialGraph = {
        Nodes = Map.empty
        Edges = []
        SymbolTable = Map.empty
        EntryPoints = []
        SourceFiles = context.SourceFiles
        CompilationOrder = []
    }
    
    modules |> List.fold (fun graph modOrNs ->
        let (SynModuleOrNamespace(longId, _, _, decls, _, _, _, range, _)) = modOrNs
        let modulePath = longId |> List.map (fun id -> id.idText) |> String.concat "."
        let symbol = tryCorrelateSymbol range fileName context.CorrelationContext
        let moduleNode = createNode (sprintf "Module:%s" modulePath) range fileName symbol None
        
        let graph' = { graph with Nodes = Map.add moduleNode.Id.Value moduleNode graph.Nodes }
        
        // Add to symbol table if correlated
        let graph'' = 
            match symbol with
            | Some sym -> 
                { graph' with SymbolTable = Map.add sym.DisplayName sym graph'.SymbolTable }
            | None -> graph'
        
        // Check if this is an entry point
        let graph''' = 
            decls |> List.fold (fun g decl ->
                match decl with
                | SynModuleDecl.Let(_, bindings, _) ->
                    bindings |> List.fold (fun g2 binding ->
                        let (SynBinding(_, _, _, _, attrs, _, _, pat, _, _, _, _, _)) = binding
                        let hasEntryPoint = 
                            attrs |> List.exists (fun attrList ->
                                attrList.Attributes |> List.exists (fun attr ->
                                    match attr.TypeName with
                                    | SynLongIdent(ids, _, _) ->
                                        ids |> List.exists (fun id -> 
                                            id.idText = "EntryPoint" || 
                                            id.idText = "EntryPointAttribute"
                                        )
                                )
                            )
                        
                        if hasEntryPoint then
                            // FIXED: Actually add the entry point to the graph!
                            let funcName = 
                                match pat with
                                | SynPat.Named(SynIdent(ident, _), _, _, _) -> ident.idText
                                | _ -> "unknown"
                            
                            printfn "[PSG Builder] Found EntryPoint: %s at %s:%d:%d" 
                                funcName fileName range.Start.Line range.Start.Column
                            
                            // Create entry point identifier
                            let entryPointId = 
                                let bindingSymbol = tryCorrelateSymbol range fileName context.CorrelationContext
                                match bindingSymbol with
                                | Some sym -> SymbolNode(sym.GetHashCode(), sym.FullName)
                                | None -> RangeNode(fileName, range.Start.Line, range.Start.Column, range.End.Line, range.End.Column)
                            
                            // Add to entry points list
                            { g2 with EntryPoints = entryPointId :: g2.EntryPoints }
                        else 
                            g2
                    ) g
                | _ -> g
            ) graph''
        
        // Process all declarations
        decls |> List.fold (fun g decl ->
            processModuleDecl decl (Some moduleNode.Id) fileName context g
        ) graph'''
    ) initialGraph

/// Build complete PSG from project results
let buildProgramSemanticGraph 
    (checkResults: FSharpCheckProjectResults) 
    (parseResults: FSharpParseFileResults[]) : ProgramSemanticGraph =
    
    printfn "[PSG] Building Program Semantic Graph"
    printfn "[PSG] Parse results: %d files" parseResults.Length
    
    // Create enhanced correlation context
    let correlationContext = createContext checkResults
    
    printfn "[PSG] Symbol uses found: %d" correlationContext.SymbolUses.Length
    
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
                Some (processImplFile implFile context)
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
    
    // Set compilation order based on file order
    let finalGraph = 
        { mergedGraph with 
            CompilationOrder = parseResults |> Array.map (fun pr -> pr.FileName) |> List.ofArray 
        }
    
    printfn "[PSG] Complete: %d nodes, %d edges, %d entry points"
        finalGraph.Nodes.Count 
        finalGraph.Edges.Length 
        finalGraph.EntryPoints.Length
        
    finalGraph