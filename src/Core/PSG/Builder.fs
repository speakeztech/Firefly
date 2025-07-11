module Core.PSG.Builder

open System.IO
open FSharp.Compiler.Symbols
open FSharp.Compiler.Text
open FSharp.Compiler.Syntax
open FSharp.Compiler.CodeAnalysis
open Core.PSG.Types
open Core.PSG.Correlation
open Core.PSG.SymbolAnalysis
open Core.FCS.Helpers

/// Build context for PSG construction
type BuildContext = {
    CheckResults: FSharpCheckProjectResults
    ParseResults: FSharpParseFileResults[]
    CorrelationContext: CorrelationContext
    SourceFiles: Map<string, string>
}

/// Create a PSG node from syntax element
let createNode 
    (syntaxKind: string) 
    (range: range) 
    (fileName: string) 
    (symbol: FSharpSymbol option) 
    (parentId: NodeId option) =
    
    let nodeId = 
        match symbol with
        | Some sym -> NodeId.FromSymbol(sym)
        | None -> NodeId.FromRange(fileName, range)
    
    {
        Id = nodeId
        SyntaxKind = syntaxKind
        Symbol = symbol
        Range = range
        SourceFile = fileName
        ParentId = parentId
        Children = []
    }

/// Process a module declaration
let rec processModuleDecl 
    (decl: SynModuleDecl) 
    (parentId: NodeId option) 
    (fileName: string) 
    (context: BuildContext) 
    (graph: ProgramSemanticGraph) =
    
    match decl with
    | SynModuleDecl.Let(isRec, bindings, range) ->
        let declNode = createNode "LetDeclaration" range fileName None parentId
        let graph' = { graph with Nodes = Map.add declNode.Id.Value declNode graph.Nodes }
        
        bindings 
        |> List.fold (fun g binding -> 
            processBinding binding (Some declNode.Id) fileName context g
        ) graph'
        
    | SynModuleDecl.Types(typeDefs, range) ->
        let declNode = createNode "TypeDeclaration" range fileName None parentId
        let graph' = { graph with Nodes = Map.add declNode.Id.Value declNode graph.Nodes }
        
        typeDefs 
        |> List.fold (fun g typeDef -> 
            processTypeDef typeDef (Some declNode.Id) fileName context g
        ) graph'
        
    | SynModuleDecl.NestedModule(componentInfo, _, decls, _, range, _) ->
        let symbol = tryCorrelateSymbol range fileName context.CorrelationContext
        let moduleNode = createNode "NestedModule" range fileName symbol parentId
        let graph' = { graph with Nodes = Map.add moduleNode.Id.Value moduleNode graph.Nodes }
        
        decls 
        |> List.fold (fun g decl -> 
            processModuleDecl decl (Some moduleNode.Id) fileName context g
        ) graph'
        
    | SynModuleDecl.Open(_, range) ->
        let openNode = createNode "OpenDeclaration" range fileName None parentId
        { graph with Nodes = Map.add openNode.Id.Value openNode graph.Nodes }
        
    | _ -> graph

/// Process a binding
and processBinding 
    (binding: SynBinding) 
    (parentId: NodeId option) 
    (fileName: string) 
    (context: BuildContext) 
    (graph: ProgramSemanticGraph) =
    
    let (SynBinding(_, _, _, _, _, _, valData, headPat, _, expr, bindingRange, _, _)) = binding
    
    // Get symbol from correlation
    let symbol = tryCorrelateSymbol bindingRange fileName context.CorrelationContext
    let bindingNode = createNode "Binding" bindingRange fileName symbol parentId
    
    // Add to graph
    let graph' = { graph with Nodes = Map.add bindingNode.Id.Value bindingNode graph.Nodes }
    
    // Add symbol to symbol table if present
    let graph'' = 
        match symbol with
        | Some sym -> 
            { graph' with SymbolTable = Map.add (sym.DisplayName) sym graph'.SymbolTable }
        | None -> graph'
    
    // Process expression
    processExpression expr (Some bindingNode.Id) fileName context graph''

/// Find the nearest function binding in the parent chain
and findNearestFunctionParent (parentId: NodeId option) (graph: ProgramSemanticGraph) : NodeId option =
    match parentId with
    | None -> None
    | Some pid ->
        match Map.tryFind pid.Value graph.Nodes with
        | Some node when node.Symbol.IsSome && isFunction node.Symbol.Value -> Some pid
        | Some node when node.SyntaxKind = "Binding" && node.Symbol.IsSome -> Some pid  // Any binding
        | Some node -> findNearestFunctionParent node.ParentId graph
        | None -> None

/// Enhanced processExpression with smart parent linking
and processExpression 
    (expr: SynExpr) 
    (parentId: NodeId option) 
    (fileName: string) 
    (context: BuildContext) 
    (graph: ProgramSemanticGraph) =
    
    let symbol = tryCorrelateSymbol expr.Range fileName context.CorrelationContext
    let exprKind = 
        match expr with
        | SynExpr.App _ -> "Application"
        | SynExpr.Lambda _ -> "Lambda"
        | SynExpr.LetOrUse _ -> "LetOrUse"
        | SynExpr.Match _ -> "Match"
        | SynExpr.Ident _ -> "Identifier"
        | SynExpr.LongIdent _ -> "LongIdentifier"
        | SynExpr.Const _ -> "Constant"
        | SynExpr.Sequential _ -> "Sequential"
        | SynExpr.IfThenElse _ -> "IfThenElse"
        | SynExpr.TypeApp _ -> "TypeApplication"
        | SynExpr.Paren _ -> "Parenthesized"
        | SynExpr.Tuple _ -> "Tuple"
        | _ -> "Expression"
    
    let exprNode = createNode exprKind expr.Range fileName symbol parentId
    let graph' = { graph with Nodes = Map.add exprNode.Id.Value exprNode graph.Nodes }

    let graph'' = 
        match parentId with
        | Some pid ->
            let edge : PSGEdge = { Source = pid; Target = exprNode.Id; Kind = ChildOf }
            printfn "[PSG Builder] Created ChildOf edge: %s -> %s (effective)" pid.Value exprNode.Id.Value
            { graph' with Edges = edge :: graph'.Edges }
        | None -> 
            printfn "[PSG Builder] No parent for node: %s" exprNode.Id.Value
            graph'

    // Process subexpressions and create edges
    match expr with
    | SynExpr.App(_, _, funcExpr, argExpr, _) ->
        // Process function and argument expressions - USE GRAPH'' WITH CHILOF EDGE
        let graph''' = processExpression funcExpr (Some exprNode.Id) fileName context graph''
        let graph'''' = processExpression argExpr (Some exprNode.Id) fileName context graph'''
        
        // Create direct edge from Application to target function for reachability
        match funcExpr with
        | SynExpr.Ident ident ->
            let identName = ident.idText
            printfn "[PSG Builder] Processing App: %s" identName
            // Try correlation context first, then local table
            let symbolOpt = 
                tryCorrelateSymbol funcExpr.Range fileName context.CorrelationContext
                |> Option.orElse (Map.tryFind identName graph''''.SymbolTable)
            
            match symbolOpt with
            | Some targetSymbol ->
                printfn "[PSG Builder] ✅ Created CallsFunction edge: %s -> %s" 
                    exprNode.Id.Value targetSymbol.FullName
                let edge : PSGEdge = { Source = exprNode.Id; Target = NodeId.FromSymbol(targetSymbol); Kind = CallsFunction }
                { graph'''' with Edges = edge :: graph''''.Edges }
            | None -> 
                printfn "[PSG Builder] ❌ Symbol not found: %s" identName
                graph''''
            
        | SynExpr.LongIdent(_, lid, _, _) ->
            let fullName = lid.LongIdent |> List.map (fun id -> id.idText) |> String.concat "."
            match Map.tryFind fullName graph''''.SymbolTable with
            | Some targetSymbol ->
                let edge : PSGEdge = {
                    Source = exprNode.Id
                    Target = NodeId.FromSymbol(targetSymbol)
                    Kind = CallsFunction
                }
                { graph'''' with Edges = edge :: graph''''.Edges }
            | None -> 
                // Try to find in correlation context for external functions
                let correlatedSymbol = tryCorrelateSymbol funcExpr.Range fileName context.CorrelationContext
                match correlatedSymbol with
                | Some targetSymbol ->
                    let edge : PSGEdge = {
                        Source = exprNode.Id
                        Target = NodeId.FromSymbol(targetSymbol)
                        Kind = CallsFunction
                    }
                    { graph'''' with Edges = edge :: graph''''.Edges }
                | None -> graph''''
        | _ -> graph''''
        
    | SynExpr.Lambda(_, _, _, bodyExpr, _, _, _) ->
        processExpression bodyExpr (Some exprNode.Id) fileName context graph''

    | SynExpr.LetOrUse(_, _, bindings, bodyExpr, _, _) ->
        let graph''' = 
            bindings 
            |> List.fold (fun g b -> processBinding b (Some exprNode.Id) fileName context g) graph''
        processExpression bodyExpr (Some exprNode.Id) fileName context graph'''
        
    | SynExpr.Match(_, matchExpr, clauses, _, _) ->
        let graph''' = processExpression matchExpr (Some exprNode.Id) fileName context graph''
        clauses
        |> List.fold (fun g clause ->
            let (SynMatchClause(pat, whenExpr, resultExpr, _, _, _)) = clause
            let g' = processPattern pat (Some exprNode.Id) fileName context g
            let g'' = 
                match whenExpr with
                | Some e -> processExpression e (Some exprNode.Id) fileName context g'
                | None -> g'
            processExpression resultExpr (Some exprNode.Id) fileName context g''
        ) graph'''
        
    | SynExpr.Sequential(_, _, expr1, expr2, _, _) ->
        let g' = processExpression expr1 (Some exprNode.Id) fileName context graph''
        processExpression expr2 (Some exprNode.Id) fileName context g'
        
    | SynExpr.IfThenElse(ifExpr, thenExpr, elseExpr, _, _, _, _) ->
        let g' = processExpression ifExpr (Some exprNode.Id) fileName context graph''
        let g'' = processExpression thenExpr (Some exprNode.Id) fileName context g'
        match elseExpr with
        | Some expr -> processExpression expr (Some exprNode.Id) fileName context g''
        | None -> g''
        
    | SynExpr.TypeApp(expr, _, _, _, _, _, _) ->
        processExpression expr (Some exprNode.Id) fileName context graph''
        
    | SynExpr.Paren(expr, _, _, _) ->
        processExpression expr (Some exprNode.Id) fileName context graph''
        
    | SynExpr.Tuple(_, exprs, _, _) ->
        exprs 
        |> List.fold (fun g e -> processExpression e (Some exprNode.Id) fileName context g) graph''
        
    | SynExpr.LongIdent(_, lid, _, _) ->
        let fullName = lid.LongIdent |> List.map (fun id -> id.idText) |> String.concat "."
        match Map.tryFind fullName graph''.SymbolTable with
        | Some targetSymbol ->
            let edge : PSGEdge = {
                Source = exprNode.Id
                Target = NodeId.FromSymbol(targetSymbol)
                Kind = SymRef
            }
            { graph'' with Edges = edge :: graph''.Edges }
        | None -> graph''
        
    | _ -> graph''

/// Process a pattern
and processPattern 
    (pat: SynPat) 
    (parentId: NodeId option) 
    (fileName: string) 
    (context: BuildContext) 
    (graph: ProgramSemanticGraph) =
    
    let symbol = tryCorrelateSymbol pat.Range fileName context.CorrelationContext
    let patNode = createNode "Pattern" pat.Range fileName symbol parentId
    let graph' = { graph with Nodes = Map.add patNode.Id.Value patNode graph.Nodes }
    
    match pat with
    | SynPat.Named(synIdent, _, _, _) ->
        let (SynIdent(ident, _)) = synIdent
        match symbol with
        | Some sym ->
            { graph' with SymbolTable = Map.add ident.idText sym graph'.SymbolTable }
        | None -> graph'
    | _ -> graph'

/// Process a type definition
and processTypeDef 
    (typeDef: SynTypeDefn) 
    (parentId: NodeId option) 
    (fileName: string) 
    (context: BuildContext) 
    (graph: ProgramSemanticGraph) =
    
    let (SynTypeDefn(componentInfo, typeRepr, members, _, range, _)) = typeDef
    let symbol = tryCorrelateSymbol range fileName context.CorrelationContext
    let typeNode = createNode "TypeDefinition" range fileName symbol parentId
    
    let graph' = { graph with Nodes = Map.add typeNode.Id.Value typeNode graph.Nodes }
    
    members 
    |> List.fold (fun g member' -> 
        processMember member' (Some typeNode.Id) fileName context g
    ) graph'

/// Process a member definition
and processMember 
    (memberDefn: SynMemberDefn) 
    (parentId: NodeId option) 
    (fileName: string) 
    (context: BuildContext) 
    (graph: ProgramSemanticGraph) =
    
    match memberDefn with
    | SynMemberDefn.Member(binding, range) ->
        processBinding binding parentId fileName context graph
    | _ -> graph

/// Build PSG from parsed implementation file
let buildFromImplementationFile 
    (implFile: ParsedImplFileInput) 
    (context: BuildContext) =
    
    let (ParsedImplFileInput(fileName, _, _, _, _, modules, _, _, _)) = implFile
    
    let initialGraph = {
        Nodes = Map.empty
        Edges = []
        SymbolTable = Map.empty
        EntryPoints = []
        SourceFiles = context.SourceFiles
        CompilationOrder = []
    }
    
    let finalGraph = 
        modules 
        |> List.fold (fun graph moduleOrNs ->
            let (SynModuleOrNamespace(_, _, _, decls, _, _, _, range, _)) = moduleOrNs
            let symbol = tryCorrelateSymbol range fileName context.CorrelationContext
            let moduleNode = createNode "Module" range fileName symbol None
            let graph' = { graph with Nodes = Map.add moduleNode.Id.Value moduleNode graph.Nodes }
            
            let graph'' = 
                match symbol with
                | Some sym -> 
                    { graph' with SymbolTable = Map.add sym.DisplayName sym graph'.SymbolTable }
                | None -> graph'
            
            decls 
            |> List.fold (fun g decl -> 
                processModuleDecl decl (Some moduleNode.Id) fileName context g
            ) graph''
        ) initialGraph
        
    printfn "[PSG Debug] Built graph for %s: %d nodes" 
        (Path.GetFileName fileName) finalGraph.Nodes.Count
        
    finalGraph

/// Build complete PSG from project results
let buildProgramSemanticGraph 
    (checkResults: FSharpCheckProjectResults) 
    (parseResults: FSharpParseFileResults[]) =
    
    printfn "[PSG Debug] CheckResults has errors: %b" checkResults.HasCriticalErrors
    printfn "[PSG Debug] AssemblyContents files: %d" checkResults.AssemblyContents.ImplementationFiles.Length
    
    let correlationContext = createContext checkResults
    
    printfn "[PSG Debug] Symbol uses found: %d across %d files" 
        correlationContext.SymbolUses.Length
        correlationContext.FileIndex.Count
    
    let sourceFiles = 
        parseResults
        |> Array.map (fun pr -> 
            let fileName = pr.FileName
            let content = 
                try File.ReadAllText(fileName) 
                with _ -> ""
            fileName, content
        )
        |> Map.ofArray
    
    let buildContext = {
        CheckResults = checkResults
        ParseResults = parseResults
        CorrelationContext = correlationContext
        SourceFiles = sourceFiles
    }
    
    let graphs = 
        parseResults
        |> Array.choose (fun pr ->
            match pr.ParseTree with
            | ParsedInput.ImplFile implFile ->
                Some (buildFromImplementationFile implFile buildContext)
            | _ -> None
        )
    
    printfn "[PSG Debug] Built %d file graphs" graphs.Length
    
    let mergedGraph = 
        graphs 
        |> Array.fold (fun acc g ->
            {
                Nodes = Map.fold (fun m k v -> Map.add k v m) acc.Nodes g.Nodes
                Edges = g.Edges @ acc.Edges
                SymbolTable = Map.fold (fun m k v -> Map.add k v m) acc.SymbolTable g.SymbolTable
                EntryPoints = g.EntryPoints @ acc.EntryPoints
                SourceFiles = acc.SourceFiles
                CompilationOrder = parseResults |> Array.map (fun pr -> pr.FileName) |> Array.toList
            }
        ) {
            Nodes = Map.empty
            Edges = []
            SymbolTable = Map.empty
            EntryPoints = []
            SourceFiles = sourceFiles
            CompilationOrder = []
        }
    
    let fullSymbolTable =
        correlationContext.SymbolUses
        |> Array.filter (fun su -> su.IsFromDefinition)
        |> Array.fold (fun table su ->
            let key = su.Symbol.DisplayName
            Map.add key su.Symbol table
        ) mergedGraph.SymbolTable
    
    printfn "[PSG Debug] Symbol table size: %d (was %d)" fullSymbolTable.Count mergedGraph.SymbolTable.Count
    
    let entryPoints = 
        getEntryPointSymbols correlationContext.SymbolUses
        |> Array.map NodeId.FromSymbol
        |> Array.toList
    
    let referenceEdges =
        correlationContext.SymbolUses
        |> Array.filter (fun su -> su.IsFromUse)
        |> Array.choose (fun useSym ->
            let defOpt = 
                correlationContext.SymbolUses
                |> Array.tryFind (fun def -> 
                    def.IsFromDefinition && 
                    def.Symbol = useSym.Symbol
                )
            
            match defOpt with
            | Some def when useSym.Range <> def.Range ->
                let edge : PSGEdge = {
                    Source = NodeId.FromRange(useSym.Range.FileName, useSym.Range)
                    Target = NodeId.FromSymbol(def.Symbol)
                    Kind = SymRef
                }
                Some edge
            | _ -> None
        )
        |> Array.toList
    
    printfn "[PSG Debug] Created %d reference edges" referenceEdges.Length
    
    let finalGraph = { 
        mergedGraph with 
            EntryPoints = entryPoints
            SymbolTable = fullSymbolTable
            Edges = referenceEdges @ mergedGraph.Edges
    }
    
    // Add validation
    let childOfEdges = finalGraph.Edges |> List.filter (fun e -> e.Kind = ChildOf)
    printfn "[PSG Debug] Final graph ChildOf edges: %d" childOfEdges.Length
    
    Success finalGraph

/// Validate PSG structure
let validateGraph (graph: ProgramSemanticGraph) =
    let errors = ResizeArray<PSGError>()
    
    graph.Edges |> List.iter (fun edge ->
        if not (Map.containsKey edge.Source.Value graph.Nodes) then
            errors.Add {
                Message = sprintf "Edge source node not found: %s" edge.Source.Value
                Location = None
                ErrorKind = InvalidNode
            }
        if not (Map.containsKey edge.Target.Value graph.Nodes) then
            errors.Add {
                Message = sprintf "Edge target node not found: %s" edge.Target.Value
                Location = None
                ErrorKind = InvalidNode
            }
    )
    
    if graph.EntryPoints.IsEmpty then
        errors.Add {
            Message = "No entry points found in PSG"
            Location = None
            ErrorKind = BuilderError
        }
    
    if errors.Count > 0 then
        Failure (errors.ToArray() |> Array.toList)
    else
        Success graph