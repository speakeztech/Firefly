module Core.PSG.Builder

open System
open System.IO
open FSharp.Compiler.Symbols
open FSharp.Compiler.Text
open FSharp.Compiler.Syntax
open FSharp.Compiler.CodeAnalysis
open Core.PSG.Types
open Core.PSG.Correlation
open Core.PSG.SymbolAnalysis
open Core.FCS.ProjectContext

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

/// Process an expression
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
        | SynExpr.Const _ -> "Constant"
        | _ -> "Expression"
    
    let exprNode = createNode exprKind expr.Range fileName symbol parentId
    let graph' = { graph with Nodes = Map.add exprNode.Id.Value exprNode graph.Nodes }
    
    // Process subexpressions and create edges
    match expr with
    | SynExpr.App(_, _, funcExpr, argExpr, _) ->
        let graph'' = processExpression funcExpr (Some exprNode.Id) fileName context graph'
        processExpression argExpr (Some exprNode.Id) fileName context graph''
        
    | SynExpr.Lambda(_, _, _, bodyExpr, _, _, _) ->
        processExpression bodyExpr (Some exprNode.Id) fileName context graph'
        
    | SynExpr.LetOrUse(_, _, bindings, bodyExpr, _, _) ->
        let graph'' = 
            bindings 
            |> List.fold (fun g b -> processBinding b (Some exprNode.Id) fileName context g) graph'
        processExpression bodyExpr (Some exprNode.Id) fileName context graph''
        
    | SynExpr.Ident ident ->
        // Create reference edge if we can resolve the symbol
        match symbol with
        | Some sym ->
            let edge = {
                Source = exprNode.Id
                Target = NodeId.FromSymbol(sym)
                Kind = EdgeKind.References
            }
            { graph' with Edges = edge :: graph'.Edges }
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
    
    // Process members
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
    
    // Initialize empty graph
    let initialGraph = {
        Nodes = Map.empty
        Edges = []
        SymbolTable = Map.empty
        EntryPoints = []
        SourceFiles = context.SourceFiles
        CompilationOrder = []
    }
    
    // Process all modules
    modules 
    |> List.fold (fun graph moduleOrNs ->
        let (SynModuleOrNamespace(_, _, _, decls, _, _, _, range, _)) = moduleOrNs
        let symbol = tryCorrelateSymbol range fileName context.CorrelationContext
        let moduleNode = createNode "Module" range fileName symbol None
        let graph' = { graph with Nodes = Map.add moduleNode.Id.Value moduleNode graph.Nodes }
        
        decls 
        |> List.fold (fun g decl -> 
            processModuleDecl decl (Some moduleNode.Id) fileName context g
        ) graph'
    ) initialGraph

/// Build complete PSG from project results
let buildProgramSemanticGraph 
    (checkResults: FSharpCheckProjectResults) 
    (parseResults: FSharpParseFileResults[]) =
    
    // Create correlation context
    let correlationContext = createContext checkResults
    
    // Load source files
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
    
    // Create build context
    let buildContext = {
        CheckResults = checkResults
        ParseResults = parseResults
        CorrelationContext = correlationContext
        SourceFiles = sourceFiles
    }
    
    // Build PSG for each file and merge
    let graphs = 
        parseResults
        |> Array.choose (fun pr ->
            match pr.ParseTree with
            | ParsedInput.ImplFile implFile ->
                Some (buildFromImplementationFile implFile buildContext)
            | _ -> None
        )
    
    // Merge all graphs
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
    
    // Find entry points
    let entryPoints = 
        getEntryPointSymbols correlationContext.SymbolUses
        |> Array.map NodeId.FromSymbol
        |> Array.toList
    
    Success { mergedGraph with EntryPoints = entryPoints }

/// Validate PSG structure
let validateGraph (graph: ProgramSemanticGraph) =
    let errors = ResizeArray<PSGError>()
    
    // Check all edges reference valid nodes
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
    
    // Check entry points exist
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