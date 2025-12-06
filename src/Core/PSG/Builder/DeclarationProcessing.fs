/// Declaration processing for PSG construction (types, members, modules)
module Core.PSG.Construction.DeclarationProcessing

open FSharp.Compiler.Text
open FSharp.Compiler.Syntax
open FSharp.Compiler.Symbols
open Core.PSG.Types
open Core.PSG.Construction.Types
open Core.PSG.Construction.SymbolCorrelation
open Core.PSG.Construction.BindingProcessing

/// Process a type definition (type Foo = ...)
let rec processTypeDefn (typeDefn: SynTypeDefn) parentId fileName (context: BuildContext) graph =
    let (SynTypeDefn(componentInfo, typeRepr, members, implicitCtor, range, trivia)) = typeDefn
    let (SynComponentInfo(attributes, typeParams, constraints, longId, xmlDoc, preferPostfix, accessibility, range2)) = componentInfo

    let typeName = longId |> List.map (fun id -> id.idText) |> String.concat "."
    let syntaxKind = sprintf "TypeDefn:%s" typeName
    let symbol = tryCorrelateSymbolWithContext range fileName syntaxKind context.CorrelationContext
    let typeNode = createNode syntaxKind range fileName symbol parentId

    let graph' = { graph with Nodes = Map.add typeNode.Id.Value typeNode graph.Nodes }
    let graph'' = addChildToParent typeNode.Id parentId graph'

    let graph''' =
        match symbol with
        | Some sym -> { graph'' with SymbolTable = Map.add sym.DisplayName sym graph''.SymbolTable }
        | None -> graph''

    // Process members (methods, properties, etc.)
    members |> List.fold (fun g memberDefn ->
        processMemberDefn memberDefn (Some typeNode.Id) fileName context g
    ) graph'''

/// Process a member definition (member this.Foo = ...)
and processMemberDefn (memberDefn: SynMemberDefn) parentId fileName (context: BuildContext) graph =
    match memberDefn with
    | SynMemberDefn.Member(binding, range) ->
        processBinding binding parentId fileName context graph

    | SynMemberDefn.LetBindings(bindings, _, _, range) ->
        bindings |> List.fold (fun g binding ->
            processBinding binding parentId fileName context g
        ) graph

    // For now, skip other member types (abstract, interface, etc.) with a node marker
    | other ->
        let memberTypeName = other.GetType().Name
        let markerNode = createNode (sprintf "MemberDefn:%s" memberTypeName) other.Range fileName None parentId
        let graph' = { graph with Nodes = Map.add markerNode.Id.Value markerNode graph.Nodes }
        addChildToParent markerNode.Id parentId graph'

/// Process a module declaration
and processModuleDecl decl parentId fileName (context: BuildContext) graph =
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
        let (SynComponentInfo(attributes, typeArgs, constraints, longId, xmlDoc, preferPostfix, accessibility, range2)) = componentInfo
        let moduleName = longId |> List.map (fun id -> id.idText) |> String.concat "."

        let syntaxKind = sprintf "NestedModule:%s" moduleName
        let symbol = tryCorrelateSymbolWithContext range fileName syntaxKind context.CorrelationContext
        let nestedModuleNode = createNode syntaxKind range fileName symbol parentId

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

    // Type declarations (type Foo = ..., type Bar = { ... }, etc.)
    | SynModuleDecl.Types(typeDefns, range) ->
        let typesNode = createNode "TypeDeclarations" range fileName None parentId
        let graph' = { graph with Nodes = Map.add typesNode.Id.Value typesNode graph.Nodes }
        let graph'' = addChildToParent typesNode.Id parentId graph'

        typeDefns |> List.fold (fun g typeDefn ->
            processTypeDefn typeDefn (Some typesNode.Id) fileName context g
        ) graph''

    // Hard stop on unhandled module declarations
    | other ->
        let declTypeName = other.GetType().Name
        failwithf "[BUILDER] ERROR: Unhandled module declaration type '%s' in file %s. PSG construction cannot continue with unknown AST nodes."
            declTypeName fileName

/// Process implementation file
let processImplFile (implFile: SynModuleOrNamespace) (context: BuildContext) graph =
    let (SynModuleOrNamespace(name, _, _, decls, _, _, _, range, _)) = implFile

    let moduleName = name |> List.map (fun i -> i.idText) |> String.concat "."
    let fileName = range.FileName

    let syntaxKind = sprintf "Module:%s" moduleName
    let symbol = tryCorrelateSymbolWithContext range fileName syntaxKind context.CorrelationContext
    let moduleNode = createNode syntaxKind range fileName symbol None

    let graph' = { graph with Nodes = Map.add moduleNode.Id.Value moduleNode graph.Nodes }

    let graph'' =
        match symbol with
        | Some sym ->
            { graph' with SymbolTable = Map.add sym.DisplayName sym graph'.SymbolTable }
        | None -> graph'

    decls
    |> List.fold (fun acc decl ->
        processModuleDecl decl (Some moduleNode.Id) fileName context acc) graph''
