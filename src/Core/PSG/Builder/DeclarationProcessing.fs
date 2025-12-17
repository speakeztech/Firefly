/// Declaration processing for PSG construction (types, members, modules)
module Core.PSG.Construction.DeclarationProcessing

open FSharp.Compiler.Text
open FSharp.Compiler.Syntax
open FSharp.Compiler.Symbols
open Core.PSG.Types
open Core.PSG.Construction.Types
open Core.PSG.Construction.SymbolCorrelation
open Core.PSG.Construction.BindingProcessing
open Core.PSG.Construction.ExpressionProcessing

/// Process a type definition (type Foo = ...)
let rec processTypeDefn (typeDefn: SynTypeDefn) parentId fileName (context: BuildContext) graph =
    let (SynTypeDefn(componentInfo, typeRepr, members, implicitCtor, range, trivia)) = typeDefn
    let (SynComponentInfo(attributes, typeParams, constraints, longId, xmlDoc, preferPostfix, accessibility, range2)) = componentInfo

    let symbol = tryCorrelateSymbolOptional range fileName "TypeDefn" context.CorrelationContext
    let typeNode = createNode (SKDecl DTypeDefn) range fileName symbol parentId

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

    // Other member types (abstract, interface, etc.) - create marker node
    | other ->
        let markerNode = createNode (SKDecl DMemberDefn) other.Range fileName None parentId
        let graph' = { graph with Nodes = Map.add markerNode.Id.Value markerNode graph.Nodes }
        addChildToParent markerNode.Id parentId graph'

/// Process a module declaration
and processModuleDecl decl parentId fileName (context: BuildContext) graph =
    match decl with
    | SynModuleDecl.Let(_, bindings, range) ->
        let letDeclNode = createNode (SKDecl DLetDecl) range fileName None parentId
        let graph' = { graph with Nodes = Map.add letDeclNode.Id.Value letDeclNode graph.Nodes }
        let graph'' = addChildToParent letDeclNode.Id parentId graph'

        bindings |> List.fold (fun g binding ->
            processBinding binding (Some letDeclNode.Id) fileName context g
        ) graph''

    | SynModuleDecl.Open(_, range) ->
        let openNode = createNode (SKDecl DOpen) range fileName None parentId
        let graph' = { graph with Nodes = Map.add openNode.Id.Value openNode graph.Nodes }
        addChildToParent openNode.Id parentId graph'

    | SynModuleDecl.NestedModule(componentInfo, _, decls, _, range, _) ->
        let (SynComponentInfo(attributes, typeArgs, constraints, longId, xmlDoc, preferPostfix, accessibility, range2)) = componentInfo

        let symbol = tryCorrelateSymbolOptional range fileName "NestedModule" context.CorrelationContext
        let nestedModuleNode = createNode (SKDecl DNestedModule) range fileName symbol parentId

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
        let typesNode = createNode (SKDecl DTypesGroup) range fileName None parentId
        let graph' = { graph with Nodes = Map.add typesNode.Id.Value typesNode graph.Nodes }
        let graph'' = addChildToParent typesNode.Id parentId graph'

        typeDefns |> List.fold (fun g typeDefn ->
            processTypeDefn typeDefn (Some typesNode.Id) fileName context g
        ) graph''

    // Hash directives (#nowarn, #light, etc.) - just skip these
    | SynModuleDecl.HashDirective(_, _) ->
        graph

    // Module-level expressions (do bindings, side effects)
    | SynModuleDecl.Expr(expr, range) ->
        let exprNode = createNode (SKDecl DModuleExpr) range fileName None parentId
        let graph' = { graph with Nodes = Map.add exprNode.Id.Value exprNode graph.Nodes }
        let graph'' = addChildToParent exprNode.Id parentId graph'
        processExpression expr (Some exprNode.Id) fileName context graph''

    // Module abbreviations (module M = Some.Other.Module)
    | SynModuleDecl.ModuleAbbrev(ident, longId, range) ->
        let abbrevNode = createNode (SKDecl DModuleAbbrev) range fileName None parentId
        let graph' = { graph with Nodes = Map.add abbrevNode.Id.Value abbrevNode graph.Nodes }
        addChildToParent abbrevNode.Id parentId graph'

    // Attributes at module level
    | SynModuleDecl.Attributes(attributes, range) ->
        let attrNode = createNode (SKDecl DAttribute) range fileName None parentId
        let graph' = { graph with Nodes = Map.add attrNode.Id.Value attrNode graph.Nodes }
        addChildToParent attrNode.Id parentId graph'

    // Exception definitions
    | SynModuleDecl.Exception(exnDefn, range) ->
        let exnNode = createNode (SKDecl DException) range fileName None parentId
        let graph' = { graph with Nodes = Map.add exnNode.Id.Value exnNode graph.Nodes }
        addChildToParent exnNode.Id parentId graph'

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

    let symbol = tryCorrelateSymbolOptional range fileName "Module" context.CorrelationContext
    let moduleNode = createNode (SKDecl DModule) range fileName symbol None

    let graph' = { graph with Nodes = Map.add moduleNode.Id.Value moduleNode graph.Nodes }

    let graph'' =
        match symbol with
        | Some sym ->
            { graph' with SymbolTable = Map.add sym.DisplayName sym graph'.SymbolTable }
        | None -> graph'

    decls
    |> List.fold (fun acc decl ->
        processModuleDecl decl (Some moduleNode.Id) fileName context acc) graph''
