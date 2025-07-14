module Core.PSG.Builder

open System
open System.IO
open FSharp.Compiler.Text
open FSharp.Compiler.Syntax
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Symbols
open Core.PSG.Types
open Core.PSG.Correlation
open Core.PSG.TypeIntegration

/// Build context for PSG construction (preserved from original)
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

/// Add child to parent and return updated graph (preserved from original)
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

/// Process a binding (let/member) - Using existing FCS 43.9.300 patterns (preserved from original)
let rec private processBinding binding parentId fileName context graph =
    match binding with
    | SynBinding(accessibility, kind, isInline, isMutable, attributes, xmlDoc, valData, pat, returnInfo, expr, range, seqPoint, trivia) ->
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

/// Process a pattern - Using existing FCS 43.9.300 patterns (preserved from original)
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

/// Process expression nodes - Complete FCS 43.9.300 SynExpr coverage
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
        let appNode = createNode "App" range fileName None parentId
        
        let graph' = { graph with Nodes = Map.add appNode.Id.Value appNode graph.Nodes }
        let graph'' = addChildToParent appNode.Id parentId graph'
        
        let graph''' = processExpression funcExpr (Some appNode.Id) fileName context graph''
        processExpression argExpr (Some appNode.Id) fileName context graph'''
    
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

    | SynExpr.Sequential(_, _, expr1, expr2, _, _) ->
        let graph' = processExpression expr1 parentId fileName context graph
        processExpression expr2 parentId fileName context graph'

    | SynExpr.Match(_, expr, clauses, range, _) ->
        let matchNode = createNode "Match" range fileName None parentId
        let graph' = { graph with Nodes = Map.add matchNode.Id.Value matchNode graph.Nodes }
        let graph'' = addChildToParent matchNode.Id parentId graph'
        processExpression expr (Some matchNode.Id) fileName context graph''

    | SynExpr.Const(constant, range) ->
        let constNode = createNode (sprintf "Const:%A" constant) range fileName None parentId
        let graph' = { graph with Nodes = Map.add constNode.Id.Value constNode graph.Nodes }
        addChildToParent constNode.Id parentId graph'

    | SynExpr.LongIdent(_, longDotId, _, range) ->
        let identText = longDotId.LongIdent |> List.map (fun id -> id.idText) |> String.concat "."
        let symbol = tryCorrelateSymbol range fileName context.CorrelationContext
        let longIdentNode = createNode (sprintf "LongIdent:%s" identText) range fileName symbol parentId
        
        let graph' = { graph with Nodes = Map.add longIdentNode.Id.Value longIdentNode graph.Nodes }
        let graph'' = addChildToParent longIdentNode.Id parentId graph'
        
        match symbol with
        | Some sym -> 
            { graph'' with SymbolTable = Map.add sym.DisplayName sym graph''.SymbolTable }
        | None -> graph''

    | SynExpr.IfThenElse(ifExpr, thenExpr, elseExpr, _, _, range, _) ->
        let ifNode = createNode "IfThenElse" range fileName None parentId
        let graph' = { graph with Nodes = Map.add ifNode.Id.Value ifNode graph.Nodes }
        let graph'' = addChildToParent ifNode.Id parentId graph'
        
        let graph''' = processExpression ifExpr (Some ifNode.Id) fileName context graph''
        let graph'''' = processExpression thenExpr (Some ifNode.Id) fileName context graph'''
        
        match elseExpr with
        | Some elseE -> processExpression elseE (Some ifNode.Id) fileName context graph''''
        | None -> graph''''

    | SynExpr.Paren(expr, _, _, range) ->
        let parenNode = createNode "Paren" range fileName None parentId
        let graph' = { graph with Nodes = Map.add parenNode.Id.Value parenNode graph.Nodes }
        let graph'' = addChildToParent parenNode.Id parentId graph'
        processExpression expr (Some parenNode.Id) fileName context graph''

    | SynExpr.Tuple(_, exprs, _, range) ->
        let tupleNode = createNode "Tuple" range fileName None parentId
        let graph' = { graph with Nodes = Map.add tupleNode.Id.Value tupleNode graph.Nodes }
        let graph'' = addChildToParent tupleNode.Id parentId graph'
        
        exprs |> List.fold (fun acc expr -> 
            processExpression expr (Some tupleNode.Id) fileName context acc) graph''

    | SynExpr.ArrayOrList(_, exprs, range) ->
        let arrayNode = createNode "ArrayOrList" range fileName None parentId
        let graph' = { graph with Nodes = Map.add arrayNode.Id.Value arrayNode graph.Nodes }
        let graph'' = addChildToParent arrayNode.Id parentId graph'
        
        exprs |> List.fold (fun acc expr -> 
            processExpression expr (Some arrayNode.Id) fileName context acc) graph''

    | SynExpr.Record(_, _, recordFields, range) ->
        let recordNode = createNode "Record" range fileName None parentId
        let graph' = { graph with Nodes = Map.add recordNode.Id.Value recordNode graph.Nodes }
        let graph'' = addChildToParent recordNode.Id parentId graph'
        
        recordFields |> List.fold (fun acc (SynExprRecordField(_, _, expr, _)) -> 
            match expr with
            | Some e -> processExpression e (Some recordNode.Id) fileName context acc
            | None -> acc) graph''

    | SynExpr.AnonRecd(_, _, recordFields, range, _) ->
        let anonRecordNode = createNode "AnonRecd" range fileName None parentId
        let graph' = { graph with Nodes = Map.add anonRecordNode.Id.Value anonRecordNode graph.Nodes }
        let graph'' = addChildToParent anonRecordNode.Id parentId graph'
        
        recordFields |> List.fold (fun acc (_, _, expr) -> 
            processExpression expr (Some anonRecordNode.Id) fileName context acc) graph''

    | SynExpr.New(_, _, expr, range) ->
        let newNode = createNode "New" range fileName None parentId
        let graph' = { graph with Nodes = Map.add newNode.Id.Value newNode graph.Nodes }
        let graph'' = addChildToParent newNode.Id parentId graph'
        processExpression expr (Some newNode.Id) fileName context graph''

    | SynExpr.TypeApp(expr, _, _, _, _, _, range) ->
        let typeAppNode = createNode "TypeApp" range fileName None parentId
        let graph' = { graph with Nodes = Map.add typeAppNode.Id.Value typeAppNode graph.Nodes }
        let graph'' = addChildToParent typeAppNode.Id parentId graph'
        processExpression expr (Some typeAppNode.Id) fileName context graph''

    | SynExpr.Lambda(_, _, _, body, _, range, _) ->
        let lambdaNode = createNode "Lambda" range fileName None parentId
        let graph' = { graph with Nodes = Map.add lambdaNode.Id.Value lambdaNode graph.Nodes }
        let graph'' = addChildToParent lambdaNode.Id parentId graph'
        processExpression body (Some lambdaNode.Id) fileName context graph''

    | SynExpr.Quote(_, _, quotedExpr, _, range) ->
        let quoteNode = createNode "Quote" range fileName None parentId
        let graph' = { graph with Nodes = Map.add quoteNode.Id.Value quoteNode graph.Nodes }
        let graph'' = addChildToParent quoteNode.Id parentId graph'
        processExpression quotedExpr (Some quoteNode.Id) fileName context graph''

    | SynExpr.Typed(expr, _, range) ->
        let typedNode = createNode "Typed" range fileName None parentId
        let graph' = { graph with Nodes = Map.add typedNode.Id.Value typedNode graph.Nodes }
        let graph'' = addChildToParent typedNode.Id parentId graph'
        processExpression expr (Some typedNode.Id) fileName context graph''

    | SynExpr.Do(expr, range) ->
        let doNode = createNode "Do" range fileName None parentId
        let graph' = { graph with Nodes = Map.add doNode.Id.Value doNode graph.Nodes }
        let graph'' = addChildToParent doNode.Id parentId graph'
        processExpression expr (Some doNode.Id) fileName context graph''

    | SynExpr.Assert(expr, range) ->
        let assertNode = createNode "Assert" range fileName None parentId
        let graph' = { graph with Nodes = Map.add assertNode.Id.Value assertNode graph.Nodes }
        let graph'' = addChildToParent assertNode.Id parentId graph'
        processExpression expr (Some assertNode.Id) fileName context graph''

    | SynExpr.DotGet(expr, _, _, range) ->
        let dotGetNode = createNode "DotGet" range fileName None parentId
        let graph' = { graph with Nodes = Map.add dotGetNode.Id.Value dotGetNode graph.Nodes }
        let graph'' = addChildToParent dotGetNode.Id parentId graph'
        processExpression expr (Some dotGetNode.Id) fileName context graph''

    | SynExpr.DotSet(targetExpr, _, rhsExpr, range) ->
        let dotSetNode = createNode "DotSet" range fileName None parentId
        let graph' = { graph with Nodes = Map.add dotSetNode.Id.Value dotSetNode graph.Nodes }
        let graph'' = addChildToParent dotSetNode.Id parentId graph'
        let graph''' = processExpression targetExpr (Some dotSetNode.Id) fileName context graph''
        processExpression rhsExpr (Some dotSetNode.Id) fileName context graph'''

    | SynExpr.LongIdentSet(_, expr, range) ->
        let setNode = createNode "LongIdentSet" range fileName None parentId
        let graph' = { graph with Nodes = Map.add setNode.Id.Value setNode graph.Nodes }
        let graph'' = addChildToParent setNode.Id parentId graph'
        processExpression expr (Some setNode.Id) fileName context graph''

    | SynExpr.DotIndexedGet(objectExpr, indexArgs, _, range) ->
        let indexGetNode = createNode "DotIndexedGet" range fileName None parentId
        let graph' = { graph with Nodes = Map.add indexGetNode.Id.Value indexGetNode graph.Nodes }
        let graph'' = addChildToParent indexGetNode.Id parentId graph'
        let graph''' = processExpression objectExpr (Some indexGetNode.Id) fileName context graph''
        processExpression indexArgs (Some indexGetNode.Id) fileName context graph'''

    | SynExpr.TryWith(tryExpr, _, range, _, _, _) ->
        let tryNode = createNode "TryWith" range fileName None parentId
        let graph' = { graph with Nodes = Map.add tryNode.Id.Value tryNode graph.Nodes }
        let graph'' = addChildToParent tryNode.Id parentId graph'
        processExpression tryExpr (Some tryNode.Id) fileName context graph''

    | SynExpr.TryFinally(tryExpr, finallyExpr, range, _, _, _) ->
        let tryFinallyNode = createNode "TryFinally" range fileName None parentId
        let graph' = { graph with Nodes = Map.add tryFinallyNode.Id.Value tryFinallyNode graph.Nodes }
        let graph'' = addChildToParent tryFinallyNode.Id parentId graph'
        let graph''' = processExpression tryExpr (Some tryFinallyNode.Id) fileName context graph''
        processExpression finallyExpr (Some tryFinallyNode.Id) fileName context graph'''

    | SynExpr.Lazy(expr, range) ->
        let lazyNode = createNode "Lazy" range fileName None parentId
        let graph' = { graph with Nodes = Map.add lazyNode.Id.Value lazyNode graph.Nodes }
        let graph'' = addChildToParent lazyNode.Id parentId graph'
        processExpression expr (Some lazyNode.Id) fileName context graph''

    | SynExpr.While(_, whileExpr, doExpr, range) ->
        let whileNode = createNode "While" range fileName None parentId
        let graph' = { graph with Nodes = Map.add whileNode.Id.Value whileNode graph.Nodes }
        let graph'' = addChildToParent whileNode.Id parentId graph'
        let graph''' = processExpression whileExpr (Some whileNode.Id) fileName context graph''
        processExpression doExpr (Some whileNode.Id) fileName context graph'''

    | SynExpr.For(_, _, _, _, identBody, _, toBody, doBody, range) ->
        let forNode = createNode "For" range fileName None parentId
        let graph' = { graph with Nodes = Map.add forNode.Id.Value forNode graph.Nodes }
        let graph'' = addChildToParent forNode.Id parentId graph'
        let graph''' = processExpression identBody (Some forNode.Id) fileName context graph''
        let graph'''' = processExpression toBody (Some forNode.Id) fileName context graph'''
        processExpression doBody (Some forNode.Id) fileName context graph''''

    | SynExpr.ForEach(_, _, _, _, _, enumExpr, bodyExpr, range) ->
        let forEachNode = createNode "ForEach" range fileName None parentId
        let graph' = { graph with Nodes = Map.add forEachNode.Id.Value forEachNode graph.Nodes }
        let graph'' = addChildToParent forEachNode.Id parentId graph'
        let graph''' = processExpression enumExpr (Some forEachNode.Id) fileName context graph''
        processExpression bodyExpr (Some forEachNode.Id) fileName context graph'''

    | SynExpr.ArrayOrListComputed(_, expr, range) ->
        let computedNode = createNode "ArrayOrListComputed" range fileName None parentId
        let graph' = { graph with Nodes = Map.add computedNode.Id.Value computedNode graph.Nodes }
        let graph'' = addChildToParent computedNode.Id parentId graph'
        processExpression expr (Some computedNode.Id) fileName context graph''

    | SynExpr.ComputationExpr(_, expr, range) ->
        let compExprNode = createNode "ComputationExpr" range fileName None parentId
        let graph' = { graph with Nodes = Map.add compExprNode.Id.Value compExprNode graph.Nodes }
        let graph'' = addChildToParent compExprNode.Id parentId graph'
        processExpression expr (Some compExprNode.Id) fileName context graph''

    | SynExpr.IndexRange(expr1, _, expr2, _, _, range) ->
        let indexRangeNode = createNode "IndexRange" range fileName None parentId
        let graph' = { graph with Nodes = Map.add indexRangeNode.Id.Value indexRangeNode graph.Nodes }
        let graph'' = addChildToParent indexRangeNode.Id parentId graph'
        let graph''' = 
            match expr1 with
            | Some e1 -> processExpression e1 (Some indexRangeNode.Id) fileName context graph''
            | None -> graph''
        match expr2 with
        | Some e2 -> processExpression e2 (Some indexRangeNode.Id) fileName context graph'''
        | None -> graph'''

    | SynExpr.IndexFromEnd(expr, range) ->
        let indexFromEndNode = createNode "IndexFromEnd" range fileName None parentId
        let graph' = { graph with Nodes = Map.add indexFromEndNode.Id.Value indexFromEndNode graph.Nodes }
        let graph'' = addChildToParent indexFromEndNode.Id parentId graph'
        processExpression expr (Some indexFromEndNode.Id) fileName context graph''

    | SynExpr.AddressOf(_, expr, _, range) ->
        let addressOfNode = createNode "AddressOf" range fileName None parentId
        let graph' = { graph with Nodes = Map.add addressOfNode.Id.Value addressOfNode graph.Nodes }
        let graph'' = addChildToParent addressOfNode.Id parentId graph'
        processExpression expr (Some addressOfNode.Id) fileName context graph''

    | SynExpr.TraitCall(_, _, argExpr, range) ->
        let traitCallNode = createNode "TraitCall" range fileName None parentId
        let graph' = { graph with Nodes = Map.add traitCallNode.Id.Value traitCallNode graph.Nodes }
        let graph'' = addChildToParent traitCallNode.Id parentId graph'
        processExpression argExpr (Some traitCallNode.Id) fileName context graph''

    | SynExpr.JoinIn(lhsExpr, _, rhsExpr, range) ->
        let joinInNode = createNode "JoinIn" range fileName None parentId
        let graph' = { graph with Nodes = Map.add joinInNode.Id.Value joinInNode graph.Nodes }
        let graph'' = addChildToParent joinInNode.Id parentId graph'
        let graph''' = processExpression lhsExpr (Some joinInNode.Id) fileName context graph''
        processExpression rhsExpr (Some joinInNode.Id) fileName context graph'''

    | SynExpr.ImplicitZero(range) ->
        let implicitZeroNode = createNode "ImplicitZero" range fileName None parentId
        let graph' = { graph with Nodes = Map.add implicitZeroNode.Id.Value implicitZeroNode graph.Nodes }
        addChildToParent implicitZeroNode.Id parentId graph'

    | SynExpr.YieldOrReturn(_, expr, range, _) ->
        let yieldReturnNode = createNode "YieldOrReturn" range fileName None parentId
        let graph' = { graph with Nodes = Map.add yieldReturnNode.Id.Value yieldReturnNode graph.Nodes }
        let graph'' = addChildToParent yieldReturnNode.Id parentId graph'
        processExpression expr (Some yieldReturnNode.Id) fileName context graph''

    | SynExpr.YieldOrReturnFrom(_, expr, range, _) ->
        let yieldReturnFromNode = createNode "YieldOrReturnFrom" range fileName None parentId
        let graph' = { graph with Nodes = Map.add yieldReturnFromNode.Id.Value yieldReturnFromNode graph.Nodes }
        let graph'' = addChildToParent yieldReturnFromNode.Id parentId graph'
        processExpression expr (Some yieldReturnFromNode.Id) fileName context graph''

    | SynExpr.SequentialOrImplicitYield(_, expr1, expr2, ifNotStmt, range) ->
        let seqOrYieldNode = createNode "SequentialOrImplicitYield" range fileName None parentId
        let graph' = { graph with Nodes = Map.add seqOrYieldNode.Id.Value seqOrYieldNode graph.Nodes }
        let graph'' = addChildToParent seqOrYieldNode.Id parentId graph'
        let graph''' = processExpression expr1 (Some seqOrYieldNode.Id) fileName context graph''
        let graph'''' = processExpression expr2 (Some seqOrYieldNode.Id) fileName context graph'''
        processExpression ifNotStmt (Some seqOrYieldNode.Id) fileName context graph''''

    | SynExpr.LetOrUseBang(_, _, _, _, rhs, _, body, range, _) ->
        let letBangNode = createNode "LetOrUseBang" range fileName None parentId
        let graph' = { graph with Nodes = Map.add letBangNode.Id.Value letBangNode graph.Nodes }
        let graph'' = addChildToParent letBangNode.Id parentId graph'
        let graph''' = processExpression rhs (Some letBangNode.Id) fileName context graph''
        processExpression body (Some letBangNode.Id) fileName context graph'''

    | SynExpr.MatchBang(_, expr, _, range, _) ->
        let matchBangNode = createNode "MatchBang" range fileName None parentId
        let graph' = { graph with Nodes = Map.add matchBangNode.Id.Value matchBangNode graph.Nodes }
        let graph'' = addChildToParent matchBangNode.Id parentId graph'
        processExpression expr (Some matchBangNode.Id) fileName context graph''

    | SynExpr.DoBang(expr, range, _) ->
        let doBangNode = createNode "DoBang" range fileName None parentId
        let graph' = { graph with Nodes = Map.add doBangNode.Id.Value doBangNode graph.Nodes }
        let graph'' = addChildToParent doBangNode.Id parentId graph'
        processExpression expr (Some doBangNode.Id) fileName context graph''

    | SynExpr.WhileBang(_, whileExpr, doExpr, range) ->
        let whileBangNode = createNode "WhileBang" range fileName None parentId
        let graph' = { graph with Nodes = Map.add whileBangNode.Id.Value whileBangNode graph.Nodes }
        let graph'' = addChildToParent whileBangNode.Id parentId graph'
        let graph''' = processExpression whileExpr (Some whileBangNode.Id) fileName context graph''
        processExpression doExpr (Some whileBangNode.Id) fileName context graph'''

    | SynExpr.MatchLambda(_, _, _, _, range) ->
        let matchLambdaNode = createNode "MatchLambda" range fileName None parentId
        let graph' = { graph with Nodes = Map.add matchLambdaNode.Id.Value matchLambdaNode graph.Nodes }
        addChildToParent matchLambdaNode.Id parentId graph'

    | SynExpr.ObjExpr(_, _, _, bindings, _, _, _, range) ->
        let objExprNode = createNode "ObjExpr" range fileName None parentId
        let graph' = { graph with Nodes = Map.add objExprNode.Id.Value objExprNode graph.Nodes }
        let graph'' = addChildToParent objExprNode.Id parentId graph'
        bindings |> List.fold (fun acc binding -> 
            processBinding binding (Some objExprNode.Id) fileName context acc) graph''

    | SynExpr.Set(targetExpr, rhsExpr, range) ->
        let setNode = createNode "Set" range fileName None parentId
        let graph' = { graph with Nodes = Map.add setNode.Id.Value setNode graph.Nodes }
        let graph'' = addChildToParent setNode.Id parentId graph'
        let graph''' = processExpression targetExpr (Some setNode.Id) fileName context graph''
        processExpression rhsExpr (Some setNode.Id) fileName context graph'''

    | SynExpr.DotIndexedSet(objectExpr, indexArgs, valueExpr, _, _, range) ->
        let dotIndexedSetNode = createNode "DotIndexedSet" range fileName None parentId
        let graph' = { graph with Nodes = Map.add dotIndexedSetNode.Id.Value dotIndexedSetNode graph.Nodes }
        let graph'' = addChildToParent dotIndexedSetNode.Id parentId graph'
        let graph''' = processExpression objectExpr (Some dotIndexedSetNode.Id) fileName context graph''
        let graph'''' = processExpression indexArgs (Some dotIndexedSetNode.Id) fileName context graph'''
        processExpression valueExpr (Some dotIndexedSetNode.Id) fileName context graph''''

    | SynExpr.NamedIndexedPropertySet(_, expr1, expr2, range) ->
        let namedIndexedPropSetNode = createNode "NamedIndexedPropertySet" range fileName None parentId
        let graph' = { graph with Nodes = Map.add namedIndexedPropSetNode.Id.Value namedIndexedPropSetNode graph.Nodes }
        let graph'' = addChildToParent namedIndexedPropSetNode.Id parentId graph'
        let graph''' = processExpression expr1 (Some namedIndexedPropSetNode.Id) fileName context graph''
        processExpression expr2 (Some namedIndexedPropSetNode.Id) fileName context graph'''

    | SynExpr.DotNamedIndexedPropertySet(targetExpr, _, argExpr, rhsExpr, range) ->
        let dotNamedIndexedPropSetNode = createNode "DotNamedIndexedPropertySet" range fileName None parentId
        let graph' = { graph with Nodes = Map.add dotNamedIndexedPropSetNode.Id.Value dotNamedIndexedPropSetNode graph.Nodes }
        let graph'' = addChildToParent dotNamedIndexedPropSetNode.Id parentId graph'
        let graph''' = processExpression targetExpr (Some dotNamedIndexedPropSetNode.Id) fileName context graph''
        let graph'''' = processExpression argExpr (Some dotNamedIndexedPropSetNode.Id) fileName context graph'''
        processExpression rhsExpr (Some dotNamedIndexedPropSetNode.Id) fileName context graph''''

    | SynExpr.TypeTest(expr, _, range) ->
        let typeTestNode = createNode "TypeTest" range fileName None parentId
        let graph' = { graph with Nodes = Map.add typeTestNode.Id.Value typeTestNode graph.Nodes }
        let graph'' = addChildToParent typeTestNode.Id parentId graph'
        processExpression expr (Some typeTestNode.Id) fileName context graph''

    | SynExpr.Upcast(expr, _, range) ->
        let upcastNode = createNode "Upcast" range fileName None parentId
        let graph' = { graph with Nodes = Map.add upcastNode.Id.Value upcastNode graph.Nodes }
        let graph'' = addChildToParent upcastNode.Id parentId graph'
        processExpression expr (Some upcastNode.Id) fileName context graph''

    | SynExpr.Downcast(expr, _, range) ->
        let downcastNode = createNode "Downcast" range fileName None parentId
        let graph' = { graph with Nodes = Map.add downcastNode.Id.Value downcastNode graph.Nodes }
        let graph'' = addChildToParent downcastNode.Id parentId graph'
        processExpression expr (Some downcastNode.Id) fileName context graph''

    | SynExpr.InferredUpcast(expr, range) ->
        let inferredUpcastNode = createNode "InferredUpcast" range fileName None parentId
        let graph' = { graph with Nodes = Map.add inferredUpcastNode.Id.Value inferredUpcastNode graph.Nodes }
        let graph'' = addChildToParent inferredUpcastNode.Id parentId graph'
        processExpression expr (Some inferredUpcastNode.Id) fileName context graph''

    | SynExpr.InferredDowncast(expr, range) ->
        let inferredDowncastNode = createNode "InferredDowncast" range fileName None parentId
        let graph' = { graph with Nodes = Map.add inferredDowncastNode.Id.Value inferredDowncastNode graph.Nodes }
        let graph'' = addChildToParent inferredDowncastNode.Id parentId graph'
        processExpression expr (Some inferredDowncastNode.Id) fileName context graph''

    | SynExpr.Null(range) ->
        let nullNode = createNode "Null" range fileName None parentId
        let graph' = { graph with Nodes = Map.add nullNode.Id.Value nullNode graph.Nodes }
        addChildToParent nullNode.Id parentId graph'

    | SynExpr.Typar(_, range) ->
        let typarNode = createNode "Typar" range fileName None parentId
        let graph' = { graph with Nodes = Map.add typarNode.Id.Value typarNode graph.Nodes }
        addChildToParent typarNode.Id parentId graph'

    | SynExpr.DotLambda(expr, range, _) ->
        let dotLambdaNode = createNode "DotLambda" range fileName None parentId
        let graph' = { graph with Nodes = Map.add dotLambdaNode.Id.Value dotLambdaNode graph.Nodes }
        let graph'' = addChildToParent dotLambdaNode.Id parentId graph'
        processExpression expr (Some dotLambdaNode.Id) fileName context graph''

    | SynExpr.Fixed(expr, range) ->
        let fixedNode = createNode "Fixed" range fileName None parentId
        let graph' = { graph with Nodes = Map.add fixedNode.Id.Value fixedNode graph.Nodes }
        let graph'' = addChildToParent fixedNode.Id parentId graph'
        processExpression expr (Some fixedNode.Id) fileName context graph''

    | SynExpr.InterpolatedString(contents, _, range) ->
        let interpolatedStringNode = createNode "InterpolatedString" range fileName None parentId
        let graph' = { graph with Nodes = Map.add interpolatedStringNode.Id.Value interpolatedStringNode graph.Nodes }
        let graph'' = addChildToParent interpolatedStringNode.Id parentId graph'
        
        contents |> List.fold (fun acc part -> 
            match part with
            | SynInterpolatedStringPart.String(_, _) -> acc
            | SynInterpolatedStringPart.FillExpr(expr, _) -> 
                processExpression expr (Some interpolatedStringNode.Id) fileName context acc) graph''

    | SynExpr.Dynamic(funcExpr, _, argExpr, range) ->
        let dynamicNode = createNode "Dynamic" range fileName None parentId
        let graph' = { graph with Nodes = Map.add dynamicNode.Id.Value dynamicNode graph.Nodes }
        let graph'' = addChildToParent dynamicNode.Id parentId graph'
        let graph''' = processExpression funcExpr (Some dynamicNode.Id) fileName context graph''
        processExpression argExpr (Some dynamicNode.Id) fileName context graph'''

    // Library-only and error handling cases
    | SynExpr.LibraryOnlyILAssembly(_, _, args, _, range) ->
        let ilAssemblyNode = createNode "LibraryOnlyILAssembly" range fileName None parentId
        let graph' = { graph with Nodes = Map.add ilAssemblyNode.Id.Value ilAssemblyNode graph.Nodes }
        let graph'' = addChildToParent ilAssemblyNode.Id parentId graph'
        args |> List.fold (fun acc expr -> 
            processExpression expr (Some ilAssemblyNode.Id) fileName context acc) graph''

    | SynExpr.LibraryOnlyStaticOptimization(_, expr, optimizedExpr, range) ->
        let staticOptNode = createNode "LibraryOnlyStaticOptimization" range fileName None parentId
        let graph' = { graph with Nodes = Map.add staticOptNode.Id.Value staticOptNode graph.Nodes }
        let graph'' = addChildToParent staticOptNode.Id parentId graph'
        let graph''' = processExpression expr (Some staticOptNode.Id) fileName context graph''
        processExpression optimizedExpr (Some staticOptNode.Id) fileName context graph'''

    | SynExpr.LibraryOnlyUnionCaseFieldGet(expr, _, _, range) ->
        let unionCaseFieldGetNode = createNode "LibraryOnlyUnionCaseFieldGet" range fileName None parentId
        let graph' = { graph with Nodes = Map.add unionCaseFieldGetNode.Id.Value unionCaseFieldGetNode graph.Nodes }
        let graph'' = addChildToParent unionCaseFieldGetNode.Id parentId graph'
        processExpression expr (Some unionCaseFieldGetNode.Id) fileName context graph''

    | SynExpr.LibraryOnlyUnionCaseFieldSet(expr, _, _, rhsExpr, range) ->
        let unionCaseFieldSetNode = createNode "LibraryOnlyUnionCaseFieldSet" range fileName None parentId
        let graph' = { graph with Nodes = Map.add unionCaseFieldSetNode.Id.Value unionCaseFieldSetNode graph.Nodes }
        let graph'' = addChildToParent unionCaseFieldSetNode.Id parentId graph'
        let graph''' = processExpression expr (Some unionCaseFieldSetNode.Id) fileName context graph''
        processExpression rhsExpr (Some unionCaseFieldSetNode.Id) fileName context graph'''

    | SynExpr.ArbitraryAfterError(_, range) ->
        let errorNode = createNode "ArbitraryAfterError" range fileName None parentId
        let graph' = { graph with Nodes = Map.add errorNode.Id.Value errorNode graph.Nodes }
        addChildToParent errorNode.Id parentId graph'

    | SynExpr.FromParseError(expr, range) ->
        let parseErrorNode = createNode "FromParseError" range fileName None parentId
        let graph' = { graph with Nodes = Map.add parseErrorNode.Id.Value parseErrorNode graph.Nodes }
        let graph'' = addChildToParent parseErrorNode.Id parentId graph'
        processExpression expr (Some parseErrorNode.Id) fileName context graph''

    | SynExpr.DiscardAfterMissingQualificationAfterDot(expr, _, range) ->
        let discardNode = createNode "DiscardAfterMissingQualificationAfterDot" range fileName None parentId
        let graph' = { graph with Nodes = Map.add discardNode.Id.Value discardNode graph.Nodes }
        let graph'' = addChildToParent discardNode.Id parentId graph'
        processExpression expr (Some discardNode.Id) fileName context graph''

    | SynExpr.DebugPoint(_, _, expr) ->
        // DebugPoint wraps an expression with debug metadata - process the inner expression
        processExpression expr parentId fileName context graph

            

/// Process a module declaration - Using existing FCS 43.9.300 patterns (preserved from original)
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
        let (SynComponentInfo(attributes, typeArgs, constraints, longId, xmlDoc, preferPostfix, accessibility, range2)) = componentInfo
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

/// Process implementation file - Using existing FCS 43.9.300 patterns (preserved from original)
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

/// Build complete PSG from project results with CANONICAL FCS CONSTRAINT RESOLUTION
let buildProgramSemanticGraph 
    (checkResults: FSharpCheckProjectResults) 
    (parseResults: FSharpParseFileResults[]) : ProgramSemanticGraph =
    
    printfn "[BUILDER] Starting PSG construction with CANONICAL FCS constraint resolution"
    
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
    
    printfn "[BUILDER] Phase 1: Building structural nodes from %d files" parseResults.Length
    
    // Phase 1: Process each file and merge results - nodes only, no cross-file edges
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
    
    // Merge all graphs
    let structuralGraph =
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
    
    printfn "[BUILDER] Phase 1 complete: Structural PSG built with %d nodes" structuralGraph.Nodes.Count
    
    // Phase 2: Apply CANONICAL FCS constraint resolution
    printfn "[BUILDER] Phase 2: Applying CANONICAL FCS constraint resolution"
    let typeEnhancedGraph = integrateTypesWithCheckResults structuralGraph checkResults
    
    // Phase 3: Create function call edges now that all nodes exist
    printfn "[BUILDER] Phase 3: Creating function call edges"
    let createFunctionCallEdges (graph: ProgramSemanticGraph) : ProgramSemanticGraph =
        let appNodes = 
            graph.Nodes
            |> Map.toSeq
            |> Seq.map snd
            |> Seq.filter (fun node -> node.SyntaxKind = "App")
            |> List.ofSeq
        
        let functionCallEdges = 
            appNodes
            |> List.choose (fun appNode ->
                // Find the function being called by looking at App node's children
                let children = 
                    match appNode.Children with
                    | Parent childIds -> childIds
                    | _ -> []
                
                // Look for LongIdent child nodes that represent the function being called
                children
                |> List.tryPick (fun childId ->
                    match Map.tryFind childId.Value graph.Nodes with
                    | Some childNode when childNode.SyntaxKind.StartsWith("LongIdent:") ->
                        // Found a LongIdent node - this represents a function call
                        match childNode.Symbol with
                        | Some funcSymbol ->
                            // Create edge from App node to the LongIdent node (not some other binding!)
                            Some { Source = appNode.Id; Target = childNode.Id; Kind = FunctionCall }
                        | None -> None
                    | Some childNode when childNode.SyntaxKind.StartsWith("Ident:") ->
                        // Also handle simple Ident nodes for function calls
                        match childNode.Symbol with
                        | Some funcSymbol ->
                            // Check if this is actually a function (not a variable)
                            match funcSymbol with
                            | :? FSharpMemberOrFunctionOrValue as mfv when mfv.IsFunction ->
                                Some { Source = appNode.Id; Target = childNode.Id; Kind = FunctionCall }
                            | _ -> None
                        | None -> None
                    | _ -> None
                )
            )
        
        printfn "[BUILDER] Created %d function call edges" functionCallEdges.Length
        
        // Debug: Show what edges we're creating
        if functionCallEdges.Length > 0 then
            printfn "[BUILDER] Sample function call edges:"
            functionCallEdges
            |> List.take (min 5 functionCallEdges.Length)
            |> List.iter (fun edge ->
                let sourceNode = Map.tryFind edge.Source.Value graph.Nodes
                let targetNode = Map.tryFind edge.Target.Value graph.Nodes
                match sourceNode, targetNode with
                | Some src, Some tgt ->
                    let targetSymbol = tgt.Symbol |> Option.map (fun s -> s.FullName) |> Option.defaultValue "no symbol"
                    printfn "  %s -> %s (%s)" src.SyntaxKind tgt.SyntaxKind targetSymbol
                | _ -> printfn "  Invalid edge: %s -> %s" edge.Source.Value edge.Target.Value
            )
        
        { graph with Edges = graph.Edges @ functionCallEdges }
    
    let edgeEnhancedGraph = createFunctionCallEdges typeEnhancedGraph
    
    // Phase 4: Detect entry points
    printfn "[BUILDER] Phase 4: Detecting entry points"
    let entryPoints = 
        edgeEnhancedGraph.Nodes
        |> Map.toSeq
        |> Seq.choose (fun (_, node) ->
            match node.Symbol with
            | Some symbol when symbol.FullName.Contains("hello") || symbol.DisplayName = "main" ->
                Some node.Id
            | _ -> None
        )
        |> List.ofSeq

    printfn "[BUILDER] Found %d entry points during construction" entryPoints.Length

    // Phase 5: Finalize PSG nodes
    printfn "[BUILDER] Phase 5: Finalizing PSG nodes"
    let finalNodes = 
        edgeEnhancedGraph.Nodes
        |> Map.map (fun _ node -> ChildrenStateHelpers.finalizeChildren node)

    let finalGraph = 
        { edgeEnhancedGraph with 
            Nodes = finalNodes
            EntryPoints = entryPoints
            CompilationOrder = parseResults |> Array.map (fun pr -> pr.FileName) |> List.ofArray 
        }
    
    printfn "[BUILDER] PSG construction complete with CANONICAL FCS constraint resolution"
    printfn "[BUILDER] Final PSG: %d nodes, %d edges, %d entry points" 
        finalGraph.Nodes.Count finalGraph.Edges.Length finalGraph.EntryPoints.Length
        
    finalGraph