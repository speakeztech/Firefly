/// Expression processing for PSG construction
module Core.PSG.Construction.ExpressionProcessing

open FSharp.Compiler.Text
open FSharp.Compiler.Syntax
open FSharp.Compiler.Symbols
open Core.PSG.Types
open Core.PSG.Construction.Types
open Core.PSG.Construction.SymbolCorrelation
open Core.PSG.Construction.PatternProcessing

// Forward declaration for mutual recursion with BindingProcessing
let mutable private processBindingWithUseFlagRef : (SynBinding -> NodeId option -> string -> BuildContext -> ProgramSemanticGraph -> bool -> ProgramSemanticGraph) option = None

/// Set the binding processor (called from BindingProcessing to break circular dependency)
let setBindingProcessor processor =
    processBindingWithUseFlagRef <- Some processor

/// Process an expression node in the PSG
let rec processExpression (expr: SynExpr) (parentId: NodeId option) (fileName: string)
                          (context: BuildContext) (graph: ProgramSemanticGraph) : ProgramSemanticGraph =
    match expr with

    // ENHANCED: Method calls like obj.Method() - FCS 43.9.300 compatible
    | SynExpr.DotGet(expr, _, longDotId, range) ->
        let methodName =
            longDotId.LongIdent |> List.map (fun id -> id.idText) |> String.concat "."

        let syntaxKind = sprintf "MethodCall:%s" methodName
        let methodSymbol = tryCorrelateSymbolWithContext range fileName syntaxKind context.CorrelationContext
        let methodCallNode = createNode syntaxKind range fileName methodSymbol parentId

        let graph' = { graph with Nodes = Map.add methodCallNode.Id.Value methodCallNode graph.Nodes }
        let graph'' = addChildToParent methodCallNode.Id parentId graph'

        let graph''' =
            match methodSymbol with
            | Some sym ->
                let updatedSymbolTable = Map.add sym.DisplayName sym graph''.SymbolTable
                let methodRefEdge = {
                    Source = methodCallNode.Id
                    Target = methodCallNode.Id
                    Kind = SymRef
                }
                { graph'' with
                    SymbolTable = updatedSymbolTable
                    Edges = methodRefEdge :: graph''.Edges }
            | None ->
                printfn "[BUILDER] Warning: Method call '%s' at %s has no symbol correlation" methodName (range.ToString())
                graph''

        processExpression expr (Some methodCallNode.Id) fileName context graph'''

    // ENHANCED: Generic type applications like stackBuffer<byte>
    | SynExpr.TypeApp(expr, _, typeArgs, _, _, _, range) ->
        let typeArgNames =
            typeArgs
            |> List.choose (fun t ->
                match t with
                | SynType.LongIdent(SynLongIdent(idents, _, _)) ->
                    Some (idents |> List.map (fun id -> id.idText) |> String.concat ".")
                | _ -> None)
            |> String.concat ", "

        let syntaxKind =
            if System.String.IsNullOrEmpty(typeArgNames) then "TypeApp:Generic"
            else sprintf "TypeApp:%s" typeArgNames

        let genericSymbol = tryCorrelateSymbolWithContext range fileName syntaxKind context.CorrelationContext

        let typeAppNode = createNode syntaxKind range fileName genericSymbol parentId
        let graph' = { graph with Nodes = Map.add typeAppNode.Id.Value typeAppNode graph.Nodes }
        let graph'' = addChildToParent typeAppNode.Id parentId graph'

        let graph''' =
            match genericSymbol with
            | Some sym ->
                let updatedSymbolTable = Map.add sym.DisplayName sym graph''.SymbolTable
                let typeInstEdge = {
                    Source = typeAppNode.Id
                    Target = typeAppNode.Id
                    Kind = TypeInstantiation []
                }
                { graph'' with
                    SymbolTable = updatedSymbolTable
                    Edges = typeInstEdge :: graph''.Edges }
            | None ->
                printfn "[BUILDER] Warning: Generic type application at %s has no symbol correlation" (range.ToString())
                graph''

        processExpression expr (Some typeAppNode.Id) fileName context graph'''

    // ENHANCED: Function calls with better correlation
    | SynExpr.App(_, _, funcExpr, argExpr, range) ->
        let syntaxKind = "App:FunctionCall"
        let symbol = tryCorrelateSymbolWithContext range fileName syntaxKind context.CorrelationContext
        let appNode = createNode syntaxKind range fileName symbol parentId

        let graph' = { graph with Nodes = Map.add appNode.Id.Value appNode graph.Nodes }
        let graph'' = addChildToParent appNode.Id parentId graph'

        let graph''' =
            match symbol with
            | Some sym ->
                { graph'' with SymbolTable = Map.add sym.DisplayName sym graph''.SymbolTable }
            | None -> graph''

        let graph'''' = processExpression funcExpr (Some appNode.Id) fileName context graph'''
        let graph''''' = processExpression argExpr (Some appNode.Id) fileName context graph''''

        // Create function call edges
        let functionCallEdges =
            match funcExpr with
            | SynExpr.Ident ident ->
                match tryCorrelateSymbolWithContext ident.idRange fileName (sprintf "Ident:%s" ident.idText) context.CorrelationContext with
                | Some funcSym when isFunction funcSym ->
                    [{ Source = appNode.Id; Target = appNode.Id; Kind = FunctionCall }]
                | _ -> []
            | SynExpr.LongIdent(_, longDotId, _, _) ->
                let identText = longDotId.LongIdent |> List.map (fun id -> id.idText) |> String.concat "."
                match tryCorrelateSymbolWithContext longDotId.Range fileName (sprintf "LongIdent:%s" identText) context.CorrelationContext with
                | Some funcSym when isFunction funcSym ->
                    [{ Source = appNode.Id; Target = appNode.Id; Kind = FunctionCall }]
                | _ -> []
            | _ -> []

        { graph''''' with Edges = graph'''''.Edges @ functionCallEdges }

    // ENHANCED: Match expressions with union case detection
    | SynExpr.Match(_, expr, clauses, range, _) ->
        let matchNode = createNode "Match" range fileName None parentId
        let graph' = { graph with Nodes = Map.add matchNode.Id.Value matchNode graph.Nodes }
        let graph'' = addChildToParent matchNode.Id parentId graph'

        let graph''' = processExpression expr (Some matchNode.Id) fileName context graph''

        clauses
        |> List.fold (fun acc clause ->
            let (SynMatchClause(pat, whenExpr, resultExpr, _, _, _)) = clause

            let clauseRange = pat.Range
            let clauseNode = createNode "MatchClause" clauseRange fileName None (Some matchNode.Id)
            let graphAcc' = { acc with Nodes = Map.add clauseNode.Id.Value clauseNode acc.Nodes }
            let graphAcc'' = addChildToParent clauseNode.Id (Some matchNode.Id) graphAcc'

            let graphAcc''' = processPattern pat (Some clauseNode.Id) fileName context graphAcc''

            let graphAcc'''' =
                match whenExpr with
                | Some whenE -> processExpression whenE (Some clauseNode.Id) fileName context graphAcc'''
                | None -> graphAcc'''

            processExpression resultExpr (Some clauseNode.Id) fileName context graphAcc''''
        ) graph'''

    | SynExpr.Ident ident ->
        let syntaxKind = sprintf "Ident:%s" ident.idText
        let symbol = tryCorrelateSymbolWithContext ident.idRange fileName syntaxKind context.CorrelationContext
        let identNode = createNode syntaxKind ident.idRange fileName symbol parentId

        let graph' = { graph with Nodes = Map.add identNode.Id.Value identNode graph.Nodes }
        let graph'' = addChildToParent identNode.Id parentId graph'

        match symbol with
        | Some sym ->
            { graph'' with SymbolTable = Map.add sym.DisplayName sym graph''.SymbolTable }
        | None -> graph''

    | SynExpr.LongIdent(_, longDotId, _, range) ->
        let identText = longDotId.LongIdent |> List.map (fun id -> id.idText) |> String.concat "."

        // ENHANCED: Detect method calls in LongIdent (like buffer.AsReadOnlySpan)
        let syntaxKind, isMethodCall =
            let parts = longDotId.LongIdent |> List.map (fun id -> id.idText)
            if parts.Length > 1 then
                let methodName = parts |> List.last
                // Check if this looks like a method call pattern
                if methodName = "AsReadOnlySpan" || methodName = "Pointer" || methodName = "Length" ||
                   methodName.StartsWith("get_") || methodName.StartsWith("set_") then
                    (sprintf "LongIdent:MethodCall:%s" methodName, true)
                else
                    (sprintf "LongIdent:%s" identText, false)
            else
                (sprintf "LongIdent:%s" identText, false)

        let symbol = tryCorrelateSymbolWithContext range fileName syntaxKind context.CorrelationContext
        let longIdentNode = createNode syntaxKind range fileName symbol parentId

        let graph' = { graph with Nodes = Map.add longIdentNode.Id.Value longIdentNode graph.Nodes }
        let graph'' = addChildToParent longIdentNode.Id parentId graph'

        let graph''' =
            match symbol with
            | Some sym ->
                let updatedSymbolTable = Map.add sym.DisplayName sym graph''.SymbolTable

                // Create method reference edge for method calls
                if isMethodCall then
                    let methodRefEdge = {
                        Source = longIdentNode.Id
                        Target = longIdentNode.Id
                        Kind = SymRef
                    }
                    { graph'' with
                        SymbolTable = updatedSymbolTable
                        Edges = methodRefEdge :: graph''.Edges }
                else
                    { graph'' with SymbolTable = updatedSymbolTable }
            | None ->
                if isMethodCall then
                    printfn "[BUILDER] Warning: Method call in LongIdent '%s' at %s has no symbol correlation" identText (range.ToString())
                graph''

        graph'''

    | SynExpr.LetOrUse(isUse, _, bindings, body, range, _) ->
        let syntaxKind = if isUse then "LetOrUse:Use" else "LetOrUse:Let"
        let symbol = tryCorrelateSymbolWithContext range fileName syntaxKind context.CorrelationContext
        let letNode = createNode syntaxKind range fileName symbol parentId

        let graph' = { graph with Nodes = Map.add letNode.Id.Value letNode graph.Nodes }
        let graph'' = addChildToParent letNode.Id parentId graph'

        let graph''' =
            match symbol with
            | Some sym ->
                { graph'' with SymbolTable = Map.add sym.DisplayName sym graph''.SymbolTable }
            | None -> graph''

        let graph'''' =
            match processBindingWithUseFlagRef with
            | Some processBindingWithUseFlag ->
                bindings
                |> List.fold (fun acc binding ->
                    processBindingWithUseFlag binding (Some letNode.Id) fileName context acc isUse) graph'''
            | None ->
                failwith "Binding processor not initialized"

        processExpression body (Some letNode.Id) fileName context graph''''

    | SynExpr.Sequential(_, _, expr1, expr2, _, _) ->
        let graph' = processExpression expr1 parentId fileName context graph
        processExpression expr2 parentId fileName context graph'

    | SynExpr.Const(constant, range) ->
        let constNode = createNode (sprintf "Const:%A" constant) range fileName None parentId
        let graph' = { graph with Nodes = Map.add constNode.Id.Value constNode graph.Nodes }
        addChildToParent constNode.Id parentId graph'

    | SynExpr.TryWith(tryExpr, withCases, range, _, _, trivia) ->
        let tryWithNode = createNode "TryWith" range fileName None parentId
        let graph' = { graph with Nodes = Map.add tryWithNode.Id.Value tryWithNode graph.Nodes }
        let graph'' = addChildToParent tryWithNode.Id parentId graph'

        // Process try block
        let graph''' = processExpression tryExpr (Some tryWithNode.Id) fileName context graph''

        // Process with clauses
        withCases
        |> List.fold (fun acc clause ->
            let (SynMatchClause(pat, whenExpr, resultExpr, _, _, _)) = clause
            let clauseNode = createNode "WithClause" pat.Range fileName None (Some tryWithNode.Id)
            let graphAcc' = { acc with Nodes = Map.add clauseNode.Id.Value clauseNode acc.Nodes }
            let graphAcc'' = addChildToParent clauseNode.Id (Some tryWithNode.Id) graphAcc'
            let graphAcc''' = processPattern pat (Some clauseNode.Id) fileName context graphAcc''
            processExpression resultExpr (Some clauseNode.Id) fileName context graphAcc'''
        ) graph'''

    | SynExpr.TryFinally(tryExpr, finallyExpr, range, _, _, trivia) ->
        let tryFinallyNode = createNode "TryFinally" range fileName None parentId
        let graph' = { graph with Nodes = Map.add tryFinallyNode.Id.Value tryFinallyNode graph.Nodes }
        let graph'' = addChildToParent tryFinallyNode.Id parentId graph'

        // Process try block
        let graph''' = processExpression tryExpr (Some tryFinallyNode.Id) fileName context graph''

        // Process finally block
        processExpression finallyExpr (Some tryFinallyNode.Id) fileName context graph'''

    // CRITICAL: Parenthesized expressions - must process inner expression with same parent
    | SynExpr.Paren(innerExpr, _, _, range) ->
        processExpression innerExpr parentId fileName context graph

    // Tuple expressions
    | SynExpr.Tuple(_, exprs, _, range) ->
        let tupleNode = createNode "Tuple" range fileName None parentId
        let graph' = { graph with Nodes = Map.add tupleNode.Id.Value tupleNode graph.Nodes }
        let graph'' = addChildToParent tupleNode.Id parentId graph'
        exprs |> List.fold (fun g expr ->
            processExpression expr (Some tupleNode.Id) fileName context g) graph''

    // Lambda expressions
    | SynExpr.Lambda(_, _, _, body, _, range, _) ->
        let lambdaNode = createNode "Lambda" range fileName None parentId
        let graph' = { graph with Nodes = Map.add lambdaNode.Id.Value lambdaNode graph.Nodes }
        let graph'' = addChildToParent lambdaNode.Id parentId graph'
        processExpression body (Some lambdaNode.Id) fileName context graph''

    // If-then-else expressions
    | SynExpr.IfThenElse(condExpr, thenExpr, elseExprOpt, _, _, range, _) ->
        let ifNode = createNode "IfThenElse" range fileName None parentId
        let graph' = { graph with Nodes = Map.add ifNode.Id.Value ifNode graph.Nodes }
        let graph'' = addChildToParent ifNode.Id parentId graph'

        let graph''' = processExpression condExpr (Some ifNode.Id) fileName context graph''
        let graph'''' = processExpression thenExpr (Some ifNode.Id) fileName context graph'''
        match elseExprOpt with
        | Some elseExpr -> processExpression elseExpr (Some ifNode.Id) fileName context graph''''
        | None -> graph''''

    // While loops
    | SynExpr.While(_, condExpr, bodyExpr, range) ->
        let whileNode = createNode "WhileLoop" range fileName None parentId
        let graph' = { graph with Nodes = Map.add whileNode.Id.Value whileNode graph.Nodes }
        let graph'' = addChildToParent whileNode.Id parentId graph'

        let graph''' = processExpression condExpr (Some whileNode.Id) fileName context graph''
        processExpression bodyExpr (Some whileNode.Id) fileName context graph'''

    // Mutable variable assignment (counter <- counter + 1)
    | SynExpr.LongIdentSet(longDotId, rhsExpr, range) ->
        let varName = longDotId.LongIdent |> List.map (fun id -> id.idText) |> String.concat "."
        let syntaxKind = sprintf "MutableSet:%s" varName
        let setNode = createNode syntaxKind range fileName None parentId
        let graph' = { graph with Nodes = Map.add setNode.Id.Value setNode graph.Nodes }
        let graph'' = addChildToParent setNode.Id parentId graph'
        processExpression rhsExpr (Some setNode.Id) fileName context graph''

    // For loops (for i = start to end do ...)
    | SynExpr.For(_, _, ident, _, startExpr, _, endExpr, bodyExpr, range) ->
        let syntaxKind = sprintf "ForLoop:%s" ident.idText
        let forNode = createNode syntaxKind range fileName None parentId
        let graph' = { graph with Nodes = Map.add forNode.Id.Value forNode graph.Nodes }
        let graph'' = addChildToParent forNode.Id parentId graph'
        let graph''' = processExpression startExpr (Some forNode.Id) fileName context graph''
        let graph'''' = processExpression endExpr (Some forNode.Id) fileName context graph'''
        processExpression bodyExpr (Some forNode.Id) fileName context graph''''

    // ForEach loops (for item in collection do ...)
    | SynExpr.ForEach(_, _, _, _, pat, enumExpr, bodyExpr, range) ->
        let forEachNode = createNode "ForEachLoop" range fileName None parentId
        let graph' = { graph with Nodes = Map.add forEachNode.Id.Value forEachNode graph.Nodes }
        let graph'' = addChildToParent forEachNode.Id parentId graph'
        let graph''' = processPattern pat (Some forEachNode.Id) fileName context graph''
        let graph'''' = processExpression enumExpr (Some forEachNode.Id) fileName context graph'''
        processExpression bodyExpr (Some forEachNode.Id) fileName context graph''''

    // Type-annotated expressions (expr : type)
    | SynExpr.Typed(innerExpr, typeSig, range) ->
        processExpression innerExpr parentId fileName context graph

    // SRTP trait calls (statically resolved type parameters)
    | SynExpr.TraitCall(typeArgs, memberSig, argExpr, range) ->
        let traitNode = createNode "TraitCall" range fileName None parentId
        let graph' = { graph with Nodes = Map.add traitNode.Id.Value traitNode graph.Nodes }
        let graph'' = addChildToParent traitNode.Id parentId graph'
        processExpression argExpr (Some traitNode.Id) fileName context graph''

    // Computed arrays/lists ([| for ... |], [ for ... ])
    | SynExpr.ArrayOrListComputed(isArray, innerExpr, range) ->
        let kind = if isArray then "ArrayComputed" else "ListComputed"
        let arrNode = createNode kind range fileName None parentId
        let graph' = { graph with Nodes = Map.add arrNode.Id.Value arrNode graph.Nodes }
        let graph'' = addChildToParent arrNode.Id parentId graph'
        processExpression innerExpr (Some arrNode.Id) fileName context graph''

    // Literal arrays/lists ([a; b], [|a; b|])
    | SynExpr.ArrayOrList(isArray, exprs, range) ->
        let kind = if isArray then "Array" else "List"
        let arrNode = createNode kind range fileName None parentId
        let graph' = { graph with Nodes = Map.add arrNode.Id.Value arrNode graph.Nodes }
        let graph'' = addChildToParent arrNode.Id parentId graph'
        exprs |> List.fold (fun g expr ->
            processExpression expr (Some arrNode.Id) fileName context g
        ) graph''

    // Address-of operator (&expr or &&expr)
    | SynExpr.AddressOf(isByRef, innerExpr, refRange, range) ->
        let addrNode = createNode "AddressOf" range fileName None parentId
        let graph' = { graph with Nodes = Map.add addrNode.Id.Value addrNode graph.Nodes }
        let graph'' = addChildToParent addrNode.Id parentId graph'
        processExpression innerExpr (Some addrNode.Id) fileName context graph''

    // Object construction (new Type(...))
    | SynExpr.New(isProtected, typeName, argExpr, range) ->
        let newNode = createNode "New" range fileName None parentId
        let graph' = { graph with Nodes = Map.add newNode.Id.Value newNode graph.Nodes }
        let graph'' = addChildToParent newNode.Id parentId graph'
        processExpression argExpr (Some newNode.Id) fileName context graph''

    // Null literal
    | SynExpr.Null(range) ->
        let nullNode = createNode "Const:Null" range fileName None parentId
        let graph' = { graph with Nodes = Map.add nullNode.Id.Value nullNode graph.Nodes }
        addChildToParent nullNode.Id parentId graph'

    // Simple mutable variable set (x <- expr)
    | SynExpr.Set(targetExpr, rhsExpr, range) ->
        let setNode = createNode "MutableSet" range fileName None parentId
        let graph' = { graph with Nodes = Map.add setNode.Id.Value setNode graph.Nodes }
        let graph'' = addChildToParent setNode.Id parentId graph'
        let graph''' = processExpression targetExpr (Some setNode.Id) fileName context graph''
        processExpression rhsExpr (Some setNode.Id) fileName context graph'''

    // Match lambda (function | pat -> expr | ...)
    | SynExpr.MatchLambda(isExnMatch, keywordRange, clauses, spBind, range) ->
        let matchLambdaNode = createNode "MatchLambda" range fileName None parentId
        let graph' = { graph with Nodes = Map.add matchLambdaNode.Id.Value matchLambdaNode graph.Nodes }
        let graph'' = addChildToParent matchLambdaNode.Id parentId graph'

        clauses |> List.fold (fun graphAcc clause ->
            let (SynMatchClause(pat, whenExpr, resultExpr, clauseRange, _, _)) = clause
            let clauseNode = createNode "MatchClause" clauseRange fileName None (Some matchLambdaNode.Id)
            let graphAcc' = { graphAcc with Nodes = Map.add clauseNode.Id.Value clauseNode graphAcc.Nodes }
            let graphAcc'' = addChildToParent clauseNode.Id (Some matchLambdaNode.Id) graphAcc'
            let graphAcc''' = processPattern pat (Some clauseNode.Id) fileName context graphAcc''
            let graphAcc'''' =
                match whenExpr with
                | Some whenE -> processExpression whenE (Some clauseNode.Id) fileName context graphAcc'''
                | None -> graphAcc'''
            processExpression resultExpr (Some clauseNode.Id) fileName context graphAcc''''
        ) graph''

    // Hard stop on unhandled expressions
    | other ->
        let exprTypeName = other.GetType().Name
        let range = other.Range
        failwithf "[BUILDER] ERROR: Unhandled expression type '%s' at %s in file %s. PSG construction cannot continue with unknown AST nodes."
            exprTypeName (range.ToString()) fileName
