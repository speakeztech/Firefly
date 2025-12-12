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

// ===================================================================
// Short-circuit Boolean Operator Detection
// ===================================================================

/// Detect if an expression is the && or || operator
/// Returns Some "&&" or Some "||" if it's a boolean operator, None otherwise
let private tryGetBooleanOperator (expr: SynExpr) : string option =
    match expr with
    | SynExpr.Ident ident ->
        match ident.idText with
        | "op_BooleanAnd" -> Some "&&"
        | "op_BooleanOr" -> Some "||"
        | _ -> None
    | SynExpr.LongIdent(_, longDotId, _, _) ->
        let lastIdent = longDotId.LongIdent |> List.tryLast
        match lastIdent with
        | Some ident ->
            match ident.idText with
            | "op_BooleanAnd" -> Some "&&"
            | "op_BooleanOr" -> Some "||"
            | _ -> None
        | None -> None
    | _ -> None

/// Try to extract a short-circuit boolean operation from a curried App expression
/// Pattern: App(App(op, leftArg), rightArg) where op is && or ||
/// Returns Some (operator, leftExpr, rightExpr) if it matches
let private tryExtractBooleanOp (funcExpr: SynExpr) (rightArg: SynExpr) : (string * SynExpr * SynExpr) option =
    match funcExpr with
    | SynExpr.App(_, _, innerFuncExpr, leftArg, _) ->
        match tryGetBooleanOperator innerFuncExpr with
        | Some op -> Some (op, leftArg, rightArg)
        | None -> None
    | _ -> None

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

        let typeAppNode = createNodeWithKind syntaxKind (SKExpr ETypeApp) range fileName genericSymbol parentId
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
    // FIRST: Check for short-circuit boolean operators (&&, ||) and desugar to IfThenElse
    | SynExpr.App(_, _, funcExpr, argExpr, range) ->
        match tryExtractBooleanOp funcExpr argExpr with
        | Some (op, leftExpr, rightExpr) ->
            // Desugar boolean operators to IfThenElse for correct control flow semantics
            // a || b  =>  if a then true else b
            // a && b  =>  if a then b else false
            let syntaxKind = sprintf "IfThenElse:BoolOp(%s)" op
            let ifNode = createNode syntaxKind range fileName None parentId
            let graph' = { graph with Nodes = Map.add ifNode.Id.Value ifNode graph.Nodes }
            let graph'' = addChildToParent ifNode.Id parentId graph'

            // Process the condition (left operand)
            let graph''' = processExpression leftExpr (Some ifNode.Id) fileName context graph''

            // For ||: then branch is "true", else branch is right operand
            // For &&: then branch is right operand, else branch is "false"
            match op with
            | "||" ->
                // Create a synthetic "true" constant node for the then branch
                let trueNode = createNode "Const:Bool true" range fileName None (Some ifNode.Id)
                let graph'''' = { graph''' with Nodes = Map.add trueNode.Id.Value trueNode graph'''.Nodes }
                let graph''''' = addChildToParent trueNode.Id (Some ifNode.Id) graph''''
                // Process the else branch (right operand)
                processExpression rightExpr (Some ifNode.Id) fileName context graph'''''
            | "&&" ->
                // Process the then branch (right operand)
                let graph'''' = processExpression rightExpr (Some ifNode.Id) fileName context graph'''
                // Create a synthetic "false" constant node for the else branch
                let falseNode = createNode "Const:Bool false" range fileName None (Some ifNode.Id)
                let graph''''' = { graph'''' with Nodes = Map.add falseNode.Id.Value falseNode graph''''.Nodes }
                addChildToParent falseNode.Id (Some ifNode.Id) graph'''''
            | _ ->
                // Should not happen, but fall through to normal App processing
                graph'''

        | None ->
            // Normal function application
            let syntaxKind = "App:FunctionCall"
            let symbol = tryCorrelateSymbolWithContext range fileName syntaxKind context.CorrelationContext
            let appNode = createNodeWithKind syntaxKind (SKExpr EApp) range fileName symbol parentId

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
        let matchNode = createNodeWithKind "Match" (SKExpr EMatch) range fileName None parentId
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
        let identNode = createNodeWithKind syntaxKind (SKExpr EIdent) ident.idRange fileName symbol parentId

        let graph' = { graph with Nodes = Map.add identNode.Id.Value identNode graph.Nodes }
        let graph'' = addChildToParent identNode.Id parentId graph'

        match symbol with
        | Some sym ->
            { graph'' with SymbolTable = Map.add sym.DisplayName sym graph''.SymbolTable }
        | None -> graph''

    | SynExpr.LongIdent(_, longDotId, _, range) ->
        let parts = longDotId.LongIdent
        let identText = parts |> List.map (fun id -> id.idText) |> String.concat "."

        // ENHANCED: Detect property/method access patterns (like oldValue.Length)
        // When we have multiple parts and the last is a property/method, decompose into
        // a PropertyAccess node with the receiver as a child
        let isPropertyAccess =
            if List.length parts > 1 then
                let methodName = (List.last parts).idText
                methodName = "AsReadOnlySpan" || methodName = "Pointer" || methodName = "Length" ||
                methodName.StartsWith("get_") || methodName.StartsWith("set_")
            else
                false

        if isPropertyAccess && List.length parts >= 2 then
            // Decompose: receiver.Property becomes PropertyAccess:Property with receiver as child
            let propertyName = (List.last parts).idText
            let receiverParts = parts |> List.take (List.length parts - 1)
            let receiverText = receiverParts |> List.map (fun id -> id.idText) |> String.concat "."

            // Create the PropertyAccess node
            let syntaxKind = sprintf "PropertyAccess:%s" propertyName
            let symbol = tryCorrelateSymbolWithContext range fileName syntaxKind context.CorrelationContext
            let propAccessNode = createNode syntaxKind range fileName symbol parentId

            let graph' = { graph with Nodes = Map.add propAccessNode.Id.Value propAccessNode graph.Nodes }
            let graph'' = addChildToParent propAccessNode.Id parentId graph'

            // Create the receiver node as a child
            // Use the range of the first ident for the receiver
            let receiverRange = (List.head receiverParts).idRange
            let receiverKind =
                if List.length receiverParts = 1 then sprintf "Ident:%s" receiverText
                else sprintf "LongIdent:%s" receiverText
            let receiverSymbol = tryCorrelateSymbolWithContext receiverRange fileName receiverKind context.CorrelationContext
            let receiverNode = createNode receiverKind receiverRange fileName receiverSymbol (Some propAccessNode.Id)

            let graph''' = { graph'' with Nodes = Map.add receiverNode.Id.Value receiverNode graph'' .Nodes }
            let graph'''' = addChildToParent receiverNode.Id (Some propAccessNode.Id) graph'''

            // Add symbols to table
            let graph''''' =
                match symbol with
                | Some sym ->
                    let st = Map.add sym.DisplayName sym graph''''.SymbolTable
                    let methodRefEdge = {
                        Source = propAccessNode.Id
                        Target = propAccessNode.Id
                        Kind = SymRef
                    }
                    { graph'''' with SymbolTable = st; Edges = methodRefEdge :: graph''''.Edges }
                | None ->
                    printfn "[BUILDER] Warning: Property access '%s' at %s has no symbol correlation" identText (range.ToString())
                    graph''''

            match receiverSymbol with
            | Some recSym ->
                { graph''''' with SymbolTable = Map.add recSym.DisplayName recSym graph'''''.SymbolTable }
            | None -> graph'''''

        else
            // Simple identifier or qualified name without property access
            let syntaxKind = sprintf "LongIdent:%s" identText
            let symbol = tryCorrelateSymbolWithContext range fileName syntaxKind context.CorrelationContext
            let longIdentNode = createNode syntaxKind range fileName symbol parentId

            let graph' = { graph with Nodes = Map.add longIdentNode.Id.Value longIdentNode graph.Nodes }
            let graph'' = addChildToParent longIdentNode.Id parentId graph'

            match symbol with
            | Some sym ->
                { graph'' with SymbolTable = Map.add sym.DisplayName sym graph''.SymbolTable }
            | None -> graph''

    | SynExpr.LetOrUse(isUse, _, bindings, body, range, _) ->
        let syntaxKind = if isUse then "LetOrUse:Use" else "LetOrUse:Let"
        let symbol = tryCorrelateSymbolWithContext range fileName syntaxKind context.CorrelationContext
        let letNode = createNodeWithKind syntaxKind (SKExpr ELetOrUse) range fileName symbol parentId

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

    | SynExpr.Sequential(_, _, expr1, expr2, range, _) ->
        // Create a Sequential node to wrap the sequence of expressions
        let seqNode = createNodeWithKind "Sequential" (SKExpr ESequential) range fileName None parentId
        let graph' = { graph with Nodes = Map.add seqNode.Id.Value seqNode graph.Nodes }
        let graph'' = addChildToParent seqNode.Id parentId graph'
        let graph''' = processExpression expr1 (Some seqNode.Id) fileName context graph''
        processExpression expr2 (Some seqNode.Id) fileName context graph'''

    | SynExpr.Const(constant, range) ->
        // Extract the constant value and create a clean SyntaxKind
        let syntaxKind, constValue =
            match constant with
            | SynConst.String(text, _, _) -> "Const:String", Some (StringValue text)
            | SynConst.Int32 i -> sprintf "Const:Int32 %d" i, Some (Int32Value i)
            | SynConst.Int64 i -> sprintf "Const:Int64 %d" i, Some (Int64Value i)
            | SynConst.Double d -> sprintf "Const:Float %f" d, Some (FloatValue d)
            | SynConst.Bool b -> sprintf "Const:Bool %b" b, Some (BoolValue b)
            | SynConst.Char c -> sprintf "Const:Char '%c'" c, Some (CharValue c)
            | SynConst.Byte b -> sprintf "Const:Byte %duy" b, Some (ByteValue b)
            | SynConst.Unit -> "Const:Unit", Some UnitValue
            | other -> sprintf "Const:%A" other, None  // Fallback for uncommon cases

        let constNode = createNodeWithKind syntaxKind (SKExpr EConst) range fileName None parentId
        // Set the ConstantValue on the node
        let constNodeWithValue = { constNode with ConstantValue = constValue }
        let graph' = { graph with Nodes = Map.add constNodeWithValue.Id.Value constNodeWithValue graph.Nodes }
        let graph'' = addChildToParent constNodeWithValue.Id parentId graph'

        // Register string literals in the PSG for later emission as globals
        match constant with
        | SynConst.String(text, _, _) ->
            let hash = uint32 (text.GetHashCode())
            { graph'' with StringLiterals = Map.add hash text graph''.StringLiterals }
        | _ -> graph''

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
        let whileNode = createNodeWithKind "WhileLoop" (SKExpr EWhileLoop) range fileName None parentId
        let graph' = { graph with Nodes = Map.add whileNode.Id.Value whileNode graph.Nodes }
        let graph'' = addChildToParent whileNode.Id parentId graph'

        let graph''' = processExpression condExpr (Some whileNode.Id) fileName context graph''
        processExpression bodyExpr (Some whileNode.Id) fileName context graph'''

    // Mutable variable assignment (counter <- counter + 1)
    | SynExpr.LongIdentSet(longDotId, rhsExpr, range) ->
        let varName = longDotId.LongIdent |> List.map (fun id -> id.idText) |> String.concat "."
        let syntaxKind = sprintf "MutableSet:%s" varName
        // Look up the symbol from the symbol table to enable def-use edge creation
        let symbol = Map.tryFind varName graph.SymbolTable
        let setNode = createNodeWithKind syntaxKind (SKExpr EMutableSet) range fileName symbol parentId
        let graph' = { graph with Nodes = Map.add setNode.Id.Value setNode graph.Nodes }
        let graph'' = addChildToParent setNode.Id parentId graph'
        processExpression rhsExpr (Some setNode.Id) fileName context graph''

    // For loops (for i = start to end do ...)
    | SynExpr.For(_, _, ident, _, startExpr, _, endExpr, bodyExpr, range) ->
        let syntaxKind = sprintf "ForLoop:%s" ident.idText
        let forNode = createNodeWithKind syntaxKind (SKExpr EForLoop) range fileName None parentId
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
    // Extract the member name from the trait signature for proper emission
    | SynExpr.TraitCall(supportTys, traitSig, argExpr, range) ->
        // Extract member name from the trait signature
        let memberName =
            match traitSig with
            | SynMemberSig.Member(SynValSig(_, SynIdent(ident, _), _, _, _, _, _, _, _, _, _, _), _, _, _) ->
                Some ident.idText
            | _ -> None

        let syntaxKind =
            match memberName with
            | Some name -> sprintf "TraitCall:%s" name
            | None -> "TraitCall"

        // Try to correlate with the resolved symbol from FCS
        let symbol = tryCorrelateSymbolWithContext range fileName syntaxKind context.CorrelationContext
        let traitNode = createNode syntaxKind range fileName symbol parentId
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

    // Indexed access (arr.[idx])
    | SynExpr.DotIndexedGet(objectExpr, indexArgs, dotRange, range) ->
        let indexNode = createNode "DotIndexedGet" range fileName None parentId
        let graph' = { graph with Nodes = Map.add indexNode.Id.Value indexNode graph.Nodes }
        let graph'' = addChildToParent indexNode.Id parentId graph'
        // Process the object being indexed
        let graph''' = processExpression objectExpr (Some indexNode.Id) fileName context graph''
        // Process the index argument
        let graph'''' = processExpression indexArgs (Some indexNode.Id) fileName context graph'''
        graph''''

    // Indexed set (arr.[idx] <- value)
    | SynExpr.DotIndexedSet(objectExpr, indexArgs, valueExpr, leftOfSetRange, dotRange, range) ->
        let setNode = createNode "DotIndexedSet" range fileName None parentId
        let graph' = { graph with Nodes = Map.add setNode.Id.Value setNode graph.Nodes }
        let graph'' = addChildToParent setNode.Id parentId graph'
        // Process the object being indexed
        let graph''' = processExpression objectExpr (Some setNode.Id) fileName context graph''
        // Process the index argument
        let graph'''' = processExpression indexArgs (Some setNode.Id) fileName context graph'''
        // Process the value being assigned
        let graph''''' = processExpression valueExpr (Some setNode.Id) fileName context graph''''
        graph'''''

    // Interpolated strings ($"Hello, {name}!")
    | SynExpr.InterpolatedString(parts, _, range) ->
        let interpNode = createNode "InterpolatedString" range fileName None parentId
        let graph' = { graph with Nodes = Map.add interpNode.Id.Value interpNode graph.Nodes }
        let graph'' = addChildToParent interpNode.Id parentId graph'

        // Process each part - strings are literals, FillExpr contains expressions to interpolate
        parts |> List.fold (fun graphAcc part ->
            match part with
            | SynInterpolatedStringPart.String(text, range) ->
                // Include the string content in the SyntaxKind and add to StringLiterals
                let hash = uint32 (hash text)
                let strPartNode = createNode (sprintf "InterpolatedStringPart:String:%u" hash) range fileName None (Some interpNode.Id)
                let graphAcc' =
                    { graphAcc with
                        Nodes = Map.add strPartNode.Id.Value strPartNode graphAcc.Nodes
                        StringLiterals = Map.add hash text graphAcc.StringLiterals }
                addChildToParent strPartNode.Id (Some interpNode.Id) graphAcc'
            | SynInterpolatedStringPart.FillExpr(fillExpr, qualifiers) ->
                let fillNode = createNode "InterpolatedStringPart:Fill" fillExpr.Range fileName None (Some interpNode.Id)
                let graphAcc' = { graphAcc with Nodes = Map.add fillNode.Id.Value fillNode graphAcc.Nodes }
                let graphAcc'' = addChildToParent fillNode.Id (Some interpNode.Id) graphAcc'
                processExpression fillExpr (Some fillNode.Id) fileName context graphAcc''
        ) graph''

    // Record expressions ({ Field1 = value1; Field2 = value2 } or { existingRecord with Field = newValue })
    | SynExpr.Record(baseInfo, copyInfo, recordFields, range) ->
        // Determine if this is a copy-update expression or a new record
        let syntaxKind =
            match copyInfo with
            | Some _ -> "Record:CopyUpdate"
            | None -> "Record:New"

        let recordNode = createNode syntaxKind range fileName None parentId
        let graph' = { graph with Nodes = Map.add recordNode.Id.Value recordNode graph.Nodes }
        let graph'' = addChildToParent recordNode.Id parentId graph'

        // Process copy source if this is a copy-update expression
        let graph''' =
            match copyInfo with
            | Some (copyExpr, _) ->
                processExpression copyExpr (Some recordNode.Id) fileName context graph''
            | None -> graph''

        // Process each field assignment
        // FCS 43.9.300: SynExprRecordField has 4 fields: fieldName, equalsRange, expr, blockSeparator
        // RecordFieldName = SynLongIdent * bool
        let graph'''' =
            recordFields
            |> List.fold (fun graphAcc (SynExprRecordField(fieldName, _equalsRange, exprOpt, _blockSep)) ->
                // Extract field name from RecordFieldName (which is SynLongIdent * bool)
                let (fieldIdent, _) = fieldName
                let fieldNameStr =
                    fieldIdent.LongIdent
                    |> List.map (fun id -> id.idText)
                    |> String.concat "."

                // Use the field identifier's range for the node
                let fieldRange = fieldIdent.Range

                // Create a node for the field assignment
                let fieldKind = sprintf "RecordField:%s" fieldNameStr
                let fieldNode = createNode fieldKind fieldRange fileName None (Some recordNode.Id)
                let graphAcc' = { graphAcc with Nodes = Map.add fieldNode.Id.Value fieldNode graphAcc.Nodes }
                let graphAcc'' = addChildToParent fieldNode.Id (Some recordNode.Id) graphAcc'

                // Process the field value expression if present
                match exprOpt with
                | Some fieldExpr ->
                    processExpression fieldExpr (Some fieldNode.Id) fileName context graphAcc''
                | None ->
                    // Field without expression (shorthand syntax: { Name } instead of { Name = Name })
                    graphAcc''
            ) graph'''

        graph''''

    // Do expressions (do expr - used for side effects)
    | SynExpr.Do(expr, range) ->
        let doNode = createNode "Do" range fileName None parentId
        let graph' = { graph with Nodes = Map.add doNode.Id.Value doNode graph.Nodes }
        let graph'' = addChildToParent doNode.Id parentId graph'
        processExpression expr (Some doNode.Id) fileName context graph''

    // Assert expressions (assert expr)
    | SynExpr.Assert(expr, range) ->
        let assertNode = createNode "Assert" range fileName None parentId
        let graph' = { graph with Nodes = Map.add assertNode.Id.Value assertNode graph.Nodes }
        let graph'' = addChildToParent assertNode.Id parentId graph'
        processExpression expr (Some assertNode.Id) fileName context graph''

    // Lazy expressions (lazy expr)
    | SynExpr.Lazy(expr, range) ->
        let lazyNode = createNode "Lazy" range fileName None parentId
        let graph' = { graph with Nodes = Map.add lazyNode.Id.Value lazyNode graph.Nodes }
        let graph'' = addChildToParent lazyNode.Id parentId graph'
        processExpression expr (Some lazyNode.Id) fileName context graph''

    // Type test expressions (expr :? Type)
    | SynExpr.TypeTest(expr, targetType, range) ->
        let typeTestNode = createNode "TypeTest" range fileName None parentId
        let graph' = { graph with Nodes = Map.add typeTestNode.Id.Value typeTestNode graph.Nodes }
        let graph'' = addChildToParent typeTestNode.Id parentId graph'
        processExpression expr (Some typeTestNode.Id) fileName context graph''

    // Upcast expressions (expr :> Type)
    | SynExpr.Upcast(expr, targetType, range) ->
        let upcastNode = createNode "Upcast" range fileName None parentId
        let graph' = { graph with Nodes = Map.add upcastNode.Id.Value upcastNode graph.Nodes }
        let graph'' = addChildToParent upcastNode.Id parentId graph'
        processExpression expr (Some upcastNode.Id) fileName context graph''

    // Downcast expressions (expr :?> Type)
    | SynExpr.Downcast(expr, targetType, range) ->
        let downcastNode = createNode "Downcast" range fileName None parentId
        let graph' = { graph with Nodes = Map.add downcastNode.Id.Value downcastNode graph.Nodes }
        let graph'' = addChildToParent downcastNode.Id parentId graph'
        processExpression expr (Some downcastNode.Id) fileName context graph''

    // Inferred upcast (upcast expr)
    | SynExpr.InferredUpcast(expr, range) ->
        let inferredUpcastNode = createNode "InferredUpcast" range fileName None parentId
        let graph' = { graph with Nodes = Map.add inferredUpcastNode.Id.Value inferredUpcastNode graph.Nodes }
        let graph'' = addChildToParent inferredUpcastNode.Id parentId graph'
        processExpression expr (Some inferredUpcastNode.Id) fileName context graph''

    // Inferred downcast (downcast expr)
    | SynExpr.InferredDowncast(expr, range) ->
        let inferredDowncastNode = createNode "InferredDowncast" range fileName None parentId
        let graph' = { graph with Nodes = Map.add inferredDowncastNode.Id.Value inferredDowncastNode graph.Nodes }
        let graph'' = addChildToParent inferredDowncastNode.Id parentId graph'
        processExpression expr (Some inferredDowncastNode.Id) fileName context graph''

    // Hard stop on unhandled expressions
    | other ->
        let exprTypeName = other.GetType().Name
        let range = other.Range
        failwithf "[BUILDER] ERROR: Unhandled expression type '%s' at %s in file %s. PSG construction cannot continue with unknown AST nodes."
            exprTypeName (range.ToString()) fileName
