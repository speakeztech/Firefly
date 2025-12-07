/// Pattern processing for PSG construction
module Core.PSG.Construction.PatternProcessing

open FSharp.Compiler.Text
open FSharp.Compiler.Syntax
open FSharp.Compiler.Symbols
open Core.PSG.Types
open Core.PSG.Correlation
open Core.PSG.Construction.Types
open Core.PSG.Construction.SymbolCorrelation

/// Extract symbol from pattern - this is the correct way to get a binding's symbol
let rec extractSymbolFromPattern (pat: SynPat) (fileName: string) (context: CorrelationContext) : FSharpSymbol option =
    match pat with
    | SynPat.Named(synIdent, _, _, range) ->
        let (SynIdent(ident, _)) = synIdent
        let syntaxKind = sprintf "Pattern:Named:%s" ident.idText
        tryCorrelateSymbolWithContext range fileName syntaxKind context
    | SynPat.LongIdent(longIdent, _, _, _, _, range) ->
        let (SynLongIdent(idents, _, _)) = longIdent
        let identText = idents |> List.map (fun id -> id.idText) |> String.concat "."
        let syntaxKind = sprintf "Pattern:LongIdent:%s" identText
        tryCorrelateSymbolWithContext range fileName syntaxKind context
    | SynPat.Paren(innerPat, _) ->
        extractSymbolFromPattern innerPat fileName context
    | SynPat.Typed(innerPat, _, _) ->
        extractSymbolFromPattern innerPat fileName context
    | _ -> None

/// Process a pattern node in the PSG
let rec processPattern (pat: SynPat) (parentId: NodeId option) (fileName: string)
                       (context: BuildContext) (graph: ProgramSemanticGraph) : ProgramSemanticGraph =
    match pat with
    | SynPat.Named(synIdent, _, _, range) ->
        let (SynIdent(ident, _)) = synIdent
        let syntaxKind = sprintf "Pattern:Named:%s" ident.idText
        let symbol = tryCorrelateSymbolWithContext range fileName syntaxKind context.CorrelationContext
        let patNode = createNode syntaxKind range fileName symbol parentId

        let graph' = { graph with Nodes = Map.add patNode.Id.Value patNode graph.Nodes }
        let graph'' = addChildToParent patNode.Id parentId graph'

        match symbol with
        | Some sym ->
            { graph'' with SymbolTable = Map.add sym.DisplayName sym graph''.SymbolTable }
        | None -> graph''

    | SynPat.LongIdent(longIdent, _, _, argPats, _, range) ->
        let (SynLongIdent(idents, _, _)) = longIdent
        let identText = idents |> List.map (fun id -> id.idText) |> String.concat "."

        // Detect union case patterns
        let syntaxKind =
            if identText = "Ok" || identText = "Error" || identText = "Some" || identText = "None" then
                sprintf "Pattern:UnionCase:%s" identText
            else
                sprintf "Pattern:LongIdent:%s" identText

        let symbol = tryCorrelateSymbolWithContext range fileName syntaxKind context.CorrelationContext
        let patNode = createNode syntaxKind range fileName symbol parentId

        let graph' = { graph with Nodes = Map.add patNode.Id.Value patNode graph.Nodes }
        let graph'' = addChildToParent patNode.Id parentId graph'

        // Process argument patterns (e.g., "length" in "Ok length")
        let graph''' =
            match argPats with
            | SynArgPats.Pats pats ->
                pats |> List.fold (fun g pat ->
                    processPattern pat (Some patNode.Id) fileName context g
                ) graph''
            | SynArgPats.NamePatPairs(pairs, _, _) ->
                pairs |> List.fold (fun g (pair: (Ident * range option * SynPat)) ->
                    let (_, _, pat) = pair
                    processPattern pat (Some patNode.Id) fileName context g
                ) graph''

        match symbol with
        | Some sym ->
            let updatedSymbolTable = Map.add sym.DisplayName sym graph'''.SymbolTable

            // Create union case reference edge for Ok/Error patterns
            if identText = "Ok" || identText = "Error" then
                let unionCaseEdge = {
                    Source = patNode.Id
                    Target = patNode.Id
                    Kind = SymbolUse
                }
                { graph''' with
                    SymbolTable = updatedSymbolTable
                    Edges = unionCaseEdge :: graph'''.Edges }
            else
                { graph''' with SymbolTable = updatedSymbolTable }
        | None ->
            printfn "[BUILDER] Warning: Pattern '%s' at %s has no symbol correlation" identText (range.ToString())
            graph'''

    // Parenthesized patterns - transparent, process inner
    | SynPat.Paren(innerPat, range) ->
        processPattern innerPat parentId fileName context graph

    // Typed patterns (pat : type) - process inner pattern
    | SynPat.Typed(innerPat, typeSig, range) ->
        processPattern innerPat parentId fileName context graph

    // Tuple patterns ((a, b, c))
    | SynPat.Tuple(isStruct, pats, commaRanges, range) ->
        let tupleNode = createNode "Pattern:Tuple" range fileName None parentId
        let graph' = { graph with Nodes = Map.add tupleNode.Id.Value tupleNode graph.Nodes }
        let graph'' = addChildToParent tupleNode.Id parentId graph'
        pats |> List.fold (fun g pat ->
            processPattern pat (Some tupleNode.Id) fileName context g
        ) graph''

    // Wildcard pattern (_)
    | SynPat.Wild(range) ->
        let wildNode = createNode "Pattern:Wildcard" range fileName None parentId
        let graph' = { graph with Nodes = Map.add wildNode.Id.Value wildNode graph.Nodes }
        addChildToParent wildNode.Id parentId graph'

    // Constant patterns (match with 0 | 1 | ...)
    | SynPat.Const(constant, range) ->
        let constNode = createNode "Pattern:Const" range fileName None parentId
        let graph' = { graph with Nodes = Map.add constNode.Id.Value constNode graph.Nodes }
        addChildToParent constNode.Id parentId graph'

    // Array or list patterns ([], [a; b], [||], [|a; b|])
    | SynPat.ArrayOrList(isArray, pats, range) ->
        let kind = if isArray then "Pattern:Array" else "Pattern:List"
        let arrNode = createNode kind range fileName None parentId
        let graph' = { graph with Nodes = Map.add arrNode.Id.Value arrNode graph.Nodes }
        let graph'' = addChildToParent arrNode.Id parentId graph'
        pats |> List.fold (fun g pat ->
            processPattern pat (Some arrNode.Id) fileName context g
        ) graph''

    // List cons pattern (head :: tail)
    | SynPat.ListCons(headPat, tailPat, range, trivia) ->
        let consNode = createNode "Pattern:ListCons" range fileName None parentId
        let graph' = { graph with Nodes = Map.add consNode.Id.Value consNode graph.Nodes }
        let graph'' = addChildToParent consNode.Id parentId graph'
        let graph''' = processPattern headPat (Some consNode.Id) fileName context graph''
        processPattern tailPat (Some consNode.Id) fileName context graph'''

    // As pattern (pat as name)
    | SynPat.As(lhsPat, rhsPat, range) ->
        let asNode = createNode "Pattern:As" range fileName None parentId
        let graph' = { graph with Nodes = Map.add asNode.Id.Value asNode graph.Nodes }
        let graph'' = addChildToParent asNode.Id parentId graph'
        let graph''' = processPattern lhsPat (Some asNode.Id) fileName context graph''
        processPattern rhsPat (Some asNode.Id) fileName context graph'''

    // Type test pattern (:? type)
    | SynPat.IsInst(typeSig, range) ->
        let isInstNode = createNode "Pattern:IsInst" range fileName None parentId
        let graph' = { graph with Nodes = Map.add isInstNode.Id.Value isInstNode graph.Nodes }
        addChildToParent isInstNode.Id parentId graph'

    // Null pattern
    | SynPat.Null(range) ->
        let nullNode = createNode "Pattern:Null" range fileName None parentId
        let graph' = { graph with Nodes = Map.add nullNode.Id.Value nullNode graph.Nodes }
        addChildToParent nullNode.Id parentId graph'

    // Hard stop on unhandled patterns
    | other ->
        let patTypeName = other.GetType().Name
        let range = other.Range
        failwithf "[BUILDER] ERROR: Unhandled pattern type '%s' at %s in file %s. PSG construction cannot continue with unknown AST nodes."
            patTypeName (range.ToString()) fileName
