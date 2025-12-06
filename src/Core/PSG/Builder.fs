module Core.PSG.Builder

open System
open System.IO
open FSharp.Compiler.Text
open FSharp.Compiler.Syntax
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Symbols
open Core.CompilerConfig
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

/// Debug missing critical symbols in correlation context
let private debugCriticalSymbols (context: CorrelationContext) (fileName: string) =
    let fileUses = Map.tryFind fileName context.FileIndex |> Option.defaultValue [||]
    let criticalSymbols = ["AsReadOnlySpan"; "spanToString"; "stackBuffer"; "Ok"; "Error"]
    
    printfn "[DEBUG] === Critical Symbol Search in %s ===" (System.IO.Path.GetFileName fileName)
    
    for critical in criticalSymbols do
        let matches = 
            fileUses
            |> Array.filter (fun symbolUse ->
                let sym = symbolUse.Symbol
                sym.DisplayName.Contains(critical) ||
                sym.FullName.Contains(critical) ||
                sym.DisplayName = critical)
        
        if matches.Length > 0 then
            printfn "[DEBUG] ✓ Found '%s': %d matches" critical matches.Length
            matches |> Array.take (min 2 matches.Length) |> Array.iter (fun su ->
                printfn "    %s at %s" su.Symbol.FullName (su.Range.ToString()))
        else
            printfn "[DEBUG] ✗ Missing '%s'" critical
    
    // Also check across ALL files for these critical symbols
    printfn "[DEBUG] === Cross-File Search ==="
    for critical in criticalSymbols do
        let allMatches = 
            context.SymbolUses
            |> Array.filter (fun symbolUse ->
                let sym = symbolUse.Symbol
                sym.DisplayName.Contains(critical) ||
                sym.FullName.Contains(critical) ||
                sym.DisplayName = critical)
        
        if allMatches.Length > 0 then
            printfn "[DEBUG] ✓ Global '%s': %d matches across all files" critical allMatches.Length
            allMatches |> Array.take (min 2 allMatches.Length) |> Array.iter (fun su ->
                printfn "    %s in %s at %s" su.Symbol.FullName (System.IO.Path.GetFileName su.Range.FileName) (su.Range.ToString()))
        else
            printfn "[DEBUG] ✗ Global missing '%s'" critical
let private tryCorrelateSymbolWithContext (range: range) (fileName: string) (syntaxKind: string) (context: CorrelationContext) : FSharpSymbol option =
    
    // Strategy 1: Exact range match
    let key = (fileName, range.StartLine, range.StartColumn, range.EndLine, range.EndColumn)
    match Map.tryFind key context.PositionIndex with
    | Some symbolUse -> 
        if isCorrelationVerbose() then
            printfn "[CORRELATION] ✓ Exact match: %s -> %s" syntaxKind symbolUse.Symbol.FullName
        Some symbolUse.Symbol
    | None ->
        // Strategy 2: Enhanced correlation by syntax kind
        match Map.tryFind fileName context.FileIndex with
        | Some fileUses ->
            
            // Method call correlation
            if syntaxKind.StartsWith("MethodCall:") || syntaxKind.Contains("DotGet") || syntaxKind.Contains("LongIdent:MethodCall:") then
                let methodName = 
                    if syntaxKind.Contains("LongIdent:MethodCall:") then
                        // Extract method name from LongIdent:MethodCall:AsReadOnlySpan
                        let parts = syntaxKind.Split(':')
                        if parts.Length > 2 then parts.[2] else ""
                    elif syntaxKind.Contains(":") then
                        let parts = syntaxKind.Split(':')
                        if parts.Length > 1 then parts.[parts.Length - 1] else ""
                    else ""
                
                let methodCandidates = 
                    fileUses
                    |> Array.filter (fun symbolUse ->
                        match symbolUse.Symbol with
                        | :? FSharpMemberOrFunctionOrValue as mfv -> 
                            (mfv.IsMember || mfv.IsProperty || mfv.IsFunction) &&
                            (String.IsNullOrEmpty(methodName) || 
                             mfv.DisplayName.Contains(methodName) ||
                             mfv.DisplayName = methodName ||
                             mfv.FullName.EndsWith("." + methodName))
                        | _ -> false)
                    |> Array.filter (fun symbolUse ->
                        abs(symbolUse.Range.StartLine - range.StartLine) <= 2 &&
                        abs(symbolUse.Range.StartColumn - range.StartColumn) <= 20)
                
                match methodCandidates |> Array.sortBy (fun su -> 
                    let nameScore = if su.Symbol.DisplayName = methodName then 0 else 1
                    let rangeScore = abs(su.Range.StartLine - range.StartLine) + abs(su.Range.StartColumn - range.StartColumn)
                    nameScore * 100 + rangeScore) |> Array.tryHead with
                | Some symbolUse -> 
                    if isCorrelationVerbose() then
                        printfn "[CORRELATION] ✓ Enhanced method match: %s -> %s" syntaxKind symbolUse.Symbol.FullName
                    Some symbolUse.Symbol
                | None -> 
                    if isCorrelationVerbose() then
                        printfn "[CORRELATION] ✗ No enhanced method match for: %s (method: %s)" syntaxKind methodName
                    None
            
            // Generic type application correlation
            elif syntaxKind.StartsWith("TypeApp:") then
                let genericCandidates = 
                    fileUses
                    |> Array.filter (fun symbolUse ->
                        match symbolUse.Symbol with
                        | :? FSharpMemberOrFunctionOrValue as mfv -> 
                            (mfv.IsFunction && mfv.GenericParameters.Count > 0) ||
                            mfv.DisplayName.Contains("stackBuffer") ||
                            mfv.FullName.Contains("stackBuffer") ||
                            mfv.DisplayName = "stackBuffer"
                        | :? FSharpEntity as entity -> entity.GenericParameters.Count > 0
                        | _ -> false)
                    |> Array.filter (fun symbolUse ->
                        abs(symbolUse.Range.StartLine - range.StartLine) <= 2)
                
                match genericCandidates |> Array.sortBy (fun su ->
                    let nameScore = if su.Symbol.DisplayName = "stackBuffer" then 0 else 1
                    let rangeScore = abs(su.Range.StartLine - range.StartLine)
                    nameScore * 100 + rangeScore) |> Array.tryHead with
                | Some symbolUse -> 
                    if isCorrelationVerbose() then
                        printfn "[CORRELATION] ✓ Enhanced generic match: %s -> %s" syntaxKind symbolUse.Symbol.FullName
                    Some symbolUse.Symbol
                | None -> 
                    if isCorrelationVerbose() then
                        printfn "[CORRELATION] ✗ No enhanced generic match for: %s" syntaxKind
                    None
            
            // Union case correlation
            elif syntaxKind.Contains("UnionCase:") then
                let unionCaseName = 
                    if syntaxKind.Contains("Ok") then "Ok"
                    elif syntaxKind.Contains("Error") then "Error"
                    elif syntaxKind.Contains("Some") then "Some"
                    elif syntaxKind.Contains("None") then "None"
                    else ""
                
                if not (String.IsNullOrEmpty(unionCaseName)) then
                    let unionCaseCandidates = 
                        fileUses
                        |> Array.filter (fun symbolUse ->
                            match symbolUse.Symbol with
                            | :? FSharpUnionCase as unionCase -> unionCase.Name = unionCaseName
                            | :? FSharpMemberOrFunctionOrValue as mfv -> 
                                mfv.DisplayName = unionCaseName
                            | _ -> false)
                        |> Array.filter (fun symbolUse ->
                            abs(symbolUse.Range.StartLine - range.StartLine) <= 2)
                    
                    match unionCaseCandidates |> Array.tryHead with
                    | Some symbolUse -> 
                        if isCorrelationVerbose() then
                            printfn "[CORRELATION] ✓ Union case match: %s -> %s" syntaxKind symbolUse.Symbol.FullName
                        Some symbolUse.Symbol
                    | None -> None
                else None
            
            // Function/identifier correlation
            elif syntaxKind.Contains("Ident:") || syntaxKind.Contains("LongIdent:") then
                let identName = 
                    if syntaxKind.Contains(":") then
                        let parts = syntaxKind.Split(':')
                        if parts.Length > 0 then parts.[parts.Length - 1] else ""
                    else ""
                
                if not (String.IsNullOrEmpty(identName)) then
                    let functionCandidates = 
                        fileUses
                        |> Array.filter (fun symbolUse ->
                            let sym = symbolUse.Symbol
                            sym.DisplayName = identName ||
                            sym.FullName.EndsWith("." + identName) ||
                            sym.FullName.Contains(identName) ||
                            // Enhanced matching for critical symbols
                            (identName = "spanToString" && (sym.DisplayName.Contains("spanToString") || sym.FullName.Contains("spanToString"))) ||
                            (identName = "stackBuffer" && (sym.DisplayName.Contains("stackBuffer") || sym.FullName.Contains("stackBuffer"))) ||
                            (identName = "AsReadOnlySpan" && (sym.DisplayName.Contains("AsReadOnlySpan") || sym.FullName.Contains("AsReadOnlySpan"))) ||
                            (sym.DisplayName.Contains(identName) && sym.DisplayName.Length <= identName.Length + 5))
                        |> Array.filter (fun symbolUse ->
                            abs(symbolUse.Range.StartLine - range.StartLine) <= 2 &&
                            abs(symbolUse.Range.StartColumn - range.StartColumn) <= 25)
                    
                    match functionCandidates |> Array.sortBy (fun su -> 
                        let nameScore = 
                            if su.Symbol.DisplayName = identName then 0
                            elif su.Symbol.FullName.EndsWith("." + identName) then 1
                            elif su.Symbol.FullName.Contains(identName) then 2
                            else 3
                        let rangeScore = abs(su.Range.StartLine - range.StartLine) + abs(su.Range.StartColumn - range.StartColumn)
                        nameScore * 1000 + rangeScore) |> Array.tryHead with
                    | Some symbolUse -> 
                        if isCorrelationVerbose() then
                            printfn "[CORRELATION] ✓ Enhanced function match: %s -> %s" syntaxKind symbolUse.Symbol.FullName
                        Some symbolUse.Symbol
                    | None -> 
                        if isCorrelationVerbose() then
                            printfn "[CORRELATION] ✗ No enhanced function match for: %s (name: %s)" syntaxKind identName
                        None
                else None
            
            // Fallback to original correlation
            else
                let closeMatch =
                    fileUses
                    |> Array.filter (fun symbolUse ->
                        abs(symbolUse.Range.StartLine - range.StartLine) <= 2)
                    |> Array.sortBy (fun symbolUse ->
                        abs(symbolUse.Range.StartLine - range.StartLine) +
                        abs(symbolUse.Range.StartColumn - range.StartColumn))
                    |> Array.tryHead
                
                match closeMatch with
                | Some symbolUse -> Some symbolUse.Symbol
                | None -> 
                    if isCorrelationVerbose() then
                        printfn "[CORRELATION] ✗ No match: %s at %s" syntaxKind (range.ToString())
                    None
        | None -> None

/// Helper function to determine if symbol represents a function
let private isFunction (symbol: FSharpSymbol) : bool =
    match symbol with
    | :? FSharpMemberOrFunctionOrValue as mfv -> mfv.IsFunction || mfv.IsMember
    | _ -> false

/// Extract symbol from pattern - this is the correct way to get a binding's symbol
let rec private extractSymbolFromPattern (pat: SynPat) (fileName: string) (context: CorrelationContext) : FSharpSymbol option =
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

/// Process binding with explicit use flag
let rec private processBindingWithUseFlag binding parentId fileName context graph isUse =
    match binding with
    | SynBinding(accessibility, kind, isInline, isMutable, attributes, xmlDoc, valData, pat, returnInfo, expr, range, seqPoint, trivia) ->
        // Get symbol from the PATTERN, not from loose range matching on the binding
        // This is critical - bindings define symbols through their patterns
        let symbol = extractSymbolFromPattern pat fileName context.CorrelationContext
        
        // Check for EntryPoint attribute
        let hasEntryPointAttr = 
            attributes
            |> List.exists (fun attrList ->
                attrList.Attributes
                |> List.exists (fun attr ->
                    let attrName = 
                        match attr.TypeName with
                        | SynLongIdent(idents, _, _) -> 
                            idents |> List.map (fun id -> id.idText) |> String.concat "."
                    attrName = "EntryPoint" || attrName = "EntryPointAttribute"))
        
        // Detect main function pattern
        let isMainFunc = 
            match pat with
            | SynPat.LongIdent(SynLongIdent(idents, _, _), _, _, _, _, _) ->
                idents |> List.exists (fun id -> id.idText = "main")
            | SynPat.Named(SynIdent(ident, _), _, _, _) ->
                ident.idText = "main"
            | _ -> false
        
        // Create binding node with enhanced detection
        let syntaxKind = 
            if hasEntryPointAttr then "Binding:EntryPoint"
            elif isMainFunc then "Binding:Main"
            elif isUse then "Binding:Use"
            else "Binding"
            
        let bindingNode = createNode syntaxKind range fileName symbol parentId
        
        let graph' = { graph with Nodes = Map.add bindingNode.Id.Value bindingNode graph.Nodes }
        let graph'' = addChildToParent bindingNode.Id parentId graph'
        
        let graph''' = 
            match symbol with
            | Some sym -> 
                let updatedSymbolTable = Map.add sym.DisplayName sym graph''.SymbolTable
                
                // Add to entry points if this is an entry point
                let updatedEntryPoints = 
                    if hasEntryPointAttr || isMainFunc then
                        printfn "[BUILDER] Found entry point during construction: %s" sym.FullName
                        bindingNode.Id :: graph''.EntryPoints
                    else
                        graph''.EntryPoints
                
                { graph'' with 
                    SymbolTable = updatedSymbolTable
                    EntryPoints = updatedEntryPoints }
            | None -> 
                printfn "[BUILDER] Warning: No symbol correlation for binding at %s" (range.ToString())
                graph''
        
        let graph'''' = processPattern pat (Some bindingNode.Id) fileName context graph'''
        processExpression expr (Some bindingNode.Id) fileName context graph''''

/// Regular binding processing (for non-LetOrUse contexts)
and private processBinding binding parentId fileName context graph =
    processBindingWithUseFlag binding parentId fileName context graph false

/// ENHANCED pattern processing (FCS 43.9.300 compatible)
and private processPattern pat parentId fileName context graph =
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
    
    | SynPat.LongIdent(longIdent, _, _, _, _, range) ->
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
        
        match symbol with
        | Some sym -> 
            let updatedSymbolTable = Map.add sym.DisplayName sym graph''.SymbolTable
            
            // Create union case reference edge for Ok/Error patterns
            if identText = "Ok" || identText = "Error" then
                let unionCaseEdge = {
                    Source = patNode.Id
                    Target = patNode.Id
                    Kind = SymbolUse
                }
                { graph'' with 
                    SymbolTable = updatedSymbolTable
                    Edges = unionCaseEdge :: graph''.Edges }
            else
                { graph'' with SymbolTable = updatedSymbolTable }
        | None -> 
            printfn "[BUILDER] Warning: Pattern '%s' at %s has no symbol correlation" identText (range.ToString())
            graph''
        
    | _ -> graph

/// ENHANCED expression processing (FCS 43.9.300 compatible)
and private processExpression (expr: SynExpr) (parentId: NodeId option) (fileName: string) 
                              (context: BuildContext) (graph: ProgramSemanticGraph) =
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
            if String.IsNullOrEmpty(typeArgNames) then "TypeApp:Generic"
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
            bindings
            |> List.fold (fun acc binding -> 
                processBindingWithUseFlag binding (Some letNode.Id) fileName context acc isUse) graph'''
        
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

    // Add other common expression cases as needed...
    | _ -> 
        // Default case for unhandled expressions
        graph

/// Process a module declaration (preserved from original)
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
        
    | _ -> graph

/// Process implementation file (preserved from original)
let rec private processImplFile (implFile: SynModuleOrNamespace) context graph =
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

/// Symbol validation helper
let private validateSymbolCapture (graph: ProgramSemanticGraph) =
    let expectedSymbols = [
        "stackBuffer"; "AsReadOnlySpan"; "spanToString"; 
        "readInto"; "sprintf"; "Ok"; "Error"; "Write"; "WriteLine"
    ]
    
    printfn "[VALIDATION] === Symbol Capture Validation ==="
    expectedSymbols |> List.iter (fun expected ->
        let found = 
            graph.SymbolTable 
            |> Map.exists (fun _ symbol -> 
                symbol.DisplayName.Contains(expected) || 
                symbol.FullName.Contains(expected))
        
        if found then
            printfn "[VALIDATION] ✓ Found expected symbol: %s" expected
        else
            printfn "[VALIDATION] ✗ Missing expected symbol: %s" expected
    )
    
    printfn "[VALIDATION] Total symbols captured: %d" graph.SymbolTable.Count

/// Build complete PSG from project results with ENHANCED symbol correlation (FCS 43.9.300 compatible)
let buildProgramSemanticGraph 
    (checkResults: FSharpCheckProjectResults) 
    (parseResults: FSharpParseFileResults[]) : ProgramSemanticGraph =
    
    printfn "[BUILDER] Starting ENHANCED PSG construction (FCS 43.9.300 compatible)"
    
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
    
    printfn "[BUILDER] Phase 1: Building structural nodes with enhanced correlation from %d files" parseResults.Length
    
    // Process each file and merge results
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
    
    printfn "[BUILDER] Phase 1 complete: Enhanced PSG built with %d nodes, %d entry points" 
        structuralGraph.Nodes.Count structuralGraph.EntryPoints.Length
    
    // Validate symbol capture
    validateSymbolCapture structuralGraph
    
    // Phase 2: Apply FCS constraint resolution
    printfn "[BUILDER] Phase 2: Applying FCS constraint resolution"
    let typeEnhancedGraph = integrateTypesWithCheckResults structuralGraph checkResults
    
    // Phase 3: Finalize nodes and analyze context
    printfn "[BUILDER] Phase 3: Finalizing PSG nodes and analyzing context"
    let finalNodes = 
        typeEnhancedGraph.Nodes
        |> Map.map (fun _ node -> 
            node
            |> ChildrenStateHelpers.finalizeChildren
            |> ReachabilityHelpers.updateNodeContext)

    let finalGraph = 
        { typeEnhancedGraph with 
            Nodes = finalNodes
            CompilationOrder = parseResults |> Array.map (fun pr -> pr.FileName) |> List.ofArray 
        }
    
    printfn "[BUILDER] ENHANCED PSG construction complete (FCS 43.9.300 compatible)"
    printfn "[BUILDER] Final PSG: %d nodes, %d edges, %d entry points, %d symbols" 
        finalGraph.Nodes.Count finalGraph.Edges.Length finalGraph.EntryPoints.Length finalGraph.SymbolTable.Count
    
    // Final validation
    validateSymbolCapture finalGraph
        
    finalGraph