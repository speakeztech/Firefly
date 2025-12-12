module Core.PSG.Correlation

open System
open FSharp.Compiler.Symbols
open FSharp.Compiler.Text
open FSharp.Compiler.Syntax
open FSharp.Compiler.CodeAnalysis
open Core.CompilerConfig
open Core.PSG.Types

/// Correlation context for PSG building
type CorrelationContext = {
    SymbolUses: FSharpSymbolUse[]
    PositionIndex: Map<(string * int * int * int * int), FSharpSymbolUse>
    FileIndex: Map<string, FSharpSymbolUse[]>
}

/// Check if a range contains a position
let rangeContainsPosition (range: range) (pos: pos) =
    range.StartLine <= pos.Line && 
    pos.Line <= range.EndLine &&
    (pos.Line > range.StartLine || pos.Column >= range.StartColumn) &&
    (pos.Line < range.EndLine || pos.Column <= range.EndColumn)

/// Check if two ranges overlap
let rangesOverlap (r1: range) (r2: range) =
    r1.FileName = r2.FileName &&
    r1.StartLine <= r2.EndLine &&
    r2.StartLine <= r1.EndLine &&
    (r1.StartLine <> r2.EndLine || r1.StartColumn <= r2.EndColumn) &&
    (r2.StartLine <> r1.EndLine || r2.StartColumn <= r1.EndColumn)

/// Create correlation context from FCS results
let createContext (checkResults: FSharpCheckProjectResults) =
    let allSymbolUses = checkResults.GetAllUsesOfAllSymbols() |> Array.ofSeq

    let positionIndex = 
        allSymbolUses
        |> Array.map (fun symbolUse ->
            let r = symbolUse.Range
            let key = (r.FileName, r.StartLine, r.StartColumn, r.EndLine, r.EndColumn)
            (key, symbolUse)
        )
        |> Map.ofArray
    
    let fileIndex =
        allSymbolUses
        |> Array.groupBy (fun symbolUse -> symbolUse.Range.FileName)
        |> Map.ofArray
    
    {
        SymbolUses = allSymbolUses
        PositionIndex = positionIndex
        FileIndex = fileIndex
    }

/// Enhanced correlation debugging for specific missing symbols (no-op in production)
let debugMissingSymbols (_context: CorrelationContext) (_fileName: string) = ()

/// Enhanced symbol correlation with syntax context and specialized matching
let tryCorrelateSymbolEnhanced (range: range) (fileName: string) (syntaxKind: string) (context: CorrelationContext) : FSharpSymbol option =
    
    // Strategy 1: Exact range match (existing)
    let key = (fileName, range.StartLine, range.StartColumn, range.EndLine, range.EndColumn)
    match Map.tryFind key context.PositionIndex with
    | Some symbolUse -> 
        if isCorrelationVerbose() then
            printfn "[CORRELATION] ✓ Exact match: %s -> %s" syntaxKind symbolUse.Symbol.FullName
        Some symbolUse.Symbol
    | None ->
        // Strategy 2: Enhanced range-based search with syntax-aware filtering
        match Map.tryFind fileName context.FileIndex with
        | Some fileUses ->
            
            // Strategy 2a: Method call specific correlation
            if syntaxKind.StartsWith("MethodCall:") || syntaxKind.Contains("DotGet") then
                let methodName = 
                    if syntaxKind.Contains(":") then
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
                             mfv.DisplayName = methodName)
                        | _ -> false)
                    |> Array.filter (fun symbolUse ->
                        rangesOverlap symbolUse.Range range ||
                        (abs(symbolUse.Range.StartLine - range.StartLine) <= 1 &&
                         abs(symbolUse.Range.StartColumn - range.StartColumn) <= 15))
                
                match methodCandidates |> Array.sortBy (fun su -> 
                    abs(su.Range.StartLine - range.StartLine) + abs(su.Range.StartColumn - range.StartColumn)) |> Array.tryHead with
                | Some symbolUse -> 
                    if isCorrelationVerbose() then
                        printfn "[CORRELATION] ✓ Method match: %s -> %s" syntaxKind symbolUse.Symbol.FullName
                    Some symbolUse.Symbol
                | None -> 
                    if isCorrelationVerbose() then
                        printfn "[CORRELATION] ✗ No method match for: %s" syntaxKind
                    None
            
            // Strategy 2b: Generic type application correlation
            elif syntaxKind.StartsWith("TypeApp:") then
                let genericCandidates = 
                    fileUses
                    |> Array.filter (fun symbolUse ->
                        match symbolUse.Symbol with
                        | :? FSharpMemberOrFunctionOrValue as mfv -> 
                            (mfv.IsFunction && (mfv.GenericParameters.Count > 0 || mfv.DisplayName.Contains("stackBuffer"))) ||
                            mfv.DisplayName.Contains("stackBuffer")
                        | :? FSharpEntity as entity -> entity.GenericParameters.Count > 0
                        | _ -> false)
                    |> Array.filter (fun symbolUse ->
                        rangesOverlap symbolUse.Range range ||
                        (symbolUse.Range.StartLine = range.StartLine &&
                         abs(symbolUse.Range.StartColumn - range.StartColumn) <= 10))
                
                match genericCandidates |> Array.tryHead with
                | Some symbolUse -> 
                    if isCorrelationVerbose() then
                        printfn "[CORRELATION] ✓ Generic match: %s -> %s" syntaxKind symbolUse.Symbol.FullName
                    Some symbolUse.Symbol
                | None -> 
                    if isCorrelationVerbose() then
                        printfn "[CORRELATION] ✗ No generic match for: %s" syntaxKind
                    None
            
            // Strategy 2c: Union case constructor correlation (Ok, Error, Some, None)
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
                                mfv.DisplayName = unionCaseName ||
                                mfv.FullName.EndsWith("." + unionCaseName)
                            | :? FSharpEntity as entity -> 
                                entity.DisplayName = unionCaseName ||
                                entity.FullName.EndsWith("." + unionCaseName)
                            | _ -> false)
                        |> Array.filter (fun symbolUse ->
                            rangesOverlap symbolUse.Range range ||
                            (abs(symbolUse.Range.StartLine - range.StartLine) <= 2))
                    
                    match unionCaseCandidates |> Array.tryHead with
                    | Some symbolUse -> 
                        if isCorrelationVerbose() then
                            printfn "[CORRELATION] ✓ Union case match: %s -> %s" syntaxKind symbolUse.Symbol.FullName
                        Some symbolUse.Symbol
                    | None -> 
                        if isCorrelationVerbose() then
                            printfn "[CORRELATION] ✗ No union case match for: %s" syntaxKind
                        None
                else None
            
            // Strategy 2d: Function call correlation by name matching
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
                            symbolUse.Symbol.DisplayName = identName ||
                            symbolUse.Symbol.FullName.EndsWith("." + identName) ||
                            symbolUse.Symbol.FullName.Contains(identName) ||
                            (symbolUse.Symbol.DisplayName.Contains(identName) && 
                             symbolUse.Symbol.DisplayName.Length <= identName.Length + 5))
                        |> Array.filter (fun symbolUse ->
                            rangesOverlap symbolUse.Range range ||
                            (abs(symbolUse.Range.StartLine - range.StartLine) <= 1 &&
                             abs(symbolUse.Range.StartColumn - range.StartColumn) <= 15))
                    
                    match functionCandidates |> Array.sortBy (fun su -> 
                        let nameScore = if su.Symbol.DisplayName = identName then 0 else 1
                        let rangeScore = abs(su.Range.StartLine - range.StartLine) + abs(su.Range.StartColumn - range.StartColumn)
                        nameScore * 100 + rangeScore) |> Array.tryHead with
                    | Some symbolUse -> 
                        if isCorrelationVerbose() then
                            printfn "[CORRELATION] ✓ Function match: %s -> %s" syntaxKind symbolUse.Symbol.FullName
                        Some symbolUse.Symbol
                    | None -> 
                        if isCorrelationVerbose() then
                            printfn "[CORRELATION] ✗ No function match for: %s (name: %s)" syntaxKind identName
                        None
                else None
            
            // Strategy 3: Fallback to original logic for other cases
            else
                let containmentMatch = 
                    fileUses
                    |> Array.tryFind (fun symbolUse ->
                        let sr = symbolUse.Range
                        rangeContainsPosition sr range.Start &&
                        rangeContainsPosition sr range.End)
                
                match containmentMatch with
                | Some symbolUse -> 
                    if isCorrelationVerbose() then
                        printfn "[CORRELATION] ✓ Containment match: %s -> %s" syntaxKind symbolUse.Symbol.FullName
                    Some symbolUse.Symbol
                | None ->
                    let overlapMatch =
                        fileUses
                        |> Array.tryFind (fun symbolUse ->
                            rangesOverlap symbolUse.Range range)
                    
                    match overlapMatch with
                    | Some symbolUse -> 
                        if isCorrelationVerbose() then
                            printfn "[CORRELATION] ✓ Overlap match: %s -> %s" syntaxKind symbolUse.Symbol.FullName
                        Some symbolUse.Symbol
                    | None ->
                        let closeMatch =
                            fileUses
                            |> Array.filter (fun symbolUse ->
                                abs(symbolUse.Range.StartLine - range.StartLine) <= 2)
                            |> Array.sortBy (fun symbolUse ->
                                abs(symbolUse.Range.StartLine - range.StartLine) +
                                abs(symbolUse.Range.StartColumn - range.StartColumn))
                            |> Array.tryHead
                        
                        match closeMatch with
                        | Some symbolUse -> 
                            if isCorrelationVerbose() then
                                printfn "[CORRELATION] ✓ Close match: %s -> %s" syntaxKind symbolUse.Symbol.FullName
                            Some symbolUse.Symbol
                        | None -> 
                            if isCorrelationVerbose() then
                                printfn "[CORRELATION] ✗ No match: %s at %s" syntaxKind (range.ToString())
                            None
        | None -> 
            if isCorrelationVerbose() then
                printfn "[CORRELATION] ✗ No file index for: %s" fileName
            None

/// Try to correlate a syntax node with its symbol using enhanced matching (ORIGINAL API)
let tryCorrelateSymbol (range: range) (fileName: string) (context: CorrelationContext) : FSharpSymbol option =
    // For backward compatibility, use enhanced correlation with generic syntax kind
    tryCorrelateSymbolEnhanced range fileName "Generic" context

/// Enhanced correlation function that passes syntax context (NEW API)
let tryCorrelateSymbolWithContext (range: range) (fileName: string) (syntaxKind: string) (context: CorrelationContext) : FSharpSymbol option =
    
    // First try enhanced correlation
    match tryCorrelateSymbolEnhanced range fileName syntaxKind context with
    | Some symbol -> Some symbol
    | None ->
        // Fallback to original correlation for backward compatibility
        match tryCorrelateSymbol range fileName context with
        | Some symbol ->
            if isCorrelationVerbose() then
                printfn "[CORRELATION] ✓ Fallback correlation: %s -> %s" syntaxKind symbol.FullName
            Some symbol
        | None -> None

/// Visitor pattern for AST traversal with correlation
type CorrelationVisitor = {
    OnBinding: SynBinding -> range -> FSharpSymbol option -> unit
    OnExpression: SynExpr -> range -> FSharpSymbol option -> unit
    OnPattern: SynPat -> range -> FSharpSymbol option -> unit
    OnType: SynType -> range -> FSharpSymbol option -> unit
    OnMember: SynMemberDefn -> range -> FSharpSymbol option -> unit
}

/// Traverse AST with correlation
let rec traverseWithCorrelation (node: obj) (fileName: string) (context: CorrelationContext) (visitor: CorrelationVisitor) =
    match node with
    | :? SynBinding as binding ->
        let symbol = tryCorrelateSymbol binding.RangeOfBindingWithRhs fileName context
        visitor.OnBinding binding binding.RangeOfBindingWithRhs symbol
        
    | :? SynExpr as expr ->
        let symbol = tryCorrelateSymbol expr.Range fileName context
        visitor.OnExpression expr expr.Range symbol
        
    | :? SynPat as pat ->
        let symbol = tryCorrelateSymbol pat.Range fileName context
        visitor.OnPattern pat pat.Range symbol
        
    | :? SynType as synType ->
        let symbol = tryCorrelateSymbol synType.Range fileName context
        visitor.OnType synType synType.Range symbol
        
    | :? SynModuleDecl as moduleDecl ->
        match moduleDecl with
        | SynModuleDecl.Let(_, bindings, _) ->
            bindings |> List.iter (fun b -> traverseWithCorrelation b fileName context visitor)
        | SynModuleDecl.Types(typeDefs, _) ->
            typeDefs |> List.iter (fun td -> traverseWithCorrelation td fileName context visitor)
        | SynModuleDecl.Open(_, _) ->
            ()
        | SynModuleDecl.NestedModule(_, _, decls, _, _, _) ->
            decls |> List.iter (fun d -> traverseWithCorrelation d fileName context visitor)
        | _ -> ()
        
    | :? SynTypeDefn as typeDef ->
        let (SynTypeDefn(_, _, members, _, _, _)) = typeDef
        members |> List.iter (fun m -> traverseWithCorrelation m fileName context visitor)
        
    | :? SynMemberDefn as memberDef ->
        match memberDef with
        | SynMemberDefn.Member(binding, _) ->
            traverseWithCorrelation binding fileName context visitor
        | _ -> ()
        
    | _ -> ()

/// Build correlation map for a module
let correlateModule 
    (moduleDecl: SynModuleOrNamespace) 
    (fileName: string) 
    (context: CorrelationContext) =
    
    let correlations = ResizeArray<(range * FSharpSymbol)>()
    
    let visitor = {
        OnBinding = fun _ range symbol -> 
            symbol |> Option.iter (fun s -> correlations.Add(range, s))
        OnExpression = fun _ range symbol -> 
            symbol |> Option.iter (fun s -> correlations.Add(range, s))
        OnPattern = fun _ range symbol -> 
            symbol |> Option.iter (fun s -> correlations.Add(range, s))
        OnType = fun _ range symbol -> 
            symbol |> Option.iter (fun s -> correlations.Add(range, s))
        OnMember = fun _ range symbol -> 
            symbol |> Option.iter (fun s -> correlations.Add(range, s))
    }
    
    let (SynModuleOrNamespace(_, _, _, decls, _, _, _, _, _)) = moduleDecl
    decls |> List.iter (fun decl -> traverseWithCorrelation decl fileName context visitor)
    
    correlations.ToArray()

/// Find all correlations for a parsed file
let correlateFile 
    (parsedInput: ParsedInput) 
    (context: CorrelationContext) =
    
    match parsedInput with
    | ParsedInput.ImplFile(ParsedImplFileInput(fileName, _, _, _, _, modules, _, _, _)) ->
        if isCorrelationVerbose() then
            printfn "[Correlation] Processing file: %s with %d modules" 
                (System.IO.Path.GetFileName fileName) modules.Length
            
        let correlations = 
            modules
            |> List.map (fun m -> correlateModule m fileName context)
            |> Array.concat
            
        if isCorrelationVerbose() then
            printfn "[Correlation] Found %d correlations in %s" 
                correlations.Length (System.IO.Path.GetFileName fileName)
            
        correlations
    | ParsedInput.SigFile _ ->
        [||]

/// Generate correlation statistics for debugging
type CorrelationStats = {
    TotalSymbols: int
    CorrelatedNodes: int
    UncorrelatedNodes: int
    SymbolsByKind: Map<string, int>
    FileCoverage: Map<string, float>
}

let generateStats (context: CorrelationContext) (correlations: (range * FSharpSymbol)[]) =
    let symbolsByKind =
        context.SymbolUses
        |> Array.groupBy (fun su -> 
            match su.Symbol with
            | :? FSharpMemberOrFunctionOrValue -> "MemberOrFunctionOrValue"
            | :? FSharpEntity -> "Entity"
            | :? FSharpGenericParameter -> "GenericParameter"
            | :? FSharpParameter -> "Parameter"
            | :? FSharpActivePatternCase -> "ActivePatternCase"
            | :? FSharpUnionCase -> "UnionCase"
            | :? FSharpField -> "Field"
            | _ -> "Other"
        )
        |> Array.map (fun (kind, uses) -> kind, uses.Length)
        |> Map.ofArray
    
    let fileCoverage =
        context.FileIndex
        |> Map.map (fun fileName fileUses ->
            let correlatedCount = 
                correlations 
                |> Array.filter (fun (r, _) -> r.FileName = fileName) 
                |> Array.length
            float correlatedCount / float fileUses.Length * 100.0
        )
    
    {
        TotalSymbols = context.SymbolUses.Length
        CorrelatedNodes = correlations.Length
        UncorrelatedNodes = context.SymbolUses.Length - correlations.Length
        SymbolsByKind = symbolsByKind
        FileCoverage = fileCoverage
    }