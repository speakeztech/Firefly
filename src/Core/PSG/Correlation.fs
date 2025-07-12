module Core.PSG.Correlation

open System
open FSharp.Compiler.Symbols
open FSharp.Compiler.Text
open FSharp.Compiler.Syntax
open FSharp.Compiler.CodeAnalysis
open Core.PSG.Types

/// Correlation context for PSG building
type CorrelationContext = {
    SymbolUses: FSharpSymbolUse[]
    PositionIndex: Map<(string * int * int * int * int), FSharpSymbolUse>
    FileIndex: Map<string, FSharpSymbolUse[]>
}

/// Check if a range contains a position
let private rangeContainsPosition (range: range) (pos: pos) =
    range.StartLine <= pos.Line && 
    pos.Line <= range.EndLine &&
    (pos.Line > range.StartLine || pos.Column >= range.StartColumn) &&
    (pos.Line < range.EndLine || pos.Column <= range.EndColumn)

/// Check if two ranges overlap
let private rangesOverlap (r1: range) (r2: range) =
    r1.FileName = r2.FileName &&
    r1.StartLine <= r2.EndLine &&
    r2.StartLine <= r1.EndLine &&
    (r1.StartLine <> r2.EndLine || r1.StartColumn <= r2.EndColumn) &&
    (r2.StartLine <> r1.EndLine || r2.StartColumn <= r1.EndColumn)

/// Create correlation context from FCS results
let createContext (checkResults: FSharpCheckProjectResults) =
    let allSymbolUses = checkResults.GetAllUsesOfAllSymbols() |> Array.ofSeq
    
    printfn "[Correlation] Found %d total symbol uses" allSymbolUses.Length
    printfn "[Correlation] Definition count: %d" 
        (allSymbolUses |> Array.filter (fun su -> su.IsFromDefinition) |> Array.length)
    printfn "[Correlation] Use count: %d" 
        (allSymbolUses |> Array.filter (fun su -> su.IsFromUse) |> Array.length)
    
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

/// Try to correlate a syntax node with its symbol using enhanced matching
let tryCorrelateSymbol (range: range) (fileName: string) (context: CorrelationContext) : FSharpSymbol option =
    // Strategy 1: Try exact range match first
    let key = (fileName, range.StartLine, range.StartColumn, range.EndLine, range.EndColumn)
    match Map.tryFind key context.PositionIndex with
    | Some symbolUse -> Some symbolUse.Symbol
    | None ->
        // Strategy 2: Search for symbols in the same file with overlapping or nearby ranges
        match Map.tryFind fileName context.FileIndex with
        | Some fileUses ->
            // First try exact containment
            let containmentMatch = 
                fileUses
                |> Array.tryFind (fun symbolUse ->
                    let sr = symbolUse.Range
                    rangeContainsPosition sr range.Start &&
                    rangeContainsPosition sr range.End
                )
            
            match containmentMatch with
            | Some symbolUse -> Some symbolUse.Symbol
            | None ->
                // Try overlapping ranges
                let overlapMatch =
                    fileUses
                    |> Array.tryFind (fun symbolUse ->
                        rangesOverlap symbolUse.Range range
                    )
                
                match overlapMatch with
                | Some symbolUse -> Some symbolUse.Symbol
                | None ->
                    // Last resort: find closest symbol within 2 lines
                    let closeMatch =
                        fileUses
                        |> Array.filter (fun symbolUse ->
                            abs(symbolUse.Range.StartLine - range.StartLine) <= 2
                        )
                        |> Array.sortBy (fun symbolUse ->
                            abs(symbolUse.Range.StartLine - range.StartLine) +
                            abs(symbolUse.Range.StartColumn - range.StartColumn)
                        )
                        |> Array.tryHead
                    
                    match closeMatch with
                    | Some symbolUse -> Some symbolUse.Symbol
                    | None -> None
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
        printfn "[Correlation] Processing file: %s with %d modules" 
            (System.IO.Path.GetFileName fileName) modules.Length
            
        let correlations = 
            modules
            |> List.map (fun m -> correlateModule m fileName context)
            |> Array.concat
            
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