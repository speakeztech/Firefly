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

/// Create correlation context from FCS results
let createContext (checkResults: FSharpCheckProjectResults) =
    // Use GetAllUsesOfAllSymbols() instead of GetAllUsesOfAllSymbolsInFile()
    let allSymbolUses = checkResults.GetAllUsesOfAllSymbols()
    
    printfn "[Correlation Debug] Found %d total symbol uses" allSymbolUses.Length
    printfn "[Correlation Debug] Definition count: %d" 
        (allSymbolUses |> Array.filter (fun su -> su.IsFromDefinition) |> Array.length)
    printfn "[Correlation Debug] Use count: %d" 
        (allSymbolUses |> Array.filter (fun su -> su.IsFromUse) |> Array.length)
    
    let positionIndex = 
        allSymbolUses
        |> Array.map (fun symbolUse ->
            let r = symbolUse.Range
            let key = (r.FileName, r.Start.Line, r.Start.Column, r.End.Line, r.End.Column)
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

/// Correlate a syntax node with its symbol using range matching
let tryCorrelateSymbol (range: range) (fileName: string) (context: CorrelationContext) =
    // Try exact range match first
    let key = (fileName, range.Start.Line, range.Start.Column, range.End.Line, range.End.Column)
    match Map.tryFind key context.PositionIndex with
    | Some symbolUse -> Some symbolUse.Symbol
    | None ->
        // Fall back to searching overlapping ranges
        match Map.tryFind fileName context.FileIndex with
        | Some fileUses ->
            // First try to find exact containment
            let exactMatch = 
                fileUses
                |> Array.tryFind (fun symbolUse ->
                    let sr = symbolUse.Range
                    sr.Start.Line = range.Start.Line && 
                    sr.Start.Column = range.Start.Column &&
                    sr.End.Line = range.End.Line && 
                    sr.End.Column = range.End.Column
                )
            
            match exactMatch with
            | Some symbolUse -> Some symbolUse.Symbol
            | None ->
                // Try to find symbol that contains this range
                fileUses
                |> Array.tryFind (fun symbolUse ->
                    let sr = symbolUse.Range
                    // Symbol range contains the query range
                    sr.Start.Line <= range.Start.Line && 
                    sr.End.Line >= range.End.Line &&
                    (sr.Start.Line < range.Start.Line || sr.Start.Column <= range.Start.Column) &&
                    (sr.End.Line > range.End.Line || sr.End.Column >= range.End.Column)
                )
                |> Option.map (fun symbolUse -> symbolUse.Symbol)
        | None -> None

/// Visitor pattern for syntax tree traversal with correlation
type CorrelationVisitor = {
    OnBinding: SynBinding -> range -> FSharpSymbol option -> unit
    OnExpression: SynExpr -> range -> FSharpSymbol option -> unit
    OnPattern: SynPat -> range -> FSharpSymbol option -> unit
    OnType: SynType -> range -> FSharpSymbol option -> unit
    OnMember: SynMemberDefn -> range -> FSharpSymbol option -> unit
}

/// Traverse syntax tree with correlation
let rec traverseWithCorrelation 
    (node: obj) 
    (fileName: string) 
    (context: CorrelationContext) 
    (visitor: CorrelationVisitor) =
    
    match node with
    | :? SynBinding as binding ->
        let symbol = tryCorrelateSymbol binding.RangeOfBindingWithRhs fileName context
        visitor.OnBinding binding binding.RangeOfBindingWithRhs symbol
        // Extract expression from binding using pattern matching
        let (SynBinding(_, _, _, _, _, _, _, _, _, expr, _, _, _)) = binding
        traverseWithCorrelation expr fileName context visitor
        
    | :? SynExpr as expr ->
        let symbol = tryCorrelateSymbol expr.Range fileName context
        visitor.OnExpression expr expr.Range symbol
        
        // Traverse subexpressions
        match expr with
        | SynExpr.App(_, _, funcExpr, argExpr, _) ->
            traverseWithCorrelation funcExpr fileName context visitor
            traverseWithCorrelation argExpr fileName context visitor
            
        | SynExpr.Lambda(_, _, _, bodyExpr, _, _, _) ->
            traverseWithCorrelation bodyExpr fileName context visitor
            
        | SynExpr.LetOrUse(_, _, bindings, bodyExpr, _, _) ->
            bindings |> List.iter (fun b -> traverseWithCorrelation b fileName context visitor)
            traverseWithCorrelation bodyExpr fileName context visitor
            
        | SynExpr.Match(_, expr, clauses, _, _) ->
            traverseWithCorrelation expr fileName context visitor
            clauses |> List.iter (fun clause ->
                let (SynMatchClause(pat, whenExpr, resultExpr, _, _, _)) = clause // 6 args with trivia
                traverseWithCorrelation pat fileName context visitor
                whenExpr |> Option.iter (fun e -> traverseWithCorrelation e fileName context visitor)
                traverseWithCorrelation resultExpr fileName context visitor
            )
            
        | SynExpr.Sequential(_, _, expr1, expr2, _, _) ->
            traverseWithCorrelation expr1 fileName context visitor
            traverseWithCorrelation expr2 fileName context visitor
            
        | SynExpr.Ident _ | SynExpr.LongIdent _ | SynExpr.Const _ ->
            () // Leaf nodes, no further traversal needed
            
        | _ -> () // Handle other expression types as needed
        
    | :? SynPat as pattern ->
        let symbol = tryCorrelateSymbol pattern.Range fileName context
        visitor.OnPattern pattern pattern.Range symbol
        
        // Traverse sub-patterns
        match pattern with
        | SynPat.Named(_, _, _, _) | SynPat.Wild _ | SynPat.Const _ ->
            () // Leaf patterns
        | SynPat.Paren(pat, _) ->
            traverseWithCorrelation pat fileName context visitor
        | SynPat.Tuple(_, pats, _, _) ->
            pats |> List.iter (fun p -> traverseWithCorrelation p fileName context visitor)
        | _ -> ()
        
    | :? SynType as synType ->
        let symbol = tryCorrelateSymbol synType.Range fileName context
        visitor.OnType synType synType.Range symbol
        
    | :? SynModuleDecl as moduleDecl ->
        // Handle module declarations
        match moduleDecl with
        | SynModuleDecl.Let(_, bindings, _) ->
            bindings |> List.iter (fun b -> traverseWithCorrelation b fileName context visitor)
        | SynModuleDecl.Types(typeDefs, _) ->
            typeDefs |> List.iter (fun td -> traverseWithCorrelation td fileName context visitor)
        | SynModuleDecl.Open(target, _) ->
            () // No traversal needed for opens
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
        printfn "[Correlation Debug] Processing file: %s with %d modules" 
            (System.IO.Path.GetFileName fileName) modules.Length
            
        let correlations = 
            modules
            |> List.map (fun m -> correlateModule m fileName context)
            |> Array.concat
            
        printfn "[Correlation Debug] Found %d correlations in %s" 
            correlations.Length (System.IO.Path.GetFileName fileName)
            
        correlations
    | ParsedInput.SigFile _ ->
        // Signature files not yet supported
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
            | :? FSharpStaticParameter -> "StaticParameter"
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
            let correlatedInFile = 
                correlations 
                |> Array.filter (fun (r, _) -> r.FileName = fileName)
                |> Array.length
            float correlatedInFile / float fileUses.Length
        )
    
    {
        TotalSymbols = context.SymbolUses.Length
        CorrelatedNodes = correlations.Length
        UncorrelatedNodes = context.SymbolUses.Length - correlations.Length
        SymbolsByKind = symbolsByKind
        FileCoverage = fileCoverage
    }