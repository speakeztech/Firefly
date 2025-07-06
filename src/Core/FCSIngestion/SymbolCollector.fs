module Core.FCSIngestion.SymbolCollector

open FSharp.Compiler.Symbols
open FSharp.Compiler.CodeAnalysis
open Core.FCSIngestion.ProjectProcessor

/// Classification of symbols for compilation analysis
type SymbolCategory =
    | EntryPoint              // Program entry points
    | AlloyNative            // Alloy/Fidelity framework symbols
    | SafePrimitive          // F# primitives that don't allocate
    | SafeNativeInterop      // NativePtr and related
    | AllocatingFunction     // Known heap-allocating functions
    | BCLType               // BCL types that should be avoided
    | UserDefined           // User-defined types and functions
    | Unknown               // Needs further analysis

/// Analyzed symbol with metadata
type CollectedSymbol = {
    Symbol: FSharpSymbol
    FullName: string
    DisplayName: string
    Assembly: string
    Category: SymbolCategory
    IsAllocation: bool
    AlloyReplacement: string option
    DeclaringEntity: string option
}

/// Symbol dependency information
type SymbolDependency = {
    FromSymbol: string
    ToSymbol: string
    DependencyKind: string  // "calls", "references", "inherits", etc.
}

/// Result of symbol collection and analysis
type SymbolCollectionResult = {
    /// All unique symbols indexed by full name
    AllSymbols: Map<string, CollectedSymbol>
    
    /// Entry point symbols
    EntryPoints: CollectedSymbol[]
    
    /// Symbols that allocate heap memory
    AllocatingSymbols: CollectedSymbol[]
    
    /// BCL references that need replacement
    BCLReferences: CollectedSymbol[]
    
    /// Symbol dependency graph
    Dependencies: SymbolDependency[]
    
    /// Statistics
    Statistics: SymbolStatistics
}

and SymbolStatistics = {
    TotalSymbols: int
    UserDefinedSymbols: int
    AlloySymbols: int
    AllocatingSymbols: int
    BCLReferences: int
    EntryPoints: int
}

/// XParsec-based symbol classification - establishing the compiler pattern
module private SymbolClassificationParsers =
    open XParsec
    open XParsec.CharParsers
    open XParsec.Parsers

    /// Parse specific operator symbols (actual FCS names)
    let pOperatorSymbol (symbol: string) = pstring symbol

    /// Parse any arithmetic operator (using actual symbolic names)
    let pArithmeticOperator =
        choice [
            pOperatorSymbol "(+)"
            pOperatorSymbol "(-)" 
            pOperatorSymbol "(*)"
            pOperatorSymbol "(/)"
            pOperatorSymbol "(%)"
        ]

    /// Parse any comparison operator (using actual symbolic names)
    let pComparisonOperator =
        choice [
            pOperatorSymbol "(=)"
            pOperatorSymbol "(<>)"
            pOperatorSymbol "(<)"
            pOperatorSymbol "(<=)"
            pOperatorSymbol "(>)"
            pOperatorSymbol "(>=)"
        ]

    /// Parse any bitwise operator (using actual symbolic names)
    let pBitwiseOperator =
        choice [
            pOperatorSymbol "(&&&)"
            pOperatorSymbol "(|||)"
            pOperatorSymbol "(~~~)"
            pOperatorSymbol "(<<<)"
            pOperatorSymbol "(>>>)"
        ]

    /// Parse any boolean operator (using actual intrinsic names)
    let pBooleanOperator =
        choice [
            pOperatorSymbol "(&&)"
            pOperatorSymbol "(||)"
        ]

    /// Parse utility operators
    let pUtilityOperator =
        choice [
            pOperatorSymbol "(|>)"
            pOperatorSymbol "(~&&)"
        ]

    /// Parse any operator
    let pAnyOperator =
        choice [
            pArithmeticOperator
            pComparisonOperator
            pBitwiseOperator
            pBooleanOperator
            pUtilityOperator
        ]

    /// Parse mathematical function names
    let pMathFunctionName =
        choice [
            pstring "abs"; pstring "acos"; pstring "asin"; pstring "atan"; pstring "atan2"
            pstring "cos"; pstring "sin"; pstring "tan"; pstring "exp"
            pstring "log"; pstring "log10"; pstring "sqrt"; pstring "pown"
            pstring "sign"; pstring "max"; pstring "min"
        ]

    /// Parse utility function names
    let pUtilityFunctionName =
        choice [
            pstring "id"; pstring "ignore"; pstring "``not``"
        ]

    /// Parse type conversion function names
    let pTypeConversionName =
        choice [
            pstring "byte"; pstring "char"; pstring "int"; pstring "int64"
        ]

    /// Parse language primitive names
    let pLanguagePrimitiveName =
        choice [
            pstring "GenericZero"; pstring "GenericOne"
            pstring "DecimalWithMeasure"; pstring "Float32WithMeasure"
            pstring "FloatWithMeasure"; pstring "Int32WithMeasure"; pstring "Int64WithMeasure"
        ]

    /// Parse operators namespace with any operator
    let pOperatorsNamespace =
        pstring "Microsoft.FSharp.Core.Operators." >>. choice [
            pAnyOperator
            pMathFunctionName
            pUtilityFunctionName
            pTypeConversionName
            pstring "Unchecked.defaultof"
        ]

    /// Parse language primitives namespace
    let pLanguagePrimitivesNamespace =
        pstring "Microsoft.FSharp.Core.LanguagePrimitives." >>. choice [
            pLanguagePrimitiveName
            pstring "IntrinsicOperators." >>. choice [
                pOperatorSymbol "(&&)"
                pOperatorSymbol "(||)" 
                pOperatorSymbol "(~&&)"
            ]
        ]

    /// Parse any safe F# primitive
    let pSafePrimitive =
        choice [
            pOperatorsNamespace
            pLanguagePrimitivesNamespace
        ]

    /// Parse known allocating function patterns
    let pAllocatingFunction =
        choice [
            pstring "Microsoft.FSharp.Collections.Array.zeroCreate"
            pstring "Microsoft.FSharp.Collections.Array.create"
            pstring "Microsoft.FSharp.Collections.Array.init"
            pstring "Microsoft.FSharp.Collections.List.init"
            pstring "Microsoft.FSharp.Collections.List.replicate"
            pstring "Microsoft.FSharp.Collections.List.ofArray"
            pstring "Microsoft.FSharp.Collections.List.toArray"
            pstring "Microsoft.FSharp.Collections.Seq.toArray"
            pstring "Microsoft.FSharp.Collections.Seq.toList"
            pstring "Microsoft.FSharp.Core.ExtraTopLevelOperators.sprintf"
            pstring "Microsoft.FSharp.Core.ExtraTopLevelOperators.printfn"
            pstring "System.String.Concat"
        ]

    /// Parse Alloy/Fidelity framework symbols
    let pAlloyFramework =
        choice [
            pstring "Alloy."
            pstring "Fidelity."
            pstring "System.ValueOption`1"
            pstring "voption"
        ]

    /// Parse native interop operations
    let pNativeInterop =
        pstring "Microsoft.FSharp.NativeInterop."

    /// Parse BCL System/Microsoft prefixes
    let pBCLPrefix =
        choice [
            pstring "System."
            pstring "Microsoft."
        ]

    /// Test if a parser matches the beginning of input (prefix matching)
    let parsePrefix parser input =
        match parser (Reader.ofString input ()) with
        | Ok _ -> true
        | Error _ -> false

    /// Classify a symbol's full name using XParsec combinators
    let classifySymbolName (fullName: string) : SymbolCategory * bool =
        // Try parsers in priority order - most specific first
        if parsePrefix pSafePrimitive fullName then
            SafePrimitive, false
        elif parsePrefix pAllocatingFunction fullName then  
            AllocatingFunction, true
        elif parsePrefix pAlloyFramework fullName then
            AlloyNative, false
        elif parsePrefix pNativeInterop fullName then
            SafeNativeInterop, false
        elif parsePrefix pBCLPrefix fullName then
            BCLType, true
        else
            Unknown, false

/// Known allocating patterns in F# and BCL
module private AllocationPatterns =
    let allocatingFunctions = 
        Set.ofList [
            // F# Collections
            "Microsoft.FSharp.Collections.Array.zeroCreate"
            "Microsoft.FSharp.Collections.Array.create"
            "Microsoft.FSharp.Collections.Array.init"
            "Microsoft.FSharp.Collections.List.init"
            "Microsoft.FSharp.Collections.List.replicate"
            "Microsoft.FSharp.Collections.List.ofArray"
            "Microsoft.FSharp.Collections.List.toArray"
            "Microsoft.FSharp.Collections.Seq.toArray"
            "Microsoft.FSharp.Collections.Seq.toList"
            
            // F# Core
            "Microsoft.FSharp.Core.ExtraTopLevelOperators.sprintf"
            "Microsoft.FSharp.Core.ExtraTopLevelOperators.printfn"
            "Microsoft.FSharp.Core.Operators.box"
            "Microsoft.FSharp.Core.Operators.raise"
            "Microsoft.FSharp.Core.String.concat"
            
            // BCL
            "System.String.Concat"
            "System.String.Format"
            "System.Console.WriteLine"
        ]
    
    let alloyReplacements = 
        Map.ofList [
            // Array operations
            ("Microsoft.FSharp.Collections.Array.zeroCreate", "Alloy.Memory.stackBuffer")
            ("Microsoft.FSharp.Collections.Array.create", "Alloy.Memory.stackBufferWithValue")
            ("Microsoft.FSharp.Collections.Array.length", "Alloy.Memory.bufferLength")
            
            // String/text operations  
            ("Microsoft.FSharp.Core.ExtraTopLevelOperators.sprintf", "Alloy.Text.formatStackBuffer")
            ("Microsoft.FSharp.Core.ExtraTopLevelOperators.printfn", "Alloy.Console.writeLine")
            ("System.Console.WriteLine", "Alloy.Console.writeLine")
            
            // Math operations
            ("System.Math.Sin", "Alloy.Math.sin")
            ("System.Math.Cos", "Alloy.Math.cos")
            ("System.Math.Sqrt", "Alloy.Math.sqrt")
        ]

/// Check if a function is an entry point
let private isEntryPoint (func: FSharpMemberOrFunctionOrValue) =
    // Check for [<EntryPoint>] attribute
    let hasEntryPointAttr = 
        func.Attributes 
        |> Seq.exists (fun attr -> 
            attr.AttributeType.BasicQualifiedName = "Microsoft.FSharp.Core.EntryPointAttribute")
    
    // Check for main function
    let isMainFunction = 
        func.LogicalName = "main" && 
        func.IsModuleValueOrMember &&
        not func.IsMember &&
        func.CurriedParameterGroups.Count = 0
    
    hasEntryPointAttr || isMainFunction

/// Analyze a single symbol using XParsec-based classification
let private analyzeSymbol (symbol: FSharpSymbol) : CollectedSymbol =
    // Safely get the full name, handling voption edge case
    let fullName = 
        try
            symbol.FullName
        with
        | ex when ex.Message.Contains("does not have a qualified name") ->
            match symbol with
            | :? FSharpEntity as entity when entity.DisplayName.Contains("voption") ->
                "System.ValueOption`1"
            | :? FSharpEntity as entity when entity.DisplayName.Contains("ValueOption") ->
                "System.ValueOption`1"
            | :? FSharpEntity as entity ->
                match entity.Namespace with
                | Some ns -> sprintf "%s.%s" ns entity.DisplayName
                | None -> sprintf "Global.%s" entity.DisplayName
            | _ ->
                sprintf "Unknown.%s" symbol.DisplayName
        | ex ->
            printfn "[SymbolCollector] Warning: Could not get FullName for symbol '%s': %s" symbol.DisplayName ex.Message
            sprintf "Unknown.%s" symbol.DisplayName
    
    let assembly = symbol.Assembly.SimpleName
    
    let category, isAllocation = 
        match symbol with
        | :? FSharpMemberOrFunctionOrValue as func when isEntryPoint func ->
            EntryPoint, false
        | :? FSharpMemberOrFunctionOrValue as func when assembly.StartsWith("Alloy") || assembly.StartsWith("Fidelity") ->
            AlloyNative, false
        | :? FSharpMemberOrFunctionOrValue as func when func.DeclaringEntity.IsSome && func.DeclaringEntity.Value.Assembly.SimpleName = symbol.Assembly.SimpleName ->
            UserDefined, false
        | _ ->
            // Use XParsec-based classification for external symbols
            SymbolClassificationParsers.classifySymbolName fullName
    
    {
        Symbol = symbol
        FullName = fullName
        DisplayName = symbol.DisplayName
        Assembly = assembly
        Category = category
        IsAllocation = isAllocation
        AlloyReplacement = AllocationPatterns.alloyReplacements.TryFind(fullName)
        DeclaringEntity = 
            match symbol with
            | :? FSharpMemberOrFunctionOrValue as func -> 
                func.DeclaringEntity |> Option.map (fun e -> 
                    try e.FullName 
                    with _ -> sprintf "%s.%s" (e.Namespace |> Option.defaultValue "Global") e.DisplayName)
            | _ -> None
    }

/// Build symbol dependencies from FCS data using a single principled approach
let private buildDependencies (processed: ProcessedProject) =
    printfn "[SymbolCollector] Building dependencies from %d symbol uses..." processed.SymbolUses.Length
    
    // Create a lookup of declarations by file and location
    let declarations = 
        processed.SymbolUses
        |> Array.choose (fun symbolUse ->
            match symbolUse.Symbol with
            | :? FSharpMemberOrFunctionOrValue as func when func.DeclarationLocation.IsSome ->
                let declLocation = func.DeclarationLocation.Value
                Some {|
                    Symbol = func.FullName
                    File = declLocation.FileName
                    StartLine = declLocation.StartLine
                    EndLine = declLocation.EndLine
                |}
            | _ -> None)
    
    // For each symbol use, find the containing declaration
    let dependencies = 
        processed.SymbolUses
        |> Array.choose (fun symbolUse ->
            match symbolUse.Range with
            | Some useRange ->
                let usedSymbol = symbolUse.Symbol.FullName
                
                // Find the declaration that contains this usage
                let containingDeclaration = 
                    declarations
                    |> Array.tryFind (fun decl ->
                        decl.File = useRange.FileName &&
                        decl.StartLine <= useRange.StartLine &&
                        useRange.EndLine <= decl.EndLine &&
                        decl.Symbol <> usedSymbol) // Don't create self-dependencies
                
                match containingDeclaration with
                | Some decl ->
                    Some {
                        FromSymbol = decl.Symbol
                        ToSymbol = usedSymbol
                        DependencyKind = "calls"
                    }
                | None -> None
            | None -> None)
        |> Array.distinctBy (fun dep -> (dep.FromSymbol, dep.ToSymbol))
    
    printfn "[SymbolCollector] Built %d dependencies" dependencies.Length
    
    // Debug: show sample dependencies
    if dependencies.Length > 0 then
        printfn "[SymbolCollector] Sample dependencies:"
        dependencies 
        |> Array.truncate 5 
        |> Array.iter (fun dep -> printfn "  %s -> %s" dep.FromSymbol dep.ToSymbol)
    
    dependencies

/// Report on problematic symbols
let private reportProblematicSymbols (collected: SymbolCollectionResult) =
    if collected.AllocatingSymbols.Length > 0 then
        printfn "\n[SymbolCollector] ⚠️  Allocating symbols found:"
        collected.AllocatingSymbols
        |> Array.truncate 10
        |> Array.iter (fun sym ->
            match sym.AlloyReplacement with
            | Some replacement ->
                printfn "  %s → %s" sym.DisplayName replacement
            | None ->
                printfn "  %s (no Alloy replacement)" sym.DisplayName)
                
        if collected.AllocatingSymbols.Length > 10 then
            printfn "  ... and %d more" (collected.AllocatingSymbols.Length - 10)
    
    if collected.BCLReferences.Length > 0 then
        printfn "\n[SymbolCollector] ⚠️  BCL references found:"
        collected.BCLReferences
        |> Array.truncate 5
        |> Array.iter (fun sym ->
            printfn "  %s from %s" sym.FullName sym.Assembly)

/// Create a dependency graph suitable for reachability analysis
let createDependencyGraph (collected: SymbolCollectionResult) =
    collected.Dependencies
    |> Array.groupBy (fun dep -> dep.FromSymbol)
    |> Array.map (fun (from, deps) ->
        from, deps |> Array.map (fun d -> d.ToSymbol) |> Set.ofArray)
    |> Map.ofArray

/// Collect and analyze all symbols from processed project
let collectSymbols (processed: ProcessedProject) =
    printfn "[SymbolCollector] Starting symbol collection and analysis..."
    
    // Extract unique symbols
    let allSymbols = 
        processed.SymbolUses
        |> Array.map (fun symbolUse -> symbolUse.Symbol)
        |> Array.distinctBy (fun s -> s.FullName)
    
    printfn "[SymbolCollector] Found %d unique symbols" allSymbols.Length
    
    // Analyze each symbol
    let collectedSymbols = 
        allSymbols
        |> Array.map analyzeSymbol
        |> Array.map (fun sym -> sym.FullName, sym)
        |> Map.ofArray
    
    // Find entry points
    let entryPoints = 
        collectedSymbols
        |> Map.toArray
        |> Array.map snd
        |> Array.filter (fun s -> s.Category = EntryPoint)
    
    // Find allocating symbols
    let allocatingSymbols =
        collectedSymbols
        |> Map.toArray
        |> Array.map snd
        |> Array.filter (fun s -> s.IsAllocation)
    
    // Find BCL references
    let bclReferences =
        collectedSymbols
        |> Map.toArray
        |> Array.map snd
        |> Array.filter (fun s -> s.Category = BCLType)
    
    // Build dependencies using single principled approach
    let dependencies = buildDependencies processed
    
    // Calculate statistics
    let stats = {
        TotalSymbols = collectedSymbols.Count
        UserDefinedSymbols = collectedSymbols |> Map.filter (fun _ s -> s.Category = UserDefined) |> Map.count
        AlloySymbols = collectedSymbols |> Map.filter (fun _ s -> s.Category = AlloyNative) |> Map.count
        AllocatingSymbols = allocatingSymbols.Length
        BCLReferences = bclReferences.Length
        EntryPoints = entryPoints.Length
    }
    
    printfn "[SymbolCollector] Analysis complete:"
    printfn "  Entry points: %d" stats.EntryPoints
    printfn "  User defined: %d" stats.UserDefinedSymbols
    printfn "  Alloy symbols: %d" stats.AlloySymbols
    printfn "  Allocating: %d" stats.AllocatingSymbols
    printfn "  BCL references: %d" stats.BCLReferences
    printfn "  Dependencies: %d" dependencies.Length
    
    let result = {
        AllSymbols = collectedSymbols
        EntryPoints = entryPoints
        AllocatingSymbols = allocatingSymbols
        BCLReferences = bclReferences
        Dependencies = dependencies
        Statistics = stats
    }
    
    // Report on problematic symbols
    reportProblematicSymbols result
    
    result