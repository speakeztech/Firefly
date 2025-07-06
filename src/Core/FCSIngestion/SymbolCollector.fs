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

/// Improved XParsec-based symbol classification for Firefly compiler
module private SymbolClassificationParsers =
    open XParsec
    open XParsec.CharParsers
    open XParsec.Parsers

    /// Parse specific operator symbols (actual FCS names)
    let pOperatorSymbol (symbol: string) = pstring symbol

    /// Parse any arithmetic operator
    let pArithmeticOperator =
        choice [
            pOperatorSymbol "(+)"
            pOperatorSymbol "(-)" 
            pOperatorSymbol "(*)"
            pOperatorSymbol "(/)"
            pOperatorSymbol "(%)"
            pOperatorSymbol "(~-)"
            pOperatorSymbol "(~+)"
        ]

    /// Parse comparison operators
    let pComparisonOperator =
        choice [
            pOperatorSymbol "(=)"
            pOperatorSymbol "(<>)"
            pOperatorSymbol "(<)"
            pOperatorSymbol "(<=)"
            pOperatorSymbol "(>)"
            pOperatorSymbol "(>=)"
        ]

    /// Parse bitwise operators
    let pBitwiseOperator =
        choice [
            pOperatorSymbol "(&&&)"
            pOperatorSymbol "(|||)"
            pOperatorSymbol "(^^^)"
            pOperatorSymbol "(~~~)"
            pOperatorSymbol "(<<<)"
            pOperatorSymbol "(>>>)"
        ]

    /// Parse logical operators
    let pLogicalOperator =
        choice [
            pOperatorSymbol "(&&)"
            pOperatorSymbol "(||)"
            pOperatorSymbol "(not)"
        ]

    /// Parse utility operators
    let pUtilityOperator =
        choice [
            pOperatorSymbol "(|>)"
            pOperatorSymbol "(<|)"
            pOperatorSymbol "(>>)"
            pOperatorSymbol "(<<)"
            pOperatorSymbol "(|)"
            pOperatorSymbol "ignore"
            pOperatorSymbol "id"
        ]

    /// Parse all safe operators
    let pSafeOperator =
        choice [
            pArithmeticOperator
            pComparisonOperator
            pBitwiseOperator
            pLogicalOperator
            pUtilityOperator
        ]

    /// Parse compile-time attributes (never allocate at runtime)
    let pCompileTimeAttribute =
        choice [
            pstring "AbstractClassAttribute"
            pstring "AutoOpenAttribute"
            pstring "CompiledNameAttribute"
            pstring "EntryPointAttribute"
            pstring "LiteralAttribute"
            pstring "RequireQualifiedAccessAttribute"
            pstring "StructAttribute"
            pstring "NoComparisonAttribute"
            pstring "NoEqualityAttribute"
            pstring "CustomComparisonAttribute"
            pstring "CustomEqualityAttribute"
            pstring "StructuralComparisonAttribute"
            pstring "StructuralEqualityAttribute"
            pstring "CLIMutableAttribute"
            pstring "CompilerGeneratedAttribute"
            pstring "DebuggerNonUserCodeAttribute"
            pstring "GeneratedCodeAttribute"
        ]

    /// Parse basic F# types that don't allocate when used as types
    let pBasicFSharpType =
        choice [
            pstring "int"
            pstring "int32"
            pstring "int64"
            pstring "uint32"
            pstring "uint64"
            pstring "float"
            pstring "float32"
            pstring "double"
            pstring "byte"
            pstring "sbyte"
            pstring "char"
            pstring "bool"
            pstring "unit"
            pstring "nativeint"
            pstring "unativeint"
        ]

    /// Parse safe F# Core functions (mathematical, no allocation)
    let pSafeFSharpCoreFunction =
        choice [
            // Math functions
            pstring "abs"; pstring "acos"; pstring "asin"; pstring "atan"; pstring "atan2"
            pstring "cos"; pstring "sin"; pstring "tan"; pstring "exp"
            pstring "log"; pstring "log10"; pstring "sqrt"; pstring "pown"
            pstring "sign"; pstring "max"; pstring "min"; pstring "round"
            pstring "ceil"; pstring "floor"; pstring "truncate"
            
            // Type conversions (safe when not boxing)
            pstring "byte"; pstring "char"; pstring "int"; pstring "int32"; pstring "int64"
            pstring "uint32"; pstring "uint64"; pstring "float"; pstring "float32"
            pstring "double"; pstring "decimal"; pstring "nativeint"; pstring "unativeint"
            
            // Utility functions
            pstring "id"; pstring "ignore"; pstring "not"
        ]

    /// Parse F# Core operators namespace (mostly safe)
    let pFSharpCoreOperators =
        pstring "Microsoft.FSharp.Core.Operators." >>. choice [
            pSafeOperator
            pSafeFSharpCoreFunction
            pstring "Unchecked.defaultof"  // Safe stack allocation
            pstring "Unchecked.unbox"      // Safe when used correctly
            pstring "Unchecked.compare"    // Safe structural comparison
        ]

    /// Parse F# LanguagePrimitives (mostly safe intrinsics)
    let pFSharpLanguagePrimitives =
        pstring "Microsoft.FSharp.Core.LanguagePrimitives." >>. choice [
            pstring "GenericZero"; pstring "GenericOne"
            pstring "AdditionDynamic"; pstring "SubtractionDynamic"
            pstring "MultiplyDynamic"; pstring "DivisionDynamic"
            pstring "EqualityComparer"; pstring "GenericEquality"
            pstring "GenericComparison"; pstring "PhysicalEquality"
            pstring "IntrinsicOperators." >>. choice [
                pOperatorSymbol "(&&)"; pOperatorSymbol "(||)"
                pOperatorSymbol "(~&&)"; pOperatorSymbol "(~||)"
            ]
        ]

    /// Parse option/voption operations (safe when using ValueOption)
    let pSafeOptionOperations =
        choice [
            pstring "Some"; pstring "None"  // When referring to ValueOption
            pstring "ValueSome"; pstring "ValueNone"
            pstring "isSome"; pstring "isNone"
            pstring "get"  // Safe for ValueOption
        ]

    /// Parse known allocating F# functions
    let pAllocatingFSharpFunction =
        choice [
            // Array creation functions
            pstring "Microsoft.FSharp.Collections.Array.zeroCreate"
            pstring "Microsoft.FSharp.Collections.Array.create"
            pstring "Microsoft.FSharp.Collections.Array.init"
            pstring "Microsoft.FSharp.Collections.Array.copy"
            pstring "Microsoft.FSharp.Collections.Array.append"
            pstring "Microsoft.FSharp.Collections.Array.concat"
            
            // List operations (all allocate)
            pstring "Microsoft.FSharp.Collections.List.init"
            pstring "Microsoft.FSharp.Collections.List.replicate"
            pstring "Microsoft.FSharp.Collections.List.ofArray"
            pstring "Microsoft.FSharp.Collections.List.toArray"
            pstring "Microsoft.FSharp.Collections.List.append"
            pstring "Microsoft.FSharp.Collections.List.concat"
            pstring "Microsoft.FSharp.Collections.List.rev"
            pstring "Microsoft.FSharp.Collections.List.sort"
            pstring "Microsoft.FSharp.Collections.List.map"
            pstring "Microsoft.FSharp.Collections.List.filter"
            
            // Sequence operations (lazy but allocate when materialized)
            pstring "Microsoft.FSharp.Collections.Seq.toArray"
            pstring "Microsoft.FSharp.Collections.Seq.toList"
            pstring "Microsoft.FSharp.Collections.Seq.cache"
            pstring "Microsoft.FSharp.Collections.Seq.map"
            pstring "Microsoft.FSharp.Collections.Seq.filter"
            
            // String operations
            pstring "Microsoft.FSharp.Core.ExtraTopLevelOperators.sprintf"
            pstring "Microsoft.FSharp.Core.ExtraTopLevelOperators.printfn"
            pstring "Microsoft.FSharp.Core.Printf.sprintf"
            pstring "Microsoft.FSharp.Core.Printf.printfn"
            
            // Boxing operations
            pstring "Microsoft.FSharp.Core.Operators.box"
            pstring "Microsoft.FSharp.Core.Operators.unbox"
            pstring "Microsoft.FSharp.Core.Operators.ref"
            
            // Exception handling
            pstring "Microsoft.FSharp.Core.Operators.raise"
            pstring "Microsoft.FSharp.Core.Operators.failwith"
            pstring "Microsoft.FSharp.Core.Operators.invalidArg"
        ]

    /// Parse BCL allocating functions (System namespace)
    let pAllocatingBCLFunction =
        choice [
            pstring "System.String.Concat"
            pstring "System.String.Format"
            pstring "System.String.Join"
            pstring "System.Console.WriteLine"
            pstring "System.Console.Write"
            pstring "System.Text.StringBuilder"
            pstring "System.Array.Copy"
            pstring "System.Array.Resize"
            pstring "System.Collections.Generic.List"
        ]

    /// Parse Alloy/Fidelity framework symbols
    let pAlloyFramework =
        choice [
            pstring "Alloy."
            pstring "Fidelity."
            pstring "System.ValueOption`1"  // Safe alternative to Option
            pstring "voption"               // Type alias
        ]

    /// Parse NativePtr operations (safe when used correctly)
    let pNativeInterop =
        pstring "Microsoft.FSharp.NativeInterop." >>. choice [
            pstring "NativePtr.stackalloc"
            pstring "NativePtr.read"
            pstring "NativePtr.write"
            pstring "NativePtr.get"
            pstring "NativePtr.set"
            pstring "NativePtr.add"
            pstring "NativePtr.ofNativeInt"
            pstring "NativePtr.toNativeInt"
        ]

    /// Parse all safe F# primitives
    let pSafeFSharpPrimitive =
        choice [
            pFSharpCoreOperators
            pFSharpLanguagePrimitives
            pNativeInterop
            pSafeOptionOperations
            pBasicFSharpType
        ]

    /// Test if a parser matches the beginning of input
    let parsePrefix parser input =
        match parser (Reader.ofString input ()) with
        | Ok _ -> true
        | Error _ -> false

    /// Enhanced symbol classification with proper precedence
    let classifySymbolName (fullName: string) (displayName: string) : SymbolCategory * bool =
        // Handle display name patterns first for simple cases
        match displayName with
        | name when parsePrefix pCompileTimeAttribute name ->
            SafePrimitive, false  // Attributes don't exist at runtime
        | name when parsePrefix pBasicFSharpType name ->
            SafePrimitive, false  // Basic types don't allocate
        | name when parsePrefix pSafeOperator name ->
            SafePrimitive, false  // Operators are safe
        | "list" | "seq" | "array" when not (fullName.Contains("Alloy")) ->
            AllocatingFunction, true  // BCL collections allocate
        | "::" | "op_ColonColon" ->
            AllocatingFunction, true  // List cons allocates
        | "FSharp" ->
            SafePrimitive, false  // Namespace/assembly name, not allocating
        | _ ->
            // Use full name for detailed classification
            if parsePrefix pSafeFSharpPrimitive fullName then
                SafePrimitive, false
            elif parsePrefix pAllocatingFSharpFunction fullName then  
                AllocatingFunction, true
            elif parsePrefix pAllocatingBCLFunction fullName then
                AllocatingFunction, true
            elif parsePrefix pAlloyFramework fullName then
                AlloyNative, false
            elif fullName.StartsWith("Microsoft.FSharp.Core") then
                // Most F# Core is safe unless explicitly listed as allocating
                SafePrimitive, false
            elif fullName.StartsWith("System.") && not (fullName.StartsWith("System.ValueOption")) then
                BCLType, true  // Most System types should be replaced with Alloy
            elif fullName.StartsWith("Microsoft.") && not (fullName.StartsWith("Microsoft.FSharp")) then
                BCLType, true  // Non-F# Microsoft types
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

/// Enhanced symbol analysis with improved classification
let private analyzeSymbol (symbol: FSharpSymbol) : CollectedSymbol =
    // Safely get the full name, handling edge cases
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
            // Don't spam warnings for common cases
            if not (symbol.DisplayName = "FSharp") then
                printfn "[SymbolCollector] Warning: Could not get FullName for symbol '%s': %s" symbol.DisplayName ex.Message
            sprintf "Unknown.%s" symbol.DisplayName
    
    let displayName = symbol.DisplayName
    let assembly = symbol.Assembly.SimpleName
    
    let category, isAllocation = 
        match symbol with
        | :? FSharpMemberOrFunctionOrValue as func when isEntryPoint func ->
            EntryPoint, false
        | :? FSharpMemberOrFunctionOrValue as func when assembly.StartsWith("Alloy") || assembly.StartsWith("Fidelity") ->
            AlloyNative, false
        | :? FSharpMemberOrFunctionOrValue as func when 
            func.DeclaringEntity.IsSome && 
            func.DeclaringEntity.Value.Assembly.SimpleName = symbol.Assembly.SimpleName ->
            UserDefined, false
        | _ ->
            // Use enhanced classification with both names
            SymbolClassificationParsers.classifySymbolName fullName displayName
    
    {
        Symbol = symbol
        FullName = fullName
        DisplayName = displayName
        Assembly = assembly
        Category = category
        IsAllocation = isAllocation
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
    
    // Create a lookup of declarations by file and location using symbolUse ranges
    let declarations = 
        processed.SymbolUses
        |> Array.choose (fun symbolUse ->
            match symbolUse.Symbol with
            | :? FSharpMemberOrFunctionOrValue as func ->
                // Use the symbol use range as the declaration location
                // Filter for symbols that appear to be declarations (not references)
                if symbolUse.IsFromDefinition then
                    let declRange = symbolUse.Range
                    Some {|
                        Symbol = func.FullName
                        File = declRange.FileName
                        StartLine = declRange.StartLine
                        EndLine = declRange.EndLine
                    |}
                else
                    None
            | _ -> None)
    
    // For each symbol use, find the containing declaration
    let dependencies = 
        processed.SymbolUses
        |> Array.choose (fun symbolUse ->
            let useRange = symbolUse.Range
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

/// Report on problematic symbols that should cause compilation errors
let private reportProblematicSymbols (collected: SymbolCollectionResult) =
    if collected.AllocatingSymbols.Length > 0 then
        printfn "\n[SymbolCollector] ❌ COMPILATION ERRORS: Allocating symbols found:"
        collected.AllocatingSymbols
        |> Array.truncate 10
        |> Array.iter (fun sym ->
            printfn "  ERROR: %s - heap allocation not allowed in Firefly" sym.DisplayName)
                
        if collected.AllocatingSymbols.Length > 10 then
            printfn "  ... and %d more allocation errors" (collected.AllocatingSymbols.Length - 10)
        
        printfn "\n  Firefly requires zero-allocation code. Use Alloy framework alternatives."
    
    if collected.BCLReferences.Length > 0 then
        printfn "\n[SymbolCollector] ❌ COMPILATION ERRORS: Unsupported BCL references:"
        collected.BCLReferences
        |> Array.truncate 5
        |> Array.iter (fun sym ->
            printfn "  ERROR: %s from %s - BCL not supported, use Alloy framework" sym.FullName sym.Assembly)
        
        if collected.BCLReferences.Length > 5 then
            printfn "  ... and %d more BCL errors" (collected.BCLReferences.Length - 5)
        
        printfn "\n  Firefly compiles to native code without BCL dependencies."

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