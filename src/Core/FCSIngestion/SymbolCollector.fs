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

/// Known allocating patterns in F# and BCL
module private AllocationPatterns =
    let allocatingFunctions = Set.ofList [
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
    
    let alloyReplacements = Map.ofList [
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

/// Collect and analyze all symbols from processed project
let rec collectSymbols (processed: ProcessedProject) =
    printfn "[SymbolCollector] Starting symbol collection and analysis..."
    
    // Extract unique symbols
    let allSymbols = 
        processed.SymbolUses
        |> Array.map (fun symbols -> symbols.Symbol)
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
    
    // Build dependencies
    let dependencies = buildDependencies processed.SymbolUses
    
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
    printfn "  Allocating: %d" stats.AllocatingSymbols
    printfn "  BCL references: %d" stats.BCLReferences
    
    {
        AllSymbols = collectedSymbols
        EntryPoints = entryPoints
        AllocatingSymbols = allocatingSymbols
        BCLReferences = bclReferences
        Dependencies = dependencies
        Statistics = stats
    }

/// Analyze a single symbol
and private analyzeSymbol (symbol: FSharpSymbol) : CollectedSymbol =
    let fullName = symbol.FullName
    let assembly = symbol.Assembly.SimpleName
    
    let category, isAllocation = 
        match symbol with
        | :? FSharpMemberOrFunctionOrValue as func ->
            // Check for entry point
            if isEntryPoint func then
                EntryPoint, false
            // Check assembly
            elif assembly.StartsWith("Alloy") || assembly.StartsWith("Fidelity") then
                AlloyNative, false
            elif fullName.StartsWith("Microsoft.FSharp.NativeInterop") then
                SafeNativeInterop, false
            elif AllocationPatterns.allocatingFunctions.Contains(fullName) then
                AllocatingFunction, true
            elif fullName.StartsWith("System.") || fullName.StartsWith("Microsoft.") then
                BCLType, true
            elif func.DeclaringEntity.IsSome && func.DeclaringEntity.Value.Assembly.SimpleName = symbol.Assembly.SimpleName then
                UserDefined, false
            else
                Unknown, false
                
        | :? FSharpEntity as entity ->
            if assembly.StartsWith("Alloy") || assembly.StartsWith("Fidelity") then
                AlloyNative, false
            elif entity.FullName.StartsWith("System.") then
                BCLType, true
            elif entity.Assembly.SimpleName = symbol.Assembly.SimpleName then
                UserDefined, false
            else
                Unknown, false
                
        | _ -> Unknown, false
    
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
                func.DeclaringEntity |> Option.map (fun e -> e.FullName)
            | _ -> None
    }

/// Check if a function is an entry point
and private isEntryPoint (func: FSharpMemberOrFunctionOrValue) =
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

/// Build symbol dependencies from uses
and private buildDependencies (symbolUses: FSharpSymbolUse[]) =
    symbolUses
    |> Array.choose (fun uses ->
        // Find the containing symbol (function/type that uses this symbol)
        uses.Symbol.DeclarationLocation
        |> Option.bind (fun loc ->
            // This is simplified - in real implementation would need to
            // track which symbol contains this usage location
            None  // TODO: Implement containment tracking
        ))
    |> Array.distinct

/// Create a dependency graph suitable for reachability analysis
and createDependencyGraph (collected: SymbolCollectionResult) =
    // Create adjacency list representation
    collected.Dependencies
    |> Array.groupBy (fun dep -> dep.FromSymbol)
    |> Array.map (fun (from, deps) ->
        from, deps |> Array.map (fun d -> d.ToSymbol) |> Set.ofArray)
    |> Map.ofArray

/// Report on problematic symbols
and reportProblematicSymbols (collected: SymbolCollectionResult) =
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