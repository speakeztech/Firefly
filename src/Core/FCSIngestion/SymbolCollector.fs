module Core.FCSIngestion.SymbolCollector

open System
open FSharp.Compiler.Symbols
open FSharp.Compiler.CodeAnalysis
open Core.FCSIngestion.ProjectProcessor
open Core.Types.TypeSystem

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

/// Symbol dependency information for computation graph
type SymbolDependency = {
    FromSymbol: string           // Function that makes the call
    ToSymbol: string             // Function being called
    CallSite: string             // Source file location
    DependencyKind: CallType     // Nature of the dependency
}

and CallType =
    | ConstructorCall          // Type instantiation (union cases, records, classes)
    | AlloyLibraryCall         // Calls into Alloy/Fidelity framework
    | IntraAssemblyCall        // Functions within the same assembly being compiled
    | ExternalLibraryCall      // Functions from other assemblies (FSharp.Core, etc.)
    | LanguagePrimitive        // F# compiler intrinsics and operators

/// Result of computational symbol analysis
type SymbolCollectionResult = {
    /// All unique symbols indexed by full name
    AllSymbols: Map<string, CollectedSymbol>
    
    /// Entry point symbols  
    EntryPoints: CollectedSymbol[]
    
    /// Symbols that allocate heap memory
    AllocatingSymbols: CollectedSymbol[]
    
    /// BCL references that need replacement
    BCLReferences: CollectedSymbol[]
    
    /// Computational dependency graph (function calls only)
    ComputationGraph: SymbolDependency[]
    
    /// Reachable functions from entry points
    ReachableFunctions: Set<string>
    
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
    ComputationalDependencies: int
    ReachableFunctions: int
    EliminatedFunctions: int
}

/// Unified XParsec-based symbol classification system
module private SymbolClassification =
    open XParsec
    open XParsec.CharParsers
    open XParsec.Parsers

    /// Test if a parser matches at the beginning of input
    let parsePrefix parser input =
        match parser (Reader.ofString input ()) with
        | Ok _ -> true
        | Error _ -> false

    /// Parse specific string literals
    let pLiteral (s: string) = pstring s

    /// Parse F# operators (safe, non-allocating)
    let pSafeOperator =
        choice [
            // Arithmetic operators
            pLiteral "Microsoft.FSharp.Core.Operators.(+)"
            pLiteral "Microsoft.FSharp.Core.Operators.(-)"
            pLiteral "Microsoft.FSharp.Core.Operators.(*)"
            pLiteral "Microsoft.FSharp.Core.Operators.(/)"
            pLiteral "Microsoft.FSharp.Core.Operators.(%)"
            pLiteral "Microsoft.FSharp.Core.Operators.(~-)"
            pLiteral "Microsoft.FSharp.Core.Operators.(~+)"
            
            // Comparison operators
            pLiteral "Microsoft.FSharp.Core.Operators.(=)"
            pLiteral "Microsoft.FSharp.Core.Operators.(<>)"
            pLiteral "Microsoft.FSharp.Core.Operators.(<)"
            pLiteral "Microsoft.FSharp.Core.Operators.(<=)"
            pLiteral "Microsoft.FSharp.Core.Operators.(>)"
            pLiteral "Microsoft.FSharp.Core.Operators.(>=)"
            
            // Logical operators
            pLiteral "Microsoft.FSharp.Core.Operators.(&&)"
            pLiteral "Microsoft.FSharp.Core.Operators.(||)"
            pLiteral "Microsoft.FSharp.Core.Operators.not"
            
            // Bitwise operators
            pLiteral "Microsoft.FSharp.Core.Operators.(&&&)"
            pLiteral "Microsoft.FSharp.Core.Operators.(|||)"
            pLiteral "Microsoft.FSharp.Core.Operators.(^^^)"
            pLiteral "Microsoft.FSharp.Core.Operators.(~~~)"
            pLiteral "Microsoft.FSharp.Core.Operators.(<<<)"
            pLiteral "Microsoft.FSharp.Core.Operators.(>>>)"
            
            // Utility operators
            pLiteral "Microsoft.FSharp.Core.Operators.(|>)"
            pLiteral "Microsoft.FSharp.Core.Operators.(<|)"
            pLiteral "Microsoft.FSharp.Core.Operators.(>>)"
            pLiteral "Microsoft.FSharp.Core.Operators.(<<)"
            pLiteral "Microsoft.FSharp.Core.Operators.ignore"
            pLiteral "Microsoft.FSharp.Core.Operators.id"
        ]

    /// Parse safe F# Core functions (mathematical, type conversions)
    let pSafeFSharpCore =
        pLiteral "Microsoft.FSharp.Core." >>. choice [
            // Operators namespace (already covered above but for consistency)
            pLiteral "Operators.abs"
            pLiteral "Operators.max"
            pLiteral "Operators.min"
            pLiteral "Operators.sign"
            
            // Type conversions (safe when not boxing)
            pLiteral "Operators.byte"
            pLiteral "Operators.char"
            pLiteral "Operators.int"
            pLiteral "Operators.int32"
            pLiteral "Operators.int64"
            pLiteral "Operators.uint32"
            pLiteral "Operators.uint64"
            pLiteral "Operators.float"
            pLiteral "Operators.float32"
            pLiteral "Operators.double"
            pLiteral "Operators.decimal"
            
            // Language primitives (intrinsics)
            pLiteral "LanguagePrimitives.GenericZero"
            pLiteral "LanguagePrimitives.GenericOne"
            pLiteral "LanguagePrimitives.GenericEquality"
            pLiteral "LanguagePrimitives.GenericComparison"
            pLiteral "LanguagePrimitives.PhysicalEquality"
        ]

    /// Parse NativePtr operations (safe when used correctly)
    let pSafeNativeInterop =
        pLiteral "Microsoft.FSharp.NativeInterop.NativePtr." >>. choice [
            pLiteral "stackalloc"
            pLiteral "read"
            pLiteral "write"
            pLiteral "get"
            pLiteral "set"
            pLiteral "add"
            pLiteral "ofNativeInt"
            pLiteral "toNativeInt"
        ]

    /// Parse Alloy/Fidelity framework operations (explicitly safe)
    let pAlloyFramework =
        choice [
            pLiteral "Alloy."
            pLiteral "Fidelity."
            pLiteral "System.ValueOption"  // Safe alternative to Option
        ]

    /// Parse allocating F# Collections operations
    let pAllocatingCollections =
        pLiteral "Microsoft.FSharp.Collections." >>. choice [
            // Array operations that allocate
            pLiteral "Array.zeroCreate"
            pLiteral "Array.create"
            pLiteral "Array.init"
            pLiteral "Array.copy"
            pLiteral "Array.append"
            pLiteral "Array.concat"
            pLiteral "Array.map"
            pLiteral "Array.mapi"
            pLiteral "Array.filter"
            pLiteral "Array.choose"
            pLiteral "Array.collect"
            pLiteral "Array.replicate"
            pLiteral "Array.rev"
            pLiteral "Array.sort"
            pLiteral "Array.sortBy"
            pLiteral "Array.sortWith"
            
            // List operations (all allocate new list nodes)
            pLiteral "List.init"
            pLiteral "List.replicate"
            pLiteral "List.ofArray"
            pLiteral "List.toArray"
            pLiteral "List.append"
            pLiteral "List.concat"
            pLiteral "List.rev"
            pLiteral "List.sort"
            pLiteral "List.sortBy"
            pLiteral "List.sortWith"
            pLiteral "List.map"
            pLiteral "List.mapi"
            pLiteral "List.filter"
            pLiteral "List.choose"
            pLiteral "List.collect"
            
            // Sequence operations
            pLiteral "Seq.toArray"
            pLiteral "Seq.toList"
            pLiteral "Seq.cache"
            pLiteral "Seq.map"
            pLiteral "Seq.mapi"
            pLiteral "Seq.filter"
            pLiteral "Seq.choose"
            pLiteral "Seq.collect"
            pLiteral "Seq.ofArray"
            pLiteral "Seq.ofList"
        ]

    /// Parse allocating F# Core operations
    let pAllocatingFSharpCore =
        pLiteral "Microsoft.FSharp.Core." >>. choice [
            // String operations
            pLiteral "ExtraTopLevelOperators.sprintf"
            pLiteral "ExtraTopLevelOperators.printf"
            pLiteral "ExtraTopLevelOperators.printfn"
            pLiteral "ExtraTopLevelOperators.eprintf"
            pLiteral "ExtraTopLevelOperators.eprintfn"
            pLiteral "Printf.sprintf"
            pLiteral "Printf.printf"
            pLiteral "Printf.printfn"
            
            // Boxing operations
            pLiteral "Operators.box"
            pLiteral "Operators.unbox"
            pLiteral "Operators.ref"
            
            // Exception handling (allocates exception objects)
            pLiteral "Operators.raise"
            pLiteral "Operators.failwith"
            pLiteral "Operators.failwithf"
            pLiteral "Operators.invalidArg"
            pLiteral "Operators.invalidOp"
            pLiteral "Operators.nullArg"
        ]

    /// Parse allocating BCL operations
    let pAllocatingBCL =
        choice [
            // System.String operations
            pLiteral "System.String.Concat"
            pLiteral "System.String.Format"
            pLiteral "System.String.Join"
            pLiteral "System.String.Copy"
            pLiteral "System.String.Substring"
            pLiteral "System.String.Replace"
            pLiteral "System.String.Insert"
            pLiteral "System.String.Remove"
            pLiteral "System.String.ToUpper"
            pLiteral "System.String.ToLower"
            pLiteral "System.String.Trim"
            pLiteral "System.String.Split"
            
            // System.Console operations
            pLiteral "System.Console.WriteLine"
            pLiteral "System.Console.Write"
            pLiteral "System.Console.ReadLine"
            
            // System.Collections operations
            pLiteral "System.Collections.Generic.List"
            pLiteral "System.Collections.Generic.Dictionary"
            pLiteral "System.Collections.Generic.HashSet"
            pLiteral "System.Array.Copy"
            pLiteral "System.Array.Resize"
            
            // System.Text operations
            pLiteral "System.Text.StringBuilder"
            pLiteral "System.Text.Encoding.GetString"
            pLiteral "System.Text.Encoding.GetBytes"
        ]

    /// Parse any BCL type (should be replaced with Alloy)
    let pBCLType =
        choice [
            // System types (excluding System.ValueOption which is safe)
            pLiteral "System.String"
            pLiteral "System.Console"
            pLiteral "System.Collections"
            pLiteral "System.Text"
            pLiteral "System.Array"
            pLiteral "System.IO"
            pLiteral "System.Threading"
            pLiteral "System.Net"
            pLiteral "System.Data"
            pLiteral "System.Xml"
            
            // Microsoft types (excluding Microsoft.FSharp which is handled separately)
            pLiteral "Microsoft.Win32"
            pLiteral "Microsoft.VisualBasic"
            pLiteral "Microsoft.CSharp"
        ]

    /// Unified classification function using XParsec
    let classifySymbol (fullName: string) (displayName: string) : SymbolCategory * bool =
        // Priority-based classification using XParsec
        if parsePrefix pSafeOperator fullName then
            (SafePrimitive, false)
        elif parsePrefix pSafeFSharpCore fullName then
            (SafePrimitive, false)
        elif parsePrefix pSafeNativeInterop fullName then
            (SafeNativeInterop, false)
        elif parsePrefix pAlloyFramework fullName then
            (AlloyNative, false)
        elif parsePrefix pAllocatingCollections fullName then
            (AllocatingFunction, true)
        elif parsePrefix pAllocatingFSharpCore fullName then
            (AllocatingFunction, true)
        elif parsePrefix pAllocatingBCL fullName then
            (AllocatingFunction, true)
        elif parsePrefix pBCLType fullName then
            (BCLType, true)
        elif fullName.StartsWith("Microsoft.FSharp.Core") then
            // Unmatched F# Core; assume safe unless proven otherwise
            (SafePrimitive, false)
        else
            // Default classification
            (Unknown, false)

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
        not func.IsMember
    
    let result = hasEntryPointAttr || isMainFunction
    if result then
        printfn "[DIAGNOSTIC] Entry point found: %s (attr: %b, main: %b)" func.FullName hasEntryPointAttr isMainFunction
    result

/// Analyze a single symbol using unified XParsec classification
let private analyzeSymbol (symbol: FSharpSymbol) : CollectedSymbol =
    // Safely get the full name, handling edge cases
    let fullName = 
        try
            symbol.FullName
        with
        | ex when ex.Message.Contains("does not have a qualified name") ->
            match symbol with
            | :? FSharpEntity as entity ->
                match entity.Namespace with
                | Some ns -> sprintf "%s.%s" ns entity.DisplayName
                | None -> sprintf "Global.%s" entity.DisplayName
            | _ ->
                sprintf "Unknown.%s" symbol.DisplayName
        | ex ->
            if not (symbol.DisplayName = "FSharp") then
                printfn "[SymbolCollector] Warning: Could not get FullName for symbol '%s': %s" symbol.DisplayName ex.Message
            sprintf "Unknown.%s" symbol.DisplayName
    
    let displayName = symbol.DisplayName
    let assembly = symbol.Assembly.SimpleName
    
    let category, isAllocation = 
        match symbol with
        | :? FSharpMemberOrFunctionOrValue as func when isEntryPoint func ->
            (EntryPoint, false)
        | :? FSharpMemberOrFunctionOrValue as func when assembly.StartsWith("Alloy") || assembly.StartsWith("Fidelity") ->
            printfn "[DIAGNOSTIC] Alloy symbol by assembly: %s (assembly: %s)" fullName assembly
            (AlloyNative, false)
        | :? FSharpMemberOrFunctionOrValue as func when 
            func.DeclaringEntity.IsSome && 
            func.DeclaringEntity.Value.Assembly.SimpleName = symbol.Assembly.SimpleName ->
            // DIAGNOSTIC: Check if this should be Alloy
            let declaring = func.DeclaringEntity.Value
            let namespaceName = declaring.Namespace |> Option.defaultValue "None"
            let entityName = declaring.DisplayName
            if namespaceName = "Alloy" || entityName.StartsWith("Alloy") then
                printfn "[DIAGNOSTIC] Alloy symbol by namespace: %s (namespace: %s, entity: %s)" fullName namespaceName entityName
                (AlloyNative, false)
            else
                printfn "[DIAGNOSTIC] User defined: %s (namespace: %s, entity: %s)" fullName namespaceName entityName
                (UserDefined, false)
        | _ ->
            // Use unified XParsec classification
            let (cat, alloc) = SymbolClassification.classifySymbol fullName displayName
            if cat = AlloyNative then
                printfn "[DIAGNOSTIC] Alloy symbol by pattern: %s" fullName
            elif cat = AllocatingFunction then
                printfn "[DIAGNOSTIC] Allocating function: %s" fullName
            cat, alloc
    
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

/// Build computational dependency graph (functional calls only) - ROBUST VERSION
let private buildComputationGraph (processed: ProcessedProject) : SymbolDependency[] =
    printfn "[SymbolCollector] Building computation graph from %d symbol uses..." processed.SymbolUses.Length
    
    // DIAGNOSTIC: Sample symbol uses focused on user code
    printfn "[DIAGNOSTIC] User code symbol uses:"
    processed.SymbolUses
    |> Array.filter (fun symbolUse -> 
        let fileName = System.IO.Path.GetFileName(symbolUse.Range.FileName)
        fileName.Contains("HelloWorld") || fileName.Contains("01_"))
    |> Array.truncate 20
    |> Array.iteri (fun i symbolUse ->
        let symType = symbolUse.Symbol.GetType().Name
        let fileName = System.IO.Path.GetFileName(symbolUse.Range.FileName)
        printfn "  [%02d] %s (%s) at %s:%d - IsFromUse: %b, IsFromDefinition: %b" 
            i symbolUse.Symbol.DisplayName symType fileName symbolUse.Range.StartLine symbolUse.IsFromUse symbolUse.IsFromDefinition)
    
    // NEW APPROACH: Group symbol uses by file and line, then look for function call patterns
    let symbolUsesByLocation = 
        processed.SymbolUses
        |> Array.groupBy (fun su -> (su.Range.FileName, su.Range.StartLine))
        |> Map.ofArray
    
    // Find function definitions first
    let functionDefinitions = 
        processed.SymbolUses
        |> Array.choose (fun symbolUse ->
            match symbolUse.Symbol with
            | :? FSharpMemberOrFunctionOrValue as func when 
                func.IsFunction && 
                symbolUse.IsFromDefinition ->
                Some {|
                    Function = func
                    FullName = func.FullName
                    FileName = symbolUse.Range.FileName
                    StartLine = symbolUse.Range.StartLine
                    EndLine = symbolUse.Range.EndLine
                |}
            | _ -> None)
    
    printfn "[DIAGNOSTIC] Function definitions found: %d" functionDefinitions.Length
    functionDefinitions |> Array.iter (fun fdef ->
        let fileName = System.IO.Path.GetFileName(fdef.FileName)
        printfn "  DEF: %s at %s:%d-%d" fdef.FullName fileName fdef.StartLine fdef.EndLine)
    
    // SIMPLIFIED: Find function calls using symbol use patterns
    let functionalDeps = 
        processed.SymbolUses
        |> Array.choose (fun symbolUse ->
            match symbolUse.Symbol with
            | :? FSharpMemberOrFunctionOrValue as func when 
                symbolUse.IsFromUse &&           // This is a usage, not a definition
                func.IsFunction &&               // It's a function
                not func.IsProperty &&           // Not a property
                not func.IsEvent &&              // Not an event
                not func.IsCompilerGenerated ->  // Not compiler generated
                
                // Find which function this call occurs within using SIMPLER logic
                let containingFunction = 
                    functionDefinitions
                    |> Array.tryFind (fun fdef ->
                        fdef.FileName = symbolUse.Range.FileName &&
                        fdef.StartLine < symbolUse.Range.StartLine &&  // Function starts before the call
                        symbolUse.Range.StartLine < fdef.EndLine &&    // Call is before function ends
                        fdef.FullName <> func.FullName)                // Don't create self-references
                
                match containingFunction with
                | Some container ->
                    let dependencyKind = 
                        if func.IsConstructor then ConstructorCall
                        elif func.FullName.StartsWith("Alloy.") || func.FullName.StartsWith("Fidelity.") then AlloyLibraryCall
                        elif func.Assembly.SimpleName = symbolUse.Symbol.Assembly.SimpleName then IntraAssemblyCall
                        elif func.Assembly.SimpleName = "FSharp.Core" then LanguagePrimitive
                        else ExternalLibraryCall
                    
                    let dep = {
                        FromSymbol = container.FullName
                        ToSymbol = func.FullName
                        CallSite = sprintf "%s:%d" symbolUse.Range.FileName symbolUse.Range.StartLine
                        DependencyKind = dependencyKind
                    }
                    
                    // DIAGNOSTIC: Log important dependencies
                    if container.FullName.Contains("main") || container.FullName.Contains("hello") || 
                       func.FullName.Contains("hello") || func.FullName.Contains("stackBuffer") || 
                       func.FullName.Contains("Prompt") || func.FullName.Contains("WriteLine") then
                        printfn "[DIAGNOSTIC] KEY DEPENDENCY: %s -> %s (%A)" container.FullName func.FullName dependencyKind
                    
                    Some dep
                | None ->
                    // DIAGNOSTIC: Log when we can't find a container for key functions
                    if func.FullName.Contains("hello") || func.FullName.Contains("stackBuffer") || 
                       func.FullName.Contains("Prompt") || func.FullName.Contains("WriteLine") then
                        let fileName = System.IO.Path.GetFileName(symbolUse.Range.FileName)
                        printfn "[DIAGNOSTIC] NO CONTAINER for %s at %s:%d" func.FullName fileName symbolUse.Range.StartLine
                        
                        // Show nearby function definitions for debugging
                        let nearbyFunctions = 
                            functionDefinitions
                            |> Array.filter (fun fdef -> 
                                fdef.FileName = symbolUse.Range.FileName &&
                                abs (fdef.StartLine - symbolUse.Range.StartLine) < 10)
                        
                        printfn "[DIAGNOSTIC]   Nearby functions:"
                        nearbyFunctions |> Array.iter (fun fdef ->
                            printfn "[DIAGNOSTIC]     %s at lines %d-%d" fdef.FullName fdef.StartLine fdef.EndLine)
                    None
            | _ -> None)
        |> Array.distinctBy (fun dep -> (dep.FromSymbol, dep.ToSymbol))
    
    printfn "[SymbolCollector] Built computation graph with %d functional dependencies" functionalDeps.Length
    
    // DIAGNOSTIC: Show all dependencies with better categorization
    printfn "[DIAGNOSTIC] Dependencies by category:"
    
    let userCodeDeps = functionalDeps |> Array.filter (fun dep -> 
        dep.FromSymbol.Contains("Examples.") || dep.ToSymbol.Contains("Examples."))
    printfn "[DIAGNOSTIC] User code dependencies (%d):" userCodeDeps.Length
    userCodeDeps |> Array.iter (fun dep ->
        printfn "  USER: %s -> %s (%A)" dep.FromSymbol dep.ToSymbol dep.DependencyKind)
    
    let alloyDeps = functionalDeps |> Array.filter (fun dep -> 
        dep.FromSymbol.Contains("Alloy.") || dep.ToSymbol.Contains("Alloy."))
    printfn "[DIAGNOSTIC] Alloy dependencies (%d):" alloyDeps.Length
    alloyDeps |> Array.truncate 10 |> Array.iter (fun dep ->
        printfn "  ALLOY: %s -> %s (%A)" dep.FromSymbol dep.ToSymbol dep.DependencyKind)
    
    functionalDeps

/// Trace computational reachability from entry points
let private traceComputationalReachability (entryPoints: string[]) (computationGraph: SymbolDependency[]) : Set<string> =
    printfn "[SymbolCollector] Tracing computational reachability from %d entry points..." entryPoints.Length
    
    // DIAGNOSTIC: Show entry points
    printfn "[DIAGNOSTIC] Entry points:"
    entryPoints |> Array.iteri (fun i ep -> printfn "  [%d] %s" i ep)
    
    let dependencyMap = 
        computationGraph
        |> Array.groupBy (fun dep -> dep.FromSymbol)
        |> Map.ofArray
        |> Map.map (fun _ deps -> deps |> Array.map (fun d -> d.ToSymbol) |> Set.ofArray)
    
    // DIAGNOSTIC: Show dependency map structure
    printfn "[DIAGNOSTIC] Dependency map structure:"
    dependencyMap |> Map.iter (fun caller callees ->
        printfn "  %s -> [%s]" caller (String.concat "; " callees))
    
    let rec traverse (visited: Set<string>) (toVisit: string list) : Set<string> =
        match toVisit with
        | [] -> 
            printfn "[DIAGNOSTIC] Traversal complete, visited: %d functions" visited.Count
            visited
        | current :: remaining ->
            if Set.contains current visited then
                traverse visited remaining
            else
                printfn "[DIAGNOSTIC] Visiting: %s" current
                let newVisited = Set.add current visited
                let callees = 
                    Map.tryFind current dependencyMap 
                    |> Option.defaultValue Set.empty
                    |> Set.toList
                
                if callees.Length > 0 then
                    printfn "[DIAGNOSTIC] %s calls: [%s]" current (String.concat "; " callees)
                
                traverse newVisited (callees @ remaining)
    
    let reachable = traverse Set.empty (entryPoints |> Array.toList)
    printfn "[SymbolCollector] Found %d computationally reachable functions" reachable.Count
    
    // DIAGNOSTIC: Show all reachable functions
    printfn "[DIAGNOSTIC] All reachable functions:"
    reachable |> Set.toArray |> Array.sort |> Array.iteri (fun i func ->
        printfn "  [%02d] %s" i func)
    
    reachable

/// Report on problematic symbols
let private reportProblematicSymbols (allocatingSymbols: CollectedSymbol[]) (bclReferences: CollectedSymbol[]) =
    if allocatingSymbols.Length > 0 then
        printfn "\n[SymbolCollector] ❌ COMPILATION ERRORS: Allocating symbols found:"
        allocatingSymbols
        |> Array.truncate 10
        |> Array.iter (fun sym ->
            printfn "  ERROR: %s - heap allocation not allowed in Firefly" sym.DisplayName)
                
        if allocatingSymbols.Length > 10 then
            printfn "  ... and %d more allocation errors" (allocatingSymbols.Length - 10)
        
        printfn "\n  Firefly requires zero-allocation code. Use Alloy framework alternatives."
    
    if bclReferences.Length > 0 then
        printfn "\n[SymbolCollector] ❌ COMPILATION ERRORS: Unsupported BCL references:"
        bclReferences
        |> Array.truncate 5
        |> Array.iter (fun sym ->
            printfn "  ERROR: %s from %s - BCL not supported, use Alloy framework" sym.FullName sym.Assembly)
        
        if bclReferences.Length > 5 then
            printfn "  ... and %d more BCL errors" (bclReferences.Length - 5)
        
        printfn "\n  Firefly compiles to native code without BCL dependencies."

/// Main symbol collection function with computational analysis
let collectSymbols (processed: ProcessedProject) : SymbolCollectionResult =
    printfn "[SymbolCollector] Starting computational symbol analysis..."
    
    // Basic symbol classification using unified XParsec approach
    let allSymbols = 
        processed.SymbolUses
        |> Array.map (fun symbolUse -> symbolUse.Symbol)
        |> Array.distinctBy (fun s -> s.FullName)
        |> Array.map analyzeSymbol
        |> Array.map (fun sym -> sym.FullName, sym)
        |> Map.ofArray
    
    let entryPoints = 
        allSymbols
        |> Map.toArray
        |> Array.map snd
        |> Array.filter (fun s -> s.Category = EntryPoint)
    
    let allocatingSymbols =
        allSymbols
        |> Map.toArray
        |> Array.map snd
        |> Array.filter (fun s -> s.IsAllocation)
    
    let bclReferences =
        allSymbols
        |> Map.toArray
        |> Array.map snd
        |> Array.filter (fun s -> s.Category = BCLType)
    
    // Build computational dependency graph
    let computationGraph = buildComputationGraph processed
    
    // Trace reachable functions from entry points
    let entryPointNames = entryPoints |> Array.map (fun ep -> ep.FullName)
    let reachableFunctions = traceComputationalReachability entryPointNames computationGraph
    
    // Calculate statistics
    let totalFunctions = allSymbols |> Map.filter (fun _ s -> s.Category = UserDefined || s.Category = AlloyNative) |> Map.count
    let stats = {
        TotalSymbols = allSymbols.Count
        UserDefinedSymbols = allSymbols |> Map.filter (fun _ s -> s.Category = UserDefined) |> Map.count
        AlloySymbols = allSymbols |> Map.filter (fun _ s -> s.Category = AlloyNative) |> Map.count
        AllocatingSymbols = allocatingSymbols.Length
        BCLReferences = bclReferences.Length
        EntryPoints = entryPoints.Length
        ComputationalDependencies = computationGraph.Length
        ReachableFunctions = reachableFunctions.Count
        EliminatedFunctions = totalFunctions - reachableFunctions.Count
    }
    
    printfn "[SymbolCollector] Analysis complete:"
    printfn "  Total symbols: %d" stats.TotalSymbols
    printfn "  Computational dependencies: %d" stats.ComputationalDependencies
    printfn "  Reachable functions: %d" stats.ReachableFunctions
    printfn "  Eliminated functions: %d" stats.EliminatedFunctions
    
    let result = {
        AllSymbols = allSymbols
        EntryPoints = entryPoints
        AllocatingSymbols = allocatingSymbols
        BCLReferences = bclReferences
        ComputationGraph = computationGraph
        ReachableFunctions = reachableFunctions
        Statistics = stats
    }
    
    // Report on problematic symbols
    reportProblematicSymbols allocatingSymbols bclReferences
    
    result

/// Create a dependency graph suitable for reachability analysis
let createDependencyGraph (collected: SymbolCollectionResult) =
    collected.ComputationGraph
    |> Array.groupBy (fun dep -> dep.FromSymbol)
    |> Array.map (fun (from, deps) ->
        from, deps |> Array.map (fun d -> d.ToSymbol) |> Set.ofArray)
    |> Map.ofArray