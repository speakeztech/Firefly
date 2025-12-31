module Core.Meta.AlloyHints

open FSharp.Native.Compiler.Symbols
open Core.Meta.Parser

/// Validated MLIR hint with platform compatibility
type ValidatedHint = {
    Hint: MLIRHint
    Compatibility: PlatformCompatibility
    Priority: HintPriority
}

and PlatformCompatibility =
    | Universal
    | PlatformSpecific of platforms: string list
    | RequiresCapability of capability: string

and HintPriority =
    | Required      // Must be followed
    | Suggested     // Should be followed unless incompatible
    | Optional      // Can be ignored if better option exists

/// Validate MLIR hints against platform capabilities
let validateHint (hint: MLIRHint) (platform: string) =
    // Check dialect support
    let dialectSupported = 
        match hint.Dialect with
        | "llvm" | "arith" | "memref" -> true  // Always available
        | "gpu" -> platform.Contains("gpu")
        | "vector" -> platform.Contains("simd") || platform.Contains("vector")
        | _ -> false
    
    if not dialectSupported then None
    else
        let compatibility = 
            match hint.Dialect with
            | "gpu" -> RequiresCapability "gpu"
            | "vector" -> RequiresCapability "simd"
            | _ -> Universal
        
        let priority = 
            match hint.MemoryStrategy with
            | Some StackOnly -> Required  // Safety critical
            | Some (StaticAllocation) -> Suggested
            | _ -> Optional
        
        Some {
            Hint = hint
            Compatibility = compatibility
            Priority = priority
        }

/// Merge hints from different sources
let mergeHints (explicit: MLIRHint option) (inferred: MLIRHint option) =
    match explicit, inferred with
    | Some e, Some i ->
        // Explicit hints override inferred ones
        Some {
            Dialect = e.Dialect
            Operation = e.Operation
            Parameters = Map.fold (fun m k v -> Map.add k v m) i.Parameters e.Parameters
            MemoryStrategy = e.MemoryStrategy |> Option.orElse i.MemoryStrategy
        }
    | Some e, None -> Some e
    | None, Some i -> Some i
    | None, None -> None

/// Get comprehensive hints for a symbol
let getComprehensiveHints (symbol: FSharpSymbol) (platform: string) =
    // Get explicit hints from XML documentation
    let explicitHint = getSymbolHints symbol
    
    // Get inferred hints based on symbol type
    let inferredHint = 
        match symbol with
        | :? FSharpMemberOrFunctionOrValue as mfv ->
            Patterns.suggestForRecursion mfv
        | :? FSharpEntity as entity ->
            Patterns.suggestForCollection entity
        | _ -> None
    
    // Merge and validate
    let merged = mergeHints explicitHint inferredHint
    
    merged |> Option.bind (fun hint -> validateHint hint platform)

/// Hint database for common patterns
module HintDatabase =
    type PatternHint = {
        Pattern: string
        Description: string
        Hint: MLIRHint
        Example: string
    }
    
    let commonPatterns = [
        {
            Pattern = "TailRecursion"
            Description = "Tail recursive functions can use simple loops"
            Hint = {
                Dialect = "scf"
                Operation = "for"
                Parameters = Map.ofList ["step", "1"]
                MemoryStrategy = Some StackOnly
            }
            Example = "let rec sum acc = function | [] -> acc | h::t -> sum (acc + h) t"
        }
        
        {
            Pattern = "MapReduce"
            Description = "Map-reduce patterns can use parallel execution"
            Hint = {
                Dialect = "scf"
                Operation = "parallel"
                Parameters = Map.ofList ["chunk_size", "1024"]
                MemoryStrategy = Some (PooledAllocation "parallel_pool")
            }
            Example = "data |> Array.map f |> Array.reduce (+)"
        }
        
        {
            Pattern = "StreamProcessing"
            Description = "Stream processing benefits from streaming buffers"
            Hint = {
                Dialect = "memref"
                Operation = "view"
                Parameters = Map.ofList ["stride", "1"]
                MemoryStrategy = Some (StreamingBuffer 4096)
            }
            Example = "stream |> Stream.map transform |> Stream.filter predicate"
        }
    ]
    
    /// Find matching pattern for code
    let findPattern (expr: FSharpExpr) =
        // Simplified pattern matching
        // Real implementation would use more sophisticated analysis
        None

/// Generate MLIR hints report
let generateHintsReport (symbols: (FSharpSymbol * ValidatedHint option) list) =
    let withHints = symbols |> List.choose (fun (s, h) -> h |> Option.map (fun hint -> s, hint))
    let withoutHints = symbols |> List.filter (fun (_, h) -> Option.isNone h) |> List.map fst
    
    {|
        TotalSymbols = symbols.Length
        SymbolsWithHints = withHints.Length
        RequiredHints = withHints |> List.filter (fun (_, h) -> h.Priority = Required) |> List.length
        SuggestedHints = withHints |> List.filter (fun (_, h) -> h.Priority = Suggested) |> List.length
        OptionalHints = withHints |> List.filter (fun (_, h) -> h.Priority = Optional) |> List.length
        SymbolsWithoutHints = withoutHints
    |}