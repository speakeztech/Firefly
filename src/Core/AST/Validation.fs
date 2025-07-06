module Core.AST.Validation

open System
open FSharp.Compiler.Symbols
open FSharp.Compiler.Text
open Core.AST.Extraction
open Core.XParsec.Foundation


/// Convert FCS range to SourcePosition
let rangeToPosition (range: range) : SourcePosition = {
    Line = range.Start.Line
    Column = range.Start.Column
    File = range.FileName
    Offset = 0
}

/// Combine multiple compiler results into a single result
let combineResults (results: CompilerResult<'T> list) : CompilerResult<'T list> =
    let successes = ResizeArray<'T>()
    let failures = ResizeArray<FireflyError>()
    
    for result in results do
        match result with
        | Success value -> successes.Add(value)
        | CompilerFailure errors -> failures.AddRange(errors)
    
    if failures.Count = 0 then
        Success (successes |> List.ofSeq)
    else
        CompilerFailure (failures |> List.ofSeq)

// Import allocation types explicitly
type AllocationType = 
    | HeapAllocation | ObjectConstruction | CollectionAllocation

type AllocationSite = {
    TypeName: string
    Location: range
    AllocationType: AllocationType
}

// ===================================================================
// Allocation Detection Logic
// ===================================================================

/// Determine if a function call results in heap allocation
let private isAllocatingFunction (fullName: string) : bool =
    let allocatingPatterns = [
        "Microsoft.FSharp.Collections.List."
        "Microsoft.FSharp.Collections.Array.create"
        "Microsoft.FSharp.Collections.Array.zeroCreate" 
        "Microsoft.FSharp.Collections.Array.init"
        "Microsoft.FSharp.Core.Printf."
        "System.String."
        "System.Text.StringBuilder"
        "System.Collections.Generic"
    ]
    
    let safePatterns = [
        "Microsoft.FSharp.NativeInterop.NativePtr."
        "Microsoft.FSharp.Core.Operators."
        "Alloy."
        "Fidelity."
        "System.Math."
        "Microsoft.FSharp.Core.Option.get"
        "Microsoft.FSharp.Core.Option.isSome"
        "Microsoft.FSharp.Core.Option.isNone"
    ]
    
    // First check if it's explicitly safe
    if safePatterns |> List.exists fullName.StartsWith then 
        false
    else 
        // Then check if it's allocating
        allocatingPatterns |> List.exists fullName.StartsWith

// ===================================================================
// Simplified Expression Analysis for Allocations
// ===================================================================

/// Detect allocation sites using simplified expression analysis
let private detectAllocationsInExpr (expr: FSharpExpr) : AllocationSite list =
    let allocations = ResizeArray<AllocationSite>()
    
    try
        let rec traverse (expr: FSharpExpr) =
            // Check if this expression type might indicate allocation
            if expr.Type.HasTypeDefinition then
                let typeDef = expr.Type.TypeDefinition
                let typeName = typeDef.QualifiedName
                
                // Check for known allocating types
                if isAllocatingFunction typeName then
                    allocations.Add({
                        TypeName = typeName
                        Location = expr.Range
                        AllocationType = HeapAllocation
                    })
                elif typeName.StartsWith("System.") && not (typeName.StartsWith("System.Math")) then
                    allocations.Add({
                        TypeName = typeName
                        Location = expr.Range
                        AllocationType = ObjectConstruction
                    })
            
            // Recursively check sub-expressions using the correct API
            for subExpr in expr.ImmediateSubExpressions do
                traverse subExpr
        
        traverse expr
    with
    | ex ->
        eprintfn "Warning: Could not fully analyze allocations in expression: %s" ex.Message
    
    allocations |> List.ofSeq

// ===================================================================
// Zero-Allocation Verification
// ===================================================================

/// Verify that reachable functions contain no heap allocations
let verifyZeroAllocation (reachableFunctions: TypedFunction[]) : Result<unit, AllocationSite[]> =
    let allAllocations = 
        reachableFunctions
        |> Array.collect (fun func -> 
            try
                detectAllocationsInExpr func.Body |> Array.ofList
            with
            | ex ->
                eprintfn "Warning: Failed to analyze allocations in %s: %s" func.FullName ex.Message
                [||])
    
    if Array.isEmpty allAllocations then 
        Ok () 
    else 
        Error allAllocations

/// Get detailed allocation report
let getAllocationReport (allocationSites: AllocationSite[]) : string =
    if Array.isEmpty allocationSites then
        "✓ Zero-allocation verification passed - no heap allocations detected"
    else
        let sb = System.Text.StringBuilder()
        sb.AppendLine("❌ Zero-allocation verification failed:") |> ignore
        sb.AppendLine() |> ignore
        
        let groupedAllocations = 
            allocationSites 
            |> Array.groupBy (fun site -> site.AllocationType)
        
        for (allocType, sites) in groupedAllocations do
            sb.AppendLine($"{allocType} allocations:") |> ignore
            for site in sites do
                let location = $"{site.Location.FileName}({site.Location.Start.Line},{site.Location.Start.Column})"
                sb.AppendLine($"  • {site.TypeName} at {location}") |> ignore
            sb.AppendLine() |> ignore
        
        sb.ToString()

// ===================================================================
// Function Validation Pipeline
// ===================================================================

/// Comprehensive validation of a function
let validateFunction (func: TypedFunction) : CompilerResult<unit> =
    try
        // Check for allocations using simplified analysis
        let allocations = detectAllocationsInExpr func.Body
        if List.isEmpty allocations then
            Success ()
        else
            let errors = allocations |> List.map (fun site ->
                let pos = rangeToPosition site.Location
                TypeCheckError(
                    func.FullName, 
                    $"Allocation detected: {site.TypeName} ({site.AllocationType})", 
                    pos))
            CompilerFailure errors
    with
    | ex ->
        let pos = rangeToPosition func.Range
        CompilerFailure [InternalError("validateFunction", ex.Message, Some ex.StackTrace)]

/// Validate all reachable functions in the program
let validateProgram (reachableFunctions: TypedFunction[]) : CompilerResult<unit> =
    let validationResults = 
        reachableFunctions 
        |> Array.map validateFunction
        |> Array.toList
    
    combineResults validationResults
    |> function
        | Success _ -> Success ()
        | CompilerFailure errors -> CompilerFailure errors

/// Get validation statistics
let getValidationStatistics (functions: TypedFunction[]) : Map<string, int> =
    let mutable totalAllocations = 0
    let mutable functionsWithAllocations = 0
    let mutable totalFunctions = functions.Length
    
    for func in functions do
        try
            let allocations = detectAllocationsInExpr func.Body
            if not (List.isEmpty allocations) then
                functionsWithAllocations <- functionsWithAllocations + 1
                totalAllocations <- totalAllocations + allocations.Length
        with
        | _ -> () // Continue processing other functions
    
    Map.ofList [
        ("TotalFunctions", totalFunctions)
        ("FunctionsWithAllocations", functionsWithAllocations)
        ("TotalAllocationSites", totalAllocations)
        ("CleanFunctions", totalFunctions - functionsWithAllocations)
    ]