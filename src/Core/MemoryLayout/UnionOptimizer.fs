module Core.MemoryLayout.UnionOptimizer

open System
open Core.XParsec.Foundation

/// Union layout strategy with optimization information
type UnionLayoutStrategy =
    | TaggedUnion of tagSize: int * maxPayloadSize: int * alignment: int
    | EnumOptimization of enumValues: int list * underlyingType: string
    | OptionOptimization of someType: string * nullRepresentation: bool
    | SingleCase of caseType: string * isNewtype: bool
    | EmptyUnion of errorMessage: string

/// Union layout with complete memory information
type UnionLayout = {
    Strategy: UnionLayoutStrategy
    TotalSize: int
    Alignment: int
    TagOffset: int
    PayloadOffset: int
    CaseMap: Map<string, int>
    IsZeroAllocation: bool
}

/// Layout analysis state for tracking transformations
type LayoutAnalysisState = {
    UnionLayouts: Map<string, UnionLayout>
    TypeMappings: Map<string, string>
    TransformationHistory: (string * string) list
}

/// Type size and alignment calculations
module TypeSizeCalculation =
    
    /// Calculates the size of a type in bytes
    let rec calculateTypeSize (typeName: string) : int =
        match typeName with
        | "int" | "int32" -> 4
        | "int64" -> 8
        | "float" | "float32" -> 4
        | "float64" | "double" -> 8
        | "bool" -> 1
        | "byte" | "uint8" -> 1
        | "char" -> 2
        | "unit" -> 0
        | _ when typeName.StartsWith("array") -> 8  // Pointer size
        | _ when typeName.StartsWith("(") && typeName.EndsWith(")") -> 
            // Tuple type - sum of component sizes
            8  // Simplified for now
        | _ -> 8  // Default to pointer size for unknown types
    
    /// Calculates the alignment requirement for a type
    let rec calculateTypeAlignment (typeName: string) : int =
        match typeName with
        | "int" | "int32" -> 4
        | "int64" -> 8
        | "float" | "float32" -> 4
        | "float64" | "double" -> 8
        | "bool" -> 1
        | "byte" | "uint8" -> 1
        | "char" -> 2
        | "unit" -> 1
        | _ -> 8  // Conservative alignment

/// Layout strategy analysis
module LayoutStrategyAnalysis =
    
    /// Analyzes a general union for layout strategy
    let analyzeGeneralUnion (name: string) (cases: (string * string option) list) : CompilerResult<UnionLayoutStrategy> =
        try
            // Check if all cases are nullary (enum-like)
            let allNullary = cases |> List.forall (fun (_, optType) -> optType.IsNone)
            
            if allNullary then
                let enumValues = [0 .. cases.Length - 1]
                let underlyingType = 
                    if cases.Length <= 256 then "uint8"
                    elif cases.Length <= 65536 then "uint16"
                    else "uint32"
                Success (EnumOptimization(enumValues, underlyingType))
            elif cases.Length = 1 then
                let (caseName, caseType) = cases.Head
                match caseType with
                | Some typ -> Success (SingleCase(typ, true))
                | None -> Success (SingleCase("unit", false))
            else
                // General tagged union
                let maxPayloadSize = 
                    cases
                    |> List.choose snd
                    |> List.map TypeSizeCalculation.calculateTypeSize
                    |> List.fold max 0
                
                let maxAlignment = 
                    cases
                    |> List.choose snd
                    |> List.map TypeSizeCalculation.calculateTypeAlignment
                    |> List.fold max 1
                
                Success (TaggedUnion(1, maxPayloadSize, maxAlignment))
                
        with ex ->
            CompilerFailure [InternalError("union analysis", sprintf "Failed to analyze union %s" name, Some ex.Message)]

/// Union layout computation
module UnionLayoutComputation =
    
    /// Computes the complete layout for a union type
    let computeUnionLayout (name: string) (cases: (string * string option) list) : CompilerResult<UnionLayout> =
        match LayoutStrategyAnalysis.analyzeGeneralUnion name cases with
        | Success strategy ->
            let (totalSize, alignment, tagOffset, payloadOffset, isZeroAlloc) =
                match strategy with
                | EnumOptimization(_, underlyingType) ->
                    let size = TypeSizeCalculation.calculateTypeSize underlyingType
                    (size, size, 0, 0, true)
                    
                | SingleCase(caseType, _) ->
                    let size = TypeSizeCalculation.calculateTypeSize caseType
                    let align = TypeSizeCalculation.calculateTypeAlignment caseType
                    (size, align, 0, 0, true)
                    
                | TaggedUnion(tagSize, maxPayloadSize, maxAlignment) ->
                    let alignedPayloadOffset = ((tagSize + maxAlignment - 1) / maxAlignment) * maxAlignment
                    let totalSize = alignedPayloadOffset + maxPayloadSize
                    (totalSize, maxAlignment, 0, alignedPayloadOffset, true)
                    
                | EmptyUnion _ ->
                    (0, 1, 0, 0, true)
                    
                | _ ->
                    (8, 8, 0, 0, false)
            
            let caseMap = 
                cases 
                |> List.mapi (fun i (caseName, _) -> (caseName, i))
                |> Map.ofList
            
            Success {
                Strategy = strategy
                TotalSize = totalSize
                Alignment = alignment
                TagOffset = tagOffset
                PayloadOffset = payloadOffset
                CaseMap = caseMap
                IsZeroAllocation = isZeroAlloc
            }
            
        | CompilerFailure errors -> CompilerFailure errors

/// Result helpers
module ResultHelpers =
    
    /// Combines a list of results into a single result
    let rec combineResults (results: CompilerResult<'T> list) : CompilerResult<'T list> =
        match results with
        | [] -> Success []
        | Success x :: rest ->
            match combineResults rest with
            | Success xs -> Success (x :: xs)
            | CompilerFailure errors -> CompilerFailure errors
        | CompilerFailure errors :: rest ->
            match combineResults rest with
            | Success _ -> CompilerFailure errors
            | CompilerFailure moreErrors -> CompilerFailure (errors @ moreErrors)

/// Public API for union optimization
module Optimize =
    
    /// Optimizes a union type definition
    let optimizeUnion (name: string) (cases: (string * string option) list) : CompilerResult<UnionLayout> =
        UnionLayoutComputation.computeUnionLayout name cases
    
    /// Creates an initial analysis state
    let createInitialState() : LayoutAnalysisState = {
        UnionLayouts = Map.empty
        TypeMappings = Map.empty
        TransformationHistory = []
    }
    
    /// Adds a union layout to the state
    let addUnionLayout (name: string) (layout: UnionLayout) (state: LayoutAnalysisState) : LayoutAnalysisState = {
        state with 
            UnionLayouts = Map.add name layout state.UnionLayouts
            TransformationHistory = (name, sprintf "Added layout: %A" layout.Strategy) :: state.TransformationHistory
    }
    
    /// Validates that all layouts are zero-allocation
    let validateZeroAllocation (state: LayoutAnalysisState) : CompilerResult<unit> =
        let nonZeroAlloc = 
            state.UnionLayouts 
            |> Map.toList
            |> List.filter (fun (_, layout) -> not layout.IsZeroAllocation)
            |> List.map fst
        
        if nonZeroAlloc.IsEmpty then
            Success ()
        else
            CompilerFailure [
                ConversionError(
                    "zero-allocation validation",
                    "union layouts",
                    "zero-allocation layouts",
                    sprintf "The following unions may cause heap allocations: %s" (String.concat ", " nonZeroAlloc)
                )
            ]