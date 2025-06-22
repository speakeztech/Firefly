module Dabbit.UnionLayouts.FixedLayoutCompiler

open System
open Core.XParsec.Foundation
open Dabbit.Parsing.OakAst

/// Union layout strategy with optimization information
type UnionLayoutStrategy =
    | TaggedUnion of tagSize: int * maxPayloadSize: int * alignment: int
    | EnumOptimization of enumValues: int list * underlyingType: OakType
    | OptionOptimization of someType: OakType * nullRepresentation: bool
    | SingleCase of caseType: OakType * isNewtype: bool
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
    TypeMappings: Map<string, OakType>
    TransformationHistory: (string * string) list
}

/// Type size and alignment calculations
module TypeSizeCalculation =
    
    /// Calculates the size of an Oak type in bytes
    let rec calculateTypeSize (oakType: OakType) : int =
        match oakType with
        | IntType -> 4
        | FloatType -> 4  
        | BoolType -> 1
        | StringType -> 8  // Pointer size
        | UnitType -> 0
        | ArrayType _ -> 8  // Pointer size
        | FunctionType(_, _) -> 8  // Function pointer
        | StructType fields ->
            fields |> List.sumBy (snd >> calculateTypeSize)
        | UnionType cases ->
            // For unanalyzed unions, estimate conservatively
            let maxCaseSize = 
                cases
                |> List.map (fun (_, optType) ->
                    match optType with
                    | Some t -> calculateTypeSize t
                    | None -> 0)
                |> List.fold max 0
            1 + maxCaseSize  // Tag + largest payload
    
    /// Calculates the alignment requirement for an Oak type
    let rec calculateTypeAlignment (oakType: OakType) : int =
        match oakType with
        | IntType -> 4
        | FloatType -> 4
        | BoolType -> 1
        | StringType -> 8
        | UnitType -> 1
        | ArrayType _ -> 8
        | FunctionType(_, _) -> 8
        | StructType fields ->
            if fields.IsEmpty then 1
            else fields |> List.map (snd >> calculateTypeAlignment) |> List.max
        | UnionType _ -> 8  // Conservative alignment

/// Layout strategy analysis
module LayoutStrategyAnalysis =
    
    /// Analyzes a general union for layout strategy
    let analyzeGeneralUnion (name: string) (cases: (string * OakType option) list) : CompilerResult<UnionLayoutStrategy> =
        try
            // Check if all cases are nullary (enum-like)
            let allNullary = cases |> List.forall (fun (_, optType) -> optType.IsNone)
            
            if allNullary then
                let enumValues = [0 .. cases.Length - 1]
                let underlyingType = 
                    if cases.Length <= 256 then IntType
                    else IntType
                Success (EnumOptimization(enumValues, underlyingType))
            else
                // Calculate payload sizes for tagged union
                let payloadSizes = 
                    cases 
                    |> List.map (fun (_, optType) ->
                        match optType with
                        | Some t -> TypeSizeCalculation.calculateTypeSize t
                        | None -> 0)
                
                let maxPayloadSize = List.max (0 :: payloadSizes)
                let tagSize = 
                    if cases.Length <= 256 then 1
                    elif cases.Length <= 65536 then 2
                    else 4
                
                let alignment = max tagSize (if maxPayloadSize > 0 then 8 else tagSize)
                Success (TaggedUnion(tagSize, maxPayloadSize, alignment))
        
        with ex ->
            CompilerFailure [ConversionError("general union analysis", name, "layout strategy", ex.Message)]
    
    /// Analyzes union cases to determine optimal layout strategy
    let analyzeUnionStrategy (name: string) (cases: (string * OakType option) list) : CompilerResult<UnionLayoutStrategy> =
        try
            if cases.IsEmpty then
                Success (EmptyUnion("Empty unions are not supported"))
            
            elif cases.Length = 1 then
                match cases.[0] with
                | (_, Some caseType) ->
                    Success (SingleCase(caseType, true))
                | (_, None) ->
                    Success (EnumOptimization([0], IntType))
            
            elif cases.Length = 2 then
                match cases with
                | [("None", None); ("Some", Some someType)] | [("Some", Some someType); ("None", None)] ->
                    Success (OptionOptimization(someType, true))
                | _ ->
                    analyzeGeneralUnion name cases
            
            else
                analyzeGeneralUnion name cases
        
        with ex ->
            CompilerFailure [ConversionError("layout strategy analysis", name, "layout strategy", ex.Message)]

/// Layout computation
module LayoutComputation =
    
    /// Computes the memory layout for a union based on its strategy
    let computeUnionLayout (name: string) (strategy: UnionLayoutStrategy) (cases: (string * OakType option) list) : CompilerResult<UnionLayout> =
        try
            match strategy with
            | SingleCase(caseType, isNewtype) ->
                let size = TypeSizeCalculation.calculateTypeSize caseType
                let alignment = TypeSizeCalculation.calculateTypeAlignment caseType
                Success {
                    Strategy = strategy
                    TotalSize = size
                    Alignment = alignment
                    TagOffset = -1  // No tag
                    PayloadOffset = 0
                    CaseMap = Map.ofList [(fst cases.[0], 0)]
                    IsZeroAllocation = true
                }
            
            | OptionOptimization(someType, useNull) ->
                let size = TypeSizeCalculation.calculateTypeSize someType
                let alignment = TypeSizeCalculation.calculateTypeAlignment someType
                let caseMap = 
                    cases 
                    |> List.mapi (fun i (caseName, _) -> (caseName, i))
                    |> Map.ofList
                Success {
                    Strategy = strategy
                    TotalSize = size
                    Alignment = alignment
                    TagOffset = -1  // No explicit tag
                    PayloadOffset = 0
                    CaseMap = caseMap
                    IsZeroAllocation = true
                }
            
            | EnumOptimization(enumValues, underlyingType) ->
                let size = TypeSizeCalculation.calculateTypeSize underlyingType
                let alignment = TypeSizeCalculation.calculateTypeAlignment underlyingType
                let caseMap = 
                    cases 
                    |> List.mapi (fun i (caseName, _) -> (caseName, i))
                    |> Map.ofList
                Success {
                    Strategy = strategy
                    TotalSize = size
                    Alignment = alignment
                    TagOffset = 0
                    PayloadOffset = -1  // No payload
                    CaseMap = caseMap
                    IsZeroAllocation = true
                }
            
            | TaggedUnion(tagSize, maxPayloadSize, alignment) ->
                let tagOffset = 0
                let payloadOffset = (tagSize + alignment - 1) / alignment * alignment  // Align payload
                let totalSize = payloadOffset + maxPayloadSize
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
                    IsZeroAllocation = true  // Stack-allocated
                }
            
            | EmptyUnion(errorMsg) ->
                CompilerFailure [ConversionError("layout computation", name, "valid layout", errorMsg)]
        
        with ex ->
            CompilerFailure [ConversionError("layout computation", name, "union layout", ex.Message)]

/// Type transformation
module TypeTransformation =
    
    /// Transforms a union type declaration to use fixed layout
    let transformUnionTypeDeclaration (name: string) (cases: (string * OakType option) list) (state: LayoutAnalysisState) : CompilerResult<OakType * LayoutAnalysisState> =
        match LayoutStrategyAnalysis.analyzeUnionStrategy name cases with
        | Success strategy ->
            match LayoutComputation.computeUnionLayout name strategy cases with
            | Success layout ->
                // Update state with computed layout
                let newState = {
                    state with 
                        UnionLayouts = Map.add name layout state.UnionLayouts
                        TransformationHistory = (name, strategy.ToString()) :: state.TransformationHistory
                }
                
                // Transform to appropriate target type based on strategy
                let transformedType = 
                    match strategy with
                    | SingleCase(caseType, _) ->
                        caseType
                    
                    | OptionOptimization(someType, _) ->
                        someType  // Special handling in codegen
                    
                    | EnumOptimization(_, underlyingType) ->
                        underlyingType
                    
                    | TaggedUnion(tagSize, payloadSize, _) ->
                        // Create struct with tag and payload
                        let tagType = IntType
                        let payloadType = ArrayType(IntType)  // Byte array representation
                        StructType [("tag", tagType); ("payload", payloadType)]
                    
                    | EmptyUnion(errorMsg) ->
                        failwith errorMsg  // This should have been caught earlier
                
                let finalState = {
                    newState with TypeMappings = Map.add name transformedType newState.TypeMappings
                }
                
                Success (transformedType, finalState)
            
            | CompilerFailure errors -> CompilerFailure errors
        
        | CompilerFailure errors -> CompilerFailure errors

/// Expression and declaration transformation
module Transformation =
    
    /// Helper function to transform a list of expressions
    let rec transformExpressionList (expressions: OakExpression list) (state: LayoutAnalysisState) (acc: OakExpression list) : CompilerResult<OakExpression list * LayoutAnalysisState> =
        match expressions with
        | [] -> Success (List.rev acc, state)
        | expr :: rest ->
            match transformExpression expr state with
            | Success (transformedExpr, newState) ->
                transformExpressionList rest newState (transformedExpr :: acc)
            | CompilerFailure errors -> CompilerFailure errors
    
    /// Transforms expressions (placeholder for layout-aware transformations)
    and transformExpression (expr: OakExpression) (state: LayoutAnalysisState) : CompilerResult<OakExpression * LayoutAnalysisState> =
        match expr with

        | Match(matchExpr, cases) ->
            match transformExpression matchExpr state with
            | Success (transformedExpr, state1) ->
                // Helper function to transform a single case
                let rec transformCases remaining accCases currentState =
                    match remaining with
                    | [] -> Success (List.rev accCases, currentState)
                    | (pattern, caseExpr) :: rest ->
                        match transformExpression caseExpr currentState with
                        | Success (transformedCaseExpr, newState) ->
                            transformCases rest ((pattern, transformedCaseExpr) :: accCases) newState
                        | CompilerFailure errors -> CompilerFailure errors
                
                // Transform all cases
                match transformCases cases [] state1 with
                | Success (transformedCases, finalState) ->
                    Success (Match(transformedExpr, transformedCases), finalState)
                | CompilerFailure errors -> CompilerFailure errors
            | CompilerFailure errors -> CompilerFailure errors
            
        | Variable _ | Literal _ ->
            Success (expr, state)
        
        | Application(func, args) ->
            match transformExpression func state with
            | Success (transformedFunc, state1) ->
                transformExpressionList args state1 []
                |> function
                   | Success (transformedArgs, finalState) ->
                       Success (Application(transformedFunc, transformedArgs), finalState)
                   | CompilerFailure errors -> CompilerFailure errors
            | CompilerFailure errors -> CompilerFailure errors
        
        | Let(name, value, body) ->
            match transformExpression value state with
            | Success (transformedValue, state1) ->
                match transformExpression body state1 with
                | Success (transformedBody, state2) ->
                    Success (Let(name, transformedValue, transformedBody), state2)
                | CompilerFailure errors -> CompilerFailure errors
            | CompilerFailure errors -> CompilerFailure errors
        
        | IfThenElse(cond, thenExpr, elseExpr) ->
            match transformExpression cond state with
            | Success (transformedCond, state1) ->
                match transformExpression thenExpr state1 with
                | Success (transformedThen, state2) ->
                    match transformExpression elseExpr state2 with
                    | Success (transformedElse, state3) ->
                        Success (IfThenElse(transformedCond, transformedThen, transformedElse), state3)
                    | CompilerFailure errors -> CompilerFailure errors
                | CompilerFailure errors -> CompilerFailure errors
            | CompilerFailure errors -> CompilerFailure errors
        
        | Sequential(first, second) ->
            match transformExpression first state with
            | Success (transformedFirst, state1) ->
                match transformExpression second state1 with
                | Success (transformedSecond, state2) ->
                    Success (Sequential(transformedFirst, transformedSecond), state2)
                | CompilerFailure errors -> CompilerFailure errors
            | CompilerFailure errors -> CompilerFailure errors
        
        | FieldAccess(target, fieldName) ->
            match transformExpression target state with
            | Success (transformedTarget, state1) ->
                Success (FieldAccess(transformedTarget, fieldName), state1)
            | CompilerFailure errors -> CompilerFailure errors
        
        | MethodCall(target, methodName, args) ->
            match transformExpression target state with
            | Success (transformedTarget, state1) ->
                transformExpressionList args state1 []
                |> function
                   | Success (transformedArgs, finalState) ->
                       Success (MethodCall(transformedTarget, methodName, transformedArgs), finalState)
                   | CompilerFailure errors -> CompilerFailure errors
            | CompilerFailure errors -> CompilerFailure errors
        
        | Lambda(params', body) ->
            match transformExpression body state with
            | Success (transformedBody, state1) ->
                Success (Lambda(params', transformedBody), state1)
            | CompilerFailure errors -> CompilerFailure errors
        
        | IOOperation(ioType, args) ->
            transformExpressionList args state []
            |> function
               | Success (transformedArgs, finalState) ->
                   Success (IOOperation(ioType, transformedArgs), finalState)
               | CompilerFailure errors -> CompilerFailure errors
    
    /// Transforms a declaration to use fixed layouts
    let transformDeclaration (decl: OakDeclaration) (state: LayoutAnalysisState) : CompilerResult<OakDeclaration * LayoutAnalysisState> =
        match decl with
        | FunctionDecl(name, params', returnType, body) ->
            match transformExpression body state with
            | Success (transformedBody, newState) ->
                Success (FunctionDecl(name, params', returnType, transformedBody), newState)
            | CompilerFailure errors -> CompilerFailure errors
        
        | EntryPoint(expr) ->
            match transformExpression expr state with
            | Success (transformedExpr, newState) ->
                Success (EntryPoint(transformedExpr), newState)
            | CompilerFailure errors -> CompilerFailure errors
        
        | TypeDecl(name, oakType) ->
            match oakType with
            | UnionType cases ->
                match TypeTransformation.transformUnionTypeDeclaration name cases state with
                | Success (transformedType, newState) ->
                    Success (TypeDecl(name, transformedType), newState)
                | CompilerFailure errors -> CompilerFailure errors
            | _ ->
                Success (decl, state)
        
        | ExternalDecl(_, _, _, _) ->
            Success (decl, state)

/// Module transformation
module ModuleTransformation =
    
    /// Helper function to transform a list of declarations
    let rec transformDeclarationList (declarations: OakDeclaration list) (state: LayoutAnalysisState) (acc: OakDeclaration list) : CompilerResult<OakDeclaration list * LayoutAnalysisState> =
        match declarations with
        | [] -> Success (List.rev acc, state)
        | decl :: rest ->
            match Transformation.transformDeclaration decl state with
            | Success (transformedDecl, newState) ->
                transformDeclarationList rest newState (transformedDecl :: acc)
            | CompilerFailure errors -> CompilerFailure errors
    
    /// Transforms a complete module
    let transformModule (module': OakModule) (state: LayoutAnalysisState) : CompilerResult<OakModule * LayoutAnalysisState> =
        match transformDeclarationList module'.Declarations state [] with
        | Success (transformedDeclarations, finalState) ->
            Success ({ module' with Declarations = transformedDeclarations }, finalState)
        | CompilerFailure errors -> CompilerFailure errors

/// Layout validation
module LayoutValidation =
    
    /// Validates that a layout is zero-allocation
    let validateZeroAllocationLayout (layout: UnionLayout) : CompilerResult<unit> =
        if layout.IsZeroAllocation then
            Success ()
        else
            CompilerFailure [ConversionError("layout validation", "computed layout", "zero-allocation layout", "Layout may cause heap allocations")]
    
    /// Validates that all computed layouts are zero-allocation
    let validateAllLayoutsZeroAllocation (state: LayoutAnalysisState) : CompilerResult<unit> =
        let validationResults = 
            state.UnionLayouts
            |> Map.toList
            |> List.map (fun (name, layout) ->
                match validateZeroAllocationLayout layout with
                | Success () -> Success ()
                | CompilerFailure errors -> 
                    CompilerFailure (errors |> List.map (fun e -> 
                        ConversionError("layout validation", name, "zero-allocation", e.ToString()))))
        
        match ResultHelpers.combineResults validationResults with
        | Success _ -> Success ()
        | CompilerFailure errors -> CompilerFailure errors

/// Helper function to transform a list of modules
let rec transformModuleList (modules: OakModule list) (state: LayoutAnalysisState) (acc: OakModule list) : CompilerResult<OakModule list * LayoutAnalysisState> =
    match modules with
    | [] -> Success (List.rev acc, state)
    | module' :: rest ->
        match ModuleTransformation.transformModule module' state with
        | Success (transformedModule, newState) ->
            transformModuleList rest newState (transformedModule :: acc)
        | CompilerFailure errors -> CompilerFailure errors

/// Main fixed layout compilation entry point
let compileFixedLayouts (program: OakProgram) : CompilerResult<OakProgram> =
    if program.Modules.IsEmpty then
        CompilerFailure [ConversionError("fixed layout compilation", "empty program", "program with fixed layouts", "Program must contain at least one module")]
    else
        let initialState = {
            UnionLayouts = Map.empty
            TypeMappings = Map.empty
            TransformationHistory = []
        }
        
        // Transform all modules
        match transformModuleList program.Modules initialState [] with
        | Success (transformedModules, finalState) ->
            // Validate all layouts are zero-allocation
            match LayoutValidation.validateAllLayoutsZeroAllocation finalState with
            | Success () ->
                Success { program with Modules = transformedModules }
            | CompilerFailure errors -> CompilerFailure errors
        | CompilerFailure errors -> CompilerFailure errors

/// Gets the computed layout for a union type by name
let getUnionLayout (program: OakProgram) (unionName: string) : CompilerResult<UnionLayout option> =
    // Find the union type declaration
    let unionDeclaration = 
        program.Modules
        |> List.collect (fun m -> m.Declarations)
        |> List.tryFind (function
            | TypeDecl(name, UnionType _) when name = unionName -> true
            | _ -> false)
    
    match unionDeclaration with
    | Some (TypeDecl(name, UnionType cases)) ->
        match LayoutStrategyAnalysis.analyzeUnionStrategy name cases with
        | Success strategy ->
            match LayoutComputation.computeUnionLayout name strategy cases with
            | Success layout ->
                Success (Some layout)
            | CompilerFailure errors -> CompilerFailure errors
        | CompilerFailure errors -> CompilerFailure errors
    | _ -> 
        Success None

/// Helper function to validate a list of unions
let rec validateUnionList (unions: (string * (string * OakType option) list) list) (acc: bool list) : CompilerResult<bool list> =
    match unions with
    | [] -> Success (List.rev acc)
    | (name, cases) :: rest ->
        // Create a minimal program to validate this union
        let dummyModule = { Name = "temp"; Declarations = [TypeDecl(name, UnionType cases)] }
        let dummyProgram = { Modules = [dummyModule] }
        
        match getUnionLayout dummyProgram name with
        | Success (Some layout) -> 
            match LayoutValidation.validateZeroAllocationLayout layout with
            | Success () -> validateUnionList rest (layout.IsZeroAllocation :: acc)
            | CompilerFailure _ -> validateUnionList rest (false :: acc)
        | Success None -> 
            validateUnionList rest (true :: acc)  // No layout computed means no issues
        | CompilerFailure errors -> CompilerFailure errors

/// Validates that all union layouts in a program are zero-allocation
let validateZeroAllocationLayouts (program: OakProgram) : CompilerResult<bool> =
    let unionTypes = 
        program.Modules
        |> List.collect (fun m -> m.Declarations)
        |> List.choose (function
            | TypeDecl(name, UnionType cases) -> Some (name, cases)
            | _ -> None)
    
    match validateUnionList unionTypes [] with
    | Success results -> Success (List.forall id results)
    | CompilerFailure errors -> CompilerFailure errors

/// Helper function to collect layout statistics
let rec collectStatistics (unions: (string * (string * OakType option) list) list) (acc: (string * string) list) : CompilerResult<(string * string) list> =
    match unions with
    | [] -> Success (List.rev acc)
    | (name, cases) :: rest ->
        // Create a minimal program to analyze this union
        let dummyModule = { Name = "temp"; Declarations = [TypeDecl(name, UnionType cases)] }
        let dummyProgram = { Modules = [dummyModule] }
        
        match getUnionLayout dummyProgram name with
        | Success (Some layout) ->
            let strategyName = 
                match layout.Strategy with
                | SingleCase(_, _) -> "SingleCase"
                | OptionOptimization(_, _) -> "OptionOptimization"
                | EnumOptimization(_, _) -> "EnumOptimization"
                | TaggedUnion(_, _, _) -> "TaggedUnion"
                | EmptyUnion(_) -> "EmptyUnion"
            let statistic = sprintf "%s (Size: %d, Alignment: %d)" strategyName layout.TotalSize layout.Alignment
            collectStatistics rest ((name, statistic) :: acc)
        | Success None ->
            collectStatistics rest acc
        | CompilerFailure errors -> CompilerFailure errors

/// Analyzes layout statistics for a program
let analyzeLayoutStatistics (program: OakProgram) : CompilerResult<Map<string, string>> =
    try
        let unionTypes = 
            program.Modules
            |> List.collect (fun m -> m.Declarations)
            |> List.choose (function
                | TypeDecl(name, UnionType cases) -> Some (name, cases)
                | _ -> None)
        
        match collectStatistics unionTypes [] with
        | Success statistics -> Success (Map.ofList statistics)
        | CompilerFailure errors -> CompilerFailure errors
    
    with ex ->
        CompilerFailure [ConversionError("layout statistics", "program analysis", "layout statistics", ex.Message)]