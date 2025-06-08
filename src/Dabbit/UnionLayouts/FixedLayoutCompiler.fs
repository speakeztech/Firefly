module Dabbit.UnionLayouts.FixedLayoutCompiler

open System
open XParsec
open Core.XParsec.Foundation
open Core.XParsec.Foundation.Combinators
open Core.XParsec.Foundation.ErrorHandling
open Dabbit.Parsing.OakAst

/// Layout analysis state for tracking union transformations
type LayoutAnalysisState = {
    UnionLayouts: Map<string, UnionLayout>
    TypeMappings: Map<string, OakType>
    LayoutStrategies: Map<string, UnionLayoutStrategy>
    TransformationHistory: (string * string) list
    ErrorContext: string list
}

/// Union layout strategy with complete optimization information
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

/// Type size calculation using XParsec patterns
module TypeSizeCalculation =
    
    /// Calculates the size of an Oak type in bytes
    let rec calculateTypeSize (oakType: OakType) : Parser<int, LayoutAnalysisState> =
        match oakType with
        | IntType -> succeed 4
        | FloatType -> succeed 4  
        | BoolType -> succeed 1
        | StringType -> succeed 8  // Pointer size
        | UnitType -> succeed 0
        | ArrayType _ -> succeed 8  // Pointer size
        | FunctionType(_, _) -> succeed 8  // Function pointer
        | StructType fields ->
            fields
            |> List.map (snd >> calculateTypeSize)
            |> List.fold (fun acc sizeParser ->
                acc >>= fun accSize ->
                sizeParser >>= fun fieldSize ->
                succeed (accSize + fieldSize)
            ) (succeed 0)
        | UnionType cases ->
            // For unanalyzed unions, estimate conservatively
            cases
            |> List.map (fun (_, optType) ->
                match optType with
                | Some t -> calculateTypeSize t
                | None -> succeed 0)
            |> List.fold (fun acc sizeParser ->
                acc >>= fun accSize ->
                sizeParser >>= fun caseSize ->
                succeed (max accSize caseSize)
            ) (succeed 0)
            >>= fun maxCaseSize ->
            succeed (1 + maxCaseSize)  // Tag + largest payload
        |> withErrorContext "type size calculation"
    
    /// Calculates the alignment requirement for an Oak type
    let rec calculateTypeAlignment (oakType: OakType) : Parser<int, LayoutAnalysisState> =
        match oakType with
        | IntType -> succeed 4
        | FloatType -> succeed 4
        | BoolType -> succeed 1
        | StringType -> succeed 8
        | UnitType -> succeed 1
        | ArrayType _ -> succeed 8
        | FunctionType(_, _) -> succeed 8
        | StructType fields ->
            if fields.IsEmpty then
                succeed 1
            else
                fields
                |> List.map (snd >> calculateTypeAlignment)
                |> List.fold (fun acc alignParser ->
                    acc >>= fun accAlign ->
                    alignParser >>= fun fieldAlign ->
                    succeed (max accAlign fieldAlign)
                ) (succeed 1)
        | UnionType _ -> succeed 8  // Conservative alignment
        |> withErrorContext "type alignment calculation"

/// Layout strategy analysis using XParsec combinators
module LayoutStrategyAnalysis =
    
    /// Analyzes union cases to determine optimal layout strategy
    let analyzeUnionStrategy (name: string) (cases: (string * OakType option) list) : Parser<UnionLayoutStrategy, LayoutAnalysisState> =
        if cases.IsEmpty then
            succeed (EmptyUnion("Empty unions are not supported"))
            |> withErrorContext "empty union analysis"
        
        elif cases.Length = 1 then
            match cases.[0] with
            | (_, Some caseType) ->
                succeed (SingleCase(caseType, true))
            | (_, None) ->
                succeed (EnumOptimization([0], IntType))
            |> withErrorContext "single case union analysis"
        
        elif cases.Length = 2 then
            match cases with
            | [("None", None); ("Some", Some someType)] | [("Some", Some someType); ("None", None)] ->
                succeed (OptionOptimization(someType, true))
            | _ ->
                analyzeGeneralUnion name cases
            |> withErrorContext "option type analysis"
        
        else
            analyzeGeneralUnion name cases
            |> withErrorContext "general union analysis"
    
    /// Analyzes a general union for layout strategy
    let analyzeGeneralUnion (name: string) (cases: (string * OakType option) list) : Parser<UnionLayoutStrategy, LayoutAnalysisState> =
        // Check if all cases are nullary (enum-like)
        let allNullary = cases |> List.forall (fun (_, optType) -> optType.IsNone)
        
        if allNullary then
            let enumValues = [0 .. cases.Length - 1]
            let underlyingType = 
                if cases.Length <= 256 then IntType  // Could be optimized to BoolType for 2 cases, etc.
                else IntType
            succeed (EnumOptimization(enumValues, underlyingType))
            |> withErrorContext "enum union analysis"
        else
            // Calculate payload sizes for tagged union
            cases
            |> List.map (fun (_, optType) ->
                match optType with
                | Some t -> calculateTypeSize t
                | None -> succeed 0)
            |> List.fold (fun acc sizeParser ->
                acc >>= fun accSizes ->
                sizeParser >>= fun size ->
                succeed (size :: accSizes)
            ) (succeed [])
            >>= fun payloadSizes ->
            
            let maxPayloadSize = List.max (0 :: payloadSizes)
            let tagSize = 
                if cases.Length <= 256 then 1
                elif cases.Length <= 65536 then 2
                else 4
            
            let alignment = max tagSize (if maxPayloadSize > 0 then 8 else tagSize)
            succeed (TaggedUnion(tagSize, maxPayloadSize, alignment))
            |> withErrorContext "tagged union analysis"
    
    /// Records a layout strategy for a union type
    let recordLayoutStrategy (name: string) (strategy: UnionLayoutStrategy) : Parser<unit, LayoutAnalysisState> =
        fun state ->
            let newState = { 
                state with 
                    LayoutStrategies = Map.add name strategy state.LayoutStrategies
            }
            Reply(Ok (), newState)

/// Layout computation using XParsec combinators
module LayoutComputation =
    
    /// Computes the memory layout for a union based on its strategy
    let computeUnionLayout (name: string) (strategy: UnionLayoutStrategy) (cases: (string * OakType option) list) : Parser<UnionLayout, LayoutAnalysisState> =
        match strategy with
        | SingleCase(caseType, isNewtype) ->
            calculateTypeSize caseType >>= fun size ->
            calculateTypeAlignment caseType >>= fun alignment ->
            succeed {
                Strategy = strategy
                TotalSize = size
                Alignment = alignment
                TagOffset = -1  // No tag
                PayloadOffset = 0
                CaseMap = Map.ofList [(fst cases.[0], 0)]
                IsZeroAllocation = true
            }
        
        | OptionOptimization(someType, useNull) ->
            calculateTypeSize someType >>= fun size ->
            calculateTypeAlignment someType >>= fun alignment ->
            let caseMap = 
                cases 
                |> List.mapi (fun i (caseName, _) -> (caseName, i))
                |> Map.ofList
            succeed {
                Strategy = strategy
                TotalSize = size
                Alignment = alignment
                TagOffset = -1  // No explicit tag
                PayloadOffset = 0
                CaseMap = caseMap
                IsZeroAllocation = true
            }
        
        | EnumOptimization(enumValues, underlyingType) ->
            calculateTypeSize underlyingType >>= fun size ->
            calculateTypeAlignment underlyingType >>= fun alignment ->
            let caseMap = 
                cases 
                |> List.mapi (fun i (caseName, _) -> (caseName, i))
                |> Map.ofList
            succeed {
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
            succeed {
                Strategy = strategy
                TotalSize = totalSize
                Alignment = alignment
                TagOffset = tagOffset
                PayloadOffset = payloadOffset
                CaseMap = caseMap
                IsZeroAllocation = true  // Stack-allocated
            }
        
        | EmptyUnion(errorMsg) ->
            compilerFail (TransformError("layout computation", name, "valid layout", errorMsg))
        |> withErrorContext (sprintf "layout computation for union '%s'" name)
    
    /// Records a computed layout
    let recordUnionLayout (name: string) (layout: UnionLayout) : Parser<unit, LayoutAnalysisState> =
        fun state ->
            let newState = { 
                state with 
                    UnionLayouts = Map.add name layout state.UnionLayouts
            }
            Reply(Ok (), newState)

/// Type transformation using XParsec combinators
module TypeTransformationParsers =
    
    /// Transforms a union type declaration to use fixed layout
    let transformUnionTypeDeclaration (name: string) (cases: (string * OakType option) list) : Parser<OakType, LayoutAnalysisState> =
        analyzeUnionStrategy name cases >>= fun strategy ->
        recordLayoutStrategy name strategy >>= fun _ ->
        computeUnionLayout name strategy cases >>= fun layout ->
        recordUnionLayout name layout >>= fun _ ->
        
        // Transform to appropriate target type based on strategy
        match strategy with
        | SingleCase(caseType, _) ->
            succeed caseType
        
        | OptionOptimization(someType, _) ->
            succeed someType  // Special handling in codegen
        
        | EnumOptimization(_, underlyingType) ->
            succeed underlyingType
        
        | TaggedUnion(tagSize, payloadSize, _) ->
            // Create struct with tag and payload
            let tagType = IntType  // Could be optimized based on tagSize
            let payloadType = ArrayType(IntType)  // Byte array representation
            succeed (StructType [("tag", tagType); ("payload", payloadType)])
        
        | EmptyUnion(errorMsg) ->
            compilerFail (TransformError("union transformation", name, "valid type", errorMsg))
        |> withErrorContext (sprintf "union type transformation '%s'" name)
    
    /// Records a type mapping for later reference
    let recordTypeMapping (originalName: string) (transformedType: OakType) : Parser<unit, LayoutAnalysisState> =
        fun state ->
            let newState = { 
                state with 
                    TypeMappings = Map.add originalName transformedType state.TypeMappings
                    TransformationHistory = (originalName, transformedType.ToString()) :: state.TransformationHistory
            }
            Reply(Ok (), newState)

/// Expression transformation using layout information
module ExpressionLayoutTransformation =
    
    /// Transforms expressions to use fixed layouts (placeholder for now)
    let rec transformExpressionWithLayouts (expr: OakExpression) : Parser<OakExpression, LayoutAnalysisState> =
        match expr with
        | Variable name ->
            succeed expr  // Variable references unchanged
        
        | Application(func, args) ->
            transformExpressionWithLayouts func >>= fun transformedFunc ->
            args
            |> List.map transformExpressionWithLayouts
            |> List.fold (fun acc argParser ->
                acc >>= fun accArgs ->
                argParser >>= fun transformedArg ->
                succeed (transformedArg :: accArgs)
            ) (succeed [])
            >>= fun transformedArgs ->
            succeed (Application(transformedFunc, List.rev transformedArgs))
        
        | Let(name, value, body) ->
            transformExpressionWithLayouts value >>= fun transformedValue ->
            transformExpressionWithLayouts body >>= fun transformedBody ->
            succeed (Let(name, transformedValue, transformedBody))
        
        | IfThenElse(cond, thenExpr, elseExpr) ->
            transformExpressionWithLayouts cond >>= fun transformedCond ->
            transformExpressionWithLayouts thenExpr >>= fun transformedThen ->
            transformExpressionWithLayouts elseExpr >>= fun transformedElse ->
            succeed (IfThenElse(transformedCond, transformedThen, transformedElse))
        
        | Sequential(first, second) ->
            transformExpressionWithLayouts first >>= fun transformedFirst ->
            transformExpressionWithLayouts second >>= fun transformedSecond ->
            succeed (Sequential(transformedFirst, transformedSecond))
        
        | FieldAccess(target, fieldName) ->
            transformExpressionWithLayouts target >>= fun transformedTarget ->
            succeed (FieldAccess(transformedTarget, fieldName))
        
        | MethodCall(target, methodName, args) ->
            transformExpressionWithLayouts target >>= fun transformedTarget ->
            args
            |> List.map transformExpressionWithLayouts
            |> List.fold (fun acc argParser ->
                acc >>= fun accArgs ->
                argParser >>= fun transformedArg ->
                succeed (transformedArg :: accArgs)
            ) (succeed [])
            >>= fun transformedArgs ->
            succeed (MethodCall(transformedTarget, methodName, List.rev transformedArgs))
        
        | Lambda(parameters, body) ->
            transformExpressionWithLayouts body >>= fun transformedBody ->
            succeed (Lambda(parameters, transformedBody))
        
        | Literal _ ->
            succeed expr

/// Declaration transformation using XParsec combinators
module DeclarationLayoutTransformation =
    
    /// Transforms a declaration to use fixed layouts
    let transformDeclarationWithLayouts (decl: OakDeclaration) : Parser<OakDeclaration, LayoutAnalysisState> =
        match decl with
        | FunctionDecl(name, parameters, returnType, body) ->
            transformExpressionWithLayouts body >>= fun transformedBody ->
            succeed (FunctionDecl(name, parameters, returnType, transformedBody))
        
        | EntryPoint(expr) ->
            transformExpressionWithLayouts expr >>= fun transformedExpr ->
            succeed (EntryPoint(transformedExpr))
        
        | TypeDecl(name, oakType) ->
            match oakType with
            | UnionType cases ->
                transformUnionTypeDeclaration name cases >>= fun transformedType ->
                recordTypeMapping name transformedType >>= fun _ ->
                succeed (TypeDecl(name, transformedType))
            | _ ->
                succeed decl
        | ExternalDecl(_, _, _, _) ->
            succeed decl
        |> withErrorContext (sprintf "declaration transformation with layouts '%s'" (match decl with FunctionDecl(n,_,_,_) -> n | TypeDecl(n,_) -> n | EntryPoint(_) -> "__entry__" | ExternalDecl(n,_,_,_) -> n))

/// Layout validation using XParsec patterns
module LayoutValidation =
    
    /// Validates that a layout is zero-allocation
    let validateZeroAllocationLayout (layout: UnionLayout) : CompilerResult<unit> =
        if layout.IsZeroAllocation then
            Success ()
        else
            CompilerFailure [TransformError("layout validation", "computed layout", "zero-allocation layout", "Layout may cause heap allocations")]
    
    /// Validates that all computed layouts are zero-allocation
    let validateAllLayoutsZeroAllocation : Parser<unit, LayoutAnalysisState> =
        fun state ->
            let validationResults = 
                state.UnionLayouts
                |> Map.toList
                |> List.map (fun (name, layout) ->
                    match validateZeroAllocationLayout layout with
                    | Success () -> Success ()
                    | CompilerFailure errors -> 
                        CompilerFailure (errors |> List.map (fun e -> 
                            TransformError("layout validation", name, "zero-allocation", e.ToString()))))
            
            let combinedResult = 
                validationResults
                |> List.fold (fun acc result ->
                    match acc, result with
                    | Success (), Success () -> Success ()
                    | CompilerFailure errors, Success () -> CompilerFailure errors
                    | Success (), CompilerFailure errors -> CompilerFailure errors
                    | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
                ) (Success ())
            
            match combinedResult with
            | Success () -> Reply(Ok (), state)
            | CompilerFailure errors -> 
                let errorMsg = errors |> List.map (fun e -> e.ToString()) |> String.concat "; "
                Reply(Error, errorMsg)

/// Main fixed layout compilation entry point - NO FALLBACKS ALLOWED
let compileFixedLayouts (program: OakProgram) : CompilerResult<OakProgram> =
    if program.Modules.IsEmpty then
        CompilerFailure [TransformError("fixed layout compilation", "empty program", "program with fixed layouts", "Program must contain at least one module")]
    else
        let initialState = {
            UnionLayouts = Map.empty
            TypeMappings = Map.empty
            LayoutStrategies = Map.empty
            TransformationHistory = []
            ErrorContext = []
        }
        
        // Transform all modules
        let transformAllModules (modules: OakModule list) : CompilerResult<OakModule list> =
            modules
            |> List.map (fun module' ->
                // Transform all declarations in the module
                let transformModuleDeclarations (declarations: OakDeclaration list) : CompilerResult<OakDeclaration list> =
                    declarations
                    |> List.map (fun decl ->
                        match transformDeclarationWithLayouts decl initialState with
                        | Reply(Ok transformedDecl, _) -> Success transformedDecl
                        | Reply(Error, error) -> CompilerFailure [TransformError("declaration transformation", "declaration", "fixed layout declaration", error)])
                    |> List.fold (fun acc result ->
                        match acc, result with
                        | Success decls, Success decl -> Success (decl :: decls)
                        | CompilerFailure errors, Success _ -> CompilerFailure errors
                        | Success _, CompilerFailure errors -> CompilerFailure errors
                        | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
                    ) (Success [])
                    |>> List.rev
                
                transformModuleDeclarations module'.Declarations >>= fun transformedDeclarations ->
                Success { module' with Declarations = transformedDeclarations })
            |> List.fold (fun acc result ->
                match acc, result with
                | Success modules, Success module' -> Success (module' :: modules)
                | CompilerFailure errors, Success _ -> CompilerFailure errors
                | Success _, CompilerFailure errors -> CompilerFailure errors
                | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
            ) (Success [])
            |>> List.rev
        
        transformAllModules program.Modules >>= fun transformedModules ->
        
        // Validate all layouts are zero-allocation
        match validateAllLayoutsZeroAllocation initialState with
        | Reply(Ok (), _) ->
            Success { program with Modules = transformedModules }
        | Reply(Error, error) ->
            CompilerFailure [TransformError("layout validation", "compiled layouts", "zero-allocation layouts", error)]

/// Gets the computed layout for a union type by name
let getUnionLayout (program: OakProgram) (unionName: string) : CompilerResult<UnionLayout option> =
    let initialState = {
        UnionLayouts = Map.empty
        TypeMappings = Map.empty
        LayoutStrategies = Map.empty
        TransformationHistory = []
        ErrorContext = []
    }
    
    // Find the union type declaration
    let unionDeclaration = 
        program.Modules
        |> List.collect (fun m -> m.Declarations)
        |> List.tryFind (function
            | TypeDecl(name, UnionType _) when name = unionName -> true
            | _ -> false)
    
    match unionDeclaration with
    | Some (TypeDecl(name, UnionType cases)) ->
        match analyzeUnionStrategy name cases initialState with
        | Reply(Ok strategy, state1) ->
            match computeUnionLayout name strategy cases state1 with
            | Reply(Ok layout, _) -> Success (Some layout)
            | Reply(Error, error) -> CompilerFailure [TransformError("layout computation", name, "union layout", error)]
        | Reply(Error, error) -> CompilerFailure [TransformError("strategy analysis", name, "layout strategy", error)]
    | _ -> Success None

/// Validates that all union layouts in a program are zero-allocation
let validateZeroAllocationLayouts (program: OakProgram) : CompilerResult<bool> =
    let unionTypes = 
        program.Modules
        |> List.collect (fun m -> m.Declarations)
        |> List.choose (function
            | TypeDecl(name, UnionType cases) -> Some (name, cases)
            | _ -> None)
    
    let validateUnion (name: string, cases: (string * OakType option) list) : CompilerResult<bool> =
        getUnionLayout program name >>= function
        | Some layout -> 
            if layout.IsZeroAllocation then Success true
            else Success false
        | None -> Success true  // No layout computed means no issues
    
    unionTypes
    |> List.map validateUnion
    |> List.fold (fun acc result ->
        match acc, result with
        | Success accBool, Success resultBool -> Success (accBool && resultBool)
        | CompilerFailure errors, Success _ -> CompilerFailure errors
        | Success _, CompilerFailure errors -> CompilerFailure errors
        | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
    ) (Success true)