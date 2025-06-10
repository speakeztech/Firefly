module Dabbit.UnionLayouts.FixedLayoutCompiler

open System
open Core.XParsec.Foundation
open Dabbit.Parsing.OakAst

/// Layout analysis for tracking union transformations
type LayoutAnalysisContext = {
    UnionLayouts: Map<string, UnionLayout>
    TypeMappings: Map<string, OakType>
    LayoutStrategies: Map<string, UnionLayoutStrategy>
    TransformationHistory: (string * string) list
}

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

/// Type size calculation using simplified logic
module TypeSizeCalculation =
    
    /// Calculates the size of an Oak type in bytes
    let calculateTypeSize (oakType: OakType) : CompilerResult<int> =
        match oakType with
        | IntType -> Success 4
        | FloatType -> Success 4  
        | BoolType -> Success 1
        | StringType -> Success 8  // Pointer size
        | UnitType -> Success 0
        | ArrayType _ -> Success 8  // Pointer size
        | FunctionType(_, _) -> Success 8  // Function pointer
        | StructType fields ->
            fields
            |> List.map (snd >> calculateTypeSize)
            |> List.fold (fun acc sizeResult ->
                match acc, sizeResult with
                | Success accSize, Success fieldSize -> Success (accSize + fieldSize)
                | CompilerFailure errors, Success _ -> CompilerFailure errors
                | Success _, CompilerFailure errors -> CompilerFailure errors
                | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
            ) (Success 0)
        | UnionType cases ->
            // For unanalyzed unions, estimate conservatively
            cases
            |> List.map (fun (_, optType) ->
                match optType with
                | Some t -> calculateTypeSize t
                | None -> Success 0)
            |> List.fold (fun acc sizeResult ->
                match acc, sizeResult with
                | Success accSize, Success caseSize -> Success (max accSize caseSize)
                | CompilerFailure errors, Success _ -> CompilerFailure errors
                | Success _, CompilerFailure errors -> CompilerFailure errors
                | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
            ) (Success 0)
            |> fun maxResult ->
                match maxResult with
                | Success maxCaseSize -> Success (1 + maxCaseSize)  // Tag + largest payload
                | CompilerFailure errors -> CompilerFailure errors
    
    /// Calculates the alignment requirement for an Oak type
    let calculateTypeAlignment (oakType: OakType) : CompilerResult<int> =
        match oakType with
        | IntType -> Success 4
        | FloatType -> Success 4
        | BoolType -> Success 1
        | StringType -> Success 8
        | UnitType -> Success 1
        | ArrayType _ -> Success 8
        | FunctionType(_, _) -> Success 8
        | StructType fields ->
            if fields.IsEmpty then
                Success 1
            else
                fields
                |> List.map (snd >> calculateTypeAlignment)
                |> List.fold (fun acc alignResult ->
                    match acc, alignResult with
                    | Success accAlign, Success fieldAlign -> Success (max accAlign fieldAlign)
                    | CompilerFailure errors, Success _ -> CompilerFailure errors
                    | Success _, CompilerFailure errors -> CompilerFailure errors
                    | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
                ) (Success 1)
        | UnionType _ -> Success 8  // Conservative alignment

/// Layout strategy analysis
module LayoutStrategyAnalysis =
    
    /// Analyzes union cases to determine optimal layout strategy
    let analyzeUnionStrategy (name: string) (cases: (string * OakType option) list) : CompilerResult<UnionLayoutStrategy> =
        if cases.IsEmpty then
            CompilerFailure [CompilerError("layout analysis", "Empty unions are not supported", None)]
        
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
    
    /// Analyzes a general union for layout strategy
    and analyzeGeneralUnion (name: string) (cases: (string * OakType option) list) : CompilerResult<UnionLayoutStrategy> =
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
            cases
            |> List.map (fun (_, optType) ->
                match optType with
                | Some t -> TypeSizeCalculation.calculateTypeSize t
                | None -> Success 0)
            |> List.fold (fun acc sizeResult ->
                match acc, sizeResult with
                | Success accSizes, Success size -> Success (size :: accSizes)
                | CompilerFailure errors, Success _ -> CompilerFailure errors
                | Success _, CompilerFailure errors -> CompilerFailure errors
                | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
            ) (Success [])
            |> fun payloadSizesResult ->
                match payloadSizesResult with
                | Success payloadSizes ->
                    let maxPayloadSize = List.max (0 :: payloadSizes)
                    let tagSize = 
                        if cases.Length <= 256 then 1
                        elif cases.Length <= 65536 then 2
                        else 4
                    
                    let alignment = max tagSize (if maxPayloadSize > 0 then 8 else tagSize)
                    Success (TaggedUnion(tagSize, maxPayloadSize, alignment))
                | CompilerFailure errors -> CompilerFailure errors

/// Layout computation
module LayoutComputation =
    
    /// Computes the memory layout for a union based on its strategy
    let computeUnionLayout (name: string) (strategy: UnionLayoutStrategy) (cases: (string * OakType option) list) : CompilerResult<UnionLayout> =
        match strategy with
        | SingleCase(caseType, isNewtype) ->
            TypeSizeCalculation.calculateTypeSize caseType >>= fun size ->
            TypeSizeCalculation.calculateTypeAlignment caseType >>= fun alignment ->
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
            TypeSizeCalculation.calculateTypeSize someType >>= fun size ->
            TypeSizeCalculation.calculateTypeAlignment someType >>= fun alignment ->
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
            TypeSizeCalculation.calculateTypeSize underlyingType >>= fun size ->
            TypeSizeCalculation.calculateTypeAlignment underlyingType >>= fun alignment ->
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
            CompilerFailure [CompilerError("layout computation", errorMsg, None)]

/// Type transformation
module TypeTransformation =
    
    /// Transforms a union type declaration to use fixed layout
    let transformUnionTypeDeclaration (name: string) (cases: (string * OakType option) list) : CompilerResult<OakType> =
        LayoutStrategyAnalysis.analyzeUnionStrategy name cases >>= fun strategy ->
        LayoutComputation.computeUnionLayout name strategy cases >>= fun layout ->
        
        // Transform to appropriate target type based on strategy
        match strategy with
        | SingleCase(caseType, _) ->
            Success caseType
        
        | OptionOptimization(someType, _) ->
            Success someType  // Special handling in codegen
        
        | EnumOptimization(_, underlyingType) ->
            Success underlyingType
        
        | TaggedUnion(tagSize, payloadSize, _) ->
            // Create struct with tag and payload
            let tagType = IntType  // Could be optimized based on tagSize
            let payloadType = ArrayType(IntType)  // Byte array representation
            Success (StructType [("tag", tagType); ("payload", payloadType)])
        
        | EmptyUnion(errorMsg) ->
            CompilerFailure [CompilerError("union transformation", errorMsg, None)]

/// Expression transformation using layout information
module ExpressionTransformation =
    
    /// Transforms expressions to use fixed layouts (simplified for POC)
    let rec transformExpressionWithLayouts (expr: OakExpression) : CompilerResult<OakExpression> =
        match expr with
        | Variable name ->
            Success expr  // Variable references unchanged
        
        | Application(func, args) ->
            transformExpressionWithLayouts func >>= fun transformedFunc ->
            args
            |> List.map transformExpressionWithLayouts
            |> List.fold (fun acc argResult ->
                match acc, argResult with
                | Success accArgs, Success transformedArg -> Success (transformedArg :: accArgs)
                | CompilerFailure errors, Success _ -> CompilerFailure errors
                | Success _, CompilerFailure errors -> CompilerFailure errors
                | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
            ) (Success [])
            >>= fun transformedArgs ->
            Success (Application(transformedFunc, List.rev transformedArgs))
        
        | Let(name, value, body) ->
            transformExpressionWithLayouts value >>= fun transformedValue ->
            transformExpressionWithLayouts body >>= fun transformedBody ->
            Success (Let(name, transformedValue, transformedBody))
        
        | IfThenElse(cond, thenExpr, elseExpr) ->
            transformExpressionWithLayouts cond >>= fun transformedCond ->
            transformExpressionWithLayouts thenExpr >>= fun transformedThen ->
            transformExpressionWithLayouts elseExpr >>= fun transformedElse ->
            Success (IfThenElse(transformedCond, transformedThen, transformedElse))
        
        | Sequential(first, second) ->
            transformExpressionWithLayouts first >>= fun transformedFirst ->
            transformExpressionWithLayouts second >>= fun transformedSecond ->
            Success (Sequential(transformedFirst, transformedSecond))
        
        | FieldAccess(target, fieldName) ->
            transformExpressionWithLayouts target >>= fun transformedTarget ->
            Success (FieldAccess(transformedTarget, fieldName))
        
        | MethodCall(target, methodName, args) ->
            transformExpressionWithLayouts target >>= fun transformedTarget ->
            args
            |> List.map transformExpressionWithLayouts
            |> List.fold (fun acc argResult ->
                match acc, argResult with
                | Success accArgs, Success transformedArg -> Success (transformedArg :: accArgs)
                | CompilerFailure errors, Success _ -> CompilerFailure errors
                | Success _, CompilerFailure errors -> CompilerFailure errors
                | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
            ) (Success [])
            >>= fun transformedArgs ->
            Success (MethodCall(transformedTarget, methodName, List.rev transformedArgs))
        
        | Lambda(parameters, body) ->
            transformExpressionWithLayouts body >>= fun transformedBody ->
            Success (Lambda(parameters, transformedBody))
        
        | Literal _ ->
            Success expr
        
        | IOOperation(ioType, args) ->
            args
            |> List.map transformExpressionWithLayouts
            |> List.fold (fun acc argResult ->
                match acc, argResult with
                | Success accArgs, Success transformedArg -> Success (transformedArg :: accArgs)
                | CompilerFailure errors, Success _ -> CompilerFailure errors
                | Success _, CompilerFailure errors -> CompilerFailure errors
                | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
            ) (Success [])
            >>= fun transformedArgs ->
            Success (IOOperation(ioType, List.rev transformedArgs))

/// Declaration transformation
module DeclarationTransformation =
    
    /// Transforms a declaration to use fixed layouts
    let transformDeclarationWithLayouts (decl: OakDeclaration) : CompilerResult<OakDeclaration> =
        match decl with
        | FunctionDecl(name, parameters, returnType, body) ->
            ExpressionTransformation.transformExpressionWithLayouts body >>= fun transformedBody ->
            Success (FunctionDecl(name, parameters, returnType, transformedBody))
        
        | EntryPoint(expr) ->
            ExpressionTransformation.transformExpressionWithLayouts expr >>= fun transformedExpr ->
            Success (EntryPoint(transformedExpr))
        
        | TypeDecl(name, oakType) ->
            match oakType with
            | UnionType cases ->
                TypeTransformation.transformUnionTypeDeclaration name cases >>= fun transformedType ->
                Success (TypeDecl(name, transformedType))
            | _ ->
                Success decl
                
        | ExternalDecl(_, _, _, _) ->
            Success decl

/// Layout validation
module LayoutValidation =
    
    /// Validates that a layout is zero-allocation
    let validateZeroAllocationLayout (layout: UnionLayout) : CompilerResult<unit> =
        if layout.IsZeroAllocation then
            Success ()
        else
            CompilerFailure [CompilerError("layout validation", "Layout may cause heap allocations", None)]

/// Main fixed layout compilation entry point
let compileFixedLayouts (program: OakProgram) : CompilerResult<OakProgram> =
    if program.Modules.IsEmpty then
        CompilerFailure [CompilerError("fixed layout compilation", "Program must contain at least one module", None)]
    else
        // Transform all modules
        let transformAllModules (modules: OakModule list) : CompilerResult<OakModule list> =
            modules
            |> List.map (fun module' ->
                // Transform all declarations in the module
                let transformModuleDeclarations (declarations: OakDeclaration list) : CompilerResult<OakDeclaration list> =
                    declarations
                    |> List.map DeclarationTransformation.transformDeclarationWithLayouts
                    |> List.fold (fun acc result ->
                        match acc, result with
                        | Success decls, Success decl -> Success (decl :: decls)
                        | CompilerFailure errors, Success _ -> CompilerFailure errors
                        | Success _, CompilerFailure errors -> CompilerFailure errors
                        | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
                    ) (Success [])
                    |> fun result ->
                        match result with
                        | Success decls -> Success (List.rev decls)
                        | CompilerFailure errors -> CompilerFailure errors
                
                transformModuleDeclarations module'.Declarations >>= fun transformedDeclarations ->
                Success { module' with Declarations = transformedDeclarations })
            |> List.fold (fun acc result ->
                match acc, result with
                | Success modules, Success module' -> Success (module' :: modules)
                | CompilerFailure errors, Success _ -> CompilerFailure errors
                | Success _, CompilerFailure errors -> CompilerFailure errors
                | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
            ) (Success [])
            |> fun result ->
                match result with
                | Success modules -> Success (List.rev modules)
                | CompilerFailure errors -> CompilerFailure errors
        
        transformAllModules program.Modules >>= fun transformedModules ->
        Success { program with Modules = transformedModules }

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
        LayoutStrategyAnalysis.analyzeUnionStrategy name cases >>= fun strategy ->
        LayoutComputation.computeUnionLayout name strategy cases >>= fun layout ->
        Success (Some layout)
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