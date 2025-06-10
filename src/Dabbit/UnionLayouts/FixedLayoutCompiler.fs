module Dabbit.UnionLayouts.FixedLayoutCompiler

open System
open Firefly.Core.XParsec.Foundation
open Dabbit.Parsing.OakAst

// ======================================
// Compiler Result Types
// ======================================

/// Compiler result type for transformations
type CompilerResult<'T> =
    | Success of 'T
    | CompilerFailure of CompilerError list

and CompilerError =
    | TransformError of phase: string * input: string * expected: string * message: string
    | ParseError of position: ParsePosition * message: string * context: string list
    | CompilerError of phase: string * message: string * details: string option

and ParsePosition = {
    Line: int
    Column: int
    File: string
    Offset: int
}

/// Layout analysis metadata stored in parser state
type LayoutAnalysisMetadata = {
    UnionLayouts: Map<string, UnionLayout>
    TypeMappings: Map<string, OakType>
    LayoutStrategies: Map<string, UnionLayoutStrategy>
    TransformationHistory: (string * string) list
}

/// Union layout strategy with complete optimization information
and UnionLayoutStrategy =
    | TaggedUnion of tagSize: int * maxPayloadSize: int * alignment: int
    | EnumOptimization of enumValues: int list * underlyingType: OakType
    | OptionOptimization of someType: OakType * nullRepresentation: bool
    | SingleCase of caseType: OakType * isNewtype: bool
    | EmptyUnion of errorMessage: string

/// Union layout with complete memory information
and UnionLayout = {
    Strategy: UnionLayoutStrategy
    TotalSize: int
    Alignment: int
    TagOffset: int
    PayloadOffset: int
    CaseMap: Map<string, int>
    IsZeroAllocation: bool
}

/// Compiler result type for transformations
type CompilerResult<'T> =
    | Success of 'T
    | CompilerFailure of CompilerError list

and CompilerError =
    | TransformError of phase: string * input: string * expected: string * message: string
    | ParseError of position: ParsePosition * message: string * context: string list
    | CompilerError of phase: string * message: string * details: string option

and ParsePosition = {
    Line: int
    Column: int
    File: string
    Offset: int
}

// ======================================
// Utility Functions for State Management
// ======================================

/// Gets layout analysis metadata from parser state
let getLayoutMetadata : Parser<LayoutAnalysisMetadata> =
    getMetadata "LayoutAnalysis" 
    |>> function 
        | Some (:? LayoutAnalysisMetadata as meta) -> meta
        | _ -> {
            UnionLayouts = Map.empty
            TypeMappings = Map.empty
            LayoutStrategies = Map.empty
            TransformationHistory = []
        }

/// Updates layout analysis metadata in parser state
let setLayoutMetadata (meta: LayoutAnalysisMetadata) : Parser<unit> =
    addMetadata "LayoutAnalysis" meta

/// Updates union layouts in metadata
let updateUnionLayouts (f: Map<string, UnionLayout> -> Map<string, UnionLayout>) : Parser<unit> =
    getLayoutMetadata >>= fun meta ->
    setLayoutMetadata { meta with UnionLayouts = f meta.UnionLayouts }

/// Updates layout strategies in metadata
let updateLayoutStrategies (f: Map<string, UnionLayoutStrategy> -> Map<string, UnionLayoutStrategy>) : Parser<unit> =
    getLayoutMetadata >>= fun meta ->
    setLayoutMetadata { meta with LayoutStrategies = f meta.LayoutStrategies }

/// Records a transformation in history
let recordTransformation (name: string) (result: string) : Parser<unit> =
    getLayoutMetadata >>= fun meta ->
    let newHistory = (name, result) :: meta.TransformationHistory
    setLayoutMetadata { meta with TransformationHistory = newHistory }

// ======================================
// Type Size Calculation
// ======================================

/// Calculates the size of an Oak type in bytes
let rec calculateTypeSize (oakType: OakType) : Parser<int> =
    match oakType with
    | IntType -> succeed 4
    | FloatType -> succeed 4  
    | BoolType -> succeed 1
    | StringType -> succeed 8  // Pointer size
    | UnitType -> succeed 0
    | ArrayType _ -> succeed 8  // Pointer size
    | FunctionType(_, _) -> succeed 8  // Function pointer
    | StructType fields ->
        let calculateFieldSizes fields =
            let rec loop acc = function
                | [] -> succeed acc
                | (_, fieldType) :: rest ->
                    calculateTypeSize fieldType >>= fun size ->
                    loop (acc + size) rest
            loop 0 fields
        calculateFieldSizes fields
    | UnionType cases ->
        // For unanalyzed unions, estimate conservatively
        let calculateCaseSizes cases =
            let rec loop maxSize = function
                | [] -> succeed maxSize
                | (_, optType) :: rest ->
                    match optType with
                    | Some t -> 
                        calculateTypeSize t >>= fun size ->
                        loop (max maxSize size) rest
                    | None -> loop maxSize rest
            loop 0 cases
        calculateCaseSizes cases >>= fun maxCaseSize ->
        succeed (1 + maxCaseSize)  // Tag + largest payload

/// Calculates the alignment requirement for an Oak type
let rec calculateTypeAlignment (oakType: OakType) : Parser<int> =
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
            let calculateFieldAlignments fields =
                let rec loop maxAlign = function
                    | [] -> succeed maxAlign
                    | (_, fieldType) :: rest ->
                        calculateTypeAlignment fieldType >>= fun align ->
                        loop (max maxAlign align) rest
                loop 1 fields
            calculateFieldAlignments fields
    | UnionType _ -> succeed 8  // Conservative alignment

// ======================================
// Layout Strategy Analysis
// ======================================

/// Analyzes union cases to determine optimal layout strategy
let analyzeUnionStrategy (name: string) (cases: (string * OakType option) list) : Parser<UnionLayoutStrategy> =
    if cases.IsEmpty then
        succeed (EmptyUnion("Empty unions are not supported"))
    elif cases.Length = 1 then
        match cases.[0] with
        | (_, Some caseType) -> succeed (SingleCase(caseType, true))
        | (_, None) -> succeed (EnumOptimization([0], IntType))
    elif cases.Length = 2 then
        match cases with
        | [("None", None); ("Some", Some someType)] | [("Some", Some someType); ("None", None)] ->
            succeed (OptionOptimization(someType, true))
        | _ -> analyzeGeneralUnion name cases
    else
        analyzeGeneralUnion name cases

/// Analyzes a general union for layout strategy
and analyzeGeneralUnion (name: string) (cases: (string * OakType option) list) : Parser<UnionLayoutStrategy> =
    let allNullary = cases |> List.forall (fun (_, optType) -> optType.IsNone)
    
    if allNullary then
        let enumValues = [0 .. cases.Length - 1]
        let underlyingType = 
            if cases.Length <= 256 then IntType
            else IntType
        succeed (EnumOptimization(enumValues, underlyingType))
    else
        // Calculate payload sizes for tagged union
        let calculatePayloadSizes cases =
            let rec loop maxSize = function
                | [] -> succeed maxSize
                | (_, optType) :: rest ->
                    match optType with
                    | Some t -> 
                        calculateTypeSize t >>= fun size ->
                        loop (max maxSize size) rest
                    | None -> loop maxSize rest
            loop 0 cases
        
        calculatePayloadSizes cases >>= fun maxPayloadSize ->
        let tagSize = 
            if cases.Length <= 256 then 1
            elif cases.Length <= 65536 then 2
            else 4
        let alignment = max tagSize (if maxPayloadSize > 0 then 8 else tagSize)
        succeed (TaggedUnion(tagSize, maxPayloadSize, alignment))

/// Records a layout strategy for a union type
let recordLayoutStrategy (name: string) (strategy: UnionLayoutStrategy) : Parser<unit> =
    updateLayoutStrategies (Map.add name strategy)

// ======================================
// Layout Computation
// ======================================

/// Computes the memory layout for a union based on its strategy
let computeUnionLayout (name: string) (strategy: UnionLayoutStrategy) (cases: (string * OakType option) list) : Parser<UnionLayout> =
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
        fail errorMsg

/// Records a computed layout
let recordUnionLayout (name: string) (layout: UnionLayout) : Parser<unit> =
    updateUnionLayouts (Map.add name layout)

// ======================================
// Type Transformation
// ======================================

/// Transforms a union type declaration to use fixed layout
let transformUnionTypeDeclaration (name: string) (cases: (string * OakType option) list) : Parser<OakType> =
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
        fail errorMsg

/// Records a type mapping for later reference
let recordTypeMapping (originalName: string) (transformedType: OakType) : Parser<unit> =
    recordTransformation originalName (transformedType.ToString()) >>= fun _ ->
    getLayoutMetadata >>= fun meta ->
    setLayoutMetadata { meta with TypeMappings = Map.add originalName transformedType meta.TypeMappings }

// ======================================
// Expression Transformation
// ======================================

/// Transforms expressions to use fixed layouts
let rec transformExpressionWithLayouts (expr: OakExpression) : Parser<OakExpression> =
    match expr with
    | Variable name ->
        succeed expr  // Variable references unchanged
    
    | Application(func, args) ->
        transformExpressionWithLayouts func >>= fun transformedFunc ->
        let transformArgs args =
            let rec loop acc = function
                | [] -> succeed (List.rev acc)
                | arg :: rest ->
                    transformExpressionWithLayouts arg >>= fun transformedArg ->
                    loop (transformedArg :: acc) rest
            loop [] args
        transformArgs args >>= fun transformedArgs ->
        succeed (Application(transformedFunc, transformedArgs))
    
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
        let transformArgs args =
            let rec loop acc = function
                | [] -> succeed (List.rev acc)
                | arg :: rest ->
                    transformExpressionWithLayouts arg >>= fun transformedArg ->
                    loop (transformedArg :: acc) rest
            loop [] args
        transformArgs args >>= fun transformedArgs ->
        succeed (MethodCall(transformedTarget, methodName, transformedArgs))
    
    | Lambda(parameters, body) ->
        transformExpressionWithLayouts body >>= fun transformedBody ->
        succeed (Lambda(parameters, transformedBody))
    
    | Literal _ ->
        succeed expr
    
    | IOOperation(ioType, args) ->
        let transformArgs args =
            let rec loop acc = function
                | [] -> succeed (List.rev acc)
                | arg :: rest ->
                    transformExpressionWithLayouts arg >>= fun transformedArg ->
                    loop (transformedArg :: acc) rest
            loop [] args
        transformArgs args >>= fun transformedArgs ->
        succeed (IOOperation(ioType, transformedArgs))

// ======================================
// Declaration Transformation
// ======================================

/// Transforms a declaration to use fixed layouts
let transformDeclarationWithLayouts (decl: OakDeclaration) : Parser<OakDeclaration> =
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

// ======================================
// Layout Validation
// ======================================

/// Validates that a layout is zero-allocation
let validateZeroAllocationLayout (layout: UnionLayout) : CompilerResult<unit> =
    if layout.IsZeroAllocation then
        Success ()
    else
        CompilerFailure [TransformError("layout validation", "computed layout", "zero-allocation layout", "Layout may cause heap allocations")]

/// Validates that all computed layouts are zero-allocation
let validateAllLayoutsZeroAllocation : Parser<unit> =
    getLayoutMetadata >>= fun meta ->
    let layouts = meta.UnionLayouts |> Map.toList |> List.map snd
    let allValid = layouts |> List.forall (fun layout -> layout.IsZeroAllocation)
    if allValid then
        succeed ()
    else
        fail "Some layouts may cause heap allocations"

// ======================================
// Main Compilation Entry Point
// ======================================

/// Compiles fixed layouts for a program - NO FALLBACKS ALLOWED
let compileFixedLayouts (program: OakProgram) : CompilerResult<OakProgram> =
    if program.Modules.IsEmpty then
        CompilerFailure [TransformError("fixed layout compilation", "empty program", "program with fixed layouts", "Program must contain at least one module")]
    else
        let transformModules (modules: OakModule list) =
            let rec loop acc = function
                | [] -> succeed (List.rev acc)
                | module' :: rest ->
                    let transformDeclarations (declarations: OakDeclaration list) =
                        let rec declLoop acc = function
                            | [] -> succeed (List.rev acc)
                            | decl :: rest ->
                                transformDeclarationWithLayouts decl >>= fun transformedDecl ->
                                declLoop (transformedDecl :: acc) rest
                        declLoop [] declarations
                    
                    transformDeclarations module'.Declarations >>= fun transformedDeclarations ->
                    let transformedModule = { module' with Declarations = transformedDeclarations }
                    loop (transformedModule :: acc) rest
            loop [] modules
        
        match runParser (transformModules program.Modules) "" with
        | Ok transformedModules ->
            let transformedProgram = { program with Modules = transformedModules }
            
            // Validate all layouts are zero-allocation
            match runParser validateAllLayoutsZeroAllocation "" with
            | Ok () -> Success transformedProgram
            | Error error -> CompilerFailure [TransformError("layout validation", "compiled layouts", "zero-allocation layouts", error)]
        
        | Error error ->
            CompilerFailure [TransformError("module transformation", "Oak AST", "transformed Oak AST", error)]

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
        let analyzeLayout = 
            analyzeUnionStrategy name cases >>= fun strategy ->
            computeUnionLayout name strategy cases >>= fun layout ->
            succeed (Some layout)
        
        match runParser analyzeLayout "" with
        | Ok result -> Success result
        | Error error -> CompilerFailure [TransformError("layout computation", name, "union layout", error)]
    
    | _ -> Success None

/// Validates that all union layouts in a program are zero-allocation
let validateZeroAllocationLayouts (program: OakProgram) : CompilerResult<bool> =
    let unionTypes = 
        program.Modules
        |> List.collect (fun m -> m.Declarations)
        |> List.choose (function
            | TypeDecl(name, UnionType cases) -> Some (name, cases)
            | _ -> None)
    
    let validateUnions = 
        unionTypes 
        |> List.map (fun (name, cases) -> getUnionLayout program name)
        |> List.fold (fun acc result ->
            match acc, result with
            | Success accBool, Success (Some layout) -> Success (accBool && layout.IsZeroAllocation)
            | Success accBool, Success None -> Success accBool
            | CompilerFailure errors, Success _ -> CompilerFailure errors
            | Success _, CompilerFailure errors -> CompilerFailure errors
            | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
        ) (Success true)
    
    validateUnions