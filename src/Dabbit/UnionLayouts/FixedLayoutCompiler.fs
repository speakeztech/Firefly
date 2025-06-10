module Dabbit.UnionLayouts.FixedLayoutCompiler

open System
open XParsec
open Firefly.Core.XParsec.Foundation
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

/// XParsec parser type for layout analysis
type LayoutParser<'T> = LayoutAnalysisState -> Result<'T * LayoutAnalysisState, string>

/// Parser combinators for layout analysis
module LayoutParsers =
    
    /// Returns a successful result with the given value
    let succeed (value: 'T) : LayoutParser<'T> =
        fun state -> Ok(value, state)
    
    /// Returns a failed result with the given error message
    let fail (message: string) : LayoutParser<'T> =
        fun state -> Error message
    
    /// Monadic bind for parser combinators
    let bind (parser: LayoutParser<'A>) (f: 'A -> LayoutParser<'B>) : LayoutParser<'B> =
        fun state ->
            match parser state with
            | Ok(value, newState) -> f value newState
            | Error msg -> Error msg
    
    /// Infix operator for bind
    let (>>=) = bind
    
    /// Maps a function over the result of a parser
    let map (f: 'A -> 'B) (parser: LayoutParser<'A>) : LayoutParser<'B> =
        parser >>= (fun value -> succeed (f value))
    
    /// Infix operator for map
    let (|>>) parser f = map f parser
    
    /// Adds error context to a parser
    let withErrorContext (context: string) (parser: LayoutParser<'T>) : LayoutParser<'T> =
        fun state ->
            match parser state with
            | Ok result -> Ok result
            | Error msg -> Error (sprintf "%s: %s" context msg)

/// Type size calculation using parser combinators
module TypeSizeCalculation =
    
    /// Calculates the size of an Oak type in bytes
    let rec calculateTypeSize (oakType: OakType) : LayoutParser<int> =
        match oakType with
        | IntType -> LayoutParsers.succeed 4
        | FloatType -> LayoutParsers.succeed 4  
        | BoolType -> LayoutParsers.succeed 1
        | StringType -> LayoutParsers.succeed 8  // Pointer size
        | UnitType -> LayoutParsers.succeed 0
        | ArrayType _ -> LayoutParsers.succeed 8  // Pointer size
        | FunctionType(_, _) -> LayoutParsers.succeed 8  // Function pointer
        | StructType fields ->
            let rec calculateFieldSizes acc fields =
                match fields with
                | [] -> LayoutParsers.succeed acc
                | (_, fieldType) :: rest ->
                    calculateTypeSize fieldType >>= fun fieldSize ->
                    calculateFieldSizes (acc + fieldSize) rest
            calculateFieldSizes 0 fields
        | UnionType cases ->
            // For unanalyzed unions, estimate conservatively
            let rec calculateCaseSizes maxSize cases =
                match cases with
                | [] -> LayoutParsers.succeed maxSize
                | (_, optType) :: rest ->
                    match optType with
                    | Some t -> 
                        calculateTypeSize t >>= fun caseSize ->
                        calculateCaseSizes (max maxSize caseSize) rest
                    | None -> calculateCaseSizes maxSize rest
            calculateCaseSizes 0 cases >>= fun maxCaseSize ->
            LayoutParsers.succeed (1 + maxCaseSize)  // Tag + largest payload
        |> LayoutParsers.withErrorContext "type size calculation"
    
    /// Calculates the alignment requirement for an Oak type
    let rec calculateTypeAlignment (oakType: OakType) : LayoutParser<int> =
        match oakType with
        | IntType -> LayoutParsers.succeed 4
        | FloatType -> LayoutParsers.succeed 4
        | BoolType -> LayoutParsers.succeed 1
        | StringType -> LayoutParsers.succeed 8
        | UnitType -> LayoutParsers.succeed 1
        | ArrayType _ -> LayoutParsers.succeed 8
        | FunctionType(_, _) -> LayoutParsers.succeed 8
        | StructType fields ->
            if fields.IsEmpty then
                LayoutParsers.succeed 1
            else
                let rec calculateFieldAlignments maxAlign fields =
                    match fields with
                    | [] -> LayoutParsers.succeed maxAlign
                    | (_, fieldType) :: rest ->
                        calculateTypeAlignment fieldType >>= fun fieldAlign ->
                        calculateFieldAlignments (max maxAlign fieldAlign) rest
                calculateFieldAlignments 1 fields
        | UnionType _ -> LayoutParsers.succeed 8  // Conservative alignment
        |> LayoutParsers.withErrorContext "type alignment calculation"

/// Layout strategy analysis using parser combinators
module LayoutStrategyAnalysis =
    
    /// State manipulation functions
    let pushScopeWithParams (parameters: Set<string>) : LayoutParser<unit> =
        fun state ->
            Ok((), state)
    
    let popScope : LayoutParser<unit> =
        fun state ->
            Ok((), state)
    
    let bindVariable (varName: string) : LayoutParser<unit> =
        fun state ->
            Ok((), state)
    
    /// Analyzes a general union for layout strategy
    let analyzeGeneralUnion (name: string) (cases: (string * OakType option) list) : LayoutParser<UnionLayoutStrategy> =
        // Check if all cases are nullary (enum-like)
        let allNullary = cases |> List.forall (fun (_, optType) -> optType.IsNone)
        
        if allNullary then
            let enumValues = [0 .. cases.Length - 1]
            let underlyingType = 
                if cases.Length <= 256 then IntType  // Could be optimized to BoolType for 2 cases, etc.
                else IntType
            LayoutParsers.succeed (EnumOptimization(enumValues, underlyingType))
            |> LayoutParsers.withErrorContext "enum union analysis"
        else
            // Calculate payload sizes for tagged union
            let rec calculatePayloadSizes acc cases =
                match cases with
                | [] -> LayoutParsers.succeed acc
                | (_, optType) :: rest ->
                    match optType with
                    | Some t -> 
                        TypeSizeCalculation.calculateTypeSize t >>= fun size ->
                        calculatePayloadSizes (size :: acc) rest
                    | None -> calculatePayloadSizes (0 :: acc) rest
            
            calculatePayloadSizes [] cases >>= fun payloadSizes ->
            
            let maxPayloadSize = List.max (0 :: payloadSizes)
            let tagSize = 
                if cases.Length <= 256 then 1
                elif cases.Length <= 65536 then 2
                else 4
            
            let alignment = max tagSize (if maxPayloadSize > 0 then 8 else tagSize)
            LayoutParsers.succeed (TaggedUnion(tagSize, maxPayloadSize, alignment))
            |> LayoutParsers.withErrorContext "tagged union analysis"
    
    /// Records a layout strategy for a union type
    let recordLayoutStrategy (name: string) (strategy: UnionLayoutStrategy) : LayoutParser<unit> =
        fun state ->
            let newState = { 
                state with 
                    LayoutStrategies = Map.add name strategy state.LayoutStrategies
            }
            Ok((), newState)

/// Layout computation using parser combinators
module LayoutComputation =
    
    /// Computes the memory layout for a union based on its strategy
    let computeUnionLayout (name: string) (strategy: UnionLayoutStrategy) (cases: (string * OakType option) list) : LayoutParser<UnionLayout> =
        match strategy with
        | SingleCase(caseType, isNewtype) ->
            TypeSizeCalculation.calculateTypeSize caseType >>= fun size ->
            TypeSizeCalculation.calculateTypeAlignment caseType >>= fun alignment ->
            LayoutParsers.succeed {
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
            LayoutParsers.succeed {
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
            LayoutParsers.succeed {
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
            LayoutParsers.succeed {
                Strategy = strategy
                TotalSize = totalSize
                Alignment = alignment
                TagOffset = tagOffset
                PayloadOffset = payloadOffset
                CaseMap = caseMap
                IsZeroAllocation = true  // Stack-allocated
            }
        
        | EmptyUnion(errorMsg) ->
            LayoutParsers.fail errorMsg
        |> LayoutParsers.withErrorContext (sprintf "layout computation for union '%s'" name)
    
    /// Records a computed layout
    let recordUnionLayout (name: string) (layout: UnionLayout) : LayoutParser<unit> =
        fun state ->
            let newState = { 
                state with 
                    UnionLayouts = Map.add name layout state.UnionLayouts
            }
            Ok((), newState)

/// Type transformation using parser combinators
module TypeTransformationParsers =
    
    /// Transforms a union type declaration to use fixed layout
    let transformUnionTypeDeclaration (name: string) (cases: (string * OakType option) list) : LayoutParser<OakType> =
        LayoutStrategyAnalysis.analyzeUnionStrategy name cases >>= fun strategy ->
        LayoutStrategyAnalysis.recordLayoutStrategy name strategy >>= fun _ ->
        LayoutComputation.computeUnionLayout name strategy cases >>= fun layout ->
        LayoutComputation.recordUnionLayout name layout >>= fun _ ->
        
        // Transform to appropriate target type based on strategy
        match strategy with
        | SingleCase(caseType, _) ->
            LayoutParsers.succeed caseType
        
        | OptionOptimization(someType, _) ->
            LayoutParsers.succeed someType  // Special handling in codegen
        
        | EnumOptimization(_, underlyingType) ->
            LayoutParsers.succeed underlyingType
        
        | TaggedUnion(tagSize, payloadSize, _) ->
            // Create struct with tag and payload
            let tagType = IntType  // Could be optimized based on tagSize
            let payloadType = ArrayType(IntType)  // Byte array representation
            LayoutParsers.succeed (StructType [("tag", tagType); ("payload", payloadType)])
        
        | EmptyUnion(errorMsg) ->
            LayoutParsers.fail errorMsg
        |> LayoutParsers.withErrorContext (sprintf "union type transformation '%s'" name)
    
    /// Records a type mapping for later reference
    let recordTypeMapping (originalName: string) (transformedType: OakType) : LayoutParser<unit> =
        fun state ->
            let newState = { 
                state with 
                    TypeMappings = Map.add originalName transformedType state.TypeMappings
                    TransformationHistory = (originalName, transformedType.ToString()) :: state.TransformationHistory
            }
            Ok((), newState)

/// Expression transformation using layout information
module ExpressionLayoutTransformation =
    
    /// Transforms expressions to use fixed layouts (placeholder for now)
    let rec transformExpressionWithLayouts (expr: OakExpression) : LayoutParser<OakExpression> =
        match expr with
        | Variable name ->
            LayoutParsers.succeed expr  // Variable references unchanged
        
        | Application(func, args) ->
            transformExpressionWithLayouts func >>= fun transformedFunc ->
            transformArguments args >>= fun transformedArgs ->
            LayoutParsers.succeed (Application(transformedFunc, transformedArgs))
        
        | Let(name, value, body) ->
            transformExpressionWithLayouts value >>= fun transformedValue ->
            transformExpressionWithLayouts body >>= fun transformedBody ->
            LayoutParsers.succeed (Let(name, transformedValue, transformedBody))
        
        | IfThenElse(cond, thenExpr, elseExpr) ->
            transformExpressionWithLayouts cond >>= fun transformedCond ->
            transformExpressionWithLayouts thenExpr >>= fun transformedThen ->
            transformExpressionWithLayouts elseExpr >>= fun transformedElse ->
            LayoutParsers.succeed (IfThenElse(transformedCond, transformedThen, transformedElse))
        
        | Sequential(first, second) ->
            transformExpressionWithLayouts first >>= fun transformedFirst ->
            transformExpressionWithLayouts second >>= fun transformedSecond ->
            LayoutParsers.succeed (Sequential(transformedFirst, transformedSecond))
        
        | FieldAccess(target, fieldName) ->
            transformExpressionWithLayouts target >>= fun transformedTarget ->
            LayoutParsers.succeed (FieldAccess(transformedTarget, fieldName))
        
        | MethodCall(target, methodName, args) ->
            transformExpressionWithLayouts target >>= fun transformedTarget ->
            transformArguments args >>= fun transformedArgs ->
            LayoutParsers.succeed (MethodCall(transformedTarget, methodName, transformedArgs))
        
        | Lambda(parameters, body) ->
            transformExpressionWithLayouts body >>= fun transformedBody ->
            LayoutParsers.succeed (Lambda(parameters, transformedBody))
        
        | Literal _ ->
            LayoutParsers.succeed expr
        
        | IOOperation(ioType, args) ->
            transformArguments args >>= fun transformedArgs ->
            LayoutParsers.succeed (IOOperation(ioType, transformedArgs))
    
    and transformArguments (args: OakExpression list) : LayoutParser<OakExpression list> =
        let rec transformArgs acc remaining =
            match remaining with
            | [] -> LayoutParsers.succeed (List.rev acc)
            | arg :: rest ->
                transformExpressionWithLayouts arg >>= fun transformedArg ->
                transformArgs (transformedArg :: acc) rest
        transformArgs [] args

/// Declaration transformation using parser combinators
module DeclarationLayoutTransformation =
    
    /// Transforms a declaration to use fixed layouts
    let transformDeclarationWithLayouts (decl: OakDeclaration) : LayoutParser<OakDeclaration list> =
        match decl with
        | FunctionDecl(name, parameters, returnType, body) ->
            ExpressionLayoutTransformation.transformExpressionWithLayouts body >>= fun transformedBody ->
            LayoutParsers.succeed [FunctionDecl(name, parameters, returnType, transformedBody)]
        
        | EntryPoint(expr) ->
            ExpressionLayoutTransformation.transformExpressionWithLayouts expr >>= fun transformedExpr ->
            LayoutParsers.succeed [EntryPoint(transformedExpr)]
        
        | TypeDecl(name, oakType) ->
            match oakType with
            | UnionType cases ->
                TypeTransformationParsers.transformUnionTypeDeclaration name cases >>= fun transformedType ->
                TypeTransformationParsers.recordTypeMapping name transformedType >>= fun _ ->
                LayoutParsers.succeed [TypeDecl(name, transformedType)]
            | _ ->
                LayoutParsers.succeed [decl]
        | ExternalDecl(_, _, _, _) ->
            LayoutParsers.succeed [decl]
        |> LayoutParsers.withErrorContext (sprintf "declaration transformation with layouts '%s'" (match decl with FunctionDecl(n,_,_,_) -> n | TypeDecl(n,_) -> n | EntryPoint(_) -> "__entry__" | ExternalDecl(n,_,_,_) -> n))

/// Module transformation using parser combinators
module ModuleTransformationParsers =
    
    /// Builds global scope from function declarations
    let buildGlobalScope (declarations: OakDeclaration list) : Set<string> =
        declarations
        |> List.choose (function
            | FunctionDecl(name, _, _, _) -> Some name
            | _ -> None)
        |> Set.ofList
    
    /// Transforms a complete module
    let transformModule (module': OakModule) : LayoutParser<OakModule> =
        let globalScope = buildGlobalScope module'.Declarations
        
        fun state ->
            let initialState = { 
                state with 
                    UnionLayouts = Map.empty
                    TypeMappings = Map.empty
                    LayoutStrategies = Map.empty
                    TransformationHistory = []
                    ErrorContext = []
            }
            
            // Transform all declarations
            let rec transformAllDeclarations acc remaining state =
                match remaining with
                | [] -> Ok(List.rev acc, state)
                | decl :: rest ->
                    match DeclarationLayoutTransformation.transformDeclarationWithLayouts decl state with
                    | Ok(transformedDecls, newState) ->
                        transformAllDeclarations (transformedDecls @ acc) rest newState
                    | Error msg -> Error msg
            
            match transformAllDeclarations [] module'.Declarations initialState with
            | Ok(transformedDeclarations, finalState) ->
                let transformedModule = { module' with Declarations = transformedDeclarations }
                Ok(transformedModule, finalState)
            | Error msg -> Error msg
        |> LayoutParsers.withErrorContext (sprintf "module transformation '%s'" module'.Name)

/// Closure elimination validation
module LayoutValidation =
    
    /// Validates that a layout is zero-allocation
    let validateZeroAllocationLayout (layout: UnionLayout) : CompilerResult<unit> =
        if layout.IsZeroAllocation then
            Success ()
        else
            CompilerFailure [CompilerError("layout validation", "Layout may cause heap allocations", Some "computed layout")]
    
    /// Validates that all computed layouts are zero-allocation
    let validateAllLayoutsZeroAllocation : LayoutParser<unit> =
        fun state ->
            let validationResults = 
                state.UnionLayouts
                |> Map.toList
                |> List.map (fun (name, layout) ->
                    validateZeroAllocationLayout layout)
            
            let hasErrors = 
                validationResults
                |> List.exists (function CompilerFailure _ -> true | Success _ -> false)
            
            if hasErrors then
                Error "Some layouts failed zero-allocation validation"
            else
                Ok((), state)

/// Main fixed layout compilation entry point - NO FALLBACKS ALLOWED
let compileFixedLayouts (program: OakProgram) : CompilerResult<OakProgram> =
    if program.Modules.IsEmpty then
        CompilerFailure [CompilerError("fixed layout compilation", "Program must contain at least one module", Some "empty program")]
    else
        let initialState = {
            UnionLayouts = Map.empty
            TypeMappings = Map.empty
            LayoutStrategies = Map.empty
            TransformationHistory = []
            ErrorContext = []
        }
        
        // Transform all modules
        let rec transformAllModules acc remaining state =
            match remaining with
            | [] -> Ok(List.rev acc, state)
            | module' :: rest ->
                match ModuleTransformationParsers.transformModule module' state with
                | Ok(transformedModule, newState) ->
                    transformAllModules (transformedModule :: acc) rest newState
                | Error msg -> Error msg
        
        match transformAllModules [] program.Modules initialState with
        | Ok(transformedModules, finalState) ->
            let transformedProgram = { program with Modules = transformedModules }
            
            // Validate all layouts are zero-allocation
            match LayoutValidation.validateAllLayoutsZeroAllocation finalState with
            | Ok((), _) ->
                Success transformedProgram
            | Error msg ->
                CompilerFailure [CompilerError("layout validation", msg, Some "compiled layouts")]
        
        | Error msg ->
            CompilerFailure [CompilerError("module transformation", msg, Some "module transformation")]

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
        match LayoutStrategyAnalysis.analyzeUnionStrategy name cases initialState with
        | Ok(strategy, state1) ->
            match LayoutComputation.computeUnionLayout name strategy cases state1 with
            | Ok(layout, _) -> Success (Some layout)
            | Error msg -> CompilerFailure [CompilerError("layout computation", msg, Some name)]
        | Error msg -> CompilerFailure [CompilerError("strategy analysis", msg, Some name)]
    | _ -> Success None

/// Validates that all union layouts in a program are zero-allocation
let validateZeroAllocationLayouts (program: OakProgram) : CompilerResult<bool> =
    let unionTypes = 
        program.Modules
        |> List.collect (fun m -> m.Declarations)
        |> List.choose (function
            | TypeDecl(name, UnionType cases) -> Some (name, cases)
            | _ -> None)
    
    let rec validateUnions acc remaining =
        match remaining with
        | [] -> Success acc
        | (name, cases) :: rest ->
            match getUnionLayout program name with
            | Success (Some layout) -> 
                if layout.IsZeroAllocation then 
                    validateUnions acc rest
                else 
                    validateUnions false rest
            | Success None -> validateUnions acc rest  // No layout computed means no issues
            | CompilerFailure errors -> CompilerFailure errors
    
    validateUnions true unionTypes