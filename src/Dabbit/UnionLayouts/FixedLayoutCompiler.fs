module Dabbit.UnionLayouts.FixedLayoutCompiler

open System
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

/// Captured variable information for closure analysis
type CapturedVariable = {
    Name: string
    Type: OakType
    OriginalName: string
    CaptureContext: string
    IsParameter: bool
}

/// Lifted closure representation
type LiftedClosure = {
    Name: string
    OriginalLambda: OakExpression
    Parameters: (string * OakType) list
    CapturedVars: CapturedVariable list
    Body: OakExpression
    ReturnType: OakType
    CallSites: string list
}

/// Result type for compiler operations
type CompilerResult<'T> =
    | Success of 'T
    | CompilerFailure of CompilerError list

and CompilerError =
    | ParseError of position: ParsePosition * message: string * context: string list
    | TransformError of phase: string * input: string * expected: string * message: string
    | CompilerError of phase: string * message: string * details: string option

and ParsePosition = {
    Line: int
    Column: int
    File: string
    Offset: int
}

/// Bind operator for CompilerResult
let (>>=) result f =
    match result with
    | Success value -> f value
    | CompilerFailure errors -> CompilerFailure errors

/// Map operator for CompilerResult  
let (|>>) result f =
    match result with
    | Success value -> Success (f value)
    | CompilerFailure errors -> CompilerFailure errors

/// Type size calculation using Foundation parsers
module TypeSizeCalculation =
    
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
            let fieldSizes = fields |> List.map (snd >> calculateTypeSize)
            List.fold (fun acc sizeParser ->
                acc >>= fun accSize ->
                sizeParser >>= fun fieldSize ->
                succeed (accSize + fieldSize)
            ) (succeed 0) fieldSizes
        | UnionType cases ->
            // For unanalyzed unions, estimate conservatively
            let caseSizes = cases |> List.map (fun (_, optType) ->
                match optType with
                | Some t -> calculateTypeSize t
                | None -> succeed 0)
            List.fold (fun acc sizeParser ->
                acc >>= fun accSize ->
                sizeParser >>= fun caseSize ->
                succeed (max accSize caseSize)
            ) (succeed 0) caseSizes
            >>= fun maxCaseSize ->
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
                let alignments = fields |> List.map (snd >> calculateTypeAlignment)
                List.fold (fun acc alignParser ->
                    acc >>= fun accAlign ->
                    alignParser >>= fun fieldAlign ->
                    succeed (max accAlign fieldAlign)
                ) (succeed 1) alignments
        | UnionType _ -> succeed 8  // Conservative alignment

/// Layout strategy analysis using Foundation combinators
module LayoutStrategyAnalysis =
    
    /// Analyzes a general union for layout strategy
    let analyzeGeneralUnion (name: string) (cases: (string * OakType option) list) : Parser<UnionLayoutStrategy> =
        // Check if all cases are nullary (enum-like)
        let allNullary = cases |> List.forall (fun (_, optType) -> optType.IsNone)
        
        if allNullary then
            let enumValues = [0 .. cases.Length - 1]
            let underlyingType = 
                if cases.Length <= 256 then IntType  // Could be optimized to BoolType for 2 cases, etc.
                else IntType
            succeed (EnumOptimization(enumValues, underlyingType))
        else
            // Calculate payload sizes for tagged union
            let payloadSizes = cases |> List.map (fun (_, optType) ->
                match optType with
                | Some t -> calculateTypeSize t
                | None -> succeed 0)
            
            List.fold (fun acc sizeParser ->
                acc >>= fun accSizes ->
                sizeParser >>= fun size ->
                succeed (size :: accSizes)
            ) (succeed []) payloadSizes
            >>= fun sizes ->
            
            let maxPayloadSize = List.max (0 :: sizes)
            let tagSize = 
                if cases.Length <= 256 then 1
                elif cases.Length <= 65536 then 2
                else 4
            
            let alignment = max tagSize (if maxPayloadSize > 0 then 8 else tagSize)
            succeed (TaggedUnion(tagSize, maxPayloadSize, alignment))

    /// Analyzes union cases to determine optimal layout strategy
    let analyzeUnionStrategy (name: string) (cases: (string * OakType option) list) : Parser<UnionLayoutStrategy> =
        if cases.IsEmpty then
            succeed (EmptyUnion("Empty unions are not supported"))
        
        elif cases.Length = 1 then
            match cases.[0] with
            | (_, Some caseType) ->
                succeed (SingleCase(caseType, true))
            | (_, None) ->
                succeed (EnumOptimization([0], IntType))
        
        elif cases.Length = 2 then
            match cases with
            | [("None", None); ("Some", Some someType)] | [("Some", Some someType); ("None", None)] ->
                succeed (OptionOptimization(someType, true))
            | _ ->
                analyzeGeneralUnion name cases
        
        else
            analyzeGeneralUnion name cases
    
    /// Records a layout strategy for a union type
    let recordLayoutStrategy (name: string) (strategy: UnionLayoutStrategy) : Parser<unit> =
        updateState (fun state ->
            let stateMap = Map.add "layoutStrategies" (Map.add name strategy Map.empty) state.Metadata
            { state with Metadata = stateMap })

/// Layout computation using Foundation combinators
module LayoutComputation =
    
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
            fail (sprintf "Layout computation failed: %s" errorMsg)
    
    /// Records a computed layout
    let recordUnionLayout (name: string) (layout: UnionLayout) : Parser<unit> =
        updateState (fun state ->
            let stateMap = Map.add "unionLayouts" (Map.add name layout Map.empty) state.Metadata
            { state with Metadata = stateMap })

/// Type transformation using Foundation combinators
module TypeTransformationParsers =
    
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
            fail (sprintf "Union transformation failed: %s" errorMsg)
    
    /// Records a type mapping for later reference
    let recordTypeMapping (originalName: string) (transformedType: OakType) : Parser<unit> =
        updateState (fun state ->
            let stateMap = Map.add "typeMappings" (Map.add originalName transformedType Map.empty) state.Metadata
            { state with Metadata = stateMap })

/// Expression transformation using layout information
module ExpressionLayoutTransformation =
    
    /// Transforms expressions to use fixed layouts
    let rec transformExpressionWithLayouts (expr: OakExpression) : Parser<OakExpression> =
        match expr with
        | Variable name ->
            succeed expr  // Variable references unchanged
        
        | Application(func, args) ->
            transformExpressionWithLayouts func >>= fun transformedFunc ->
            List.fold (fun acc argParser ->
                acc >>= fun accArgs ->
                argParser >>= fun transformedArg ->
                succeed (transformedArg :: accArgs)
            ) (succeed []) (List.map transformExpressionWithLayouts args)
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
            List.fold (fun acc argParser ->
                acc >>= fun accArgs ->
                argParser >>= fun transformedArg ->
                succeed (transformedArg :: accArgs)
            ) (succeed []) (List.map transformExpressionWithLayouts args)
            >>= fun transformedArgs ->
            succeed (MethodCall(transformedTarget, methodName, List.rev transformedArgs))
        
        | Lambda(parameters, body) ->
            transformExpressionWithLayouts body >>= fun transformedBody ->
            succeed (Lambda(parameters, transformedBody))
        
        | Literal _ ->
            succeed expr
        
        | IOOperation(ioType, args) ->
            List.fold (fun acc argParser ->
                acc >>= fun accArgs ->
                argParser >>= fun transformedArg ->
                succeed (transformedArg :: accArgs)
            ) (succeed []) (List.map transformExpressionWithLayouts args)
            >>= fun transformedArgs ->
            succeed (IOOperation(ioType, List.rev transformedArgs))

/// Declaration transformation using Foundation combinators
module DeclarationLayoutTransformation =
    
    /// Transforms a declaration to use fixed layouts
    let transformDeclarationWithLayouts (decl: OakDeclaration) : Parser<OakDeclaration list> =
        match decl with
        | FunctionDecl(name, parameters, returnType, body) ->
            transformExpressionWithLayouts body >>= fun transformedBody ->
            succeed [FunctionDecl(name, parameters, returnType, transformedBody)]
        
        | EntryPoint(expr) ->
            transformExpressionWithLayouts expr >>= fun transformedExpr ->
            succeed [EntryPoint(transformedExpr)]
        
        | TypeDecl(name, oakType) ->
            match oakType with
            | UnionType cases ->
                transformUnionTypeDeclaration name cases >>= fun transformedType ->
                recordTypeMapping name transformedType >>= fun _ ->
                succeed [TypeDecl(name, transformedType)]
            | _ ->
                succeed [decl]
        
        | ExternalDecl(_, _, _, _) ->
            succeed [decl]

/// Module transformation using Foundation combinators
module ModuleTransformationParsers =
    
    /// Builds global scope from function declarations
    let buildGlobalScope (declarations: OakDeclaration list) : Set<string> =
        declarations
        |> List.choose (function
            | FunctionDecl(name, _, _, _) -> Some name
            | _ -> None)
        |> Set.ofList
    
    /// Transforms a complete module
    let transformModule (module': OakModule) : Parser<OakModule> =
        let globalScope = buildGlobalScope module'.Declarations
        
        // Transform all declarations
        List.fold (fun acc declParser ->
            acc >>= fun accDecls ->
            declParser >>= fun newDecls ->
            succeed (accDecls @ newDecls)
        ) (succeed []) (List.map transformDeclarationWithLayouts module'.Declarations)
        >>= fun transformedDeclarations ->
        succeed { module' with Declarations = transformedDeclarations }

/// Closure elimination validation
module ClosureValidation =
    
    /// Validates that no closures remain in an expression
    let rec validateNoClosures (expr: OakExpression) : CompilerResult<unit> =
        match expr with
        | Lambda(_, _) ->
            CompilerFailure [TransformError("closure validation", "lambda expression", "eliminated closure", "Lambda expression found after closure elimination")]
        
        | Application(func, args) ->
            validateNoClosures func >>= fun _ ->
            List.fold (fun acc result ->
                match acc, result with
                | Success (), Success () -> Success ()
                | CompilerFailure errors, Success () -> CompilerFailure errors
                | Success (), CompilerFailure errors -> CompilerFailure errors
                | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
            ) (Success ()) (List.map validateNoClosures args)
        
        | Let(_, value, body) ->
            validateNoClosures value >>= fun _ ->
            validateNoClosures body
        
        | IfThenElse(cond, thenExpr, elseExpr) ->
            validateNoClosures cond >>= fun _ ->
            validateNoClosures thenExpr >>= fun _ ->
            validateNoClosures elseExpr
        
        | Sequential(first, second) ->
            validateNoClosures first >>= fun _ ->
            validateNoClosures second
        
        | FieldAccess(target, _) ->
            validateNoClosures target
        
        | MethodCall(target, _, args) ->
            validateNoClosures target >>= fun _ ->
            List.fold (fun acc result ->
                match acc, result with
                | Success (), Success () -> Success ()
                | CompilerFailure errors, Success () -> CompilerFailure errors
                | Success (), CompilerFailure errors -> CompilerFailure errors
                | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
            ) (Success ()) (List.map validateNoClosures args)
        
        | Variable _ | Literal _ ->
            Success ()
        
        | IOOperation(_, args) ->
            List.fold (fun acc result ->
                match acc, result with
                | Success (), Success () -> Success ()
                | CompilerFailure errors, Success () -> CompilerFailure errors
                | Success (), CompilerFailure errors -> CompilerFailure errors
                | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
            ) (Success ()) (List.map validateNoClosures args)
    
    /// Validates that a declaration contains no closures
    let validateDeclarationNoClosures (decl: OakDeclaration) : CompilerResult<unit> =
        match decl with
        | FunctionDecl(_, _, _, body) -> validateNoClosures body
        | EntryPoint(expr) -> validateNoClosures expr
        | TypeDecl(_, _) -> Success ()
        | ExternalDecl(_, _, _, _) -> Success ()
    
    /// Validates that a program contains no closures
    let validateProgramNoClosures (program: OakProgram) : CompilerResult<unit> =
        program.Modules
        |> List.collect (fun m -> m.Declarations)
        |> List.map validateDeclarationNoClosures
        |> List.fold (fun acc result ->
            match acc, result with
            | Success (), Success () -> Success ()
            | CompilerFailure errors, Success () -> CompilerFailure errors
            | Success (), CompilerFailure errors -> CompilerFailure errors
            | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
        ) (Success ())

/// Main fixed layout compilation entry point
let compileFixedLayouts (program: OakProgram) : CompilerResult<OakProgram> =
    if program.Modules.IsEmpty then
        CompilerFailure [TransformError("fixed layout compilation", "empty program", "program with fixed layouts", "Program must contain at least one module")]
    else
        let initialState = createInitialState ""
        
        // Transform all modules
        let transformAllModules (modules: OakModule list) : CompilerResult<OakModule list> =
            let rec transformModulesRec acc remaining =
                match remaining with
                | [] -> Success (List.rev acc)
                | module' :: rest ->
                    match runParser (transformModule module') "" with
                    | Ok transformedModule -> transformModulesRec (transformedModule :: acc) rest
                    | Error error -> CompilerFailure [TransformError("module transformation", module'.Name, "transformed module", error)]
            
            transformModulesRec [] modules
        
        transformAllModules program.Modules >>= fun transformedModules ->
        let transformedProgram = { program with Modules = transformedModules }
        
        // Validate that no closures remain
        ClosureValidation.validateProgramNoClosures transformedProgram >>= fun _ ->
        
        Success transformedProgram

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
        match runParser (analyzeUnionStrategy name cases >>= fun strategy ->
                        computeUnionLayout name strategy cases) "" with
        | Ok layout -> Success (Some layout)
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
    
    let validateUnion (name: string, cases: (string * OakType option) list) : CompilerResult<bool> =
        getUnionLayout program name >>= function
        | Some layout -> 
            if layout.IsZeroAllocation then Success true
            else Success false
        | None -> Success true  // No layout computed means no issues
    
    List.fold (fun acc result ->
        match acc, result with
        | Success accBool, Success resultBool -> Success (accBool && resultBool)
        | CompilerFailure errors, Success _ -> CompilerFailure errors
        | Success _, CompilerFailure errors -> CompilerFailure errors
        | CompilerFailure errors1, CompilerFailure errors2 -> CompilerFailure (errors1 @ errors2)
    ) (Success true) (List.map validateUnion unionTypes)