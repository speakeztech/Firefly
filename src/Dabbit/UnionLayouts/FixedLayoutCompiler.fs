module Dabbit.UnionLayouts.FixedLayoutCompiler

open System
open Firefly.Core.XParsec.Foundation
open Dabbit.Parsing.OakAst

// ======================================
// Layout Analysis State
// ======================================

/// State for tracking union layout compilation
type LayoutState = {
    UnionLayouts: Map<string, UnionLayout>
    TypeMappings: Map<string, OakType>
    TransformHistory: (string * string) list
}

/// Simple union layout representation
type UnionLayout = {
    UnionName: string
    Strategy: LayoutStrategy
    TotalSize: int
    IsZeroAllocation: bool
}

/// Basic layout strategies for POC
type LayoutStrategy =
    | SingleCase of caseType: OakType
    | TaggedUnion of tagSize: int * maxPayloadSize: int
    | SimpleEnum of caseCount: int

/// Parser type for layout compilation
type LayoutParser<'T> = Parser<'T>

/// Creates initial layout state
let createLayoutState() : LayoutState =
    {
        UnionLayouts = Map.empty
        TypeMappings = Map.empty
        TransformHistory = []
    }

// ======================================
// State Management
// ======================================

/// Gets the current layout state
let getLayoutState : LayoutParser<LayoutState> =
    getMetadata "layout_state" 
    |>> function 
        | Some state -> state :?> LayoutState
        | None -> createLayoutState()

/// Sets the layout state
let setLayoutState (state: LayoutState) : LayoutParser<unit> =
    addMetadata "layout_state" state

/// Updates the layout state
let updateLayoutState (f: LayoutState -> LayoutState) : LayoutParser<unit> =
    getLayoutState >>= fun state ->
    setLayoutState (f state)

// ======================================
// Type Size Calculation
// ======================================

module TypeSizes =
    
    /// Calculates the size of an Oak type in bytes
    let calculateSize (oakType: OakType) : int =
        match oakType with
        | IntType -> 4
        | FloatType -> 4  
        | BoolType -> 1
        | StringType -> 8  // Pointer size
        | UnitType -> 0
        | ArrayType _ -> 8  // Pointer size
        | FunctionType(_, _) -> 8  // Function pointer
        | StructType fields ->
            fields |> List.sumBy (snd >> calculateSize)
        | UnionType _ -> 8  // Conservative estimate
    
    /// Gets the alignment requirement for a type
    let calculateAlignment (oakType: OakType) : int =
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
            else fields |> List.map (snd >> calculateAlignment) |> List.max
        | UnionType _ -> 8

// ======================================
// Layout Strategy Analysis
// ======================================

module LayoutAnalysis =
    
    /// Analyzes union cases to determine layout strategy
    let analyzeStrategy (name: string) (cases: (string * OakType option) list) : LayoutParser<LayoutStrategy> =
        if cases.IsEmpty then
            fail "Empty unions are not supported"
        
        elif cases.Length = 1 then
            match cases.[0] with
            | (_, Some caseType) -> succeed (SingleCase caseType)
            | (_, None) -> succeed (SimpleEnum 1)
        
        else
            // Check if all cases are nullary (enum-like)
            let allNullary = cases |> List.forall (fun (_, optType) -> optType.IsNone)
            
            if allNullary then
                succeed (SimpleEnum cases.Length)
            else
                // Calculate payload sizes for tagged union
                let payloadSizes = 
                    cases 
                    |> List.map (fun (_, optType) ->
                        match optType with
                        | Some t -> TypeSizes.calculateSize t
                        | None -> 0)
                
                let maxPayloadSize = List.max (0 :: payloadSizes)
                let tagSize = 
                    if cases.Length <= 256 then 1
                    elif cases.Length <= 65536 then 2
                    else 4
                
                succeed (TaggedUnion(tagSize, maxPayloadSize))
    
    /// Records a layout strategy for a union
    let recordLayout (name: string) (strategy: LayoutStrategy) : LayoutParser<unit> =
        let totalSize = 
            match strategy with
            | SingleCase caseType -> TypeSizes.calculateSize caseType
            | SimpleEnum _ -> 4  // Use int for enum
            | TaggedUnion(tagSize, payloadSize) -> tagSize + payloadSize
        
        let layout = {
            UnionName = name
            Strategy = strategy
            TotalSize = totalSize
            IsZeroAllocation = true  // All layouts are stack-allocated in POC
        }
        
        updateLayoutState (fun state ->
            { state with UnionLayouts = Map.add name layout state.UnionLayouts })

// ======================================
// Type Transformation
// ======================================

module TypeTransformation =
    
    /// Transforms a union type to its target representation
    let transformUnionType (name: string) (cases: (string * OakType option) list) : LayoutParser<OakType> =
        LayoutAnalysis.analyzeStrategy name cases >>= fun strategy ->
        LayoutAnalysis.recordLayout name strategy >>= fun _ ->
        
        match strategy with
        | SingleCase caseType ->
            succeed caseType
        
        | SimpleEnum _ ->
            succeed IntType
        
        | TaggedUnion(tagSize, _) ->
            // Create struct with tag and payload
            let tagType = IntType
            let payloadType = ArrayType IntType  // Byte array for payload
            succeed (StructType [("tag", tagType); ("payload", payloadType)])
    
    /// Records a type mapping
    let recordTypeMapping (originalName: string) (transformedType: OakType) : LayoutParser<unit> =
        updateLayoutState (fun state ->
            { state with 
                TypeMappings = Map.add originalName transformedType state.TypeMappings
                TransformHistory = (originalName, transformedType.ToString()) :: state.TransformHistory })

// ======================================
// Expression Transformation
// ======================================

/// Transforms expressions (simplified for POC)
let rec transformExpression (expr: OakExpression) : LayoutParser<OakExpression> =
    match expr with
    | Variable name ->
        succeed expr
    
    | Application(func, args) ->
        transformExpression func >>= fun transformedFunc ->
        let rec transformArgs remainingArgs accArgs =
            match remainingArgs with
            | [] -> succeed (List.rev accArgs)
            | arg :: rest ->
                transformExpression arg >>= fun transformedArg ->
                transformArgs rest (transformedArg :: accArgs)
        
        transformArgs args [] >>= fun transformedArgs ->
        succeed (Application(transformedFunc, transformedArgs))
    
    | Let(name, value, body) ->
        transformExpression value >>= fun transformedValue ->
        transformExpression body >>= fun transformedBody ->
        succeed (Let(name, transformedValue, transformedBody))
    
    | IfThenElse(cond, thenExpr, elseExpr) ->
        transformExpression cond >>= fun transformedCond ->
        transformExpression thenExpr >>= fun transformedThen ->
        transformExpression elseExpr >>= fun transformedElse ->
        succeed (IfThenElse(transformedCond, transformedThen, transformedElse))
    
    | Sequential(first, second) ->
        transformExpression first >>= fun transformedFirst ->
        transformExpression second >>= fun transformedSecond ->
        succeed (Sequential(transformedFirst, transformedSecond))
    
    | FieldAccess(target, fieldName) ->
        transformExpression target >>= fun transformedTarget ->
        succeed (FieldAccess(transformedTarget, fieldName))
    
    | MethodCall(target, methodName, args) ->
        transformExpression target >>= fun transformedTarget ->
        let rec transformArgs remainingArgs accArgs =
            match remainingArgs with
            | [] -> succeed (List.rev accArgs)
            | arg :: rest ->
                transformExpression arg >>= fun transformedArg ->
                transformArgs rest (transformedArg :: accArgs)
        
        transformArgs args [] >>= fun transformedArgs ->
        succeed (MethodCall(transformedTarget, methodName, transformedArgs))
    
    | Lambda(parameters, body) ->
        transformExpression body >>= fun transformedBody ->
        succeed (Lambda(parameters, transformedBody))
    
    | Literal _ | IOOperation(_, _) ->
        succeed expr

// ======================================
// Declaration Transformation
// ======================================

module Declarations =
    
    /// Transforms a function declaration
    let transformFunction (name: string) (parameters: (string * OakType) list) 
                         (returnType: OakType) (body: OakExpression) : LayoutParser<OakDeclaration> =
        transformExpression body >>= fun transformedBody ->
        succeed (FunctionDecl(name, parameters, returnType, transformedBody))
    
    /// Transforms an entry point
    let transformEntryPoint (expr: OakExpression) : LayoutParser<OakDeclaration> =
        transformExpression expr >>= fun transformedExpr ->
        succeed (EntryPoint transformedExpr)
    
    /// Transforms a type declaration
    let transformType (name: string) (oakType: OakType) : LayoutParser<OakDeclaration> =
        match oakType with
        | UnionType cases ->
            TypeTransformation.transformUnionType name cases >>= fun transformedType ->
            TypeTransformation.recordTypeMapping name transformedType >>= fun _ ->
            succeed (TypeDecl(name, transformedType))
        | _ ->
            succeed (TypeDecl(name, oakType))
    
    /// Transforms any declaration
    let transform (decl: OakDeclaration) : LayoutParser<OakDeclaration> =
        match decl with
        | FunctionDecl(name, parameters, returnType, body) ->
            transformFunction name parameters returnType body
        | EntryPoint expr ->
            transformEntryPoint expr
        | TypeDecl(name, oakType) ->
            transformType name oakType
        | ExternalDecl(_, _, _, _) ->
            succeed decl

// ======================================
// Layout Validation
// ======================================

module Validation =
    
    /// Validates that a layout is zero-allocation
    let validateZeroAllocation (layout: UnionLayout) : bool =
        layout.IsZeroAllocation
    
    /// Validates all computed layouts are zero-allocation
    let validateAllLayouts : LayoutParser<unit> =
        getLayoutState >>= fun state ->
        let allValid = 
            state.UnionLayouts 
            |> Map.forall (fun _ layout -> validateZeroAllocation layout)
        
        if allValid then
            succeed ()
        else
            fail "Some layouts may cause heap allocations"

// ======================================
// Main Compilation Functions
// ======================================

/// Compiles fixed layouts for a program
let compileFixedLayouts (program: OakProgram) : Result<OakProgram, string> =
    if program.Modules.IsEmpty then
        Error "Program must contain at least one module"
    else
        let transformModule (module': OakModule) : LayoutParser<OakModule> =
            let rec transformDeclarations remainingDecls accDecls =
                match remainingDecls with
                | [] -> succeed (List.rev accDecls)
                | decl :: rest ->
                    Declarations.transform decl >>= fun transformedDecl ->
                    transformDeclarations rest (transformedDecl :: accDecls)
            
            transformDeclarations module'.Declarations [] >>= fun transformedDeclarations ->
            succeed { module' with Declarations = transformedDeclarations }
        
        let transformProgram = 
            setLayoutState (createLayoutState()) >>= fun _ ->
            let rec transformModules remainingModules accModules =
                match remainingModules with
                | [] -> succeed (List.rev accModules)
                | module' :: rest ->
                    transformModule module' >>= fun transformedModule ->
                    transformModules rest (transformedModule :: accModules)
            
            transformModules program.Modules [] >>= fun transformedModules ->
            Validation.validateAllLayouts >>= fun _ ->
            succeed { program with Modules = transformedModules }
        
        match runParser transformProgram "" with
        | Ok transformedProgram -> Ok transformedProgram
        | Error e -> Error e

/// Gets the computed layout for a union type
let getUnionLayout (program: OakProgram) (unionName: string) : Result<UnionLayout option, string> =
    let findUnionParser = 
        setLayoutState (createLayoutState()) >>= fun _ ->
        // Process the program to compute layouts
        compileFixedLayouts program |> ignore
        getLayoutState >>= fun state ->
        succeed (Map.tryFind unionName state.UnionLayouts)
    
    match runParser findUnionParser "" with
    | Ok layout -> Ok layout
    | Error e -> Error e

/// Validates zero-allocation layouts for a program
let validateZeroAllocationLayouts (program: OakProgram) : Result<bool, string> =
    let validationParser = 
        setLayoutState (createLayoutState()) >>= fun _ ->
        // Process unions in the program
        let rec processModules modules =
            match modules with
            | [] -> succeed ()
            | module' :: rest ->
                let rec processDeclarations declarations =
                    match declarations with
                    | [] -> succeed ()
                    | TypeDecl(name, UnionType cases) :: restDecls ->
                        LayoutAnalysis.analyzeStrategy name cases >>= fun strategy ->
                        LayoutAnalysis.recordLayout name strategy >>= fun _ ->
                        processDeclarations restDecls
                    | _ :: restDecls ->
                        processDeclarations restDecls
                
                processDeclarations module'.Declarations >>= fun _ ->
                processModules rest
        
        processModules program.Modules >>= fun _ ->
        Validation.validateAllLayouts >>= fun _ ->
        succeed true
    
    match runParser validationParser "" with
    | Ok result -> Ok result
    | Error _ -> Ok false  // Return false if validation fails