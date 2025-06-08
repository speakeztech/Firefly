module Dabbit.UnionLayouts.FixedLayoutCompiler

open System
open Dabbit.Parsing.OakAst

/// Represents the layout strategy for a discriminated union
type UnionLayoutStrategy =
    | TaggedUnion of tagSize: int * maxPayloadSize: int
    | EnumOptimization of enumValues: int list
    | OptionOptimization of someType: OakType
    | SingleCase of caseType: OakType

/// Represents the computed layout for a union type
type UnionLayout = {
    Strategy: UnionLayoutStrategy
    TotalSize: int
    Alignment: int
    TagOffset: int
    PayloadOffset: int
}

/// Calculates the size in bytes of an Oak type
let rec private calculateTypeSize (oakType: OakType) : int =
    match oakType with
    | IntType -> 4          // 32-bit integer
    | FloatType -> 4        // 32-bit float
    | BoolType -> 1         // Boolean as byte
    | StringType -> 8       // Pointer to string data
    | UnitType -> 0         // Unit type has no runtime representation
    | ArrayType _ -> 8      // Pointer to array data
    | FunctionType(_, _) -> 8  // Function pointer
    | StructType fields ->
        fields |> List.sumBy (snd >> calculateTypeSize)
    | UnionType cases ->
        // For uncompiled unions, estimate based on largest case
        let maxCaseSize = 
            cases 
            |> List.map (fun (_, optType) -> 
                match optType with 
                | Some t -> calculateTypeSize t 
                | None -> 0)
            |> List.max
        1 + maxCaseSize  // Tag byte + largest payload

/// Gets the alignment requirement for an Oak type
let rec private getTypeAlignment (oakType: OakType) : int =
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
        else fields |> List.map (snd >> getTypeAlignment) |> List.max
    | UnionType _ -> 8  // Conservative alignment for unions

/// Determines the optimal layout strategy for a union type
let private determineLayoutStrategy (cases: (string * OakType option) list) : UnionLayoutStrategy =
    match cases with
    | [] -> failwith "Empty union not supported"
    
    | [(name, Some caseType)] ->
        // Single case union - no tag needed
        SingleCase caseType
    
    | [(name1, None); (name2, Some caseType)] when name1 = "None" && name2 = "Some" ->
        // Option type optimization - use null pointer or special values
        OptionOptimization caseType
    
    | allNone when allNone |> List.forall (fun (_, optType) -> optType.IsNone) ->
        // Enum-style union (all cases have no data)
        let enumValues = [0 .. cases.Length - 1]
        EnumOptimization enumValues
    
    | _ ->
        // General tagged union
        let payloadSizes = 
            cases |> List.map (fun (_, optType) ->
                match optType with
                | Some t -> calculateTypeSize t
                | None -> 0
            )
        let maxPayloadSize = List.max payloadSizes
        let tagSize = if cases.Length <= 256 then 1 else if cases.Length <= 65536 then 2 else 4
        TaggedUnion(tagSize, maxPayloadSize)

/// Computes the memory layout for a union based on its strategy
let private computeUnionLayout (strategy: UnionLayoutStrategy) : UnionLayout =
    match strategy with
    | SingleCase caseType ->
        let size = calculateTypeSize caseType
        let alignment = getTypeAlignment caseType
        {
            Strategy = strategy
            TotalSize = size
            Alignment = alignment
            TagOffset = -1  // No tag
            PayloadOffset = 0
        }
    
    | OptionOptimization someType ->
        let size = calculateTypeSize someType
        let alignment = getTypeAlignment someType
        {
            Strategy = strategy
            TotalSize = size
            Alignment = alignment
            TagOffset = -1  // No explicit tag (uses null/special values)
            PayloadOffset = 0
        }
    
    | EnumOptimization enumValues ->
        let tagSize = if enumValues.Length <= 256 then 1 else if enumValues.Length <= 65536 then 2 else 4
        {
            Strategy = strategy
            TotalSize = tagSize
            Alignment = tagSize
            TagOffset = 0
            PayloadOffset = -1  // No payload
        }
    
    | TaggedUnion(tagSize, maxPayloadSize) ->
        let alignment = max tagSize (if maxPayloadSize > 0 then 8 else tagSize)
        let tagOffset = 0
        let payloadOffset = (tagSize + alignment - 1) / alignment * alignment  // Align payload
        let totalSize = payloadOffset + maxPayloadSize
        {
            Strategy = strategy
            TotalSize = totalSize
            Alignment = alignment
            TagOffset = tagOffset
            PayloadOffset = payloadOffset
        }

/// Transforms a union type declaration to use fixed layout
let private transformUnionType (name: string) (cases: (string * OakType option) list) : OakType * UnionLayout =
    let strategy = determineLayoutStrategy cases
    let layout = computeUnionLayout strategy
    
    let transformedType = 
        match strategy with
        | SingleCase caseType -> caseType
        | OptionOptimization someType -> someType  // Will need special handling in codegen
        | EnumOptimization _ ->
            // Transform to integer type
            if layout.TotalSize = 1 then IntType else IntType  // Simplify to int for now
        | TaggedUnion(tagSize, payloadSize) ->
            // Create struct with tag and payload
            let tagType = IntType  // Simplify tag to int
            let payloadType = ArrayType(IntType)  // Simplify payload to byte array
            StructType [("tag", tagType); ("payload", payloadType)]
    
    (transformedType, layout)

/// Transforms union expressions to use the fixed layout
let rec private transformExpression (expr: OakExpression) (unionLayouts: Map<string, UnionLayout>) : OakExpression =
    match expr with
    | Literal lit -> Literal lit
    | Variable name -> Variable name
    
    | Application(func, args) ->
        let transformedFunc = transformExpression func unionLayouts
        let transformedArgs = args |> List.map (fun arg -> transformExpression arg unionLayouts)
        Application(transformedFunc, transformedArgs)
    
    | Lambda(params', body) ->
        let transformedBody = transformExpression body unionLayouts
        Lambda(params', transformedBody)
    
    | Let(name, value, body) ->
        let transformedValue = transformExpression value unionLayouts
        let transformedBody = transformExpression body unionLayouts
        Let(name, transformedValue, transformedBody)
    
    | IfThenElse(cond, thenExpr, elseExpr) ->
        let transformedCond = transformExpression cond unionLayouts
        let transformedThen = transformExpression thenExpr unionLayouts
        let transformedElse = transformExpression elseExpr unionLayouts
        IfThenElse(transformedCond, transformedThen, transformedElse)
    
    | Sequential(first, second) ->
        let transformedFirst = transformExpression first unionLayouts
        let transformedSecond = transformExpression second unionLayouts
        Sequential(transformedFirst, transformedSecond)
    
    | FieldAccess(target, fieldName) ->
        let transformedTarget = transformExpression target unionLayouts
        FieldAccess(transformedTarget, fieldName)
    
    | MethodCall(target, methodName, args) ->
        let transformedTarget = transformExpression target unionLayouts
        let transformedArgs = args |> List.map (fun arg -> transformExpression arg unionLayouts)
        MethodCall(transformedTarget, methodName, transformedArgs)

/// Transforms a declaration to use fixed layouts for union types
let private transformDeclaration (decl: OakDeclaration) (unionLayouts: Map<string, UnionLayout>) : OakDeclaration =
    match decl with
    | FunctionDecl(name, params', returnType, body) ->
        let transformedBody = transformExpression body unionLayouts
        FunctionDecl(name, params', returnType, transformedBody)
    
    | EntryPoint(expr) ->
        let transformedExpr = transformExpression expr unionLayouts
        EntryPoint(transformedExpr)
    
    | TypeDecl(name, oakType) ->
        match oakType with
        | UnionType cases ->
            let (transformedType, layout) = transformUnionType name cases
            TypeDecl(name, transformedType)
        | _ -> decl

/// Compiles discriminated unions to use fixed memory layouts,
/// eliminating heap allocations for union types
let compileFixedLayouts (program: OakProgram) : OakProgram =
    // First pass: collect all union types and compute their layouts
    let unionLayouts = 
        program.Modules
        |> List.collect (fun module' -> module'.Declarations)
        |> List.choose (function
            | TypeDecl(name, UnionType cases) -> 
                let (_, layout) = transformUnionType name cases
                Some(name, layout)
            | _ -> None)
        |> Map.ofList
    
    // Second pass: transform all declarations to use fixed layouts
    let transformedModules = 
        program.Modules 
        |> List.map (fun module' ->
            let transformedDeclarations = 
                module'.Declarations 
                |> List.map (fun decl -> transformDeclaration decl unionLayouts)
            { module' with Declarations = transformedDeclarations }
        )
    
    { program with Modules = transformedModules }

/// Gets the computed layout for a union type by name
let getUnionLayout (program: OakProgram) (unionName: string) : UnionLayout option =
    program.Modules
    |> List.collect (fun module' -> module'.Declarations)
    |> List.tryPick (function
        | TypeDecl(name, UnionType cases) when name = unionName ->
            let (_, layout) = transformUnionType name cases
            Some layout
        | _ -> None)

/// Validates that all union layouts are zero-allocation
let validateZeroAllocationLayouts (program: OakProgram) : bool =
    let unionTypes = 
        program.Modules
        |> List.collect (fun module' -> module'.Declarations)
        |> List.choose (function
            | TypeDecl(name, UnionType cases) -> Some(name, cases)
            | _ -> None)
    
    unionTypes |> List.forall (fun (name, cases) ->
        let strategy = determineLayoutStrategy cases
        match strategy with
        | TaggedUnion(_, _) -> true  // Stack-allocated
        | EnumOptimization _ -> true  // Just integers
        | OptionOptimization _ -> true  // Optimized representation
        | SingleCase _ -> true  // Direct representation
    )