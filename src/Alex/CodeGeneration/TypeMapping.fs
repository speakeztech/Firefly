module Alex.CodeGeneration.TypeMapping

open System
open FSharp.Compiler.Symbols
open FSharp.Compiler.Syntax
open Core.MemoryLayout.LayoutAnalyzer
open Core.Types.TypeSystem  // Import MLIRType from Core

/// Type mapping context preserving FCS information
type TypeContext = {
    TypeMap: Map<string, MLIRType>  // Use string keys instead of FSharpType
    Symbols: Map<string, FSharpSymbol>
    Generics: Map<string, FSharpGenericParameter>
}

/// Create mapping for basic types
let private basicTypeMap = 
    [
        "System.Int32", MLIRTypes.i32
        "System.Int64", MLIRTypes.i64
        "System.Single", MLIRTypes.f32
        "System.Double", MLIRTypes.f64
        "System.Boolean", MLIRTypes.i1
        "System.Byte", MLIRTypes.i8
        "System.Void", MLIRTypes.void_
        "System.String", MLIRTypes.memref MLIRTypes.i8
    ] |> Map.ofList

/// Get string representation of FSharpType for caching
let private getTypeKey (fsType: FSharpType) =
    fsType.Format(FSharpDisplayContext.Empty)

/// Map a struct type to MLIR
let rec private mapStructType (ctx: TypeContext) (fsType: FSharpType) : MLIRType * TypeContext =
    let fields = 
        match fsType.TypeDefinition with
        | def when def.IsFSharpRecord ->
            def.FSharpFields |> Seq.toList
        | _ ->
            fsType.TypeDefinition.FSharpFields |> Seq.toList
    
    let (fieldTypes, ctx') = 
        fields |> List.fold (fun (types, c) field ->
            let (ft, c') = mapType c field.FieldType
            ((field.Name, ft) :: types, c')) ([], ctx)
    
    (MLIRTypes.struct_ (List.rev fieldTypes), ctx')

/// Map a union type to MLIR
and private mapUnionType (ctx: TypeContext) (entity: FSharpEntity) : MLIRType * TypeContext =
    let cases = entity.UnionCases |> Seq.toList
    
    let (caseTypes, ctx') = 
        cases |> List.fold (fun (types, c) case ->
            let fields = case.Fields |> Seq.toList
            let (fieldTypes, c') = 
                fields |> List.fold (fun (ftypes, ctx'') field ->
                    let (ft, ctx''') = mapType ctx'' field.FieldType
                    ((field.Name, ft) :: ftypes, ctx''')) ([], c)
            
            let caseType = 
                if List.isEmpty fieldTypes then
                    MLIRTypes.i8  // Empty case, just a tag
                else
                    MLIRTypes.struct_ (List.rev fieldTypes)
            
            ((case.Name, caseType) :: types, c')) ([], ctx)
    
    // Create a union struct with tag and data fields
    (MLIRTypes.struct_ [("tag", MLIRTypes.i8); ("data", MLIRTypes.struct_ (List.rev caseTypes))], ctx')

/// Map FSharp type to MLIR type
and mapType (ctx: TypeContext) (fsType: FSharpType) : MLIRType * TypeContext =
    let typeKey = getTypeKey fsType
    
    // Check cache first
    match Map.tryFind typeKey ctx.TypeMap with
    | Some mlirType -> (mlirType, ctx)
    | None ->
        let (mlirType, ctx') = 
            if fsType.IsAbbreviation then
                mapType ctx fsType.AbbreviatedType
            elif fsType.IsTupleType then
                let (elemTypes, ctx') = 
                    fsType.GenericArguments 
                    |> Seq.fold (fun (types, c) t ->
                        let (mt, c') = mapType c t
                        ((sprintf "Item%d" (List.length types + 1), mt) :: types, c')) ([], ctx)
                (MLIRTypes.struct_ (List.rev elemTypes), ctx')
            elif fsType.IsFunctionType then
                let domain, range = 
                    match fsType.GenericArguments |> Seq.toList with
                    | [d; r] -> (d, r)
                    | _ -> failwith "Invalid function type"
                let (domainType, ctx1) = mapType ctx domain
                let (rangeType, ctx2) = mapType ctx1 range
                (MLIRTypes.func [domainType] rangeType, ctx2)
            elif fsType.IsGenericParameter then
                // Generic parameters default to i64
                (MLIRTypes.i64, ctx)
            elif fsType.HasTypeDefinition then
                // Check if it's an array type by name
                if fsType.TypeDefinition.FullName.StartsWith("Microsoft.FSharp.Core.array") ||
                   fsType.TypeDefinition.FullName.StartsWith("System.Array") then
                    match fsType.GenericArguments |> Seq.tryHead with
                    | Some elemType ->
                        let (mlirElem, ctx') = mapType ctx elemType
                        (MLIRTypes.memref mlirElem, ctx')
                    | None ->
                        (MLIRTypes.memref MLIRTypes.i8, ctx)
                else
                    match fsType.TypeDefinition.TryFullName with
                    | Some fullName when Map.containsKey fullName basicTypeMap ->
                        (Map.find fullName basicTypeMap, ctx)
                    | _ when fsType.TypeDefinition.IsValueType ->
                        mapStructType ctx fsType
                    | _ when fsType.TypeDefinition.IsFSharpUnion ->
                        mapUnionType ctx fsType.TypeDefinition
                    | _ ->
                        (MLIRTypes.memref MLIRTypes.i8, ctx)  // Reference types default to pointers
            else
                (MLIRTypes.i32, ctx)  // Default fallback
        
        // Cache the result
        let updatedTypeMap = Map.add typeKey mlirType ctx'.TypeMap
        (mlirType, { ctx' with TypeMap = updatedTypeMap })

/// Map F# symbol to MLIR type
let mapSymbol (ctx: TypeContext) (symbol: FSharpSymbol) : MLIRType * TypeContext =
    match symbol with
    | :? FSharpMemberOrFunctionOrValue as mfv ->
        mapType ctx mfv.FullType
    | :? FSharpEntity as entity ->
        // Construct the proper type for the entity
        if entity.IsFSharpAbbreviation then
            mapType ctx entity.AbbreviatedType
        elif entity.IsArrayType then
            // Array types need their element type
            match entity.ArrayRank with
            | 1 -> 
                // Get element type if available
                match entity.FSharpFields |> Seq.tryHead with
                | Some field -> mapType ctx field.FieldType
                | None -> (MLIRTypes.memref MLIRTypes.i8, ctx)  // Default
            | _ -> (MLIRTypes.memref MLIRTypes.i8, ctx)  // Multi-dimensional array
        elif entity.IsFSharpRecord then
            // Record: struct of all fields
            mapStructType ctx (entity.AsType())
        elif entity.IsFSharpUnion then
            // Union: use our union analysis
            mapUnionType ctx entity
        elif entity.IsValueType then
            // Value type: struct of fields
            mapStructType ctx (entity.AsType())
        elif entity.IsClass then
            // Class: reference type
            (MLIRTypes.memref MLIRTypes.i8, ctx)
        elif entity.IsInterface then
            // Interface: reference type
            (MLIRTypes.memref MLIRTypes.i8, ctx)
        else
            // Enum or other: treat as i32
            (MLIRTypes.i32, ctx)
    | :? FSharpField as field ->
        mapType ctx field.FieldType
    | _ -> (MLIRTypes.i64, ctx)

/// Type context builder functions
module TypeContextBuilder =
    /// Create initial context
    let create() = {
        TypeMap = Map.empty
        Symbols = Map.empty
        Generics = Map.empty
    }
    
    /// Add symbol to context
    let addSymbol name symbol ctx = {
        ctx with Symbols = Map.add name symbol ctx.Symbols
    }
    
    /// Add generic parameter
    let addGeneric name param ctx = {
        ctx with Generics = Map.add name param ctx.Generics
    }

/// Type conversion functions
module TypeConversion =
    /// Convert F# AST type to MLIR type
    let rec synTypeToMLIRType (synType: SynType) : MLIRType =
        match synType with
        | SynType.LongIdent(SynLongIdent(ids, _, _)) ->
            let typeName = ids |> List.map (fun id -> id.idText) |> String.concat "."
            match typeName with
            | "int" | "System.Int32" -> MLIRTypes.i32
            | "int64" | "System.Int64" -> MLIRTypes.i64
            | "float" | "System.Double" -> MLIRTypes.f64
            | "float32" | "single" | "System.Single" -> MLIRTypes.f32
            | "bool" | "System.Boolean" -> MLIRTypes.i1
            | "byte" | "System.Byte" -> MLIRTypes.i8
            | "unit" | "System.Void" -> MLIRTypes.void_
            | "string" | "System.String" -> MLIRTypes.memref MLIRTypes.i8
            | _ -> MLIRTypes.i32  // Default
        | SynType.App(typeName, _, innerTypes, _, _, _, _) ->
            match typeName with
            | SynType.LongIdent(SynLongIdent(ids, _, _)) ->
                let typeName = ids |> List.map (fun id -> id.idText) |> String.concat "."
                match typeName with
                | "array" | "System.Array" ->
                    match innerTypes with
                    | [innerType] -> 
                        let elemType = synTypeToMLIRType innerType
                        MLIRTypes.memref elemType
                    | _ -> MLIRTypes.memref MLIRTypes.i8
                | _ -> MLIRTypes.i32
            | _ -> MLIRTypes.i32
        | SynType.Array(rank, elementType, _) ->
            let elemType = synTypeToMLIRType elementType
            MLIRTypes.memref elemType
        | SynType.Tuple(isStruct, segments, _) ->
            // Extract the actual types from segments
            let fieldTypes = 
                segments 
                |> List.mapi (fun i segment -> 
                    match segment with
                    | SynTupleTypeSegment.Type(t) -> 
                        (sprintf "Item%d" (i + 1), synTypeToMLIRType t))
            MLIRTypes.struct_ fieldTypes
        | SynType.Fun(argType, returnType, _, trivia) ->
            let argType = synTypeToMLIRType argType
            let retType = synTypeToMLIRType returnType
            MLIRTypes.func [argType] retType
        | SynType.Paren(innerType, _) ->
            synTypeToMLIRType innerType
        | _ -> MLIRTypes.i32  // Default fallback

    /// Get bit width of an MLIR type
    let getBitWidth (t: MLIRType) : int =
        match t.BitWidth with
        | Some width -> width
        | None -> 32  // Default

/// Operations on TypeContext for type resolution and registration
module TypeContextOps =
    /// Try to resolve a type by name from the context
    let tryResolveType (ctx: TypeContext) (typeName: string) : MLIRType option =
        Map.tryFind typeName ctx.TypeMap
    
    /// Register a function's type signature in the context
    let registerFunction (ctx: TypeContext) (functionName: string) 
                        (paramTypes: MLIRType list) (returnType: MLIRType) : TypeContext =
        // For functions, we store the complete function type
        let funcType = MLIRTypes.func paramTypes returnType
        { ctx with TypeMap = Map.add functionName funcType ctx.TypeMap }
    
    /// Register a type mapping in the context
    let registerType (ctx: TypeContext) (typeName: string) (mlirType: MLIRType) : TypeContext =
        { ctx with TypeMap = Map.add typeName mlirType ctx.TypeMap }
    
    /// Check if a type is already registered
    let isTypeRegistered (ctx: TypeContext) (typeName: string) : bool =
        Map.containsKey typeName ctx.TypeMap
    
    /// Get all registered type names
    let getRegisteredTypes (ctx: TypeContext) : string list =
        ctx.TypeMap |> Map.toList |> List.map fst