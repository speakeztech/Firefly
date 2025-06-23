module Core.MLIRGeneration.TypeMapping

open FSharp.Compiler.Symbols
open Core.MLIRGeneration.TypeSystem
open Core.MemoryLayout.LayoutAnalyzer

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

/// Map FSharp type to MLIR type
let rec mapType (ctx: TypeContext) (fsType: FSharpType) : MLIRType * TypeContext =
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
                        (mt :: types, c')) ([], ctx)
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
                        mapUnionType ctx fsType
                    | _ ->
                        (MLIRTypes.memref MLIRTypes.i8, ctx)  // Reference type
            else
                (MLIRTypes.i64, ctx)  // Unknown type
        
        // Cache the result
        let ctx'' = { ctx' with TypeMap = Map.add typeKey mlirType ctx'.TypeMap }
        (mlirType, ctx'')

/// Map struct type preserving field information
and mapStructType ctx fsType =
    let fields = fsType.TypeDefinition.FSharpFields |> Seq.toList
    let (fieldTypes, ctx') = 
        fields |> List.fold (fun (types, c) field ->
            let (ft, c') = mapType c field.FieldType
            (ft :: types, c')) ([], ctx)
    (MLIRTypes.struct_ (List.rev fieldTypes), ctx')

/// Map union type with layout optimization
and mapUnionType ctx (entity: FSharpEntity) =
    let cases = entity.UnionCases |> Seq.toList
    
    match UnionAnalysis.analyzeStrategy entity with
    | Some (Enum _) -> 
        (MLIRTypes.i32, ctx)
    | Some (Single caseType) ->
        mapType ctx caseType
    | Some (Option someType) ->
        let (someMLIR, ctx') = mapType ctx someType
        (MLIRTypes.nullable someMLIR, ctx')
    | _ ->
        // Tagged union: tag + largest payload
        // Calculate the maximum size needed
        let maxSize = 
            cases 
            |> List.map (fun case ->
                case.Fields |> Seq.sumBy (fun f -> 
                    let (t, _) = mapType ctx f.FieldType
                    match t.Width with
                    | Some w -> w
                    | None -> 8))  // Default size
            |> List.fold max 0
        (MLIRTypes.struct_ [MLIRTypes.i32; MLIRTypes.array MLIRTypes.i8 maxSize], ctx)

/// Get MLIR type for a symbol
let resolveSymbolType (ctx: TypeContext) (symbol: FSharpSymbol) =
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
            | 1 -> (MLIRTypes.memref MLIRTypes.i8, ctx)  // Single dimension array
            | n -> (MLIRTypes.memref MLIRTypes.i8, ctx)  // Multi-dimensional array
        elif entity.IsFSharpRecord then
            // Record: struct of all fields
            let fields = entity.FSharpFields |> Seq.toList
            let (fieldTypes, ctx') = 
                fields |> List.fold (fun (types, c) field ->
                    let (ft, c') = mapType c field.FieldType
                    (ft :: types, c')) ([], ctx)
            (MLIRTypes.struct_ (List.rev fieldTypes), ctx')
        elif entity.IsFSharpUnion then
            // Union: use our union analysis
            mapUnionType ctx entity
        elif entity.IsValueType then
            // Value type: struct of fields
            let fields = entity.FSharpFields |> Seq.toList
            let (fieldTypes, ctx') = 
                fields |> List.fold (fun (types, c) field ->
                    let (ft, c') = mapType c field.FieldType
                    (ft :: types, c')) ([], ctx)
            (MLIRTypes.struct_ (List.rev fieldTypes), ctx')
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