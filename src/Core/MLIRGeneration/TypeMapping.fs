module Core.MLIRGeneration.TypeMapping

open FSharp.Compiler.Symbols
open Core.MemoryLayout.LayoutAnalyzer

/// Type mapping context preserving FCS information
type TypeContext = {
    TypeMap: Map<FSharpType, MLIRType>
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

/// Map FSharp type to MLIR type
let rec mapType (ctx: TypeContext) (fsType: FSharpType) : MLIRType * TypeContext =
    // Check cache first
    match Map.tryFind fsType ctx.TypeMap with
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
            elif fsType.IsArrayType then
                let elemType = fsType.GenericArguments |> Seq.head
                let (mlirElem, ctx') = mapType ctx elemType
                (MLIRTypes.memref mlirElem, ctx')
            elif fsType.HasTypeDefinition then
                match fsType.TypeDefinition.TryFullName with
                | Some fullName when Map.containsKey fullName basicTypeMap ->
                    (Map.find fullName basicTypeMap, ctx)
                | _ when fsType.TypeDefinition.IsValueType ->
                    mapStructType ctx fsType
                | _ when fsType.TypeDefinition.IsUnion ->
                    mapUnionType ctx fsType
                | _ ->
                    (MLIRTypes.memref MLIRTypes.i8, ctx)  // Reference type
            else
                (MLIRTypes.i64, ctx)  // Unknown type
        
        // Cache the result
        let ctx'' = { ctx' with TypeMap = Map.add fsType mlirType ctx'.TypeMap }
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
and mapUnionType ctx fsType =
    let cases = fsType.TypeDefinition.UnionCases |> Seq.toList
    
    match UnionAnalysis.analyzeStrategy fsType.TypeDefinition with
    | Some (Enum _) -> 
        (MLIRTypes.i32, ctx)
    | Some (Single caseType) ->
        mapType ctx caseType
    | Some (Option someType) ->
        let (someMLIR, ctx') = mapType ctx someType
        (MLIRTypes.nullable someMLIR, ctx')
    | _ ->
        // Tagged union: tag + largest payload
        let layout = analyze fsType
        (MLIRTypes.struct_ [MLIRTypes.i32; MLIRTypes.array MLIRTypes.i8 layout.Size], ctx)

/// Get MLIR type for a symbol
let resolveSymbolType (ctx: TypeContext) (symbol: FSharpSymbol) =
    match symbol with
    | :? FSharpMemberOrFunctionOrValue as mfv ->
        mapType ctx mfv.FullType
    | :? FSharpEntity as entity when entity.IsValueType || entity.IsClass ->
        let fsType = entity.AsType()
        mapType ctx fsType
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