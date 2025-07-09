module Core.MemoryLayout.LayoutAnalyzer

open FSharp.Compiler.Symbols
open FSharp.Compiler.Syntax

/// Layout strategy for discriminated unions
type UnionLayoutStrategy =
    | Tagged of tagSize: int * maxPayloadSize: int
    | Enum of count: int
    | Option of someType: FSharpType
    | Single of caseType: FSharpType

/// Memory layout information
type TypeLayout = {
    Size: int
    Alignment: int
    Strategy: UnionLayoutStrategy option
}

/// Compute size and alignment for FSharp types
module TypeMetrics =
    let rec sizeOf (fsType: FSharpType) : int =
        if fsType.IsAbbreviation then
            sizeOf fsType.AbbreviatedType
        elif fsType.IsTupleType then
            fsType.GenericArguments 
            |> Seq.sumBy sizeOf
        elif fsType.IsFunctionType then
            8 // Function pointer
        elif fsType.HasTypeDefinition && 
             (fsType.TypeDefinition.FullName.StartsWith("Microsoft.FSharp.Core.array") ||
              fsType.TypeDefinition.FullName.StartsWith("System.Array")) then
            8 // Array reference
        else
            match fsType.TypeDefinition.TryFullName with
            | Some "System.Int32" -> 4
            | Some "System.Int64" -> 8
            | Some "System.Single" -> 4
            | Some "System.Double" -> 8
            | Some "System.Boolean" -> 1
            | Some "System.Byte" -> 1
            | Some "System.String" -> 8
            | _ when fsType.HasTypeDefinition && fsType.TypeDefinition.IsValueType ->
                // Struct: sum of field sizes
                fsType.TypeDefinition.FSharpFields
                |> Seq.sumBy (fun f -> sizeOf f.FieldType)
            | _ -> 8 // Reference type
    
    let rec alignmentOf (fsType: FSharpType) : int =
        if fsType.IsAbbreviation then
            alignmentOf fsType.AbbreviatedType
        elif fsType.IsTupleType then
            fsType.GenericArguments
            |> Seq.map alignmentOf
            |> Seq.max
        else
            match fsType.TypeDefinition.TryFullName with
            | Some "System.Int32" -> 4
            | Some "System.Int64" -> 8
            | Some "System.Single" -> 4
            | Some "System.Double" -> 8
            | Some "System.Boolean" -> 1
            | Some "System.Byte" -> 1
            | _ when fsType.HasTypeDefinition && fsType.TypeDefinition.IsValueType ->
                // Struct: max alignment of fields
                fsType.TypeDefinition.FSharpFields
                |> Seq.map (fun f -> alignmentOf f.FieldType)
                |> Seq.fold max 1
            | _ -> 8 // Reference type alignment

/// Analyze union layout strategy
module UnionAnalysis =
    open TypeMetrics
    
    let analyzeStrategy (unionType: FSharpEntity) =
        let cases = unionType.UnionCases |> Seq.toList
        
        match cases with
        | [] -> None
        | [single] ->
            match single.Fields |> Seq.toList with
            | [] -> Some (Enum 1)
            | [field] -> Some (Single field.FieldType)
            | fields -> 
                let size = fields |> List.sumBy (fun f -> sizeOf f.FieldType)
                Some (Tagged(1, size))
        | _ ->
            let allNullary = cases |> List.forall (fun c -> c.Fields.Count = 0)
            if allNullary then
                Some (Enum cases.Length)
            else
                match cases with
                | [none; some] | [some; none] when none.Name = "None" && some.Name = "Some" ->
                    some.Fields
                    |> Seq.tryHead
                    |> Option.map (fun f -> Option f.FieldType)
                | _ ->
                    let maxPayload = 
                        cases 
                        |> List.map (fun c ->
                            c.Fields |> Seq.sumBy (fun f -> sizeOf f.FieldType))
                        |> List.fold max 0
                    let tagSize = if cases.Length <= 256 then 1 else 4
                    Some (Tagged(tagSize, maxPayload))

/// Compute complete layout for a type
let analyze (fsType: FSharpType) : TypeLayout =
    let size = TypeMetrics.sizeOf fsType
    let alignment = TypeMetrics.alignmentOf fsType
    let strategy = 
        if fsType.HasTypeDefinition && fsType.TypeDefinition.IsFSharpUnion then
            UnionAnalysis.analyzeStrategy fsType.TypeDefinition
        else None
    
    { Size = size; Alignment = alignment; Strategy = strategy }