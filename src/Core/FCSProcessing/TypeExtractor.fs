module Core.FCSProcessing.TypeExtractor

open FSharp.Compiler.Symbols
open FSharp.Compiler.CodeAnalysis
open Dabbit.CodeGeneration.TypeMapping

/// Extract and catalog all type information
type ExtractedTypes = {
    Entities: Map<string, FSharpEntity>
    Members: Map<string, FSharpMemberOrFunctionOrValue>
    Unions: Map<string, FSharpUnionCase list>
    Fields: Map<string, FSharpField list>
}

/// Extract types from checked results
let extractFromCheckResults (checkResults: FSharpCheckFileAnswer) =
    match checkResults with
    | FSharpCheckFileAnswer.Succeeded results ->
        let rec processEntity (path: string list) (entity: FSharpEntity) (acc: ExtractedTypes) =
            let fullName = (path @ [entity.DisplayName]) |> String.concat "."
            let acc' = { acc with Entities = Map.add fullName entity acc.Entities }
            
            // Process union cases
            let acc'' = 
                if entity.IsFSharpUnion then
                    let cases = entity.UnionCases |> Seq.toList
                    { acc' with Unions = Map.add fullName cases acc'.Unions }
                else acc'
            
            // Process fields
            let acc''' = 
                if entity.IsFSharpRecord || entity.IsValueType then
                    let fields = entity.FSharpFields |> Seq.toList
                    { acc'' with Fields = Map.add fullName fields acc''.Fields }
                else acc''
            
            // Process members
            let acc'''' = 
                entity.MembersFunctionsAndValues
                |> Seq.fold (fun a mfv ->
                    let memberName = fullName + "." + mfv.DisplayName
                    { a with Members = Map.add memberName mfv a.Members }) acc'''
            
            // Process nested entities
            entity.NestedEntities
            |> Seq.fold (fun acc nestedEntity -> processEntity (path @ [entity.DisplayName]) nestedEntity acc) acc''''
        
        let empty = { 
            Entities = Map.empty
            Members = Map.empty
            Unions = Map.empty
            Fields = Map.empty 
        }
        
        results.PartialAssemblySignature.Entities
        |> Seq.fold (fun acc entity -> processEntity [] entity acc) empty
        |> Some
    
    | _ -> None

/// Build type context from extracted types
let buildTypeContext (extracted: ExtractedTypes) =
    let ctx = TypeContextBuilder.create()
    
    // Add all entities to context
    let ctx' = 
        extracted.Entities 
        |> Map.fold (fun c name entity ->
            TypeContextBuilder.addSymbol name (entity :> FSharpSymbol) c) ctx
    
    // Add all members to context
    extracted.Members
    |> Map.fold (fun c name mfv ->
        TypeContextBuilder.addSymbol name (mfv :> FSharpSymbol) c) ctx'

/// Get entity by name
let getEntity (extracted: ExtractedTypes) (name: string) : FSharpEntity option =
    Map.tryFind name extracted.Entities

/// Get member by name
let getMember (extracted: ExtractedTypes) (name: string) : FSharpMemberOrFunctionOrValue option =
    Map.tryFind name extracted.Members

/// Get union cases for a type
let getUnionCases (extracted: ExtractedTypes) (typeName: string) : FSharpUnionCase list option =
    Map.tryFind typeName extracted.Unions

/// Get fields for a type
let getFields (extracted: ExtractedTypes) (typeName: string) : FSharpField list option =
    Map.tryFind typeName extracted.Fields

/// Check if a type is a union
let isUnionType (extracted: ExtractedTypes) (typeName: string) : bool =
    Map.containsKey typeName extracted.Unions

/// Check if a type is a record
let isRecordType (extracted: ExtractedTypes) (typeName: string) : bool =
    match Map.tryFind typeName extracted.Entities with
    | Some entity -> entity.IsFSharpRecord
    | None -> false

/// Get all type names
let getAllTypeNames (extracted: ExtractedTypes) : string list =
    extracted.Entities |> Map.toList |> List.map fst

/// Get all member names
let getAllMemberNames (extracted: ExtractedTypes) : string list =
    extracted.Members |> Map.toList |> List.map fst