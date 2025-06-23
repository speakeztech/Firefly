module Core.FCSProcessing.TypeExtractor

open FSharp.Compiler.Symbols
open FSharp.Compiler.CodeAnalysis

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
        let rec processEntity (path: string list) (entity: FSharpEntity) acc =
            let fullName = (path @ [entity.DisplayName]) |> String.concat "."
            let acc' = { acc with Entities = Map.add fullName entity acc.Entities }
            
            // Process union cases
            let acc'' = 
                if entity.IsUnion then
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
            |> Seq.fold (processEntity (path @ [entity.DisplayName])) acc''''
        
        let empty = { Entities = Map.empty; Members = Map.empty; Unions = Map.empty; Fields = Map.empty }
        
        results.PartialAssemblySignature.Entities
        |> Seq.fold (processEntity []) empty
        |> Some
    
    | _ -> None

/// Build type context from extracted types
let buildTypeContext (extracted: ExtractedTypes) =
    let ctx = Core.MLIRGeneration.TypeMapping.TypeContextBuilder.create()
    
    // Add all entities and members to context
    let ctx' = 
        extracted.Entities 
        |> Map.fold (fun c name entity ->
            Core.MLIRGeneration.TypeMapping.TypeContextBuilder.addSymbol name (entity :> FSharpSymbol) c) ctx
    
    extracted.Members
    |> Map.fold (fun c name mfv ->
        Core.MLIRGeneration.TypeMapping.TypeContextBuilder.addSymbol name (mfv :> FSharpSymbol) c) ctx'