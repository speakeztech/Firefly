module Core.AST.Extraction

open FSharp.Compiler.Symbols
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Text

type TypedFunction = {
    Symbol: FSharpMemberOrFunctionOrValue
    FullName: string
    Range: range
    Body: FSharpExpr
    Module: string
    IsEntryPoint: bool
}


let extractFunctions (checkResults: FSharpCheckProjectResults) : TypedFunction[] =
    // Get the assembly signature for proper type resolution
    let assembly = checkResults.AssemblySignature
    
    // Track all types defined in this project
    let projectDefinedTypes = 
        assembly.Entities
        |> Seq.collect (fun entity -> 
            seq { 
                yield entity.FullName
                for nestedEntity in entity.NestedEntities do
                    yield nestedEntity.FullName 
            })
        |> Set.ofSeq
    
    // Extract with proper source tracking
    checkResults.AssemblyContents.ImplementationFiles
    |> Seq.collect (fun implFile ->
        let moduleName = implFile.QualifiedName
        let sourceFile = implFile.FileName
        
        let rec processDeclarations (decls: FSharpImplementationFileDeclaration list) =
            decls |> List.choose (fun decl ->
                match decl with
                | FSharpImplementationFileDeclaration.MemberOrFunctionOrValue(value, _, body) 
                    when value.IsFunction ->
                    let isEntryPoint = 
                        value.Attributes |> Seq.exists (fun attr -> 
                            attr.AttributeType.DisplayName = "EntryPoint") ||
                        (value.LogicalName = "main" && value.IsModuleValueOrMember)
                    
                    Some {
                        Symbol = value
                        FullName = value.FullName
                        Range = body.Range
                        Body = body
                        Module = moduleName
                        IsEntryPoint = isEntryPoint
                    }
                | FSharpImplementationFileDeclaration.Entity(_, subDecls) ->
                    processDeclarations subDecls |> List.tryHead
                | _ -> None)
        
        processDeclarations implFile.Declarations)
    |> Array.ofSeq