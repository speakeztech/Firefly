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
    checkResults.AssemblyContents.ImplementationFiles
    |> Seq.collect (fun implFile ->
        let moduleName = implFile.QualifiedName
        
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