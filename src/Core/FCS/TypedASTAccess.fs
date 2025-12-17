module Core.FCS.TypedASTAccess

open FSharp.Compiler.Symbols

/// Simplified view of typed declarations
type TypedDeclaration = {
    Symbol: FSharpSymbol option  // Made optional to handle init actions
    Expression: FSharpExpr option
    NestedDeclarations: TypedDeclaration list
}

/// Extract typed declarations from implementation file
let extractDeclarations (implFile: FSharpImplementationFileContents) =
    let rec processDeclaration = function
        | FSharpImplementationFileDeclaration.Entity (entity, subDecls) ->
            {
                Symbol = Some (entity :> FSharpSymbol)
                Expression = None
                NestedDeclarations = subDecls |> List.map processDeclaration
            }
        
        | FSharpImplementationFileDeclaration.MemberOrFunctionOrValue (mfv, args, expr) ->
            {
                Symbol = Some (mfv :> FSharpSymbol)
                Expression = Some expr
                NestedDeclarations = []
            }
        
        | FSharpImplementationFileDeclaration.InitAction expr ->
            // Module initialization has no specific symbol
            {
                Symbol = None
                Expression = Some expr
                NestedDeclarations = []
            }
    
    implFile.Declarations |> List.map processDeclaration

/// Simplified representation of a function
type Function = {
    Symbol: FSharpMemberOrFunctionOrValue
    Parameters: FSharpParameter list list  
    Body: FSharpExpr
    ReturnType: FSharpType
}

/// Extract all functions from implementation files
let extractFunctions (implFiles: FSharpImplementationFileContents list) =
    implFiles
    |> List.collect (fun implFile ->
        let decls = extractDeclarations implFile
        
        let rec collectFunctions (decl: TypedDeclaration) =
            match decl.Symbol with
            | Some symbol ->
                match symbol with
                | :? FSharpMemberOrFunctionOrValue as mfv when mfv.IsFunction ->
                    match decl.Expression with
                    | Some expr ->
                        [{
                            Symbol = mfv
                            Parameters = 
                                mfv.CurriedParameterGroups 
                                |> Seq.map (fun group -> group |> List.ofSeq) 
                                |> List.ofSeq  
                            Body = expr
                            ReturnType = mfv.ReturnParameter.Type
                        }]
                    | None -> []
                | _ ->
                    // Not a function, but check nested declarations
                    decl.NestedDeclarations |> List.collect collectFunctions
            | None ->
                // No symbol (e.g., init action), check nested declarations
                decl.NestedDeclarations |> List.collect collectFunctions
        
        decls |> List.collect collectFunctions
    )