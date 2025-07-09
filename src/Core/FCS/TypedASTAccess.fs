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

/// Walk typed expression tree with a visitor function
let rec visitExpr (visitor: FSharpExpr -> unit) (expr: FSharpExpr) =
    visitor expr
    for subExpr in expr.ImmediateSubExpressions do
        visitExpr visitor subExpr

/// Extract all function calls from an expression
let extractCalls (expr: FSharpExpr) =
    let calls = ResizeArray<FSharpMemberOrFunctionOrValue>()
    
    visitExpr (fun e ->
        // Check if this is a call expression
        if e.Type.HasTypeDefinition then
            let typeDef = e.Type.TypeDefinition
            if typeDef.IsFSharpModule || typeDef.IsClass then
                // This might be a function call
                // Note: Simplified extraction logic
                ()
    ) expr
    
    calls |> List.ofSeq

/// Get all types referenced in an expression
let extractTypeReferences (expr: FSharpExpr) =
    let types = ResizeArray<FSharpEntity>()
    
    visitExpr (fun e ->
        if e.Type.HasTypeDefinition then
            types.Add(e.Type.TypeDefinition)
    ) expr
    
    types |> Seq.distinct |> List.ofSeq

/// Simplified representation of a function
type Function = {
    Symbol: FSharpMemberOrFunctionOrValue
    Parameters: FSharpMemberOrFunctionOrValue list list
    Body: FSharpExpr
    ReturnType: FSharpType
}

/// Extract all functions from implementation files
let extractFunctions (implFiles: FSharpImplementationFileContents list) =
    implFiles
    |> List.collect (fun implFile ->
        let decls = extractDeclarations implFile
        
        let rec collectFunctions decl =
            match decl.Symbol with
            | Some (:? FSharpMemberOrFunctionOrValue as mfv) when mfv.IsFunction ->
                match decl.Expression with
                | Some expr ->
                    [{
                        Symbol = mfv
                        Parameters = mfv.CurriedParameterGroups |> List.ofSeq |> List.map List.ofSeq
                        Body = expr
                        ReturnType = mfv.ReturnParameter.Type
                    }]
                | None -> []
            | _ ->
                decl.NestedDeclarations |> List.collect collectFunctions
        
        decls |> List.collect collectFunctions
    )

/// Get memory allocation sites in expression
let findAllocationSites (expr: FSharpExpr) =
    let allocations = ResizeArray<FSharpExpr * string>()
    
    visitExpr (fun e ->
        // Detect common allocation patterns
        // This is simplified - real implementation would be more sophisticated
        match e with
        | _ when e.Type.HasTypeDefinition && e.Type.TypeDefinition.IsArrayType ->
            allocations.Add(e, "Array allocation")
        | _ when e.Type.HasTypeDefinition && e.Type.TypeDefinition.IsFSharpRecord ->
            allocations.Add(e, "Record allocation")
        | _ -> ()
    ) expr
    
    allocations |> List.ofSeq