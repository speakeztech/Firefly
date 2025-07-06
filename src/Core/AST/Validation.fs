module Core.AST.Validation

open System
open FSharp.Compiler.Symbols
open FSharp.Compiler.Text
open Core.AST.Extraction

type AllocationType = | HeapAllocation | ObjectConstruction | CollectionAllocation

type AllocationSite = {
    TypeName: string
    Location: range
    AllocationType: AllocationType
}

let private isAllocatingFunction (fullName: string) : bool =
    let allocatingPatterns = [
        "Microsoft.FSharp.Collections.List."
        "Microsoft.FSharp.Collections.Array.create"
        "Microsoft.FSharp.Core.Printf."
        "System.String."
    ]
    let safePatterns = [
        "Microsoft.FSharp.NativeInterop.NativePtr."
        "Microsoft.FSharp.Core.Operators."
        "Alloy."
    ]
    
    if safePatterns |> List.exists fullName.StartsWith then false
    else allocatingPatterns |> List.exists fullName.StartsWith

let rec private detectAllocationsInExpr (expr: FSharpExpr) : AllocationSite list =
    let allocations = ResizeArray<AllocationSite>()
    
    let rec traverse expr =
        match expr with
        | FSharpExpr.Call(_, memberFunc, _, _, args) ->
            if isAllocatingFunction memberFunc.FullName then
                allocations.Add({
                    TypeName = memberFunc.FullName
                    Location = expr.Range
                    AllocationType = HeapAllocation
                })
            args |> List.iter traverse
        | FSharpExpr.NewObject(objType, _, args) ->
            allocations.Add({
                TypeName = objType.TypeDefinition.FullName
                Location = expr.Range
                AllocationType = ObjectConstruction
            })
            args |> List.iter traverse
        | FSharpExpr.Application(funcExpr, _, args) ->
            traverse funcExpr
            args |> List.iter traverse
        | FSharpExpr.Let(binding, body) ->
            traverse binding.Body
            traverse body
        | FSharpExpr.Sequential(expr1, expr2) ->
            traverse expr1
            traverse expr2
        | _ -> ()
    
    traverse expr
    allocations |> List.ofSeq

let verifyZeroAllocation (reachableFunctions: TypedFunction[]) : Result<unit, AllocationSite[]> =
    let allAllocations = 
        reachableFunctions
        |> Array.collect (fun func -> detectAllocationsInExpr func.Body |> Array.ofList)
    
    if Array.isEmpty allAllocations then Ok () else Error allAllocations