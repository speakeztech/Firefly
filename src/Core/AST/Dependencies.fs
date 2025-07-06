module Core.AST.Dependencies

open System
open FSharp.Compiler.Symbols
open FSharp.Compiler.Text
open Core.AST.Extraction

type DependencyType =
    | DirectCall | AlloyLibraryCall | ObjectConstruction | ExternalCall

type Dependency = {
    From: string
    To: string
    CallSite: range
    Type: DependencyType
}

let private classifyDependency (targetFullName: string) : DependencyType =
    if targetFullName.StartsWith("Alloy.") || targetFullName.StartsWith("Fidelity.") then AlloyLibraryCall
    elif targetFullName.Contains("..ctor") then ObjectConstruction
    elif targetFullName.StartsWith("Microsoft.FSharp.") then ExternalCall
    else DirectCall

let rec private extractFromExpr (containingFunction: string) (expr: FSharpExpr) : Dependency list =
    let deps = ResizeArray<Dependency>()
    
    let addDep targetName range =
        if targetName <> containingFunction then
            deps.Add({
                From = containingFunction
                To = targetName
                CallSite = range
                Type = classifyDependency targetName
            })
    
    let rec traverse expr =
        match expr with
        | FSharpExpr.Call(objExprOpt, memberFunc, _, _, args) ->
            addDep memberFunc.FullName expr.Range
            objExprOpt |> Option.iter traverse
            args |> List.iter traverse
        | FSharpExpr.NewObject(objType, _, args) ->
            addDep objType.TypeDefinition.FullName expr.Range
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
        | FSharpExpr.IfThenElse(condExpr, thenExpr, elseExprOpt) ->
            traverse condExpr
            traverse thenExpr
            elseExprOpt |> Option.iter traverse
        | FSharpExpr.Value(valueToGet) when valueToGet.IsFunction ->
            addDep valueToGet.FullName expr.Range
        | _ -> ()
    
    traverse expr
    deps |> List.ofSeq

let buildDependencies (functions: TypedFunction[]) : Dependency[] =
    functions
    |> Array.collect (fun func ->
        extractFromExpr func.FullName func.Body |> Array.ofList)