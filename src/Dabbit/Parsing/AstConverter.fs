module Dabbit.Parsing.AstConverter

open System
open System.Text.RegularExpressions
open Dabbit.Parsing.OakAst

/// Represents a simple F# expression for parsing
type SimpleExpression =
    | FunctionCall of name: string * args: string list
    | StringLiteral of string
    | Variable of string
    | Sequence of SimpleExpression list

/// Converts a simple expression to Oak AST expression
let rec private simpleExpressionToOak (expr: SimpleExpression) : OakExpression =
    match expr with
    | FunctionCall("printf", [text]) ->
        OakExpression.Application(OakExpression.Variable("printf"), [OakExpression.Literal(OakLiteral.StringLiteral(text))])
    | FunctionCall("printfn", [format; arg]) ->
        OakExpression.Application(OakExpression.Variable("printfn"), [OakExpression.Literal(OakLiteral.StringLiteral(format)); OakExpression.Variable(arg)])
    | FunctionCall("printfn", [text]) ->
        OakExpression.Application(OakExpression.Variable("printfn"), [OakExpression.Literal(OakLiteral.StringLiteral(text))])
    | FunctionCall("readLine", [varName]) ->
        OakExpression.Application(OakExpression.Variable("readLine"), [])
    | FunctionCall(name, args) ->
        let argExprs = args |> List.map (fun arg -> OakExpression.Variable(arg))
        OakExpression.Application(OakExpression.Variable(name), argExprs)
    | SimpleExpression.StringLiteral(text) ->
        OakExpression.Literal(OakLiteral.StringLiteral(text))
    | SimpleExpression.Variable(name) ->
        OakExpression.Variable(name)
    | Sequence(exprs) ->
        let oakExprs = exprs |> List.map simpleExpressionToOak
        match oakExprs with
        | [] -> OakExpression.Literal(OakLiteral.UnitLiteral)
        | [single] -> single
        | first :: rest ->
            rest |> List.fold (fun acc expr -> OakExpression.Sequential(acc, expr)) first

/// Parses a basic F# function call like printf "text" or printfn "Hello, %s!" name
let private parseFunctionCall (line: string) : SimpleExpression option =
    // Match printf "text" or similar patterns
    let printfPattern = @"printf\s+""([^""]*)""\s*$"
    let printfnPattern = @"printfn\s+""([^""]*)""\s+(\w+)\s*$"
    let printfnSimplePattern = @"printfn\s+""([^""]*)""\s*$"
    let readLinePattern = @"(\w+)\s*=\s*stdin\.ReadLine\(\)\s*$"
    
    let printfMatch = Regex.Match(line.Trim(), printfPattern)
    let printfnMatch = Regex.Match(line.Trim(), printfnPattern)
    let printfnSimpleMatch = Regex.Match(line.Trim(), printfnSimplePattern)
    let readLineMatch = Regex.Match(line.Trim(), readLinePattern)
    
    if printfMatch.Success then
        Some (FunctionCall("printf", [printfMatch.Groups.[1].Value]))
    elif printfnMatch.Success then
        Some (FunctionCall("printfn", [printfnMatch.Groups.[1].Value; printfnMatch.Groups.[2].Value]))
    elif printfnSimpleMatch.Success then
        Some (FunctionCall("printfn", [printfnSimpleMatch.Groups.[1].Value]))
    elif readLineMatch.Success then
        Some (FunctionCall("readLine", [readLineMatch.Groups.[1].Value]))
    else
        None

/// Parses F# source code into Oak AST
let parseAndConvertToOakAst (sourceCode: string) : OakProgram =
    let lines = sourceCode.Split('\n') |> Array.map (fun line -> line.Trim()) |> Array.filter (fun line -> not (String.IsNullOrEmpty(line)))
    
    // Extract module name
    let moduleName = 
        lines 
        |> Array.tryFind (fun line -> line.StartsWith("module "))
        |> Option.map (fun line -> line.Replace("module ", "").Trim())
        |> Option.defaultValue "Main"
    
    // Find function definitions
    let functionLines = 
        lines 
        |> Array.mapi (fun i line -> (i, line))
        |> Array.filter (fun (i, line) -> line.StartsWith("let ") && line.Contains("()"))
    
    let declarations = ResizeArray<OakDeclaration>()
    
    // Parse each function
    for (startIndex, funcLine) in functionLines do
        let funcName = 
            let pattern = @"let\s+(\w+)\s*\(\)"
            let match' = Regex.Match(funcLine, pattern)
            if match'.Success then match'.Groups.[1].Value else "unknown"
        
        // Find function body (lines until next function or end)
        let bodyStartIndex = startIndex + 1
        let bodyEndIndex = 
            lines 
            |> Array.mapi (fun i line -> (i, line))
            |> Array.skip (bodyStartIndex)
            |> Array.tryFind (fun (i, line) -> line.StartsWith("let ") || not (line.StartsWith(" ")))
            |> Option.map fst
            |> Option.defaultValue lines.Length
        
        let bodyLines = 
            lines.[bodyStartIndex..bodyEndIndex-1]
            |> Array.filter (fun line -> not (String.IsNullOrWhiteSpace(line)))
            |> Array.map (fun line -> line.Trim())
        
        // Parse function body
        let bodyExpressions = 
            bodyLines 
            |> Array.choose parseFunctionCall
            |> Array.toList
        
        let bodyExpression = 
            match bodyExpressions with
            | [] -> OakExpression.Literal(OakLiteral.UnitLiteral)
            | [single] -> simpleExpressionToOak single
            | multiple -> 
                let oakExprs = multiple |> List.map simpleExpressionToOak
                oakExprs |> List.reduce (fun acc expr -> OakExpression.Sequential(acc, expr))
        
        let funcDecl = OakDeclaration.FunctionDecl(funcName, [], OakType.UnitType, bodyExpression)
        declarations.Add(funcDecl)
    
    // Look for top-level function calls (like hello())
    let topLevelCalls = 
        lines 
        |> Array.filter (fun line -> 
            not (line.StartsWith("module ")) && 
            not (line.StartsWith("let ")) &&
            not (line.StartsWith(" ")) &&
            line.Contains("()"))
        |> Array.choose (fun line ->
            let pattern = @"(\w+)\s*\(\)"
            let match' = Regex.Match(line.Trim(), pattern)
            if match'.Success then 
                Some (OakExpression.Application(OakExpression.Variable(match'.Groups.[1].Value), []))
            else None)
    
    // If we have top-level calls, create an entry point
    if topLevelCalls.Length > 0 then
        let entryExpression = 
            if topLevelCalls.Length = 1 then
                topLevelCalls.[0]
            else
                topLevelCalls |> Array.reduce (fun acc expr -> OakExpression.Sequential(acc, expr))
        
        declarations.Add(OakDeclaration.EntryPoint(entryExpression))
    
    // If no functions were found, create a simple main
    if declarations.Count = 0 then
        declarations.Add(OakDeclaration.EntryPoint(OakExpression.Literal(OakLiteral.IntLiteral(0))))
    
    { 
        Modules = [{ 
            Name = moduleName
            Declarations = declarations |> Seq.toList 
        }] 
    }