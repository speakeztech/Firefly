module Dabbit.Parsing.AstConverter

open System
open FSharp.Compiler.Syntax
open FSharp.Compiler.Text
open FSharp.Compiler.SyntaxTree
open Dabbit.Parsing.OakAst

/// Result type containing the original ASTs and converted program
type ASTConversionResult = {
    FCSAstText: string
    OakAstText: string
    OakProgram: OakProgram
}

// Helper functions
let getIdentPath (idents: Ident list) =
    idents |> List.map (fun id -> id.idText) |> String.concat "."

// Type conversion
let rec synTypeToOakType (synType: SynType) : OakType =
    match synType with
    | SynType.LongIdent(lid) ->
        let name = getIdentPath lid.LongIdent
        match name.ToLowerInvariant() with
        | "int" | "int32" | "system.int32" -> IntType
        | "float" | "double" | "system.double" -> FloatType
        | "bool" | "boolean" | "system.boolean" -> BoolType
        | "string" | "system.string" -> StringType
        | "unit" -> UnitType
        | _ -> StructType([])
    
    | SynType.App(typeName, _, typeArgs, _, _, _, _) ->
        match typeName with
        | SynType.LongIdent(lid) when 
            let name = getIdentPath lid.LongIdent
            name = "array" || name = "[]" || name = "list" ->
            match typeArgs with
            | [elemType] -> ArrayType(synTypeToOakType elemType)
            | _ -> UnitType
        | _ -> UnitType
    
    | SynType.Fun(paramType, returnType, _, _) ->
        FunctionType([synTypeToOakType paramType], synTypeToOakType returnType)
    
    | SynType.Tuple(_, segments, _) ->
        let fields = 
            segments 
            |> List.mapi (fun i segment -> 
                let fieldType = 
                    match segment with
                    | SynTupleTypeSegment.Type(t) -> synTypeToOakType t
                    | _ -> UnitType
                (sprintf "Item%d" (i+1), fieldType))
        StructType(fields)
    
    | SynType.Array(_, elemType, _) ->
        ArrayType(synTypeToOakType elemType)
    
    | _ -> UnitType

// Convert literals
let rec synConstToOakLiteral (constant: SynConst) : OakLiteral =
    match constant with
    | SynConst.Int32(n) -> IntLiteral(n)
    | SynConst.Double(f) -> FloatLiteral(f)
    | SynConst.Bool(b) -> BoolLiteral(b)
    | SynConst.String(s, _, _) -> StringLiteral(s)
    | SynConst.Unit -> UnitLiteral
    | _ -> UnitLiteral

// Expression conversion
let rec synExprToOakExpression (expr: SynExpr) : OakExpression =
    match expr with
    | SynExpr.Const(constant, _) ->
        Literal(synConstToOakLiteral constant)
    
    | SynExpr.Ident(ident) ->
        Variable(ident.idText)
    
    | SynExpr.LongIdent(_, lid, _, _) ->
        Variable(getIdentPath lid.LongIdent)
    
    | SynExpr.App(_, _, funcExpr, argExpr, _) ->
        let func = synExprToOakExpression funcExpr
        let arg = synExprToOakExpression argExpr
        
        // Special handling for printfn/printf
        match func with
        | Variable funcName when funcName = "printf" || funcName = "printfn" ->
            match arg with
            | Literal (StringLiteral formatStr) ->
                // Simple case: printf "Hello, world!"
                if funcName = "printf" then
                    IOOperation(Printf(formatStr), [])
                else
                    IOOperation(Printfn(formatStr), [])
            | _ ->
                // Default fallback for more complex printf patterns
                Application(func, [arg])
        | _ ->
            Application(func, [arg])
    
    | SynExpr.LetOrUse(_, _, bindings, bodyExpr, _, _) ->
        match bindings with
        | [binding] ->
            match binding with
            | SynBinding(_, _, _, _, _, _, _, pat, _, valExpr, _, _, _) ->
                // Extract name from pattern
                match pat with
                | SynPat.Named(_, ident, _, _, _) ->
                    Let(ident.idText, synExprToOakExpression valExpr, synExprToOakExpression bodyExpr)
                | SynPat.LongIdent(LongIdentWithDots(ident, _), _, _, _, _, _) ->
                    Let(getIdentPath ident, synExprToOakExpression valExpr, synExprToOakExpression bodyExpr)
                | _ -> 
                    Let("_", synExprToOakExpression valExpr, synExprToOakExpression bodyExpr)
        | _ -> Literal(UnitLiteral)
    
    | SynExpr.IfThenElse(condExpr, thenExpr, elseExprOpt, _, _, _, _) ->
        let cond = synExprToOakExpression condExpr
        let thenBranch = synExprToOakExpression thenExpr
        let elseBranch = 
            match elseExprOpt with
            | Some(elseExpr) -> synExprToOakExpression elseExpr
            | None -> Literal(UnitLiteral)
        IfThenElse(cond, thenBranch, elseBranch)
    
    | SynExpr.Sequential(_, _, first, second, _, _) ->
        Sequential(synExprToOakExpression first, synExprToOakExpression second)
    
    | _ -> Literal(UnitLiteral) // Default fallback

// Function declaration conversion
let synBindingToOakFunctionDecl (binding: SynBinding) : OakDeclaration option =
    match binding with
    | SynBinding(_, _, _, _, _, _, valData, SynPat.Named(SynPat.Wild(_), ident, _, _, _), returnTypeOpt, expr, _, _, _) ->
        // Check for EntryPoint attribute
        let isEntryPoint = 
            valData.Attributes 
            |> List.exists (fun attrList -> 
                attrList.Attributes 
                |> List.exists (fun attr -> 
                    attr.TypeName.ToString().Contains("EntryPoint")))
                    
        if isEntryPoint then
            Some(EntryPoint(synExprToOakExpression expr))
        else
            let returnType = 
                match returnTypeOpt with
                | Some t -> synTypeToOakType t
                | None -> UnitType
                
            Some(FunctionDecl(ident.idText, [], returnType, synExprToOakExpression expr))
            
    | SynBinding(_, _, _, _, _, _, _, SynPat.LongIdent(LongIdentWithDots(ident, _), _, _, args, _, _), returnTypeOpt, expr, _, _, _) ->
        // Process function parameters
        let parameters = 
            args 
            |> List.choose (fun arg ->
                match arg with
                | SynPat.Named(_, ident, isThis, _, _) ->
                    Some (ident.idText, UnitType) // Default to UnitType as type info is missing
                | _ -> None)
                
        let returnType = 
            match returnTypeOpt with
            | Some t -> synTypeToOakType t
            | None -> UnitType
                
        Some(FunctionDecl(getIdentPath ident, parameters, returnType, synExprToOakExpression expr))
        
    | _ -> None

// Module declaration conversion
let convertModuleDecl (decl: SynModuleDecl) : OakDeclaration list =
    match decl with
    | SynModuleDecl.Let(_, bindings, _) ->
        bindings 
        |> List.choose synBindingToOakFunctionDecl
    
    | _ -> [] // Skip other declaration types for now

// Module conversion
let convertSynModuleOrNamespace (mdl: SynModuleOrNamespace) : OakModule =
    match mdl with
    | SynModuleOrNamespace(longId, _, _, decls, _, _, _, _, _) ->
        let moduleName = getIdentPath longId
        let declarations = decls |> List.collect convertModuleDecl
        { Name = moduleName; Declarations = declarations }

// Main entry point for conversion
let convertToOakAst (parsedInput: ParsedInput) : OakProgram =
    match parsedInput with
    | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
        { Modules = modules |> List.map convertSynModuleOrNamespace }
    
    | _ -> { Modules = [] }

// Convert F# source code to Oak AST - CRITICAL MISSING FUNCTION
let parseAndConvertToOakAstWithIntermediate (sourceCode: string) : ASTConversionResult =
    try
        // Parse the F# source code using F# Compiler Services
        let sourceText = SourceText.ofString sourceCode
        let parsingOptions = FSharp.Compiler.CodeAnalysis.FSharpParsingOptions.Default
        
        let parseResults = 
            FSharp.Compiler.CodeAnalysis.FSharpParsingOptions.GetParsingOptionsFromCommandLineArgs(
                parsingOptions.SourceFiles,
                [| "--noframework"; "--mlcompatibility" |],
                parsingOptions.ConditionalDefines)
            |> fun opts -> FSharp.Compiler.CodeAnalysis.FSharpParsingOptions.WithDefines opts ["FIREFLY"]
            |> fun opts -> FSharp.Compiler.CodeAnalysis.FSharpChecker.Create().Parse(sourceText, "input.fs", opts)
            |> Async.RunSynchronously
        
        match parseResults.ParseTree with
        | Some parsedInput ->
            // Convert to text representation for debugging/intermediates
            let fcsAstText = sprintf "%A" parsedInput
            
            // Convert to our Oak AST
            let oakProgram = convertToOakAst parsedInput
            let oakAstText = sprintf "%A" oakProgram
            
            { 
                FCSAstText = fcsAstText
                OakAstText = oakAstText
                OakProgram = oakProgram 
            }
        | None ->
            let errorMessage = 
                parseResults.Diagnostics 
                |> Array.map (fun diag -> diag.Message)
                |> String.concat "\n"
                
            // Return empty program with error info
            { 
                FCSAstText = sprintf "Parse error: %s" errorMessage
                OakAstText = "No AST generated due to parse errors"
                OakProgram = { Modules = [] }
            }
    with ex ->
        // Handle any exceptions during parsing/conversion
        { 
            FCSAstText = sprintf "Exception: %s\n%s" ex.Message ex.StackTrace
            OakAstText = "No AST generated due to exception"
            OakProgram = { Modules = [] }
        }

// Public API
let convertSynType (synType: SynType) : OakType =
    synTypeToOakType synType

let convertSynExpr (expr: SynExpr) : OakExpression =
    synExprToOakExpression expr