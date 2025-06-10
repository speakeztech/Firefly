module Dabbit.Parsing.AstConverter

open System
open FSharp.Compiler.Syntax
open FSharp.Compiler.Text
open Dabbit.Parsing.OakAst

/// Result type for F# to Oak AST conversion with diagnostics
type ASTConversionResult = {
    OakProgram: OakProgram
    Diagnostics: string list
}

/// Core F# AST to Oak AST mapping functions
module AstMapping =
    
    /// Maps basic types from F# to Oak representation
    let rec mapBasicType (typeName: string) : OakType =
        match typeName.ToLowerInvariant() with
        | "int" | "int32" | "system.int32" -> IntType
        | "float" | "double" | "system.double" -> FloatType
        | "bool" | "boolean" | "system.boolean" -> BoolType
        | "string" | "system.string" -> StringType
        | "unit" -> UnitType
        | _ when typeName.StartsWith("array") || typeName.Contains("[]") -> 
            ArrayType(IntType) // Simplified - would need to extract element type
        | _ -> StructType([]) // Default for unknown types
    
    /// Maps F# literal to Oak literal with simplified approach
    let mapLiteral (constant: SynConst) : OakLiteral =
        match constant with
        | SynConst.Int32 n -> IntLiteral(n)
        | SynConst.Double f -> FloatLiteral(f)
        | SynConst.Bool b -> BoolLiteral(b)
        | SynConst.String(s, _, _) -> StringLiteral(s)
        | SynConst.Unit -> UnitLiteral
        | _ -> UnitLiteral // Default for other literals
    
    /// Gets a name from an identifier, handling both simple and qualified names
    let getIdentifierName (ident: Ident) : string = ident.idText
    
    /// Gets a name from a list of identifiers (for qualified names)
    let getQualifiedName (idents: Ident list) : string =
        String.concat "." [for id in idents -> id.idText]
    
    /// Simplified expression mapper focusing on the most common cases
    let rec mapExpression (expr: SynExpr) : OakExpression =
        match expr with
        | SynExpr.Const(constant, _) ->
            Literal(mapLiteral constant)
        
        | SynExpr.Ident(ident) ->
            Variable(getIdentifierName ident)
        
        // Basic function application
        | SynExpr.App(_, _, funcExpr, argExpr, _) ->
            let func = mapExpression funcExpr
            let arg = mapExpression argExpr
            
            // Special case for printf/printfn
            match func with
            | Variable funcName when funcName = "printf" || funcName = "printfn" ->
                match arg with
                | Literal (StringLiteral formatStr) ->
                    if funcName = "printf" then
                        IOOperation(Printf(formatStr), [])
                    else
                        IOOperation(Printfn(formatStr), [])
                | _ -> Application(func, [arg])
            | _ -> Application(func, [arg])
        
        // Let bindings
        | SynExpr.LetOrUse(_, _, bindings, bodyExpr, _, _) ->
            // Extract just the first binding for simplicity
            match bindings with
            | binding :: _ ->
                match binding with
                | SynBinding(_, _, _, _, _, _, _, headPat, _, expr, _, _, _) ->
                    // Try to extract name from pattern with a more generic approach
                    let name = 
                        match headPat with
                        | SynPat.Named(_, id, _, _, _) -> getIdentifierName id
                        | _ -> "_"  // Default name if pattern can't be matched
                    
                    Let(name, mapExpression expr, mapExpression bodyExpr)
                | _ -> mapExpression bodyExpr
            | [] -> mapExpression bodyExpr
        
        // Basic if-then-else
        | SynExpr.IfThenElse(condExpr, thenExpr, elseExprOpt, _, _, _, _) ->
            let cond = mapExpression condExpr
            let thenBranch = mapExpression thenExpr
            let elseBranch = 
                match elseExprOpt with
                | Some(elseExpr) -> mapExpression elseExpr
                | None -> Literal(UnitLiteral)
            IfThenElse(cond, thenBranch, elseBranch)
        
        // Sequential expressions
        | SynExpr.Sequential(_, _, first, second, _, _) ->
            Sequential(mapExpression first, mapExpression second)
        
        // Default for other expressions
        | _ -> Literal(UnitLiteral)
    
    /// Maps a module or namespace declarations to Oak declarations with simplified approach
    let mapModuleDeclarations (decls: SynModuleDecl list) : OakDeclaration list =
        // Process only let bindings and function declarations for simplicity
        decls |> List.collect (function
            | SynModuleDecl.Let(_, bindings, _) ->
                bindings |> List.choose (function
                    | SynBinding(_, _, _, _, attrs, _, _, headPat, _, expr, _, _, _) ->
                        // Try to extract name and detect if it's an entry point
                        match headPat with
                        | SynPat.Named(_, id, _, _, _) ->
                            let name = getIdentifierName id
                            
                            // Check if it has EntryPoint attribute (simplified check)
                            let isEntryPoint = 
                                attrs |> List.exists (fun attrList ->
                                    attrList.Attributes |> List.exists (fun attr ->
                                        match attr.TypeName with
                                        | LongIdentWithDots(idents, _) -> 
                                            idents |> List.exists (fun id -> 
                                                id.idText.Contains("EntryPoint"))))
                            
                            if isEntryPoint then
                                Some(EntryPoint(mapExpression expr))
                            else
                                Some(FunctionDecl(name, [], UnitType, mapExpression expr))
                        | _ -> None
                )
            | _ -> []
        )
    
    /// Maps a module to Oak module with simplified approach
    let mapModule (mdl: SynModuleOrNamespace) : OakModule =
        match mdl with
        | SynModuleOrNamespace(ids, _, _, decls, _, _, _, _, _) ->
            let moduleName = getQualifiedName ids
            let declarations = mapModuleDeclarations decls
            { Name = moduleName; Declarations = declarations }

/// Main entry point for F# to Oak AST conversion with simpler approach
let parseAndConvertToOakAst (sourceCode: string) : OakProgram =
    try
        // Use FSharp.Compiler.Service to parse the F# source code
        let sourceText = SourceText.ofString sourceCode
        let checker = FSharp.Compiler.CodeAnalysis.FSharpChecker.Create()
        
        // Create basic parsing options
        let parsingOptions = FSharp.Compiler.CodeAnalysis.FSharpParsingOptions.Default
        
        // Parse the source code
        let parseResults = checker.ParseFile("input.fs", sourceText, parsingOptions) |> Async.RunSynchronously
        
        // Extract modules from parse tree, if available
        let modules =
            match parseResults.ParseTree with
            | None -> []
            | Some parsedInput -> 
                match parsedInput with
                | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
                    modules |> List.map AstMapping.mapModule
                | _ -> []
        
        // Return program with extracted modules
        { Modules = modules }
    with ex ->
        // Return empty program in case of errors
        { Modules = [] }

/// Full conversion with diagnostic information
let parseAndConvertWithDiagnostics (sourceCode: string) : ASTConversionResult =
    try
        // Parse source code
        let sourceText = SourceText.ofString sourceCode
        let checker = FSharp.Compiler.CodeAnalysis.FSharpChecker.Create()
        let parsingOptions = FSharp.Compiler.CodeAnalysis.FSharpParsingOptions.Default
        let parseResults = checker.ParseFile("input.fs", sourceText, parsingOptions) |> Async.RunSynchronously
        
        // Collect diagnostics
        let diagnostics =
            parseResults.Diagnostics
            |> Array.map (fun diag -> diag.Message)
            |> Array.toList
        
        // Convert to Oak AST
        let oakProgram =
            match parseResults.ParseTree with
            | None -> { Modules = [] }
            | Some parsedInput ->
                match parsedInput with
                | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
                    { Modules = modules |> List.map AstMapping.mapModule }
                | _ -> { Modules = [] }
        
        // Return result with program and diagnostics
        { OakProgram = oakProgram; Diagnostics = diagnostics }
    with ex ->
        // Return empty program with exception information
        { 
            OakProgram = { Modules = [] }
            Diagnostics = [sprintf "Exception: %s" ex.Message]
        }