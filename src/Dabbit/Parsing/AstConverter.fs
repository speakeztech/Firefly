module Dabbit.Parsing.AstConverter

open FSharp.Compiler.Syntax
open FSharp.Compiler.SyntaxTree
open FSharp.Compiler.Text
open Fantomas.Core
open Fantomas.FCS
open Dabbit.Parsing.OakAst

/// Converts F# AST to Oak AST
let convertToOakAst (parsedInput: ParsedInput) : OakProgram =
    let modules = 
        match parsedInput with
        | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _)) ->
            modules |> List.map (fun parsedModule ->
                match parsedModule with
                | SynModuleOrNamespace(idents, _, isModule, decls, _, _, _, range) ->
                    let moduleName = String.concat "." (idents |> List.map (fun ident -> ident.idText))

                    let oakDecls = decls |> List.choose (fun decl ->
                        match decl with
                        | SynModuleDecl.Let(_, bindings, _) ->
                            Some(FunctionDecl("placeholder", [], UnitType, Literal(UnitLiteral)))
                        | SynModuleDecl.Types(typeDefs, _) ->
                            Some(TypeDecl("placeholder", UnitType))
                        | SynModuleDecl.DoExpr(_, expr, _) ->
                            Some(EntryPoint(Literal(UnitLiteral)))
                        | _ -> None
                    )

                    { Name = moduleName; Declarations = oakDecls }
            )
        | _ -> []

    { Modules = modules }

/// Parses F# code and converts to Oak AST
let parseAndConvertToOakAst (sourceCode: string) : OakProgram =
    let sourceText = SourceText.ofString sourceCode
    let (parseTree, _) = CodeFormatter.ParseFile(sourceCode, "temp.fs", [], [||] |> List.ofArray)
    convertToOakAst parseTree
