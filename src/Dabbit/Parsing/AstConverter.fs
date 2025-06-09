module Dabbit.Parsing.AstConverter

open FSharp.Compiler.Syntax
open FSharp.Compiler.Text
open Dabbit.Parsing.OakAst

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
        Application(func, [arg])
    
    | SynExpr.LetOrUse(_, _, bindings, bodyExpr, _, _) ->
        // Process a let binding - just handle the simplest case
        match bindings with
        | [binding] ->
            match binding with
            | SynBinding(_, _, _, _, _, _, _, pat, _, valExpr, _, _, _) ->
                // Extract name from pattern without detailed pattern matching
                let name = "_" // Default name
                Let(name, synExprToOakExpression valExpr, synExprToOakExpression bodyExpr)
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

// Function declaration conversion - simplified to avoid complex pattern matching
let synBindingToOakFunctionDecl (binding: SynBinding) : OakDeclaration option =
    // Default fallback that avoids complex pattern matching
    Some(FunctionDecl("unknownFunction", [], UnitType, Literal(UnitLiteral)))

// Module declaration conversion
let convertModuleDecl (decl: SynModuleDecl) : OakDeclaration list =
    match decl with
    | SynModuleDecl.Let(_, bindings, _) ->
        bindings 
        |> List.choose (fun _ -> 
            // Default fallback without complex pattern matching
            Some(FunctionDecl("unknownFunction", [], UnitType, Literal(UnitLiteral))))
    
    | _ -> [] // Skip other declaration types for now

// Module conversion
let convertSynModuleOrNamespace (mdl: SynModuleOrNamespace) : OakModule =
    match mdl with
    | SynModuleOrNamespace(longId, _, _, _, _, _, _, _, identifiers) ->
        let moduleName = getIdentPath longId
        { Name = moduleName; Declarations = [] }

// Main entry point
let convertToOakAst (parsedInput: ParsedInput) : OakProgram =
    match parsedInput with
    | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
        { Modules = modules |> List.map convertSynModuleOrNamespace }
    
    | _ -> { Modules = [] }

// Public API
let convertSynType (synType: SynType) : OakType =
    synTypeToOakType synType

let convertSynExpr (expr: SynExpr) : OakExpression =
    synExprToOakExpression expr