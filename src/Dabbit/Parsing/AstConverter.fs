module Dabbit.Parsing.AstConverter

open System
open FSharp.Compiler.Syntax
open FSharp.Compiler.Text
open Fantomas.FCS
open Fantomas.Core
open Dabbit.Parsing.OakAst

/// Result type for F# to Oak AST conversion with diagnostics
type ASTConversionResult = {
    OakProgram: OakProgram
    Diagnostics: string list
}

/// Helper active patterns for handling F# AST
module private AstPatterns = 
    // Helper for extracting identifier text
    let (|IdentText|) (ident: Ident) = ident.idText
    
    // Helper for handling long identifiers
    let (|LongIdentText|) (lid: SynLongIdent) = 
        String.concat "." [for id in lid.LongIdent -> id.idText]

/// Core F# AST to Oak AST mapping functions
module AstMapping =
    open AstPatterns
    
    /// Maps Fantomas/FCS type to Oak type with proper type system mapping
    let rec mapType (synType: SynType) : OakType =
        match synType with
        | SynType.LongIdent(id) ->
            let name = id.ToString()
            match name.ToLowerInvariant() with
            | "int" | "int32" | "system.int32" -> IntType
            | "float" | "double" | "system.double" -> FloatType
            | "bool" | "boolean" | "system.boolean" -> BoolType
            | "string" | "system.string" -> StringType
            | "unit" -> UnitType
            | _ -> 
                // This would be extended to handle custom types better
                StructType([])
        
        | SynType.App(typeName, _, typeArgs, _, _, _, _) ->
            match typeName with
            | SynType.LongIdent(id) ->
                let name = id.ToString()
                if name = "array" || name = "[]" || name = "list" then
                    match typeArgs with
                    | [elemType] -> ArrayType(mapType elemType)
                    | _ -> UnitType  // Malformed generic
                else 
                    UnitType  // Unknown generic type
            | _ -> UnitType
        
        | SynType.Fun(paramType, returnType, _, _) ->
            FunctionType([mapType paramType], mapType returnType)
        
        | SynType.Tuple(_, segments, _) ->
            let fields = 
                segments 
                |> List.mapi (fun i segment -> 
                    let fieldType = 
                        match segment with
                        | SynTupleTypeSegment.Type(t) -> mapType t
                        | _ -> UnitType
                    (sprintf "Item%d" (i+1), fieldType))
            StructType(fields)
        
        | SynType.Array(_, elemType, _) ->
            ArrayType(mapType elemType)
        
        | _ -> UnitType
    
    /// Maps F# literal to Oak literal with proper value conversion
    let mapLiteral (constant: SynConst) : OakLiteral =
        match constant with
        | SynConst.Int32(n) -> IntLiteral(n)
        | SynConst.Double(f) -> FloatLiteral(f)
        | SynConst.Bool(b) -> BoolLiteral(b)
        | SynConst.String(s, _, _) -> StringLiteral(s)
        | SynConst.Unit -> UnitLiteral
        | SynConst.Char(c) -> StringLiteral(c.ToString())
        | SynConst.Byte(b) -> IntLiteral(int b)
        | SynConst.UInt16(u) -> IntLiteral(int u)
        | SynConst.Int16(s) -> IntLiteral(int s)
        | SynConst.UInt32(u) -> IntLiteral(int u)
        | SynConst.UInt64(u) -> IntLiteral(int u)
        | SynConst.Int64(i) -> IntLiteral(int i)
        | SynConst.SByte(s) -> IntLiteral(int s)
        | _ -> UnitLiteral
    
    /// Core recursive expression mapper using Fantomas's rich AST model
    let rec mapExpression (expr: SynExpr) : OakExpression =
        match expr with
        | SynExpr.Const(constant, _) ->
            Literal(mapLiteral constant)
        
        | SynExpr.Ident(IdentText name) ->
            Variable(name)
        
        | SynExpr.LongIdent(_, LongIdentText name, _, _) ->
            Variable(name)
        
        | SynExpr.App(_, isInfix, funcExpr, argExpr, _) ->
            let func = mapExpression funcExpr
            let arg = mapExpression argExpr
            
            // Special handling for printfn/printf using pattern matching
            match func with
            | Variable funcName when funcName = "printf" || funcName = "printfn" ->
                match arg with
                | Literal (StringLiteral formatStr) ->
                    // Simple case: printf "Hello, world!"
                    if funcName = "printf" then
                        IOOperation(Printf(formatStr), [])
                    else
                        IOOperation(Printfn(formatStr), [])
                | Application(Literal(StringLiteral formatStr), args) ->
                    // Handle printf "%d %s" arg1 arg2
                    if funcName = "printf" then
                        IOOperation(Printf(formatStr), args)
                    else
                        IOOperation(Printfn(formatStr), args)
                | _ ->
                    // Default fallback for complex cases
                    Application(func, [arg])
            | _ ->
                if isInfix then
                    // Handle infix operators properly
                    match func with
                    | Variable op when op = "op_Addition" -> 
                        // Detect common arithmetic operations
                        Application(Variable("+"), [mapExpression funcExpr; arg])
                    | Variable op when op = "op_Subtraction" -> 
                        Application(Variable("-"), [mapExpression funcExpr; arg])
                    | Variable op when op = "op_Multiply" -> 
                        Application(Variable("*"), [mapExpression funcExpr; arg])
                    | Variable op when op = "op_Division" -> 
                        Application(Variable("/"), [mapExpression funcExpr; arg])
                    | _ -> 
                        Application(func, [arg])
                else
                    Application(func, [arg])
        
        | SynExpr.LetOrUse(_, _, bindings, bodyExpr, _, _) ->
            // Map let-bindings one by one
            bindings 
            |> List.fold (fun bodyAcc binding ->
                match binding with
                | SynBinding(_, _, _, _, _, _, _, pat, returnTypeOpt, valExpr, _, _, _) ->
                    // Extract name from pattern using different approaches
                    let name =
                        match pat with
                        | SynPat.Named(_, id, _, _, _) -> 
                            // For Named pattern, just convert to string to avoid type issues
                            id.ToString()
                        | SynPat.LongIdent(lid, _, _, _, _, _) -> 
                            // Using correct 6-arg pattern for LongIdent
                            lid.ToString()
                        | _ -> "_"
                            
                    Let(name, mapExpression valExpr, bodyAcc)
            ) (mapExpression bodyExpr)
        
        | SynExpr.IfThenElse(condExpr, thenExpr, elseExprOpt, _, _, _, _) ->
            let cond = mapExpression condExpr
            let thenBranch = mapExpression thenExpr
            let elseBranch = 
                match elseExprOpt with
                | Some(elseExpr) -> mapExpression elseExpr
                | None -> Literal(UnitLiteral)
            IfThenElse(cond, thenBranch, elseBranch)
        
        | SynExpr.Sequential(_, _, first, second, _, _) ->
            Sequential(mapExpression first, mapExpression second)
        
        | SynExpr.Lambda(_, _, pats, body, _, trivia) ->
            // Map lambda parameters - a key improvement over the original
            let parameters = 
                pats 
                |> List.choose (function
                    | SynPat.Named(_, IdentText name, _, _, _) -> 
                        Some (name, UnitType)  // Type inference would improve this
                    | SynPat.Typed(SynPat.Named(_, IdentText name, _, _, _), synType, _) ->
                        Some (name, mapType synType)
                    | _ -> None)
            
            Lambda(parameters, mapExpression body)
        
        | SynExpr.Match(_, expr, clauses, _, _) ->
            // Map pattern matching - important for F# idioms
            let scrutinee = mapExpression expr
            
            // Map each clause to an if-then-else chain
            let foldedMatch =
                clauses
                |> List.fold (fun elseExpr clause ->
                    match clause with
                    | SynMatchClause(pattern, whenExpr, result, _, _, _) ->
                        // This is a simplification - a full implementation would handle patterns better
                        let condition = 
                            match pattern with
                            | SynPat.Const(constant, _) ->
                                Application(Variable("="), [scrutinee; Literal(mapLiteral constant)])
                            | SynPat.Named(_, IdentText name, _, _, _) ->
                                Application(Variable("="), [scrutinee; Variable(name)])
                            | SynPat.Wild(_) ->
                                // Wildcard pattern always matches
                                Literal(BoolLiteral(true))
                            | _ ->
                                // More complex patterns would need proper handling
                                Literal(BoolLiteral(true))
                                
                        // Add when clause if present
                        let finalCondition =
                            match whenExpr with
                            | Some guard -> 
                                Application(Variable("&&"), [condition; mapExpression guard])
                            | None -> condition
                            
                        IfThenElse(finalCondition, mapExpression result, elseExpr)
                ) (Literal(UnitLiteral))  // Default case
                
            foldedMatch
            
        | SynExpr.DotGet(expr, _, LongIdentText fieldName, _) ->
            let target = mapExpression expr
            FieldAccess(target, fieldName)
            
        | SynExpr.DotIndexedGet(expr, indexArgs, _, _) ->
            let target = mapExpression expr
            // Map indexed access (arrays, collections)
            let indices = 
                indexArgs 
                |> List.collect (fun (SynIndexerArg(exprs, _, _)) -> 
                    exprs |> List.map mapExpression)
            
            // Create a function application for indexed access
            match indices with
            | [index] -> 
                // Common case: single index
                Application(Variable("get_Item"), [target; index])
            | _ -> 
                // Multiple indices or none
                Application(Variable("get_Item"), target :: indices)
            
        | SynExpr.ArrayOrList(_, elements, _) ->
            // Map arrays/lists to ArrayLiteral
            let mappedElements = elements |> List.map mapExpression
            Literal(ArrayLiteral(mappedElements))
            
        | _ -> 
            // Default fallback for unsupported expressions
            Literal(UnitLiteral)
    
    /// Maps F# binding to function declaration or value binding
    let mapBinding (binding: SynBinding) : OakDeclaration option =
        match binding with
        | SynBinding(_, _, _, _, _, _, valData, SynPat.Named(SynPat.Wild(_), IdentText name, _, _, _), returnTypeOpt, expr, _, _, _) ->
            // Check for EntryPoint attribute
            let isEntryPoint = 
                valData.Attributes 
                |> List.exists (fun attrList -> 
                    attrList.Attributes 
                    |> List.exists (fun attr -> 
                        attr.TypeName.ToString().Contains("EntryPoint")))
                        
            if isEntryPoint then
                Some(EntryPoint(mapExpression expr))
            else
                let returnType = 
                    match returnTypeOpt with
                    | Some t -> mapType t
                    | None -> UnitType
                    
                Some(FunctionDecl(name, [], returnType, mapExpression expr))
                
        | SynBinding(_, _, _, _, _, _, _, SynPat.LongIdent(LongIdentText name, _, _, args, _, _), returnTypeOpt, expr, _, _, _) ->
            // Process function parameters with better type information
            let parameters = 
                args 
                |> List.choose (function
                    | SynPat.Named(_, IdentText name, _, _, _) ->
                        Some (name, UnitType)
                    | SynPat.Typed(SynPat.Named(_, IdentText name, _, _, _), synType, _) ->
                        // Extract type information when available
                        Some (name, mapType synType)
                    | _ -> None)
                    
            let returnType = 
                match returnTypeOpt with
                | Some t -> mapType t
                | None -> UnitType
                    
            Some(FunctionDecl(name, parameters, returnType, mapExpression expr))
            
        | _ -> None
    
    /// Maps a module declaration to Oak declarations
    let mapModuleDecl (decl: SynModuleDecl) : OakDeclaration list =
        match decl with
        | SynModuleDecl.Let(_, bindings, _) ->
            bindings 
            |> List.choose mapBinding
        
        | SynModuleDecl.Types(typeDefns, _) ->
            typeDefns
            |> List.collect (fun typeDef ->
                match typeDef with
                | SynTypeDefn(componentInfo, typeRepr, members, _, _) ->
                    let typeName = componentInfo.ToString()
                    
                    // Map type representation
                    let oakType =
                        match typeRepr with
                        | SynTypeDefnRepr.Simple(SynTypeDefnSimpleRepr.Union(_, cases, _), _) ->
                            // Map discriminated union
                            let unionCases =
                                cases
                                |> List.map (fun (SynUnionCase(_, ident, caseType, _, _, _, _)) ->
                                    let caseTypeOption = 
                                        match caseType with
                                        | SynUnionCaseType.UnionCaseFields(fields) ->
                                            if fields.IsEmpty then None
                                            else
                                                // For simplicity, just use the first field type
                                                match fields with
                                                | [SynField(_, _, Some(fieldType), _, _, _, _, _)] ->
                                                    Some(mapType fieldType)
                                                | _ -> 
                                                    Some(StructType(
                                                        fields 
                                                        |> List.mapi (fun i (SynField(_, _, fieldType, _, _, _, _, _)) ->
                                                            (sprintf "Field%d" i, 
                                                             match fieldType with 
                                                             | Some t -> mapType t 
                                                             | None -> UnitType))
                                                    ))
                                        | SynUnionCaseType.UnionCaseFullType(fullType, _) ->
                                            Some(mapType fullType)
                                    
                                    (ident.idText, caseTypeOption))
                            
                            UnionType(unionCases)
                        
                        | SynTypeDefnRepr.Simple(SynTypeDefnSimpleRepr.Record(_, fields, _), _) ->
                            // Map record type
                            let recordFields =
                                fields
                                |> List.choose (fun (SynField(_, _, fieldType, ident, _, _, _, _)) ->
                                    match ident, fieldType with
                                    | Some id, Some t -> Some (id.idText, mapType t)
                                    | _ -> None)
                            
                            StructType(recordFields)
                        
                        | _ -> UnitType
                    
                    // Create type declaration
                    [TypeDecl(typeName, oakType)]
                    
                    // Also map member declarations
                    @ (members 
                      |> List.choose (function
                          | SynMemberDefn.Member(binding, _) -> mapBinding binding
                          | _ -> None))
            )
        
        | SynModuleDecl.Open(_, _, _) ->
            // Track imports for module resolution
            []
            
        | _ -> [] // Skip other declaration types for now
    
    /// Maps module or namespace to Oak module
    let mapModule (mdl: SynModuleOrNamespace) : OakModule =
        let moduleName = mdl.ToString()
        let declarations = mdl.Decls |> List.collect mapModuleDecl
        { Name = moduleName; Declarations = declarations }

/// Main entry point for F# to Oak AST conversion
let parseAndConvertToOakAst (sourceCode: string) : OakProgram =
    try
        // Use Fantomas to parse the F# source code with better error handling
        let sourceText = SourceText.ofString sourceCode
        let checker = FSharp.Compiler.CodeAnalysis.FSharpChecker.Create()
        
        // Create parsing options with Firefly-specific defines
        let parsingOptions = 
            { FSharp.Compiler.CodeAnalysis.FSharpParsingOptions.Default with
                ConditionalDefines = ["FIREFLY"] }
        
        // Parse the source code
        let parseResults = 
            checker.ParseFile("input.fs", sourceText, parsingOptions)
            |> Async.RunSynchronously
        
        match parseResults.ParseTree with
        | Some parsedInput ->
            // Convert parsed input to Oak AST
            match parsedInput with
            | ParsedInput.ImplFile(implFile) ->
                { Modules = implFile.Modules |> List.map AstMapping.mapModule }
            | _ -> 
                { Modules = [] }
        | None ->
            // Return empty program if parsing failed
            { Modules = [] }
    with ex ->
        // Handle any exceptions during parsing/conversion
        { Modules = [] }

/// Full conversion with diagnostic information
let parseAndConvertWithDiagnostics (sourceCode: string) : ASTConversionResult =
    try
        // Use Fantomas for parsing with diagnostics
        let sourceText = SourceText.ofString sourceCode
        let checker = FSharp.Compiler.CodeAnalysis.FSharpChecker.Create()
        
        // Create parsing options with Firefly-specific defines
        let parsingOptions = 
            { FSharp.Compiler.CodeAnalysis.FSharpParsingOptions.Default with
                ConditionalDefines = ["FIREFLY"] }
        
        // Parse the source code
        let parseResults = 
            checker.ParseFile("input.fs", sourceText, parsingOptions)
            |> Async.RunSynchronously
        
        let diagnostics =
            parseResults.Diagnostics
            |> Array.map (fun diag -> diag.Message)
            |> Array.toList
        
        match parseResults.ParseTree with
        | Some parsedInput ->
            // Convert parsed input to Oak AST
            match parsedInput with
            | ParsedInput.ImplFile(implFile) ->
                let oakProgram = { Modules = implFile.Modules |> List.map AstMapping.mapModule }
                { OakProgram = oakProgram; Diagnostics = diagnostics }
            | _ -> 
                { OakProgram = { Modules = [] }; Diagnostics = "Unsupported input format" :: diagnostics }
        | None ->
            // Return empty program with parse error diagnostics
            { OakProgram = { Modules = [] }; Diagnostics = "Parse error" :: diagnostics }
    with ex ->
        // Handle any exceptions during parsing/conversion
        { 
            OakProgram = { Modules = [] }; 
            Diagnostics = [sprintf "Exception: %s" ex.Message; ex.StackTrace] 
        }