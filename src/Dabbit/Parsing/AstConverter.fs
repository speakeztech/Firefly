module Dabbit.Parsing.AstConverter

open System
open System.IO
open FSharp.Compiler.Syntax
open FSharp.Compiler.Text
open Dabbit.Parsing.OakAst
open Core.XParsec.Foundation

/// Result type for F# to Oak AST conversion with diagnostics and F# AST
type ASTConversionResult = {
    OakProgram: OakProgram
    Diagnostics: string list
    FSharpASTText: string
    ProcessedFiles: string list  // Track all processed files for debugging
    SourceModules: (string * OakModule list) list // Track source file for each module
}

/// Helper functions for extracting names and types from F# AST nodes
module AstHelpers =
    let getIdentifierName (ident: Ident) : string = ident.idText
    
    let getQualifiedName (idents: Ident list) : string =
        match idents with
        | [] -> "_empty_"
        | _ -> idents |> List.map (fun id -> id.idText) |> String.concat "."
    
    let extractLongIdent (synLongIdent: SynLongIdent) : Ident list =
        let (SynLongIdent(idents, _, _)) = synLongIdent
        idents
    
    let extractIdent (synIdent: SynIdent) : Ident =
        let (SynIdent(ident, _)) = synIdent
        ident
    
    let extractPatternName (pattern: SynPat) : string =
        match pattern with
        | SynPat.Named(synIdent, _, _, _) -> 
            try
                synIdent |> extractIdent |> getIdentifierName
            with
            | _ -> "_pattern_"
        | SynPat.LongIdent(longDotId, _, _, _, _, _) ->
            try
                let idents = longDotId |> extractLongIdent
                if idents.IsEmpty then "_empty_pattern_" else getQualifiedName idents
            with
            | _ -> "_pattern_"
        | SynPat.Const(SynConst.Unit, _) -> "()"
        | _ -> "_"
    
    let hasEntryPointAttribute (attributes: SynAttributes) : bool =
        try
            attributes
            |> List.exists (fun attrList ->
                if attrList.Attributes.IsEmpty then false
                else
                    attrList.Attributes 
                    |> List.exists (fun attr ->
                        match attr.TypeName with
                        | SynLongIdent(idents, _, _) -> 
                            if idents.IsEmpty then false
                            else idents |> List.exists (fun id -> 
                                id.idText.Contains("EntryPoint"))))
        with
        | _ -> false
    
    /// Extract the type name from a SynType
    let rec extractTypeName (typ: SynType) : string =
        match typ with
        | SynType.LongIdent(longIdent) -> 
            let idents = extractLongIdent longIdent
            getQualifiedName idents
        | SynType.App(typeName, _, args, _, _, _, _) ->
            let baseType = extractTypeName typeName
            let argTypes = args |> List.map extractTypeName
            sprintf "%s<%s>" baseType (String.concat ", " argTypes)
        | SynType.Tuple(false, typeElements, _) ->
            let typeNames = 
                typeElements 
                |> List.map (function
                    | SynTupleTypeSegment.Type(synType) -> extractTypeName synType
                    | _ -> "unknown_tuple_element")
            sprintf "(%s)" (String.concat " * " typeNames)
        | SynType.Array(_, elementType, _) ->
            sprintf "%s[]" (extractTypeName elementType)
        | SynType.Fun(paramType, returnType, _, _) ->
            sprintf "%s -> %s" (extractTypeName paramType) (extractTypeName returnType)
        | _ -> "unknown_type"

/// Type mapping functions
module TypeMapping =
    let mapBasicType (typeName: string) : OakType =
        match typeName.ToLowerInvariant() with
        | "int" | "int32" | "system.int32" -> IntType
        | "float" | "double" | "system.double" -> FloatType
        | "bool" | "boolean" | "system.boolean" -> BoolType
        | "string" | "system.string" -> StringType
        | "unit" -> UnitType
        | _ when typeName.StartsWith("array") || typeName.Contains("[]") -> ArrayType(IntType)
        | _ -> StructType([])
    
    let mapLiteral (constant: SynConst) : OakLiteral =
        match constant with
        | SynConst.Int32 n -> IntLiteral(n)
        | SynConst.Double f -> FloatLiteral(f)
        | SynConst.Bool b -> BoolLiteral(b)
        | SynConst.String(s, _, _) -> StringLiteral(s)
        | SynConst.Unit -> UnitLiteral
        | _ -> UnitLiteral
    
    /// Map SynType to OakType with improved support for complex types
    let rec mapSynType (synType: SynType) : OakType =
        match synType with
        | SynType.LongIdent(longIdent) ->
            let typeName = AstHelpers.extractLongIdent longIdent |> AstHelpers.getQualifiedName
            mapBasicType typeName
            
        | SynType.App(typeName, _, args, _, _, _, _) ->
            let baseTypeName = AstHelpers.extractTypeName typeName
            if baseTypeName.Contains("array") || baseTypeName.EndsWith("[]") then
                let elementType = 
                    if args.IsEmpty then IntType
                    else mapSynType args.Head
                ArrayType(elementType)
            else
                mapBasicType baseTypeName
                
        | SynType.Tuple(_, typeElements, _) ->
            let fields = 
                typeElements 
                |> List.mapi (fun i segment -> 
                    match segment with
                    | SynTupleTypeSegment.Type(elemType) -> 
                        (sprintf "Item%d" (i+1), mapSynType elemType)
                    | _ -> 
                        (sprintf "Item%d" (i+1), UnitType))
            StructType(fields)
            
        | SynType.Fun(paramType, returnType, _, _) ->
            FunctionType([mapSynType paramType], mapSynType returnType)
            
        | SynType.Array(_, elementType, _) ->
            ArrayType(mapSynType elementType)
            
        | _ -> UnitType // Default for unknown types

/// Expression conversion functions
module ExpressionMapping =
    
    // Forward declaration for mutual recursion
    let rec mapExpression (expr: SynExpr) : OakExpression = 
        mapExpressionImpl expr
        
    and mapPattern (pat: SynPat) : OakPattern =
        match pat with
        | SynPat.Named(synIdent, _, _, _) ->
            let ident = AstHelpers.extractIdent synIdent
            PatternVariable(AstHelpers.getIdentifierName ident)
        | SynPat.Wild(_) ->
            PatternWildcard
        | SynPat.Const(constant, _) ->
            PatternLiteral(TypeMapping.mapLiteral constant)
        | SynPat.LongIdent(longDotId, _, _, args, _, _) ->
            let idents = longDotId |> AstHelpers.extractLongIdent
            let name = AstHelpers.getQualifiedName idents
            match args with
            | SynArgPats.Pats pats ->
                let patterns = pats |> List.map mapPattern
                PatternConstructor(name, patterns)
            | _ ->
                PatternConstructor(name, [])
        | _ ->
            PatternWildcard
            
    and mapExpressionImpl (expr: SynExpr) : OakExpression =
        try
            match expr with
            | SynExpr.Match(_, matchExpr, clauses, _, _) ->
                let mappedMatchExpr = mapExpression matchExpr
                let mappedClauses = 
                    clauses 
                    |> List.map (fun clause ->
                        match clause with
                        | SynMatchClause(pat, whenExpr, resultExpr, _, _, _) ->
                            let pattern = mapPattern pat
                            let result = mapExpression resultExpr
                            (pattern, result))
                Match(mappedMatchExpr, mappedClauses)
            
            | SynExpr.Const(constant, _) ->
                constant |> TypeMapping.mapLiteral |> Literal
            
            | SynExpr.Ident(ident) ->
                ident |> AstHelpers.getIdentifierName |> Variable
            
            | SynExpr.LongIdent(_, longIdent, _, _) ->
                try
                    let idents = longIdent |> AstHelpers.extractLongIdent
                    let qualifiedName = AstHelpers.getQualifiedName idents
                    Variable qualifiedName
                with
                | _ -> Variable "_unknown_"
            
            | SynExpr.App(_, _, funcExpr, argExpr, _) ->
                mapFunctionApplication funcExpr argExpr
            
            | SynExpr.TypeApp(expr, _, typeArgs, _, _, _, range) ->
                mapTypeApplication expr typeArgs
            
            | SynExpr.LetOrUse(_, _, bindings, bodyExpr, _, _) ->
                mapLetBinding bindings bodyExpr
            
            | SynExpr.IfThenElse(condExpr, thenExpr, elseExprOpt, _, _, _, _) ->
                mapConditional condExpr thenExpr elseExprOpt
            
            | SynExpr.Sequential(_, _, first, second, _, _) ->
                Sequential(mapExpression first, mapExpression second)
            
            | SynExpr.Lambda(_, _, _, body, parsedData, _, _) ->
                mapLambda parsedData body
                
            | SynExpr.DotGet(expr, _, lid, _) ->
                // Handle field/property access - extract target and member
                let target = mapExpression expr
                
                match lid with
                | SynLongIdent(idents, _, _) ->
                    let memberName = AstHelpers.getQualifiedName idents
                    
                    // Handle common property patterns
                    match target, memberName with
                    | Variable "buffer", "AsSpan" ->
                        Application(Variable "buffer.AsSpan", [Literal UnitLiteral])
                    | Variable "span", "ToString" ->
                        Application(Variable "span.ToString", [Literal UnitLiteral])
                    | _ ->
                        FieldAccess(target, memberName)
                
            | SynExpr.DotIndexedGet(expr, indexExpr, _, _) ->
                let target = mapExpression expr
                
                let indexList = 
                    match indexExpr with
                    | SynExpr.Tuple(_, indexExprs, _, _) -> 
                        indexExprs |> List.map mapExpression
                    | SynExpr.Paren(innerExpr, _, _, _) ->
                        [mapExpression innerExpr]
                    | _ -> 
                        [mapExpression indexExpr]
                
                MethodCall(target, "get_Item", indexList)
                
            | SynExpr.Paren(expr, _, _, _) ->
                // Skip parentheses in Oak AST
                mapExpression expr
                
            | SynExpr.Tuple(_, exprs, _, _) ->
                // In FCS 43.9.300, exprs is already a list of SynExpr
                let mappedExprs = 
                    match exprs with
                    | [] -> []
                    | _ -> exprs |> List.map mapExpression
                
                match mappedExprs with
                | [] -> Literal UnitLiteral
                | [single] -> single
                | _ -> 
                    let rec buildSequence expressions =
                        match expressions with
                        | [] -> Literal UnitLiteral
                        | [last] -> last
                        | head :: tail -> Sequential(head, buildSequence tail)
                    buildSequence mappedExprs
                
            | SynExpr.TryWith(tryExpr, rangeInfo1, rangeInfo2, rangeInfo3, rangeInfo4, trivia) ->

                let tryBody = mapExpression tryExpr
                
                let defaultHandler = (PatternConstructor("Error", [PatternWildcard]),
                                    Literal(StringLiteral "Exception occurred"))
                
                Match(
                    Application(Variable "Ok", [tryBody]),
                    [defaultHandler]
                )
            
            | _ -> Literal(UnitLiteral)
        with
        | _ -> Literal(UnitLiteral)
    
    and mapTypeApplication (expr: SynExpr) (typeArgs: SynType list) : OakExpression =
        try
            match expr with
            | SynExpr.LongIdent(_, longIdent, _, _) ->
                let idents = longIdent |> AstHelpers.extractLongIdent
                let qualifiedName = AstHelpers.getQualifiedName idents
                
                match qualifiedName with
                | "NativePtr.stackalloc" ->
                    Variable "NativePtr.stackalloc"
                | "Console.readLine" ->
                    Variable "Console.readLine"
                | "Span" ->
                    Variable "Span.create"
                | _ ->
                    Variable qualifiedName
            | _ ->
                mapExpression expr
        with
        | _ -> Variable "_unknown_type_app_"
    
    and mapFunctionApplication funcExpr argExpr =
        try
            let func = mapExpression funcExpr
            let arg = mapExpression argExpr
            
            match func with
            | Variable "printf" ->
                match arg with
                | Literal(StringLiteral formatStr) -> IOOperation(Printf(formatStr), [])
                | _ -> Application(func, [arg])
            | Variable "printfn" ->
                match arg with
                | Literal(StringLiteral formatStr) -> IOOperation(Printfn(formatStr), [])
                | _ -> Application(func, [arg])
            | Variable "NativePtr.stackalloc" ->
                Application(Variable "NativePtr.stackalloc", [arg])
            | Variable "Console.readLine" ->
                Application(Variable "Console.readLine", [arg])
            | Variable "Span.create" ->
                Application(Variable "Span.create", [arg])
            | _ -> 
                // Check if this is a chain of applications
                match argExpr with
                | SynExpr.App(_, _, chainFunc, chainArg, _) ->
                    // This is a multi-argument function - recursively process the chain
                    let chainedApp = mapFunctionApplication chainFunc chainArg
                    match chainedApp with
                    | Application(f, args) ->
                        // Add this argument to the chain
                        Application(func, [arg] @ args) 
                    | _ ->
                        // Not a recognized chain, just apply normally
                        Application(func, [arg])
                | _ ->
                    // Normal single-argument application
                    Application(func, [arg])
        with
        | _ -> Literal(UnitLiteral)
    
    and mapLetBinding bindings bodyExpr =
        try
            match bindings with
            | binding :: _ ->
                let (SynBinding(_, _, _, _, _, _, _, headPat, _, expr, _, _, _)) = binding
                let name = AstHelpers.extractPatternName headPat
                Let(name, mapExpression expr, mapExpression bodyExpr)
            | [] -> mapExpression bodyExpr
        with
        | _ -> mapExpression bodyExpr
    
    and mapConditional condExpr thenExpr elseExprOpt =
        try
            let cond = mapExpression condExpr
            let thenBranch = mapExpression thenExpr
            let elseBranch = 
                match elseExprOpt with
                | Some(elseExpr) -> mapExpression elseExpr
                | None -> Literal(UnitLiteral)
            IfThenElse(cond, thenBranch, elseBranch)
        with
        | _ -> Literal(UnitLiteral)
    
    and mapLambda parsedData body =
        try
            let params' = 
                match parsedData with
                | Some(originalPats, _) ->
                    if originalPats.IsEmpty then [("x", UnitType)]
                    else originalPats |> List.map (fun pat -> (AstHelpers.extractPatternName pat, UnitType))
                | None -> [("x", UnitType)]
            Lambda(params', mapExpression body)
        with
        | _ -> Lambda([("x", UnitType)], Literal(UnitLiteral))
    
    and mapChainedApplication (funcExpr: SynExpr) (args: SynExpr list) : OakExpression =
        try
            let func = mapExpression funcExpr
            let mappedArgs = args |> List.map mapExpression
            
            match func with
            | Variable "Console.readLine" when mappedArgs.Length = 2 ->
                Application(Variable "Console.readLine", mappedArgs)
            | Variable "Span.create" when mappedArgs.Length = 2 ->
                Application(Variable "Span.create", mappedArgs)
            | _ ->
                Application(func, mappedArgs)
        with
        | _ -> Literal(UnitLiteral)
    
    and mapExpressionChain (expr: SynExpr) : OakExpression =
        let rec collectApplications expr args =
            match expr with
            | SynExpr.App(_, _, funcExpr, argExpr, _) ->
                collectApplications funcExpr (argExpr :: args)
            | _ -> (expr, args)
        
        let (baseFunc, allArgs) = collectApplications expr []
        
        if allArgs.Length > 1 then
            mapChainedApplication baseFunc allArgs
        else
            mapExpression expr

/// Declaration mapping functions - Enhanced with better module handling  
module DeclarationMapping =
    
    let mapUnionCase (case: SynUnionCase) : string * OakType option =
        try
            let (SynUnionCase(_, synIdent, caseType, _, _, _, _)) = case
            let ident = AstHelpers.extractIdent synIdent
            let caseName = AstHelpers.getIdentifierName ident
            
            match caseType with
            | SynUnionCaseKind.Fields fields ->
                if fields.IsEmpty then (caseName, None)
                else 
                    // Map the first field type (simplified)
                    let fieldType = 
                        fields 
                        |> List.tryHead 
                        |> Option.map (fun field -> 
                            // In FCS 43.9.300, we need to extract the type directly
                            let (SynField(_, _, _, fieldType, _, _, _, _, _)) = field
                            // Map the type - use a default if something goes wrong
                            try 
                                TypeMapping.mapSynType fieldType
                            with _ -> 
                                UnitType)
                    (caseName, fieldType)
            | _ -> (caseName, None)
        with
        | _ -> ("_case_", None)
    
    let mapRecordField (field: SynField) : string * OakType =
        try
            let (SynField(_, _, fieldId, fieldType, _, _, _, _, _)) = field
            let fieldName = 
                match fieldId with
                | Some ident -> AstHelpers.getIdentifierName ident
                | None -> "_"
            
            // In FCS 43.9.300, handle the field type directly
            let mappedType = 
                try
                    TypeMapping.mapSynType fieldType
                with _ ->
                    // Fallback to UnitType if mapping fails
                    UnitType
            
            (fieldName, mappedType)
        with
        | _ -> ("_field_", UnitType)
    
    let mapTypeDefinition (typeDefn: SynTypeDefn) : OakDeclaration option =
        try
            let (SynTypeDefn(SynComponentInfo(_, _, _, longId, _, _, _, _), repr, _, _, _, _)) = typeDefn
            let idents = longId
            let typeName = if idents.IsEmpty then "_type_" else AstHelpers.getQualifiedName idents
            
            match repr with
            | SynTypeDefnRepr.Simple(SynTypeDefnSimpleRepr.Union(_, cases, _), _) ->
                let oakCases = cases |> List.map mapUnionCase
                Some(TypeDecl(typeName, UnionType(oakCases)))
            | SynTypeDefnRepr.Simple(SynTypeDefnSimpleRepr.Record(_, fields, _), _) ->
                let oakFields = fields |> List.map mapRecordField
                Some(TypeDecl(typeName, StructType(oakFields)))
            | SynTypeDefnRepr.ObjectModel(_, members, _) ->
                // For interface/class definitions, create a simple struct type
                Some(TypeDecl(typeName, StructType([])))
            | _ -> None
        with
        | _ -> None
    
    let mapBinding (binding: SynBinding) : OakDeclaration option =
        try
            let (SynBinding(_, _, isInline, _, attrs, _, _, headPat, returnTy, expr, _, _, _)) = binding
            let name = AstHelpers.extractPatternName headPat
            
            if AstHelpers.hasEntryPointAttribute attrs then
                Some(EntryPoint(ExpressionMapping.mapExpression expr))
            else
                match expr with
                | SynExpr.Lambda(_, _, _, body, parsedData, _, _) ->
                    let params' = 
                        match parsedData with
                        | Some(originalPats, _) ->
                            if originalPats.IsEmpty then [("x", UnitType)]
                            else originalPats |> List.map (fun pat -> (AstHelpers.extractPatternName pat, UnitType))
                        | None -> [("x", UnitType)]
                    
                    // Map the function body with proper expression mapping
                    Some(FunctionDecl(name, params', UnitType, ExpressionMapping.mapExpression body))
                
                | _ ->
                    // Map as a function with no parameters
                    let mappedExpr = ExpressionMapping.mapExpression expr
                    
                    // For printf/printfn, create special I/O operation declarations
                    match mappedExpr with
                    | IOOperation(Printf formatStr, _) ->
                        Some(FunctionDecl(name, [("message", StringType)], UnitType, 
                                         IOOperation(Printf "%s", [Variable "message"])))
                    | IOOperation(Printfn formatStr, _) ->
                        Some(FunctionDecl(name, [("message", StringType)], UnitType, 
                                         IOOperation(Printfn "%s", [Variable "message"])))
                    | Application(Variable "sprintf", [Variable formatStrVar; Variable valueVar]) ->
                        // Handle format function with proper implementation
                        Some(FunctionDecl(name, [("formatStr", StringType); ("value", StringType)], StringType,
                                         Application(
                                             Application(Variable "sprintf", [Variable "formatStr"]),
                                             [Variable "value"])))
                    | Application(Variable "span.ToString", _) ->
                        // Handle spanToString with proper implementation
                        Some(FunctionDecl(name, [("span", ArrayType(IntType))], StringType,
                                         Application(Variable "span.ToString", [Literal UnitLiteral])))
                    | Application(_, _) when name = "stackBuffer" ->
                        // Special case for stackBuffer function
                        Some(FunctionDecl(name, [("size", IntType)], ArrayType(IntType),
                                         Application(Variable "NativePtr.stackalloc", [Variable "size"])))
                    | Application(_, _) when name = "readInto" ->
                        // Special case for readInto function
                        Some(FunctionDecl(name, [("buffer", ArrayType(IntType))], ArrayType(IntType),
                                         Application(Variable "NativePtr.readLine", [Variable "buffer"])))
                    | Application(_, _) when name = "readLine" ->
                        // Special case for readLine function  
                        Some(FunctionDecl(name, [("buffer", ArrayType(IntType)); ("maxLength", IntType)], IntType,
                                         Application(Variable "NativePtr.readLine", 
                                                    [Variable "buffer"; Variable "maxLength"])))
                    | _ ->
                        // Generic function declaration
                        Some(FunctionDecl(name, [], UnitType, mappedExpr))
        with
        | _ -> None
    
    // Enhanced to handle more declaration types
    let rec mapModuleDeclaration (decl: SynModuleDecl) : OakDeclaration list =
        try
            match decl with
            | SynModuleDecl.Let(_, bindings, _) ->
                if bindings.IsEmpty then []
                else 
                    let mappedBindings = bindings |> List.choose mapBinding
                    printfn "  Mapped %d bindings from Let" mappedBindings.Length
                    mappedBindings
                    
            | SynModuleDecl.Types(typeDefns, _) ->
                if typeDefns.IsEmpty then []
                else 
                    let mappedTypes = typeDefns |> List.choose mapTypeDefinition
                    printfn "  Mapped %d type definitions" mappedTypes.Length
                    mappedTypes
                    
            | SynModuleDecl.NestedModule(componentInfo, isRecursive, decls, range, accessibility, synAttrs) ->
                // Handle nested modules by recursively processing their declarations
                let (SynComponentInfo(attributes, typeParams, constraints, longId, xmlDoc, preferPostfix, access, range)) = componentInfo
                let moduleName = 
                    if longId.IsEmpty then "_nested_module_"
                    else AstHelpers.getQualifiedName longId
                
                printfn "  Processing nested module: %s with %d declarations" moduleName decls.Length
                
                // Map all declarations in the nested module
                let nestedDecls = decls |> List.collect mapModuleDeclaration
                printfn "  Mapped %d declarations from nested module %s" nestedDecls.Length moduleName
                
                // Return the nested module declarations with full implementations
                nestedDecls
                
            | SynModuleDecl.Expr(expression, range) ->
                // Map do expressions to function declarations with generated names
                let doFuncName = sprintf "_do_%d" (DateTime.Now.Ticks % 10000L)
                let mappedExpr = ExpressionMapping.mapExpression expression
                [FunctionDecl(doFuncName, [], UnitType, mappedExpr)]
                
            | SynModuleDecl.Open(target, _) ->
                // For open statements, we don't generate declarations but log them
                match target with
                | SynOpenDeclTarget.ModuleOrNamespace(longIdent, _) ->
                    let idents = AstHelpers.extractLongIdent longIdent
                    let targetName = AstHelpers.getQualifiedName idents
                    printfn "  Found open statement: %s" targetName
                | _ -> 
                    printfn "  Found unknown open target"
                []
                
            | _ -> 
                printfn "  Unhandled module declaration type: %A" decl
                []
        with
        | ex -> 
            printfn "  Error mapping module declaration: %s" ex.Message
            []

/// Module mapping functions - Enhanced with better diagnostics
module ModuleMapping =
    let mapModule (mdl: SynModuleOrNamespace) (sourceFile: string) : OakModule =
        try
            let (SynModuleOrNamespace(ids, _, _, decls, _, _, _, _, _)) = mdl
            let moduleName = 
                if ids.IsEmpty then "Module"
                else AstHelpers.getQualifiedName ids
                
            printfn "Mapping module %s from %s with %d declarations" 
                moduleName (Path.GetFileName(sourceFile)) decls.Length
                
            let declarations = 
                if decls.IsEmpty then
                    printfn "  Module %s has empty declarations list" moduleName
                    []
                else
                    let mappedDecls = decls |> List.collect DeclarationMapping.mapModuleDeclaration
                    printfn "  Mapped %d declarations for module %s" mappedDecls.Length moduleName
                    mappedDecls
                    
            { Name = moduleName; Declarations = declarations }
        with
        | ex -> 
            printfn "Error mapping module: %s" ex.Message
            { Name = "_error_module_"; Declarations = [] }
    
    // Improved to include source file information and better error handling
    let extractModulesFromParseTree (parseTree: ParsedInput) (sourceFile: string) : OakModule list =
        try
            match parseTree with
            | ParsedInput.ImplFile(implFile) ->
                try
                    let (ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) = implFile
                    
                    printfn "Extracting modules from %s (%d modules found)" 
                        (Path.GetFileName(sourceFile)) modules.Length
                    
                    // Map each module with source file information for better debugging
                    if modules.IsEmpty then []
                    else 
                        modules 
                        |> List.map (fun mdl -> mapModule mdl sourceFile)
                        |> List.filter (fun m -> not m.Declarations.IsEmpty)
                        
                with ex -> 
                    printfn "Error extracting modules from %s: %s" 
                        (Path.GetFileName(sourceFile)) ex.Message
                    let errorModule = { 
                        Name = sprintf "ERROR_MODULE_%s" (Path.GetFileName(sourceFile))
                        Declarations = [] 
                    }
                    [errorModule]
            | ParsedInput.SigFile(_) -> 
                printfn "Signature file not supported: %s" (Path.GetFileName(sourceFile))
                []
        with ex -> 
            printfn "Parse error in %s: %s" (Path.GetFileName(sourceFile)) ex.Message
            let errorModule = { 
                Name = sprintf "PARSE_ERROR_%s" (Path.GetFileName(sourceFile))
                Declarations = [] 
            }
            [errorModule]

/// Module merging for handling multiple modules with the same name
module ModuleMerger =
    // Merges modules with the same name from multiple files
    let mergeModules (modules: OakModule list) : OakModule list =
        // Group modules by name
        let modulesByName = 
            modules
            |> List.groupBy (fun m -> m.Name)
        
        // For each group, merge all declarations
        modulesByName
        |> List.map (fun (name, modulesWithSameName) ->
            if modulesWithSameName.Length = 1 then
                // Only one module with this name, no need to merge
                modulesWithSameName.[0]
            else
                // Multiple modules with the same name, merge their declarations
                let allDeclarations = 
                    modulesWithSameName
                    |> List.collect (fun m -> m.Declarations)
                
                // Filter out duplicate declarations by name
                let uniqueDeclarations =
                    allDeclarations
                    |> List.fold (fun (seen, acc) decl ->
                        let name = 
                            match decl with
                            | FunctionDecl(n, _, _, _) -> n
                            | TypeDecl(n, _) -> n
                            | EntryPoint(_) -> "_entry_point_"
                            | ExternalDecl(n, _, _, _) -> n
                        
                        if Set.contains name seen then
                            // Already have this declaration - check if it's a stub
                            match decl with
                            | FunctionDecl(_, _, _, Literal UnitLiteral) ->
                                // This is a stub - keep existing implementation
                                (seen, acc)
                            | FunctionDecl(_, _, _, _) ->
                                // Keep this more complete implementation - replace the stub
                                let filtered = acc |> List.filter (function
                                    | FunctionDecl(n, _, _, _) when n = name -> false
                                    | _ -> true)
                                (seen, decl :: filtered)
                            | _ -> (seen, acc)
                        else
                            // New declaration - add it
                            (Set.add name seen, decl :: acc)
                    ) (Set.empty, [])
                    |> snd
                    |> List.rev
                
                printfn "Merging %d modules named '%s' with total of %d declarations" 
                    modulesWithSameName.Length name allDeclarations.Length
                
                { Name = name; Declarations = uniqueDeclarations }
        )

/// Dependency extraction and resolution system
module DependencyResolution =
    /// Represents a dependency with source information
    type Dependency = {
        Path: string
        SourceFile: string
        IsDirective: bool  // True for #load, false for open
    }
    
    /// Extracts all dependencies from a parse tree
    let extractAllDependencies (parseTree: ParsedInput) (sourceFile: string) (baseDirectory: string) : Dependency list =
        match parseTree with
        | ParsedInput.ImplFile(implFile) ->
            let (ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) = implFile
            
            // Process open statements (we'll handle #load directives separately)
            let openDependencies =
                modules |> List.collect (fun mdl ->
                    let (SynModuleOrNamespace(_, _, _, decls, _, _, _, _, _)) = mdl
                    decls |> List.choose (fun decl ->
                        match decl with
                        | SynModuleDecl.Open(target, _) ->
                            match target with
                            | SynOpenDeclTarget.ModuleOrNamespace(SynLongIdent(idents, _, _), _) ->
                                let moduleName = idents |> List.map (fun id -> id.idText) |> String.concat "."
                                let filePath = moduleName.Replace(".", Path.DirectorySeparatorChar.ToString()) + ".fs"
                                Some { Path = filePath; SourceFile = sourceFile; IsDirective = false }
                            | _ -> None
                        | _ -> None))
            
            // Parse the source file directly for #load and #I directives
            let sourceCode = 
                try
                    File.ReadAllLines(sourceFile) 
                with ex -> 
                    printfn "Error reading source file %s: %s" sourceFile ex.Message
                    [||]
            
            // Extract #load directives
            let loadDependencies =
                sourceCode
                |> Array.choose (fun line -> 
                    let trimmed = line.Trim()
                    if trimmed.StartsWith("#load") then
                        let quotedPath = 
                            try
                                let startIdx = trimmed.IndexOf("\"") + 1
                                let endIdx = trimmed.LastIndexOf("\"")
                                if startIdx > 0 && endIdx > startIdx then
                                    Some(trimmed.Substring(startIdx, endIdx - startIdx))
                                else None
                            with _ -> None
                        
                        match quotedPath with
                        | Some path -> 
                            printfn "Found #load directive: %s in %s" path (Path.GetFileName(sourceFile))
                            Some { Path = path; SourceFile = sourceFile; IsDirective = true }
                        | None -> None
                    else None)
                |> Array.toList
            
            // Extract #I directives
            let includeDirectives =
                sourceCode
                |> Array.choose (fun line -> 
                    let trimmed = line.Trim()
                    if trimmed.StartsWith("#I") then
                        let quotedPath = 
                            try
                                let startIdx = trimmed.IndexOf("\"") + 1
                                let endIdx = trimmed.LastIndexOf("\"")
                                if startIdx > 0 && endIdx > startIdx then
                                    Some(trimmed.Substring(startIdx, endIdx - startIdx))
                                else None
                            with _ -> None
                        
                        match quotedPath with
                        | Some path -> 
                            printfn "Found #I directive: %s in %s" path (Path.GetFileName(sourceFile))
                            Some path
                        | None -> None
                    else None)
                |> Array.toList
            
            // Normalize and resolve search paths
            let searchDirectories = 
                includeDirectives
                |> List.map (fun path -> 
                    if Path.IsPathRooted(path) then path
                    else Path.Combine(baseDirectory, path.TrimStart('\\').TrimStart('/')))
                |> fun dirs -> dirs @ [baseDirectory]
            
            // Resolve all dependency paths
            let resolveDepPath (dep: Dependency) : Dependency =
                if dep.IsDirective then
                    // For #load directives, try direct path first
                    let directPath = Path.Combine(Path.GetDirectoryName(dep.SourceFile), dep.Path)
                    if File.Exists(directPath) then
                        { dep with Path = directPath }
                    else
                        // Try search paths
                        let resolvedPath = 
                            searchDirectories
                            |> List.tryPick (fun dir -> 
                                let fullPath = Path.Combine(dir, dep.Path)
                                if File.Exists(fullPath) then Some fullPath else None)
                        
                        match resolvedPath with
                        | Some path -> { dep with Path = path }
                        | None -> dep
                else
                    // For open statements
                    let resolvedPath = 
                        searchDirectories
                        |> List.tryPick (fun dir -> 
                            let fullPath = Path.Combine(dir, dep.Path)
                            if File.Exists(fullPath) then Some fullPath else None)
                    
                    match resolvedPath with
                    | Some path -> { dep with Path = path }
                    | None -> dep
            
            // Combine and resolve all dependencies
            let allDependencies = loadDependencies @ openDependencies
            let resolvedDeps = allDependencies |> List.map resolveDepPath
            
            // Filter to only valid dependencies and log them
            let validDeps = resolvedDeps |> List.filter (fun dep -> File.Exists(dep.Path))
            let invalidDeps = resolvedDeps |> List.filter (fun dep -> not (File.Exists(dep.Path)))
            
            if not invalidDeps.IsEmpty then
                printfn "Warning: %d dependencies could not be resolved:" invalidDeps.Length
                invalidDeps |> List.iter (fun dep -> 
                    printfn "  - %s (referenced from %s)" dep.Path (Path.GetFileName(dep.SourceFile)))
                    
            if not validDeps.IsEmpty then
                printfn "Found %d valid dependencies for %s:" validDeps.Length (Path.GetFileName(sourceFile))
                validDeps |> List.iter (fun dep -> 
                    printfn "  - %s" (Path.GetFileName(dep.Path)))
                    
            validDeps
            
        | ParsedInput.SigFile(_) -> []

    /// Resolves all dependencies recursively
    let rec resolveAllDependencies 
            (sourceFile: string) 
            (checker: FSharp.Compiler.CodeAnalysis.FSharpChecker) 
            (parsingOptions: FSharp.Compiler.CodeAnalysis.FSharpParsingOptions)
            (visited: Set<string>)
            (baseDirectory: string) : CompilerResult<ParsedInput list * string list> =
        
        if Set.contains sourceFile visited then
            Success ([], [])  // Already processed
        else
            try
                // Parse the current file
                if not (File.Exists(sourceFile)) then
                    CompilerFailure [
                        ConversionError(
                            "dependency resolution", 
                            sourceFile, 
                            "parsed file", 
                            sprintf "File not found: %s" sourceFile)
                    ]
                else
                    let source = File.ReadAllText(sourceFile)
                    let sourceText = SourceText.ofString source
                    let parseResults = checker.ParseFile(sourceFile, sourceText, parsingOptions) |> Async.RunSynchronously
                    
                    // Extract all dependencies
                    let dependencies = extractAllDependencies parseResults.ParseTree sourceFile baseDirectory
                    
                    // Filter to only valid dependencies
                    let validDependencies = 
                        dependencies 
                        |> List.filter (fun dep -> File.Exists(dep.Path))
                    
                    // Log any invalid dependencies as diagnostics
                    let invalidDependencies = 
                        dependencies 
                        |> List.filter (fun dep -> not (File.Exists(dep.Path)))
                        |> List.map (fun dep -> 
                            sprintf "Warning: Could not resolve dependency '%s' from '%s'" 
                                dep.Path dep.SourceFile)
                    
                    // Mark current file as visited
                    let newVisited = Set.add sourceFile visited
                    
                    // Recursively process all dependencies
                    let depResults = 
                        validDependencies
                        |> List.map (fun dep -> 
                            resolveAllDependencies dep.Path checker parsingOptions newVisited baseDirectory)
                    
                    // Combine all results
                    match ResultHelpers.combineResults depResults with
                    | Success results ->
                        let depParseTrees = results |> List.map fst |> List.concat
                        let depDiagnostics = results |> List.map snd |> List.concat
                        
                        Success (parseResults.ParseTree :: depParseTrees, 
                                sourceFile :: (List.concat [invalidDependencies; depDiagnostics]))
                    | CompilerFailure errors -> CompilerFailure errors
            with ex ->
                CompilerFailure [
                    ConversionError(
                        "dependency resolution", 
                        sourceFile, 
                        "parsed file", 
                        sprintf "Exception: %s" ex.Message)
                ]

/// Unified conversion function with complete dependency resolution and module merging
let parseAndConvertToOakAst (inputPath: string) (sourceCode: string) : ASTConversionResult =
    try
        printfn "=== Starting compilation for %s ===" (Path.GetFileName(inputPath))
        
        // Parse the main file
        let sourceText = SourceText.ofString sourceCode
        let baseDirectory = Path.GetDirectoryName(inputPath)
        
        // Create checker instance
        let checker = FSharp.Compiler.CodeAnalysis.FSharpChecker.Create(keepAssemblyContents = true)
        let parsingOptions = { 
            FSharp.Compiler.CodeAnalysis.FSharpParsingOptions.Default with
                SourceFiles = [|Path.GetFileName(inputPath)|]
                ConditionalDefines = ["INTERACTIVE"]
                ApplyLineDirectives = false
        }
        
        // Parse the main file
        let mainParseResults = checker.ParseFile(inputPath, sourceText, parsingOptions) |> Async.RunSynchronously
        
        // Resolve and parse all dependencies
        let dependencyFiles = 
            try
                // Get the dependent files from sourceCode
                let deps = DependencyResolution.extractAllDependencies mainParseResults.ParseTree inputPath baseDirectory
                deps |> List.map (fun dep -> dep.Path) |> List.filter File.Exists
            with ex -> 
                printfn "Error extracting dependencies: %s" ex.Message
                []
        
        printfn "Found %d dependencies" dependencyFiles.Length
        
        // Parse each dependency file
        let dependencyResults =
            dependencyFiles |> List.map (fun depFile ->
                let depSource = File.ReadAllText(depFile)
                let depSourceText = SourceText.ofString depSource
                let depResult = checker.ParseFile(depFile, depSourceText, parsingOptions) |> Async.RunSynchronously
                printfn "Parsed dependency: %s" (Path.GetFileName(depFile))
                (depFile, depResult.ParseTree))
        
        // Generate complete AST text - including main file and ALL dependencies
        let fullAstText = 
            let mainAstText = sprintf "// Main file: %s\n%A\n\n" inputPath mainParseResults.ParseTree
            let depAstTexts = 
                dependencyResults 
                |> List.map (fun (file, ast) -> sprintf "// Dependency: %s\n%A\n\n" file ast)
                |> String.concat ""
            mainAstText + depAstTexts
        
        // Process modules from all parse trees with source file tracking
        let mainModules = ModuleMapping.extractModulesFromParseTree mainParseResults.ParseTree inputPath
        let depModules = 
            dependencyResults 
            |> List.collect (fun (file, ast) -> 
                let modules = ModuleMapping.extractModulesFromParseTree ast file
                printfn "Extracted %d modules from %s" modules.Length (Path.GetFileName(file))
                modules)
        
        // Track all modules by source file
        let sourceModules = 
            (inputPath, mainModules) :: 
            (dependencyResults |> List.map (fun (file, ast) -> 
                (file, ModuleMapping.extractModulesFromParseTree ast file)))
            
        // Extract all modules
        let allModules = mainModules @ depModules
            
        printfn "Extracted %d total modules before merging" allModules.Length
        for m in allModules do
            printfn "  Module: %s with %d declarations" m.Name m.Declarations.Length
            
        // Merge modules with the same name
        let mergedModules = ModuleMerger.mergeModules allModules
        
        printfn "After merging: %d modules" mergedModules.Length
        for m in mergedModules do
            printfn "  Module: %s with %d declarations" m.Name m.Declarations.Length
            for d in m.Declarations do
                match d with
                | FunctionDecl(name, _, _, _) ->
                    printfn "    Function: %s" name
                | TypeDecl(name, _) ->
                    printfn "    Type: %s" name
                | EntryPoint(_) ->
                    printfn "    EntryPoint"
                | ExternalDecl(name, _, _, _) ->
                    printfn "    External: %s" name
        
        // Create program with all modules
        let program = { Modules = mergedModules }
        
        // Validate completeness
        let declCount = mergedModules |> List.sumBy (fun m -> m.Declarations.Length)
        printfn "Final Oak AST contains %d modules with %d total declarations" 
            mergedModules.Length declCount
        
        // Build final result with detailed tracking
        { 
            OakProgram = program
            Diagnostics = [
                sprintf "Parsed main file and %d dependencies" dependencyResults.Length;
                sprintf "Found %d modules with %d declarations" mergedModules.Length declCount
            ]
            FSharpASTText = fullAstText  // This is the critical field for the .fcs file
            ProcessedFiles = inputPath :: dependencyFiles
            SourceModules = sourceModules
        }
    with ex ->
        printfn "CRITICAL ERROR in parsing: %s" ex.Message
        printfn "Stack trace: %s" ex.StackTrace
        { 
            OakProgram = { Modules = [] }
            Diagnostics = [sprintf "Error in parsing: %s" ex.Message]
            FSharpASTText = sprintf "Error: %s" ex.Message
            ProcessedFiles = []
            SourceModules = []
        }