module Alex.Pipeline.FCSPipeline

open FSharp.Compiler.Syntax
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Symbols
open Core.Types.MLIRTypes 
open Alex.CodeGeneration.TypeMapping
open Alex.Bindings.SymbolRegistry
open Alex.Bindings.PatternLibrary
open Alex.Analysis.ReachabilityAnalyzer
open Alex.Transformations.ClosureElimination
open Alex.Transformations.StackAllocation.Application

/// Convert F# type to MLIR type with full fidelity
let rec private convertFSharpTypeToMLIR (fsharpType: FSharpType) (typeCtx: TypeContext) : MLIRType =
    if fsharpType.IsAbbreviation then
        convertFSharpTypeToMLIR fsharpType.AbbreviatedType typeCtx
    elif fsharpType.IsFunctionType then
        let domainType = convertFSharpTypeToMLIR fsharpType.GenericArguments.[0] typeCtx
        let rangeType = convertFSharpTypeToMLIR fsharpType.GenericArguments.[1] typeCtx
        MLIRTypes.func [domainType] rangeType
    elif fsharpType.IsTupleType then
        let elementTypes = 
            fsharpType.GenericArguments 
            |> Seq.map (fun t -> convertFSharpTypeToMLIR t typeCtx)
            |> Seq.toList
        MLIRTypes.tuple elementTypes
    elif fsharpType.IsGenericParameter then
        // For generic parameters, use opaque type placeholder
        MLIRTypes.opaque (sprintf "type_param_%s" fsharpType.GenericParameter.Name)
    else
        match fsharpType.TypeDefinition.TryFullName with
        | Some "System.Int32" -> MLIRTypes.i32
        | Some "System.Int64" -> MLIRTypes.i64
        | Some "System.Single" -> MLIRTypes.f32
        | Some "System.Double" -> MLIRTypes.f64
        | Some "System.Boolean" -> MLIRTypes.i1
        | Some "System.Byte" -> MLIRTypes.i8
        | Some "System.String" -> MLIRTypes.memref MLIRTypes.i8
        | Some "System.Void" | Some "Microsoft.FSharp.Core.unit" -> MLIRTypes.void_
        | Some typeName when typeName.StartsWith("Microsoft.FSharp.Core.FSharpOption") ->
            let innerType = convertFSharpTypeToMLIR fsharpType.GenericArguments.[0] typeCtx
            MLIRTypes.variant [("Some", innerType); ("None", MLIRTypes.void_)]
        | Some typeName when typeName.StartsWith("Microsoft.FSharp.Collections.FSharpList") ->
            let elementType = convertFSharpTypeToMLIR fsharpType.GenericArguments.[0] typeCtx
            MLIRTypes.memref elementType
        | Some typeName when fsharpType.HasTypeDefinition && fsharpType.TypeDefinition.IsArrayType ->
            let elementType = convertFSharpTypeToMLIR fsharpType.GenericArguments.[0] typeCtx
            MLIRTypes.memref elementType
        | Some typeName ->
            // For user-defined types, check if we have a registered mapping
            match TypeContextOps.tryResolveType typeCtx typeName with
            | Some mlirType -> mlirType
            | None ->
                // Build struct type from record/class fields
                if fsharpType.TypeDefinition.IsFSharpRecord then
                    let fields = 
                        fsharpType.TypeDefinition.FSharpFields
                        |> Seq.map (fun field -> 
                            (field.Name, convertFSharpTypeToMLIR field.FieldType typeCtx))
                        |> Seq.toList
                    MLIRTypes.struct_ fields
                else
                    // Default to opaque type for unhandled cases
                    MLIRTypes.opaque typeName
        | None -> MLIRTypes.opaque "unknown"

/// Extract complete type information from a member or value
let private extractMemberTypeInfo (symbol: FSharpMemberOrFunctionOrValue) (typeCtx: TypeContext) =
    let parameterTypes = 
        symbol.CurriedParameterGroups
        |> Seq.collect (fun group -> 
            group |> Seq.map (fun param -> convertFSharpTypeToMLIR param.Type typeCtx))
        |> Seq.toList
    
    let returnType = convertFSharpTypeToMLIR symbol.ReturnParameter.Type typeCtx
    
    (parameterTypes, returnType)

/// Process type information from FSharpCheckFileResults
let extractTypeInformation (checkResults: FSharpCheckFileResults) (typeCtx: TypeContext) =
    let mutable updatedContext = typeCtx
    
    // Process all symbol uses to build type mappings
    checkResults.GetAllUsesOfAllSymbolsInFile()
    |> Seq.toArray
    |> Array.iter (fun symbolUse ->
        match symbolUse.Symbol with
        | :? FSharpMemberOrFunctionOrValue as memberSymbol when memberSymbol.IsModuleValueOrMember ->
            let (paramTypes, returnType) = extractMemberTypeInfo memberSymbol updatedContext
            let fullName = 
                match memberSymbol.DeclaringEntity with
                | Some entity -> sprintf "%s.%s" entity.FullName memberSymbol.DisplayName
                | None -> memberSymbol.FullName
            
            // Register the type mapping
            updatedContext <- TypeContextOps.registerFunction updatedContext fullName paramTypes returnType
            
        | :? FSharpEntity as entity when entity.IsFSharpRecord ->
            // Process record type
            let fields = 
                entity.FSharpFields
                |> Seq.map (fun field -> 
                    (field.Name, convertFSharpTypeToMLIR field.FieldType updatedContext))
                |> Seq.toList
            let structType = MLIRTypes.struct_ fields
            updatedContext <- TypeContextOps.registerType updatedContext entity.FullName structType
            
        | :? FSharpEntity as entity when entity.IsFSharpUnion ->
            // Process discriminated union
            let cases = 
                entity.UnionCases
                |> Seq.map (fun case ->
                    let fieldTypes = 
                        case.Fields  // Changed from UnionCaseFields to Fields
                        |> Seq.map (fun field -> convertFSharpTypeToMLIR field.FieldType updatedContext)
                        |> Seq.toList
                    let caseType = 
                        match fieldTypes with
                        | [] -> MLIRTypes.void_
                        | [single] -> single
                        | multiple -> MLIRTypes.tuple multiple
                    (case.Name, caseType))
                |> Seq.toList
            let variantType = MLIRTypes.variant cases
            updatedContext <- TypeContextOps.registerType updatedContext entity.FullName variantType
            
        | :? FSharpEntity as entity when entity.IsClass || entity.IsInterface ->
            // For classes and interfaces, register as opaque for now
            let opaqueType = MLIRTypes.opaque entity.FullName
            updatedContext <- TypeContextOps.registerType updatedContext entity.FullName opaqueType
            
        | _ -> ())
    
    // Process type definitions from declarations
    match checkResults.ImplementationFile with
    | Some implFile ->
        implFile.Declarations
        |> List.iter (fun decl ->
            match decl with
            | FSharpImplementationFileDeclaration.Entity(entity, subDecls) ->
                if entity.IsFSharpModule then
                    // Process module members
                    subDecls |> List.iter (fun subDecl ->
                        match subDecl with
                        | FSharpImplementationFileDeclaration.MemberOrFunctionOrValue(memberVal, _, _) ->
                            let (paramTypes, returnType) = extractMemberTypeInfo memberVal updatedContext
                            let fullName = sprintf "%s.%s" entity.FullName memberVal.DisplayName
                            updatedContext <- TypeContextOps.registerFunction updatedContext fullName paramTypes returnType
                        | _ -> ())
                        
            | FSharpImplementationFileDeclaration.MemberOrFunctionOrValue(memberVal, _, _) ->
                let (paramTypes, returnType) = extractMemberTypeInfo memberVal updatedContext
                updatedContext <- TypeContextOps.registerFunction updatedContext memberVal.FullName paramTypes returnType
                
            | _ -> ())
    | None -> ()
    
    updatedContext

/// Convert extracted symbol to resolved symbol with type information
let private toResolvedSymbol (extracted: Core.FCSIngestion.SymbolExtraction.ExtractedSymbol) : ResolvedSymbol =
    // Try to find a pattern in our library
    match findByName extracted.QualifiedName with
    | Some pattern ->
        {
            QualifiedName = extracted.QualifiedName
            ShortName = extracted.ShortName
            ParameterTypes = fst pattern.TypeSig
            ReturnType = snd pattern.TypeSig
            Operation = pattern.OpPattern
            Namespace = 
                if List.isEmpty extracted.ModulePath then "Global" 
                else String.concat "." extracted.ModulePath
            SourceLibrary = "alloy"
            RequiresExternal = 
                match pattern.OpPattern with
                | ExternalCall(_, Some _) -> true
                | _ -> false
        }
    | None ->
        // Create a default symbol
        {
            QualifiedName = extracted.QualifiedName
            ShortName = extracted.ShortName
            ParameterTypes = []
            ReturnType = MLIRTypes.i32  // Default type
            Operation = DirectCall extracted.QualifiedName
            Namespace = 
                if List.isEmpty extracted.ModulePath then "Global" 
                else String.concat "." extracted.ModulePath
            SourceLibrary = "user"
            RequiresExternal = false
        }

/// Extract symbols with type information from check results
let extractTypedSymbols (checkResults: FSharpCheckFileResults) (ast: ParsedInput) (typeCtx: TypeContext) : ResolvedSymbol list =
    // Start with unified extraction
    let baseSymbols = 
        Core.FCSIngestion.SymbolExtraction.extractSymbolsFromParsedInput ast
        |> List.map toResolvedSymbol
    
    // Enhance symbols with type information from check results
    let symbolMap = 
        checkResults.GetAllUsesOfAllSymbolsInFile()
        |> Seq.toArray
        |> Array.choose (fun symbolUse ->
            match symbolUse.Symbol with
            | :? FSharpMemberOrFunctionOrValue as memberSymbol when memberSymbol.IsModuleValueOrMember ->
                let (paramTypes, returnType) = extractMemberTypeInfo memberSymbol typeCtx
                let fullName = 
                    match memberSymbol.DeclaringEntity with
                    | Some entity -> sprintf "%s.%s" entity.FullName memberSymbol.DisplayName
                    | None -> memberSymbol.FullName
                Some (fullName, (paramTypes, returnType))
            | _ -> None)
        |> Map.ofArray
    
    // Merge type information with base symbols
    baseSymbols
    |> List.map (fun symbol ->
        match Map.tryFind symbol.QualifiedName symbolMap with
        | Some (paramTypes, returnType) ->
            { symbol with 
                ParameterTypes = paramTypes
                ReturnType = returnType }
        | None -> symbol)

/// Apply AST transformations (closure elimination, stack allocation, etc.)
let applyTransformations (ast: ParsedInput) (typeCtx: TypeContext) (reachableSymbols: Set<string>) =
    // Apply closure elimination if there are bindings
    let afterClosure = 
        match ast with
        | ParsedInput.ImplFile(ParsedImplFileInput(fileName, isScript, qualName, pragmas, hashDirectives, modules, isLastCompiled, isExe, _)) ->
            let transformedModules = 
                modules |> List.map (fun m ->
                    match m with
                    | SynModuleOrNamespace(lid, isRec, kind, decls, xml, attrs, access, range, trivia) ->
                        // Extract and transform bindings
                        let transformedDecls = 
                            decls |> List.map (function
                                | SynModuleDecl.Let(isRec, bindings, range) ->
                                    if List.isEmpty bindings then 
                                        SynModuleDecl.Let(isRec, bindings, range)
                                    else
                                        let transformedBindings = transformModule bindings
                                        SynModuleDecl.Let(isRec, transformedBindings, range)
                                | decl -> decl)
                        SynModuleOrNamespace(lid, isRec, kind, transformedDecls, xml, attrs, access, range, trivia))
            ParsedInput.ImplFile(ParsedImplFileInput(fileName, isScript, qualName, pragmas, hashDirectives, transformedModules, isLastCompiled, isExe, Set.empty))
        | sig_ -> sig_
    
    // Apply stack allocation transformation
    let afterStack = applyStackAllocation afterClosure typeCtx reachableSymbols
    
    afterStack