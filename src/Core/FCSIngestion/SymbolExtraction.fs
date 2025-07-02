module Core.FCSIngestion.SymbolExtraction

open FSharp.Compiler.Syntax
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Symbols

/// Basic symbol information extracted from AST
type ExtractedSymbol = {
    /// Fully qualified name (e.g., "Alloy.Memory.stackBuffer")
    QualifiedName: string
    /// Short name without qualification (e.g., "stackBuffer")
    ShortName: string
    /// Module path as list (e.g., ["Alloy"; "Memory"])
    ModulePath: string list
    /// Symbol kind
    Kind: SymbolKind
}

and SymbolKind =
    | Value
    | Function
    | Type
    | Module
    | Member of typeName: string

/// Combine module path with symbol name to create qualified name
let private qualifyName (modulePath: string list) (name: string) =
    if List.isEmpty modulePath then name
    else (modulePath @ [name]) |> String.concat "."

/// Extract symbol from a pattern
let rec extractSymbolFromPattern (modulePath: string list) (pat: SynPat) : ExtractedSymbol option =
    match pat with
    | SynPat.Named(SynIdent(ident, _), _, _, _) ->
        Some {
            QualifiedName = qualifyName modulePath ident.idText
            ShortName = ident.idText
            ModulePath = modulePath
            Kind = Value
        }
    | SynPat.LongIdent(longIdent, _, _, _, _, _) ->
        let (SynLongIdent(ids, _, _)) = longIdent
        let fullName = ids |> List.map (fun id -> id.idText) |> String.concat "."
        Some {
            QualifiedName = qualifyName modulePath fullName
            ShortName = 
                match List.tryLast ids with
                | Some id -> id.idText
                | None -> fullName
            ModulePath = modulePath
            Kind = Value
        }
    | SynPat.Paren(pat, _) ->
        extractSymbolFromPattern modulePath pat
    | SynPat.Typed(pat, _, _) ->
        extractSymbolFromPattern modulePath pat
    | _ -> None

/// Extract member name from pattern
let extractMemberName (pat: SynPat) : string option =
    match pat with
    | SynPat.LongIdent(SynLongIdent([_; memberName], _, _), _, _, _, _, _) ->
        Some memberName.idText
    | _ -> None

/// Extract all symbols from a module declaration
let rec extractSymbolsFromDecl (modulePath: string list) (decl: SynModuleDecl) : ExtractedSymbol list =
    match decl with
    | SynModuleDecl.Let(_, bindings, _) ->
        bindings |> List.choose (fun (SynBinding(_, _, _, _, _, _, _, pat, _, _, _, _, _)) ->
            extractSymbolFromPattern modulePath pat)
    
    | SynModuleDecl.Types(types, _) ->
        types |> List.collect (fun (SynTypeDefn(SynComponentInfo(_, _, _, longId, _, _, _, _), _, memberDefns, _, _, _)) ->
            let typeName = longId.Head.idText
            let typeQualifiedName = qualifyName modulePath typeName
            
            // The type itself
            let typeSymbol = {
                QualifiedName = typeQualifiedName
                ShortName = typeName
                ModulePath = modulePath
                Kind = Type
            }
            
            // Members of the type
            let memberSymbols = 
                memberDefns |> List.choose (fun memberDefn ->
                    match memberDefn with
                    | SynMemberDefn.Member(SynBinding(_, _, _, _, _, _, _, pat, _, _, _, _, _), _) ->
                        match extractMemberName pat with
                        | Some memberName -> 
                            Some {
                                QualifiedName = typeQualifiedName + "." + memberName
                                ShortName = memberName
                                ModulePath = modulePath @ [typeName]
                                Kind = Member typeName
                            }
                        | None -> None
                    | _ -> None)
            
            typeSymbol :: memberSymbols)
    
    | SynModuleDecl.NestedModule(SynComponentInfo(_, _, _, longId, _, _, _, _), _, decls, _, _, _) ->
        let nestedModuleName = longId.Head.idText
        let nestedPath = modulePath @ [nestedModuleName]
        
        // The module itself
        let moduleSymbol = {
            QualifiedName = qualifyName modulePath nestedModuleName
            ShortName = nestedModuleName
            ModulePath = modulePath
            Kind = Module
        }
        
        // Symbols within the module
        let nestedSymbols = decls |> List.collect (extractSymbolsFromDecl nestedPath)
        
        moduleSymbol :: nestedSymbols
        
    | _ -> []

/// Extract all symbols from a parsed input file
let extractSymbolsFromParsedInput (ast: ParsedInput) : ExtractedSymbol list =
    match ast with
    | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
        modules 
        |> List.collect (fun (SynModuleOrNamespace(longId, _, _, decls, _, _, _, _, _)) ->
            let modulePath = longId |> List.map (fun id -> id.idText)
            decls |> List.collect (extractSymbolsFromDecl modulePath))
    | ParsedInput.SigFile(ParsedSigFileInput(_, _, _, _, modules, _, _)) ->
        modules
        |> List.collect (fun (SynModuleOrNamespaceSig(longId, _, _, decls, _, _, _, _, _)) ->
            let modulePath = longId |> List.map (fun id -> id.idText)
            // TODO: Extract from signature declarations
            [])

/// Extract module path from parsed input
let extractModulePath (input: ParsedInput) : string list =
    match input with
    | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
        match modules with
        | SynModuleOrNamespace(longId, _, _, _, _, _, _, _, _) :: _ ->
            longId |> List.map (fun id -> id.idText)
        | [] -> []
    | ParsedInput.SigFile(ParsedSigFileInput(_, _, _, _, modules, _, _)) ->
        match modules with
        | SynModuleOrNamespaceSig(longId, _, _, _, _, _, _, _, _) :: _ ->
            longId |> List.map (fun id -> id.idText)
        | [] -> []

/// Group symbols by module for efficient lookup
let groupSymbolsByModule (symbols: ExtractedSymbol list) : Map<string, Set<string>> =
    symbols
    |> List.groupBy (fun s -> 
        if List.isEmpty s.ModulePath then "" 
        else String.concat "." s.ModulePath)
    |> List.map (fun (moduleName, syms) ->
        (moduleName, syms |> List.map (fun s -> s.ShortName) |> Set.ofList))
    |> Map.ofList

/// Build a symbol to file mapping
let buildSymbolToFileMap (filesAndSymbols: (string * ExtractedSymbol list) list) : Map<string, string> =
    filesAndSymbols
    |> List.collect (fun (filePath, symbols) ->
        symbols |> List.map (fun symbol -> (symbol.QualifiedName, filePath)))
    |> Map.ofList

// For compatibility with check results (when we have FSharpCheckFileResults)
/// Extract symbols from type-checked results
let extractFromCheckResults (checkResults: FSharpCheckFileResults) (fileName: string) =
    let assemblySig = checkResults.PartialAssemblySignature
    let shortName = System.IO.Path.GetFileName(fileName)
    
    {|
        FileName = fileName
        Modules = 
            assemblySig.Entities 
            |> Seq.filter (fun e -> e.IsFSharpModule && e.DeclarationLocation.FileName.EndsWith(shortName))
            |> Seq.map (fun e -> e.FullName)
            |> Seq.toArray
        Types = 
            assemblySig.Entities
            |> Seq.filter (fun e -> not e.IsFSharpModule && e.DeclarationLocation.FileName.EndsWith(shortName))
            |> Seq.map (fun e -> {| Name = e.FullName; Kind = e.DisplayName |})
            |> Seq.toArray
        Functions = 
            assemblySig.Entities
            |> Seq.collect (fun e -> e.MembersFunctionsAndValues)
            |> Seq.filter (fun f -> f.DeclarationLocation.FileName.EndsWith(shortName))
            |> Seq.map (fun f -> {| Name = f.FullName; Signature = f.FullType.Format(FSharpDisplayContext.Empty) |})
            |> Seq.toArray
    |}