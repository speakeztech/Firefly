module Core.FCSIngestion.AstMerger

open FSharp.Compiler.Syntax

/// Result of merging multiple ASTs - uses only primitive types
type MergedASTCollection = {
    /// The main file path
    MainFile: string
    
    /// All ASTs indexed by file path
    FileASTs: Map<string, ParsedInput>
    
    /// Order of files for processing
    FileOrder: string list
    
    /// Module dependencies extracted from ASTs (file -> modules it imports)
    ModuleImports: Map<string, Set<string>>
}

/// Extract module imports from an AST
let extractModuleImports (ast: ParsedInput) : Set<string> =
    match ast with
    | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
        modules 
        |> List.collect (fun (SynModuleOrNamespace(_, _, _, decls, _, _, _, _, _)) ->
            decls |> List.choose (fun decl ->
                match decl with
                | SynModuleDecl.Open(target, _) ->
                    match target with
                    | SynOpenDeclTarget.ModuleOrNamespace(SynLongIdent(ids, _, _), _) ->
                        Some (String.concat "." (ids |> List.map (fun id -> id.idText)))
                    | _ -> None
                | _ -> None))
        |> Set.ofList
    | _ -> Set.empty

/// Merge multiple ASTs into a unified collection structure
let mergeCompilationUnit (astList: (string * ParsedInput) list) (mainFile: string) : MergedASTCollection =
    // Build the file map
    let fileAsts = astList |> Map.ofList
    
    // Extract the file order
    let fileOrder = astList |> List.map fst
    
    // Extract module imports from each file
    let moduleImports = 
        astList
        |> List.map (fun (filePath, ast) ->
            (filePath, extractModuleImports ast))
        |> Map.ofList
    
    {
        MainFile = mainFile
        FileASTs = fileAsts
        FileOrder = fileOrder
        ModuleImports = moduleImports
    }

/// Count the number of top-level declarations in an AST
let countTopLevelDeclarations (ast: ParsedInput) : int =
    match ast with
    | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
        modules 
        |> List.sumBy (fun (SynModuleOrNamespace(_, _, _, decls, _, _, _, _, _)) ->
            decls.Length)
    | ParsedInput.SigFile(ParsedSigFileInput(_, _, _, _, modules, _, _)) ->
        modules 
        |> List.sumBy (fun (SynModuleOrNamespaceSig(_, _, _, decls, _, _, _, _, _)) ->
            decls.Length)

/// Check if an AST represents a script file based on its structure
let isScriptAST (ast: ParsedInput) : bool =
    match ast with
    | ParsedInput.ImplFile(ParsedImplFileInput(_, isScript, _, _, _, _, _, _, _)) -> isScript
    | _ -> false