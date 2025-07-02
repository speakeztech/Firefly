module Dabbit.Analysis.CompilationUnit

open System.IO
open FSharp.Compiler.Syntax
open Core.XParsec.Foundation
open Core.Utilities.IntermediateWriter
open ReachabilityAnalyzer
open DependencyGraphBuilder
open AstPruner

/// Parser function type - accepts file path and returns parsed AST
type FileParser = string -> Async<CompilerResult<ParsedInput>>

/// Represents a single source file in the compilation unit
type SourceFile = {
    /// Absolute path to the file
    Path: string
    /// Parsed AST
    Ast: ParsedInput
    /// Files this file loads via #load
    LoadedFiles: string list
    /// Module path extracted from AST
    ModulePath: string list
    /// Symbols defined in this file
    DefinedSymbols: Set<string>
}

/// Compilation unit representing all files in the program
type CompilationUnit = {
    /// Main entry file
    MainFile: string
    /// All source files by path
    SourceFiles: Map<string, SourceFile>
    /// Global symbol to file mapping
    SymbolToFile: Map<string, string>
    /// File dependency graph (who loads who)
    FileDependencies: Map<string, Set<string>>
}

/// Result of compilation unit analysis
type CompilationUnitAnalysis = {
    /// The compilation unit
    Unit: CompilationUnit
    /// Global reachability result
    GlobalReachability: ReachabilityResult
    /// Per-file reachable symbols
    PerFileReachable: Map<string, Set<string>>
    /// Pruned ASTs for each file
    PrunedAsts: Map<string, ParsedInput>
}

/// Extract #load directives from an AST
let private extractLoadDirectives (ast: ParsedInput) : string list =
    match ast with
    | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, hashDirectives, _, _, _, _)) ->
        hashDirectives 
        |> List.choose (function
            | ParsedHashDirective("load", args, _) ->
                match args with
                | [ParsedHashDirectiveArgument.String (file, _, _)] -> Some file
                | _ -> None
            | _ -> None)
    | _ -> []

/// Extract module path from AST
let private extractModulePath (ast: ParsedInput) : string list =
    match ast with
    | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
        match modules with
        | (SynModuleOrNamespace(longId, _, _, _, _, _, _, _, _)) :: _ ->
            longId |> List.map (fun id -> id.idText)
        | [] -> []
    | _ -> []

/// Extract all symbols defined in a file
let private extractFileSymbols (modulePath: string list) (ast: ParsedInput) : Set<string> =
    match ast with
    | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
        modules 
        |> List.collect (fun (SynModuleOrNamespace(_, _, _, decls, _, _, _, _, _)) ->
            decls |> List.collect (extractSymbolsFromDecl modulePath))
        |> Set.ofList
    | _ -> Set.empty

/// Recursively gather all files in the compilation unit
let rec private gatherSourceFiles (parser: FileParser) (basePath: string) (filePath: string) (gathered: Map<string, SourceFile>) : Async<CompilerResult<Map<string, SourceFile>>> = async {
    let absolutePath = 
        if Path.IsPathRooted(filePath) then filePath
        else Path.Combine(basePath, filePath)
    
    if Map.containsKey absolutePath gathered then
        return Success gathered
    else
        // Parse this file using provided parser
        let! parseResult = parser absolutePath
        match parseResult with
        | CompilerFailure errors -> 
            return CompilerFailure errors
        | Success ast ->
            // Extract information
            let loads = extractLoadDirectives ast
            let absoluteLoads = 
                loads |> List.map (fun f -> 
                    if Path.IsPathRooted(f) then f
                    else Path.Combine(Path.GetDirectoryName(absolutePath), f))
            
            let modulePath = extractModulePath ast
            let symbols = extractFileSymbols modulePath ast
            
            let sourceFile = {
                Path = absolutePath
                Ast = ast
                LoadedFiles = absoluteLoads
                ModulePath = modulePath
                DefinedSymbols = symbols
            }
            
            let gathered' = Map.add absolutePath sourceFile gathered
            
            // Recursively gather loaded files
            let! finalResult = 
                absoluteLoads 
                |> List.fold (fun asyncResult loadPath -> async {
                    let! result = asyncResult
                    match result with
                    | CompilerFailure _ -> return result
                    | Success gathered -> 
                        return! gatherSourceFiles parser basePath loadPath gathered
                }) (async.Return (Success gathered'))
            
            return finalResult
}

/// Build a compilation unit from a main file
let buildCompilationUnit (parser: FileParser) (mainFile: string) (mainAst: ParsedInput) : Async<CompilerResult<CompilationUnit>> = async {
    // Start with the main file
    let mainPath = Path.GetFullPath(mainFile)
    let mainLoads = extractLoadDirectives mainAst
    let mainModulePath = extractModulePath mainAst
    let mainSymbols = extractFileSymbols mainModulePath mainAst
    
    let mainSourceFile = {
        Path = mainPath
        Ast = mainAst
        LoadedFiles = mainLoads |> List.map (fun f ->
            if Path.IsPathRooted(f) then f
            else Path.Combine(Path.GetDirectoryName(mainPath), f))
        ModulePath = mainModulePath
        DefinedSymbols = mainSymbols
    }
    
    // Gather all dependencies
    let initialMap = Map.ofList [(mainPath, mainSourceFile)]
    let! gatherResult = gatherSourceFiles parser (Path.GetDirectoryName(mainPath)) mainPath initialMap
    
    match gatherResult with
    | CompilerFailure errors -> return CompilerFailure errors
    | Success allFiles ->
        // Build symbol to file mapping
        let symbolToFile =
            allFiles 
            |> Map.toList
            |> List.collect (fun (filePath, sourceFile) ->
                sourceFile.DefinedSymbols 
                |> Set.toList
                |> List.map (fun symbol -> (symbol, filePath)))
            |> Map.ofList
        
        // Build file dependency graph
        let fileDeps =
            allFiles
            |> Map.map (fun _ sourceFile -> Set.ofList sourceFile.LoadedFiles)
        
        return Success {
            MainFile = mainPath
            SourceFiles = allFiles
            SymbolToFile = symbolToFile
            FileDependencies = fileDeps
        }
}

/// Analyze a compilation unit
let analyzeCompilationUnit (unit: CompilationUnit) : CompilationUnitAnalysis =
    // Use ReachabilityAnalyzer's multi-file analysis function
    let parsedInputs = 
        unit.SourceFiles 
        |> Map.toList
        |> List.map (fun (path, sourceFile) -> (path, sourceFile.Ast))
    
    // Perform analysis using existing infrastructure
    let reachabilityResult = analyzeFromParsedInputs parsedInputs
    
    // Map reachable symbols back to files
    let perFileReachable =
        unit.SourceFiles
        |> Map.map (fun filePath sourceFile ->
            reachabilityResult.Reachable
            |> Set.filter (fun symbol ->
                match Map.tryFind symbol unit.SymbolToFile with
                | Some file -> file = filePath
                | None -> false))
    
    // Prune each file with its reachable set
    let prunedAsts =
        unit.SourceFiles
        |> Map.map (fun filePath sourceFile ->
            let fileReachable = 
                Map.tryFind filePath perFileReachable 
                |> Option.defaultValue Set.empty
            
            prune fileReachable sourceFile.Ast)
    
    {
        Unit = unit
        GlobalReachability = reachabilityResult
        PerFileReachable = perFileReachable
        PrunedAsts = prunedAsts
    }

/// Analyze compilation unit with dependency loading
let analyzeWithDependencies (parser: FileParser) (mainFile: string) (mainAst: ParsedInput) (writeIntermediates: bool) (intermediatesDir: string option) : Async<CompilerResult<CompilationUnitAnalysis>> = async {
    // Build compilation unit
    let! unitResult = buildCompilationUnit parser mainFile mainAst
    
    match unitResult with
    | CompilerFailure errors -> return CompilerFailure errors
    | Success unit ->
        printfn "Compilation unit built:"
        printfn "  Main file: %s" unit.MainFile
        printfn "  Total files: %d" (Map.count unit.SourceFiles)
        printfn "  Total symbols: %d" (Map.count unit.SymbolToFile)
        
        // Analyze
        let analysis = analyzeCompilationUnit unit
        
        printfn "Analysis complete:"
        printfn "  Global reachable: %d" (Set.count analysis.GlobalReachability.Reachable)
        printfn "  Reduction: %.1f%%" 
            ((float analysis.GlobalReachability.Statistics.EliminatedSymbols / 
              float analysis.GlobalReachability.Statistics.TotalSymbols) * 100.0)
        
        // Write intermediate files if requested
        if writeIntermediates then
            match intermediatesDir with
            | Some dir ->
                for KeyValue(filePath, prunedAst) in analysis.PrunedAsts do
                    let fileName = Path.GetFileNameWithoutExtension(filePath)
                    let outputPath = Path.Combine(dir, fileName + ".ra.fcs")
                    // Use the IntermediateWriter module to write the AST
                    let astText = sprintf "%A" prunedAst
                    writeFile filePath fileName "ra.fcs"  astText
                    printfn "  Wrote: %s" outputPath
            | None ->
                printfn "  No intermediates directory specified"
        
        return Success analysis
}

/// Get the pruned AST for a specific file
let getPrunedAst (analysis: CompilationUnitAnalysis) (filePath: string) : ParsedInput option =
    Map.tryFind filePath analysis.PrunedAsts

/// Get reachability statistics for a specific file
let getFileStats (analysis: CompilationUnitAnalysis) (filePath: string) : (int * int) option =
    match Map.tryFind filePath analysis.Unit.SourceFiles with
    | Some sourceFile ->
        let totalSymbols = Set.count sourceFile.DefinedSymbols
        let reachableSymbols = 
            Map.tryFind filePath analysis.PerFileReachable
            |> Option.map Set.count
            |> Option.defaultValue 0
        Some (totalSymbols, reachableSymbols)
    | None -> None

/// Check if the main file requires multi-file analysis
let requiresMultiFileAnalysis (ast: ParsedInput) : bool =
    not (List.isEmpty (extractLoadDirectives ast))

/// Build compilation unit from pre-parsed ASTs
let buildCompilationUnitFromParsed (parsedUnits: (string * ParsedInput) list) : Async<CompilerResult<CompilationUnit>> = async {
    match parsedUnits with
    | [] -> return CompilerFailure [InternalError("buildCompilationUnitFromParsed", "No parsed units provided", None)]
    | (mainPath, mainAst) :: rest ->
        // Extract information from main file
        let mainLoads = extractLoadDirectives mainAst
        let mainModulePath = extractModulePath mainAst
        let mainSymbols = extractFileSymbols mainModulePath mainAst
        
        let mainSourceFile = {
            Path = mainPath
            Ast = mainAst
            LoadedFiles = mainLoads |> List.map (fun f ->
                if Path.IsPathRooted(f) then f
                else Path.Combine(Path.GetDirectoryName(mainPath), f))
            ModulePath = mainModulePath
            DefinedSymbols = mainSymbols
        }
        
        // Build source files map from all parsed units
        let allSourceFiles = 
            parsedUnits
            |> List.map (fun (path, ast) ->
                let loads = extractLoadDirectives ast
                let modulePath = extractModulePath ast
                let symbols = extractFileSymbols modulePath ast
                
                let sourceFile = {
                    Path = path
                    Ast = ast
                    LoadedFiles = loads |> List.map (fun f ->
                        if Path.IsPathRooted(f) then f
                        else Path.Combine(Path.GetDirectoryName(path), f))
                    ModulePath = modulePath
                    DefinedSymbols = symbols
                }
                (path, sourceFile))
            |> Map.ofList
        
        // Build symbol to file mapping
        let symbolToFile =
            allSourceFiles 
            |> Map.toList
            |> List.collect (fun (filePath, sourceFile) ->
                sourceFile.DefinedSymbols 
                |> Set.toList
                |> List.map (fun symbol -> (symbol, filePath)))
            |> Map.ofList
        
        // Build file dependency graph
        let fileDeps =
            allSourceFiles
            |> Map.map (fun _ sourceFile -> Set.ofList sourceFile.LoadedFiles)
        
        return Success {
            MainFile = mainPath
            SourceFiles = allSourceFiles
            SymbolToFile = symbolToFile
            FileDependencies = fileDeps
        }
}