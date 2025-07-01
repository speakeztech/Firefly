module Core.Utilities.IntermediateWriter

open System
open System.IO
open System.Text.Json
open FSharp.Compiler.Syntax

/// Options for JSON serialization
let private jsonOptions = 
    let opts = JsonSerializerOptions()
    opts.WriteIndented <- true
    opts

/// Ensure directory exists and is clean
let prepareIntermediatesDirectory (intermediatesDir: string option) =
    match intermediatesDir with
    | Some dir ->
        if Directory.Exists(dir) then
            printfn "  Clearing intermediates directory..."
            Directory.Delete(dir, true)
        Directory.CreateDirectory(dir) |> ignore
        printfn "  Intermediates directory cleared"
    | None -> ()

/// Write a file to the intermediates directory
let writeFile (directory: string) (baseName: string) (extension: string) (content: string) =
    let filePath = Path.Combine(directory, baseName + extension)
    File.WriteAllText(filePath, content)
    printfn "  Wrote %s (%d bytes)" (Path.GetFileName(filePath)) content.Length

/// Write AST to file with formatted output
let writeAST (directory: string) (baseName: string) (ast: ParsedInput) =
    let content = sprintf "%A" ast
    writeFile directory baseName ".ast" content

/// Write AST metadata as JSON
let writeASTMetadata (directory: string) (baseName: string) (metadata: obj) =
    let json = JsonSerializer.Serialize(metadata, jsonOptions)
    writeFile directory baseName ".ast.json" json

/// Write symbol information as JSON
let writeSymbols (directory: string) (baseName: string) (symbols: obj) =
    let json = JsonSerializer.Serialize(symbols, jsonOptions)
    writeFile directory baseName ".symbols.json" json

/// Write compilation state summary
let writeCompilationSummary (directory: string) (summary: obj) =
    let json = JsonSerializer.Serialize(summary, jsonOptions)
    writeFile directory "compilation_state" ".json" json

/// Write compilation order information
let writeCompilationOrder (directory: string) (files: string list) =
    let order = {| Files = files; Count = files.Length |}
    let json = JsonSerializer.Serialize(order, jsonOptions)
    writeFile directory "compilation_order" ".json" json

/// Structure for organizing intermediate files
type IntermediateFileStructure = {
    RootDir: string
    FCSDir: string
    InitialDir: string
    CompilationUnitsDir: string
    SymbolTablesDir: string
    PrunedDir: string
    SummaryDir: string
}

/// Create the standard intermediate file structure
let createIntermediateStructure (rootDir: string) =
    let structure = {
        RootDir = rootDir
        FCSDir = Path.Combine(rootDir, "fcs")
        InitialDir = Path.Combine(rootDir, "fcs", "initial")
        CompilationUnitsDir = Path.Combine(rootDir, "fcs", "initial", "compilation_units")
        SymbolTablesDir = Path.Combine(rootDir, "fcs", "initial", "symbol_tables")
        PrunedDir = Path.Combine(rootDir, "fcs", "pruned")
        SummaryDir = Path.Combine(rootDir, "fcs", "summary")
    }
    
    // Create all directories
    Directory.CreateDirectory(structure.CompilationUnitsDir) |> ignore
    Directory.CreateDirectory(structure.SymbolTablesDir) |> ignore
    Directory.CreateDirectory(structure.PrunedDir) |> ignore
    Directory.CreateDirectory(structure.SummaryDir) |> ignore
    
    structure

/// Write initial parsing results
let writeParsingResults (structure: IntermediateFileStructure) (order: int) (fileName: string) (ast: ParsedInput) (diagnostics: FSharp.Compiler.Diagnostics.FSharpDiagnostic array) =
    let baseName = sprintf "%02d_%s" order (Path.GetFileName(fileName))
    
    // Write AST
    writeAST structure.CompilationUnitsDir baseName ast
    
    // Write metadata
    let metadata = {|
        FileName = fileName
        Order = order
        HasErrors = diagnostics.Length > 0
        DiagnosticsCount = diagnostics.Length
        Diagnostics = diagnostics |> Array.map (fun d -> 
            {| Message = d.Message; Line = d.StartLine; Column = d.StartColumn |})
    |}
    writeASTMetadata structure.CompilationUnitsDir baseName metadata

/// Write symbol extraction results
let writeSymbolExtractionResults (structure: IntermediateFileStructure) (fileName: string) (symbols: obj) (cumulativeCount: int) =
    let baseName = Path.GetFileNameWithoutExtension(fileName)
    writeSymbols structure.CompilationUnitsDir baseName symbols
    
    // Write cumulative state
    let stateFile = sprintf "after_%s" baseName
    let state = {| TotalSymbols = cumulativeCount; File = fileName |}
    let json = JsonSerializer.Serialize(state, jsonOptions)
    writeFile structure.SymbolTablesDir stateFile ".json" json

/// Write pruned ASTs
let writePrunedAST (structure: IntermediateFileStructure) (order: int) (fileName: string) (ast: ParsedInput) =
    let baseName = sprintf "%02d_%s" order (Path.GetFileName(fileName))
    writeFile structure.PrunedDir baseName ".pruned.ast" (sprintf "%A" ast)

/// Write intermediate ASTs from analysis results
let writeIntermediateASTs (intermediatesDir: string) (prunedAsts: Map<string, ParsedInput>) =
    let prunedDir = Path.Combine(intermediatesDir, "fcs", "pruned", "compilation_units")
    Directory.CreateDirectory(prunedDir) |> ignore
    
    prunedAsts 
    |> Map.toList
    |> List.iteri (fun i (filePath, ast) ->
        let baseName = sprintf "%02d_%s" i (Path.GetFileName(filePath))
        writeFile prunedDir baseName ".pruned.ast" (sprintf "%A" ast))

/// Write MLIR output
let writeMLIR (intermediatesDir: string) (baseName: string) (mlirContent: string) =
    writeFile intermediatesDir baseName ".mlir" mlirContent

/// Write LLVM IR output
let writeLLVMIR (intermediatesDir: string) (baseName: string) (llvmContent: string) =
    writeFile intermediatesDir baseName ".ll" llvmContent

/// Create a compilation summary report
let writeCompilationReport (structure: IntermediateFileStructure) (statistics: obj) =
    let json = JsonSerializer.Serialize(statistics, jsonOptions)
    writeFile structure.SummaryDir "compilation_report" ".json" json