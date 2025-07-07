module Core.PSG.IntermediateWriter

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization
open FSharp.Compiler.Syntax
open FSharp.Compiler.Text
open FSharp.Compiler.Symbols
open FSharp.Compiler.CodeAnalysis
open Core.PSG.Types
open Core.PSG.XmlDocParser

/// Converter for range to JSON
type RangeJsonConverter() =
    inherit JsonConverter<range>()
    
    override _.Read(reader, typeToConvert, options) =
        failwith "Range deserialization not implemented"
        
    override _.Write(writer, value, options) =
        writer.WriteStartObject()
        writer.WriteString("FileName", value.FileName)
        writer.WriteNumber("StartLine", value.Start.Line)
        writer.WriteNumber("StartColumn", value.Start.Column)
        writer.WriteNumber("EndLine", value.End.Line)
        writer.WriteNumber("EndColumn", value.End.Column)
        writer.WriteEndObject()

/// JSON serialization options
let jsonOptions =
    let options = JsonSerializerOptions()
    options.WriteIndented <- true
    options.Converters.Add(RangeJsonConverter())
    options.Converters.Add(JsonFSharpConverter())
    options

/// Write typed AST to file
let writeTypedAST (fileName: string) (checkResults: FSharpCheckProjectResults) =
    try
        let outputPath = Path.ChangeExtension(fileName, ".typ.ast")
        
        // Create a simplified representation for serialization
        let simplifiedAST =
            checkResults.AssemblyContents.ImplementationFiles
            |> Seq.map (fun implFile ->
                {|
                    FileName = implFile.FileName
                    QualifiedName = implFile.QualifiedName
                    Declarations = implFile.Declarations |> Seq.length
                    HasEntryPoint = 
                        implFile.Declarations
                        |> Seq.exists (fun decl ->
                            match decl with
                            | FSharpImplementationFileDeclaration.MemberOrFunctionOrValue(mfv, _, _) ->
                                mfv.Attributes
                                |> Seq.exists (fun attr -> 
                                    attr.AttributeType.BasicQualifiedName = "EntryPointAttribute" ||
                                    attr.AttributeType.BasicQualifiedName = "System.EntryPointAttribute")
                            | _ -> false)
                |}
            )
            |> Seq.toArray
            
        // Serialize to JSON
        let json = JsonSerializer.Serialize(simplifiedAST, jsonOptions)
        File.WriteAllText(outputPath, json)
        
        printfn "Typed AST written to %s" outputPath
        true
    with ex ->
        printfn "Error writing typed AST: %s" ex.Message
        false

/// Write symbolic AST to file
let writeSymbolicAST (fileName: string) (symbolUses: FSharpSymbolUse[]) =
    try
        let outputPath = Path.ChangeExtension(fileName, ".sym.ast")
        
        // Create a simplified representation for serialization
        let simplifiedSymbols =
            symbolUses
            |> Array.map (fun symbolUse ->
                {|
                    SymbolName = symbolUse.Symbol.DisplayName
                    SymbolType = symbolUse.Symbol.GetType().Name
                    IsDefinition = symbolUse.IsFromDefinition
                    FileName = symbolUse.Range.FileName
                    StartLine = symbolUse.Range.Start.Line
                    StartColumn = symbolUse.Range.Start.Column
                    EndLine = symbolUse.Range.End.Line
                    EndColumn = symbolUse.Range.End.Column
                |}
            )
            
        // Serialize to JSON
        let json = JsonSerializer.Serialize(simplifiedSymbols, jsonOptions)
        File.WriteAllText(outputPath, json)
        
        printfn "Symbolic AST written to %s" outputPath
        true
    with ex ->
        printfn "Error writing symbolic AST: %s" ex.Message
        false

/// Write XML metadata to file
let writeXmlMetadata (fileName: string) (checkResults: FSharpCheckProjectResults) =
    try
        let outputPath = Path.ChangeExtension(fileName, ".xml.ast")
        
        // Extract XML documentation from symbols
        let xmlDocs =
            checkResults.AssemblyContents.ImplementationFiles
            |> Seq.collect (fun implFile ->
                implFile.Declarations
                |> Seq.collect (fun decl ->
                    match decl with
                    | FSharpImplementationFileDeclaration.Entity (entity, _) ->
                        [|
                            match extractXmlDocFromMemberOrFunction entity with
                            | Some xmlContent -> 
                                yield {|
                                    SymbolName = entity.DisplayName
                                    SymbolType = "Entity"
                                    XmlLines = xmlContent
                                |}
                            | None -> ()
                        |]
                    | FSharpImplementationFileDeclaration.MemberOrFunctionOrValue(mfv, _, _) ->
                        [|
                            match extractXmlDocFromMemberOrFunction mfv with
                            | Some xmlContent ->
                                yield {|
                                    SymbolName = mfv.DisplayName
                                    SymbolType = "MemberOrFunctionOrValue"
                                    XmlLines = xmlContent
                                |}
                            | None -> ()
                        |]
                    | _ -> [||]
                )
            )
            |> Seq.toArray
            
        // Serialize to JSON
        let json = JsonSerializer.Serialize(xmlDocs, jsonOptions)
        File.WriteAllText(outputPath, json)
        
        printfn "XML metadata written to %s" outputPath
        true
    with ex ->
        printfn "Error writing XML metadata: %s" ex.Message
        false

/// Write PSG to file
let writePSG (fileName: string) (psg: ProgramSemanticGraph) =
    try
        let outputPath = Path.ChangeExtension(fileName, ".psg")
        
        // Create a simplified representation for serialization
        let simplifiedPSG = {|
            ModuleCount = psg.ModuleNodes.Count
            TypeCount = psg.TypeNodes.Count
            ValueCount = psg.ValueNodes.Count
            ExpressionCount = psg.ExpressionNodes.Count
            PatternCount = psg.PatternNodes.Count
            SymbolCount = psg.SymbolTable.Count
            EntryPointCount = psg.EntryPoints.Length
            Modules = 
                psg.ModuleNodes
                |> Map.toSeq
                |> Seq.map (fun (nodeId, node) ->
                    // Extract qualified path through pattern matching
                    let moduleName = 
                        match node.Syntax with
                        | SynModuleOrNamespace(longId, isRec, kind, decls, xmlDoc, attrs, access, range, trivia) ->
                            longId |> List.map (fun id -> id.idText) |> String.concat "."
                    
                    {|
                        Id = nodeId.Value
                        QualifiedName = moduleName
                        Range = node.SourceLocation.Range
                        ChildCount = node.Children.Length
                        HasMLIRMetadata = node.Metadata.IsSome
                    |}
                )
                |> Seq.toArray
            Types =
                psg.TypeNodes
                |> Map.toSeq
                |> Seq.map (fun (nodeId, node) ->
                    // Extract type name through pattern matching
                    let typeName = 
                        match node.Syntax with
                        | SynTypeDefn(typeInfo, implType, members, None, range, trivia) ->
                            match typeInfo with
                            | SynComponentInfo(_, _, _, id, _, _, _, _) ->
                                id |> List.map (fun i -> i.idText) |> String.concat "."
                        | _ -> "unknown"
                    
                    {|
                        Id = nodeId.Value
                        Name = typeName
                        Range = node.SourceLocation.Range
                        HasMLIRMetadata = node.Metadata.IsSome
                    |}
                )
                |> Seq.toArray
            Values =
                psg.ValueNodes
                |> Map.toSeq
                |> Seq.map (fun (nodeId, node) ->
                    // Extract name from binding through pattern matching
                    let bindingName =
                        match node.Syntax with
                        | SynBinding(_, _, _, _, _, _, _,  pat, _, _, _, _, _) ->
                            match pat with
                            | SynPat.Named(SynIdent(ident, _), _, _, _) -> ident.idText
                            | SynPat.LongIdent(SynLongIdent(ids, _, _), _, _, _, _, _) ->
                                ids |> List.map (fun id -> id.idText) |> String.concat "."
                            | _ -> "anonymous" // Handle other pattern cases
                    
                    {|
                        Id = nodeId.Value
                        Name = bindingName
                        Range = node.SourceLocation.Range
                        HasMLIRMetadata = node.Metadata.IsSome
                        IsEntryPoint = psg.EntryPoints |> List.contains nodeId
                    |}
                )
                |> Seq.toArray
        |}
            
        // Serialize to JSON
        let json = JsonSerializer.Serialize(simplifiedPSG, jsonOptions)
        File.WriteAllText(outputPath, json)
        
        printfn "PSG written to %s" outputPath
        true
    with ex ->
        printfn "Error writing PSG: %s" ex.Message
        false

/// Write all intermediate assets for a processed project
let writeIntermediates 
    (outputDir: string) 
    (parseResults: FSharpParseFileResults[]) 
    (checkResults: FSharpCheckProjectResults)
    (psg: ProgramSemanticGraph) =
    
    // Create output directory if it doesn't exist
    if not (Directory.Exists outputDir) then
        Directory.CreateDirectory outputDir |> ignore
    
    // Write individual files
    parseResults
    |> Array.iter (fun parseResult ->
        let fileName = Path.Combine(outputDir, Path.GetFileName(parseResult.FileName))
        
        // Write typed AST
        writeTypedAST fileName checkResults |> ignore
        
        // Write symbolic AST
        let symbolUses = checkResults.GetAllUsesOfAllSymbols()
        writeSymbolicAST fileName symbolUses |> ignore
        
        // Write XML metadata
        writeXmlMetadata fileName checkResults |> ignore
    )
    
    // Write PSG
    let mainFile = 
        match parseResults with
        | [||] -> "output.fs"
        | results -> results.[0].FileName
        
    let psgFile = Path.Combine(outputDir, Path.GetFileName(mainFile))
    writePSG psgFile psg |> ignore
    
    printfn "All intermediate assets written to %s" outputDir