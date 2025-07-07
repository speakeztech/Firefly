module Core.PSG.XmlDocParser

open System
open FSharp.Compiler.Symbols
open XParsec
open XParsec.Parsers
open XParsec.CharParsers
open Core.PSG.Types

/// Parse a quoted string (handles both single and double quotes)
let private pQuotedString = 
    choice [
        between (pchar '"') (pchar '"') (manyChars (noneOf ['"']))
        between (pchar '\'') (pchar '\'') (manyChars (noneOf ['\'']))
    ]

/// Parse an XML attribute name="value"
let private pAttribute =
    parser {
        let! name = manyChars (satisfy (fun c -> 
            (c >= 'a' && c <= 'z') || 
            (c >= 'A' && c <= 'Z') || 
            (c >= '0' && c <= '9') || 
            c = '-' || c = '_'))
        let! _ = spaces
        let! _ = pchar '='
        let! _ = spaces
        let! value = pQuotedString
        return (name, value)
    }

/// Parse a sequence of XML attributes
let private pAttributes =
    parser {
        let! attrs = many (parser {
            let! _ = spaces
            let! attr = pAttribute
            return attr
        })
        return Map.ofSeq attrs
    }

/// Parse opening tag with attributes: <tagname attr1="val1" attr2="val2">
let private pOpenTag tagName =
    parser {
        let! _ = pstring "<"
        let! _ = pstring tagName
        let! attrs = pAttributes
        let! _ = spaces
        let! _ = pchar '>'
        return attrs
    }

/// Parse self-closing tag with attributes: <tagname attr1="val1" attr2="val2" />
let private pSelfClosingTag tagName =
    parser {
        let! _ = pstring "<"
        let! _ = pstring tagName
        let! attrs = pAttributes
        let! _ = spaces
        let! _ = pstring "/>"
        return attrs
    }

/// Parse closing tag: </tagname>
let private pCloseTag tagName =
    parser {
        let! _ = pstring "</"
        let! _ = pstring tagName
        let! _ = pchar '>'
        return ()
    }

/// Parse content between tags
let private pContent =
    manyChars (noneOf ['<'; '>'])

/// Parse a complete MLIR tag section with parameters and attributes
let private pMlirSection =
    parser {
        // Parse either self-closing tag or opening tag
        let! mlirAttrs = 
            (pSelfClosingTag "mlir") <|> (pOpenTag "mlir")
        
        // Extract dialect and operation attributes
        let dialectValue = Map.tryFind "dialect" mlirAttrs
        let operationValue = Map.tryFind "operation" mlirAttrs
        
        let dialectOpt = 
            match dialectValue with
            | Some value -> 
                match DialectName.Create value with
                | Ok dialect -> Some dialect
                | Result.Error _ -> None
            | None -> None
            
        let operationOpt = 
            match operationValue with
            | Some value -> 
                match OperationName.Create value with
                | Ok operation -> Some operation
                | Result.Error _ -> None
            | None -> None
        
        // For self-closing tags, return immediately
        if mlirAttrs |> Map.containsKey "/" then
            return {
                Dialect = dialectOpt
                Operation = operationOpt
                Parameters = Map.empty
                Attributes = Map.empty
            }
        else
            // For normal tags, parse param and attr elements
            let! elements = many (choice [
                // Parse param tags
                parser {
                    let! paramAttrs = (pOpenTag "param")
                    let! content = pContent
                    let! _ = pCloseTag "param"
                    let name = Map.tryFind "name" paramAttrs |> Option.defaultValue ""
                    return Choice1Of2 (ParameterName name, ParameterValue content)
                }
                
                // Parse attr tags
                parser {
                    let! attrAttrs = (pOpenTag "attr")
                    let! content = pContent
                    let! _ = pCloseTag "attr"
                    let name = Map.tryFind "name" attrAttrs |> Option.defaultValue ""
                    return Choice2Of2 (AttributeName name, AttributeValue content)
                }
                
                // Skip any other content
                parser {
                    let! _ = manyCharsTill anyChar (lookAhead (choice [
                        (pstring "</mlir>") 
                        (pstring "<param") 
                        (pstring "<attr")
                    ]))
                    return Choice1Of2 (ParameterName "", ParameterValue "")
                }
            ])
            
            // Skip optional closing tag
            let! _ = optional (pCloseTag "mlir")
            
            // Extract parameters and attributes
            let parameters = 
                elements 
                |> Seq.choose (function 
                    | Choice1Of2 (name, value) when name <> ParameterName "" -> Some (name, value) 
                    | _ -> None)
                |> Map.ofSeq
                
            let attributes = 
                elements 
                |> Seq.choose (function 
                    | Choice2Of2 attr -> Some attr 
                    | _ -> None)
                |> Map.ofSeq
                
            return {
                Dialect = dialectOpt
                Operation = operationOpt
                Parameters = parameters
                Attributes = attributes
            }
    }

/// Parse MLIR metadata from XML documentation
let parseMLIRMetadata (xmlDoc: string[]) : MLIRMetadataParseResult = 
    if xmlDoc = null || Array.isEmpty xmlDoc then
        Missing
    else
        // Combine lines into a single string for parsing
        let xmlContent = String.Join("\n", xmlDoc)
        
        // Look for MLIR section by first finding the mlir tag
        let mlirTagParser = 
            manyCharsTill anyChar (lookAhead (pstring "<mlir")) >>. 
            pMlirSection
            
        // Use a helper function to disambiguate
        let handleParseResult result =
            match result with
            | Ok success -> Valid success.Parsed
            | _ -> 
                if xmlContent.Contains("<mlir") then
                    Invalid ["Failed to parse MLIR metadata"]
                else
                    Missing
                    
        handleParseResult (mlirTagParser (Reader.ofString xmlContent ()))

/// Extract XML documentation from a member, function, or value symbol
let extractXmlDocFromMemberOrFunction (symbol: FSharpSymbol) : string[] option =
    match symbol with
    | :? FSharpMemberOrFunctionOrValue as mfv ->
        try
            // Access XmlDocSig which appears to be a string option
            match mfv.XmlDocSig with
            | xmlContent when not (String.IsNullOrEmpty xmlContent) ->
                // Split the XML content into lines
                Some (xmlContent.Split([|'\n'|], StringSplitOptions.RemoveEmptyEntries))
            | _ -> None
        with
        | _ -> None
    | _ -> None
        
/// Get MLIR metadata for a symbol
let getMetadataForSymbol (symbol: FSharpSymbol) : MLIRMetadata option =
    match extractXmlDocFromMemberOrFunction symbol with
    | Some xmlDoc ->
        match parseMLIRMetadata xmlDoc with
        | Valid metadata -> Some metadata
        | _ -> None
    | None -> None