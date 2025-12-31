module Core.Meta.Parser

open System
open System.Xml.Linq
open FSharp.Native.Compiler.Symbols

/// MLIR hint extracted from XML documentation
type MLIRHint = {
    Dialect: string
    Operation: string
    Parameters: Map<string, string>
    MemoryStrategy: MemoryStrategy option
}

and MemoryStrategy =
    | StackOnly
    | StaticAllocation
    | PooledAllocation of poolName: string
    | StreamingBuffer of size: int

/// Pattern recognition for common F# constructs
module Patterns =
    open FSharp.Native.Compiler.Symbols
    
    /// Suggest hints for recursive functions
    let suggestForRecursion (mfv: FSharpMemberOrFunctionOrValue) =
        if mfv.IsFunction && mfv.LogicalName.Contains("rec") then
            Some {
                Dialect = "scf"
                Operation = "for"
                Parameters = Map.ofList ["step", "1"; "parallel", "false"]
                MemoryStrategy = Some StackOnly
            }
        else None
    
    /// Suggest hints for collection operations
    let suggestForCollection (entity: FSharpEntity) =
        if entity.IsArrayType then
            Some {
                Dialect = "memref"
                Operation = "alloca"
                Parameters = Map.ofList ["alignment", "16"]
                MemoryStrategy = Some StackOnly
            }
        elif entity.IsFSharp && entity.LogicalName.Contains("List") then
            Some {
                Dialect = "memref"
                Operation = "view"
                Parameters = Map.empty
                MemoryStrategy = Some (PooledAllocation "list_pool")
            }
        else None

/// Extract MLIR hints from XML documentation using System.Xml.Linq
let extractMLIRHints (xmlDoc: FSharpXmlDoc) =
    match xmlDoc with
    | FSharpXmlDoc.FromXmlText xmlText ->
        try
            let xmlString = 
                xmlText.UnprocessedLines 
                |> Array.map (fun line -> line.Trim())
                |> String.concat " "
            
            // Wrap in root element if needed
            let wrappedXml = 
                if xmlString.StartsWith("<") then xmlString
                else sprintf "<doc>%s</doc>" xmlString
            
            let doc = XDocument.Parse(wrappedXml)
            
            // Look for firefly-specific elements
            let fireflyElem = doc.Descendants(XName.Get("firefly")) |> Seq.tryHead
            
            match fireflyElem with
            | Some elem ->
                // Extract MLIR dialect info
                let mlirElem = elem.Element(XName.Get("mlir"))
                let dialectOpt = mlirElem |> Option.ofObj |> Option.bind (fun e -> e.Attribute(XName.Get("dialect")) |> Option.ofObj) |> Option.map (fun a -> a.Value)
                let opOpt = mlirElem |> Option.ofObj |> Option.bind (fun e -> e.Attribute(XName.Get("op")) |> Option.ofObj) |> Option.map (fun a -> a.Value)
                
                // Extract parameters
                let parameters = 
                    match mlirElem with
                    | null -> Map.empty
                    | mlir ->
                        mlir.Elements(XName.Get("param"))
                        |> Seq.choose (fun p ->
                            let name = p.Attribute(XName.Get("name"))
                            let value = p.Attribute(XName.Get("value"))
                            match name, value with
                            | null, _ | _, null -> None
                            | n, v -> Some (n.Value, v.Value))
                        |> Map.ofSeq
                
                // Extract memory strategy
                let memoryElem = elem.Element(XName.Get("memory"))
                let memoryStrategy =
                    match memoryElem with
                    | null -> None
                    | mem ->
                        let strategy = mem.Attribute(XName.Get("strategy"))
                        match strategy with
                        | null -> None
                        | s ->
                            match s.Value with
                            | "stack" -> Some StackOnly
                            | "static" -> Some StaticAllocation
                            | "pooled" ->
                                let pool = mem.Attribute(XName.Get("pool"))
                                Some (PooledAllocation (if pool = null then "default" else pool.Value))
                            | "streaming" ->
                                let size = mem.Attribute(XName.Get("size"))
                                match size with
                                | null -> Some (StreamingBuffer 4096)
                                | s -> 
                                    match Int32.TryParse(s.Value) with
                                    | true, sz -> Some (StreamingBuffer sz)
                                    | _ -> Some (StreamingBuffer 4096)
                            | _ -> None
                
                match dialectOpt, opOpt with
                | Some dialect, Some op ->
                    Some {
                        Dialect = dialect
                        Operation = op
                        Parameters = parameters
                        MemoryStrategy = memoryStrategy
                    }
                | _ -> None
            | None -> None
            
        with _ -> 
            // If XML parsing fails, return None
            None
    
    | _ -> None

/// Get MLIR hints for a symbol
let getSymbolHints (symbol: FSharpSymbol) =
    match symbol with
    | :? FSharpMemberOrFunctionOrValue as mfv ->
        extractMLIRHints mfv.XmlDoc
    | :? FSharpEntity as entity ->
        extractMLIRHints entity.XmlDoc
    | :? FSharpField as field ->
        extractMLIRHints field.XmlDoc
    | _ -> None