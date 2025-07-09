module Core.Meta.Parser

open XParsec
open FSharp.Compiler.Symbols
open FSharp.Compiler.Xml

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

/// Parser combinators for MLIR metadata
module Parsers =
    open XParsec.Xml
    
    let mlirDialect = 
        element "mlir" >>=. 
        attr "dialect" >>= fun dialect ->
        attr "op" >>= fun op ->
        result { Dialect = dialect; Operation = op }
    
    let mlirParam =
        element "param" >>=.
        attr "name" >>= fun name ->
        attr "value" >>= fun value ->
        result (name, value)
    
    let mlirParams =
        many mlirParam >>= fun parameters ->
        result (Map.ofList parameters)
    
    let memoryHint =
        element "memory" >>=.
        attr "strategy" >>= fun strategy ->
        match strategy with
        | "stack" -> result StackOnly
        | "static" -> result StaticAllocation
        | "pooled" -> 
            attr "pool" >>= fun poolName ->
            result (PooledAllocation poolName)
        | "streaming" ->
            attr "size" >>= fun sizeStr ->
            match System.Int32.TryParse(sizeStr) with
            | true, size -> result (StreamingBuffer size)
            | false, _ -> fail "Invalid streaming buffer size"
        | _ -> fail $"Unknown memory strategy: {strategy}"
    
    let mlirHint =
        element "firefly" >>=.
        optional mlirDialect >>= fun dialectInfo ->
        optional mlirParams >>= fun parameters ->
        optional memoryHint >>= fun memory ->
        
        match dialectInfo with
        | Some di ->
            result {
                Dialect = di.Dialect
                Operation = di.Operation
                Parameters = defaultArg parameters Map.empty
                MemoryStrategy = memory
            }
        | None -> fail "Missing MLIR dialect information"

/// Extract MLIR hints from XML documentation
let extractMLIRHints (xmlDoc: FSharpXmlDoc) =
    match xmlDoc with
    | FSharpXmlDoc.FromXmlText xmlText ->
        let xmlString = 
            xmlText.UnprocessedLines 
            |> String.concat "\n"
        
        // Parse XML documentation for Firefly-specific tags
        match parse Parsers.mlirHint xmlString with
        | Success hint -> Some hint
        | Failure _ -> None
    
    | _ -> None

/// Get MLIR hints for a symbol
let getSymbolHints (symbol: FSharpSymbol) =
    match symbol with
    | :? FSharpMemberOrFunctionOrValue as mfv ->
        extractMLIRHints mfv.XmlDoc
    | :? FSharpEntity as entity ->
        extractMLIRHints entity.XmlDoc
    | _ -> None

/// Common MLIR patterns for F# constructs
module Patterns =
    /// Suggest MLIR dialect for recursive functions
    let suggestForRecursion (func: FSharpMemberOrFunctionOrValue) =
        if func.IsCompilerGenerated then None
        else
            // Check if function is recursive
            let isRecursive = func.IsValue && func.FullName.Contains("@")
            if isRecursive then
                Some {
                    Dialect = "scf"  // Structured Control Flow
                    Operation = "while"
                    Parameters = Map.empty
                    MemoryStrategy = Some StackOnly
                }
            else None
    
    /// Suggest memory strategy for collections
    let suggestForCollection (entity: FSharpEntity) =
        if entity.IsArrayType then
            Some {
                Dialect = "memref"
                Operation = "alloc"
                Parameters = Map.ofList ["alignment", "64"]
                MemoryStrategy = Some (PooledAllocation "default")
            }
        elif entity.IsFSharpList then
            Some {
                Dialect = "llvm"
                Operation = "alloca"
                Parameters = Map.empty
                MemoryStrategy = Some StackOnly
            }
        else None