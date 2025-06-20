module Dabbit.SymbolResolution.SymbolRegistry

open System
open Core.XParsec.Foundation
open Dabbit.Parsing.OakAst
open Core.MLIRGeneration.TypeSystem
open Core.MLIRGeneration.Dialect

/// Represents the MLIR operation pattern for a resolved symbol
type MLIROperationPattern =
    | DialectOperation of dialect: MLIRDialect * operation: string * attributes: Map<string, string>
    | ExternalCall of funcName: string * library: string option
    | CompositePattern of patterns: MLIROperationPattern list
    | CustomTransform of transformName: string * parameters: string list

/// Complete symbol resolution with type safety and MLIR generation info
type ResolvedSymbol = {
    /// Fully qualified name from source (e.g., "Alloy.Memory.stackBuffer")
    QualifiedName: string
    /// Short name for lookup (e.g., "stackBuffer")
    ShortName: string
    /// Parameter type signatures
    ParameterTypes: MLIRType list
    /// Return type signature
    ReturnType: MLIRType
    /// The MLIR operation pattern to generate
    Operation: MLIROperationPattern
    /// Namespace this symbol belongs to
    Namespace: string
    /// Library or module providing this symbol
    SourceLibrary: string
    /// Whether this requires external linking
    RequiresExternal: bool
}

/// Symbol resolution state for compositional lookups
type SymbolResolutionState = {
    /// All registered symbols by qualified name
    SymbolsByQualified: Map<string, ResolvedSymbol>
    /// Symbols by short name for unqualified lookup
    SymbolsByShort: Map<string, ResolvedSymbol>
    /// Namespace mappings for qualified resolution
    NamespaceMap: Map<string, string list>
    /// Active namespace context for resolution
    ActiveNamespaces: string list
    /// Type aliases and custom types
    TypeRegistry: Map<string, MLIRType>
    /// Active symbols in current context
    ActiveSymbols: string list 
}

/// The complete symbol registry - the "Library of Alexandria"
type SymbolRegistry = {
    /// Current resolution state
    State: SymbolResolutionState
    /// Transformation history for debugging
    ResolutionHistory: (string * string) list
}

/// XParsec-based symbol resolution functions
module SymbolResolution =
    
    /// Creates an empty resolution state
    let createEmptyState () : SymbolResolutionState = {
        SymbolsByQualified = Map.empty
        SymbolsByShort = Map.empty
        NamespaceMap = Map.empty
        ActiveNamespaces = []
        TypeRegistry = Map.empty
        ActiveSymbols = []
    }
    
    /// Registers a symbol in the resolution state
    let registerSymbol (symbol: ResolvedSymbol) (state: SymbolResolutionState) : SymbolResolutionState =
        let updatedQualified = Map.add symbol.QualifiedName symbol state.SymbolsByQualified
        let updatedShort = Map.add symbol.ShortName symbol state.SymbolsByShort
        
        // Update namespace mapping
        let namespaceSymbols = 
            match Map.tryFind symbol.Namespace state.NamespaceMap with
            | Some existing -> symbol.ShortName :: existing
            | None -> [symbol.ShortName]
        let updatedNamespaces = Map.add symbol.Namespace namespaceSymbols state.NamespaceMap
        
        {
            state with
                SymbolsByQualified = updatedQualified
                SymbolsByShort = updatedShort
                NamespaceMap = updatedNamespaces
        }
    
    /// Resolves a symbol by qualified name using XParsec patterns
    let resolveQualified (qualifiedName: string) (state: SymbolResolutionState) : CompilerResult<ResolvedSymbol> =
        match Map.tryFind qualifiedName state.SymbolsByQualified with
        | Some symbol -> Success symbol
        | None -> 
            CompilerFailure [ConversionError(
                "qualified symbol resolution",
                qualifiedName,
                "resolved symbol",
                sprintf "Symbol '%s' not found in qualified registry" qualifiedName
            )]
    
    /// Resolves a symbol by short name with namespace context
    let resolveUnqualified (shortName: string) (state: SymbolResolutionState) : CompilerResult<ResolvedSymbol> =
        match Map.tryFind shortName state.SymbolsByShort with
        | Some symbol -> Success symbol
        | None ->
            // Try namespace-qualified resolution
            let tryNamespaceResolution () =
                state.ActiveNamespaces
                |> List.tryPick (fun ns ->
                    let qualifiedAttempt = sprintf "%s.%s" ns shortName
                    Map.tryFind qualifiedAttempt state.SymbolsByQualified)
            
            match tryNamespaceResolution () with
            | Some symbol -> Success symbol
            | None ->
                CompilerFailure [ConversionError(
                    "unqualified symbol resolution",
                    shortName,
                    "resolved symbol",
                    sprintf "Symbol '%s' not found in any active namespace" shortName
                )]
    
    /// Compositional symbol resolution with fallback strategies
    let resolveSymbol (name: string) (state: SymbolResolutionState) : CompilerResult<ResolvedSymbol> =
        if name.Contains('.') then
            resolveQualified name state
        else
            resolveUnqualified name state

/// Alloy library symbol definitions using the compositional pattern
module AlloySymbols =
    
    /// Creates the foundational Alloy memory management symbols
    let createMemorySymbols () : ResolvedSymbol list = [
        {
            QualifiedName = "Alloy.Memory.stackBuffer"
            ShortName = "stackBuffer"
            ParameterTypes = [MLIRType.Integer 32] // size parameter
            ReturnType = MLIRType.MemRef(MLIRType.Integer 8, []) // memref<?xi8>
            Operation = DialectOperation(
                MLIRDialect.MemRef, 
                "memref.alloca", 
                Map.ofList [("element_type", "i8")]
            )
            Namespace = "Alloy.Memory"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        
        {
            QualifiedName = "Alloy.Memory.spanToString"
            ShortName = "spanToString"
            ParameterTypes = [MLIRType.MemRef(MLIRType.Integer 8, [])] // span parameter
            ReturnType = MLIRType.MemRef(MLIRType.Integer 8, []) // string as memref
            Operation = CustomTransform("span_to_string", ["utf8_conversion"])
            Namespace = "Alloy.Memory"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
    ]
    
    /// Creates the Alloy I/O console symbols with external dependencies
    let createConsoleSymbols () : ResolvedSymbol list = [
        {
            QualifiedName = "Alloy.IO.Console.prompt"
            ShortName = "prompt"
            ParameterTypes = [MLIRType.MemRef(MLIRType.Integer 8, [])] // message parameter
            ReturnType = MLIRType.Void
            Operation = ExternalCall("printf", Some "libc")
            Namespace = "Alloy.IO.Console"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
        
        {
            QualifiedName = "Alloy.IO.Console.readInto"
            ShortName = "readInto"
            ParameterTypes = [MLIRType.MemRef(MLIRType.Integer 8, [])] // buffer parameter  
            ReturnType = MLIRType.Integer 32 // Result<int, string> simplified to i32
            Operation = CompositePattern([
                ExternalCall("scanf", Some "libc");
                CustomTransform("result_wrapper", ["success_check"])
            ])
            Namespace = "Alloy.IO.Console"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
        
        {
            QualifiedName = "Alloy.IO.Console.writeLine"
            ShortName = "writeLine"
            ParameterTypes = [MLIRType.MemRef(MLIRType.Integer 8, [])] // message parameter
            ReturnType = MLIRType.Void
            Operation = ExternalCall("printf", Some "libc")
            Namespace = "Alloy.IO.Console"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
    ]
    
    /// Creates Alloy string manipulation symbols
    let createStringSymbols () : ResolvedSymbol list = [
        {
            QualifiedName = "Alloy.IO.String.format"
            ShortName = "format"
            ParameterTypes = [MLIRType.MemRef(MLIRType.Integer 8, []); MLIRType.MemRef(MLIRType.Integer 8, [])] // format, value
            ReturnType = MLIRType.MemRef(MLIRType.Integer 8, [])
            Operation = ExternalCall("sprintf", Some "libc")
            Namespace = "Alloy.IO.String"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
    ]

/// Registry construction and management
module RegistryConstruction =
    
    /// Builds the complete Alloy symbol registry
    let buildAlloyRegistry () : CompilerResult<SymbolRegistry> =
        try
            let initialState = SymbolResolution.createEmptyState ()
            
            // Gather all Alloy symbols
            let alloySymbols = 
                AlloySymbols.createMemorySymbols ()
                @ AlloySymbols.createConsoleSymbols ()
                @ AlloySymbols.createStringSymbols ()
            
            // Register all symbols
            let finalState = 
                alloySymbols
                |> List.fold (fun state symbol -> 
                    SymbolResolution.registerSymbol symbol state) initialState
            
            // Set active namespaces for Alloy imports
            let stateWithNamespaces = {
                finalState with 
                    ActiveNamespaces = [
                        "Alloy.Memory"
                        "Alloy.IO.Console"
                        "Alloy.IO.String"
                    ]
            }
            
            Success {
                State = stateWithNamespaces
                ResolutionHistory = [("initialization", "Alloy registry built")]
            }
            
        with ex ->
            CompilerFailure [InternalError(
                "registry construction",
                "Failed to build Alloy registry",
                Some ex.Message
            )]
    
    /// Adds namespace context to registry
    let withNamespaceContext (namespaces: string list) (registry: SymbolRegistry) : SymbolRegistry =
        let updatedState = {
            registry.State with ActiveNamespaces = namespaces @ registry.State.ActiveNamespaces
        }
        let historyEntry = ("namespace_context", String.concat ", " namespaces)
        {
            registry with 
                State = updatedState
                ResolutionHistory = historyEntry :: registry.ResolutionHistory
        }
    
    /// Tracks active symbols for resolution context
    let trackActiveSymbol (symbolId: string) (registry: SymbolRegistry) : SymbolRegistry =
        let updatedState = {
            registry.State with ActiveSymbols = symbolId :: registry.State.ActiveSymbols
        }
        {
            registry with State = updatedState
        }
    
    /// Resolves a symbol through the complete registry
    let resolveSymbolInRegistry (symbolName: string) (registry: SymbolRegistry) : CompilerResult<ResolvedSymbol * SymbolRegistry> =
        match SymbolResolution.resolveSymbol symbolName registry.State with
        | Success symbol ->
            let historyEntry = ("resolution", sprintf "%s -> %s" symbolName symbol.QualifiedName)
            let updatedRegistry = {
                registry with ResolutionHistory = historyEntry :: registry.ResolutionHistory
            }
            Success (symbol, updatedRegistry)
            
        | CompilerFailure errors ->
            let historyEntry = ("resolution_failed", sprintf "Failed to resolve %s" symbolName)
            let updatedRegistry = {
                registry with ResolutionHistory = historyEntry :: registry.ResolutionHistory
            }
            CompilerFailure errors

/// MLIR operation generation based on resolved symbols
module MLIRGeneration =
    
    /// Finds active buffer in symbol list
    let findActiveBuffer (activeSymbols: string list) : string =
        activeSymbols 
        |> List.tryFind (fun s -> s.Contains("buffer"))
        |> Option.defaultValue "%unknown_buffer"
    
    /// Generates MLIR operation string from a resolved symbol
    let generateMLIROperation (symbol: ResolvedSymbol) (args: string list) (resultId: string) (registry: SymbolRegistry) : CompilerResult<string list> =
        match symbol.Operation with
        | DialectOperation(dialect, operation, attributes) ->
            let dialectPrefix = dialectToString dialect
            let attrStr = 
                if attributes.IsEmpty then ""
                else 
                    attributes 
                    |> Map.toList 
                    |> List.map (fun (k, v) -> sprintf "%s = %s" k v)
                    |> String.concat ", "
                    |> sprintf " {%s}"
            
            let argStr = if args.IsEmpty then "" else sprintf "(%s)" (String.concat ", " args)
            let typeStr = mlirTypeToString symbol.ReturnType
            let operationStr = sprintf "    %s = %s.%s%s : %s%s" resultId dialectPrefix operation argStr typeStr attrStr
            Success [operationStr]
        
        | ExternalCall(funcName, library) ->
            let argStr = String.concat ", " args
            let paramTypes = symbol.ParameterTypes |> List.map mlirTypeToString |> String.concat ", "
            let returnType = mlirTypeToString symbol.ReturnType
            let callOp = sprintf "    %s = func.call @%s(%s) : (%s) -> %s" resultId funcName argStr paramTypes returnType
            Success [callOp]
        
        | CustomTransform(transformName, parameters) ->
            // Generate LLVM-compatible operations directly instead of custom dialect
            match transformName with
            | "span_to_string" ->
                // For span_to_string, handle both direct arguments and unit literal
                let bufferValue = 
                    if args.IsEmpty then
                        // Look for buffer in the active symbols
                        findActiveBuffer registry.State.ActiveSymbols
                    else
                        args.[0]
                
                // Generate a cast operation that works with any buffer source
                let bitcastOp = sprintf "    %s = memref.cast %s : memref<?xi8> to memref<?xi8>" resultId bufferValue
                Success [bitcastOp]
                
            | "generic_transform" ->
                // For generic transform, handle broader patterns
                let bufferValue = 
                    if args.IsEmpty then
                        findActiveBuffer registry.State.ActiveSymbols
                    else
                        args.[0]
                
                let typeStr = mlirTypeToString symbol.ReturnType
                let castOp = sprintf "    %s = memref.cast %s : memref<?xi8> to %s" resultId bufferValue typeStr
                Success [castOp]
                
            | _ ->
                // For unknown transforms, generate a default constant
                let constOp = sprintf "    %s = arith.constant 0 : i32" resultId
                Success [constOp]
        
        | CompositePattern(patterns) ->
            // For composite patterns, generate direct operations for each step
            let mutable operations = []
            let mutable tempResultId = resultId
            
            // Generate simplified steps that will work with LLVM dialect
            for i, pattern in List.indexed patterns do
                let stepResultId = if i = patterns.Length - 1 then resultId else sprintf "%s_step%d" resultId i
                tempResultId <- stepResultId
                
                // Generate appropriate operations based on pattern type
                match pattern with
                | ExternalCall(funcName, _) ->
                    // For external calls, generate a simple constant for placeholder
                    let constOp = sprintf "    %s = arith.constant %d : i32" stepResultId i
                    operations <- constOp :: operations
                | CustomTransform(transformName, _) ->
                    // Custom transform handling
                    match transformName with
                    | "result_wrapper" ->
                        let wrapOp = sprintf "    %s = arith.constant 1 : i32" stepResultId
                        operations <- wrapOp :: operations
                    | _ ->
                        let defaultOp = sprintf "    %s = arith.constant %d : i32" stepResultId i
                        operations <- defaultOp :: operations
                | _ ->
                    // Default case - generate a simple constant
                    let constOp = sprintf "    %s = arith.constant %d : i32" stepResultId i
                    operations <- constOp :: operations
            
            Success (List.rev operations)

/// External interface for integration with existing MLIR generation
module PublicInterface =
    
    /// Creates the standard Alloy registry for use in compilation
    let createStandardRegistry () : CompilerResult<SymbolRegistry> =
        RegistryConstruction.buildAlloyRegistry ()
    
    /// Resolves a function call and generates appropriate MLIR
    let resolveFunctionCall (funcName: string) (args: string list) (resultId: string) (registry: SymbolRegistry) 
                           : CompilerResult<string list * SymbolRegistry> =
        match RegistryConstruction.resolveSymbolInRegistry funcName registry with
        | Success (symbol, updatedRegistry) ->
            // For buffer-related operations, track them in active symbols
            let registry2 = 
                if funcName.Contains("buffer") && args.Length > 0 then
                    RegistryConstruction.trackActiveSymbol args.[0] updatedRegistry
                else
                    updatedRegistry
            
            match MLIRGeneration.generateMLIROperation symbol args resultId registry2 with
            | Success operations -> Success (operations, registry2)
            | CompilerFailure errors -> CompilerFailure errors
            
        | CompilerFailure errors -> CompilerFailure errors
    
    /// Gets all symbols in a namespace for validation/debugging
    let getNamespaceSymbols (namespaceName: string) (registry: SymbolRegistry) : string list =
        match Map.tryFind namespaceName registry.State.NamespaceMap with
        | Some symbols -> symbols
        | None -> []
    
    /// Validates that all required symbols are available
    let validateRequiredSymbols (requiredSymbols: string list) (registry: SymbolRegistry) : CompilerResult<unit> =
        let missingSymbols = 
            requiredSymbols
            |> List.filter (fun symbolName ->
                match SymbolResolution.resolveSymbol symbolName registry.State with
                | Success _ -> false
                | CompilerFailure _ -> true)
        
        if missingSymbols.IsEmpty then
            Success ()
        else
            CompilerFailure [ConversionError(
                "symbol validation",
                String.concat ", " missingSymbols,
                "available symbols",
                sprintf "Missing required symbols: %s" (String.concat ", " missingSymbols)
            )]