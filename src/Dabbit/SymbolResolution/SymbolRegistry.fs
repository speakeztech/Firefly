module Dabbit.SymbolResolution.SymbolRegistry

open System
open Core.XParsec.Foundation
open Dabbit.Parsing.OakAst
open Core.MLIRGeneration.TypeSystem
open Core.MLIRGeneration.Dialect

/// Create MLIRTypes helper module for backwards compatibility
module MLIRTypes =
    /// Creates an integer type with specified width
    let createInteger (width: int) = {
        Category = MLIRTypeCategory.Integer
        Width = Some width
        ElementType = None
        Parameters = []
        ReturnType = None
    }

    /// Creates a float type with specified width
    let createFloat (width: int) = {
        Category = MLIRTypeCategory.Float
        Width = Some width
        ElementType = None
        Parameters = []
        ReturnType = None
    }

    /// Creates a void type
    let createVoid() = {
        Category = MLIRTypeCategory.Void
        Width = None
        ElementType = None
        Parameters = []
        ReturnType = None
    }

    /// Creates a memory reference type with element type and dimensions
    let createMemRef (elementType: MLIRType) (dimensions: int list) = {
        Category = MLIRTypeCategory.MemRef
        Width = None
        ElementType = Some elementType
        Parameters = dimensions |> List.map (fun d -> createInteger d)
        ReturnType = None
    }

    /// Creates a function type with parameters and return type
    let createFunction (inputTypes: MLIRType list) (returnType: MLIRType) = {
        Category = MLIRTypeCategory.Function
        Width = None
        ElementType = None
        Parameters = inputTypes
        ReturnType = Some returnType
    }

/// Type analysis module for compatibility checking
module TypeAnalysis =
    /// Checks if types are compatible for conversion
    let canConvertTo (fromType: MLIRType) (toType: MLIRType) : bool =
        fromType = toType || 
        (fromType.Category = MLIRTypeCategory.Integer && toType.Category = MLIRTypeCategory.Integer) ||
        (fromType.Category = MLIRTypeCategory.Float && toType.Category = MLIRTypeCategory.Float) ||
        (fromType.Category = MLIRTypeCategory.Integer && toType.Category = MLIRTypeCategory.Float) ||
        (fromType.Category = MLIRTypeCategory.MemRef && toType.Category = MLIRTypeCategory.MemRef)

/// MLIR operation pattern for resolved symbols
type MLIROperationPattern =
    | DialectOperation of dialect: MLIRDialect * operation: string * attributes: Map<string, string>
    | ExternalCall of funcName: string * library: string option
    | CompositePattern of patterns: MLIROperationPattern list
    | CustomTransform of transformName: string * parameters: string list

/// Symbol with type safety and MLIR generation info
type ResolvedSymbol = {
    QualifiedName: string                // Fully qualified name (e.g., "Alloy.Memory.stackBuffer")
    ShortName: string                    // Short name for lookup (e.g., "stackBuffer")
    ParameterTypes: MLIRType list        // Parameter type signatures
    ReturnType: MLIRType                 // Return type signature
    Operation: MLIROperationPattern      // MLIR operation pattern to generate
    Namespace: string                    // Namespace this symbol belongs to
    SourceLibrary: string                // Library or module providing this symbol
    RequiresExternal: bool               // Whether this requires external linking
}

/// Pattern-based transformation for symbols
type SymbolPattern = {
    Name: string                         // Name of the pattern
    Description: string                  // Description of what the pattern does
    QualifiedNamePattern: string         // Qualified name pattern to match
    OperationPattern: MLIROperationPattern // Operation pattern to generate
    TypeSignature: MLIRType list * MLIRType // Type signature (parameters and return type)
    AstPattern: OakExpression -> bool    // Pattern matcher for AST expressions
}

/// Symbol resolution state
type SymbolResolutionState = {
    SymbolsByQualified: Map<string, ResolvedSymbol>  // All registered symbols by qualified name
    SymbolsByShort: Map<string, ResolvedSymbol>      // Symbols by short name for unqualified lookup
    NamespaceMap: Map<string, string list>           // Namespace mappings for qualified resolution
    ActiveNamespaces: string list                    // Active namespace context for resolution
    TypeRegistry: Map<string, MLIRType>              // Type aliases and custom types
    ActiveSymbols: string list                       // Active symbols in current context
    PatternRegistry: SymbolPattern list              // Pattern registry for AST-driven transformation
}

/// The complete symbol registry
type SymbolRegistry = {
    State: SymbolResolutionState                     // Current resolution state
    ResolutionHistory: (string * string) list        // Transformation history for debugging
}

/// Pattern library utility functions
module PatternLibrary =
    let isResultReturningFunction (funcName: string) : bool =
        match funcName with
        | "readInto" | "readFile" | "parseInput" | "tryParse" -> true
        | _ -> funcName.Contains("OrNone") || 
               funcName.Contains("OrError") || 
               funcName.StartsWith("try") ||
               funcName.Contains("Result")
               
    let isVariableNamed (name: string) (expr: OakExpression) =
        match expr with
        | Variable n when n = name -> true
        | _ -> false
    
    let isApplicationOf (funcName: string) (expr: OakExpression) =
        match expr with
        | Application(Variable name, _) when name = funcName -> true
        | _ -> false
    
    let isResultReturningApplication (expr: OakExpression) =
        match expr with
        | Application(Variable name, _) when isResultReturningFunction name -> true
        | _ -> false
    
    let isResultMatchPattern (expr: OakExpression) =
        match expr with
        | Match(Application(Variable funcName, _), cases) when isResultReturningFunction funcName ->
            cases |> List.exists (fun (pattern, _) ->
                match pattern with
                | PatternConstructor("Ok", _) -> true
                | _ -> false) &&
            cases |> List.exists (fun (pattern, _) ->
                match pattern with
                | PatternConstructor("Error", _) -> true
                | _ -> false)
        | _ -> false
    
    let isStringFormatPattern (expr: OakExpression) =
        match expr with
        | Application(Variable name, [Literal(StringLiteral _); _]) 
            when name = "format" || name = "String.format" -> true
        | _ -> false
    
    /// Core library of pattern-based transformations
    let symbolPatternLibrary : SymbolPattern list = [
        // Stack allocation pattern
        {
            Name = "stack-buffer-pattern"
            Description = "Stack allocation of fixed-size buffer"
            QualifiedNamePattern = "Alloy.Memory.stackBuffer"
            OperationPattern = DialectOperation(
                MLIRDialect.MemRef, 
                "memref.alloca", 
                Map.ofList [("element_type", "i8")]
            )
            TypeSignature = ([MLIRTypes.createInteger 32], MLIRTypes.createMemRef (MLIRTypes.createInteger 8) [])
            AstPattern = fun expr ->
                match expr with
                | Application(Variable "stackBuffer", [_]) -> true
                | _ -> false
        }
        
        // NativePtr.stackalloc pattern
        {
            Name = "nativeptr-stackalloc-pattern"
            Description = "NativePtr.stackalloc with fixed size"
            QualifiedNamePattern = "NativePtr.stackalloc"
            OperationPattern = DialectOperation(
                MLIRDialect.MemRef, 
                "memref.alloca", 
                Map.ofList [("element_type", "i8")]
            )
            TypeSignature = ([MLIRTypes.createInteger 32], MLIRTypes.createMemRef (MLIRTypes.createInteger 8) [])
            AstPattern = fun expr ->
                match expr with
                | Application(Variable "NativePtr.stackalloc", [_]) -> true
                | _ -> false
        }
        
        // String format pattern
        {
            Name = "string-format-pattern"
            Description = "String formatting with format strings"
            QualifiedNamePattern = "Alloy.IO.String.format"
            OperationPattern = CompositePattern([
                ExternalCall("sprintf", Some "libc")
                CustomTransform("string_format", ["utf8_conversion"])
            ])
            TypeSignature = ([MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []; 
                            MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []],
                           MLIRTypes.createMemRef (MLIRTypes.createInteger 8) [])
            AstPattern = isStringFormatPattern
        }
        
        // Console writeLine pattern
        {
            Name = "console-writeline-pattern"
            Description = "Console output with writeLine"
            QualifiedNamePattern = "Alloy.IO.Console.writeLine"
            OperationPattern = ExternalCall("printf", Some "libc")
            TypeSignature = ([MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []], MLIRTypes.createVoid ())
            AstPattern = fun expr ->
                match expr with
                | Application(Variable "writeLine", [_]) -> true
                | _ -> false
        }
        
        // Console prompt pattern
        {
            Name = "console-prompt-pattern"
            Description = "Console prompt function"
            QualifiedNamePattern = "Alloy.IO.Console.prompt"
            OperationPattern = ExternalCall("printf", Some "libc")
            TypeSignature = ([MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []], MLIRTypes.createVoid ())
            AstPattern = fun expr ->
                match expr with
                | Application(Variable "prompt", [_]) -> true
                | _ -> false
        }
        
        // Result-returning readInto pattern
        {
            Name = "readinto-pattern"
            Description = "Console input with readInto returning Result"
            QualifiedNamePattern = "Alloy.IO.Console.readInto"
            OperationPattern = CompositePattern([
                ExternalCall("fgets", Some "libc")
                ExternalCall("strlen", Some "libc")
                CustomTransform("result_wrapper", ["success_check"])
            ])
            TypeSignature = ([MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []], MLIRTypes.createInteger 32)
            AstPattern = fun expr ->
                match expr with
                | Application(Variable "readInto", [_]) -> true
                | _ -> false
        }
        
        // Result match pattern for handling Ok/Error cases
        {
            Name = "result-match-pattern"
            Description = "Match expression for Result type with Ok/Error cases"
            QualifiedNamePattern = "Result.match"
            OperationPattern = CustomTransform("result_match", ["ok_handler"; "error_handler"])
            TypeSignature = ([MLIRTypes.createInteger 32], MLIRTypes.createMemRef (MLIRTypes.createInteger 8) [])
            AstPattern = isResultMatchPattern
        }
        
        // spanToString pattern
        {
            Name = "span-to-string-pattern"
            Description = "Convert Span to string with proper UTF8 handling"
            QualifiedNamePattern = "Alloy.Memory.spanToString"
            OperationPattern = CustomTransform("span_to_string", ["utf8_conversion"])
            TypeSignature = ([MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []], 
                           MLIRTypes.createMemRef (MLIRTypes.createInteger 8) [])
            AstPattern = fun expr ->
                match expr with
                | Application(Variable "spanToString", [_]) -> true
                | _ -> false
        }
        
        // Console.readLine pattern
        {
            Name = "console-readline-pattern"
            Description = "Read a line from the console into a buffer"
            QualifiedNamePattern = "Alloy.IO.Console.readLine"
            OperationPattern = CompositePattern([
                ExternalCall("fgets", Some "libc")
                ExternalCall("strlen", Some "libc")
            ])
            TypeSignature = ([MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []; MLIRTypes.createInteger 32],
                           MLIRTypes.createInteger 32)
            AstPattern = fun expr ->
                match expr with
                | Application(Variable "Console.readLine", [_; _]) -> true
                | _ -> false
        }
    ]
    
    let findPatternByName (qualifiedName: string) : SymbolPattern option =
        symbolPatternLibrary
        |> List.tryFind (fun pattern -> 
            pattern.QualifiedNamePattern = qualifiedName ||
            qualifiedName.EndsWith(pattern.QualifiedNamePattern))
    
    let findPatternByExpression (expr: OakExpression) : SymbolPattern option =
        symbolPatternLibrary
        |> List.tryFind (fun pattern -> pattern.AstPattern expr)

/// Symbol resolution functions
module SymbolResolution =
    let createEmptyState () : SymbolResolutionState = {
        SymbolsByQualified = Map.empty
        SymbolsByShort = Map.empty
        NamespaceMap = Map.empty
        ActiveNamespaces = []
        TypeRegistry = Map.empty
        ActiveSymbols = []
        PatternRegistry = PatternLibrary.symbolPatternLibrary
    }
    
    let registerSymbol (symbol: ResolvedSymbol) (state: SymbolResolutionState) : SymbolResolutionState =
        let namespaceSymbols = 
            match Map.tryFind symbol.Namespace state.NamespaceMap with
            | Some existing -> symbol.ShortName :: existing
            | None -> [symbol.ShortName]
            
        { state with
            SymbolsByQualified = Map.add symbol.QualifiedName symbol state.SymbolsByQualified
            SymbolsByShort = Map.add symbol.ShortName symbol state.SymbolsByShort
            NamespaceMap = Map.add symbol.Namespace namespaceSymbols state.NamespaceMap }
    
    let resolveQualified (qualifiedName: string) (state: SymbolResolutionState) : CompilerResult<ResolvedSymbol> =
        match Map.tryFind qualifiedName state.SymbolsByQualified with
        | Some symbol -> Success symbol
        | None -> 
            match PatternLibrary.findPatternByName qualifiedName with
            | Some pattern ->
                let symbol = {
                    QualifiedName = qualifiedName
                    ShortName = 
                        if qualifiedName.Contains(".") then
                            qualifiedName.Split([|'.'|], StringSplitOptions.RemoveEmptyEntries) |> Array.last
                        else qualifiedName
                    ParameterTypes = fst pattern.TypeSignature
                    ReturnType = snd pattern.TypeSignature
                    Operation = pattern.OperationPattern
                    Namespace = 
                        if qualifiedName.Contains(".") then
                            qualifiedName.Substring(0, qualifiedName.LastIndexOf('.'))
                        else ""
                    SourceLibrary = "Alloy"
                    RequiresExternal = 
                        match pattern.OperationPattern with
                        | ExternalCall(_, Some _) -> true
                        | CompositePattern patterns ->
                            patterns |> List.exists (function
                                | ExternalCall(_, Some _) -> true
                                | _ -> false)
                        | _ -> false
                }
                Success symbol
            | None ->
                CompilerFailure [ConversionError(
                    "qualified symbol resolution",
                    qualifiedName,
                    "resolved symbol",
                    sprintf "Symbol '%s' not found in qualified registry or pattern library" qualifiedName
                )]
    
    let resolveUnqualified (shortName: string) (state: SymbolResolutionState) : CompilerResult<ResolvedSymbol> =
        match Map.tryFind shortName state.SymbolsByShort with
        | Some symbol -> Success symbol
        | None ->
            let tryNamespaceResolution () =
                state.ActiveNamespaces
                |> List.tryPick (fun ns ->
                    let qualifiedAttempt = sprintf "%s.%s" ns shortName
                    Map.tryFind qualifiedAttempt state.SymbolsByQualified)
            
            match tryNamespaceResolution () with
            | Some symbol -> Success symbol
            | None ->
                let patternMatch = 
                    state.PatternRegistry
                    |> List.tryFind (fun pattern -> 
                        pattern.QualifiedNamePattern.EndsWith(shortName))
                
                match patternMatch with
                | Some pattern ->
                    let qualifiedName = 
                        if state.ActiveNamespaces.IsEmpty then shortName
                        else sprintf "%s.%s" state.ActiveNamespaces.[0] shortName
                    
                    let symbol = {
                        QualifiedName = qualifiedName
                        ShortName = shortName
                        ParameterTypes = fst pattern.TypeSignature
                        ReturnType = snd pattern.TypeSignature
                        Operation = pattern.OperationPattern
                        Namespace = if state.ActiveNamespaces.IsEmpty then "" else state.ActiveNamespaces.[0]
                        SourceLibrary = "Alloy"
                        RequiresExternal = 
                            match pattern.OperationPattern with
                            | ExternalCall(_, Some _) -> true
                            | CompositePattern patterns ->
                                patterns |> List.exists (function
                                    | ExternalCall(_, Some _) -> true
                                    | _ -> false)
                            | _ -> false
                    }
                    Success symbol
                | None ->
                    CompilerFailure [ConversionError(
                        "unqualified symbol resolution",
                        shortName,
                        "resolved symbol",
                        sprintf "Symbol '%s' not found in any active namespace or pattern library" shortName
                    )]
    
    let resolveSymbol (name: string) (state: SymbolResolutionState) : CompilerResult<ResolvedSymbol> =
        if name.Contains('.') then resolveQualified name state
        else resolveUnqualified name state

/// Alloy library symbol definitions
module AlloySymbols =
    // Memory management symbols
    let createMemorySymbols() : ResolvedSymbol list = [
        {
            QualifiedName = "Alloy.Memory.stackBuffer"
            ShortName = "stackBuffer"
            ParameterTypes = [MLIRTypes.createInteger 32]
            ReturnType = MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
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
            ParameterTypes = [MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []]
            ReturnType = MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
            Operation = CustomTransform("span_to_string", ["utf8_conversion"])
            Namespace = "Alloy.Memory"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Memory.stackalloc"
            ShortName = "stackalloc"
            ParameterTypes = [MLIRTypes.createInteger 32]
            ReturnType = MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
            Operation = DialectOperation(
                MLIRDialect.MemRef,
                "memref.alloca",
                Map.ofList [("element_type", "i8")]
            )
            Namespace = "Alloy.Memory"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
    ]
    
    // Console I/O symbols
    let createConsoleSymbols() : ResolvedSymbol list = [
        {
            QualifiedName = "Alloy.IO.Console.prompt"
            ShortName = "prompt"
            ParameterTypes = [MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []]
            ReturnType = MLIRTypes.createVoid ()
            Operation = ExternalCall("printf", Some "libc")
            Namespace = "Alloy.IO.Console"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
        {
            QualifiedName = "Alloy.IO.Console.readInto"
            ShortName = "readInto"
            ParameterTypes = [MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []]
            ReturnType = MLIRTypes.createInteger 32
            Operation = CompositePattern([
                ExternalCall("fgets", Some "libc")
                ExternalCall("strlen", Some "libc")
                CustomTransform("result_wrapper", ["success_check"])
            ])
            Namespace = "Alloy.IO.Console"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
        {
            QualifiedName = "Alloy.IO.Console.writeLine"
            ShortName = "writeLine"
            ParameterTypes = [MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []]
            ReturnType = MLIRTypes.createVoid ()
            Operation = ExternalCall("printf", Some "libc")
            Namespace = "Alloy.IO.Console"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
        {
            QualifiedName = "Alloy.IO.Console.readLine"
            ShortName = "readLine"
            ParameterTypes = [
                MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
                MLIRTypes.createInteger 32
            ]
            ReturnType = MLIRTypes.createInteger 32
            Operation = CompositePattern([
                ExternalCall("fgets", Some "libc")
                ExternalCall("strlen", Some "libc")
            ])
            Namespace = "Alloy.IO.Console"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
    ]
    
    // String manipulation symbols
    let createStringSymbols() : ResolvedSymbol list = [
        {
            QualifiedName = "Alloy.IO.String.format"
            ShortName = "format"
            ParameterTypes = [
                MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
                MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
            ] 
            ReturnType = MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
            Operation = ExternalCall("sprintf", Some "libc")
            Namespace = "Alloy.IO.String"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
        {
            QualifiedName = "Alloy.IO.String.concat"
            ShortName = "concat"
            ParameterTypes = [
                MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
                MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
            ]
            ReturnType = MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []
            Operation = ExternalCall("strcat", Some "libc")
            Namespace = "Alloy.IO.String"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
        {
            QualifiedName = "Alloy.IO.String.length"
            ShortName = "length"
            ParameterTypes = [MLIRTypes.createMemRef (MLIRTypes.createInteger 8) []]
            ReturnType = MLIRTypes.createInteger 32
            Operation = ExternalCall("strlen", Some "libc")
            Namespace = "Alloy.IO.String"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
    ]

/// Registry construction and management with pattern-based resolution
module RegistryConstruction =
    let buildAlloyRegistry() : CompilerResult<SymbolRegistry> =
        try
            let initialState = SymbolResolution.createEmptyState()
            
            // Gather all Alloy symbols
            let alloySymbols = 
                AlloySymbols.createMemorySymbols() @
                AlloySymbols.createConsoleSymbols() @
                AlloySymbols.createStringSymbols()
            
            // Register all symbols - fixed fold implementation
            let finalState = 
                (initialState, alloySymbols)
                ||> List.fold (fun state symbol -> SymbolResolution.registerSymbol symbol state)
            
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
                ResolutionHistory = [("initialization", "Alloy registry built with pattern support")]
            }
            
        with ex ->
            CompilerFailure [InternalError(
                "registry construction",
                "Failed to build Alloy registry with pattern support",
                Some ex.Message
            )]
            
    let withNamespaceContext (namespaces: string list) (registry: SymbolRegistry) : SymbolRegistry =
        { registry with 
            State = { registry.State with ActiveNamespaces = namespaces @ registry.State.ActiveNamespaces }
            ResolutionHistory = ("namespace_context", String.concat ", " namespaces) :: registry.ResolutionHistory
        }
    
    let trackActiveSymbol (symbolId: string) (registry: SymbolRegistry) : SymbolRegistry =
        { registry with 
            State = { registry.State with ActiveSymbols = symbolId :: registry.State.ActiveSymbols }
        }
    
    let resolveSymbolInRegistry (symbolName: string) (registry: SymbolRegistry) : CompilerResult<ResolvedSymbol * SymbolRegistry> =
        match SymbolResolution.resolveSymbol symbolName registry.State with
        | Success symbol ->
            let updatedRegistry = { 
                registry with 
                    ResolutionHistory = ("resolution", sprintf "%s -> %s" symbolName symbol.QualifiedName) :: registry.ResolutionHistory 
            }
            Success (symbol, updatedRegistry)
            
        | CompilerFailure errors ->
            let updatedRegistry = { 
                registry with 
                    ResolutionHistory = ("resolution_failed", sprintf "Failed to resolve %s" symbolName) :: registry.ResolutionHistory 
            }
            CompilerFailure errors

/// MLIR operation generation based on resolved symbols and patterns
module MLIRGeneration =
    let findActiveBuffer (activeSymbols: string list) : string =
        activeSymbols 
        |> List.tryFind (fun s -> s.Contains("buffer"))
        |> Option.defaultValue "%unknown_buffer"
    
    let validateArgumentTypes (symbol: ResolvedSymbol) (argSSAValues: string list) (argTypes: MLIRType list) : bool * string list =
        if argSSAValues.Length <> symbol.ParameterTypes.Length then
            (false, [sprintf "Expected %d arguments for %s, got %d" 
                    symbol.ParameterTypes.Length symbol.QualifiedName argSSAValues.Length])
        else
            let typeChecks = 
                List.zip3 argSSAValues argTypes symbol.ParameterTypes
                |> List.map (fun (argVal, actualType, expectedType) ->
                    if TypeAnalysis.canConvertTo actualType expectedType then
                        (true, "")
                    else
                        (false, sprintf "Type mismatch for %s: expected %s, got %s" 
                            argVal (mlirTypeToString expectedType) (mlirTypeToString actualType)))
            
            (List.forall fst typeChecks, typeChecks |> List.filter (not << fst) |> List.map snd)
    
    let generatePatternOperations (pattern: SymbolPattern) (args: string list) (resultId: string) : string list =
        match pattern.OperationPattern with
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
            let returnType = snd pattern.TypeSignature
            let typeStr = mlirTypeToString returnType
            
            [sprintf "    %s = %s.%s%s : %s%s" resultId dialectPrefix operation argStr typeStr attrStr]
            
        | ExternalCall(funcName, _) ->
            let paramTypes = fst pattern.TypeSignature
            let returnType = snd pattern.TypeSignature
            
            let paramTypeStrs = 
                if args.IsEmpty then ["void"]
                else paramTypes |> List.map mlirTypeToString
            
            let argStr = String.concat ", " args
            let typeStr = String.concat ", " paramTypeStrs
            let returnTypeStr = mlirTypeToString returnType
            
            [sprintf "    %s = func.call @%s(%s) : (%s) -> %s" 
                resultId funcName argStr typeStr returnTypeStr]
            
        | CompositePattern operations ->
            operations
            |> List.mapi (fun i operation ->
                let stepResultId = 
                    if i = operations.Length - 1 then resultId 
                    else sprintf "%s_step%d" resultId i
                
                let returnType = snd pattern.TypeSignature
                
                match operation with
                | ExternalCall(extFuncName, _) ->
                    let paramTypes = fst pattern.TypeSignature
                    let paramTypeStrs = 
                        if args.IsEmpty then ["void"]
                        else paramTypes |> List.map mlirTypeToString
                    
                    let argStr = String.concat ", " args
                    let typeStr = String.concat ", " paramTypeStrs
                    
                    sprintf "    %s = func.call @%s(%s) : (%s) -> %s" 
                        stepResultId extFuncName argStr typeStr (mlirTypeToString returnType)
                    
                | CustomTransform(transformName, _) ->
                    sprintf "    %s = composite.%s%d : %s" 
                        stepResultId transformName i (mlirTypeToString returnType)
                    
                | _ ->
                    sprintf "    %s = arith.constant %d : i32 // Step %d" 
                        stepResultId i i)
            
        | CustomTransform(transformName, _) ->
            match transformName with
            | "span_to_string" ->
                let bufferArg = if args.IsEmpty then "%unknown_buffer" else args.[0]
                [sprintf "    %s = memref.cast %s : memref<?xi8> to memref<?xi8>" resultId bufferArg]
                
            | "result_wrapper" ->
                [sprintf "    %s = arith.addi %s, 0x10000 : i32" 
                    resultId (if args.IsEmpty then "0" else args.[0])]
                
            | "result_match" ->
                [sprintf "    %s = arith.constant 0 : i32 // Result match handled by pattern system" resultId]
                
            | _ ->
                [sprintf "    %s = arith.constant 0 : i32 // Custom transform: %s" resultId transformName]

    let generateMLIROperation (symbol: ResolvedSymbol) (args: string list) (argTypes: MLIRType list)
                            (resultId: string) (registry: SymbolRegistry) : CompilerResult<string list * SymbolRegistry> =
        let patternMatch = 
            registry.State.PatternRegistry
            |> List.tryFind (fun pattern -> 
                pattern.QualifiedNamePattern = symbol.QualifiedName ||
                symbol.QualifiedName.EndsWith(pattern.QualifiedNamePattern))
        
        match patternMatch with
        | Some pattern -> Success (generatePatternOperations pattern args resultId, registry)
        | None ->
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
                Success ([operationStr], registry)
            
            | ExternalCall(funcName, _) ->
                let argStr = String.concat ", " args
                let paramTypes = symbol.ParameterTypes |> List.map mlirTypeToString |> String.concat ", "
                let returnType = mlirTypeToString symbol.ReturnType
                let callOp = sprintf "    %s = func.call @%s(%s) : (%s) -> %s" resultId funcName argStr paramTypes returnType
                Success ([callOp], registry)
            
            | CompositePattern operations ->
                let mutable resultOps = []
                let mutable tempRegistry = registry
                
                let operationResults =
                    operations |> List.mapi (fun i operation ->
                        match operation with
                        | ExternalCall(extFuncName, _) ->
                            let tempResultId = if i = operations.Length - 1 then resultId else sprintf "%s_step%d" resultId i
                            let paramTypes = if args.IsEmpty then "void" else String.concat ", " (symbol.ParameterTypes |> List.map mlirTypeToString)
                            sprintf "    %s = func.call @%s(%s) : (%s) -> %s" 
                                tempResultId extFuncName (String.concat ", " args)
                                paramTypes (mlirTypeToString symbol.ReturnType)
                            
                        | CustomTransform(transformName, _) ->
                            let tempResultId = if i = operations.Length - 1 then resultId else sprintf "%s_step%d" resultId i
                            sprintf "    %s = composite.%s%d : %s" 
                                tempResultId transformName i (mlirTypeToString symbol.ReturnType)
                            
                        | _ ->
                            let tempResultId = if i = operations.Length - 1 then resultId else sprintf "%s_step%d" resultId i
                            sprintf "    %s = arith.constant %d : i32 // Placeholder for operation %d" 
                                tempResultId i i)
                
                Success (operationResults, tempRegistry)
                
            | CustomTransform(transformName, _) ->
                match transformName with
                | "span_to_string" ->
                    let bufferValue = 
                        if args.IsEmpty then findActiveBuffer registry.State.ActiveSymbols
                        else args.[0]
                    
                    let bitcastOp = sprintf "    %s = memref.cast %s : memref<?xi8> to memref<?xi8>" resultId bufferValue
                    Success ([bitcastOp], registry)
                    
                | _ ->
                    let constOp = sprintf "    %s = arith.constant 0 : i32" resultId
                    Success ([constOp], registry)

/// Type-aware external interface for integration with existing MLIR generation
module PublicInterface =
    let createStandardRegistry() : CompilerResult<SymbolRegistry> =
        RegistryConstruction.buildAlloyRegistry()
    
    let getSymbolType (funcName: string) (registry: SymbolRegistry) : MLIRType option =
        match RegistryConstruction.resolveSymbolInRegistry funcName registry with
        | Success (symbol, _) -> Some symbol.ReturnType
        | CompilerFailure _ ->
            registry.State.PatternRegistry
            |> List.tryFind (fun pattern -> 
                pattern.QualifiedNamePattern = funcName ||
                funcName.EndsWith(pattern.QualifiedNamePattern))
            |> Option.map (fun pattern -> snd pattern.TypeSignature)
    
    let resolveFunctionCall (funcName: string) (args: string list) (resultId: string) (registry: SymbolRegistry)
                           : CompilerResult<string list * SymbolRegistry> =
        let patternMatch = 
            registry.State.PatternRegistry
            |> List.tryFind (fun pattern -> 
                pattern.QualifiedNamePattern = funcName ||
                funcName.EndsWith(pattern.QualifiedNamePattern))
        
        match patternMatch with
        | Some pattern -> Success (MLIRGeneration.generatePatternOperations pattern args resultId, registry)
        | None ->
            match RegistryConstruction.resolveSymbolInRegistry funcName registry with
            | Success (symbol, updatedRegistry) ->
                let argTypes = 
                    args |> List.mapi (fun i _ ->
                        if i < symbol.ParameterTypes.Length then symbol.ParameterTypes.[i]
                        else MLIRTypes.createInteger 32)
                
                let (typesValid, typeErrors) = MLIRGeneration.validateArgumentTypes symbol args argTypes
                
                if not typesValid then
                    let errorMsg = sprintf "Type validation failed for %s: %s" funcName (String.concat "; " typeErrors)
                    
                    let registry2 = { 
                        updatedRegistry with 
                            ResolutionHistory = ("type_warning", errorMsg) :: updatedRegistry.ResolutionHistory 
                    }
                    
                    MLIRGeneration.generateMLIROperation symbol args argTypes resultId registry2
                else
                    MLIRGeneration.generateMLIROperation symbol args argTypes resultId updatedRegistry
                
            | CompilerFailure _ -> 
                let defaultOp = sprintf "    %s = arith.constant 0 : i32 // Unknown function: %s" resultId funcName
                let updatedRegistry = { 
                    registry with 
                        ResolutionHistory = ("resolution_fallback", sprintf "Created fallback for %s" funcName) :: registry.ResolutionHistory 
                }
                Success ([defaultOp], updatedRegistry)

    let getNamespaceSymbols (namespaceName: string) (registry: SymbolRegistry) : string list =
        match Map.tryFind namespaceName registry.State.NamespaceMap with
        | Some symbols -> symbols
        | None -> []
    
    let validateRequiredSymbols (requiredSymbols: string list) (registry: SymbolRegistry) : CompilerResult<unit> =
        let missingSymbols = 
            requiredSymbols
            |> List.filter (fun symbolName ->
                match SymbolResolution.resolveSymbol symbolName registry.State with
                | Success _ -> false
                | CompilerFailure _ -> 
                    not (registry.State.PatternRegistry
                         |> List.exists (fun pattern -> 
                             pattern.QualifiedNamePattern = symbolName ||
                             symbolName.EndsWith(pattern.QualifiedNamePattern))))
        
        if missingSymbols.IsEmpty then Success ()
        else
            CompilerFailure [ConversionError(
                "symbol validation",
                String.concat ", " missingSymbols,
                "available symbols",
                sprintf "Missing required symbols: %s" (String.concat ", " missingSymbols)
            )]
            
    let getParameterTypes (funcName: string) (registry: SymbolRegistry) : MLIRType list option =
        match RegistryConstruction.resolveSymbolInRegistry funcName registry with
        | Success (symbol, _) -> Some symbol.ParameterTypes
        | CompilerFailure _ -> 
            registry.State.PatternRegistry
            |> List.tryFind (fun pattern -> 
                pattern.QualifiedNamePattern = funcName ||
                funcName.EndsWith(pattern.QualifiedNamePattern))
            |> Option.map (fun pattern -> fst pattern.TypeSignature)
        
    let areArgumentTypesCompatible (funcName: string) (argTypes: MLIRType list) (registry: SymbolRegistry) : bool =
        match getParameterTypes funcName registry with
        | Some paramTypes ->
            paramTypes.Length = argTypes.Length &&
            List.zip paramTypes argTypes
            |> List.forall (fun (paramType, argType) -> TypeAnalysis.canConvertTo argType paramType)
        | None -> false
    
    let findPatternByExpression (expr: OakExpression) (registry: SymbolRegistry) : SymbolPattern option =
        registry.State.PatternRegistry
        |> List.tryFind (fun pattern -> pattern.AstPattern expr)