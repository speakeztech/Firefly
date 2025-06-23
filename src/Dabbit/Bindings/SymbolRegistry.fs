module Dabbit.Bindings.SymbolRegistry

open System
open FSharp.Compiler.Syntax
open Core.XParsec.Foundation
open Core.MLIRGeneration.TypeSystem
open Core.MLIRGeneration.Dialect
open Dabbit.Bindings.PatternLibrary

/// Symbol with type safety and MLIR generation info
type ResolvedSymbol = {
    QualifiedName: string
    ShortName: string
    ParameterTypes: MLIRType list
    ReturnType: MLIRType
    Operation: MLIROperationPattern
    Namespace: string
    SourceLibrary: string
    RequiresExternal: bool
}

/// Symbol resolution state
type SymbolResolutionState = {
    SymbolsByQualified: Map<string, ResolvedSymbol>
    SymbolsByShort: Map<string, ResolvedSymbol>
    NamespaceMap: Map<string, string list>
    ActiveNamespaces: string list
    TypeRegistry: Map<string, MLIRType>
    ActiveSymbols: string list
    PatternRegistry: SymbolPattern list
}

/// The complete symbol registry
type SymbolRegistry = {
    State: SymbolResolutionState
    ResolutionHistory: (string * string) list
}

/// Symbol resolution functions
module SymbolResolution =
    let createEmptyState () : SymbolResolutionState = {
        SymbolsByQualified = Map.empty
        SymbolsByShort = Map.empty
        NamespaceMap = Map.empty
        ActiveNamespaces = []
        TypeRegistry = Map.empty
        ActiveSymbols = []
        PatternRegistry = alloyPatterns
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
            match findByName qualifiedName with
            | Some pattern ->
                let symbol = {
                    QualifiedName = qualifiedName
                    ShortName = 
                        if qualifiedName.Contains(".") then
                            qualifiedName.Split([|'.'|], StringSplitOptions.RemoveEmptyEntries) |> Array.last
                        else qualifiedName
                    ParameterTypes = fst pattern.TypeSig
                    ReturnType = snd pattern.TypeSig
                    Operation = pattern.OpPattern
                    Namespace = 
                        if qualifiedName.Contains(".") then
                            qualifiedName.Substring(0, qualifiedName.LastIndexOf('.'))
                        else ""
                    SourceLibrary = "Alloy"
                    RequiresExternal = 
                        match pattern.OpPattern with
                        | ExternalCall(_, Some _) -> true
                        | Composite patterns ->
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
                        pattern.QualifiedName.EndsWith(shortName))
                
                match patternMatch with
                | Some pattern ->
                    let qualifiedName = 
                        if state.ActiveNamespaces.IsEmpty then shortName
                        else sprintf "%s.%s" state.ActiveNamespaces.[0] shortName
                    
                    let symbol = {
                        QualifiedName = qualifiedName
                        ShortName = shortName
                        ParameterTypes = fst pattern.TypeSig
                        ReturnType = snd pattern.TypeSig
                        Operation = pattern.OpPattern
                        Namespace = if state.ActiveNamespaces.IsEmpty then "" else state.ActiveNamespaces.[0]
                        SourceLibrary = "Alloy"
                        RequiresExternal = 
                            match pattern.OpPattern with
                            | ExternalCall(_, Some _) -> true
                            | Composite patterns ->
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
            ParameterTypes = [MLIRTypes.i32]
            ReturnType = MLIRTypes.memref MLIRTypes.i8
            Operation = DialectOp(MemRef, "memref.alloca", Map["element_type", "i8"])
            Namespace = "Alloy.Memory"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Memory.spanToString"
            ShortName = "spanToString"
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i8]
            ReturnType = MLIRTypes.memref MLIRTypes.i8
            Operation = Transform("span_to_string", ["utf8_conversion"])
            Namespace = "Alloy.Memory"
            SourceLibrary = "Alloy"
            RequiresExternal = false
        }
        {
            QualifiedName = "Alloy.Memory.stackalloc"
            ShortName = "stackalloc"
            ParameterTypes = [MLIRTypes.i32]
            ReturnType = MLIRTypes.memref MLIRTypes.i8
            Operation = DialectOp(MemRef, "memref.alloca", Map["element_type", "i8"])
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
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i8]
            ReturnType = MLIRTypes.void_
            Operation = ExternalCall("printf", Some "libc")
            Namespace = "Alloy.IO.Console"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
        {
            QualifiedName = "Alloy.IO.Console.readInto"
            ShortName = "readInto"
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i8]
            ReturnType = MLIRTypes.i32
            Operation = Composite [
                ExternalCall("fgets", Some "libc")
                ExternalCall("strlen", Some "libc")
                Transform("result_wrapper", ["success_check"])
            ]
            Namespace = "Alloy.IO.Console"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
        {
            QualifiedName = "Alloy.IO.Console.writeLine"
            ShortName = "writeLine"
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i8]
            ReturnType = MLIRTypes.void_
            Operation = ExternalCall("printf", Some "libc")
            Namespace = "Alloy.IO.Console"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
        {
            QualifiedName = "Alloy.IO.Console.readLine"
            ShortName = "readLine"
            ParameterTypes = [
                MLIRTypes.memref MLIRTypes.i8
                MLIRTypes.i32
            ]
            ReturnType = MLIRTypes.i32
            Operation = Composite [
                ExternalCall("fgets", Some "libc")
                ExternalCall("strlen", Some "libc")
            ]
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
                MLIRTypes.memref MLIRTypes.i8
                MLIRTypes.memref MLIRTypes.i8
            ] 
            ReturnType = MLIRTypes.memref MLIRTypes.i8
            Operation = ExternalCall("sprintf", Some "libc")
            Namespace = "Alloy.IO.String"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
        {
            QualifiedName = "Alloy.IO.String.concat"
            ShortName = "concat"
            ParameterTypes = [
                MLIRTypes.memref MLIRTypes.i8
                MLIRTypes.memref MLIRTypes.i8
            ]
            ReturnType = MLIRTypes.memref MLIRTypes.i8
            Operation = ExternalCall("strcat", Some "libc")
            Namespace = "Alloy.IO.String"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
        {
            QualifiedName = "Alloy.IO.String.length"
            ShortName = "length"
            ParameterTypes = [MLIRTypes.memref MLIRTypes.i8]
            ReturnType = MLIRTypes.i32
            Operation = ExternalCall("strlen", Some "libc")
            Namespace = "Alloy.IO.String"
            SourceLibrary = "Alloy"
            RequiresExternal = true
        }
    ]

/// Registry construction and management
module RegistryConstruction =
    let buildAlloyRegistry() : CompilerResult<SymbolRegistry> =
        try
            let initialState = SymbolResolution.createEmptyState()
            
            // Gather all Alloy symbols
            let alloySymbols = 
                AlloySymbols.createMemorySymbols() @
                AlloySymbols.createConsoleSymbols() @
                AlloySymbols.createStringSymbols()
            
            // Register all symbols
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
        match pattern.OpPattern with
        | DialectOp(dialect, operation, attributes) ->
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
            let returnType = snd pattern.TypeSig
            let typeStr = mlirTypeToString returnType
            
            [sprintf "    %s = %s.%s%s : %s%s" resultId dialectPrefix operation argStr typeStr attrStr]
            
        | ExternalCall(funcName, _) ->
            let paramTypes = fst pattern.TypeSig
            let returnType = snd pattern.TypeSig
            
            let paramTypeStrs = 
                if args.IsEmpty then ["void"]
                else paramTypes |> List.map mlirTypeToString
            
            let argStr = String.concat ", " args
            let typeStr = String.concat ", " paramTypeStrs
            let returnTypeStr = mlirTypeToString returnType
            
            [sprintf "    %s = func.call @%s(%s) : (%s) -> %s" 
                resultId funcName argStr typeStr returnTypeStr]
            
        | Composite operations ->
            operations
            |> List.mapi (fun i operation ->
                let stepResultId = 
                    if i = operations.Length - 1 then resultId 
                    else sprintf "%s_step%d" resultId i
                
                let returnType = snd pattern.TypeSig
                
                match operation with
                | ExternalCall(extFuncName, _) ->
                    let paramTypes = fst pattern.TypeSig
                    let paramTypeStrs = 
                        if args.IsEmpty then ["void"]
                        else paramTypes |> List.map mlirTypeToString
                    
                    let argStr = String.concat ", " args
                    let typeStr = String.concat ", " paramTypeStrs
                    
                    sprintf "    %s = func.call @%s(%s) : (%s) -> %s" 
                        stepResultId extFuncName argStr typeStr (mlirTypeToString returnType)
                    
                | Transform(transformName, _) ->
                    sprintf "    %s = composite.%s%d : %s" 
                        stepResultId transformName i (mlirTypeToString returnType)
                    
                | _ ->
                    sprintf "    %s = arith.constant %d : i32 // Step %d" 
                        stepResultId i i)
            
        | Transform(transformName, _) ->
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
                pattern.QualifiedName = symbol.QualifiedName ||
                symbol.QualifiedName.EndsWith(pattern.QualifiedName))
        
        match patternMatch with
        | Some pattern -> Success (generatePatternOperations pattern args resultId, registry)
        | None ->
            match symbol.Operation with
            | DialectOp(dialect, operation, attributes) ->
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
            
            | Composite operations ->
                let operationResults =
                    operations |> List.mapi (fun i operation ->
                        match operation with
                        | ExternalCall(extFuncName, _) ->
                            let tempResultId = if i = operations.Length - 1 then resultId else sprintf "%s_step%d" resultId i
                            let paramTypes = if args.IsEmpty then "void" else String.concat ", " (symbol.ParameterTypes |> List.map mlirTypeToString)
                            sprintf "    %s = func.call @%s(%s) : (%s) -> %s" 
                                tempResultId extFuncName (String.concat ", " args)
                                paramTypes (mlirTypeToString symbol.ReturnType)
                            
                        | Transform(transformName, _) ->
                            let tempResultId = if i = operations.Length - 1 then resultId else sprintf "%s_step%d" resultId i
                            sprintf "    %s = composite.%s%d : %s" 
                                tempResultId transformName i (mlirTypeToString symbol.ReturnType)
                            
                        | _ ->
                            let tempResultId = if i = operations.Length - 1 then resultId else sprintf "%s_step%d" resultId i
                            sprintf "    %s = arith.constant %d : i32 // Placeholder for operation %d" 
                                tempResultId i i)
                
                Success (operationResults, registry)
                
            | Transform(transformName, _) ->
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
                pattern.QualifiedName = funcName ||
                funcName.EndsWith(pattern.QualifiedName))
            |> Option.map (fun pattern -> snd pattern.TypeSig)
    
    let resolveFunctionCall (funcName: string) (args: string list) (resultId: string) (registry: SymbolRegistry)
                           : CompilerResult<string list * SymbolRegistry> =
        let patternMatch = 
            registry.State.PatternRegistry
            |> List.tryFind (fun pattern -> 
                pattern.QualifiedName = funcName ||
                funcName.EndsWith(pattern.QualifiedName))
        
        match patternMatch with
        | Some pattern -> Success (MLIRGeneration.generatePatternOperations pattern args resultId, registry)
        | None ->
            match RegistryConstruction.resolveSymbolInRegistry funcName registry with
            | Success (symbol, updatedRegistry) ->
                let argTypes = 
                    args |> List.mapi (fun i _ ->
                        if i < symbol.ParameterTypes.Length then symbol.ParameterTypes.[i]
                        else MLIRTypes.i32)
                
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
                             pattern.QualifiedName = symbolName ||
                             symbolName.EndsWith(pattern.QualifiedName))))
        
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
                pattern.QualifiedName = funcName ||
                funcName.EndsWith(pattern.QualifiedName))
            |> Option.map (fun pattern -> fst pattern.TypeSig)
        
    let areArgumentTypesCompatible (funcName: string) (argTypes: MLIRType list) (registry: SymbolRegistry) : bool =
        match getParameterTypes funcName registry with
        | Some paramTypes ->
            paramTypes.Length = argTypes.Length &&
            List.zip paramTypes argTypes
            |> List.forall (fun (paramType, argType) -> TypeAnalysis.canConvertTo argType paramType)
        | None -> false
    
    let findPatternByExpression (expr: SynExpr) (registry: SymbolRegistry) : SymbolPattern option =
        registry.State.PatternRegistry
        |> List.tryFind (fun pattern -> pattern.Matcher expr)