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
        if name.Contains('.') then
            resolveQualified name state
        else
            resolveUnqualified name state

/// Type analysis functions
module TypeAnalysis =
    let rec canConvertTo (fromType: MLIRType) (toType: MLIRType) : bool =
        match fromType.Category, toType.Category with
        | MLIRTypeCategory.Integer, MLIRTypeCategory.Integer -> 
            fromType.Width <= toType.Width
        | MLIRTypeCategory.Float, MLIRTypeCategory.Float -> 
            fromType.Width <= toType.Width
        | MLIRTypeCategory.MemRef, MLIRTypeCategory.MemRef ->
            match fromType.ElementType, toType.ElementType with
            | Some f, Some t -> canConvertTo f t
            | _ -> false
        | _, _ -> fromType = toType
    
    let inferType (expr: SynExpr) : MLIRType option =
        match expr with
        | SynExpr.Const(SynConst.Int32 _, _) -> Some MLIRTypes.i32
        | SynExpr.Const(SynConst.String _, _) -> Some (MLIRTypes.memref MLIRTypes.i8)
        | _ -> None

/// Registry construction and management
module RegistryConstruction =
    let createInitialRegistry () : SymbolRegistry =
        let state = SymbolResolution.createEmptyState ()
        
        let alloySymbols = [
            { QualifiedName = "Alloy.Memory.stackBuffer"
              ShortName = "stackBuffer"
              ParameterTypes = [MLIRTypes.i32]
              ReturnType = MLIRTypes.memref MLIRTypes.i8
              Operation = DialectOp(MemRef, "alloca", Map["element_type", "i8"])
              Namespace = "Alloy.Memory"
              SourceLibrary = "Alloy"
              RequiresExternal = false }
            
            { QualifiedName = "Alloy.IO.String.format"
              ShortName = "format"
              ParameterTypes = [MLIRTypes.memref MLIRTypes.i8; MLIRTypes.memref MLIRTypes.i8]
              ReturnType = MLIRTypes.memref MLIRTypes.i8
              Operation = ExternalCall("sprintf", Some "libc")
              Namespace = "Alloy.IO.String"
              SourceLibrary = "Alloy"
              RequiresExternal = true }
            
            { QualifiedName = "Alloy.IO.Console.writeLine"
              ShortName = "writeLine"
              ParameterTypes = [MLIRTypes.memref MLIRTypes.i8]
              ReturnType = MLIRTypes.void_
              Operation = ExternalCall("printf", Some "libc")
              Namespace = "Alloy.IO.Console"
              SourceLibrary = "Alloy"
              RequiresExternal = true }
            
            { QualifiedName = "NativePtr.stackalloc"
              ShortName = "stackalloc"
              ParameterTypes = [MLIRTypes.i32]
              ReturnType = MLIRTypes.memref MLIRTypes.i8
              Operation = DialectOp(MemRef, "alloca", Map["element_type", "i8"])
              Namespace = "NativePtr"
              SourceLibrary = "Alloy"
              RequiresExternal = false }
            
            { QualifiedName = "Result.isOk"
              ShortName = "isOk"
              ParameterTypes = [MLIRTypes.i32]
              ReturnType = MLIRTypes.i1
              Operation = DialectOp(Arith, "cmpi", Map["predicate", "sge"])
              Namespace = "Result"
              SourceLibrary = "Alloy"
              RequiresExternal = false }
        ]
        
        let stateWithSymbols = 
            alloySymbols 
            |> List.fold (fun acc sym -> SymbolResolution.registerSymbol sym acc) state
        
        { State = stateWithSymbols; ResolutionHistory = [] }
    
    let buildAlloyRegistry () : CompilerResult<SymbolRegistry> =
        try
            let baseRegistry = createInitialRegistry ()
            
            let stateWithPatterns = {
                baseRegistry.State with 
                    PatternRegistry = alloyPatterns
            }
            
            Success { 
                State = stateWithPatterns
                ResolutionHistory = [("initialization", "Registry created with Alloy symbols and patterns")]
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

/// Symbol-specific MLIR operation generation from registry patterns
/// This module handles generation of MLIR operations specifically from symbols
/// and patterns in the registry, as opposed to Core.MLIRGeneration which
/// provides general MLIR generation infrastructure
module SymbolOperations =
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
    
    let rec generatePatternOperations (pattern: SymbolPattern) (args: string list) (resultId: string) : string list =
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
            let (paramTypes, returnType) = pattern.TypeSig
            let typeStr = mlirTypeToString returnType
            
            [sprintf "    %s = %s.%s%s : %s%s" resultId dialectPrefix operation argStr typeStr attrStr]
            
        | ExternalCall(funcName, _) ->
            let (paramTypes, returnType) = pattern.TypeSig
            
            let paramTypeStr = 
                if paramTypes.IsEmpty then ""
                else sprintf "(%s)" (paramTypes |> List.map mlirTypeToString |> String.concat ", ")
            
            let returnTypeStr = mlirTypeToString returnType
            
            [sprintf "    %s = func.call @%s(%s) : %s -> %s" 
                resultId funcName (String.concat ", " args) paramTypeStr returnTypeStr]
                
        | Composite operations ->
            operations |> List.collect (fun op ->
                let tempPattern = { pattern with OpPattern = op }
                generatePatternOperations tempPattern args resultId)
                
        | Transform(transformName, parameters) ->
            match transformName with
            | "span_conversion" ->
                let bufferArg = if args.IsEmpty then "%0" else args.[0]
                [sprintf "    %s = memref.cast %s : memref<?xi8> to memref<?xi8>" resultId bufferArg]
                
            | "result_wrapper" ->
                [sprintf "    %s = arith.addi %s, 0x10000 : i32" 
                    resultId (if args.IsEmpty then "0" else args.[0])]
                
            | "result_match" ->
                [sprintf "    %s = arith.constant 0 : i32 // Result match handled by pattern system" resultId]
                
            | _ ->
                [sprintf "    %s = arith.constant 0 : i32 // Custom transform: %s" resultId transformName]

    let rec generateMLIROperation (symbol: ResolvedSymbol) (args: string list) (argTypes: MLIRType list)
                                (resultId: string) (registry: SymbolRegistry) : CompilerResult<string list * SymbolRegistry> =
        let patternMatch = 
            registry.State.PatternRegistry
            |> List.tryFind (fun pattern -> 
                pattern.QualifiedName = symbol.QualifiedName ||
                symbol.QualifiedName.EndsWith(pattern.QualifiedName))
        
        match patternMatch with
        | Some pattern -> 
            Success (generatePatternOperations pattern args resultId, registry)
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
                let updatedRegistry = { 
                    registry with 
                        State = { registry.State with ActiveSymbols = resultId :: registry.State.ActiveSymbols } 
                }
                
                let paramTypeStr = 
                    if argTypes.IsEmpty then ""
                    else sprintf "(%s)" (argTypes |> List.map mlirTypeToString |> String.concat ", ")
                
                let returnTypeStr = mlirTypeToString symbol.ReturnType
                
                let callOp = 
                    sprintf "    %s = func.call @%s(%s) : %s -> %s" 
                        resultId funcName (String.concat ", " args) paramTypeStr returnTypeStr
                    
                Success ([callOp], updatedRegistry)
                
            | Composite operations ->
                let foldResult = 
                    operations |> List.fold (fun acc op ->
                        match acc with
                        | CompilerFailure _ -> acc
                        | Success (ops, reg) ->
                            let tempSymbol = { symbol with Operation = op }
                            match generateMLIROperation tempSymbol args argTypes resultId reg with
                            | Success (newOps, newReg) -> Success (ops @ newOps, newReg)
                            | CompilerFailure errors -> CompilerFailure errors
                    ) (Success ([], registry))
                foldResult
                
            | Transform(transformName, _) ->
                match transformName with
                | "cast_to_span" ->
                    let bufferValue = findActiveBuffer registry.State.ActiveSymbols
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
        | Some pattern -> Success (SymbolOperations.generatePatternOperations pattern args resultId, registry)
        | None ->
            match RegistryConstruction.resolveSymbolInRegistry funcName registry with
            | Success (symbol, updatedRegistry) ->
                let argTypes = 
                    args |> List.mapi (fun i _ ->
                        if i < symbol.ParameterTypes.Length then symbol.ParameterTypes.[i]
                        else MLIRTypes.i32)
                
                let (typesValid, typeErrors) = SymbolOperations.validateArgumentTypes symbol args argTypes
                
                if not typesValid then
                    let errorMsg = sprintf "Type validation failed for %s: %s" funcName (String.concat "; " typeErrors)
                    
                    let registry2 = { 
                        updatedRegistry with 
                            ResolutionHistory = ("type_warning", errorMsg) :: updatedRegistry.ResolutionHistory 
                    }
                    
                    SymbolOperations.generateMLIROperation symbol args argTypes resultId registry2
                else
                    SymbolOperations.generateMLIROperation symbol args argTypes resultId updatedRegistry
                
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