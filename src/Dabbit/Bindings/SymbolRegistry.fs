module Dabbit.Bindings.SymbolRegistry

open System
open FSharp.Compiler.Syntax
open Core.XParsec.Foundation
open Core.Types.Dialects
open Core.Types.TypeSystem
open Core.Types.MLIRContext
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
                        else "Global"
                    SourceLibrary = "alloy"
                    RequiresExternal = 
                        match pattern.OpPattern with
                        | ExternalCall(_, Some _) -> true
                        | _ -> false
                }
                
                Success symbol
            | None -> CompilerFailure [ConversionError("symbol_resolution", qualifiedName, "unknown", "Could not find symbol")]
    
    let resolveFromNamespace (symbolName: string) (namespace': string) (state: SymbolResolutionState) : CompilerResult<ResolvedSymbol> =
        let qualifiedName = sprintf "%s.%s" namespace' symbolName
        resolveQualified qualifiedName state
    
    let resolveSymbol (symbolName: string) (state: SymbolResolutionState) : CompilerResult<ResolvedSymbol> =
        // First try to find by short name directly
        match Map.tryFind symbolName state.SymbolsByShort with
        | Some symbol -> Success symbol
        | None ->
            // Next try resolving as a qualified name
            if symbolName.Contains(".") then
                resolveQualified symbolName state
            else
                // Try all active namespaces
                let rec tryNamespaces (namespaces: string list) =
                    match namespaces with
                    | [] -> 
                        // Last resort, try to find pattern by name
                        match findByName symbolName with
                        | Some pattern ->
                            let symbol = {
                                QualifiedName = symbolName
                                ShortName = symbolName
                                ParameterTypes = fst pattern.TypeSig
                                ReturnType = snd pattern.TypeSig
                                Operation = pattern.OpPattern
                                Namespace = "Global"
                                SourceLibrary = "alloy"
                                RequiresExternal = 
                                    match pattern.OpPattern with
                                    | ExternalCall(_, Some _) -> true
                                    | _ -> false
                            }
                            Success symbol
                        | None -> 
                            CompilerFailure [ConversionError("symbol_resolution", symbolName, "active_namespaces", 
                                                 sprintf "Could not find symbol in active namespaces: %s" 
                                                     (String.concat ", " state.ActiveNamespaces))]
                    | ns :: rest ->
                        match resolveFromNamespace symbolName ns state with
                        | Success symbol -> Success symbol
                        | CompilerFailure _ -> tryNamespaces rest
                
                tryNamespaces state.ActiveNamespaces
    
    let addNamespace (namespace': string) (state: SymbolResolutionState) : SymbolResolutionState =
        { state with ActiveNamespaces = namespace' :: state.ActiveNamespaces |> List.distinct }
        
    let addNamespaces (namespaces: string list) (state: SymbolResolutionState) : SymbolResolutionState =
        let updatedNamespaces = List.append namespaces state.ActiveNamespaces |> List.distinct
        { state with ActiveNamespaces = updatedNamespaces }
    
    let createEmptyRegistry () : SymbolRegistry = {
        State = createEmptyState()
        ResolutionHistory = []
    }

/// Symbol operations for code generation
module SymbolOperations =
    let getBitWidth (t: MLIRType) : int =
        match t.BitWidth with
        | Some width -> width
        | None -> 32  // Default to 32-bit integers
    
    let generateFunctionCall (funcName: string) (args: string list) (resultId: string) : string =
        sprintf "    %s = call @%s(%s) : () -> ()" resultId funcName (String.concat ", " args)
    
    let generateDialectCall (dialect: MLIRDialect) (operation: string) (args: string list) (resultType: string) (resultId: string) : string =
        let argsStr = if args.IsEmpty then "" else String.concat ", " args
        sprintf "    %s = %s.%s %s : %s" resultId (dialectToString dialect) operation argsStr resultType
    
    let validateArgumentTypes (symbol: ResolvedSymbol) (args: string list) (argTypes: MLIRType list) : bool * string list =
        if symbol.ParameterTypes.Length <> argTypes.Length then
            false, [sprintf "Expected %d arguments, got %d" symbol.ParameterTypes.Length argTypes.Length]
        else
            let typeResults = 
                List.zip symbol.ParameterTypes argTypes
                |> List.mapi (fun i (expected, actual) ->
                    if expected.Category = actual.Category then
                        match expected.Category with
                        | Integer | Float ->
                            let expectedWidth = getBitWidth expected
                            let actualWidth = getBitWidth actual
                            
                            // For numeric types, we need to check bit width compatibility
                            if expectedWidth = actualWidth then
                                true, None
                            else
                                false, Some (sprintf "Argument %d: expected %d bits, got %d bits" 
                                                (i + 1) expectedWidth actualWidth)
                        | _ -> 
                            // For other types, we just check category equality
                            true, None
                    else
                        false, Some (sprintf "Argument %d: expected %s, got %s" 
                                        (i + 1) (mlirTypeToString expected) (mlirTypeToString actual))
                )
            
            let isValid = typeResults |> List.forall (fun (valid, _) -> valid)
            let errors = typeResults |> List.choose (fun (_, err) -> err)
            
            isValid, errors
    
    let generateMLIROperation (symbol: ResolvedSymbol) (args: string list) (argTypes: MLIRType list) (resultId: string) (registry: SymbolRegistry) : CompilerResult<string list * SymbolRegistry> =
        match symbol.Operation with
        | DialectOp (dialect, operation, _) ->
            let resultTypeStr = mlirTypeToString symbol.ReturnType
            let op = generateDialectCall dialect operation args resultTypeStr resultId
            let updatedRegistry = { 
                registry with 
                    ResolutionHistory = ("direct_operation", sprintf "Generated %s.%s" (dialectToString dialect) operation) :: registry.ResolutionHistory 
            }
            Success ([op], updatedRegistry)
            
        | ExternalCall (funcName, _) ->
            let op = generateFunctionCall funcName args resultId
            let updatedRegistry = { 
                registry with 
                    ResolutionHistory = ("external_call", sprintf "Called external function %s" funcName) :: registry.ResolutionHistory 
            }
            Success ([op], updatedRegistry)
            
        | Composite operations ->
            // Handle a list of operations
            let errorMsg = "Composite operations not yet fully implemented"
            let updatedRegistry = { 
                registry with 
                    ResolutionHistory = ("warning", errorMsg) :: registry.ResolutionHistory 
            }
            let fallbackOp = sprintf "    %s = arith.constant 0 : i32 // Warning: %s" resultId errorMsg
            Success ([fallbackOp], updatedRegistry)
            
        | Transform (name, params') ->
            let op = sprintf "    %s = transform.%s(%s) : %s" resultId name (String.concat ", " params') (mlirTypeToString symbol.ReturnType)
            let updatedRegistry = { 
                registry with 
                    ResolutionHistory = ("transform", sprintf "Applied transform %s" name) :: registry.ResolutionHistory 
            }
            Success ([op], updatedRegistry)

/// Registry operations
module Registry =
    let createRegistry (initialPatterns: SymbolPattern list) : SymbolRegistry =
        let state = {
            SymbolsByQualified = Map.empty
            SymbolsByShort = Map.empty
            NamespaceMap = Map.empty
            ActiveNamespaces = ["Alloy"; "Core"]
            TypeRegistry = Map.empty
            ActiveSymbols = []
            PatternRegistry = initialPatterns
        }
        
        { State = state; ResolutionHistory = [] }
    
    let registerFunction (qualifiedName: string) (paramTypes: MLIRType list) (returnType: MLIRType) (opPattern: MLIROperationPattern) (registry: SymbolRegistry) : SymbolRegistry =
        let shortName = 
            if qualifiedName.Contains(".") then
                qualifiedName.Split([|'.'|], StringSplitOptions.RemoveEmptyEntries) |> Array.last
            else qualifiedName
            
        let namespace' = 
            if qualifiedName.Contains(".") then
                qualifiedName.Substring(0, qualifiedName.LastIndexOf('.'))
            else "Global"
            
        let symbol = {
            QualifiedName = qualifiedName
            ShortName = shortName
            ParameterTypes = paramTypes
            ReturnType = returnType
            Operation = opPattern
            Namespace = namespace'
            SourceLibrary = "user"
            RequiresExternal = false
        }
        
        let updatedState = SymbolResolution.registerSymbol symbol registry.State
        { registry with State = updatedState }
    
    let addNamespace (namespace': string) (registry: SymbolRegistry) : SymbolRegistry =
        let updatedState = SymbolResolution.addNamespace namespace' registry.State
        { registry with State = updatedState }
        
    let generateMLIRCall (funcName: string) (args: string list) (argTypes: MLIRType list) (resultId: string) (registry: SymbolRegistry) : CompilerResult<string list * SymbolRegistry> =
        match SymbolResolution.resolveSymbol funcName registry.State with
        | Success symbol ->
            let updatedRegistry = { 
                registry with 
                    ResolutionHistory = ("symbol_found", sprintf "Found symbol %s" funcName) :: registry.ResolutionHistory 
            }
                
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
                sprintf "Missing required symbols: %s" (String.concat ", " missingSymbols))]
    
    /// Get all resolved symbols for reachability analysis
    /// Returns a map that's compatible with ReachabilityAnalyzer's signature
    let getAllSymbols (registry: SymbolRegistry) : Map<string, obj> =
        registry.State.SymbolsByQualified
        |> Map.map (fun _ symbol -> symbol :> obj)

    /// Extract entry points from parsed AST
    let getEntryPoints (input: ParsedInput) : Set<string> =
        match input with
        | ParsedInput.ImplFile(ParsedImplFileInput(_, _, _, _, _, modules, _, _, _)) ->
            modules 
            |> List.collect (fun (SynModuleOrNamespace(_, _, _, decls, _, _, _, _, _)) ->
                decls |> List.collect (function
                    | SynModuleDecl.Let(_, bindings, _) ->
                        bindings |> List.choose (function
                            | SynBinding(_, _, _, _, attrs, _, _, SynPat.Named(SynIdent(ident, _), _, _, _), _, _, _, _, _) ->
                                let hasEntryPoint = 
                                    attrs |> List.exists (fun attr ->
                                        attr.Attributes |> List.exists (fun attr ->
                                            match attr.TypeName with
                                            | SynLongIdent(longId, _, _) ->
                                                let name = longId |> List.map (fun id -> id.idText) |> String.concat "."
                                                name = "EntryPoint" || name = "System.EntryPoint" || name = "EntryPointAttribute"
                                            ))
                                if hasEntryPoint then Some ident.idText else None
                            | _ -> None)
                    ))
            |> Set.ofList
        | _ -> Set.empty