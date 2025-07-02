module Dabbit.CodeGeneration.MLIREmitter

open System.Text
open Core.XParsec.Foundation
open Core.Types.TypeSystem
open Dabbit.Bindings.SymbolRegistry

/// Critical error for undefined MLIR generation
exception MLIRGenerationException of string * string option

/// MLIR builder type - transforms a builder state
type MLIRBuilder<'T> = MLIRBuilderState -> ('T * MLIRBuilderState)

/// Builder state with output and context
and MLIRBuilderState = {
    Output: StringBuilder
    Indent: int
    SSACounter: int
    LocalVars: Map<string, (string * MLIRType)>  // SSA name and type
    TypeContext: TypeMapping.TypeContext
    SymbolRegistry: SymbolRegistry
    RequiredExternals: Set<string>
    CurrentFunction: string option
    GeneratedFunctions: Set<string>
    LocalFunctions: Map<string, MLIRType>  
    CurrentModule: string list  // Module path for symbol resolution
    OpenedNamespaces: string list list  // List of opened namespaces
    HasErrors: bool
    ExpectedType: MLIRType option
}

/// MLIR builder computation expression
type MLIRBuilderCE() =
    member _.Return(x) : MLIRBuilder<'T> = 
        fun state -> (x, state)
    
    member _.ReturnFrom(x: MLIRBuilder<'T>) : MLIRBuilder<'T> = x
    
    member _.Bind(m: MLIRBuilder<'T>, f: 'T -> MLIRBuilder<'U>) : MLIRBuilder<'U> =
        fun state ->
            let (result, state') = m state
            (f result) state'
    
    member _.Zero() : MLIRBuilder<unit> =
        fun state -> ((), state)
    
    member _.Delay(f) : MLIRBuilder<'T> =
        fun state -> f () state
    
    member _.Run(f) : MLIRBuilder<'T> = f

/// MLIR builder instance
let mlir = MLIRBuilderCE()

/// Get current state
let getState : MLIRBuilder<MLIRBuilderState> =
    fun state -> (state, state)

/// Set state
let setState (newState: MLIRBuilderState) : MLIRBuilder<unit> =
    fun _ -> ((), newState)

/// Update state
let updateState (f: MLIRBuilderState -> MLIRBuilderState) : MLIRBuilder<unit> =
    fun state -> ((), f state)

/// Generate unique SSA name
let nextSSA (prefix: string) : MLIRBuilder<string> =
    fun state ->
        let name = sprintf "%%%s%d" prefix state.SSACounter
        let state' = { state with SSACounter = state.SSACounter + 1 }
        (name, state')

/// Emit raw text
let emit (text: string) : MLIRBuilder<unit> =
    fun state ->
        state.Output.Append(text) |> ignore
        ((), state)

/// Emit with current indentation
let emitIndented (text: string) : MLIRBuilder<unit> =
    fun state ->
        let indent = String.replicate state.Indent "  "
        state.Output.Append(indent).Append(text) |> ignore
        ((), state)

/// Emit line with indentation
let emitLine (text: string) : MLIRBuilder<unit> =
    mlir {
        do! emitIndented text
        do! emit "\n"
    }

/// Emit newline only
let newline : MLIRBuilder<unit> = emit "\n"

/// Increase indentation for nested scope
let indent (builder: MLIRBuilder<'T>) : MLIRBuilder<'T> =
    mlir {
        do! updateState (fun s -> { s with Indent = s.Indent + 1 })
        let! result = builder
        do! updateState (fun s -> { s with Indent = s.Indent - 1 })
        return result
    }

/// Add external dependency
let requireExternal (name: string) : MLIRBuilder<unit> =
    updateState (fun s -> { s with RequiredExternals = Set.add name s.RequiredExternals })

/// Bind local variable with type
let bindLocal (name: string) (ssa: string) (typ: MLIRType) : MLIRBuilder<unit> =
    updateState (fun s -> { s with LocalVars = Map.add name (ssa, typ) s.LocalVars })

/// Lookup local variable
let lookupLocal (name: string) : MLIRBuilder<(string * MLIRType) option> =
    mlir {
        let! state = getState
        return Map.tryFind name state.LocalVars
    }

/// Fail hard with error
let failHard (phase: string) (message: string) : MLIRBuilder<'T> =
    fun state ->
        let funcContext = 
            match state.CurrentFunction with
            | Some f -> sprintf " in function %s" f
            | None -> ""
        raise (MLIRGenerationException(sprintf "[%s] %s%s" phase message funcContext, state.CurrentFunction))

/// Emit stack allocation
let emitAlloca (bufferSSA: string) (size: int) : MLIRBuilder<unit> =
    mlir {
        do! emitLine (sprintf "%s = memref.alloca() : memref<%dxi8>" bufferSSA size)
    }

/// Resolve symbol using registry (placeholder implementation)
let private resolveSymbol (name: string) : MLIRBuilder<ResolvedSymbol> =
    mlir {
        let! state = getState
        // TODO: Implement actual symbol resolution when SymbolRegistry module is complete
        return! failHard "Symbol Resolution" (sprintf "Symbol resolution not yet implemented for '%s'" name)
    }

/// Create initial state
let createInitialState (typeCtx: TypeMapping.TypeContext) (symbolRegistry: SymbolRegistry) = {
    Output = StringBuilder()
    Indent = 1
    SSACounter = 0
    LocalVars = Map.empty
    TypeContext = typeCtx
    SymbolRegistry = symbolRegistry
    RequiredExternals = Set.empty
    CurrentFunction = None
    GeneratedFunctions = Set.empty
    LocalFunctions = Map.empty
    CurrentModule = []
    OpenedNamespaces = []
    HasErrors = false
    ExpectedType = None
}

/// Run builder and extract result
let runBuilder (builder: MLIRBuilder<'T>) (state: MLIRBuilderState) : 'T * MLIRBuilderState =
    builder state

/// Run builder and extract output
let runBuilderForOutput (builder: MLIRBuilder<'T>) (state: MLIRBuilderState) : string =
    let (_, finalState) = builder state
    finalState.Output.ToString()