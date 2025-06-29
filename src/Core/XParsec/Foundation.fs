module Core.XParsec.Foundation

open System
open System.Text
open XParsec

/// Position information for error reporting
type SourcePosition = {
    Line: int
    Column: int
    File: string
    Offset: int
}

/// Core error types for the Firefly compiler
type FireflyError =
    | SyntaxError of position: SourcePosition * message: string * context: string list
    | ConversionError of phase: string * source: string * target: string * message: string
    | TypeCheckError of construct: string * message: string * location: SourcePosition
    | InternalError of phase: string * message: string * details: string option
    | MLIRGenerationError of phase: string * message: string * functionName: string option

/// Result type for compiler operations - no fallbacks allowed
type CompilerResult<'T> =
    | Success of 'T
    | CompilerFailure of FireflyError list

/// Compiler state for tracking translation context
type FireflyState = {
    CurrentFile: string
    ImportedModules: string list
    TypeDefinitions: Map<string, string>
    ScopeStack: string list list
    ErrorStack: string list
}

/// MLIR-specific state for code generation
type MLIRState = {
    Output: StringBuilder
    Indent: int
    SSACounter: int
    LocalVars: Map<string, string * string>  // name -> (ssa, type)
    RequiredExternals: Set<string>
    CurrentFunction: string option
    GeneratedFunctions: Set<string>
    CurrentModule: string list
    HasErrors: bool
}

/// Combined state for MLIR generation
type MLIRBuilderState = {
    Firefly: FireflyState
    MLIR: MLIRState
}

/// MLIR combinator type - transforms MLIR state and produces results
type MLIRCombinator<'T> = MLIRBuilderState -> CompilerResult<'T * MLIRBuilderState>

/// MLIR combinators - separate from XParsec parsers
module MLIRCombinators =
    
    /// Lift a value into the MLIR context
    let lift (value: 'T): MLIRCombinator<'T> =
        fun state -> Success(value, state)
    
    /// Get current MLIR state
    let getMLIRState: MLIRCombinator<MLIRBuilderState> =
        fun state -> Success(state, state)
    
    /// Update MLIR state
    let updateMLIRState (f: MLIRBuilderState -> MLIRBuilderState): MLIRCombinator<unit> =
        fun state -> Success((), f state)
    
    /// Sequential composition of MLIR combinators
    let (>>=) (combinator: MLIRCombinator<'T>) (f: 'T -> MLIRCombinator<'U>): MLIRCombinator<'U> =
        fun state ->
            match combinator state with
            | Success(value, state') -> f value state'
            | CompilerFailure errors -> CompilerFailure errors
    
    /// Map operator for MLIRCombinator
    let (|>>) (m: MLIRCombinator<'a>) (f: 'a -> 'b): MLIRCombinator<'b> =
        m >>= (fun value -> lift (f value))
    
    /// Monadic map for lists
    let rec mapM (f: 'a -> MLIRCombinator<'b>) (list: 'a list): MLIRCombinator<'b list> =
        match list with
        | [] -> lift []
        | head :: tail ->
            f head >>= fun mappedHead ->
            mapM f tail >>= fun mappedTail ->
            lift (mappedHead :: mappedTail)
    
    /// Fail with MLIR generation error
    let fail (phase: string) (message: string): MLIRCombinator<'T> =
        fun state ->
            let location = state.MLIR.CurrentFunction
            let error = MLIRGenerationError(phase, message, location)
            CompilerFailure [error]

/// MLIR computation expression builder
type MLIRBuilder() =
    member inline _.Bind(c, f) = MLIRCombinators.(>>=) c f
    member inline _.Return(x) = MLIRCombinators.lift x
    member inline _.ReturnFrom(c) = c
    member inline _.Zero() = MLIRCombinators.lift ()
    
    member inline _.For(sequence: seq<'T>, body: 'T -> MLIRCombinator<unit>) =
        let folder acc item = 
            MLIRCombinators.(>>=) acc (fun () -> body item)
        Seq.fold folder (MLIRCombinators.lift ()) sequence

/// MLIR computation expression
let mlir = MLIRBuilder()

/// Creates initial compiler state
let initialState (fileName: string) : FireflyState = {
    CurrentFile = fileName
    ImportedModules = []
    TypeDefinitions = Map.empty
    ScopeStack = [[]]
    ErrorStack = []
}

/// Creates initial MLIR state
let initialMLIRState : MLIRState = {
    Output = StringBuilder()
    Indent = 1
    SSACounter = 0
    LocalVars = Map.empty
    RequiredExternals = Set.empty
    CurrentFunction = None
    GeneratedFunctions = Set.empty
    CurrentModule = []
    HasErrors = false
}

/// Creates combined initial state
let initialMLIRBuilderState (fileName: string) : MLIRBuilderState = {
    Firefly = initialState fileName
    MLIR = initialMLIRState
}

/// Runs an MLIR combinator and extracts the result
let runMLIRCombinator (combinator: MLIRCombinator<'T>) (initialState: MLIRBuilderState): CompilerResult<'T * string> =
    match combinator initialState with
    | Success(result, finalState) -> 
        if finalState.MLIR.HasErrors then
            CompilerFailure [InternalError("MLIR Generation", "Compilation completed with errors", None)]
        else
            Success(result, finalState.MLIR.Output.ToString())
    | CompilerFailure errors -> CompilerFailure errors