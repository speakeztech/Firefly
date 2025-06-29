module Core.XParsec.Foundation

open System
open System.Text  
open XParsec
open XParsec.CharParsers
open XParsec.Parsers

// ===================================================================
// Core Compiler Types
// ===================================================================

/// MLIR value representation
type MLIRValue = {
    SSA: string
    Type: string
    IsConstant: bool
}

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

// ===================================================================
// XParsec Re-exports - Controlled Access Point
// ===================================================================

// Character parsers
let pString str = pstring str
let pChar predicate = satisfy predicate
let pCharLiteral ch = pchar ch  
let pSpaces = spaces
let pSpaces1 = spaces1

// Combinators  
let choice parsers = choice parsers
let pBetween open_ close content = between open_ close content
let preturn value = preturn value
let many parser = many parser
let many1 parser = many1 parser
let opt parser = opt parser
let pSepBy parser separator = sepBy parser separator
let pSepBy1 parser separator = sepBy1 parser separator

// Operators
let (>>=) = (>>=)
let (>>.) = (>>.)
let (.>>) = (.>>)
let (|>>) = (|>>)
let (>>%) = (>>%)
let (<|>) = (<|>)

// ===================================================================
// MLIR Generation State
// ===================================================================

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

// ===================================================================
// MLIR-Specific Parsing Utilities
// ===================================================================

/// Parse integer values
let pInt () =
    many1 (pChar System.Char.IsDigit) 
    |>> (fun digits -> 
        let digitStr = new string(digits |> Seq.toArray)
        System.Int32.Parse(digitStr))

/// Parse identifier (alphanumeric + underscore, starting with letter or underscore)
let pIdentifier () =
    (pChar System.Char.IsLetter <|> pChar (fun c -> c = '_')) >>= fun firstChar ->
    many (pChar System.Char.IsLetterOrDigit <|> pChar (fun c -> c = '_')) >>= fun restChars ->
    let identifier = new string(Array.append [|firstChar|] (restChars |> Seq.toArray))
    preturn identifier

/// Parse SSA value names (starting with %)
let pSSAName () =
    pCharLiteral '%' >>. pIdentifier ()

/// Parse quoted string with basic escape sequences
let pQuotedString () =
    let pStringChar = pChar (fun c -> c <> '"' && c <> '\\')
    pCharLiteral '"' >>. many pStringChar >>= fun chars ->
    pCharLiteral '"' >>. preturn (new string(chars |> Seq.toArray))

/// MLIR punctuation with spacing
let pEqualsSpaced () = pSpaces >>. pCharLiteral '=' >>. pSpaces
let pColonSpaced () = pSpaces >>. pCharLiteral ':' >>. pSpaces
let pCommaSpaced () = pSpaces >>. pCharLiteral ',' >>. pSpaces

// ===================================================================
// MLIR Generation Combinators
// ===================================================================

/// MLIR combinators for code generation
module MLIRCombinators =
    
    /// Lift a value into the MLIR context
    let lift (value: 'T): MLIRCombinator<'T> =
        fun state -> Success(value, state)
    
    /// Sequential composition of MLIR combinators
    let (>>=) (combinator: MLIRCombinator<'T>) (f: 'T -> MLIRCombinator<'U>): MLIRCombinator<'U> =
        fun state ->
            match combinator state with
            | Success(value, state') -> f value state'
            | CompilerFailure errors -> CompilerFailure errors
    
    /// Fail with MLIR generation error
    let fail (phase: string) (message: string): MLIRCombinator<'T> =
        fun state ->
            let location = state.MLIR.CurrentFunction
            let error = MLIRGenerationError(phase, message, location)
            CompilerFailure [error]

    /// Get next SSA value name
    let nextSSA (prefix: string): MLIRCombinator<string> =
        fun state ->
            let newCounter = state.MLIR.SSACounter + 1
            let ssaName = sprintf "%%%s%d" prefix newCounter
            let newState = { state with MLIR = { state.MLIR with SSACounter = newCounter } }
            Success(ssaName, newState)

    /// Emit a line of MLIR code
    let emitLine (line: string): MLIRCombinator<unit> =
        fun state ->
            let indentStr = String.replicate (state.MLIR.Indent * 2) " "
            let fullLine = indentStr + line + "\n"
            state.MLIR.Output.Append(fullLine) |> ignore
            Success((), state)

/// MLIR computation expression builder
type MLIRBuilder() =
    member inline _.Bind(c, f) = MLIRCombinators.(>>=) c f
    member inline _.Return(x) = MLIRCombinators.lift x
    member inline _.ReturnFrom(c) = c
    member inline _.Zero() = MLIRCombinators.lift ()

/// MLIR computation expression
let mlir = MLIRBuilder()

// ===================================================================
// State Management
// ===================================================================

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

// ===================================================================
// Utility Functions
// ===================================================================

let isNullOrEmpty (str: string) = String.IsNullOrEmpty(str)
let isNullOrWhiteSpace (str: string) = String.IsNullOrWhiteSpace(str)
let indent (level: int) : string = String.replicate (level * 2) " "