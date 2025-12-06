/// EmissionMonad - Monadic MLIR emission
///
/// Provides a monadic interface for MLIR code generation, inspired by
/// mlir-hs's MonadNameSupply and MonadBlockBuilder patterns.
///
/// The Emit<'T> type represents a computation that:
/// - Reads from a PSGZipper (current focus + context)
/// - Modifies EmissionState (SSA counter, locals, etc.)
/// - Writes to MLIRBuilder (output buffer)
/// - Produces a value of type 'T
module Alex.CodeGeneration.EmissionMonad

open System.Text
open Core.PSG.Types
open Alex.CodeGeneration.EmissionContext
open Alex.Traversal.PSGZipper

/// The emission environment
type EmitEnv = {
    Zipper: PSGZipper
    Builder: MLIRBuilder
}

/// Emission computation: state transformer over EmissionState
/// Returns updated state + value
type Emit<'T> = EmitEnv -> EmissionState -> EmissionState * 'T

// ═══════════════════════════════════════════════════════════════════
// Monad Operations
// ═══════════════════════════════════════════════════════════════════

/// Return a value without side effects
let emit (value: 'T) : Emit<'T> =
    fun _env state -> (state, value)

/// Map a function over the result
let map (f: 'T -> 'U) (m: Emit<'T>) : Emit<'U> =
    fun env state ->
        let state', x = m env state
        (state', f x)

/// Infix map
let (|>>) (m: Emit<'T>) (f: 'T -> 'U) : Emit<'U> =
    map f m

/// Monadic bind
let bind (f: 'T -> Emit<'U>) (m: Emit<'T>) : Emit<'U> =
    fun env state ->
        let state', x = m env state
        f x env state'

/// Infix bind
let (>>=) (m: Emit<'T>) (f: 'T -> Emit<'U>) : Emit<'U> =
    bind f m

/// Sequence, keeping second result
let (>>.) (m1: Emit<'T>) (m2: Emit<'U>) : Emit<'U> =
    m1 >>= fun _ -> m2

/// Sequence, keeping first result
let (.>>) (m1: Emit<'T>) (m2: Emit<'U>) : Emit<'T> =
    m1 >>= fun x -> m2 >>= fun _ -> emit x

/// Sequence and pair
let (.>>.) (m1: Emit<'T>) (m2: Emit<'U>) : Emit<'T * 'U> =
    m1 >>= fun a -> m2 >>= fun b -> emit (a, b)

/// Applicative apply
let apply (mf: Emit<'T -> 'U>) (mx: Emit<'T>) : Emit<'U> =
    mf >>= fun f -> mx >>= fun x -> emit (f x)

/// Infix apply
let (<*>) mf mx = apply mf mx

/// Lift a 2-arg function
let map2 (f: 'T -> 'U -> 'V) (m1: Emit<'T>) (m2: Emit<'U>) : Emit<'V> =
    emit f <*> m1 <*> m2

/// Sequence a list of computations
let sequence (ms: Emit<'T> list) : Emit<'T list> =
    List.foldBack (fun m acc -> map2 (fun x xs -> x :: xs) m acc) ms (emit [])

/// Map and sequence
let traverse (f: 'T -> Emit<'U>) (xs: 'T list) : Emit<'U list> =
    xs |> List.map f |> sequence

/// Execute for side effects, ignoring results
let forEach (f: 'T -> Emit<unit>) (xs: 'T list) : Emit<unit> =
    xs |> traverse f |>> ignore

// ═══════════════════════════════════════════════════════════════════
// State Access
// ═══════════════════════════════════════════════════════════════════

/// Get the current state
let getState : Emit<EmissionState> =
    fun _env state -> (state, state)

/// Set the state
let setState (newState: EmissionState) : Emit<unit> =
    fun _env _state -> (newState, ())

/// Modify the state
let modifyState (f: EmissionState -> EmissionState) : Emit<unit> =
    fun _env state -> (f state, ())

/// Get the environment
let getEnv : Emit<EmitEnv> =
    fun env state -> (state, env)

/// Get the zipper
let getZipper : Emit<PSGZipper> =
    fun env state -> (state, env.Zipper)

/// Get the current focus node
let getFocus : Emit<PSGNode> =
    getZipper |>> fun z -> z.Focus

/// Get the builder
let getBuilder : Emit<MLIRBuilder> =
    fun env state -> (state, env.Builder)

// ═══════════════════════════════════════════════════════════════════
// Zipper Navigation
// ═══════════════════════════════════════════════════════════════════

/// Run an emission computation at a different zipper position
/// This is the key operation for zipper-based traversal
let withZipper (newZipper: PSGZipper) (m: Emit<'T>) : Emit<'T> =
    fun env state ->
        let newEnv = { env with Zipper = newZipper }
        m newEnv state

/// Navigate to a specific node and run emission there
/// The zipper is moved to focus on the node, preserving graph context
let atNode (node: PSGNode) (m: Emit<'T>) : Emit<'T> =
    getZipper >>= fun z ->
    let nodeZipper = PSGZipper.createWithState z.Graph node z.State
    withZipper nodeZipper m

/// Navigate down to first child and run emission
/// Fails if no children exist - use tryFirstChild for optional navigation
let atFirstChild (m: Emit<'T>) : Emit<'T> =
    getZipper >>= fun z ->
    match PSGZipper.down z with
    | NavOk childZipper -> withZipper childZipper m
    | NavFail msg -> failwith (sprintf "Navigation failed: %s" msg)

/// Navigate down to nth child and run emission
/// Fails if child doesn't exist - use tryNthChild for optional navigation
let atNthChild (index: int) (m: Emit<'T>) : Emit<'T> =
    getZipper >>= fun z ->
    match PSGZipper.downTo index z with
    | NavOk childZipper -> withZipper childZipper m
    | NavFail msg -> failwith (sprintf "Navigation to child %d failed: %s" index msg)

/// Traverse all children, collecting results
/// This threads state through each child in order
let traverseChildrenM (f: Emit<'T>) : Emit<'T list> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z
    let rec go acc remaining =
        match remaining with
        | [] -> emit (List.rev acc)
        | child :: rest ->
            atNode child f >>= fun result ->
            go (result :: acc) rest
    go [] children

/// Traverse children, filtering and mapping
let traverseChildrenWhere (pred: PSGNode -> bool) (f: Emit<'T>) : Emit<'T list> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z |> List.filter pred
    let rec go acc remaining =
        match remaining with
        | [] -> emit (List.rev acc)
        | child :: rest ->
            atNode child f >>= fun result ->
            go (result :: acc) rest
    go [] children

/// Execute emission at each child, returning the last result (for sequences)
let emitChildrenSequentially (f: Emit<'T>) : Emit<'T option> =
    getZipper >>= fun z ->
    let children = PSGZipper.childNodes z
    match children with
    | [] -> emit None
    | _ ->
        let rec go lastResult remaining =
            match remaining with
            | [] -> emit lastResult
            | child :: rest ->
                atNode child f >>= fun result ->
                go (Some result) rest
        go None children

// ═══════════════════════════════════════════════════════════════════
// Pattern Integration
// ═══════════════════════════════════════════════════════════════════

// Patterns are now node classifiers: PSGNode -> 'T option or (PSG * PSGNode) -> 'T option
// The zipper provides access to the graph and current focus node.

/// Run a node classifier at the current focus
/// Classifiers extract typed data from PSG nodes
let runClassifier (classifier: Core.PSG.Types.PSGNode -> 'T option) : Emit<'T option> =
    getFocus |>> classifier

/// Run a classifier that needs graph context
let runGraphClassifier (classifier: Core.PSG.Types.ProgramSemanticGraph -> Core.PSG.Types.PSGNode -> 'T option) : Emit<'T option> =
    getZipper >>= fun z ->
    emit (classifier z.Graph z.Focus)

/// Run a classifier and if it matches, run the emission continuation
let whenClassifier (classifier: Core.PSG.Types.PSGNode -> 'T option) (then': 'T -> Emit<'U>) : Emit<'U option> =
    runClassifier classifier >>= fun resultOpt ->
    match resultOpt with
    | Some value -> then' value |>> Some
    | None -> emit None

/// Run a graph classifier and if it matches, run the emission continuation
let whenGraphClassifier (classifier: Core.PSG.Types.ProgramSemanticGraph -> Core.PSG.Types.PSGNode -> 'T option) (then': 'T -> Emit<'U>) : Emit<'U option> =
    runGraphClassifier classifier >>= fun resultOpt ->
    match resultOpt with
    | Some value -> then' value |>> Some
    | None -> emit None

/// Try multiple classifier/emission pairs in order, returning first match
let tryClassifiers (classifiers: ((Core.PSG.Types.PSGNode -> 'T option) * ('T -> Emit<'U>)) list) : Emit<'U option> =
    let rec go remaining =
        match remaining with
        | [] -> emit None
        | (classifier, handler) :: rest ->
            whenClassifier classifier handler >>= fun result ->
            match result with
            | Some value -> emit (Some value)
            | None -> go rest
    go classifiers

/// Try multiple graph classifier/emission pairs in order
let tryGraphClassifiers (classifiers: ((Core.PSG.Types.ProgramSemanticGraph -> Core.PSG.Types.PSGNode -> 'T option) * ('T -> Emit<'U>)) list) : Emit<'U option> =
    getZipper >>= fun z ->
    let rec go remaining =
        match remaining with
        | [] -> emit None
        | (classifier, handler) :: rest ->
            match classifier z.Graph z.Focus with
            | Some value ->
                handler value |>> Some
            | None -> go rest
    go classifiers

/// Try multiple emission computations in order, returning first Some result
/// Each computation returns Option - None means "doesn't apply", Some means "handled"
let tryEmissions (emissions: Emit<'T option> list) : Emit<'T option> =
    let rec go remaining =
        match remaining with
        | [] -> emit None
        | e :: rest ->
            e >>= fun result ->
            match result with
            | Some value -> emit (Some value)
            | None -> go rest
    go emissions

// ═══════════════════════════════════════════════════════════════════
// SSA Operations
// ═══════════════════════════════════════════════════════════════════

/// Generate a fresh SSA value name
let freshSSA : Emit<string> =
    fun _env state ->
        let name = sprintf "%%v%d" state.SSACounter
        let state' = { state with SSACounter = state.SSACounter + 1 }
        (state', name)

/// Generate a fresh SSA value with type tracking
let freshSSAWithType (mlirType: string) : Emit<string> =
    fun _env state ->
        let name = sprintf "%%v%d" state.SSACounter
        let state' = {
            state with
                SSACounter = state.SSACounter + 1
                SSATypes = Map.add name mlirType state.SSATypes
        }
        (state', name)

/// Look up a local variable
let lookupLocal (name: string) : Emit<string option> =
    getState |>> fun s -> Map.tryFind name s.Locals

/// Look up a local variable's type
let lookupLocalType (name: string) : Emit<string option> =
    getState |>> fun s -> Map.tryFind name s.LocalTypes

/// Bind a local variable
let bindLocal (fsharpName: string) (ssaName: string) (mlirType: string) : Emit<unit> =
    modifyState (fun s -> {
        s with
            Locals = Map.add fsharpName ssaName s.Locals
            LocalTypes = Map.add fsharpName mlirType s.LocalTypes
            SSATypes = Map.add ssaName mlirType s.SSATypes
    })

/// Register a string literal
let registerStringLiteral (content: string) : Emit<string> =
    fun _env state ->
        match state.StringLiterals |> List.tryFind (fun (c, _) -> c = content) with
        | Some (_, name) -> (state, name)
        | None ->
            let name = sprintf "@str%d" (List.length state.StringLiterals)
            let state' = { state with StringLiterals = (content, name) :: state.StringLiterals }
            (state', name)

// ═══════════════════════════════════════════════════════════════════
// Bindings Bridge
// ═══════════════════════════════════════════════════════════════════

/// Create an SSAContext that is synchronized with the current EmissionState
/// This allows calling into the Bindings layer from the monad
let withSSAContext (action: SSAContext -> 'T) : Emit<'T> =
    fun env state ->
        // Create a temporary SSAContext from our state
        let ctx = SSAContext.create ()
        ctx.Counter <- state.SSACounter
        ctx.Locals <- state.Locals
        ctx.LocalTypes <- state.LocalTypes
        ctx.SSATypes <- state.SSATypes
        ctx.StringLiterals <- state.StringLiterals

        // Run the action
        let result = action ctx

        // Sync state back from SSAContext
        let state' = {
            state with
                SSACounter = ctx.Counter
                Locals = ctx.Locals
                LocalTypes = ctx.LocalTypes
                SSATypes = ctx.SSATypes
                StringLiterals = ctx.StringLiterals
        }
        (state', result)

/// Execute a Bindings function that uses MLIRBuilder and SSAContext
let invokeBinding (action: MLIRBuilder -> SSAContext -> 'T) : Emit<'T> =
    getBuilder >>= fun builder ->
    withSSAContext (fun ctx -> action builder ctx)

// ═══════════════════════════════════════════════════════════════════
// Output Operations
// ═══════════════════════════════════════════════════════════════════

/// Emit a line of MLIR
let line (text: string) : Emit<unit> =
    fun env state ->
        MLIRBuilder.line env.Builder text
        (state, ())

/// Emit a line without indentation
let lineNoIndent (text: string) : Emit<unit> =
    fun env state ->
        MLIRBuilder.lineNoIndent env.Builder text
        (state, ())

/// Emit raw text (no newline)
let raw (text: string) : Emit<unit> =
    fun env state ->
        MLIRBuilder.raw env.Builder text
        (state, ())

/// Increase indentation
let pushIndent : Emit<unit> =
    fun env state ->
        MLIRBuilder.push env.Builder
        (state, ())

/// Decrease indentation
let popIndent : Emit<unit> =
    fun env state ->
        MLIRBuilder.pop env.Builder
        (state, ())

/// Execute with increased indentation
let withIndent (body: Emit<'T>) : Emit<'T> =
    pushIndent >>. body .>> popIndent

/// Emit a block: { ... }
let block (header: string) (body: Emit<'T>) : Emit<'T> =
    line (header + " {") >>.
    withIndent body .>>
    line "}"

// ═══════════════════════════════════════════════════════════════════
// MLIR Emission Helpers
// ═══════════════════════════════════════════════════════════════════

/// Emit an arith.constant
let emitConstant (value: string) (mlirType: string) : Emit<string> =
    freshSSAWithType mlirType >>= fun name ->
    line (sprintf "%s = arith.constant %s : %s" name value mlirType) >>.
    emit name

/// Emit an i32 constant
let emitI32 (value: int) : Emit<string> =
    emitConstant (string value) "i32"

/// Emit an i64 constant
let emitI64 (value: int64) : Emit<string> =
    emitConstant (string value) "i64"

/// Emit an f32 constant
let emitF32 (value: float32) : Emit<string> =
    emitConstant (sprintf "%f" value) "f32"

/// Emit an f64 constant
let emitF64 (value: float) : Emit<string> =
    emitConstant (sprintf "%f" value) "f64"

/// Emit arith.addi
let emitAddi (a: string) (b: string) (typ: string) : Emit<string> =
    freshSSAWithType typ >>= fun result ->
    line (sprintf "%s = arith.addi %s, %s : %s" result a b typ) >>.
    emit result

/// Emit arith.subi
let emitSubi (a: string) (b: string) (typ: string) : Emit<string> =
    freshSSAWithType typ >>= fun result ->
    line (sprintf "%s = arith.subi %s, %s : %s" result a b typ) >>.
    emit result

/// Emit arith.muli
let emitMuli (a: string) (b: string) (typ: string) : Emit<string> =
    freshSSAWithType typ >>= fun result ->
    line (sprintf "%s = arith.muli %s, %s : %s" result a b typ) >>.
    emit result

/// Emit arith.divui
let emitDivui (a: string) (b: string) (typ: string) : Emit<string> =
    freshSSAWithType typ >>= fun result ->
    line (sprintf "%s = arith.divui %s, %s : %s" result a b typ) >>.
    emit result

/// Emit arith.remui
let emitRemui (a: string) (b: string) (typ: string) : Emit<string> =
    freshSSAWithType typ >>= fun result ->
    line (sprintf "%s = arith.remui %s, %s : %s" result a b typ) >>.
    emit result

/// Emit arith.cmpi
let emitCmpi (predicate: string) (a: string) (b: string) (typ: string) : Emit<string> =
    freshSSAWithType "i1" >>= fun result ->
    line (sprintf "%s = arith.cmpi %s, %s, %s : %s" result predicate a b typ) >>.
    emit result

/// Emit arith.select
let emitSelect (cond: string) (ifTrue: string) (ifFalse: string) (typ: string) : Emit<string> =
    freshSSAWithType typ >>= fun result ->
    line (sprintf "%s = arith.select %s, %s, %s : %s" result cond ifTrue ifFalse typ) >>.
    emit result

/// Emit arith.extsi (sign extend)
let emitExtsi (value: string) (fromType: string) (toType: string) : Emit<string> =
    freshSSAWithType toType >>= fun result ->
    line (sprintf "%s = arith.extsi %s : %s to %s" result value fromType toType) >>.
    emit result

/// Emit arith.trunci (truncate)
let emitTrunci (value: string) (fromType: string) (toType: string) : Emit<string> =
    freshSSAWithType toType >>= fun result ->
    line (sprintf "%s = arith.trunci %s : %s to %s" result value fromType toType) >>.
    emit result

/// Emit llvm.alloca
let emitAlloca (elemType: string) (count: string) : Emit<string> =
    freshSSAWithType "!llvm.ptr" >>= fun result ->
    line (sprintf "%s = llvm.alloca %s x %s : (i64) -> !llvm.ptr" result count elemType) >>.
    emit result

/// Emit llvm.load
let emitLoad (ptr: string) (elemType: string) : Emit<string> =
    freshSSAWithType elemType >>= fun result ->
    line (sprintf "%s = llvm.load %s : !llvm.ptr -> %s" result ptr elemType) >>.
    emit result

/// Emit llvm.store
let emitStore (value: string) (ptr: string) (elemType: string) : Emit<unit> =
    line (sprintf "llvm.store %s, %s : %s, !llvm.ptr" value ptr elemType)

/// Emit llvm.getelementptr
let emitGep (basePtr: string) (index: string) (elemType: string) : Emit<string> =
    freshSSAWithType "!llvm.ptr" >>= fun result ->
    line (sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i64) -> !llvm.ptr, %s" result basePtr index elemType) >>.
    emit result

/// Emit llvm.mlir.addressof
let emitAddressOf (globalName: string) : Emit<string> =
    freshSSAWithType "!llvm.ptr" >>= fun result ->
    line (sprintf "%s = llvm.mlir.addressof %s : !llvm.ptr" result globalName) >>.
    emit result

/// Emit func.return
let emitReturn (value: string) (typ: string) : Emit<unit> =
    line (sprintf "func.return %s : %s" value typ)

/// Emit func.return for void
let emitReturnVoid : Emit<unit> =
    line "func.return"

/// Emit cf.br (unconditional branch)
let emitBr (label: string) : Emit<unit> =
    line (sprintf "cf.br ^%s" label)

/// Emit cf.cond_br (conditional branch)
let emitCondBr (cond: string) (trueLabel: string) (falseLabel: string) : Emit<unit> =
    line (sprintf "cf.cond_br %s, ^%s, ^%s" cond trueLabel falseLabel)

/// Emit a block label
let emitBlockLabel (label: string) : Emit<unit> =
    lineNoIndent (sprintf "^%s:" label)

// ═══════════════════════════════════════════════════════════════════
// Running the Monad
// ═══════════════════════════════════════════════════════════════════

/// Run an emission computation
let run (zipper: PSGZipper) (builder: MLIRBuilder) (initial: EmissionState) (m: Emit<'T>) : EmissionState * 'T =
    let env = { Zipper = zipper; Builder = builder }
    m env initial

/// Run and extract just the value
let runValue (zipper: PSGZipper) (builder: MLIRBuilder) (initial: EmissionState) (m: Emit<'T>) : 'T =
    let _, result = run zipper builder initial m
    result

/// Run and extract the final state
let runState (zipper: PSGZipper) (builder: MLIRBuilder) (initial: EmissionState) (m: Emit<'T>) : EmissionState =
    let state', _ = run zipper builder initial m
    state'
