/// MLIRZipper - Codata-driven MLIR composition via observation and witnessing
///
/// ARCHITECTURAL FOUNDATION (Coeffects & Codata):
/// - PSGZipper navigates the INPUT (PSG tree structure)
/// - MLIRZipper composes the OUTPUT (MLIR compute graph) via OBSERVATION
/// - Together they form Alex's dual-zipper architecture
///
/// KEY VOCABULARY (from Coeffects & Codata theory):
/// - **witness** - To observe and record a computation (the fold witnesses each node)
/// - **observe** - To note a context requirement (coeffect - what we need)
/// - **yield** - To produce on demand (codata production)
/// - **bind** - To associate an observation with an identity
/// - **recall** - To retrieve a prior observation
/// - **extract** - Comonad operation: collapse accumulated context to final value
///
/// KEY DESIGN PRINCIPLES (from SpeakEZ blogs):
/// 1. SSA IS Functional Programming (Appel 1998) - F# structure maps directly to MLIR
/// 2. Codata/Pull model - observations accumulate; MLIR emerges from witnessed structure
/// 3. Coeffect tracking - observe requirements (string literals, extern funcs)
/// 4. Functional state threading - immutable state passed through
/// 5. NO string pattern matching on symbol names
module Alex.Traversal.MLIRZipper

open Alex.CodeGeneration.MLIRTypes

// ═══════════════════════════════════════════════════════════════════
// MLIR Structure Types - What we're building
// ═══════════════════════════════════════════════════════════════════

/// An MLIR operation with its result SSA value(s)
type MLIROp = {
    /// The operation text (e.g., "%v0 = arith.constant 42 : i32")
    Text: string
    /// Result SSA values produced by this op (empty for void ops)
    Results: (string * MLIRType) list
}

/// A basic block within a function
type MLIRBlock = {
    /// Block label (e.g., "bb0", "entry")
    Label: string
    /// Block arguments (for phi-like semantics)
    Arguments: (string * MLIRType) list
    /// Operations within this block (in order)
    Operations: MLIROp list
}

/// A function definition
type MLIRFunc = {
    /// Function name
    Name: string
    /// Parameter names and types
    Parameters: (string * MLIRType) list
    /// Return type
    ReturnType: MLIRType
    /// Blocks within the function (first is entry)
    Blocks: MLIRBlock list
    /// Attributes (e.g., visibility)
    Attributes: string list
}

/// Global declarations (string literals, extern funcs, buffers)
type MLIRGlobal =
    | StringLiteral of name: string * content: string * length: int
    | ExternFunc of name: string * signature: string
    | StaticBuffer of name: string * size: int

/// A complete MLIR module
type MLIRModule = {
    /// Global declarations
    Globals: MLIRGlobal list
    /// Function definitions
    Functions: MLIRFunc list
}

// ═══════════════════════════════════════════════════════════════════
// MLIRZipper Focus - Where we are in the MLIR structure
// ═══════════════════════════════════════════════════════════════════

/// Current focus within MLIR structure being built
type MLIRFocus =
    | AtModule
    | InFunction of funcName: string
    | InBlock of funcName: string * blockLabel: string

// ═══════════════════════════════════════════════════════════════════
// MLIRZipper Path - Breadcrumbs back to module level
// ═══════════════════════════════════════════════════════════════════

/// Path back through MLIR structure (breadcrumbs for navigation)
type MLIRPath =
    | Top
    | EnteredFunction of parent: MLIRPath * funcName: string * completedFuncs: MLIRFunc list
    | EnteredBlock of parent: MLIRPath * funcName: string * blockLabel: string * completedBlocks: MLIRBlock list * currentParams: (string * MLIRType) list * currentReturnType: MLIRType

// ═══════════════════════════════════════════════════════════════════
// MLIRZipper State - Functional state threaded through traversal
// ═══════════════════════════════════════════════════════════════════

/// Immutable state for MLIR composition (symmetric to EmissionState in PSGZipper)
type MLIRState = {
    /// SSA counter for unique value names
    SSACounter: int
    /// Label counter for unique block labels
    LabelCounter: int
    /// Map from PSG NodeId to (SSA value name, MLIR type)
    /// Records what SSA value was produced for each PSG node
    NodeSSA: Map<string, string * string>
    /// String literals: content -> global name (for deduplication)
    StringLiterals: Map<string, string>
    /// External functions declared
    ExternalFuncs: Set<string>
    /// Current function name (for context)
    CurrentFunction: string option
}

module MLIRState =
    /// Create initial state
    let create () : MLIRState = {
        SSACounter = 0
        LabelCounter = 0
        NodeSSA = Map.empty
        StringLiterals = Map.empty
        ExternalFuncs = Set.empty
        CurrentFunction = None
    }

    /// Yield next SSA name on demand (codata yield)
    let yieldSSA (state: MLIRState) : string * MLIRState =
        let name = sprintf "%%v%d" state.SSACounter
        name, { state with SSACounter = state.SSACounter + 1 }

    /// Yield next block label on demand (codata yield)
    let yieldLabel (state: MLIRState) : string * MLIRState =
        let name = sprintf "bb%d" state.LabelCounter
        name, { state with LabelCounter = state.LabelCounter + 1 }

    /// Bind SSA observation to a PSG node identity
    let bindNodeSSA (nodeId: string) (ssaName: string) (mlirType: string) (state: MLIRState) : MLIRState =
        { state with NodeSSA = Map.add nodeId (ssaName, mlirType) state.NodeSSA }

    /// Recall a prior SSA observation for a PSG node
    let recallNodeSSA (nodeId: string) (state: MLIRState) : (string * string) option =
        Map.tryFind nodeId state.NodeSSA

    /// Observe a string literal requirement (coeffect - context need)
    let observeStringLiteral (content: string) (state: MLIRState) : string * MLIRState =
        match Map.tryFind content state.StringLiterals with
        | Some name -> name, state
        | None ->
            let name = sprintf "str%d" state.StringLiterals.Count
            name, { state with StringLiterals = Map.add content name state.StringLiterals }

    /// Observe an external function requirement (coeffect - context need)
    let observeExternFunc (name: string) (state: MLIRState) : MLIRState =
        { state with ExternalFuncs = Set.add name state.ExternalFuncs }

    /// Set current function context
    let setCurrentFunction (name: string option) (state: MLIRState) : MLIRState =
        { state with CurrentFunction = name }

// ═══════════════════════════════════════════════════════════════════
// The MLIRZipper Type - Symmetric to PSGZipper
// ═══════════════════════════════════════════════════════════════════

/// The MLIR Composition Zipper - symmetric counterpart to PSGZipper
///
/// Where PSGZipper navigates the INPUT (PSG structure),
/// MLIRZipper composes the OUTPUT (MLIR compute graph).
///
/// Key insight: The zipper pattern works for both READING and WRITING trees.
/// - PSGZipper: Focus on current PSG node, Path to get back
/// - MLIRZipper: Focus on current MLIR position, Path with accumulated content
type MLIRZipper = {
    /// Current focus in MLIR structure
    Focus: MLIRFocus
    /// Path back to module level (with accumulated content)
    Path: MLIRPath
    /// Operations accumulated at current focus (in reverse order for efficiency)
    CurrentOps: MLIROp list
    /// Functional state (SSA counters, node mappings, literals)
    State: MLIRState
    /// Completed globals (accumulated during traversal)
    Globals: MLIRGlobal list
}

// ═══════════════════════════════════════════════════════════════════
// MLIRZipper Operations - Navigation and Emission
// ═══════════════════════════════════════════════════════════════════

module MLIRZipper =

    // ─────────────────────────────────────────────────────────────────
    // Construction
    // ─────────────────────────────────────────────────────────────────

    /// Create a new MLIRZipper at module level
    let create () : MLIRZipper = {
        Focus = AtModule
        Path = Top
        CurrentOps = []
        State = MLIRState.create ()
        Globals = []
    }

    /// Create with initial state (for continuing from previous context)
    let createWithState (state: MLIRState) : MLIRZipper = {
        Focus = AtModule
        Path = Top
        CurrentOps = []
        State = state
        Globals = []
    }

    // ─────────────────────────────────────────────────────────────────
    // State Operations (functional threading with codata vocabulary)
    // ─────────────────────────────────────────────────────────────────

    /// Map over the state
    let mapState (f: MLIRState -> MLIRState) (zipper: MLIRZipper) : MLIRZipper =
        { zipper with State = f zipper.State }

    /// Replace the state
    let withState (state: MLIRState) (zipper: MLIRZipper) : MLIRZipper =
        { zipper with State = state }

    /// Yield next SSA name on demand (codata yield)
    let yieldSSA (zipper: MLIRZipper) : string * MLIRZipper =
        let name, newState = MLIRState.yieldSSA zipper.State
        name, { zipper with State = newState }

    /// Yield next block label on demand (codata yield)
    let yieldLabel (zipper: MLIRZipper) : string * MLIRZipper =
        let name, newState = MLIRState.yieldLabel zipper.State
        name, { zipper with State = newState }

    /// Bind SSA observation to a PSG node identity
    let bindNodeSSA (nodeId: string) (ssaName: string) (mlirType: string) (zipper: MLIRZipper) : MLIRZipper =
        mapState (MLIRState.bindNodeSSA nodeId ssaName mlirType) zipper

    /// Recall a prior SSA observation for a PSG node
    let recallNodeSSA (nodeId: string) (zipper: MLIRZipper) : (string * string) option =
        MLIRState.recallNodeSSA nodeId zipper.State

    /// Observe a string literal requirement (coeffect - context need)
    let observeStringLiteral (content: string) (zipper: MLIRZipper) : string * MLIRZipper =
        let name, newState = MLIRState.observeStringLiteral content zipper.State
        // Also add to globals
        let len = content.Length + 1  // +1 for null terminator
        let strGlobal = StringLiteral (name, content, len)
        let newGlobals =
            if List.exists (fun g ->
                match g with
                | StringLiteral (n, _, _) -> n = name
                | _ -> false) zipper.Globals
            then zipper.Globals
            else strGlobal :: zipper.Globals
        name, { zipper with State = newState; Globals = newGlobals }

    /// Observe an external function requirement (coeffect - context need)
    let observeExternFunc (name: string) (signature: string) (zipper: MLIRZipper) : MLIRZipper =
        let newState = MLIRState.observeExternFunc name zipper.State
        let extGlobal = ExternFunc (name, signature)
        let newGlobals =
            if List.exists (fun g ->
                match g with
                | ExternFunc (n, _) -> n = name
                | _ -> false) zipper.Globals
            then zipper.Globals
            else extGlobal :: zipper.Globals
        { zipper with State = newState; Globals = newGlobals }

    /// Observe a static buffer requirement (coeffect - for ReadLine etc.)
    let observeStaticBuffer (name: string) (size: int) (zipper: MLIRZipper) : MLIRZipper =
        let bufGlobal = StaticBuffer (name, size)
        let newGlobals =
            if List.exists (fun g ->
                match g with
                | StaticBuffer (n, _) -> n = name
                | _ -> false) zipper.Globals
            then zipper.Globals
            else bufGlobal :: zipper.Globals
        { zipper with Globals = newGlobals }

    // ─────────────────────────────────────────────────────────────────
    // Navigation - Enter/Exit structures
    // ─────────────────────────────────────────────────────────────────

    /// Enter a function definition
    let enterFunction (name: string) (parameters: (string * MLIRType) list) (returnType: MLIRType) (zipper: MLIRZipper) : MLIRZipper =
        match zipper.Focus with
        | AtModule ->
            // Save current accumulated functions in path
            { zipper with
                Focus = InFunction name
                Path = EnteredFunction (zipper.Path, name, [])
                CurrentOps = []
                State = MLIRState.setCurrentFunction (Some name) zipper.State }
        | _ ->
            // Can only enter function from module level
            failwithf "Cannot enter function '%s' - not at module level (focus: %A)" name zipper.Focus

    /// Enter a basic block within current function
    let enterBlock (label: string) (arguments: (string * MLIRType) list) (zipper: MLIRZipper) : MLIRZipper =
        match zipper.Focus, zipper.Path with
        | InFunction funcName, EnteredFunction (parentPath, _, completedFuncs) ->
            // Save current ops as we enter a new block
            { zipper with
                Focus = InBlock (funcName, label)
                Path = EnteredBlock (zipper.Path, funcName, label, [], [], Unit)  // Store parent function state
                CurrentOps = [] }
        | InBlock (funcName, prevLabel), EnteredBlock (parentPath, fn, _, completedBlocks, funcParams, retTy) ->
            // Finish previous block, enter new one
            let prevBlock = {
                Label = prevLabel
                Arguments = []
                Operations = List.rev zipper.CurrentOps
            }
            { zipper with
                Focus = InBlock (funcName, label)
                Path = EnteredBlock (parentPath, fn, label, prevBlock :: completedBlocks, funcParams, retTy)
                CurrentOps = [] }
        | _ ->
            failwithf "Cannot enter block '%s' - not in a function" label

    /// Exit current block, returning to function level
    /// Note: Uses 'rec' to enable mutual recursion with exitFunction
    let rec exitBlock (zipper: MLIRZipper) : MLIRZipper =
        match zipper.Focus, zipper.Path with
        | InBlock (funcName, label), EnteredBlock (parentPath, fn, _, completedBlocks, funcParams, retTy) ->
            // Complete current block
            let block = {
                Label = label
                Arguments = []
                Operations = List.rev zipper.CurrentOps
            }
            { zipper with
                Focus = InFunction funcName
                Path = parentPath
                CurrentOps = [] }  // Blocks are stored in path, ops reset
        | _ ->
            failwithf "Cannot exit block - not in a block (focus: %A)" zipper.Focus

    /// Exit current function, returning to module level
    and exitFunction (zipper: MLIRZipper) : MLIRZipper * MLIRFunc =
        match zipper.Focus, zipper.Path with
        | InFunction funcName, EnteredFunction (parentPath, name, completedFuncs) ->
            // Create function with accumulated blocks/ops
            let func = {
                Name = name
                Parameters = []  // TODO: Track parameters in path
                ReturnType = Unit  // TODO: Track return type in path
                Blocks = []  // TODO: Accumulate blocks properly
                Attributes = []
            }
            { zipper with
                Focus = AtModule
                Path = parentPath
                CurrentOps = []
                State = MLIRState.setCurrentFunction None zipper.State },
            func
        | InBlock (funcName, label), _ ->
            // Exit block first, then function
            let zipper' = exitBlock zipper
            exitFunction zipper'
        | _ ->
            failwithf "Cannot exit function - not in a function (focus: %A)" zipper.Focus

    // ─────────────────────────────────────────────────────────────────
    // Witnessing - Recording observations of computation structure
    // The fold WITNESSES nodes; these functions RECORD what is observed
    // ─────────────────────────────────────────────────────────────────

    /// Witness an operation (record observation at current focus)
    let witnessOp (text: string) (results: (string * MLIRType) list) (zipper: MLIRZipper) : MLIRZipper =
        let op = { Text = text; Results = results }
        { zipper with CurrentOps = op :: zipper.CurrentOps }

    /// Witness an operation with single result
    let witnessOpWithResult (text: string) (resultSSA: string) (resultType: MLIRType) (zipper: MLIRZipper) : MLIRZipper =
        witnessOp text [(resultSSA, resultType)] zipper

    /// Witness a void operation (observation with no yielded value)
    let witnessVoidOp (text: string) (zipper: MLIRZipper) : MLIRZipper =
        witnessOp text [] zipper

    /// Witness a constant, yielding an SSA name
    let witnessConstant (value: int64) (ty: IntegerBitWidth) (zipper: MLIRZipper) : string * MLIRZipper =
        let ssaName, zipper' = yieldSSA zipper
        let tyStr = Serialize.integerBitWidth ty
        let text = sprintf "%s = arith.constant %d : %s" ssaName value tyStr
        ssaName, witnessOpWithResult text ssaName (Integer ty) zipper'

    /// Witness addressof for a string literal, yielding pointer SSA
    let witnessAddressOf (globalName: string) (zipper: MLIRZipper) : string * MLIRZipper =
        let ssaName, zipper' = yieldSSA zipper
        let text = sprintf "%s = llvm.mlir.addressof @%s : !llvm.ptr" ssaName globalName
        ssaName, witnessOpWithResult text ssaName Pointer zipper'

    /// Witness binary arithmetic operation
    let witnessArith (op: string) (lhs: string) (rhs: string) (ty: MLIRType) (zipper: MLIRZipper) : string * MLIRZipper =
        let ssaName, zipper' = yieldSSA zipper
        let tyStr = Serialize.mlirType ty
        let text = sprintf "%s = %s %s, %s : %s" ssaName op lhs rhs tyStr
        ssaName, witnessOpWithResult text ssaName ty zipper'

    /// Witness comparison operation
    let witnessCmpi (pred: string) (lhs: string) (rhs: string) (opType: MLIRType) (zipper: MLIRZipper) : string * MLIRZipper =
        let ssaName, zipper' = yieldSSA zipper
        let tyStr = Serialize.mlirType opType
        let text = sprintf "%s = arith.cmpi %s, %s, %s : %s" ssaName pred lhs rhs tyStr
        ssaName, witnessOpWithResult text ssaName (Integer I1) zipper'

    /// Witness function call
    let witnessCall (funcName: string) (args: string list) (argTypes: MLIRType list) (resultType: MLIRType) (zipper: MLIRZipper) : string * MLIRZipper =
        let ssaName, zipper' = yieldSSA zipper
        let argsStr = String.concat ", " args
        let typesStr = argTypes |> List.map Serialize.mlirType |> String.concat ", "
        let retStr = Serialize.mlirType resultType
        let text = sprintf "%s = llvm.call @%s(%s) : (%s) -> %s" ssaName funcName argsStr typesStr retStr
        ssaName, witnessOpWithResult text ssaName resultType zipper'

    /// Witness void function call
    let witnessCallVoid (funcName: string) (args: string list) (argTypes: MLIRType list) (zipper: MLIRZipper) : MLIRZipper =
        let argsStr = String.concat ", " args
        let typesStr = argTypes |> List.map Serialize.mlirType |> String.concat ", "
        let text = sprintf "llvm.call @%s(%s) : (%s) -> ()" funcName argsStr typesStr
        witnessVoidOp text zipper

    /// Witness syscall via inline assembly
    /// sysNum: SSA value for syscall number
    /// args: list of (ssaName, mlirTypeString) pairs for arguments
    /// resultType: MLIRType for return value
    let witnessSyscall (sysNum: string) (args: (string * string) list) (resultType: MLIRType) (zipper: MLIRZipper) : string * MLIRZipper =
        let ssaName, zipper' = yieldSSA zipper
        let argSSAs = args |> List.map fst
        let argTypes = args |> List.map snd
        let allSSAs = sysNum :: argSSAs
        let allTypes = "i64" :: argTypes  // syscall number is always i64
        let argsStr = String.concat ", " allSSAs
        // Linux x86-64 syscall: rax = syscall number, rdi/rsi/rdx = args
        let constraints =
            match List.length args with
            | 0 -> "={rax},{rax},~{rcx},~{r11},~{memory}"
            | 1 -> "={rax},{rax},{rdi},~{rcx},~{r11},~{memory}"
            | 2 -> "={rax},{rax},{rdi},{rsi},~{rcx},~{r11},~{memory}"
            | 3 -> "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}"
            | _ -> "={rax},{rax},{rdi},{rsi},{rdx},{r10},~{rcx},~{r11},~{memory}"
        let typesStr = String.concat ", " allTypes
        let retStr = Serialize.mlirType resultType
        let text = sprintf "%s = llvm.inline_asm has_side_effects \"syscall\", \"%s\" %s : (%s) -> %s"
                       ssaName constraints argsStr typesStr retStr
        ssaName, witnessOpWithResult text ssaName resultType zipper'

    /// Witness branch instruction
    let witnessBranch (target: string) (zipper: MLIRZipper) : MLIRZipper =
        let text = sprintf "llvm.br ^%s" target
        witnessVoidOp text zipper

    /// Witness conditional branch
    let witnessCondBranch (cond: string) (trueTarget: string) (falseTarget: string) (zipper: MLIRZipper) : MLIRZipper =
        let text = sprintf "llvm.cond_br %s, ^%s, ^%s" cond trueTarget falseTarget
        witnessVoidOp text zipper

    /// Witness return
    let witnessReturn (value: string) (ty: MLIRType) (zipper: MLIRZipper) : MLIRZipper =
        let tyStr = Serialize.mlirType ty
        let text = sprintf "llvm.return %s : %s" value tyStr
        witnessVoidOp text zipper

    /// Witness unreachable (for exit syscall paths)
    let witnessUnreachable (zipper: MLIRZipper) : MLIRZipper =
        witnessVoidOp "llvm.unreachable" zipper

    // ─────────────────────────────────────────────────────────────────
    // Extraction - Comonad extract: collapse accumulated context to final value
    // This is the ONLY place where MLIR text is actually produced
    // ─────────────────────────────────────────────────────────────────

    // Use Serialize.escape from MLIRTypes for string escaping

    /// Extract: comonad operation that collapses zipper to complete MLIR text
    /// This is the final observation - the accumulated context becomes a value
    let extract (zipper: MLIRZipper) : string =
        let sb = System.Text.StringBuilder()

        sb.AppendLine("module {") |> ignore

        // Emit globals (string literals first, then externs, then buffers)
        for glb in List.rev zipper.Globals do
            match glb with
            | StringLiteral (name, content, len) ->
                let escaped = Serialize.escape content
                sb.AppendLine(sprintf "  llvm.mlir.global internal constant @%s(\"%s\\00\") : !llvm.array<%d x i8>"
                    name escaped len) |> ignore
            | ExternFunc (name, signature) ->
                sb.AppendLine(sprintf "  llvm.func @%s%s attributes {sym_visibility = \"private\"}"
                    name signature) |> ignore
            | StaticBuffer (name, size) ->
                // Zero-initialized buffer - no initializer needed
                sb.AppendLine(sprintf "  llvm.mlir.global internal @%s() : !llvm.array<%d x i8>"
                    name size) |> ignore

        // Emit current operations (if any)
        for op in List.rev zipper.CurrentOps do
            sb.AppendLine(sprintf "  %s" op.Text) |> ignore

        sb.AppendLine("}") |> ignore

        sb.ToString()

    /// Get all emitted operations as text (without module wrapper)
    let getOperationsText (zipper: MLIRZipper) : string =
        zipper.CurrentOps
        |> List.rev
        |> List.map (fun op -> op.Text)
        |> String.concat "\n"

// ═══════════════════════════════════════════════════════════════════
// Dual-Zipper Transfer Type
// ═══════════════════════════════════════════════════════════════════

/// Result of transferring a PSG node to MLIR via zipper
type TransferResult =
    /// Produced an SSA value with type
    | TRValue of ssa: string * mlirType: string
    /// Produced no value (unit/void)
    | TRVoid
    /// Transfer failed with error
    | TRError of message: string

module TransferResult =
    let isValue = function TRValue _ -> true | _ -> false
    let isVoid = function TRVoid -> true | _ -> false
    let isError = function TRError _ -> true | _ -> false

    let getSSA = function TRValue (ssa, _) -> Some ssa | _ -> None
    let getType = function TRValue (_, t) -> Some t | _ -> None
