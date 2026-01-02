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
    /// Whether the function has internal linkage (default true for lambdas)
    IsInternal: bool
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
    /// EnteredFunction preserves parent function's ops, var bindings, and node SSAs for proper scoping
    /// Each function has its own SSA namespace - values from outer scopes are NOT valid in nested functions
    | EnteredFunction of parent: MLIRPath * funcName: string * funcParams: (string * MLIRType) list * funcReturnType: MLIRType * isInternal: bool * completedFuncs: MLIRFunc list * parentOps: MLIROp list * parentVarBindings: Map<string, string * string> * parentNodeSSAs: Map<string, string * string>
    | EnteredBlock of parent: MLIRPath * funcName: string * blockLabel: string * completedBlocks: MLIRBlock list * currentParams: (string * MLIRType) list * currentReturnType: MLIRType * isInternal: bool

// ═══════════════════════════════════════════════════════════════════
// MLIRZipper State - Functional state threaded through traversal
// ═══════════════════════════════════════════════════════════════════

/// Immutable state for MLIR composition (symmetric to EmissionState in PSGZipper)
type MLIRState = {
    /// SSA counter for unique value names
    SSACounter: int
    /// Label counter for unique block labels
    LabelCounter: int
    /// Lambda counter for unique lambda function names
    LambdaCounter: int
    /// Entry point Lambda node IDs (these become "main")
    /// Set during initialization from SemanticGraph.EntryPoints
    EntryPointLambdaIds: Set<int>
    /// Map from PSG NodeId to (SSA value name, MLIR type)
    /// Records what SSA value was produced for each PSG node
    NodeSSA: Map<string, string * string>
    /// Map from variable name to (SSA value name, MLIR type)
    /// Records SSA bindings for Lambda parameters and let bindings
    VarBindings: Map<string, string * string>
    /// String literals: content -> global name (for deduplication)
    StringLiterals: Map<string, string>
    /// External functions declared
    ExternalFuncs: Set<string>
    /// Current function name (for context)
    CurrentFunction: string option
    /// Map from function name to its MLIR return type string
    /// Used to get correct return type at call sites
    FuncReturnTypes: Map<string, string>
    /// Map from function name to its parameter count
    /// Used to filter out unit arguments at call sites
    FuncParamCounts: Map<string, int>
    /// Map from function name to its parameter types (MLIR type strings)
    /// Used to generate correct call signatures for both externs and internal functions
    FuncParamTypes: Map<string, string list>
}

module MLIRState =
    /// Create initial state with no entry point knowledge
    let create () : MLIRState = {
        SSACounter = 0
        LabelCounter = 0
        LambdaCounter = 0
        EntryPointLambdaIds = Set.empty
        NodeSSA = Map.empty
        VarBindings = Map.empty
        StringLiterals = Map.empty
        ExternalFuncs = Set.empty
        CurrentFunction = None
        FuncReturnTypes = Map.empty
        FuncParamCounts = Map.empty
        FuncParamTypes = Map.empty
    }

    /// Create state with entry point Lambda IDs
    /// These Lambdas will be named "main" instead of "lambda_N"
    let createWithEntryPoints (entryPointLambdaIds: Set<int>) : MLIRState = {
        SSACounter = 0
        LabelCounter = 0
        LambdaCounter = 0
        EntryPointLambdaIds = entryPointLambdaIds
        NodeSSA = Map.empty
        VarBindings = Map.empty
        StringLiterals = Map.empty
        ExternalFuncs = Set.empty
        CurrentFunction = None
        FuncReturnTypes = Map.empty
        FuncParamCounts = Map.empty
        FuncParamTypes = Map.empty
    }
    
    /// Record a function's return type
    let recordFuncReturnType (funcName: string) (returnType: string) (state: MLIRState) : MLIRState =
        { state with FuncReturnTypes = Map.add funcName returnType state.FuncReturnTypes }

    /// Look up a function's return type
    let lookupFuncReturnType (funcName: string) (state: MLIRState) : string option =
        Map.tryFind funcName state.FuncReturnTypes

    /// Record a function's parameter count
    let recordFuncParamCount (funcName: string) (paramCount: int) (state: MLIRState) : MLIRState =
        { state with FuncParamCounts = Map.add funcName paramCount state.FuncParamCounts }

    /// Look up a function's parameter count
    let lookupFuncParamCount (funcName: string) (state: MLIRState) : int option =
        Map.tryFind funcName state.FuncParamCounts

    /// Record a function's parameter types
    let recordFuncParamTypes (funcName: string) (paramTypes: string list) (state: MLIRState) : MLIRState =
        { state with FuncParamTypes = Map.add funcName paramTypes state.FuncParamTypes }

    /// Look up a function's parameter types
    let lookupFuncParamTypes (funcName: string) (state: MLIRState) : string list option =
        Map.tryFind funcName state.FuncParamTypes

    /// Check if a function is an extern
    let isExtern (funcName: string) (state: MLIRState) : bool =
        Set.contains funcName state.ExternalFuncs

    /// Parse an MLIR signature string and extract param types and return type
    /// e.g., "(i32, i64, i32) -> i32" -> (["i32"; "i64"; "i32"], "i32")
    let parseSignature (signature: string) : string list * string =
        // Find "->" to split params from return type
        match signature.IndexOf("->") with
        | -1 -> [], signature.Trim()
        | arrowIdx ->
            let paramsStr = signature.Substring(0, arrowIdx).Trim()
            let retStr = signature.Substring(arrowIdx + 2).Trim()
            // Extract params from "(param1, param2, ...)"
            let paramTypes =
                if paramsStr.StartsWith("(") && paramsStr.EndsWith(")") then
                    let inner = paramsStr.Substring(1, paramsStr.Length - 2).Trim()
                    if inner.Length = 0 then []
                    else inner.Split(',') |> Array.map (fun s -> s.Trim()) |> Array.toList
                else []
            paramTypes, retStr

    /// Yield next SSA name on demand (codata yield)
    let yieldSSA (state: MLIRState) : string * MLIRState =
        let name = sprintf "%%v%d" state.SSACounter
        name, { state with SSACounter = state.SSACounter + 1 }

    /// Yield next block label on demand (codata yield)
    let yieldLabel (state: MLIRState) : string * MLIRState =
        let name = sprintf "bb%d" state.LabelCounter
        name, { state with LabelCounter = state.LabelCounter + 1 }

    /// Yield next lambda function name on demand (codata yield)
    /// Entry point Lambdas become "main", others get unique names
    let yieldLambdaName (state: MLIRState) : string * MLIRState =
        let name = sprintf "lambda_%d" state.LambdaCounter
        name, { state with LambdaCounter = state.LambdaCounter + 1 }

    /// Yield lambda name for a specific node ID
    /// Entry point Lambdas (in EntryPointLambdaIds) become "main"
    let yieldLambdaNameForNode (nodeId: int) (state: MLIRState) : string * MLIRState =
        if Set.contains nodeId state.EntryPointLambdaIds then
            // This Lambda is an entry point - name it main
            "main", state  // Don't increment counter for main
        else
            let name = sprintf "lambda_%d" state.LambdaCounter
            name, { state with LambdaCounter = state.LambdaCounter + 1 }

    /// Bind SSA observation to a PSG node identity
    let bindNodeSSA (nodeId: string) (ssaName: string) (mlirType: string) (state: MLIRState) : MLIRState =
        { state with NodeSSA = Map.add nodeId (ssaName, mlirType) state.NodeSSA }

    /// Recall a prior SSA observation for a PSG node
    let recallNodeSSA (nodeId: string) (state: MLIRState) : (string * string) option =
        Map.tryFind nodeId state.NodeSSA

    /// Bind a variable name to an SSA value (for Lambda parameters)
    let bindVar (varName: string) (ssaName: string) (mlirType: string) (state: MLIRState) : MLIRState =
        { state with VarBindings = Map.add varName (ssaName, mlirType) state.VarBindings }

    /// Recall SSA value for a variable name
    let recallVar (varName: string) (state: MLIRState) : (string * string) option =
        Map.tryFind varName state.VarBindings

    /// Observe a string literal requirement (coeffect - context need)
    let observeStringLiteral (content: string) (state: MLIRState) : string * MLIRState =
        match Map.tryFind content state.StringLiterals with
        | Some name -> name, state
        | None ->
            let name = sprintf "str%d" state.StringLiterals.Count
            name, { state with StringLiterals = Map.add content name state.StringLiterals }

    /// Observe an external function requirement with signature (coeffect - context need)
    /// Parses the signature to extract param types and return type for later use
    let observeExternFuncWithSignature (name: string) (signature: string) (state: MLIRState) : MLIRState =
        let paramTypes, retType = parseSignature signature
        let state' = { state with ExternalFuncs = Set.add name state.ExternalFuncs }
        let state'' = recordFuncParamTypes name paramTypes state'
        let state''' = recordFuncParamCount name (List.length paramTypes) state''
        recordFuncReturnType name retType state'''

    /// Observe an external function requirement (coeffect - context need)
    /// Simple version that just tracks the function name
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
    /// Completed lambda functions (accumulated during traversal)
    CompletedFunctions: MLIRFunc list
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
        CompletedFunctions = []
    }

    /// Create with initial state (for continuing from previous context)
    let createWithState (state: MLIRState) : MLIRZipper = {
        Focus = AtModule
        Path = Top
        CurrentOps = []
        State = state
        Globals = []
        CompletedFunctions = []
    }

    /// Create with entry point Lambda IDs
    /// These Lambdas will be named "main" instead of "lambda_N"
    let createWithEntryPoints (entryPointLambdaIds: Set<int>) : MLIRZipper = {
        Focus = AtModule
        Path = Top
        CurrentOps = []
        State = MLIRState.createWithEntryPoints entryPointLambdaIds
        Globals = []
        CompletedFunctions = []
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

    /// Bind a variable name to an SSA value (for Lambda parameters)
    let bindVar (varName: string) (ssaName: string) (mlirType: string) (zipper: MLIRZipper) : MLIRZipper =
        mapState (MLIRState.bindVar varName ssaName mlirType) zipper

    /// Recall SSA value for a variable name
    let recallVar (varName: string) (zipper: MLIRZipper) : (string * string) option =
        MLIRState.recallVar varName zipper.State

    /// Look up a function's tracked return type
    let lookupFuncReturnType (funcName: string) (zipper: MLIRZipper) : string option =
        MLIRState.lookupFuncReturnType funcName zipper.State

    /// Look up a function's parameter count
    let lookupFuncParamCount (funcName: string) (zipper: MLIRZipper) : int option =
        MLIRState.lookupFuncParamCount funcName zipper.State

    /// Look up a function's parameter types
    let lookupFuncParamTypes (funcName: string) (zipper: MLIRZipper) : string list option =
        MLIRState.lookupFuncParamTypes funcName zipper.State

    /// Check if a function is an extern
    let isExtern (funcName: string) (zipper: MLIRZipper) : bool =
        MLIRState.isExtern funcName zipper.State

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
    /// Also parses and stores the signature for correct call-site generation
    let observeExternFunc (name: string) (signature: string) (zipper: MLIRZipper) : MLIRZipper =
        let newState = MLIRState.observeExternFuncWithSignature name signature zipper.State
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

    /// Enter a function definition with explicit visibility control
    let enterFunctionWithVisibility (name: string) (parameters: (string * MLIRType) list) (returnType: MLIRType) (isInternal: bool) (zipper: MLIRZipper) : MLIRZipper =
        // Record param types and return type for this function
        let paramTypes = parameters |> List.map (snd >> Serialize.mlirType)
        let stateWithParamTypes =
            zipper.State
            |> MLIRState.recordFuncParamTypes name paramTypes
            |> MLIRState.recordFuncParamCount name (List.length parameters)
            |> MLIRState.recordFuncReturnType name (Serialize.mlirType returnType)
        match zipper.Focus with
        | AtModule ->
            // Enter from module level - preserve NodeSSA (module-level bindings should remain visible)
            // Save NodeSSA in path so it's restored when we exit back to module level
            // Reset SSA counter for function-local numbering
            { zipper with
                Focus = InFunction name
                Path = EnteredFunction (zipper.Path, name, parameters, returnType, isInternal, [], [], Map.empty, stateWithParamTypes.NodeSSA)
                CurrentOps = []
                State = { stateWithParamTypes with
                            VarBindings = Map.empty
                            CurrentFunction = Some name
                            SSACounter = 0 } }  // Reset SSA counter for function-local numbering
        | InFunction _ ->
            // Already in a function - nested lambda
            // CRITICAL: Save current ops, var bindings, AND node SSAs in path so they're restored when we exit
            // Filter NodeSSA to only keep function references (@xxx) and pipe markers ($pipe:) - value SSAs are function-local
            // Also reset SSA counter so nested functions have their own %v0, %v1, etc.
            let globalOnlyNodeSSA =
                stateWithParamTypes.NodeSSA
                |> Map.filter (fun _ (ssa, _) -> ssa.StartsWith("@") || ssa.StartsWith("$pipe:"))
            { zipper with
                Focus = InFunction name
                Path = EnteredFunction (zipper.Path, name, parameters, returnType, isInternal, [], zipper.CurrentOps, stateWithParamTypes.VarBindings, stateWithParamTypes.NodeSSA)
                CurrentOps = []
                State = { stateWithParamTypes with
                            VarBindings = Map.empty
                            NodeSSA = globalOnlyNodeSSA
                            CurrentFunction = Some name
                            SSACounter = 0 } }  // Reset SSA counter for nested function
        | _ ->
            // Can only enter function from module level or inside another function
            failwithf "Cannot enter function '%s' - invalid focus: %A" name zipper.Focus

    /// Enter a function definition (internal by default for lambdas)
    let enterFunction (name: string) (parameters: (string * MLIRType) list) (returnType: MLIRType) (zipper: MLIRZipper) : MLIRZipper =
        enterFunctionWithVisibility name parameters returnType true zipper

    /// Enter a basic block within current function
    let enterBlock (label: string) (arguments: (string * MLIRType) list) (zipper: MLIRZipper) : MLIRZipper =
        match zipper.Focus, zipper.Path with
        | InFunction funcName, EnteredFunction (parentPath, _, funcParams, retTy, isInternal, completedFuncs, _parentOps, _parentVarBindings, _parentNodeSSAs) ->
            // Save current ops as we enter a new block
            { zipper with
                Focus = InBlock (funcName, label)
                Path = EnteredBlock (zipper.Path, funcName, label, [], funcParams, retTy, isInternal)
                CurrentOps = [] }
        | InBlock (funcName, prevLabel), EnteredBlock (parentPath, fn, _, completedBlocks, funcParams, retTy, isInternal) ->
            // Finish previous block, enter new one
            let prevBlock = {
                Label = prevLabel
                Arguments = []
                Operations = List.rev zipper.CurrentOps
            }
            { zipper with
                Focus = InBlock (funcName, label)
                Path = EnteredBlock (parentPath, fn, label, prevBlock :: completedBlocks, funcParams, retTy, isInternal)
                CurrentOps = [] }
        | _ ->
            failwithf "Cannot enter block '%s' - not in a function" label

    /// Exit current block, returning to function level
    /// Note: Uses 'rec' to enable mutual recursion with exitFunction
    let rec exitBlock (zipper: MLIRZipper) : MLIRZipper =
        match zipper.Focus, zipper.Path with
        | InBlock (funcName, label), EnteredBlock (parentPath, fn, _, completedBlocks, funcParams, retTy, _isInternal) ->
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

    /// Exit current function, returning to module level or parent function
    and exitFunction (zipper: MLIRZipper) : MLIRZipper * MLIRFunc =
        match zipper.Focus, zipper.Path with
        | InFunction funcName, EnteredFunction (parentPath, name, funcParams, funcReturnType, isInternal, completedFuncs, parentOps, parentVarBindings, parentNodeSSAs) ->
            // Create function with accumulated ops as single entry block
            let entryBlock = {
                Label = "entry"
                Arguments = []
                Operations = List.rev zipper.CurrentOps  // Reverse to get correct order
            }
            let func = {
                Name = name
                Parameters = funcParams
                ReturnType = funcReturnType
                Blocks = [entryBlock]
                Attributes = []
                IsInternal = isInternal  // Use observed visibility
            }
            // Determine new focus based on parentPath
            // If parentPath is Top, we return to AtModule
            // If parentPath is EnteredFunction, we return to InFunction (and restore CurrentFunction)
            let newFocus, newState =
                match parentPath with
                | Top ->
                    AtModule, { MLIRState.setCurrentFunction None zipper.State with VarBindings = parentVarBindings; NodeSSA = parentNodeSSAs }
                | EnteredFunction (_, parentFuncName, _, _, _, _, _, _, _) ->
                    InFunction parentFuncName, { MLIRState.setCurrentFunction (Some parentFuncName) zipper.State with VarBindings = parentVarBindings; NodeSSA = parentNodeSSAs }
                | EnteredBlock (_, parentFuncName, _, _, _, _, _) ->
                    InFunction parentFuncName, { MLIRState.setCurrentFunction (Some parentFuncName) zipper.State with VarBindings = parentVarBindings; NodeSSA = parentNodeSSAs }
            { zipper with
                Focus = newFocus
                Path = parentPath
                CurrentOps = parentOps  // CRITICAL: Restore parent function's ops
                State = newState },      // CRITICAL: State now includes restored VarBindings and NodeSSAs
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

        // Check tracked parameter count to filter out extra unit arguments
        let actualArgs, actualArgTypes =
            match MLIRState.lookupFuncParamCount funcName zipper'.State with
            | Some paramCount when paramCount < List.length args ->
                // Function has fewer parameters than arguments - truncate to match
                // This handles cases where PSG has unit arguments but function was defined without them
                List.take paramCount args, List.take paramCount argTypes
            | _ ->
                args, argTypes

        // Check if we need to convert argument types to match function declaration
        // This handles polymorphic functions where TVar -> Pointer but call site has concrete types
        let argsStr, typesStr, zipper'' =
            match MLIRState.lookupFuncParamTypes funcName zipper'.State with
            | Some funcParamTypes when List.length funcParamTypes = List.length actualArgs ->
                // Have declared param types - check for mismatches and insert conversions
                let convertedArgs, convertedTypes, z =
                    List.zip3 actualArgs actualArgTypes funcParamTypes
                    |> List.fold (fun (argsAcc, typesAcc, z) (arg, argType, declaredTypeStr) ->
                        let argTypeStr = Serialize.mlirType argType
                        if argTypeStr = declaredTypeStr then
                            // Types match - no conversion needed
                            argsAcc @ [arg], typesAcc @ [declaredTypeStr], z
                        elif declaredTypeStr = "!llvm.ptr" && argTypeStr.StartsWith("i") then
                            // Function expects ptr but we have integer - insert inttoptr
                            let convSSA, z' = yieldSSA z
                            let convText = sprintf "%s = llvm.inttoptr %s : %s to !llvm.ptr" convSSA arg argTypeStr
                            let z'' = witnessOpWithResult convText convSSA Pointer z'
                            argsAcc @ [convSSA], typesAcc @ ["!llvm.ptr"], z''
                        elif argTypeStr = "!llvm.ptr" && declaredTypeStr.StartsWith("i") then
                            // Function expects integer but we have ptr - insert ptrtoint
                            let convSSA, z' = yieldSSA z
                            let convText = sprintf "%s = llvm.ptrtoint %s : !llvm.ptr to %s" convSSA arg declaredTypeStr
                            let z'' = witnessOpWithResult convText convSSA (Serialize.deserializeType declaredTypeStr) z'
                            argsAcc @ [convSSA], typesAcc @ [declaredTypeStr], z''
                        else
                            // Other mismatch - use actual types
                            argsAcc @ [arg], typesAcc @ [argTypeStr], z
                    ) ([], [], zipper')
                String.concat ", " convertedArgs, String.concat ", " convertedTypes, z
            | _ ->
                // No declared param types - use actual types
                String.concat ", " actualArgs,
                actualArgTypes |> List.map Serialize.mlirType |> String.concat ", ",
                zipper'

        // Use tracked return type if available (for correct lowered types)
        // Otherwise fall back to the PSG-derived resultType
        let actualRetType, actualRetStr =
            match MLIRState.lookupFuncReturnType funcName zipper''.State with
            | Some trackedRetStr ->
                // Parse the tracked type back to MLIRType for witnessOpWithResult
                Serialize.deserializeType trackedRetStr, trackedRetStr
            | None ->
                resultType, Serialize.mlirType resultType
        let text = sprintf "%s = llvm.call @%s(%s) : (%s) -> %s" ssaName funcName argsStr typesStr actualRetStr
        ssaName, witnessOpWithResult text ssaName actualRetType zipper''

    /// Witness void function call
    let witnessCallVoid (funcName: string) (args: string list) (argTypes: MLIRType list) (zipper: MLIRZipper) : MLIRZipper =
        let argsStr = String.concat ", " args
        let typesStr = argTypes |> List.map Serialize.mlirType |> String.concat ", "
        let text = sprintf "llvm.call @%s(%s) : (%s) -> ()" funcName argsStr typesStr
        witnessVoidOp text zipper

    /// Witness indirect function call (call through function pointer)
    let witnessIndirectCall (funcPtr: string) (args: string list) (argTypes: MLIRType list) (resultType: MLIRType) (zipper: MLIRZipper) : string * MLIRZipper =
        // First, emit addressof for any arguments that are function symbols (@name)
        // These need to be converted to SSA values before use
        let zipper', argsResolved =
            args
            |> List.fold (fun (z, acc) arg ->
                if arg.StartsWith("@") then
                    // Function symbol - emit addressof to get SSA value
                    let addrSSA, z' = yieldSSA z
                    let addrText = sprintf "%s = llvm.mlir.addressof %s : !llvm.ptr" addrSSA arg
                    let z'' = witnessOpWithResult addrText addrSSA Pointer z'
                    z'', addrSSA :: acc
                else
                    z, arg :: acc
            ) (zipper, [])
            |> fun (z, acc) -> z, List.rev acc

        let ssaName, zipper'' = yieldSSA zipper'
        let argsStr = String.concat ", " argsResolved
        let typesStr = argTypes |> List.map Serialize.mlirType |> String.concat ", "
        let retStr = Serialize.mlirType resultType
        // Indirect call uses the function pointer SSA value, not a symbol name
        let text = sprintf "%s = llvm.call %s(%s) : !llvm.ptr, (%s) -> %s" ssaName funcPtr argsStr typesStr retStr
        ssaName, witnessOpWithResult text ssaName resultType zipper''

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
    // Lambda Support
    // ─────────────────────────────────────────────────────────────────

    /// Yield next lambda function name on demand (codata yield)
    let yieldLambdaName (zipper: MLIRZipper) : string * MLIRZipper =
        let name, newState = MLIRState.yieldLambdaName zipper.State
        name, { zipper with State = newState }

    /// Yield lambda name for a specific node ID
    /// Entry point Lambdas (in EntryPointLambdaIds) become "main"
    let yieldLambdaNameForNode (nodeId: int) (zipper: MLIRZipper) : string * MLIRZipper =
        let name, newState = MLIRState.yieldLambdaNameForNode nodeId zipper.State
        name, { zipper with State = newState }

    /// Add a completed lambda function to the zipper
    /// Also records the function's return type and parameter count for call site generation
    let addCompletedFunction (func: MLIRFunc) (zipper: MLIRZipper) : MLIRZipper =
        let returnTypeStr = Serialize.mlirType func.ReturnType
        let paramCount = List.length func.Parameters
        let state1 = MLIRState.recordFuncReturnType func.Name returnTypeStr zipper.State
        let state2 = MLIRState.recordFuncParamCount func.Name paramCount state1
        { zipper with
            CompletedFunctions = func :: zipper.CompletedFunctions
            State = state2 }

    /// Witness a lambda function definition
    /// For non-capturing lambdas, generates a function and returns its name as a pointer
    /// parameters: list of (name, type) for lambda parameters
    /// bodyOps: list of operations that form the lambda body
    /// returnType: the return type of the lambda
    let witnessLambda (parameters: (string * MLIRType) list) (bodyOps: MLIROp list) (returnType: MLIRType) (zipper: MLIRZipper) : string * MLIRZipper =
        // Generate unique lambda name
        let lambdaName, zipper1 = yieldLambdaName zipper
        
        // Create the function definition
        let func: MLIRFunc = {
            Name = lambdaName
            Parameters = parameters
            ReturnType = returnType
            Blocks = [{
                Label = "entry"
                Arguments = []
                Operations = bodyOps
            }]
            Attributes = []
            IsInternal = true
        }
        
        // Add to completed functions
        let zipper2 = addCompletedFunction func zipper1
        
        // Get address of the lambda function as a pointer
        let ptrSSA, zipper3 = yieldSSA zipper2
        let text = sprintf "%s = llvm.mlir.addressof @%s : !llvm.ptr" ptrSSA lambdaName
        let zipper4 = witnessOpWithResult text ptrSSA Pointer zipper3
        
        ptrSSA, zipper4

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

        // Emit completed lambda functions
        for func in List.rev zipper.CompletedFunctions do
            let paramsStr = 
                func.Parameters 
                |> List.mapi (fun i (name, ty) -> sprintf "%%arg%d: %s" i (Serialize.mlirType ty))
                |> String.concat ", "
            let retStr = Serialize.mlirType func.ReturnType
            let attrs = 
                if List.isEmpty func.Attributes then ""
                else sprintf " attributes {%s}" (String.concat ", " (List.map (fun a -> sprintf "sym_visibility = \"%s\"" a) func.Attributes))
            let visibility = if func.IsInternal then "internal " else ""
            sb.AppendLine(sprintf "  llvm.func %s@%s(%s) -> %s%s {" visibility func.Name paramsStr retStr attrs) |> ignore
            // For MLIR LLVM dialect, entry block should not have explicit label when function has parameters
            let hasParams = not (List.isEmpty func.Parameters)
            for i, block in List.indexed func.Blocks do
                // Skip block label for entry block (first block) when function has parameters
                if i = 0 && hasParams then
                    () // No block label - parameters serve as implicit entry block arguments
                else
                    sb.AppendLine(sprintf "  ^%s:" block.Label) |> ignore
                for op in block.Operations do
                    sb.AppendLine(sprintf "    %s" op.Text) |> ignore
            sb.AppendLine("  }") |> ignore

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
    /// Built-in operator/function (handled inline at application sites)
    | TRBuiltin of name: string

module TransferResult =
    let isValue = function TRValue _ -> true | _ -> false
    let isVoid = function TRVoid -> true | _ -> false
    let isError = function TRError _ -> true | _ -> false
    let isBuiltin = function TRBuiltin _ -> true | _ -> false

    let getSSA = function TRValue (ssa, _) -> Some ssa | _ -> None
    let getType = function TRValue (_, t) -> Some t | _ -> None
    let getBuiltinName = function TRBuiltin name -> Some name | _ -> None
