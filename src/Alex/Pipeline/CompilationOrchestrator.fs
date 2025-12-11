module Alex.Pipeline.CompilationOrchestrator

open System
open System.IO
open System.Text
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Text
open FSharp.Compiler.Symbols
open FSharp.Compiler.Symbols.FSharpExprPatterns
open Core.IngestionPipeline
open Core.FCS.ProjectContext
open Core.Utilities.IntermediateWriter
open Core.PSG.Types
open Alex.Pipeline.CompilationTypes
open Alex.Bindings.BindingTypes
open Alex.CodeGeneration.MLIRBuilder
open Alex.Traversal.PSGZipper
open Alex.Patterns.PSGPatterns

// ===================================================================
// Pipeline Integration Types
// ===================================================================

/// Extended compilation result with outputs
type CompilationResult = {
    Success: bool
    Diagnostics: CompilerError list
    Statistics: CompilationStatistics
    Intermediates: IntermediateOutputs
}

/// Intermediate file outputs
and IntermediateOutputs = {
    ProjectAnalysis: string option
    PSGRepresentation: string option
    ReachabilityAnalysis: string option
    PrunedSymbols: string option
}

/// Empty intermediates for initialization
let emptyIntermediates = {
    ProjectAnalysis = None
    PSGRepresentation = None
    ReachabilityAnalysis = None
    PrunedSymbols = None
}

// ===================================================================
// Pipeline Configuration Bridge
// ===================================================================

/// Convert compilation config to ingestion pipeline config
let createPipelineConfig (config: CompilationConfig) (intermediatesDir: string option) : PipelineConfig = {
    CacheStrategy = Balanced
    TemplateName = None
    CustomTemplateDir = None
    EnableCouplingAnalysis = config.EnableReachabilityAnalysis
    EnableMemoryOptimization = config.EnableStackAllocation
    OutputIntermediates = config.PreserveIntermediateASTs
    IntermediatesDir = intermediatesDir
}

// ===================================================================
// Type Conversions
// ===================================================================

/// Convert IngestionPipeline.DiagnosticSeverity to CompilationTypes.ErrorSeverity
let convertSeverity (severity: DiagnosticSeverity) : ErrorSeverity =
    match severity with
    | DiagnosticSeverity.Error -> ErrorSeverity.Error
    | DiagnosticSeverity.Warning -> ErrorSeverity.Warning
    | DiagnosticSeverity.Info -> ErrorSeverity.Info

/// Convert IngestionPipeline.Diagnostic to CompilerError
let convertDiagnostic (diag: Diagnostic) : CompilerError = {
    Phase = "Pipeline"
    Message = diag.Message
    Location = diag.Location
    Severity = convertSeverity diag.Severity
}

// ===================================================================
// Progress Reporting
// ===================================================================

/// Report compilation phase progress
let reportPhase (progress: ProgressCallback) (phase: CompilationPhase) (message: string) =
    progress phase message

// ===================================================================
// Statistics Collection
// ===================================================================

/// Generate meaningful compilation statistics from pipeline results
let generateStatistics (pipelineResult: PipelineResult) (startTime: DateTime) : CompilationStatistics =
    let endTime = DateTime.UtcNow
    let duration = endTime - startTime
    
    let totalFiles = 
        match pipelineResult.ProjectResults with
        | Some projectResults -> projectResults.CompilationOrder.Length
        | None -> 0
    
    let totalSymbols =
        match pipelineResult.ReachabilityAnalysis with
        | Some analysis -> analysis.PruningStatistics.TotalSymbols
        | None -> 0
    
    let reachableSymbols =
        match pipelineResult.ReachabilityAnalysis with
        | Some analysis -> analysis.PruningStatistics.ReachableSymbols
        | None -> 0
    
    let eliminatedSymbols =
        match pipelineResult.ReachabilityAnalysis with
        | Some analysis -> analysis.PruningStatistics.EliminatedSymbols
        | None -> 0
    
    {
        TotalFiles = totalFiles
        TotalSymbols = totalSymbols
        ReachableSymbols = reachableSymbols
        EliminatedSymbols = eliminatedSymbols
        CompilationTimeMs = float duration.TotalMilliseconds
    }

// ===================================================================
// MLIR Generation via Alex
// ===================================================================

/// Result of MLIR generation including any errors
type MLIRGenerationResult = {
    Content: string
    Errors: CompilerError list
    HasErrors: bool
}

/// Collect emission errors during MLIR generation
module EmissionErrors =
    let mutable private errors : CompilerError list = []

    let reset () = errors <- []
    let add (err: CompilerError) = errors <- err :: errors
    let toCompilerErrors () = errors |> List.rev
    let hasErrors () = not (List.isEmpty errors)

// ===================================================================
// PSG Node Emission - Local pattern matching at each node
// ===================================================================

/// Maximum depth for function inlining to prevent infinite recursion
let [<Literal>] MaxInliningDepth = 100

/// Emission context accumulates MLIR as we traverse
type EmissionContext = {
    Builder: StringBuilder
    SSACounter: int
    LabelCounter: int
    StringLiterals: Map<uint32, string * string>  // hash -> (content, globalName)
    NodeValues: Map<string, string * MLIRType>  // nodeId -> (ssaValue, type)
    IndentLevel: int
    CurrentFunctionName: string option
    Graph: ProgramSemanticGraph
    /// Track function bodies currently being inlined to detect infinite recursion
    InliningStack: Set<string>
    /// Current inlining depth for hard limit
    InliningDepth: int
    /// Type substitution map for generic instantiation during inlining
    /// Maps type parameter names (e.g., "^a") to concrete type names (e.g., "Microsoft.FSharp.Core.string")
    TypeSubstitutions: Map<string, string>
    /// Baker member body mappings (Phase 4: post-reachability type resolution)
    /// Used to find function bodies for inlining when PSG symbol correlation fails
    MemberBodies: Map<string, Baker.Types.MemberBodyMapping>
    /// Parameter bindings for inlined functions - maps parameter names to (ssa, type)
    /// When inlining, arguments are bound to parameters before emitting the body
    ParameterBindings: Map<string, string * MLIRType>
}

module EmissionContext =
    let create (psg: ProgramSemanticGraph) (memberBodies: Map<string, Baker.Types.MemberBodyMapping>) = {
        Builder = StringBuilder()
        SSACounter = 0
        LabelCounter = 0
        StringLiterals = Map.empty
        NodeValues = Map.empty
        IndentLevel = 0
        CurrentFunctionName = None
        Graph = psg
        InliningStack = Set.empty
        InliningDepth = 0
        TypeSubstitutions = Map.empty
        MemberBodies = memberBodies
        ParameterBindings = Map.empty
    }

    /// Add type substitutions for inlining a generic function with concrete arguments
    let withTypeSubstitutions (subs: Map<string, string>) (ctx: EmissionContext) : EmissionContext =
        { ctx with TypeSubstitutions = Map.fold (fun acc k v -> Map.add k v acc) ctx.TypeSubstitutions subs }

    /// Clear type substitutions (when exiting an inlined scope)
    let clearTypeSubstitutions (ctx: EmissionContext) : EmissionContext =
        { ctx with TypeSubstitutions = Map.empty }

    /// Apply type substitutions to a type name
    /// If the type is a generic parameter (^a), look it up in the substitution map
    let substituteType (typeName: string) (ctx: EmissionContext) : string =
        if typeName.StartsWith("^") then
            match Map.tryFind typeName ctx.TypeSubstitutions with
            | Some concrete -> concrete
            | None -> typeName  // No substitution available
        else
            typeName

    /// Push a function onto the inlining stack, returning None if already present (cycle detected)
    let tryPushInlining (funcName: string) (ctx: EmissionContext) : EmissionContext option =
        if ctx.InliningDepth >= MaxInliningDepth then
            printfn "[EMIT] ERROR: Maximum inlining depth (%d) exceeded at %s" MaxInliningDepth funcName
            None
        elif Set.contains funcName ctx.InliningStack then
            printfn "[EMIT] ERROR: Infinite recursion detected - %s is already being inlined" funcName
            printfn "[EMIT]   Current inlining stack: %A" (Set.toList ctx.InliningStack)
            None
        else
            Some { ctx with InliningStack = Set.add funcName ctx.InliningStack; InliningDepth = ctx.InliningDepth + 1 }

    /// Pop a function from the inlining stack
    let popInlining (funcName: string) (ctx: EmissionContext) : EmissionContext =
        { ctx with InliningStack = Set.remove funcName ctx.InliningStack; InliningDepth = ctx.InliningDepth - 1 }

    /// Bind parameter names to argument values for function inlining
    /// Returns a new context with the parameter bindings added
    let withParameterBindings (paramNames: string list) (argVals: (string * MLIRType) list) (ctx: EmissionContext) : EmissionContext =
        let bindings =
            List.zip paramNames argVals
            |> List.fold (fun acc (name, (ssa, ty)) ->
                if ssa <> "" then
                    printfn "[EMIT] Binding parameter '%s' to %s" name ssa
                    Map.add name (ssa, ty) acc
                else
                    acc) ctx.ParameterBindings
        { ctx with ParameterBindings = bindings }

    /// Clear parameter bindings when exiting an inlined scope
    let clearParameterBindings (ctx: EmissionContext) : EmissionContext =
        { ctx with ParameterBindings = Map.empty }

    /// Look up a parameter by name
    let lookupParameter (name: string) (ctx: EmissionContext) =
        Map.tryFind name ctx.ParameterBindings

    let indent (ctx: EmissionContext) =
        String.replicate ctx.IndentLevel "  "

    let emit (text: string) (ctx: EmissionContext) =
        ctx.Builder.Append(text) |> ignore
        ctx

    let emitLine (text: string) (ctx: EmissionContext) =
        ctx.Builder.AppendLine(indent ctx + text) |> ignore
        ctx

    let emitLineRaw (text: string) (ctx: EmissionContext) =
        ctx.Builder.AppendLine(text) |> ignore
        ctx

    let nextSSA (ctx: EmissionContext) =
        let name = sprintf "%%v%d" ctx.SSACounter
        name, { ctx with SSACounter = ctx.SSACounter + 1 }

    let nextLabel (ctx: EmissionContext) =
        let name = sprintf "bb%d" ctx.LabelCounter
        name, { ctx with LabelCounter = ctx.LabelCounter + 1 }

    let registerStringLiteral (content: string) (ctx: EmissionContext) =
        let hash = uint32 (content.GetHashCode())
        match Map.tryFind hash ctx.StringLiterals with
        | Some (_, name) -> name, ctx
        | None ->
            let name = sprintf "@str%d" ctx.StringLiterals.Count
            name, { ctx with StringLiterals = Map.add hash (content, name) ctx.StringLiterals }

    let recordNodeValue (nodeId: string) (ssa: string) (ty: MLIRType) (ctx: EmissionContext) =
        { ctx with NodeValues = Map.add nodeId (ssa, ty) ctx.NodeValues }

    let lookupNodeValue (nodeId: string) (ctx: EmissionContext) =
        Map.tryFind nodeId ctx.NodeValues

    let withIndent (ctx: EmissionContext) =
        { ctx with IndentLevel = ctx.IndentLevel + 1 }

    let withoutIndent (ctx: EmissionContext) =
        { ctx with IndentLevel = max 0 (ctx.IndentLevel - 1) }

    let getText (ctx: EmissionContext) =
        ctx.Builder.ToString()

/// Extract the full type name from an FSharpType for overload matching
let getTypeName (ftype: FSharpType option) : string option =
    match ftype with
    | None -> None
    | Some t ->
        try
            if t.HasTypeDefinition then
                // Try full name first
                match t.TypeDefinition.TryFullName with
                | Some name -> Some name
                | None ->
                    // Fallback: use FullName property which may throw for abbreviations
                    try Some t.TypeDefinition.FullName
                    with _ ->
                        // Last resort: use DisplayName
                        Some t.TypeDefinition.DisplayName
            elif t.IsGenericParameter then
                Some (sprintf "^%s" t.GenericParameter.Name)
            else
                Some (t.Format(FSharp.Compiler.Symbols.FSharpDisplayContext.Empty))
        with _ -> None

/// Select the correct overload from candidates based on argument types from PSG
/// Uses type substitution from context to resolve generic type parameters at call site
let selectOverload (candidates: SRTPOverloadCandidate list) (argNodes: PSGNode list) (ctx: EmissionContext) : SRTPOverloadCandidate option =
    // Get argument type names from PSG nodes, applying type substitution
    let argTypeNames =
        argNodes
        |> List.map (fun node ->
            match getTypeName node.Type with
            | Some typeName ->
                // Apply type substitution for generic parameters
                Some (EmissionContext.substituteType typeName ctx)
            | None -> None)

    printfn "[EMIT] MultipleOverloads: argTypes=%A (after substitution)" argTypeNames

    // Find a candidate whose parameter types match the argument types
    candidates
    |> List.tryFind (fun candidate ->
        let paramTypes = candidate.ParameterTypeNames
        printfn "[EMIT]   Checking candidate: %s with params=%A" candidate.TargetMethodFullName paramTypes

        // Skip the first parameter (the WritableString receiver) and match the rest
        // For op_Dollar, params are [WritableString; string|NativeStr]
        // And args are [WritableString; string|NativeStr]
        if paramTypes.Length <> argTypeNames.Length then
            false
        else
            // Match each parameter type against argument type
            List.zip paramTypes argTypeNames
            |> List.forall (fun (paramType, argType) ->
                match argType with
                | None -> true  // Unknown arg type - don't disqualify
                | Some at ->
                    // Check for substring match (e.g., "string" in "Microsoft.FSharp.Core.string")
                    let matches = paramType.Contains(at) || at.Contains(paramType)
                    printfn "[EMIT]     param=%s vs arg=%s => %b" paramType at matches
                    matches))

/// Build type substitution map from function parameters to concrete argument types
/// For a call like `Write "Hello"` where Write has param `s : ^a`:
/// - funcNode.Symbol has the function's type (e.g., ^a -> unit)
/// - argNodes have concrete types (e.g., string)
/// Returns a map from type parameter names (^a) to concrete types (Microsoft.FSharp.Core.string)
let buildTypeSubstitutions (funcNode: PSGNode) (argNodes: PSGNode list) : Map<string, string> =
    match funcNode.Symbol with
    | Some (:? FSharp.Compiler.Symbols.FSharpMemberOrFunctionOrValue as mfv) ->
        // Get the function's parameter types
        let paramGroups = mfv.CurriedParameterGroups |> Seq.collect id |> Seq.toList

        // Build map from generic parameter names to concrete argument types
        if paramGroups.Length <> argNodes.Length then
            Map.empty
        else
            List.zip paramGroups argNodes
            |> List.choose (fun (param, argNode) ->
                let paramType = param.Type
                if paramType.IsGenericParameter then
                    let paramName = sprintf "^%s" paramType.GenericParameter.Name
                    match getTypeName argNode.Type with
                    | Some argTypeName when not (argTypeName.StartsWith("^")) ->
                        // Concrete type - add substitution
                        printfn "[EMIT] TypeSub: %s -> %s" paramName argTypeName
                        Some (paramName, argTypeName)
                    | _ -> None
                else
                    None)
            |> Map.ofList
    | _ -> Map.empty

/// Get parameter names from a function node for binding during inlining
let getParameterNames (funcNode: PSGNode) : string list =
    match funcNode.Symbol with
    | Some (:? FSharp.Compiler.Symbols.FSharpMemberOrFunctionOrValue as mfv) ->
        mfv.CurriedParameterGroups
        |> Seq.collect id
        |> Seq.choose (fun param -> param.Name)
        |> Seq.toList
    | _ -> []

/// Map F# type to MLIR type
let mapFSharpType (ftype: FSharpType option) : MLIRType =
    match ftype with
    | None -> Integer I32  // Default
    | Some t ->
        try
            if t.HasTypeDefinition then
                match t.TypeDefinition.TryFullName with
                | Some "System.Int32" -> Integer I32
                | Some "System.Int64" -> Integer I64
                | Some "System.Int16" -> Integer I16
                | Some "System.Byte" | Some "System.SByte" -> Integer I8
                | Some "System.UInt32" -> Integer I32
                | Some "System.UInt64" -> Integer I64
                | Some "System.Boolean" -> Integer I1
                | Some "System.Single" -> Float F32
                | Some "System.Double" -> Float F64
                | Some "System.Void" | Some "Microsoft.FSharp.Core.unit" -> Unit
                | Some "System.IntPtr" | Some "System.UIntPtr" -> Pointer
                | Some n when n.Contains("nativeptr") -> Pointer
                | Some "Microsoft.FSharp.Core.string" -> Pointer  // Strings are pointers to data
                | _ -> Integer I32
            elif t.IsGenericParameter then
                Pointer
            else
                Integer I32
        with _ -> Integer I32

/// Get all children of a PSG node (no reachability filtering)
/// During emission, we traverse from known-reachable entry points, so anything
/// we encounter is needed. Reachability filtering is counterproductive here
/// because SRTP-resolved methods may not have been in the static call graph.
let getAllChildren (psg: ProgramSemanticGraph) (node: PSGNode) : PSGNode list =
    ChildrenStateHelpers.getChildrenList node
    |> List.choose (fun id -> Map.tryFind id.Value psg.Nodes)

/// Flatten tuple arguments for function calls
/// In F#, tuple-style calls like f(a, b, c) produce a single Tuple node containing the args
/// Curried calls like f a b c produce multiple separate argument nodes
/// This function normalizes both cases to a flat list of argument nodes
let flattenTupleArgs (psg: ProgramSemanticGraph) (argNodes: PSGNode list) : PSGNode list =
    match argNodes with
    | [single] when single.SyntaxKind = "Tuple" ->
        // Single tuple node - expand its children to get actual arguments
        getAllChildren psg single
    | _ ->
        // Already flat (curried style or empty)
        argNodes

/// Emit a constant value
let emitConstant (node: PSGNode) (ctx: EmissionContext) : string * MLIRType * EmissionContext =
    match node.ConstantValue with
    | Some (Int32Value i) ->
        let ssa, ctx' = EmissionContext.nextSSA ctx
        let ctx'' = EmissionContext.emitLine (sprintf "%s = arith.constant %d : i32" ssa i) ctx'
        ssa, Integer I32, ctx''
    | Some (Int64Value i) ->
        let ssa, ctx' = EmissionContext.nextSSA ctx
        let ctx'' = EmissionContext.emitLine (sprintf "%s = arith.constant %d : i64" ssa i) ctx'
        ssa, Integer I64, ctx''
    | Some (BoolValue b) ->
        let ssa, ctx' = EmissionContext.nextSSA ctx
        let value = if b then 1 else 0
        let ctx'' = EmissionContext.emitLine (sprintf "%s = arith.constant %d : i1" ssa value) ctx'
        ssa, Integer I1, ctx''
    | Some (ByteValue b) ->
        let ssa, ctx' = EmissionContext.nextSSA ctx
        let ctx'' = EmissionContext.emitLine (sprintf "%s = arith.constant %d : i8" ssa (int b)) ctx'
        ssa, Integer I8, ctx''
    | Some (FloatValue f) ->
        let ssa, ctx' = EmissionContext.nextSSA ctx
        let ctx'' = EmissionContext.emitLine (sprintf "%s = arith.constant %f : f64" ssa f) ctx'
        ssa, Float F64, ctx''
    | Some (StringValue s) ->
        // String literals become global constants
        let globalName, ctx' = EmissionContext.registerStringLiteral s ctx
        let ssa, ctx'' = EmissionContext.nextSSA ctx'
        // Load pointer to the string data
        let ctx''' = EmissionContext.emitLine (sprintf "%s = llvm.mlir.addressof %s : !llvm.ptr" ssa globalName) ctx''
        ssa, Pointer, ctx'''
    | Some UnitValue ->
        // Unit is represented as empty tuple or just nothing
        "", Unit, ctx
    | Some (CharValue c) ->
        // Characters are emitted as i8 (their byte value)
        let ssa, ctx' = EmissionContext.nextSSA ctx
        let ctx'' = EmissionContext.emitLine (sprintf "%s = arith.constant %d : i8" ssa (int c)) ctx'
        ssa, Integer I8, ctx''
    | None ->
        // Try to infer from SyntaxKind
        if node.SyntaxKind.StartsWith("Const:Int32") then
            // Parse from SyntaxKind if needed
            let ssa, ctx' = EmissionContext.nextSSA ctx
            let ctx'' = EmissionContext.emitLine (sprintf "%s = arith.constant 0 : i32" ssa) ctx'
            ssa, Integer I32, ctx''
        else
            "", Unit, ctx

/// Emit a node and return its SSA value and type
/// Note: We do NOT check IsReachable here because during emission we traverse
/// from known-reachable entry points. SRTP-resolved functions may not have been
/// in the static call graph, so their nodes won't be marked reachable.
let rec emitNode (psg: ProgramSemanticGraph) (node: PSGNode) (ctx: EmissionContext) : string * MLIRType * EmissionContext =
    match node.SyntaxKind with
        // Constants
        | k when k.StartsWith("Const:") ->
            emitConstant node ctx

        // Identifiers - look up in node values or parameter bindings
        | k when k.StartsWith("Ident:") || k.StartsWith("LongIdent:") ->
            // Try to find the value this identifier refers to
            // First check node values (let bindings), then parameter bindings (inlined function args)
            match EmissionContext.lookupNodeValue node.Id.Value ctx with
            | Some (ssa, ty) -> ssa, ty, ctx
            | None ->
                // Not found by node ID - try looking up by identifier name (for inlined parameters)
                let identName =
                    if k.StartsWith("Ident:") then k.Substring(6)
                    elif k.StartsWith("LongIdent:") then
                        let fullName = k.Substring(10)
                        // Get the last part for simple parameter lookup
                        match fullName.LastIndexOf('.') with
                        | -1 -> fullName
                        | i -> fullName.Substring(i + 1)
                    else ""
                match EmissionContext.lookupParameter identName ctx with
                | Some (ssa, ty) ->
                    printfn "[EMIT] Found parameter binding for '%s' -> %s" identName ssa
                    ssa, ty, ctx
                | None ->
                    // Not yet computed - may be a function reference
                    "", mapFSharpType node.Type, ctx

        // Sequential expressions - emit each in order, return last value
        | "Sequential" ->
            let children = getAllChildren psg node
            printfn "[EMIT] Sequential has %d children" children.Length
            for c in children do
                printfn "[EMIT]   Seq child: %s (SyntaxKind: %s)" c.Id.Value c.SyntaxKind
            let folder (_, _, ctx') child =
                printfn "[EMIT]   Emitting seq child: %s" child.SyntaxKind
                let result = emitNode psg child ctx'
                printfn "[EMIT]   Result: %s" (match result with (s, _, _) -> s)
                result
            children |> List.fold folder ("", Unit, ctx)

        // Let bindings - emit value, record binding, then continuation
        | k when k.StartsWith("LetOrUse:") || k.StartsWith("Binding:") || k.StartsWith("Binding") ->
            let children = getAllChildren psg node
            printfn "[EMIT] LetOrUse: %s has %d children" k children.Length
            for i, c in List.indexed children do
                printfn "[EMIT]   child[%d]: %s" i c.SyntaxKind
            // F# let bindings have 3 children: [pattern; value; continuation]
            // Simple bindings may have 2 children: [pattern; value]
            // For mutable bindings, we need to allocate stack space and store the value
            let processMutableBinding pattern value (continuation: PSGNode option) ctx =
                // Emit the value
                let valueSsa, valueTy, ctx' = emitNode psg value ctx
                // First emit a constant for the allocation size
                let sizeSsa, ctx'' = EmissionContext.nextSSA ctx'
                let ctx'' = EmissionContext.emitLine (sprintf "%s = arith.constant 1 : i64" sizeSsa) ctx''
                // Allocate stack space for the mutable variable
                let allocSsa, ctx''' = EmissionContext.nextSSA ctx''
                let mlirType =
                    match valueTy with
                    | Integer I8 -> "i8"
                    | Integer I16 -> "i16"
                    | Integer I32 -> "i32"
                    | Integer I64 -> "i64"
                    | Float F32 -> "f32"
                    | Float F64 -> "f64"
                    | Pointer -> "!llvm.ptr"
                    | _ -> "i32"
                let allocLine = sprintf "%s = llvm.alloca %s x %s : (i64) -> !llvm.ptr" allocSsa sizeSsa mlirType
                let ctx4 = EmissionContext.emitLine allocLine ctx'''
                // Store the value into the allocation (if we have a value)
                let ctx5 =
                    if valueSsa <> "" then
                        let storeLine = sprintf "llvm.store %s, %s : %s, !llvm.ptr" valueSsa allocSsa mlirType
                        EmissionContext.emitLine storeLine ctx4
                    else ctx4
                // Record the binding - use the ALLOCATION address, not the value
                let patternName =
                    if pattern.SyntaxKind.StartsWith("Pattern:Named:") then
                        Some (pattern.SyntaxKind.Substring(14))
                    elif pattern.SyntaxKind.StartsWith("Pattern:LongIdent:") then
                        let fullName = pattern.SyntaxKind.Substring(18)
                        match fullName.LastIndexOf('.') with
                        | -1 -> Some fullName
                        | i -> Some (fullName.Substring(i + 1))
                    else None
                let ctx6 =
                    match patternName with
                    | Some name ->
                        printfn "[EMIT] Recording mutable binding '%s' = %s (alloc at %s)" name valueSsa allocSsa
                        // Record the allocation address as a Pointer type
                        let ctx6 = EmissionContext.recordNodeValue node.Id.Value allocSsa Pointer ctx5
                        EmissionContext.withParameterBindings [name] [(allocSsa, Pointer)] ctx6
                    | None -> ctx5
                // If there's a continuation, emit it
                match continuation with
                | Some cont -> emitNode psg cont ctx6
                | None -> allocSsa, Pointer, ctx6

            let processBinding pattern value (continuation: PSGNode option) ctx =
                // Emit the value
                let ssa, ty, ctx' = emitNode psg value ctx
                // Record the binding by node ID and also by pattern name
                let ctx'' =
                    if ssa <> "" then
                        let ctx'' = EmissionContext.recordNodeValue node.Id.Value ssa ty ctx'
                        // Also record by pattern name if it's a simple identifier pattern
                        let patternName =
                            if pattern.SyntaxKind.StartsWith("Pattern:Named:") then
                                Some (pattern.SyntaxKind.Substring(14))
                            elif pattern.SyntaxKind.StartsWith("Pattern:LongIdent:") then
                                let fullName = pattern.SyntaxKind.Substring(18)
                                match fullName.LastIndexOf('.') with
                                | -1 -> Some fullName
                                | i -> Some (fullName.Substring(i + 1))
                            else None
                        match patternName with
                        | Some name ->
                            printfn "[EMIT] Recording let binding '%s' = %s" name ssa
                            EmissionContext.withParameterBindings [name] [(ssa, ty)] ctx''
                        | None -> ctx''
                    else ctx'
                // If there's a continuation, emit it
                match continuation with
                | Some cont -> emitNode psg cont ctx''
                | None -> ssa, ty, ctx''

            match children with
            // PSG LetOrUse structure: [Binding, continuation] where Binding has [Pattern, Value]
            | [bindingNode; continuation] when bindingNode.SyntaxKind.StartsWith("Binding") ->
                let bindingChildren = getAllChildren psg bindingNode
                let isMutable = bindingNode.SyntaxKind.Contains("Mutable")
                match bindingChildren with
                | [pattern; value] ->
                    if isMutable then
                        processMutableBinding pattern value (Some continuation) ctx
                    else
                        processBinding pattern value (Some continuation) ctx
                | _ ->
                    // Malformed binding - emit continuation
                    emitNode psg continuation ctx
            // PSG Binding structure: [Pattern, Value] (no continuation)
            | [pattern; value] when node.SyntaxKind.StartsWith("Binding") ->
                let isMutable = node.SyntaxKind.Contains("Mutable")
                if isMutable then
                    processMutableBinding pattern value None ctx
                else
                    processBinding pattern value None ctx
            // Legacy 3-child structure: [pattern; value; continuation]
            | [pattern; value; continuation] ->
                processBinding pattern value (Some continuation) ctx
            | value :: _ when children.Length >= 1 ->
                // Fallback - just emit value
                emitNode psg value ctx
            | _ ->
                "", Unit, ctx

        // Function application
        | k when k.StartsWith("App:") ->
            emitFunctionApp psg node ctx

        // Patterns - structural, usually pass through to child
        | k when k.StartsWith("Pattern:") ->
            let children = getAllChildren psg node
            match children with
            | [child] -> emitNode psg child ctx
            | _ -> "", Unit, ctx

        // Tuple - emit each element
        | "Tuple" ->
            let children = getAllChildren psg node
            // Emit all children and return the last (or could build a struct)
            let folder (results, ctx') child =
                let ssa, ty, ctx'' = emitNode psg child ctx'
                ((ssa, ty) :: results, ctx'')
            let results, ctx' = children |> List.fold folder ([], ctx)
            match List.rev results with
            | (ssa, ty) :: _ -> ssa, ty, ctx'
            | _ -> "", Unit, ctx'

        // Address-of - get pointer to value
        | "AddressOf" ->
            let children = getAllChildren psg node
            match children with
            | [child] ->
                let ssa, _, ctx' = emitNode psg child ctx
                // The SSA value IS the address if it's from alloca
                ssa, Pointer, ctx'
            | _ -> "", Pointer, ctx

        // TypeApp - emit the inner function call
        | k when k.StartsWith("TypeApp:") ->
            let children = getAllChildren psg node
            match children with
            | [child] -> emitNode psg child ctx
            | _ -> "", Unit, ctx

        // Default: try children
        | _ ->
            let children = getAllChildren psg node
            match children with
            | [child] -> emitNode psg child ctx
            | _ -> "", mapFSharpType node.Type, ctx

/// Emit an FSharp.NativeInterop intrinsic as a compiler primitive
and emitNativeInteropIntrinsic (psg: ProgramSemanticGraph) (funcNode: PSGNode) (argNodes: PSGNode list) (ctx: EmissionContext) : string * MLIRType * EmissionContext =
    let intrinsicName = getNativeInteropIntrinsicName funcNode |> Option.defaultValue "unknown"
    // Flatten tuple arguments for tuple-style calls
    let flatArgs = flattenTupleArgs psg argNodes
    printfn "[EMIT] NativeInterop intrinsic: %s with %d args" intrinsicName flatArgs.Length

    // Emit all arguments first
    let emitArg (argVals, ctx') argNode =
        let ssa, ty, ctx'' = emitNode psg argNode ctx'
        ((ssa, ty) :: argVals, ctx'')
    let argVals, ctx' = flatArgs |> List.fold emitArg ([], ctx)
    let argVals = List.rev argVals

    // Generate MLIR for the intrinsic based on its name
    match intrinsicName with
    | "toNativeInt" ->
        // Pointer to integer cast - essentially a no-op in LLVM (same representation)
        match argVals with
        | [(ptrSsa, _)] when ptrSsa <> "" ->
            // Just pass through - nativeptr and nativeint have same representation
            ptrSsa, Pointer, ctx'
        | _ ->
            // If no SSA value, generate a placeholder
            let ssa, ctx'' = EmissionContext.nextSSA ctx'
            ssa, Pointer, ctx''

    | "ofNativeInt" ->
        // Integer to pointer cast - essentially a no-op
        match argVals with
        | [(intSsa, _)] when intSsa <> "" ->
            intSsa, Pointer, ctx'
        | _ ->
            let ssa, ctx'' = EmissionContext.nextSSA ctx'
            ssa, Pointer, ctx''

    | "toVoidPtr" | "ofVoidPtr" ->
        // Void pointer conversions - no-ops
        match argVals with
        | [(ptrSsa, _)] when ptrSsa <> "" -> ptrSsa, Pointer, ctx'
        | _ ->
            let ssa, ctx'' = EmissionContext.nextSSA ctx'
            ssa, Pointer, ctx''

    | "add" ->
        // Pointer arithmetic: ptr + offset
        match argVals with
        | [(ptrSsa, _); (offsetSsa, _)] when ptrSsa <> "" && offsetSsa <> "" ->
            let ssa, ctx'' = EmissionContext.nextSSA ctx'
            let line = sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i64) -> !llvm.ptr" ssa ptrSsa offsetSsa
            let ctx''' = EmissionContext.emitLine line ctx''
            ssa, Pointer, ctx'''
        | _ ->
            let ssa, ctx'' = EmissionContext.nextSSA ctx'
            ssa, Pointer, ctx''

    | "get" ->
        // Array element read: ptr[index]
        match argVals with
        | [(ptrSsa, _); (indexSsa, _)] when ptrSsa <> "" ->
            let elemPtr, ctx'' = EmissionContext.nextSSA ctx'
            let ssa, ctx''' = EmissionContext.nextSSA ctx''
            let gepLine = sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i64) -> !llvm.ptr" elemPtr ptrSsa indexSsa
            let loadLine = sprintf "%s = llvm.load %s : !llvm.ptr -> i8" ssa elemPtr
            let ctx4 = EmissionContext.emitLine gepLine ctx'''
            let ctx5 = EmissionContext.emitLine loadLine ctx4
            ssa, Integer I8, ctx5
        | _ ->
            let ssa, ctx'' = EmissionContext.nextSSA ctx'
            ssa, Integer I8, ctx''

    | "set" ->
        // Array element write: ptr[index] <- value
        match argVals with
        | [(ptrSsa, _); (indexSsa, _); (valueSsa, _)] when ptrSsa <> "" ->
            let elemPtr, ctx'' = EmissionContext.nextSSA ctx'
            let gepLine = sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i64) -> !llvm.ptr" elemPtr ptrSsa indexSsa
            let storeLine = sprintf "llvm.store %s, %s : i8, !llvm.ptr" valueSsa elemPtr
            let ctx''' = EmissionContext.emitLine gepLine ctx''
            let ctx4 = EmissionContext.emitLine storeLine ctx'''
            "", Unit, ctx4
        | _ -> "", Unit, ctx'

    | "read" ->
        // Read from pointer
        match argVals with
        | [(ptrSsa, _)] when ptrSsa <> "" ->
            let ssa, ctx'' = EmissionContext.nextSSA ctx'
            let line = sprintf "%s = llvm.load %s : !llvm.ptr -> i64" ssa ptrSsa
            let ctx''' = EmissionContext.emitLine line ctx''
            ssa, Integer I64, ctx'''
        | _ ->
            let ssa, ctx'' = EmissionContext.nextSSA ctx'
            ssa, Integer I64, ctx''

    | "write" ->
        // Write to pointer
        match argVals with
        | [(ptrSsa, _); (valueSsa, _)] when ptrSsa <> "" ->
            let line = sprintf "llvm.store %s, %s : i64, !llvm.ptr" valueSsa ptrSsa
            let ctx'' = EmissionContext.emitLine line ctx'
            "", Unit, ctx''
        | _ -> "", Unit, ctx'

    | "stackalloc" ->
        // Stack allocation - emit llvm.alloca
        let ssa, ctx'' = EmissionContext.nextSSA ctx'
        // Size comes from type parameter, for now assume reasonable default
        match argVals with
        | [(sizeSsa, _)] when sizeSsa <> "" ->
            let line = sprintf "%s = llvm.alloca %s x i8 : (i64) -> !llvm.ptr" ssa sizeSsa
            let ctx''' = EmissionContext.emitLine line ctx''
            ssa, Pointer, ctx'''
        | _ ->
            ssa, Pointer, ctx''

    | "nullPtr" ->
        // Null pointer constant
        let ssa, ctx'' = EmissionContext.nextSSA ctx'
        let line = sprintf "%s = llvm.mlir.null : !llvm.ptr" ssa
        let ctx''' = EmissionContext.emitLine line ctx''
        ssa, Pointer, ctx'''

    | _ ->
        // Unknown intrinsic - emit a comment and return placeholder
        printfn "[EMIT] WARNING: Unknown NativeInterop intrinsic: %s" intrinsicName
        let ssa, ctx'' = EmissionContext.nextSSA ctx'
        ssa, Pointer, ctx''

/// Emit a function application
and emitFunctionApp (psg: ProgramSemanticGraph) (node: PSGNode) (ctx: EmissionContext) : string * MLIRType * EmissionContext =
    let children = getAllChildren psg node
    printfn "[EMIT] App: children=%d" children.Length
    match children with
    | [] -> "", Unit, ctx
    | funcNode :: argNodes ->
        printfn "[EMIT] App: funcNode=%s isExtern=%b" funcNode.SyntaxKind (isExternPrimitive funcNode)
        // Check if function is an extern primitive
        if isExternPrimitive funcNode then
            emitExternCall psg funcNode argNodes ctx
        // Check if function is an FSharp.NativeInterop intrinsic
        elif isNativeInteropIntrinsic funcNode then
            emitNativeInteropIntrinsic psg funcNode argNodes ctx
        else
            // Check for SRTP resolution on the funcNode (SRTP-dispatched calls)
            // The SRTP resolution is stored on the LongIdent/Ident node, not the App node
            let bodyLookupResult =
                match funcNode.SRTPResolution with
                | Some (FSMethod (_, methodRef, _)) ->
                    // Use the resolved method to find the body
                    printfn "[EMIT] App: SRTP resolved to %s" methodRef.FullName
                    tryGetCalledFunctionBodyByName psg ctx.MemberBodies methodRef.FullName
                | Some (FSMethodByName methodName) ->
                    // SRTP resolved to a specific method by name (extracted from FCS internals)
                    printfn "[EMIT] App: SRTP resolved to method: %s" methodName
                    tryGetCalledFunctionBodyByName psg ctx.MemberBodies methodName
                | Some BuiltIn ->
                    // Built-in operator, no body to look up
                    printfn "[EMIT] App: SRTP resolved to BuiltIn operator"
                    None
                | Some (Unresolved reason) ->
                    printfn "[EMIT] App: SRTP unresolved: %s" reason
                    tryGetCalledFunctionBody psg ctx.MemberBodies funcNode
                | Some (MultipleOverloads (traitName, candidates)) ->
                    // Multiple overloads - select based on argument types from PSG
                    // Uses type substitution from context to resolve generic parameters
                    printfn "[EMIT] App: SRTP MultipleOverloads for %s (%d candidates)" traitName candidates.Length
                    match selectOverload candidates argNodes ctx with
                    | Some selected ->
                        printfn "[EMIT] App: Selected overload: %s" selected.TargetMethodFullName
                        tryGetCalledFunctionBodyByName psg ctx.MemberBodies selected.TargetMethodFullName
                    | None ->
                        printfn "[EMIT] App: No matching overload found, falling back to normal lookup"
                        tryGetCalledFunctionBody psg ctx.MemberBodies funcNode
                | Some (FSRecordField _) | Some (FSAnonRecordField _) ->
                    // Record field access - use normal lookup
                    tryGetCalledFunctionBody psg ctx.MemberBodies funcNode
                | None ->
                    // No SRTP resolution on funcNode, use normal body lookup
                    tryGetCalledFunctionBody psg ctx.MemberBodies funcNode

            // Get function name for tracking
            let funcName =
                match funcNode.Symbol with
                | Some s -> try s.FullName with _ -> funcNode.SyntaxKind
                | None -> funcNode.SyntaxKind

            // Check if the called function has a body we should inline
            match bodyLookupResult with
            | Some bodyNode ->
                printfn "[EMIT] App: Found body for %s -> %s" funcNode.SyntaxKind bodyNode.SyntaxKind

                // Check for infinite recursion before inlining
                match EmissionContext.tryPushInlining funcName ctx with
                | None ->
                    // Recursion detected or max depth exceeded - emit error and return unit
                    failwithf "EMISSION ERROR: Infinite recursion or max depth exceeded when inlining '%s'. Stack: %A"
                        funcName (Set.toList ctx.InliningStack)
                | Some ctx' ->
                    // Build type substitutions from call site arguments to resolve generic parameters
                    // For a call like `Write "Hello"` where Write has param `s : ^a`:
                    // This creates a mapping ^a -> Microsoft.FSharp.Core.string
                    let typeSubs = buildTypeSubstitutions funcNode argNodes
                    let ctx' =
                        if Map.isEmpty typeSubs then ctx'
                        else
                            printfn "[EMIT] App: Inlining with type substitutions: %A" (Map.toList typeSubs)
                            EmissionContext.withTypeSubstitutions typeSubs ctx'

                    // Get parameter names from the function symbol
                    let paramNames = getParameterNames funcNode
                    printfn "[EMIT] App: Function %s has parameters: %A" funcName paramNames
                    printfn "[EMIT] App: argNodes count: %d" argNodes.Length
                    for i, arg in List.indexed argNodes do
                        printfn "[EMIT] App:   arg[%d]: %s (SyntaxKind: %s)" i arg.Id.Value arg.SyntaxKind

                    // First emit ALL arguments for their side effects
                    let emitArg (argVals, ctxAcc) argNode =
                        let ssa, ty, ctxAcc' = emitNode psg argNode ctxAcc
                        printfn "[EMIT] App:   emitted arg: ssa='%s' type=%A" ssa ty
                        ((ssa, ty) :: argVals, ctxAcc')
                    let argVals, ctx'' = argNodes |> List.fold emitArg ([], ctx')
                    let argVals = List.rev argVals  // Reverse to match parameter order

                    // Bind parameter names to argument values
                    let ctx'' =
                        if List.length paramNames = List.length argVals then
                            EmissionContext.withParameterBindings paramNames argVals ctx''
                        else
                            printfn "[EMIT] WARNING: Parameter count mismatch - params=%d args=%d" paramNames.Length argVals.Length
                            ctx''

                    // Inline the function body (arguments already emitted above for side effects)
                    let result, ty, ctx''' = emitNode psg bodyNode ctx''

                    // Pop the function from the inlining stack
                    let ctxFinal = EmissionContext.popInlining funcName ctx'''
                    result, ty, ctxFinal

            | None ->
                printfn "[EMIT] App: No body found for %s" funcNode.SyntaxKind
                // Unknown function - still emit args for side effects
                let emitArg (argVals, ctx') argNode =
                    let ssa, ty, ctx'' = emitNode psg argNode ctx'
                    ((ssa, ty) :: argVals, ctx'')
                let argVals, ctx' = argNodes |> List.fold emitArg ([], ctx)
                let returnType = mapFSharpType node.Type
                "", returnType, ctx'

/// Try to find the body of a called function in the PSG
and tryGetCalledFunctionBody (psg: ProgramSemanticGraph) (memberBodies: Map<string, Baker.Types.MemberBodyMapping>) (funcNode: PSGNode) : PSGNode option =
    // Look for a matching function definition in the PSG
    // The function's binding should be reachable
    match funcNode.Symbol with
    | Some sym ->
        let targetName =
            try Some sym.FullName
            with _ ->
                try Some sym.DisplayName
                with _ -> None
        match targetName with
        | Some name -> tryGetCalledFunctionBodyByName psg memberBodies name
        | None -> None
    | None -> None

/// Try to find the body of a function by its full name
/// Note: Does NOT require the binding to be marked reachable - SRTP-resolved methods
/// may not be in the call graph since SRTP dispatch is implicit
/// Uses Baker member bodies as fallback when PSG symbol correlation fails
and tryGetCalledFunctionBodyByName (psg: ProgramSemanticGraph) (memberBodies: Map<string, Baker.Types.MemberBodyMapping>) (name: string) : PSGNode option =
    // Find binding with this name - EXACT symbol matching only
    // For static members like WritableString.($), also try op_Dollar format
    let namesToTry = [
        name
        // Try with "op_Dollar" instead of "($)"
        name.Replace("($)", "op_Dollar")
    ]

    // Step 1: Symbol-based lookup in PSG
    let bySymbol =
        psg.Nodes
        |> Map.toSeq
        |> Seq.tryFind (fun (_, n) ->
            // Note: Don't check IsReachable - SRTP-resolved methods may not be in call graph
            n.SyntaxKind.StartsWith("Binding") &&
            match n.Symbol with
            | Some s ->
                try
                    let fullName = s.FullName
                    namesToTry |> List.exists (fun targetName ->
                        fullName = targetName)
                with _ -> false
            | None -> false)

    let psgResult =
        bySymbol
        |> Option.bind (fun (_, bindingNode) ->
            // Get the body - use all children, not just reachable ones (SRTP target may not be reachable)
            match bindingNode.Children with
            | Parent childIds ->
                let children = childIds |> List.choose (fun id -> Map.tryFind id.Value psg.Nodes)
                match children with
                | _ :: body :: _ -> Some body
                | _ -> None
            | _ -> None)

    // Step 2: If PSG lookup failed, try Baker member bodies
    // Baker has correlated member bodies with PSG bindings via range matching
    match psgResult with
    | Some _ -> psgResult
    | None ->
        // Try each name variant in Baker's member bodies
        namesToTry
        |> List.tryPick (fun targetName ->
            match Map.tryFind targetName memberBodies with
            | Some mapping ->
                // Baker found this member. Use its PSGBindingId to find the PSG body node.
                match mapping.PSGBindingId with
                | Some bindingId ->
                    // Find the binding node and return its body
                    match Map.tryFind bindingId.Value psg.Nodes with
                    | Some bindingNode ->
                        match bindingNode.Children with
                        | Parent childIds ->
                            let children = childIds |> List.choose (fun id -> Map.tryFind id.Value psg.Nodes)
                            match children with
                            | _ :: body :: _ ->
                                printfn "[BAKER] Found body for %s via Baker correlation" targetName
                                Some body
                            | _ -> None
                        | _ -> None
                    | None -> None
                | None ->
                    // Baker has the FSharpExpr body but no PSG correlation
                    // We can't emit FSharpExpr directly, need PSG node
                    printfn "[BAKER] Warning: Found member body for %s but no PSG binding correlation" targetName
                    None
            | None -> None)

/// Emit an extern primitive call via ExternDispatch
and emitExternCall (psg: ProgramSemanticGraph) (funcNode: PSGNode) (argNodes: PSGNode list) (ctx: EmissionContext) : string * MLIRType * EmissionContext =
    match tryExtractExternPrimitiveInfo funcNode with
    | None ->
        EmissionErrors.add {
            Phase = "MLIR"
            Message = sprintf "Could not extract extern info from %s" funcNode.SyntaxKind
            Location = Some funcNode.SourceFile
            Severity = ErrorSeverity.Warning
        }
        "", Unit, ctx
    | Some info ->
        // Flatten tuple arguments - F# tuple-style calls f(a, b, c) produce a single Tuple node
        let flatArgs = flattenTupleArgs psg argNodes
        printfn "[EMIT] ExternCall: %s with %d args (after flattening from %d)" info.EntryPoint flatArgs.Length argNodes.Length

        // Emit all arguments first
        let emitArg (argVals, ctx') argNode =
            let ssa, ty, ctx'' = emitNode psg argNode ctx'
            if ssa <> "" then
                // Parse the SSA string to get the number
                let ssaNum =
                    if ssa.StartsWith("%v") then
                        match System.Int32.TryParse(ssa.Substring(2)) with
                        | true, n -> V n
                        | _ -> V 0
                    else V 0
                ({ SSA = ssaNum; Type = ty } :: argVals, ctx'')
            else
                (argVals, ctx'')
        let argVals, ctx' = flatArgs |> List.fold emitArg ([], ctx)
        let argVals = List.rev argVals

        // Create extern primitive and dispatch
        let prim = createExternPrimitive info argVals Static

        // Run the binding via the MLIR monad
        let dispatchComputation = ExternDispatch.dispatch prim

        // Create a BuilderState to run the MLIR computation
        let builderState = BuilderState.createAt ctx'.SSACounter

        // Execute the MLIR monad - it's just a function from BuilderState to result
        let result = dispatchComputation builderState

        // Merge state back - update SSA counter
        let ctx'' = { ctx' with SSACounter = builderState.SSACounter }

        // Emit the MLIR output from the builder
        let mlirOutput = builderState.Output.ToString()
        let ctx''' =
            if mlirOutput.Length > 0 then
                mlirOutput.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
                |> Array.fold (fun c line -> EmissionContext.emitLine line c) ctx''
            else ctx''

        match result with
        | Emitted val' ->
            val'.SSA.Name, val'.Type, ctx'''
        | EmittedVoid ->
            "", Unit, ctx'''
        | NotSupported reason ->
            EmissionErrors.add {
                Phase = "MLIR"
                Message = sprintf "Binding not supported: %s - %s" info.EntryPoint reason
                Location = Some funcNode.SourceFile
                Severity = ErrorSeverity.Warning
            }
            "", info.ReturnType, ctx'''

/// Emit string literal globals
let emitStringLiterals (ctx: EmissionContext) : EmissionContext =
    ctx.StringLiterals
    |> Map.toList
    |> List.fold (fun c (_, (content, name)) ->
        let escaped = content.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\0A")
        let len = content.Length + 1  // +1 for null terminator
        EmissionContext.emitLineRaw
            (sprintf "llvm.mlir.global internal constant %s(\"%s\\00\") : !llvm.array<%d x i8>"
                name escaped len) c
    ) ctx

/// Generate MLIR from PSG using proper layered architecture
///
/// ARCHITECTURE (correct flow):
///   PSG → Zipper (traversal) → Local pattern match → Bindings (MLIR generation)
///
/// The traversal follows PSG structure. At each node, local pattern matching
/// determines the emission. Extern primitives dispatch to platform bindings.
/// MLIR accumulates in the builder (correct centralization point).
///
let generateMLIRViaAlex (psg: ProgramSemanticGraph) (memberBodies: Map<string, Baker.Types.MemberBodyMapping>) (projectName: string) (targetTriple: string) : MLIRGenerationResult =
    // Reset error collector for this compilation
    EmissionErrors.reset()

    // Register all platform bindings
    Alex.Bindings.Time.TimeBindings.registerBindings ()
    Alex.Bindings.Console.ConsoleBindings.registerBindings ()
    Alex.Bindings.Process.ProcessBindings.registerBindings ()

    // Set target platform from triple
    match TargetPlatform.parseTriple targetTriple with
    | Some platform -> ExternDispatch.setTargetPlatform platform
    | None -> ()  // Use default (auto-detect)

    // Initialize emission context with Baker member bodies
    let ctx = EmissionContext.create psg memberBodies

    // Emit module header
    let ctx = EmissionContext.emitLineRaw (sprintf "// Firefly-generated MLIR for %s" projectName) ctx
    let ctx = EmissionContext.emitLineRaw (sprintf "// Target: %s" targetTriple) ctx
    let ctx = EmissionContext.emitLineRaw "" ctx

    // Find entry points and emit each as a function
    let emitEntryPoint (ctx: EmissionContext) (entryId: NodeId) =
        match Map.tryFind entryId.Value psg.Nodes with
        | None ->
            printfn "[EMIT] Entry point %s not found in PSG" entryId.Value
            ctx
        | Some entryNode ->
            printfn "[EMIT] Entry point: %s (SyntaxKind: %s, IsReachable: %b)" entryId.Value entryNode.SyntaxKind entryNode.IsReachable
            let children = getAllChildren psg entryNode
            printfn "[EMIT] Entry point has %d children" children.Length
            for c in children do
                printfn "[EMIT]   Child: %s (SyntaxKind: %s)" c.Id.Value c.SyntaxKind
            if not entryNode.IsReachable then ctx
            else
                // Emit function signature
                let funcName = "main"
                let ctx = EmissionContext.emitLineRaw "module {" ctx
                let ctx = EmissionContext.emitLineRaw "" ctx

                // First, emit any string literals used
                // (We'll collect them during emission and add at the end)

                let ctx = { ctx with CurrentFunctionName = Some funcName }
                let ctx = EmissionContext.emitLineRaw (sprintf "  llvm.func @%s() -> i32 {" funcName) ctx
                let ctx = EmissionContext.withIndent ctx
                let ctx = EmissionContext.withIndent ctx

                // Emit entry block
                let ctx = EmissionContext.emitLine "^entry:" ctx

                // Emit the entry point body
                let resultSSA, resultType, ctx = emitNode psg entryNode ctx
                printfn "[EMIT] Result after emitting entry: SSA=%s Type=%A" resultSSA resultType

                // Emit return
                let ctx =
                    if resultSSA <> "" && resultType = Integer I32 then
                        EmissionContext.emitLine (sprintf "llvm.return %s : i32" resultSSA) ctx
                    else
                        // Default return 0
                        let ssa, ctx' = EmissionContext.nextSSA ctx
                        let ctx'' = EmissionContext.emitLine (sprintf "%s = arith.constant 0 : i32" ssa) ctx'
                        EmissionContext.emitLine (sprintf "llvm.return %s : i32" ssa) ctx''

                let ctx = EmissionContext.withoutIndent ctx
                let ctx = EmissionContext.emitLineRaw "  }" ctx
                let ctx = EmissionContext.emitLineRaw "" ctx

                // Emit string literals at module level
                let ctx = emitStringLiterals ctx

                let ctx = EmissionContext.emitLineRaw "}" ctx
                ctx

    // Emit all entry points
    let ctx = psg.EntryPoints |> List.fold emitEntryPoint ctx

    // Collect any emission errors
    let emissionErrors = EmissionErrors.toCompilerErrors()

    {
        Content = EmissionContext.getText ctx
        Errors = emissionErrors
        HasErrors = EmissionErrors.hasErrors()
    }

// ===================================================================
// Intermediate File Management
// ===================================================================

/// Collect intermediate file paths from pipeline execution
let collectIntermediates (intermediatesDir: string option) : IntermediateOutputs =
    match intermediatesDir with
    | None -> emptyIntermediates
    | Some dir ->
        let tryFindFile fileName =
            let path = Path.Combine(dir, fileName)
            if File.Exists(path) then Some path else None
        
        {
            ProjectAnalysis = tryFindFile "project.analysis.json"
            PSGRepresentation = tryFindFile "psg.summary.json"
            ReachabilityAnalysis = tryFindFile "reachability.analysis.json"
            PrunedSymbols = tryFindFile "psg.pruned.symbols.json"
        }

// ===================================================================
// Main Compilation Entry Points
// ===================================================================

/// Compile a project file using the ingestion pipeline
let compileProject 
    (projectPath: string) 
    (outputPath: string)
    (projectOptions: FSharpProjectOptions)
    (compilationConfig: CompilationConfig)
    (intermediatesDir: string option)
    (progress: ProgressCallback) = async {
    
    let startTime = DateTime.UtcNow
    
    // Convert CompilationConfig to PipelineConfig
    let pipelineConfig = createPipelineConfig compilationConfig intermediatesDir
    
    try
        // Execute the complete ingestion and analysis pipeline
        printfn "[Compilation] Starting compilation pipeline..."
        let! pipelineResult = runPipeline projectPath pipelineConfig
        
        // Convert diagnostics and generate statistics
        let diagnostics = pipelineResult.Diagnostics |> List.map convertDiagnostic
        let statistics = generateStatistics pipelineResult startTime
        let intermediates = collectIntermediates intermediatesDir
        
        // Report final results
        if pipelineResult.Success then
            printfn "[Compilation] Compilation completed successfully"
            
            match pipelineResult.ReachabilityAnalysis with
            | Some analysis ->
                printfn "[Compilation] Final statistics: %d/%d symbols reachable (%.1f%% eliminated)" 
                    analysis.PruningStatistics.ReachableSymbols
                    analysis.PruningStatistics.TotalSymbols
                    ((float analysis.PruningStatistics.EliminatedSymbols / float analysis.PruningStatistics.TotalSymbols) * 100.0)
            | None -> ()
        else
            printfn "[Compilation] Compilation failed"
        
        return {
            Success = pipelineResult.Success
            Diagnostics = diagnostics
            Statistics = statistics
            Intermediates = intermediates
        }
        
    with ex ->
        printfn "[Compilation] Compilation failed: %s" ex.Message
        return {
            Success = false
            Diagnostics = [{
                Phase = "Compilation"
                Message = ex.Message
                Location = None
                Severity = ErrorSeverity.Error
            }]
            Statistics = CompilationStatistics.empty
            Intermediates = emptyIntermediates
        }
}

/// Simplified entry point using file path
let compile 
    (projectPath: string) 
    (intermediatesDir: string option) 
    (progress: ProgressCallback) = async {
    
    // Create default compilation configuration
    let compilationConfig : CompilationConfig = {
        EnableClosureElimination = true
        EnableStackAllocation = true
        EnableReachabilityAnalysis = true
        PreserveIntermediateASTs = intermediatesDir.IsSome
        VerboseOutput = false
    }
    
    // Create F# checker and load project
    let checker = FSharpChecker.Create()
    
    try
        // Read the project file content and create ISourceText
        let content = File.ReadAllText(projectPath)
        let sourceText = SourceText.ofString content
        
        // Get project options from script
        let! (projectOptions, diagnostics) = checker.GetProjectOptionsFromScript(projectPath, sourceText)
        
        // Check for critical errors in diagnostics
        if diagnostics.Length > 0 then
            printfn "[Compilation] Project loading diagnostics:"
            for diag in diagnostics do
                printfn "  %s" diag.Message
        
        // Use a default output path
        let outputPath = Path.ChangeExtension(projectPath, ".exe")
        
        return! compileProject projectPath outputPath projectOptions compilationConfig intermediatesDir progress
    
    with ex ->
        return {
            Success = false
            Diagnostics = [{
                Phase = "ProjectLoading"
                Message = ex.Message
                Location = Some projectPath
                Severity = ErrorSeverity.Error
            }]
            Statistics = CompilationStatistics.empty
            Intermediates = emptyIntermediates
        }
}