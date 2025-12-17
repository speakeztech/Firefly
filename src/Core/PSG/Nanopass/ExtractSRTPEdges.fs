/// ExtractSRTPEdges - Nanopass to extract SRTP call relationships BEFORE reachability
///
/// This nanopass runs BEFORE reachability analysis to ensure that SRTP-dispatched
/// call targets are marked reachable. Without this, reachability follows only the
/// syntactic call graph (Console.Write → op_Dollar) and misses the actual targets
/// (writeSystemString → writeBytes → Bindings.writeBytes).
///
/// The key insight: TraitCall nodes in the typed tree know their resolution targets,
/// but reachability runs before we can access that information. This pass extracts
/// SRTP resolutions early and returns additional call relationships that get merged
/// into the semantic call graph during reachability computation.
///
/// IMPORTANT: We can't create PSG edges to target functions because those functions
/// don't have PSG nodes yet (they're unreachable). Instead, we return a map of
/// caller -> callees that gets merged into the call graph.
///
/// Reference: See plan at ~/.claude/plans/elegant-cooking-garden.md
module Core.PSG.Nanopass.ExtractSRTPEdges

open FSharp.Compiler.Symbols
open FSharp.Compiler.Symbols.FSharpExprPatterns
open FSharp.Compiler.Text
open FSharp.Compiler.CodeAnalysis
open Core.PSG.Types
open Core.CompilerConfig

// ═══════════════════════════════════════════════════════════════════════════
// Range-based Key Generation (same as ResolveSRTP for consistency)
// ═══════════════════════════════════════════════════════════════════════════

/// Generate a stable key from a range for matching TraitCalls to PSG nodes
let private rangeToKey (range: range) : string =
    sprintf "%s_%d_%d_%d_%d"
        (System.IO.Path.GetFileName range.FileName)
        range.Start.Line range.Start.Column
        range.End.Line range.End.Column

// ═══════════════════════════════════════════════════════════════════════════
// SRTP Target Resolution
// ═══════════════════════════════════════════════════════════════════════════

/// Build a map of member full names to their body expressions
/// This allows us to find what functions a resolved SRTP member calls
let private buildMemberBodyMap (checkResults: FSharpCheckProjectResults) : Map<string, FSharpExpr> =
    let mutable bodies = Map.empty

    for implFile in checkResults.AssemblyContents.ImplementationFiles do
        let rec processDecl (decl: FSharpImplementationFileDeclaration) =
            match decl with
            | FSharpImplementationFileDeclaration.Entity (entity, subDecls) ->
                subDecls |> List.iter processDecl
            | FSharpImplementationFileDeclaration.MemberOrFunctionOrValue (mfv, _args, body) ->
                try
                    let fullName = mfv.FullName
                    bodies <- bodies |> Map.add fullName body
                with _ -> ()
            | FSharpImplementationFileDeclaration.InitAction _ -> ()

        implFile.Declarations |> List.iter processDecl

    bodies

/// Extract function calls from an expression body
/// Returns list of (fullName, declarationLocation) for called functions
let private extractCallsFromBody (body: FSharpExpr) : (string * range) list =
    let mutable calls = []

    let rec walkExpr (expr: FSharpExpr) =
        try
            match expr with
            | FSharpExprPatterns.Call (objOpt, memberOrFunc, _typeArgs1, _typeArgs2, argExprs) ->
                // This is a direct function/method call
                try
                    let fullName = memberOrFunc.FullName
                    let declLoc = memberOrFunc.DeclarationLocation
                    calls <- (fullName, declLoc) :: calls
                with _ -> ()
                // Continue walking arguments
                objOpt |> Option.iter walkExpr
                argExprs |> List.iter walkExpr

            | _ ->
                // Walk sub-expressions
                for subExpr in expr.ImmediateSubExpressions do
                    try walkExpr subExpr with _ -> ()
        with _ -> ()

    walkExpr body
    calls |> List.distinct

// ═══════════════════════════════════════════════════════════════════════════
// SRTP Call Relationships
// ═══════════════════════════════════════════════════════════════════════════

/// Result of SRTP extraction: additional call relationships for the call graph
/// This is a map from caller symbol name to list of callee symbol names
type SRTPCallRelationships = Map<string, string list>

/// Extract call relationships from SRTP resolutions
/// For each TraitCall, we need to know:
/// 1. Which function contains the TraitCall (the caller)
/// 2. What functions the resolved SRTP member calls (the callees)
///
/// Since TraitCall is inlined during compilation, the caller of the TraitCall
/// effectively calls the functions that the SRTP member's body calls.
let private extractSRTPCallRelationships
    (checkResults: FSharpCheckProjectResults)
    : SRTPCallRelationships =

    // First, build a map of all member bodies
    let memberBodies = buildMemberBodyMap checkResults
    if isSRTPVerbose() then
        printfn "[ExtractSRTP] Built member body map with %d entries" memberBodies.Count

    let mutable relationships: Map<string, string list> = Map.empty
    let mutable exprCount = 0

    for implFile in checkResults.AssemblyContents.ImplementationFiles do
        // Track which function we're currently in
        let rec processDecl (currentFunction: string option) (decl: FSharpImplementationFileDeclaration) =
            try
                match decl with
                | FSharpImplementationFileDeclaration.Entity (_, subDecls) ->
                    subDecls |> List.iter (processDecl currentFunction)
                | FSharpImplementationFileDeclaration.MemberOrFunctionOrValue (mfv, _, expr) ->
                    // This MFV defines the containing function
                    let funcName = try Some mfv.FullName with _ -> None
                    walkExpr funcName expr
                | FSharpImplementationFileDeclaration.InitAction expr ->
                    // Init actions have no containing function name, use module name
                    walkExpr (Some "ModuleInit") expr
            with _ -> ()

        and walkExpr (containingFunction: string option) (expr: FSharpExpr) =
            exprCount <- exprCount + 1
            try
                match expr with
                | FSharpExprPatterns.TraitCall (sourceTypes, traitName, _memberFlags, _paramTypes, _retTypes, traitArgs) ->
                    // Resolve the trait call to its concrete implementation
                    match sourceTypes with
                    | firstType :: _ when firstType.HasTypeDefinition ->
                        let entity = firstType.TypeDefinition
                        let members = entity.MembersFunctionsAndValues

                        // Find member matching the trait name
                        let candidate =
                            members
                            |> Seq.tryFind (fun m ->
                                m.LogicalName = traitName || m.CompiledName = traitName)

                        match candidate, containingFunction with
                        | Some resolvedMember, Some caller ->
                            let resolvedFullName = resolvedMember.FullName

                            // Get what the resolved member's body calls
                            let bodyCallees =
                                match Map.tryFind resolvedFullName memberBodies with
                                | Some body ->
                                    let calls = extractCallsFromBody body
                                    if isSRTPVerbose() then
                                        printfn "[ExtractSRTP] %s calls %s (trait: %s), which calls: %A"
                                            caller resolvedFullName traitName (calls |> List.map fst)
                                    calls |> List.map fst
                                | None ->
                                    if isSRTPVerbose() then
                                        printfn "[ExtractSRTP] %s calls %s (trait: %s) - no body found"
                                            caller resolvedFullName traitName
                                    // Even if no body, the resolved member itself should be reachable
                                    [resolvedFullName]

                            // Add relationships: caller -> each callee
                            if not (List.isEmpty bodyCallees) then
                                let existingCallees =
                                    Map.tryFind caller relationships
                                    |> Option.defaultValue []
                                let newCallees =
                                    bodyCallees
                                    |> List.filter (fun c -> not (List.contains c existingCallees))
                                relationships <- Map.add caller (existingCallees @ newCallees) relationships

                        | _ -> ()
                    | _ -> ()

                    // Continue walking arguments
                    for arg in traitArgs do
                        try walkExpr containingFunction arg with _ -> ()

                | _ ->
                    // Walk sub-expressions
                    for subExpr in expr.ImmediateSubExpressions do
                        try walkExpr containingFunction subExpr with _ -> ()
            with _ -> ()

        implFile.Declarations |> List.iter (processDecl None)

    if isSRTPVerbose() then
        printfn "[ExtractSRTP] Walked %d expressions, found %d callers with SRTP relationships"
            exprCount relationships.Count
    relationships

// ═══════════════════════════════════════════════════════════════════════════
// Main Entry Point
// ═══════════════════════════════════════════════════════════════════════════

/// Result of the ExtractSRTPEdges nanopass
type ExtractSRTPResult = {
    /// Additional call relationships to merge into the semantic call graph
    /// Map from caller symbol name -> list of callee symbol names
    AdditionalCalls: SRTPCallRelationships
}

/// Run the ExtractSRTPEdges nanopass.
/// This MUST run BEFORE reachability analysis to ensure SRTP targets are reachable.
///
/// Returns additional call relationships that should be merged into the semantic
/// call graph during reachability computation. We don't modify the PSG here because
/// target functions may not have PSG nodes yet (they're unreachable).
let run (checkResults: FSharpCheckProjectResults) : ExtractSRTPResult =
    // Extract SRTP call relationships from the typed tree
    let relationships = extractSRTPCallRelationships checkResults

    // Debug: print what we found
    if isSRTPVerbose() && not (Map.isEmpty relationships) then
        printfn "[SRTP] Found SRTP call relationships:"
        for KeyValue(caller, callees) in relationships do
            printfn "  %s -> %A" caller callees

    { AdditionalCalls = relationships }
