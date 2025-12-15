/// MemberBodyMapper - Extracts member bodies from FCS typed tree
///
/// Uses TypedTreeZipper to walk the typed tree and extract:
/// 1. Member bodies (for function inlining in Alex)
/// 2. Field access info (indexed by range for O(1) lookup)
/// 3. Resolved types (indexed by range for O(1) lookup)
///
/// CRITICAL: Operates AFTER reachability analysis (Phase 3).
///
/// Phase: Baker (Phase 4) - post-reachability type resolution
module Baker.MemberBodyMapper

open FSharp.Compiler.Symbols
open FSharp.Compiler.CodeAnalysis
open Core.PSG.Types
open Baker.Types
open Baker.TypedTreeZipper

/// Filter member bodies to only include those referenced by reachable PSG nodes
let filterToReachable
    (allBodies: Map<string, MemberBodyMapping>)
    (psg: ProgramSemanticGraph)
    : Map<string, MemberBodyMapping> =

    // Collect all member names referenced by reachable nodes
    let referencedMembers =
        psg.Nodes
        |> Map.toSeq
        |> Seq.filter (fun (_, node) -> node.IsReachable)
        |> Seq.choose (fun (_, node) ->
            match node.SRTPResolution with
            | Some (FSMethod (_, mfv, _)) ->
                try Some mfv.FullName with _ -> None
            | Some (FSMethodByName name) ->
                Some name
            | Some (MultipleOverloads (_, candidates)) ->
                candidates
                |> List.tryHead
                |> Option.map (fun c -> c.TargetMethodFullName)
            | _ ->
                match node.Symbol with
                | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
                    try Some mfv.FullName with _ -> None
                | _ -> None)
        |> Set.ofSeq

    // Filter bodies to only those referenced
    allBodies
    |> Map.filter (fun key _ -> Set.contains key referencedMembers)

/// Convert CorrelationState.MemberBodies to MemberBodyMapping format
let private convertToMemberBodyMappings
    (memberBodies: Map<string, FSharpMemberOrFunctionOrValue * FSharpExpr>)
    : Map<string, MemberBodyMapping> =

    memberBodies
    |> Map.map (fun _fullName (mfv, body) ->
        {
            Member = mfv
            DeclarationRange = mfv.DeclarationLocation
            Body = body
            PSGBindingId = None  // Correlation happens at emission time via range lookup
        })

/// Add operator aliases for lookup flexibility
let private addOperatorAliases
    (bodies: Map<string, MemberBodyMapping>)
    : Map<string, MemberBodyMapping> =

    bodies
    |> Map.fold (fun acc fullName mapping ->
        let displayName = mapping.Member.DisplayName
        if displayName.StartsWith("(") && displayName.EndsWith(")") then
            let opName = displayName.Substring(1, displayName.Length - 2)
            let compiledName =
                match opName with
                | "$" -> "op_Dollar"
                | "+" -> "op_Addition"
                | "-" -> "op_Subtraction"
                | "*" -> "op_Multiply"
                | "/" -> "op_Division"
                | "%" -> "op_Modulus"
                | ">" -> "op_GreaterThan"
                | "<" -> "op_LessThan"
                | ">=" -> "op_GreaterThanOrEqual"
                | "<=" -> "op_LessThanOrEqual"
                | "=" -> "op_Equality"
                | "<>" -> "op_Inequality"
                | _ -> sprintf "op_%s" opName

            match mapping.Member.DeclaringEntity with
            | Some entity ->
                let entityName =
                    match entity.TryFullName with
                    | Some name -> name
                    | None -> entity.DisplayName
                let altKey = sprintf "%s.%s" entityName compiledName
                Map.add altKey mapping (Map.add fullName mapping acc)
            | None ->
                Map.add fullName mapping acc
        else
            Map.add fullName mapping acc
    ) Map.empty

/// Result of MemberBodyMapper run
type MapperResult = {
    /// Member body mappings (fullName -> mapping)
    MemberBodies: Map<string, MemberBodyMapping>
    /// Correlation state from TypedTreeZipper (for field access, type lookups)
    CorrelationState: CorrelationState
    /// Statistics
    Statistics: BakerStatistics
}

/// Main entry point: Extract member bodies and correlation info using TypedTreeZipper
///
/// CRITICAL: Should be called AFTER reachability analysis (Phase 3)
let run
    (psg: ProgramSemanticGraph)
    (checkResults: FSharpCheckProjectResults)
    : MapperResult =

    // Step 1: Use TypedTreeZipper to walk typed tree and extract correlations
    let implFiles = checkResults.AssemblyContents.ImplementationFiles
    let correlationState = extractTypedTreeInfo implFiles

    // Step 2: Convert member bodies to our MemberBodyMapping format
    let allBodies = convertToMemberBodyMappings correlationState.MemberBodies

    // Step 3: Add operator aliases for flexible lookup
    let withAliases = addOperatorAliases allBodies

    // Step 4: Filter to only members referenced by reachable nodes
    let reachableBodies = filterToReachable withAliases psg

    {
        MemberBodies = reachableBodies
        CorrelationState = correlationState
        Statistics = {
            TotalMembers = withAliases.Count
            MembersWithBodies = reachableBodies.Count
            MembersCorrelatedWithPSG = 0  // Correlation happens at emission time
            ProcessingTimeMs = 0.0
        }
    }
