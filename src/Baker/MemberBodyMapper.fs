/// MemberBodyMapper - Orchestrates dual-tree zipper correlation
///
/// Uses TypedTreeZipper to walk both typed tree and PSG in tandem,
/// extracting correlations at synchronized positions.
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
            PSGBindingId = None
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
    /// Correlation state from dual-tree zipper
    CorrelationState: CorrelationState
    /// Statistics
    Statistics: BakerStatistics
}

/// Main entry point: Correlate typed tree with PSG using dual-tree zipper
///
/// CRITICAL: Should be called AFTER reachability analysis (Phase 3)
let run
    (psg: ProgramSemanticGraph)
    (checkResults: FSharpCheckProjectResults)
    : MapperResult =

    // Use dual-tree zipper to correlate typed tree with PSG
    let implFiles = checkResults.AssemblyContents.ImplementationFiles
    let correlationState = correlate psg implFiles

    // Convert member bodies to MemberBodyMapping format
    let allBodies = convertToMemberBodyMappings correlationState.MemberBodies

    // Add operator aliases for flexible lookup
    let withAliases = addOperatorAliases allBodies

    {
        MemberBodies = withAliases
        CorrelationState = correlationState
        Statistics = {
            TotalMembers = withAliases.Count
            MembersWithBodies = withAliases.Count
            MembersCorrelatedWithPSG = correlationState.FieldAccess.Count + correlationState.Types.Count
            ProcessingTimeMs = 0.0
        }
    }
