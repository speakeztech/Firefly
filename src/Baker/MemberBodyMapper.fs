/// MemberBodyMapper - Extracts member bodies from FCS typed tree using TypedTreeZipper
///
/// This component uses the TypedTreeZipper to walk the FCS typed tree and PSG
/// in parallel, correlating member bodies through structural context.
///
/// CRITICAL: This operates AFTER reachability analysis (Phase 3).
/// Only members referenced by reachable PSG nodes are processed.
///
/// Phase: Baker (Phase 4) - post-reachability type resolution
module Baker.MemberBodyMapper

open System
open FSharp.Compiler.Symbols
open FSharp.Compiler.Text
open FSharp.Compiler.CodeAnalysis
open Core.PSG.Types
open Baker.Types
open Baker.TypedTreeZipper

/// Extract member bodies using the TypedTreeZipper for correlation
///
/// The zipper walks both the typed tree and PSG in parallel, using
/// structural context (the zipper's path) to maintain correlation.
/// This replaces the previous ad-hoc range matching approach.
let extractWithZipper
    (psg: ProgramSemanticGraph)
    (checkResults: FSharpCheckProjectResults)
    : Map<string, MemberBodyMapping> =

    // Use the zipper to walk all implementation files
    let implFiles = checkResults.AssemblyContents.ImplementationFiles
    let correlationState = correlateAll implFiles psg

    // Convert zipper's MemberBodies to our MemberBodyMapping format
    let memberBodies =
        correlationState.MemberBodies
        |> Map.map (fun _fullName (mfv, body) ->
            // Look up the PSG correlation if available
            let psgBindingId =
                correlationState.Correlations
                |> Map.tryPick (fun nodeId info ->
                    match info.MemberBody with
                    | Some b when System.Object.ReferenceEquals(b, body) -> Some (NodeId.Create nodeId)
                    | _ -> None)

            {
                Member = mfv
                DeclarationRange = mfv.DeclarationLocation
                Body = body
                PSGBindingId = psgBindingId
            })

    // Also add operator aliases (op_Dollar, etc.)
    let withAliases =
        memberBodies
        |> Map.fold (fun acc fullName mapping ->
            let displayName = mapping.Member.DisplayName
            if displayName.StartsWith("(") && displayName.EndsWith(")") then
                // This is an operator like "($)"
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

    withAliases

/// Filter member bodies to only include those referenced by reachable PSG nodes
/// This is the key post-reachability filtering step
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
            // Check for SRTP resolutions that reference specific members
            match node.SRTPResolution with
            | Some (FSMethod (_, mfv, _)) ->
                try Some mfv.FullName with _ -> None
            | Some (FSMethodByName name) ->
                Some name
            | Some (MultipleOverloads (_, candidates)) ->
                // All candidate method names
                candidates
                |> List.tryHead
                |> Option.map (fun c -> c.TargetMethodFullName)
            | _ ->
                // Check for direct symbol references
                match node.Symbol with
                | Some (:? FSharpMemberOrFunctionOrValue as mfv) ->
                    try Some mfv.FullName with _ -> None
                | _ -> None)
        |> Set.ofSeq

    // Filter bodies to only those referenced
    allBodies
    |> Map.filter (fun key _ -> Set.contains key referencedMembers)

/// Main entry point: Extract and process member bodies using the TypedTreeZipper
/// CRITICAL: This should be called AFTER reachability analysis (Phase 3)
let run
    (psg: ProgramSemanticGraph)
    (checkResults: FSharpCheckProjectResults)
    : BakerResult =

    // Step 1: Extract all member bodies using the zipper
    let allBodies = extractWithZipper psg checkResults

    // Step 2: Filter to only members referenced by reachable nodes
    let reachableBodies = filterToReachable allBodies psg

    let correlatedCount =
        reachableBodies
        |> Map.filter (fun _ m -> m.PSGBindingId.IsSome)
        |> Map.count

    {
        MemberBodies = reachableBodies
        Statistics = {
            TotalMembers = allBodies.Count
            MembersWithBodies = reachableBodies.Count
            MembersCorrelatedWithPSG = correlatedCount
            ProcessingTimeMs = 0.0
        }
    }
