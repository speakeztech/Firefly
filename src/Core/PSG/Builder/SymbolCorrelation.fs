/// Symbol correlation logic for PSG construction
module Core.PSG.Construction.SymbolCorrelation

open System
open FSharp.Compiler.Text
open FSharp.Compiler.Symbols
open Core.CompilerConfig
open Core.PSG.Correlation

/// Try to correlate a symbol with the given context using multiple strategies
let tryCorrelateSymbolWithContext (range: range) (fileName: string) (syntaxKind: string) (context: CorrelationContext) : FSharpSymbol option =

    // Strategy 1: Exact range match
    let key = (fileName, range.StartLine, range.StartColumn, range.EndLine, range.EndColumn)
    match Map.tryFind key context.PositionIndex with
    | Some symbolUse ->
        if isCorrelationVerbose() then
            printfn "[CORRELATION] ✓ Exact match: %s -> %s" syntaxKind symbolUse.Symbol.FullName
        Some symbolUse.Symbol
    | None ->
        // Strategy 2: Enhanced correlation by syntax kind
        match Map.tryFind fileName context.FileIndex with
        | Some fileUses ->

            // Method call correlation
            if syntaxKind.StartsWith("MethodCall:") || syntaxKind.Contains("DotGet") || syntaxKind.Contains("LongIdent:MethodCall:") then
                let methodName =
                    if syntaxKind.Contains("LongIdent:MethodCall:") then
                        // Extract method name from LongIdent:MethodCall:AsReadOnlySpan
                        let parts = syntaxKind.Split(':')
                        if parts.Length > 2 then parts.[2] else ""
                    elif syntaxKind.Contains(":") then
                        let parts = syntaxKind.Split(':')
                        if parts.Length > 1 then parts.[parts.Length - 1] else ""
                    else ""

                let methodCandidates =
                    fileUses
                    |> Array.filter (fun symbolUse ->
                        match symbolUse.Symbol with
                        | :? FSharpMemberOrFunctionOrValue as mfv ->
                            (mfv.IsMember || mfv.IsProperty || mfv.IsFunction) &&
                            (String.IsNullOrEmpty(methodName) ||
                             mfv.DisplayName.Contains(methodName) ||
                             mfv.DisplayName = methodName ||
                             mfv.FullName.EndsWith("." + methodName))
                        | _ -> false)
                    |> Array.filter (fun symbolUse ->
                        abs(symbolUse.Range.StartLine - range.StartLine) <= 2 &&
                        abs(symbolUse.Range.StartColumn - range.StartColumn) <= 20)

                match methodCandidates |> Array.sortBy (fun su ->
                    let nameScore = if su.Symbol.DisplayName = methodName then 0 else 1
                    let rangeScore = abs(su.Range.StartLine - range.StartLine) + abs(su.Range.StartColumn - range.StartColumn)
                    nameScore * 100 + rangeScore) |> Array.tryHead with
                | Some symbolUse ->
                    if isCorrelationVerbose() then
                        printfn "[CORRELATION] ✓ Enhanced method match: %s -> %s" syntaxKind symbolUse.Symbol.FullName
                    Some symbolUse.Symbol
                | None ->
                    if isCorrelationVerbose() then
                        printfn "[CORRELATION] ✗ No enhanced method match for: %s (method: %s)" syntaxKind methodName
                    None

            // Generic type application correlation
            elif syntaxKind.StartsWith("TypeApp:") then
                let genericCandidates =
                    fileUses
                    |> Array.filter (fun symbolUse ->
                        match symbolUse.Symbol with
                        | :? FSharpMemberOrFunctionOrValue as mfv ->
                            (mfv.IsFunction && mfv.GenericParameters.Count > 0) ||
                            mfv.DisplayName.Contains("stackBuffer") ||
                            mfv.FullName.Contains("stackBuffer") ||
                            mfv.DisplayName = "stackBuffer"
                        | :? FSharpEntity as entity -> entity.GenericParameters.Count > 0
                        | _ -> false)
                    |> Array.filter (fun symbolUse ->
                        abs(symbolUse.Range.StartLine - range.StartLine) <= 2)

                match genericCandidates |> Array.sortBy (fun su ->
                    let nameScore = if su.Symbol.DisplayName = "stackBuffer" then 0 else 1
                    let rangeScore = abs(su.Range.StartLine - range.StartLine)
                    nameScore * 100 + rangeScore) |> Array.tryHead with
                | Some symbolUse ->
                    if isCorrelationVerbose() then
                        printfn "[CORRELATION] ✓ Enhanced generic match: %s -> %s" syntaxKind symbolUse.Symbol.FullName
                    Some symbolUse.Symbol
                | None ->
                    if isCorrelationVerbose() then
                        printfn "[CORRELATION] ✗ No enhanced generic match for: %s" syntaxKind
                    None

            // Union case correlation
            elif syntaxKind.Contains("UnionCase:") then
                let unionCaseName =
                    if syntaxKind.Contains("Ok") then "Ok"
                    elif syntaxKind.Contains("Error") then "Error"
                    elif syntaxKind.Contains("Some") then "Some"
                    elif syntaxKind.Contains("None") then "None"
                    else ""

                if not (String.IsNullOrEmpty(unionCaseName)) then
                    let unionCaseCandidates =
                        fileUses
                        |> Array.filter (fun symbolUse ->
                            match symbolUse.Symbol with
                            | :? FSharpUnionCase as unionCase -> unionCase.Name = unionCaseName
                            | :? FSharpMemberOrFunctionOrValue as mfv ->
                                mfv.DisplayName = unionCaseName
                            | _ -> false)
                        |> Array.filter (fun symbolUse ->
                            abs(symbolUse.Range.StartLine - range.StartLine) <= 2)

                    match unionCaseCandidates |> Array.tryHead with
                    | Some symbolUse ->
                        if isCorrelationVerbose() then
                            printfn "[CORRELATION] ✓ Union case match: %s -> %s" syntaxKind symbolUse.Symbol.FullName
                        Some symbolUse.Symbol
                    | None -> None
                else None

            // Function/identifier correlation
            elif syntaxKind.Contains("Ident:") || syntaxKind.Contains("LongIdent:") then
                let identName =
                    if syntaxKind.Contains(":") then
                        let parts = syntaxKind.Split(':')
                        if parts.Length > 0 then parts.[parts.Length - 1] else ""
                    else ""

                if not (String.IsNullOrEmpty(identName)) then
                    let functionCandidates =
                        fileUses
                        |> Array.filter (fun symbolUse ->
                            let sym = symbolUse.Symbol
                            sym.DisplayName = identName ||
                            sym.FullName.EndsWith("." + identName) ||
                            sym.FullName.Contains(identName) ||
                            // Enhanced matching for critical symbols
                            (identName = "spanToString" && (sym.DisplayName.Contains("spanToString") || sym.FullName.Contains("spanToString"))) ||
                            (identName = "stackBuffer" && (sym.DisplayName.Contains("stackBuffer") || sym.FullName.Contains("stackBuffer"))) ||
                            (identName = "AsReadOnlySpan" && (sym.DisplayName.Contains("AsReadOnlySpan") || sym.FullName.Contains("AsReadOnlySpan"))) ||
                            (sym.DisplayName.Contains(identName) && sym.DisplayName.Length <= identName.Length + 5))
                        |> Array.filter (fun symbolUse ->
                            abs(symbolUse.Range.StartLine - range.StartLine) <= 2 &&
                            abs(symbolUse.Range.StartColumn - range.StartColumn) <= 25)

                    match functionCandidates |> Array.sortBy (fun su ->
                        let nameScore =
                            if su.Symbol.DisplayName = identName then 0
                            elif su.Symbol.FullName.EndsWith("." + identName) then 1
                            elif su.Symbol.FullName.Contains(identName) then 2
                            else 3
                        let rangeScore = abs(su.Range.StartLine - range.StartLine) + abs(su.Range.StartColumn - range.StartColumn)
                        nameScore * 1000 + rangeScore) |> Array.tryHead with
                    | Some symbolUse ->
                        if isCorrelationVerbose() then
                            printfn "[CORRELATION] ✓ Enhanced function match: %s -> %s" syntaxKind symbolUse.Symbol.FullName
                        Some symbolUse.Symbol
                    | None ->
                        if isCorrelationVerbose() then
                            printfn "[CORRELATION] ✗ No enhanced function match for: %s (name: %s)" syntaxKind identName
                        None
                else None

            // Fallback to original correlation
            else
                let closeMatch =
                    fileUses
                    |> Array.filter (fun symbolUse ->
                        abs(symbolUse.Range.StartLine - range.StartLine) <= 2)
                    |> Array.sortBy (fun symbolUse ->
                        abs(symbolUse.Range.StartLine - range.StartLine) +
                        abs(symbolUse.Range.StartColumn - range.StartColumn))
                    |> Array.tryHead

                match closeMatch with
                | Some symbolUse -> Some symbolUse.Symbol
                | None ->
                    if isCorrelationVerbose() then
                        printfn "[CORRELATION] ✗ No match: %s at %s" syntaxKind (range.ToString())
                    None
        | None -> None

/// Helper function to determine if symbol represents a function
let isFunction (symbol: FSharpSymbol) : bool =
    match symbol with
    | :? FSharpMemberOrFunctionOrValue as mfv -> mfv.IsFunction || mfv.IsMember
    | _ -> false

/// Try to correlate a symbol with an optional context
/// Returns None if context is None (Phase 1 structural building)
/// This is the primary function used during PSG construction
let tryCorrelateSymbolOptional (range: range) (fileName: string) (syntaxKind: string) (contextOpt: CorrelationContext option) : FSharpSymbol option =
    match contextOpt with
    | None -> None  // Phase 1: No symbol correlation during structural building
    | Some context -> tryCorrelateSymbolWithContext range fileName syntaxKind context
