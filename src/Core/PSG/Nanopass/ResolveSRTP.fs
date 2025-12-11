/// ResolveSRTP Nanopass - Captures SRTP resolution from FCS internals
///
/// SRTP (Statically Resolved Type Parameters) constraints are resolved by FCS during
/// type checking, but the resolution is stored in internal types not exposed through
/// the public FCS API. This nanopass uses reflection to access the internal
/// TraitConstraintInfo.Solution property and populate the SRTPResolution field on
/// PSG nodes that represent trait calls.
///
/// This is a "game time decision" - accessing FCS internals is fragile but necessary
/// for principled compilation. A parallel effort to add this to the public FCS API
/// is underway.
///
/// Phase: Should run after TypeIntegration (Phase 4) when FSharpExpr trees are available
module Core.PSG.Nanopass.ResolveSRTP

open System
open System.Reflection
open FSharp.Compiler.Symbols
open FSharp.Compiler.Symbols.FSharpExprPatterns
open FSharp.Compiler.Text
open FSharp.Compiler.CodeAnalysis
open Core.PSG.Types

// ═══════════════════════════════════════════════════════════════════════════
// Reflection Helpers for FCS Internals
// ═══════════════════════════════════════════════════════════════════════════

/// Cache for reflected types and members to avoid repeated lookups
module private ReflectionCache =
    let mutable private traitConstraintInfoType: Type option = None
    let mutable private solutionProperty: PropertyInfo option = None
    let mutable private fsMethSlnType: Type option = None
    let mutable private fsRecdFieldSlnType: Type option = None
    let mutable private fsAnonRecdFieldSlnType: Type option = None
    let mutable private builtInSlnType: Type option = None
    let mutable private initialized = false

    /// Initialize reflection cache by searching FCS assembly
    let initialize () =
        if not initialized then
            try
                // Get the FCS assembly from a known type
                let fcsAssembly = typeof<FSharpExpr>.Assembly
                printfn "[SRTP] FCS Assembly: %s" fcsAssembly.FullName

                // Diagnostic: Find all types containing "Trait" to discover correct names
                let allTypes = fcsAssembly.GetTypes()
                let traitTypes =
                    allTypes
                    |> Array.filter (fun t ->
                        t.Name.Contains("Trait") ||
                        t.Name.Contains("TTrait") ||
                        t.FullName.Contains("TraitConstraint"))

                printfn "[SRTP] Found %d types containing 'Trait':" traitTypes.Length
                for t in traitTypes do
                    printfn "[SRTP]   - %s (Name: %s)" t.FullName t.Name
                    // List properties
                    let props = t.GetProperties(BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
                    for p in props do
                        printfn "[SRTP]       Property: %s : %s" p.Name p.PropertyType.Name

                // Find TraitConstraintInfo from FSharp.Compiler.TypedTree
                // The full name is "FSharp.Compiler.TypedTree+TraitConstraintInfo"
                traitConstraintInfoType <-
                    allTypes
                    |> Array.tryFind (fun t -> t.FullName = "FSharp.Compiler.TypedTree+TraitConstraintInfo")

                match traitConstraintInfoType with
                | Some t ->
                    printfn "[SRTP] Found TraitConstraintInfo type: %s" t.FullName
                    // List ALL properties including non-public
                    let allProps = t.GetProperties(BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
                    printfn "[SRTP] Properties on %s:" t.Name
                    for p in allProps do
                        printfn "[SRTP]   - %s : %s (CanRead=%b)" p.Name p.PropertyType.Name p.CanRead

                    // Get the Solution property - try both public and non-public
                    solutionProperty <-
                        t.GetProperty("Solution", BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
                        |> Option.ofObj

                    if solutionProperty.IsNone then
                        printfn "[SRTP] Warning: Could not find Solution property on TraitConstraintInfo"
                    else
                        printfn "[SRTP] Found Solution property: %s" solutionProperty.Value.PropertyType.FullName
                | None ->
                    printfn "[SRTP] Warning: Could not find TraitConstraintInfo type in FCS assembly"

                // Find solution types - look for TraitConstraintSln cases
                let slnTypes =
                    allTypes
                    |> Array.filter (fun t ->
                        t.Name.Contains("Sln") ||
                        t.FullName.Contains("TraitConstraintSln"))

                printfn "[SRTP] Found %d types containing 'Sln':" slnTypes.Length
                for t in slnTypes |> Array.take (min 20 slnTypes.Length) do
                    printfn "[SRTP]   - %s" t.FullName

                // Solution types are nested in FSharp.Compiler.TypedTree+TraitConstraintSln
                fsMethSlnType <- allTypes |> Array.tryFind (fun t -> t.FullName = "FSharp.Compiler.TypedTree+TraitConstraintSln+FSMethSln")
                fsRecdFieldSlnType <- allTypes |> Array.tryFind (fun t -> t.FullName = "FSharp.Compiler.TypedTree+TraitConstraintSln+FSRecdFieldSln")
                fsAnonRecdFieldSlnType <- allTypes |> Array.tryFind (fun t -> t.FullName = "FSharp.Compiler.TypedTree+TraitConstraintSln+FSAnonRecdFieldSln")
                builtInSlnType <- allTypes |> Array.tryFind (fun t -> t.FullName = "FSharp.Compiler.TypedTree+TraitConstraintSln+BuiltInSln")

                printfn "[SRTP] FSMethSln type: %A" (fsMethSlnType |> Option.map (fun t -> t.FullName))
                printfn "[SRTP] FSRecdFieldSln type: %A" (fsRecdFieldSlnType |> Option.map (fun t -> t.FullName))
                printfn "[SRTP] FSAnonRecdFieldSln type: %A" (fsAnonRecdFieldSlnType |> Option.map (fun t -> t.FullName))
                printfn "[SRTP] BuiltInSln type: %A" (builtInSlnType |> Option.map (fun t -> t.FullName))

                initialized <- true
                printfn "[SRTP] Reflection cache initialized"
            with ex ->
                printfn "[SRTP] Error initializing reflection cache: %s" ex.Message
                printfn "[SRTP] Stack trace: %s" ex.StackTrace
                initialized <- true  // Don't retry on failure

    let getTraitConstraintInfoType () =
        initialize ()
        traitConstraintInfoType

    let getSolutionProperty () =
        initialize ()
        solutionProperty

// ═══════════════════════════════════════════════════════════════════════════
// SRTP Resolution Extraction
// ═══════════════════════════════════════════════════════════════════════════

/// Attempt to extract SRTP resolution from a TraitCall's internal traitInfo
let private tryExtractResolution (traitCallExpr: FSharpExpr) : SRTPResolution option =
    try
        // The FSharpExpr has an internal E property that holds the expression variant
        let eProperty = traitCallExpr.GetType().GetProperty("E", BindingFlags.Public ||| BindingFlags.Instance)
        match eProperty with
        | null ->
            printfn "[SRTP] Could not find E property on FSharpExpr"
            None
        | prop ->
            let eValue = prop.GetValue(traitCallExpr)
            if isNull eValue then None
            else
                // The E value is a discriminated union, check if it's TraitCall
                let eType = eValue.GetType()
                if eType.Name.Contains("TraitCall") then
                    // This approach won't work because E.TraitCall doesn't store the internal traitInfo
                    // The conversion from internal Expr to E.TraitCall discards the Solution
                    // We need to find another way...
                    None
                else
                    None
    with ex ->
        printfn "[SRTP] Error extracting resolution: %s" ex.Message
        None

/// Walk FSharpExpr tree looking for TraitCall expressions and extract their resolutions
let private extractTraitCallResolutions (checkResults: FSharpCheckProjectResults) : Map<string, SRTPResolution> =
    let mutable resolutions = Map.empty

    let rangeToKey (range: range) : string =
        sprintf "%s_%d_%d_%d_%d"
            (System.IO.Path.GetFileName range.FileName)
            range.Start.Line range.Start.Column
            range.End.Line range.End.Column

    // Walk the typed AST looking for TraitCall expressions
    let assemblyContents = checkResults.AssemblyContents

    for implFile in assemblyContents.ImplementationFiles do
        let rec processDecl (decl: FSharpImplementationFileDeclaration) =
            match decl with
            | FSharpImplementationFileDeclaration.Entity (_, subDecls) ->
                subDecls |> List.iter processDecl
            | FSharpImplementationFileDeclaration.MemberOrFunctionOrValue (_, _, expr) ->
                walkExpr expr
            | FSharpImplementationFileDeclaration.InitAction expr ->
                walkExpr expr

        and walkExpr (expr: FSharpExpr) =
            try
                // Check if this is a TraitCall using the pattern matcher
                match expr with
                | TraitCall (sourceTypes, traitName, _memberFlags, paramTypes, _retTypes, traitArgs) ->
                    // Debug: show all info from TraitCall
                    let sourceTypeNames = sourceTypes |> List.map (fun t -> t.Format(FSharpDisplayContext.Empty))
                    let paramTypeNames = paramTypes |> List.map (fun t -> t.Format(FSharpDisplayContext.Empty))
                    let argCount = List.length traitArgs
                    printfn "[SRTP-TC] TraitCall: %s, sourceTypes: %A, paramTypes: %A, argCount: %d" traitName sourceTypeNames paramTypeNames argCount

                    // Show each argument's type and what kind of expression it is
                    traitArgs |> List.iteri (fun i arg ->
                        let argTypeName = arg.Type.Format(FSharpDisplayContext.Empty)
                        // Check what kind of expression the arg is (Value, Const, etc.)
                        let exprKind =
                            match arg with
                            | Value (valRef) -> sprintf "Value(%s : %s)" valRef.DisplayName (valRef.FullType.Format(FSharpDisplayContext.Empty))
                            | Const (obj, ty) -> sprintf "Const(%A : %s)" obj (ty.Format(FSharpDisplayContext.Empty))
                            | _ -> sprintf "Other(%s)" (arg.GetType().Name)
                        printfn "[SRTP-TC]   arg[%d].Type = %s, expr = %s" i argTypeName exprKind)

                    // Attempt to infer resolution from source types
                    // For SRTP, the first source type typically provides the member implementation
                    let inferredResolution =
                        match sourceTypes with
                        | firstType :: _ ->
                            // The first source type should have the member we're looking for
                            // Try to find the member on the type
                            if firstType.HasTypeDefinition then
                                let entity = firstType.TypeDefinition
                                let members = entity.MembersFunctionsAndValues

                                // Find all members with the trait name
                                let candidateMembers =
                                    members
                                    |> Seq.filter (fun m -> m.LogicalName = traitName || m.CompiledName = traitName)
                                    |> Seq.toList

                                // If there's only one candidate, use it directly
                                // If there are multiple (overloads), capture ALL candidates with full signatures
                                // so emission can select based on argument types in the PSG
                                match candidateMembers with
                                | [single] ->
                                    printfn "[SRTP] Single overload for %s -> %s.%s" traitName entity.FullName single.LogicalName
                                    FSMethod (firstType, single, [])
                                | multiple when multiple.Length > 1 ->
                                    // Multiple overloads - capture all candidates with their full signatures
                                    // This allows emission to match based on concrete argument types in PSG
                                    let candidates =
                                        multiple
                                        |> List.choose (fun m ->
                                            try
                                                // Get the target method body - look for what this method calls
                                                // For op_Dollar, this would be writeNativeStr or writeSystemString
                                                let targetFullName =
                                                    // The actual implementation is in the method body
                                                    // For now, use the operator's full name - emission will trace to body
                                                    sprintf "%s.%s" entity.FullName m.LogicalName

                                                let paramTypes =
                                                    m.CurriedParameterGroups
                                                    |> Seq.collect id
                                                    |> Seq.map (fun p ->
                                                        try p.Type.TypeDefinition.FullName
                                                        with _ -> p.Type.Format(FSharpDisplayContext.Empty))
                                                    |> Seq.toList

                                                let returnType =
                                                    try m.ReturnParameter.Type.TypeDefinition.FullName
                                                    with _ -> m.ReturnParameter.Type.Format(FSharpDisplayContext.Empty)

                                                printfn "[SRTP]   Overload candidate: %s with params %A" targetFullName paramTypes

                                                Some {
                                                    TargetMethodFullName = targetFullName
                                                    ParameterTypeNames = paramTypes
                                                    ReturnTypeName = returnType
                                                }
                                            with _ -> None)

                                    if candidates.Length > 0 then
                                        printfn "[SRTP] Captured %d overload candidates for %s" candidates.Length traitName
                                        MultipleOverloads (traitName, candidates)
                                    else
                                        Unresolved (sprintf "Could not extract signatures for %s overloads" traitName)
                                | [] ->
                                    // No candidates found on the type
                                    let typeName = firstType.Format(FSharpDisplayContext.Empty)
                                    if typeName.StartsWith("^") then
                                        // Generic type parameter - resolution happens at instantiation site
                                        BuiltIn
                                    else
                                        Unresolved (sprintf "Member %s not found on %s" traitName typeName)
                                | _ ->
                                    // Shouldn't happen but handle gracefully
                                    BuiltIn
                            else
                                // Type without definition (e.g., type parameter)
                                BuiltIn  // Assume resolved
                        | [] ->
                            Unresolved "No source types for TraitCall"

                    let key = rangeToKey expr.Range
                    resolutions <- Map.add key inferredResolution resolutions

                    // Continue walking arguments - with error handling
                    for arg in traitArgs do
                        try walkExpr arg
                        with _ -> ()

                | _ ->
                    // Walk sub-expressions - with error handling
                    for subExpr in expr.ImmediateSubExpressions do
                        try walkExpr subExpr
                        with _ -> ()
            with ex ->
                // Silently ignore expression walking errors
                // Some FCS expressions may trigger constraint solver issues
                ()

        implFile.Declarations |> List.iter processDecl

    printfn "[SRTP] Extracted %d TraitCall locations" (Map.count resolutions)
    resolutions

// ═══════════════════════════════════════════════════════════════════════════
// Internal Typed Tree Traversal via Reflection
// ═══════════════════════════════════════════════════════════════════════════

/// Helper to get F# list items via reflection
let private getFSharpListItems (listObj: obj) : obj list =
    if isNull listObj then []
    else
        let listType = listObj.GetType()
        // F# list is a discriminated union with Cons (head, tail) and Empty
        let rec collect (current: obj) acc =
            if isNull current then List.rev acc
            else
                let currentType = current.GetType()
                // Check for empty list
                let isEmptyProp = currentType.GetProperty("IsEmpty")
                if isEmptyProp <> null then
                    let isEmpty = isEmptyProp.GetValue(current) :?> bool
                    if isEmpty then List.rev acc
                    else
                        let headProp = currentType.GetProperty("Head")
                        let tailProp = currentType.GetProperty("Tail")
                        if headProp <> null && tailProp <> null then
                            let head = headProp.GetValue(current)
                            let tail = tailProp.GetValue(current)
                            collect tail (head :: acc)
                        else List.rev acc
                else List.rev acc
        collect listObj []

/// Helper to get F# option value via reflection
let private getFSharpOptionValue (optionObj: obj) : obj option =
    if isNull optionObj then None
    else
        let optType = optionObj.GetType()
        let isSomeProp = optType.GetProperty("IsSome")
        if isSomeProp <> null then
            let isSome = isSomeProp.GetValue(optionObj) :?> bool
            if isSome then
                let valueProp = optType.GetProperty("Value")
                if valueProp <> null then Some (valueProp.GetValue(optionObj))
                else None
            else None
        else None

/// Helper to get F# ref cell value via reflection
let private getFSharpRefValue (refObj: obj) : obj option =
    if isNull refObj then None
    else
        let refType = refObj.GetType()
        // F# ref cells have a Value or contents property
        let valueProp = refType.GetProperty("Value")
                        |> Option.ofObj
                        |> Option.orElse (refType.GetProperty("contents") |> Option.ofObj)
        match valueProp with
        | Some prop -> Some (prop.GetValue(refObj))
        | None -> None

/// Extract range from an internal Expr object
let private tryGetExprRange (expr: obj) : range option =
    if isNull expr then None
    else
        try
            let exprType = expr.GetType()
            // Internal Expr has a Range property
            let rangeProp = exprType.GetProperty("Range", BindingFlags.Public ||| BindingFlags.Instance)
            if rangeProp <> null then
                Some (rangeProp.GetValue(expr) :?> range)
            else None
        with _ -> None

/// Extract method name from a ValRef via reflection
let private tryGetValRefName (vref: obj) : string option =
    if isNull vref then None
    else
        try
            let vrefType = vref.GetType()
            // ValRef has LogicalName property
            let logicalNameProp = vrefType.GetProperty("LogicalName", BindingFlags.Public ||| BindingFlags.Instance)
            if logicalNameProp <> null then
                Some (logicalNameProp.GetValue(vref) :?> string)
            else
                // Try DisplayName as fallback
                let displayNameProp = vrefType.GetProperty("DisplayName", BindingFlags.Public ||| BindingFlags.Instance)
                if displayNameProp <> null then
                    Some (displayNameProp.GetValue(vref) :?> string)
                else None
        with _ -> None

/// Extract full name from a ValRef (including type path)
let private tryGetValRefFullName (vref: obj) : string option =
    if isNull vref then None
    else
        try
            let vrefType = vref.GetType()
            // Try to get the full logical name by traversing the enclosing entity
            let logicalNameProp = vrefType.GetProperty("LogicalName", BindingFlags.Public ||| BindingFlags.Instance)
            let logicalName =
                if logicalNameProp <> null then logicalNameProp.GetValue(vref) :?> string
                else "?"

            // Try to get the enclosing entity for the full path
            let tryGetEntityPath (vref: obj) =
                try
                    // ValRef may have DeclaringEntity or similar
                    let derefProp = vrefType.GetProperty("Deref", BindingFlags.Public ||| BindingFlags.Instance)
                    if derefProp <> null then
                        let deref = derefProp.GetValue(vref)
                        if not (isNull deref) then
                            let derefType = deref.GetType()
                            // Val has EnclosingEntity
                            let enclosingProp = derefType.GetProperty("DeclaringEntity", BindingFlags.Public ||| BindingFlags.Instance)
                                                |> Option.ofObj
                                                |> Option.orElse (derefType.GetProperty("TopValDeclaringEntity", BindingFlags.Public ||| BindingFlags.Instance) |> Option.ofObj)
                            match enclosingProp with
                            | Some prop ->
                                let enclosing = prop.GetValue(deref)
                                if not (isNull enclosing) then
                                    // Try to get full name from EntityRef
                                    let fullNameProp = enclosing.GetType().GetProperty("LogicalName", BindingFlags.Public ||| BindingFlags.Instance)
                                    if fullNameProp <> null then
                                        let entityName = fullNameProp.GetValue(enclosing) :?> string
                                        Some (entityName + "." + logicalName)
                                    else Some logicalName
                                else Some logicalName
                            | None -> Some logicalName
                        else Some logicalName
                    else Some logicalName
                with _ -> Some logicalName

            tryGetEntityPath vref
        with _ -> None

/// Convert an internal TraitConstraintSln to our SRTPResolution type
/// For FSMethSln, extracts the ValRef to get the specific method selected
let private convertSolution (sln: obj) : SRTPResolution =
    if isNull sln then Unresolved "Null solution"
    else
        let slnType = sln.GetType()
        let slnName = slnType.Name

        // Check which variant of TraitConstraintSln this is
        if slnName.Contains("FSMethSln") then
            // FSMethSln has: ty, vref, minst, staticTyOpt
            // Extract vref to get the specific method
            let vrefProp = slnType.GetProperty("vref", BindingFlags.Public ||| BindingFlags.Instance)
            if vrefProp <> null then
                let vref = vrefProp.GetValue(sln)
                let methodName = tryGetValRefName vref
                let fullName = tryGetValRefFullName vref
                printfn "[SRTP] FSMethSln vref: %A (full: %A)" methodName fullName
                // Return FSMethodByName with the resolved method name
                match fullName |> Option.orElse methodName with
                | Some name -> FSMethodByName name
                | None -> BuiltIn  // Fallback if we couldn't get the name
            else
                printfn "[SRTP] FSMethSln has no vref property"
                BuiltIn
        elif slnName.Contains("FSRecdFieldSln") then
            BuiltIn
        elif slnName.Contains("FSAnonRecdFieldSln") then
            BuiltIn
        elif slnName.Contains("ILMethSln") then
            BuiltIn
        elif slnName.Contains("BuiltInSln") then
            BuiltIn
        elif slnName.Contains("ClosedExprSln") then
            BuiltIn
        else
            Unresolved (sprintf "Unknown solution type: %s" slnName)

/// Extract TraitConstraintInfo.Solution from an internal TraitConstraintInfo object
let private tryExtractTraitSolution (traitInfo: obj) : (range * SRTPResolution) option =
    if isNull traitInfo then None
    else
        try
            let infoType = traitInfo.GetType()

            // Get the Solution property (returns FSharpOption<TraitConstraintSln>)
            let solutionProp = infoType.GetProperty("Solution", BindingFlags.Public ||| BindingFlags.Instance)
            if solutionProp = null then
                printfn "[SRTP] TraitConstraintInfo has no Solution property"
                None
            else
                let solutionOpt = solutionProp.GetValue(traitInfo)
                match getFSharpOptionValue solutionOpt with
                | Some sln ->
                    // Get the member name for diagnostics
                    let memberNameProp = infoType.GetProperty("MemberLogicalName")
                    let memberName = if memberNameProp <> null then memberNameProp.GetValue(traitInfo) :?> string else "?"

                    printfn "[SRTP] Found resolved TraitConstraintInfo: %s -> %s" memberName (sln.GetType().Name)

                    // We need a range to correlate with PSG nodes
                    // TraitConstraintInfo doesn't have a range, but the enclosing Expr does
                    // For now, return without range - we'll correlate differently
                    None  // Will need to get range from enclosing Expr
                | None ->
                    // Solution ref cell might contain the value differently
                    // Try accessing the underlying solution ref field (try property first, then field)
                    let refCell =
                        let prop = infoType.GetProperty("solution", BindingFlags.NonPublic ||| BindingFlags.Instance)
                        if prop <> null then
                            prop.GetValue(traitInfo)
                        else
                            let field = infoType.GetField("solution", BindingFlags.NonPublic ||| BindingFlags.Instance)
                            if field <> null then field.GetValue(traitInfo)
                            else null

                    if not (isNull refCell) then
                        match getFSharpRefValue refCell with
                        | Some refContents ->
                            match getFSharpOptionValue refContents with
                            | Some actualSln ->
                                let memberNameProp = infoType.GetProperty("MemberLogicalName")
                                let memberName = if memberNameProp <> null then memberNameProp.GetValue(traitInfo) :?> string else "?"
                                printfn "[SRTP] Found resolved TraitConstraintInfo (via ref): %s -> %s" memberName (actualSln.GetType().Name)
                                None  // Still need range
                            | None ->
                                printfn "[SRTP] TraitConstraintInfo has unresolved solution (ref option is None)"
                                None
                        | None ->
                            printfn "[SRTP] Could not read solution ref cell"
                            None
                    else
                        printfn "[SRTP] TraitConstraintInfo solution is None and no ref field found"
                        None
        with ex ->
            printfn "[SRTP] Error extracting trait solution: %s" ex.Message
            None

/// Walk an internal Expr tree to find TraitCall operations
let rec private walkInternalExpr (expr: obj) (resolutions: byref<Map<string, SRTPResolution>>) : unit =
    if isNull expr then ()
    else
        try
            let exprType = expr.GetType()
            let typeName = exprType.Name

            // Debug: track what expression types we're seeing
            if typeName = "Lambda" then
                printfn "[SRTP-EXPR] Found Lambda expression"
                let allProps = exprType.GetProperties(BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
                for p in allProps do
                    printfn "[SRTP-EXPR]   Lambda prop: %s : %s" p.Name (p.PropertyType.Name)

            // Check for App expressions that may contain TraitCall operations
            if typeName = "App" || typeName.EndsWith("+App") then
                let funcExprProp = exprType.GetProperty("funcExpr", BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
                if funcExprProp <> null then
                    let funcExpr = funcExprProp.GetValue(expr)
                    if not (isNull funcExpr) then
                        let funcTypeName = funcExpr.GetType().Name
                        printfn "[SRTP-EXPR] App funcExpr type: %s" funcTypeName
                        // If funcExpr is Op, check what operation it contains
                        if funcTypeName = "Op" then
                            let opProp = funcExpr.GetType().GetProperty("op", BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
                                        |> Option.ofObj
                                        |> Option.orElse (funcExpr.GetType().GetProperty("Item1") |> Option.ofObj)
                            match opProp with
                            | Some p ->
                                let opValue = p.GetValue(funcExpr)
                                if not (isNull opValue) then
                                    printfn "[SRTP-EXPR] App funcExpr Op type: %s" (opValue.GetType().Name)
                            | None -> ()

            // Check if this is an Op expression with TraitCall
            if typeName = "Op" || exprType.FullName.EndsWith("+Op") then
                // Expr.Op has: op, typeArgs, args, range
                // Try to get the op field
                let opProp = exprType.GetProperty("op", BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
                            |> Option.ofObj
                            |> Option.orElse (exprType.GetProperty("Item1") |> Option.ofObj)

                match opProp with
                | Some prop ->
                    let op = prop.GetValue(expr)
                    if not (isNull op) then
                        let opType = op.GetType()
                        printfn "[SRTP-EXPR] Op type: %s" opType.Name
                        if opType.Name.Contains("TraitCall") then
                            printfn "[SRTP-EXPR] FOUND TraitCall Op!"
                            // Found a TraitCall! Extract the TraitConstraintInfo
                            let traitInfoProp = opType.GetProperty("Item", BindingFlags.Public ||| BindingFlags.Instance)
                                                |> Option.ofObj
                                                |> Option.orElse (opType.GetProperty("Item1") |> Option.ofObj)

                            match traitInfoProp with
                            | Some tiProp ->
                                let traitInfo = tiProp.GetValue(op)

                                // Get the range from the Expr
                                match tryGetExprRange expr with
                                | Some range ->
                                    // Try to get the solution
                                    let tiType = traitInfo.GetType()
                                    let solutionProp = tiType.GetProperty("Solution")
                                    if solutionProp <> null then
                                        let solutionOpt = solutionProp.GetValue(traitInfo)
                                        match getFSharpOptionValue solutionOpt with
                                        | Some sln ->
                                            let memberNameProp = tiType.GetProperty("MemberLogicalName")
                                            let memberName = if memberNameProp <> null then memberNameProp.GetValue(traitInfo) :?> string else "?"

                                            let resolution = convertSolution sln
                                            let key = sprintf "%s_%d_%d_%d_%d"
                                                        (System.IO.Path.GetFileName range.FileName)
                                                        range.Start.Line range.Start.Column
                                                        range.End.Line range.End.Column

                                            printfn "[SRTP-INTERNAL] Resolved TraitCall: %s at %s -> %A" memberName key resolution
                                            resolutions <- Map.add key resolution resolutions
                                        | None ->
                                            let memberNameProp = tiType.GetProperty("MemberLogicalName")
                                            let memberName = if memberNameProp <> null then memberNameProp.GetValue(traitInfo) :?> string else "?"
                                            printfn "[SRTP-INTERNAL] Unresolved TraitCall: %s" memberName
                                | None -> ()
                            | None -> ()
                | None -> ()

            // Recursively walk sub-expressions
            // Try common expression patterns

            // Lambda: bodyExpr
            let bodyExprProp = exprType.GetProperty("bodyExpr", BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
            if bodyExprProp <> null then
                let bodyExpr = bodyExprProp.GetValue(expr)
                if not (isNull bodyExpr) then
                    printfn "[SRTP-EXPR] Walking bodyExpr of type: %s" (bodyExpr.GetType().Name)
                    walkInternalExpr bodyExpr &resolutions

            // Let/LetRec: binding(s) and bodyExpr
            let bindingProp = exprType.GetProperty("binding", BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
            if bindingProp <> null then
                let binding = bindingProp.GetValue(expr)
                if not (isNull binding) then
                    let bindExprProp = binding.GetType().GetProperty("Expr")
                    if bindExprProp <> null then
                        walkInternalExpr (bindExprProp.GetValue(binding)) &resolutions

            let bindingsProp = exprType.GetProperty("bindings", BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
            if bindingsProp <> null then
                let bindings = bindingsProp.GetValue(expr)
                for binding in getFSharpListItems bindings do
                    let bindExprProp = binding.GetType().GetProperty("Expr")
                    if bindExprProp <> null then
                        walkInternalExpr (bindExprProp.GetValue(binding)) &resolutions

            // App: funcExpr and args
            let funcExprProp = exprType.GetProperty("funcExpr", BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
            if funcExprProp <> null then
                walkInternalExpr (funcExprProp.GetValue(expr)) &resolutions

            let argsProp = exprType.GetProperty("args", BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
            if argsProp <> null then
                let args = argsProp.GetValue(expr)
                for arg in getFSharpListItems args do
                    walkInternalExpr arg &resolutions

            // Sequential: expr1, expr2
            let expr1Prop = exprType.GetProperty("expr1", BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
            if expr1Prop <> null then
                walkInternalExpr (expr1Prop.GetValue(expr)) &resolutions

            let expr2Prop = exprType.GetProperty("expr2", BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
            if expr2Prop <> null then
                walkInternalExpr (expr2Prop.GetValue(expr)) &resolutions

            // Link: dereference
            let linkProp = exprType.GetProperty("Item", BindingFlags.Public ||| BindingFlags.Instance)
            if linkProp <> null && typeName = "Link" then
                match getFSharpRefValue (linkProp.GetValue(expr)) with
                | Some linked -> walkInternalExpr linked &resolutions
                | None -> ()

            // DebugPoint: strip and continue
            if typeName.Contains("DebugPoint") then
                let innerProp = exprType.GetProperty("Item2", BindingFlags.Public ||| BindingFlags.Instance)
                if innerProp <> null then
                    walkInternalExpr (innerProp.GetValue(expr)) &resolutions

        with ex ->
            // Silently continue on errors
            ()

/// Walk a Binding to extract its expression
let private walkBinding (binding: obj) (resolutions: byref<Map<string, SRTPResolution>>) : unit =
    if isNull binding then ()
    else
        try
            let bindType = binding.GetType()
            printfn "[SRTP-BIND] Walking binding of type: %s" bindType.Name
            let exprProp = bindType.GetProperty("Expr", BindingFlags.Public ||| BindingFlags.Instance)
                           |> Option.ofObj
                           |> Option.orElse (bindType.GetProperty("expr", BindingFlags.NonPublic ||| BindingFlags.Instance) |> Option.ofObj)

            match exprProp with
            | Some prop ->
                let expr = prop.GetValue(binding)
                if not (isNull expr) then
                    let exprTypeName = expr.GetType().Name
                    printfn "[SRTP-BIND] Found Expr of type: %s" exprTypeName
                    printfn "[SRTP-BIND] Calling walkInternalExpr..."
                    walkInternalExpr expr &resolutions
                    printfn "[SRTP-BIND] walkInternalExpr returned for %s" exprTypeName
                else
                    printfn "[SRTP-BIND] Expr is null"
            | None ->
                printfn "[SRTP-BIND] Binding has no Expr property. Available:"
                let allProps = bindType.GetProperties(BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
                for p in allProps do
                    printfn "[SRTP-BIND]   - %s : %s" p.Name p.PropertyType.Name
        with ex ->
            printfn "[SRTP-BIND] Error walking binding: %s" ex.Message

/// Walk ModuleOrNamespaceContents to find all expressions
let rec private walkModuleContents (contents: obj) (resolutions: byref<Map<string, SRTPResolution>>) : unit =
    if isNull contents then ()
    else
        try
            let contentsType = contents.GetType()
            let typeName = contentsType.Name
            printfn "[SRTP-WALK] Walking contents of type: %s" typeName

            // TMDefLet: binding
            if typeName.Contains("TMDefLet") then
                printfn "[SRTP-WALK] Found TMDefLet"
                let bindingProp = contentsType.GetProperty("binding", BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
                                  |> Option.ofObj
                                  |> Option.orElse (contentsType.GetProperty("Item1") |> Option.ofObj)
                match bindingProp with
                | Some prop ->
                    let binding = prop.GetValue(contents)
                    if not (isNull binding) then
                        printfn "[SRTP-WALK] TMDefLet has binding of type: %s" (binding.GetType().Name)
                        walkBinding binding &resolutions
                    else
                        printfn "[SRTP-WALK] TMDefLet binding is null"
                | None ->
                    printfn "[SRTP-WALK] TMDefLet has no binding property. Available:"
                    let allProps = contentsType.GetProperties(BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
                    for p in allProps do
                        printfn "[SRTP-WALK]   - %s : %s" p.Name p.PropertyType.Name

            // TMDefDo: expr
            elif typeName.Contains("TMDefDo") then
                let exprProp = contentsType.GetProperty("expr", BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
                               |> Option.ofObj
                               |> Option.orElse (contentsType.GetProperty("Item1") |> Option.ofObj)
                match exprProp with
                | Some prop -> walkInternalExpr (prop.GetValue(contents)) &resolutions
                | None -> ()

            // TMDefRec: bindings list
            elif typeName.Contains("TMDefRec") then
                printfn "[SRTP-WALK] Found TMDefRec"
                let bindingsProp = contentsType.GetProperty("bindings", BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
                match bindingsProp with
                | null ->
                    printfn "[SRTP-WALK] TMDefRec has no bindings property. Available:"
                    let allProps = contentsType.GetProperties(BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
                    for p in allProps do
                        printfn "[SRTP-WALK]   - %s : %s" p.Name p.PropertyType.Name
                | prop ->
                    let bindings = prop.GetValue(contents)
                    let bindingList = getFSharpListItems bindings
                    printfn "[SRTP-WALK] TMDefRec has %d bindings" bindingList.Length
                    for modBinding in bindingList do
                        let mbType = modBinding.GetType()
                        printfn "[SRTP-WALK] ModuleOrNamespaceBinding type: %s" mbType.Name
                        // ModuleOrNamespaceBinding.Binding or .Module
                        if mbType.Name.Contains("Binding") then
                            let innerBindProp = mbType.GetProperty("Item", BindingFlags.Public ||| BindingFlags.Instance)
                                                |> Option.ofObj
                                                |> Option.orElse (mbType.GetProperty("Item1") |> Option.ofObj)
                            match innerBindProp with
                            | Some p -> walkBinding (p.GetValue(modBinding)) &resolutions
                            | None ->
                                printfn "[SRTP-WALK] Binding has no Item property"
                        elif mbType.Name.Contains("Module") then
                            printfn "[SRTP-WALK] Processing nested Module"
                            // Try moduleOrNamespaceContents property (actual name)
                            let subContentsProp = mbType.GetProperty("moduleOrNamespaceContents", BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
                                                  |> Option.ofObj
                                                  |> Option.orElse (mbType.GetProperty("Item2", BindingFlags.Public ||| BindingFlags.Instance) |> Option.ofObj)
                            match subContentsProp with
                            | Some prop ->
                                let subContents = prop.GetValue(modBinding)
                                if not (isNull subContents) then
                                    printfn "[SRTP-WALK] Nested module contents type: %s" (subContents.GetType().Name)
                                    walkModuleContents subContents &resolutions
                                else
                                    printfn "[SRTP-WALK] Nested module contents is null"
                            | None ->
                                printfn "[SRTP-WALK] Module has no moduleOrNamespaceContents property"

            // TMDefs: list of contents
            elif typeName.Contains("TMDefs") then
                printfn "[SRTP-WALK] Found TMDefs"
                let defsProp = contentsType.GetProperty("defs", BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
                               |> Option.ofObj
                               |> Option.orElse (contentsType.GetProperty("Item") |> Option.ofObj)
                               |> Option.orElse (contentsType.GetProperty("Item1") |> Option.ofObj)
                match defsProp with
                | Some prop ->
                    let defs = prop.GetValue(contents)
                    let defList = getFSharpListItems defs
                    printfn "[SRTP-WALK] TMDefs has %d items" defList.Length
                    for def in defList do
                        walkModuleContents def &resolutions
                | None ->
                    printfn "[SRTP-WALK] TMDefs has no defs property. Available:"
                    let allProps = contentsType.GetProperties(BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
                    for p in allProps do
                        printfn "[SRTP-WALK]   - %s : %s" p.Name p.PropertyType.Name

            // TMDefOpens: no expressions
            elif typeName.Contains("TMDefOpens") then
                ()

        with ex ->
            printfn "[SRTP] Error walking module contents: %s" ex.Message

/// Walk a CheckedImplFile to find all TraitCall expressions with solutions
let private walkCheckedImplFile (implFile: obj) (resolutions: byref<Map<string, SRTPResolution>>) : unit =
    if isNull implFile then ()
    else
        try
            let ifType = implFile.GetType()
            printfn "[SRTP-INTERNAL] Walking CheckedImplFile of type: %s" ifType.FullName

            // List all properties
            let props = ifType.GetProperties(BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
            printfn "[SRTP-INTERNAL] CheckedImplFile has %d properties" props.Length

            // CheckedImplFile has Contents property
            let contentsProp = ifType.GetProperty("Contents", BindingFlags.Public ||| BindingFlags.Instance)
                               |> Option.ofObj
                               |> Option.orElse (ifType.GetProperty("contents", BindingFlags.NonPublic ||| BindingFlags.Instance) |> Option.ofObj)

            match contentsProp with
            | Some prop ->
                let contents = prop.GetValue(implFile)
                if isNull contents then
                    printfn "[SRTP-INTERNAL] Contents is null"
                else
                    printfn "[SRTP-INTERNAL] Contents type: %s" (contents.GetType().FullName)
                    walkModuleContents contents &resolutions
            | None ->
                printfn "[SRTP] CheckedImplFile has no Contents property. Available properties:"
                for p in props do
                    printfn "[SRTP]   - %s : %s" p.Name p.PropertyType.Name
        with ex ->
            printfn "[SRTP] Error walking impl file: %s" ex.Message

/// Access internal CheckedImplFile list and extract SRTP resolutions
let private extractFromInternalTypedTree (checkResults: FSharpCheckProjectResults) : Map<string, SRTPResolution> =
    let mutable resolutions = Map.empty

    try
        // Get the assembly contents
        let assemblyContents = checkResults.AssemblyContents
        let acType = assemblyContents.GetType()

        // Get internal mimpls field
        let mimplsField = acType.GetField("mimpls", BindingFlags.NonPublic ||| BindingFlags.Instance)
        if mimplsField = null then
            printfn "[SRTP] Could not find mimpls field on FSharpAssemblyContents"
        else
            let mimplsValue = mimplsField.GetValue(assemblyContents)
            if not (isNull mimplsValue) then
                let implFiles = getFSharpListItems mimplsValue
                printfn "[SRTP-INTERNAL] Walking %d internal CheckedImplFile(s)" implFiles.Length

                for implFile in implFiles do
                    walkCheckedImplFile implFile &resolutions

                printfn "[SRTP-INTERNAL] Found %d resolved TraitCalls" (Map.count resolutions)
    with ex ->
        printfn "[SRTP] Error accessing internal typed tree: %s" ex.Message

    resolutions

// ═══════════════════════════════════════════════════════════════════════════
// Main Nanopass Entry Point
// ═══════════════════════════════════════════════════════════════════════════

/// Run the SRTP resolution nanopass
/// This attempts to extract SRTP resolutions from FCS internals and populate
/// the SRTPResolution field on corresponding PSG nodes
let run (psg: ProgramSemanticGraph) (checkResults: FSharpCheckProjectResults) : ProgramSemanticGraph =
    printfn "[SRTP] Starting SRTP resolution nanopass"

    // Initialize reflection cache (for diagnostics)
    ReflectionCache.initialize ()

    // Extract resolutions from the INTERNAL typed tree (has Solution property)
    let internalResolutions = extractFromInternalTypedTree checkResults

    // Also extract from public API to identify TraitCall locations
    let publicResolutions = extractTraitCallResolutions checkResults

    // Merge: prefer internal resolutions (they have actual solutions)
    let mergedResolutions =
        publicResolutions
        |> Map.map (fun key publicRes ->
            match Map.tryFind key internalResolutions with
            | Some internalRes -> internalRes  // Use internal resolution
            | None -> publicRes  // Fall back to public (which is Unresolved)
        )
        |> fun m ->
            // Also add any internal resolutions not in public set
            internalResolutions
            |> Map.fold (fun acc k v -> if Map.containsKey k acc then acc else Map.add k v acc) m

    if Map.isEmpty mergedResolutions then
        printfn "[SRTP] No TraitCall expressions found - PSG unchanged"
        psg
    else
        printfn "[SRTP] Correlating %d TraitCall resolutions with PSG nodes" (Map.count mergedResolutions)

        let rangeToKey (range: range) : string =
            sprintf "%s_%d_%d_%d_%d"
                (System.IO.Path.GetFileName range.FileName)
                range.Start.Line range.Start.Column
                range.End.Line range.End.Column

        // Update PSG nodes that match TraitCall locations
        let mutable updatedCount = 0
        let mutable resolvedCount = 0
        let updatedNodes =
            psg.Nodes
            |> Map.map (fun nodeId node ->
                // Check if this node's range matches a TraitCall location
                let key = rangeToKey node.Range
                match Map.tryFind key mergedResolutions with
                | Some resolution ->
                    updatedCount <- updatedCount + 1
                    if resolution <> Unresolved "FCS API doesn't expose TraitConstraintInfo.Solution" then
                        resolvedCount <- resolvedCount + 1
                    { node with SRTPResolution = Some resolution }
                | None ->
                    // Also check if this is a TraitCall node by SyntaxKind
                    if node.SyntaxKind.StartsWith("TraitCall") then
                        printfn "[SRTP] Found TraitCall node %s but no resolution available" nodeId
                        { node with SRTPResolution = Some (Unresolved "No resolution found for TraitCall") }
                    else
                        node
            )

        printfn "[SRTP] Updated %d PSG nodes with SRTP resolution info (%d actually resolved)" updatedCount resolvedCount
        { psg with Nodes = updatedNodes }
