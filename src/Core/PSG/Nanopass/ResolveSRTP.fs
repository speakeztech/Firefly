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
                let allTypes = fcsAssembly.GetTypes()

                // Find TraitConstraintInfo from FSharp.Compiler.TypedTree
                traitConstraintInfoType <-
                    allTypes
                    |> Array.tryFind (fun t -> t.FullName = "FSharp.Compiler.TypedTree+TraitConstraintInfo")

                match traitConstraintInfoType with
                | Some t ->
                    // Get the Solution property - try both public and non-public
                    solutionProperty <-
                        t.GetProperty("Solution", BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
                        |> Option.ofObj
                | None -> ()

                // Solution types are nested in FSharp.Compiler.TypedTree+TraitConstraintSln
                fsMethSlnType <- allTypes |> Array.tryFind (fun t -> t.FullName = "FSharp.Compiler.TypedTree+TraitConstraintSln+FSMethSln")
                fsRecdFieldSlnType <- allTypes |> Array.tryFind (fun t -> t.FullName = "FSharp.Compiler.TypedTree+TraitConstraintSln+FSRecdFieldSln")
                fsAnonRecdFieldSlnType <- allTypes |> Array.tryFind (fun t -> t.FullName = "FSharp.Compiler.TypedTree+TraitConstraintSln+FSAnonRecdFieldSln")
                builtInSlnType <- allTypes |> Array.tryFind (fun t -> t.FullName = "FSharp.Compiler.TypedTree+TraitConstraintSln+BuiltInSln")

                initialized <- true
            with _ ->
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

                                                Some {
                                                    TargetMethodFullName = targetFullName
                                                    ParameterTypeNames = paramTypes
                                                    ReturnTypeName = returnType
                                                }
                                            with _ -> None)

                                    if candidates.Length > 0 then
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
                match fullName |> Option.orElse methodName with
                | Some name -> FSMethodByName name
                | None -> BuiltIn
            else
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
                None
            else
                let solutionOpt = solutionProp.GetValue(traitInfo)
                match getFSharpOptionValue solutionOpt with
                | Some _ ->
                    // We need a range to correlate with PSG nodes
                    // TraitConstraintInfo doesn't have a range, but the enclosing Expr does
                    None  // Will need to get range from enclosing Expr
                | None ->
                    // Solution ref cell might contain the value differently
                    // Try accessing the underlying solution ref field
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
                            | Some _ -> None  // Still need range
                            | None -> None
                        | None -> None
                    else
                        None
        with _ ->
            None

/// Walk an internal Expr tree to find TraitCall operations
let rec private walkInternalExpr (expr: obj) (resolutions: byref<Map<string, SRTPResolution>>) : unit =
    if isNull expr then ()
    else
        try
            let exprType = expr.GetType()
            let typeName = exprType.Name

            // Check if this is an Op expression with TraitCall
            if typeName = "Op" || exprType.FullName.EndsWith("+Op") then
                // Expr.Op has: op, typeArgs, args, range
                let opProp = exprType.GetProperty("op", BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
                            |> Option.ofObj
                            |> Option.orElse (exprType.GetProperty("Item1") |> Option.ofObj)

                match opProp with
                | Some prop ->
                    let op = prop.GetValue(expr)
                    if not (isNull op) then
                        let opType = op.GetType()
                        if opType.Name.Contains("TraitCall") then
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
                                            let resolution = convertSolution sln
                                            let key = sprintf "%s_%d_%d_%d_%d"
                                                        (System.IO.Path.GetFileName range.FileName)
                                                        range.Start.Line range.Start.Column
                                                        range.End.Line range.End.Column

                                            resolutions <- Map.add key resolution resolutions
                                        | None -> ()
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
            let exprProp = bindType.GetProperty("Expr", BindingFlags.Public ||| BindingFlags.Instance)
                           |> Option.ofObj
                           |> Option.orElse (bindType.GetProperty("expr", BindingFlags.NonPublic ||| BindingFlags.Instance) |> Option.ofObj)

            match exprProp with
            | Some prop ->
                let expr = prop.GetValue(binding)
                if not (isNull expr) then
                    walkInternalExpr expr &resolutions
            | None -> ()
        with _ -> ()

/// Walk ModuleOrNamespaceContents to find all expressions
let rec private walkModuleContents (contents: obj) (resolutions: byref<Map<string, SRTPResolution>>) : unit =
    if isNull contents then ()
    else
        try
            let contentsType = contents.GetType()
            let typeName = contentsType.Name

            // TMDefLet: binding
            if typeName.Contains("TMDefLet") then
                let bindingProp = contentsType.GetProperty("binding", BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
                                  |> Option.ofObj
                                  |> Option.orElse (contentsType.GetProperty("Item1") |> Option.ofObj)
                match bindingProp with
                | Some prop ->
                    let binding = prop.GetValue(contents)
                    if not (isNull binding) then
                        walkBinding binding &resolutions
                | None -> ()

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
                let bindingsProp = contentsType.GetProperty("bindings", BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
                if bindingsProp <> null then
                    let bindings = bindingsProp.GetValue(contents)
                    let bindingList = getFSharpListItems bindings
                    for modBinding in bindingList do
                        let mbType = modBinding.GetType()
                        if mbType.Name.Contains("Binding") then
                            let innerBindProp = mbType.GetProperty("Item", BindingFlags.Public ||| BindingFlags.Instance)
                                                |> Option.ofObj
                                                |> Option.orElse (mbType.GetProperty("Item1") |> Option.ofObj)
                            match innerBindProp with
                            | Some p -> walkBinding (p.GetValue(modBinding)) &resolutions
                            | None -> ()
                        elif mbType.Name.Contains("Module") then
                            let subContentsProp = mbType.GetProperty("moduleOrNamespaceContents", BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
                                                  |> Option.ofObj
                                                  |> Option.orElse (mbType.GetProperty("Item2", BindingFlags.Public ||| BindingFlags.Instance) |> Option.ofObj)
                            match subContentsProp with
                            | Some prop ->
                                let subContents = prop.GetValue(modBinding)
                                if not (isNull subContents) then
                                    walkModuleContents subContents &resolutions
                            | None -> ()

            // TMDefs: list of contents
            elif typeName.Contains("TMDefs") then
                let defsProp = contentsType.GetProperty("defs", BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance)
                               |> Option.ofObj
                               |> Option.orElse (contentsType.GetProperty("Item") |> Option.ofObj)
                               |> Option.orElse (contentsType.GetProperty("Item1") |> Option.ofObj)
                match defsProp with
                | Some prop ->
                    let defs = prop.GetValue(contents)
                    let defList = getFSharpListItems defs
                    for def in defList do
                        walkModuleContents def &resolutions
                | None -> ()

            // TMDefOpens: no expressions
            elif typeName.Contains("TMDefOpens") then
                ()

        with _ -> ()

/// Walk a CheckedImplFile to find all TraitCall expressions with solutions
let private walkCheckedImplFile (implFile: obj) (resolutions: byref<Map<string, SRTPResolution>>) : unit =
    if isNull implFile then ()
    else
        try
            let ifType = implFile.GetType()

            // CheckedImplFile has Contents property
            let contentsProp = ifType.GetProperty("Contents", BindingFlags.Public ||| BindingFlags.Instance)
                               |> Option.ofObj
                               |> Option.orElse (ifType.GetProperty("contents", BindingFlags.NonPublic ||| BindingFlags.Instance) |> Option.ofObj)

            match contentsProp with
            | Some prop ->
                let contents = prop.GetValue(implFile)
                if not (isNull contents) then
                    walkModuleContents contents &resolutions
            | None -> ()
        with _ -> ()

/// Access internal CheckedImplFile list and extract SRTP resolutions
let private extractFromInternalTypedTree (checkResults: FSharpCheckProjectResults) : Map<string, SRTPResolution> =
    let mutable resolutions = Map.empty

    try
        let assemblyContents = checkResults.AssemblyContents
        let acType = assemblyContents.GetType()

        let mimplsField = acType.GetField("mimpls", BindingFlags.NonPublic ||| BindingFlags.Instance)
        if mimplsField <> null then
            let mimplsValue = mimplsField.GetValue(assemblyContents)
            if not (isNull mimplsValue) then
                let implFiles = getFSharpListItems mimplsValue
                for implFile in implFiles do
                    walkCheckedImplFile implFile &resolutions
    with _ -> ()

    resolutions

// ═══════════════════════════════════════════════════════════════════════════
// Main Nanopass Entry Point
// ═══════════════════════════════════════════════════════════════════════════

/// Run the SRTP resolution nanopass
/// This attempts to extract SRTP resolutions from FCS internals and populate
/// the SRTPResolution field on corresponding PSG nodes
let run (psg: ProgramSemanticGraph) (checkResults: FSharpCheckProjectResults) : ProgramSemanticGraph =
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
            | Some internalRes -> internalRes
            | None -> publicRes
        )
        |> fun m ->
            internalResolutions
            |> Map.fold (fun acc k v -> if Map.containsKey k acc then acc else Map.add k v acc) m

    if Map.isEmpty mergedResolutions then
        psg
    else
        let rangeToKey (range: range) : string =
            sprintf "%s_%d_%d_%d_%d"
                (System.IO.Path.GetFileName range.FileName)
                range.Start.Line range.Start.Column
                range.End.Line range.End.Column

        // Update PSG nodes that match TraitCall locations
        let updatedNodes =
            psg.Nodes
            |> Map.map (fun nodeId node ->
                let key = rangeToKey node.Range
                match Map.tryFind key mergedResolutions with
                | Some resolution ->
                    { node with SRTPResolution = Some resolution }
                | None ->
                    if node.SyntaxKind.StartsWith("TraitCall") then
                        { node with SRTPResolution = Some (Unresolved "No resolution found for TraitCall") }
                    else
                        node
            )

        { psg with Nodes = updatedNodes }
