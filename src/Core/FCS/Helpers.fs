module Core.FCS.Helpers

open FSharp.Compiler.Symbols

/// Get the declaring entity for any symbol type
let getDeclaringEntity (symbol: FSharpSymbol) : FSharpEntity option =
    match symbol with
    | :? FSharpMemberOrFunctionOrValue as mfv -> mfv.DeclaringEntity
    | :? FSharpField as field -> field.DeclaringEntity
    | :? FSharpUnionCase as unionCase -> Some unionCase.DeclaringEntity
    | :? FSharpEntity as entity -> entity.DeclaringEntity
    | :? FSharpGenericParameter -> None
    | :? FSharpActivePatternCase -> None
    | :? FSharpParameter -> None
    | _ -> None

/// Get the full name with declaring context
let getFullNameWithContext (symbol: FSharpSymbol) : string =
    let declContext = 
        match getDeclaringEntity symbol with
        | Some entity -> entity.FullName + "."
        | None -> ""
    declContext + symbol.DisplayName

/// Check if a symbol belongs to a specific module or namespace
let symbolBelongsTo (moduleOrNamespace: FSharpEntity) (symbol: FSharpSymbol) : bool =
    match getDeclaringEntity symbol with
    | Some entity -> entity = moduleOrNamespace
    | None -> false

/// Get all members of an entity
let getEntityMembers (entity: FSharpEntity) : FSharpSymbol list =
    let members = entity.MembersFunctionsAndValues |> Seq.toList |> List.map (fun m -> m :> FSharpSymbol)
    let fields = entity.FSharpFields |> Seq.toList |> List.map (fun f -> f :> FSharpSymbol)
    let unionCases = 
        if entity.IsFSharpUnion then 
            entity.UnionCases |> Seq.toList |> List.map (fun uc -> uc :> FSharpSymbol)
        else []
    let nested = entity.NestedEntities |> Seq.toList |> List.map (fun e -> e :> FSharpSymbol)
    
    members @ fields @ unionCases @ nested

/// Get the accessibility of a symbol
let getAccessibility (symbol: FSharpSymbol) : string =
    match symbol with
    | :? FSharpEntity as entity ->
        if entity.Accessibility.IsPublic then "public"
        elif entity.Accessibility.IsPrivate then "private"
        elif entity.Accessibility.IsInternal then "internal"
        else "unknown"
    | :? FSharpMemberOrFunctionOrValue as mfv ->
        if mfv.Accessibility.IsPublic then "public"
        elif mfv.Accessibility.IsPrivate then "private"
        elif mfv.Accessibility.IsInternal then "internal"
        else "unknown"
    | :? FSharpField as field ->
        if field.Accessibility.IsPublic then "public"
        elif field.Accessibility.IsPrivate then "private"
        elif field.Accessibility.IsInternal then "internal"
        else "unknown"
    | :? FSharpUnionCase as unionCase ->
        if unionCase.Accessibility.IsPublic then "public"
        elif unionCase.Accessibility.IsPrivate then "private"
        elif unionCase.Accessibility.IsInternal then "internal"
        else "unknown"
    | _ -> "unknown"

/// Check if symbol is a function
let isFunction (symbol: FSharpSymbol) : bool =
    match symbol with
    | :? FSharpMemberOrFunctionOrValue as mfv -> mfv.IsFunction
    | _ -> false

/// Check if symbol is a type (class, record, union, etc.)
let isType (symbol: FSharpSymbol) : bool =
    match symbol with
    | :? FSharpEntity as entity -> 
        not entity.IsFSharpModule && not entity.IsNamespace
    | _ -> false

/// Check if symbol is a module
let isModule (symbol: FSharpSymbol) : bool =
    match symbol with
    | :? FSharpEntity as entity -> entity.IsFSharpModule
    | _ -> false

/// Check if symbol is a namespace
let isNamespace (symbol: FSharpSymbol) : bool =
    match symbol with
    | :? FSharpEntity as entity -> entity.IsNamespace
    | _ -> false

/// Get symbol kind as string
let getSymbolKind (symbol: FSharpSymbol) : string =
    match symbol with
    | :? FSharpEntity as entity ->
        if entity.IsNamespace then "namespace"
        elif entity.IsFSharpModule then "module"
        elif entity.IsClass then "class"
        elif entity.IsInterface then "interface"
        elif entity.IsFSharpRecord then "record"
        elif entity.IsFSharpUnion then "union"
        elif entity.IsEnum then "enum"
        elif entity.IsDelegate then "delegate"
        elif entity.IsArrayType then "array"
        else "type"
    | :? FSharpMemberOrFunctionOrValue as mfv ->
        if mfv.IsProperty then "property"
        elif mfv.IsEvent then "event"
        elif mfv.IsFunction then "function"
        elif mfv.IsMember then "member"
        elif mfv.IsValue then "value"
        else "member"
    | :? FSharpField -> "field"
    | :? FSharpUnionCase -> "union case"
    | :? FSharpGenericParameter -> "generic parameter"
    | :? FSharpParameter -> "parameter"
    | :? FSharpActivePatternCase -> "active pattern"
    | _ -> "unknown"

/// Get attributes for a symbol
let getAttributes (symbol: FSharpSymbol) : FSharpAttribute list =
    match symbol with
    | :? FSharpEntity as entity -> entity.Attributes |> Seq.toList
    | :? FSharpMemberOrFunctionOrValue as mfv -> mfv.Attributes |> Seq.toList
    | :? FSharpField as field -> field.PropertyAttributes |> Seq.toList
    | _ -> []

/// Check if symbol has a specific attribute
let hasAttribute (attributeName: string) (symbol: FSharpSymbol) : bool =
    getAttributes symbol
    |> List.exists (fun attr -> 
        attr.AttributeType.DisplayName = attributeName ||
        attr.AttributeType.FullName.EndsWith("." + attributeName)
    )

/// Get XML documentation for a symbol
let getXmlDoc (symbol: FSharpSymbol) : FSharpXmlDoc option =
    match symbol with
    | :? FSharpEntity as entity -> Some entity.XmlDoc
    | :? FSharpMemberOrFunctionOrValue as mfv -> Some mfv.XmlDoc
    | :? FSharpField as field -> Some field.XmlDoc
    | :? FSharpUnionCase as uc -> Some uc.XmlDoc
    | _ -> None