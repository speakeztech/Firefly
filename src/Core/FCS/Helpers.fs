module Core.FCS.Helpers

open FSharp.Compiler.Symbols

/// Get the declaring entity for any symbol type
let getDeclaringEntity (symbol: FSharpSymbol) : FSharpEntity option =
    match symbol with
    | :? FSharpMemberOrFunctionOrValue as mfv -> mfv.DeclaringEntity
    | :? FSharpField as field -> field.DeclaringEntity
    | :? FSharpUnionCase as unionCase -> unionCase.DeclaringEntity
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
    let members = entity.MembersFunctionsAndValues |> List.map (fun m -> m :> FSharpSymbol)
    let fields = entity.FSharpFields |> List.map (fun f -> f :> FSharpSymbol)
    let unionCases = 
        if entity.IsFSharpUnion then 
            entity.UnionCases |> List.map (fun uc -> uc :> FSharpSymbol)
        else []
    let nested = entity.NestedEntities |> List.map (fun e -> e :> FSharpSymbol)
    
    members @ fields @ unionCases @ nested

/// Get the accessibility of a symbol
let getAccessibility (symbol: FSharpSymbol) : string =
    if symbol.IsPublic then "public"
    elif symbol.IsPrivate then "private"
    elif symbol.IsInternal then "internal"
    else "unknown"

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
    | :? FSharpEntity as entity -> entity.Attributes |> List.ofSeq
    | :? FSharpMemberOrFunctionOrValue as mfv -> mfv.Attributes |> List.ofSeq
    | :? FSharpField as field -> field.PropertyAttributes |> List.ofSeq
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