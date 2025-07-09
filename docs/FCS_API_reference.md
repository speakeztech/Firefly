# F# Compiler Services API Reference

## Key FCS Types and Their Properties

### FSharpSymbol (abstract base class)
**Common Properties**:
- `DisplayName: string` - Simple name
- `FullName: string` - Fully qualified name
- `IsPublic: bool`
- `IsPrivate: bool`
- `IsInternal: bool`
- `DeclarationLocation: range` - Where it's declared

**Properties that vary by type**:
- `XmlDoc` - Not all symbol types have this
- `Attributes` - Only some types have attributes

**NO DeclaringEntity on base class!**

### FSharpEntity (types, modules, namespaces)
**Inherits**: FSharpSymbol

**Additional Properties**:
- `IsFSharpModule: bool`
- `IsNamespace: bool`
- `IsClass: bool`
- `IsInterface: bool`
- `IsFSharpRecord: bool`
- `IsFSharpUnion: bool`
- `IsArrayType: bool`
- `DeclaringEntity: FSharpEntity option` - Parent module/namespace
- `NestedEntities: FSharpEntity list`
- `MembersFunctionsAndValues: FSharpMemberOrFunctionOrValue list`
- `UnionCases: FSharpUnionCase list` (for DUs)
- `FSharpFields: FSharpField list` (for records)

### FSharpMemberOrFunctionOrValue (functions, methods, values)
**Inherits**: FSharpSymbol

**Additional Properties**:
- `IsFunction: bool`
- `IsValue: bool`
- `IsMember: bool`
- `IsProperty: bool`
- `IsModuleValueOrMember: bool`
- `DeclaringEntity: FSharpEntity option` - Containing type/module
- `CurriedParameterGroups: FSharpParameter list list`
- `ReturnParameter: FSharpParameter`
- `GenericParameters: FSharpGenericParameter list`
- `Attributes: FSharpAttribute list`

### FSharpField (record/class fields)
**Inherits**: FSharpSymbol

**Additional Properties**:
- `IsMutable: bool`
- `IsStatic: bool`
- `FieldType: FSharpType`
- `DeclaringEntity: FSharpEntity option` - Containing type

### FSharpUnionCase
**Inherits**: FSharpSymbol

**Additional Properties**:
- `UnionCaseFields: FSharpField list`
- `DeclaringEntity: FSharpEntity option` - Parent union type

### FSharpSymbolUse
**Properties**:
- `Symbol: FSharpSymbol` - The symbol being used
- `IsFromDefinition: bool` - True if this is the definition
- `IsFromUse: bool` - True if this is a use
- `Range: range` - Location of use

## Common Patterns

### Getting declaring entity for any symbol:
```fsharp
let getDeclaringEntity (symbol: FSharpSymbol) =
    match symbol with
    | :? FSharpMemberOrFunctionOrValue as mfv -> mfv.DeclaringEntity
    | :? FSharpField as field -> field.DeclaringEntity
    | :? FSharpUnionCase as uc -> uc.DeclaringEntity
    | :? FSharpEntity as entity -> entity.DeclaringEntity
    | _ -> None
```

### Checking symbol type:
```fsharp
match symbol with
| :? FSharpEntity as entity -> 
    // Handle type/module/namespace
| :? FSharpMemberOrFunctionOrValue as mfv ->
    // Handle function/method/value
| :? FSharpField as field ->
    // Handle field
| :? FSharpUnionCase as unionCase ->
    // Handle DU case
| _ ->
    // Other symbol types
```

## Important Notes

1. **Always pattern match** on specific symbol types to access type-specific properties
2. **DeclaringEntity** is NOT on the base FSharpSymbol class
3. **Use type tests** (`:?`) to check symbol types before casting
4. **Handle None cases** - DeclaringEntity can be None for top-level definitions