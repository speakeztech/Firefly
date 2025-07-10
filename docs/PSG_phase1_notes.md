# Phase 1 PSG Key Problems Identified

1. **Zero Correlation Success**
   - `psg.corr.json` is empty `[]`
   - Correlation rate is 0% with 0 correlated nodes out of 4726 symbols
   - All files show 0% coverage

2. **Self-Referencing Edges**
   - All edges point to themselves (e.g., `sym_message_652804a8` → `sym_message_652804a8`)
   - This suggests the edge creation logic is broken

3. **Empty Typed AST**
   - All typed AST files are 0 bytes
   - This means `checkResults.AssemblyContents.ImplementationFiles` is empty

4. **Limited Symbol Table**
   - Only 46 symbols captured out of 4726 total
   - Many nodes have `null` symbols

## Root Causes

1. **Correlation Traversal Not Happening**
   - The `correlateFile` function expects ParsedInput but might not be getting called correctly
   - The traversal visitor pattern isn't walking the AST properly

2. **AssemblyContents Issue**
   - FCS might not be populating AssemblyContents without proper compilation
   - Need to ensure `keepAssemblyContents = true` is working

3. **Range Matching Failure**
   - The position index might not be matching due to subtle range differences

## Remediation Plan

### 1. Fix Typed AST Generation
```fsharp
// In Builder.fs, add debugging to see what's available
let buildProgramSemanticGraph checkResults parseResults =
    printfn "[DEBUG] AssemblyContents has %d files" 
        checkResults.AssemblyContents.ImplementationFiles.Length
```

### 2. Fix Correlation Traversal
The main issue is that `correlateModule` isn't properly traversing the AST. Need to implement proper SynModuleDecl traversal:

```fsharp
// In Correlation.fs, fix traverseWithCorrelation to handle SynModuleDecl
| :? SynModuleDecl as decl ->
    match decl with
    | SynModuleDecl.Let(_, bindings, _) ->
        bindings |> List.iter (fun b -> traverseWithCorrelation b fileName context visitor)
    | SynModuleDecl.Types(typeDefs, _) ->
        // Process type definitions
    | _ -> ()
```

### 3. Fix Edge Creation
The self-referencing edges suggest the identifier resolution is broken. In `processExpression`:

```fsharp
| SynExpr.Ident ident ->
    // Should look up the target symbol, not use the same symbol
    match tryFindDefinitionForIdent ident context with
    | Some targetSymbol ->
        let edge = {
            Source = exprNode.Id
            Target = NodeId.FromSymbol(targetSymbol)
            Kind = EdgeKind.References
        }
```

### 4. Add Diagnostic Output
Add more debugging to understand what's happening:

```fsharp
// In createContext
printfn "[DEBUG] Found %d symbol uses across %d files" 
    allSymbolUses.Length 
    (allSymbolUses |> Array.map (fun su -> su.Range.FileName) |> Array.distinct |> Array.length)
```

### Next Steps

1. First, let's verify FCS is returning data by adding debug output
2. Fix the AST traversal to properly visit all syntax nodes
3. Implement proper symbol resolution for references
4. Add range tolerance for correlation matching

Would you like me to implement these fixes starting with the diagnostic additions to understand what data FCS is providing?