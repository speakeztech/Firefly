# PSG Phase 2 Notes

## Major Improvements 🎉

### 1. **Correlation is Working**
- Jumped from 0% to 5.99% correlation rate
- 283 symbols successfully correlated (up from 0)
- The `psg.corr.json` shows real correlations happening with symbols like `ValueNone`, `Some`, `printf`, etc.

### 2. **Graph Structure**
- **Nodes**: 1134 (up from 459) - Much better AST traversal
- **Edges**: 1717 (up from 89) - Real reference edges being created
- **Entry Point**: Successfully found `main` function
- **Symbol Table**: 358 symbols (up from 46)

### 3. **File Coverage Patterns**
```
01_HelloWorldDirect.fs: 25.0%    ← Best coverage
Console.fs:             15.2%    
Core.fs:                13.1%    
Math.fs:                 4.9%    
Memory.fs:               1.0%    
Text.fs:                 0.5%    ← Worst coverage
```

This gradient suggests correlation works better for:
- Simpler files with direct code
- Files with less generic/abstract code

## Remaining Challenges

### 1. **Low Overall Correlation Rate (5.99%)**
The main issue is likely the **range matching precision**. FCS provides very specific ranges for symbols, but the AST nodes might have slightly different ranges. For example:
- A function name might have range `(10,4)-(10,8)`  
- But the binding node might have range `(10,0)-(12,15)`

### 2. **Library Code Correlation**
Library files (Math.fs, Memory.fs, Text.fs) have very low correlation rates. This could be due to:
- More complex generic code
- Inline functions
- Compiler-generated symbols

### 3. **Missing AssemblyContents**
The typed AST files are still 0 bytes, suggesting `AssemblyContents` is empty. This limits our ability to cross-reference.

## Recommendations for Next Phase

### 1. **Improve Range Matching**
Add "fuzzy" range matching that looks for symbols within a node's range:
```fsharp
let tryCorrelateWithinRange (nodeRange: range) (fileName: string) (context: CorrelationContext) =
    context.SymbolUses
    |> Array.filter (fun su -> 
        su.Range.FileName = fileName &&
        nodeRange.Start.Line <= su.Range.Start.Line &&
        nodeRange.End.Line >= su.Range.End.Line
    )
    |> Array.tryHead
    |> Option.map (fun su -> su.Symbol)
```

### 2. **Add Pattern-Specific Correlation**
For patterns like function names, try correlating by name + approximate location:
```fsharp
| SynPat.Named(synIdent, _, _, _) ->
    let (SynIdent(ident, _)) = synIdent
    // Try to find symbol with matching name near this location
    tryFindSymbolByNameAndLocation ident.idText range context
```

### 3. **Debug Output Enhancement**
Add a debug file showing uncorrelated nodes to understand what's being missed:
```fsharp
let uncorrelatedNodes = 
    graph.Nodes 
    |> Map.filter (fun _ node -> node.Symbol.IsNone)
    |> Map.map (fun _ node -> 
        {| Kind = node.SyntaxKind
           Range = node.Range
           File = Path.GetFileName(node.SourceFile) |})
```

### 4. **Reachability Preview**
With 1134 nodes and 1717 edges, you're ready to start basic reachability:
- Start from the entry point
- Follow edges to find reachable nodes
- Generate `.pruned.ast` showing only reachable code

## Summary

You've made substantial progress:
- ✅ PSG structure is solid
- ✅ Edge creation is working
- ✅ Entry point detection works
- ✅ Basic correlation is functional

The 5.99% correlation rate, while low, is enough to proceed with reachability analysis. The uncorrelated nodes won't block reachability - they just won't have symbol information attached.

Ready to tackle reachability analysis next? The PSG foundation is now strong enough to support it!