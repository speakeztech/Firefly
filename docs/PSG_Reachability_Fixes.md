# Current Reachability Analysis for PSG

The reachability analysis is completely violating your design principle.

## **Current Violations:**

**`analyzeReachabilityWithBoundaries`** takes BOTH `psg: ProgramSemanticGraph` AND `symbolUses: FSharpSymbolUse[]` - **WRONG**

**It then processes raw FCS data:**
```fsharp
let definitions = symbolUses |> Array.filter (fun su -> su.IsFromDefinition)
let uses = symbolUses |> Array.filter (fun su -> su.IsFromUse)
// File distribution analysis directly on symbolUses
// findEntryPoints directly on raw FSharpSymbolUse[]
```

**`findEntryPoints`** operates on `FSharpSymbolUse[]` instead of using `PSG.EntryPoints` 

**Pipeline calls it wrong:**
```fsharp
let reachabilityResult = analyzeReachabilityWithBoundaries projectResults.SymbolUses
```

## **What Should Happen:**

**Reachability should ONLY take PSG:**
```fsharp
let analyzeReachability (psg: ProgramSemanticGraph) : LibraryAwareReachability
```

**Everything should come from PSG:**
- Entry points: `psg.EntryPoints` (already computed)
- Relationships: `psg.Edges` (already built) 
- Symbols: `psg.SymbolTable` or `psg.Nodes`
- **NO** raw `FSharpSymbolUse[]` arrays

**The PSG IS the analyzed project data.** Reachability should traverse the graph, not re-analyze the original FCS data.

Reachability analysis is doing **redundant work** and **bypassing your PSG entirely**. This violates the fundamental principle that PSG is the single source of truth for all downstream analysis.
