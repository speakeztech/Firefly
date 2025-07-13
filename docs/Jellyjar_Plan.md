# JellyJar Plan

JellyJar is a module within Firefly that uses DuckDB PGQ to capture information about the Program Semantic Graph. The goal is both to use the output of those queries as general diagnostic output at compile-time, but eventually it will also be part of a larger LSP plan for Firefly that creates a more dynamic relationship between the PSG and the developer. For now the design is to make it strictly a means to gain stats on the PSG as it's being built. Both the initial and pruned PSG should get some attention in this early version of the JellyJar module. But for now it should remain a compile-time tool the same way Firefly is being kept relatively simple/straightforward for the purposes of the "hello world" POCs.

The metaphor for the name deserves some mention as the point is to be non-destructive of the data. The metaphor is like being a child in summertime capturing a bunch of fireflies at dusk during the summertime and putting them in a small jar where they eventually synchronize their tails lighting up, and then you let them go. The DuckDB PGQ is the jellyjar that does the analysis and performs the prismatic analysis for all of the different confirmations of various attributes that have been added to the consolidated PSG. Right now it's constrained to general over-arching data to confirm the validity of creation and pruning of the PSG. But eventually it will become an interactive tool that will let developers confirm the trace of their applications back through Alloy and other libraries that support the solutions they're building.

The eventual goal is to use DuckDB PGQ as a new form of powerful design-time analysis tool. Even though the initial effort will be more of a post-compilation "batch mode" query-and-post operation to the console, all architectural decisions should be made with some eye toward that dynamic future.

## Phase 1: Conservative (JSON-based)

```fsharp
// Core writes artifacts
writeDebugOutputs psg "./build/intermediates"

// JellyJar reads back from disk
JellyJar.Analytics.fromIntermediates "./build/intermediates"
|> JellyJar.Reports.displaySummaryTables
```

## Phase 2: Elegant In-Memory Integration

### Streaming PSG Construction

```fsharp
// Core builds PSG while streaming to JellyJar
let jellyJar = JellyJar.Database.create()

let psg = 
    buildProgramSemanticGraph checkResults parseResults
    |> JellyJar.PSG.streamNodes jellyJar      // Nodes flow directly to DuckDB
    |> JellyJar.PSG.streamEdges jellyJar      // Edges flow directly  
    |> JellyJar.PSG.streamSymbols jellyJar    // Symbol table flows directly
```

### Live Analytics During Compilation

```fsharp
// Real-time compilation metrics
type CompilationObserver = {
    OnNodeAdded: PSGNode -> unit
    OnEdgeCreated: PSGEdge -> unit  
    OnSymbolCorrelated: FSharpSymbol -> unit
    OnReachabilityMarked: NodeId * bool -> unit
}

let observer = JellyJar.LiveAnalytics.create()
let psg = buildProgramSemanticGraphWithObserver observer checkResults parseResults

// JellyJar can provide real-time feedback:
// "Symbol correlation: 94/100 (94%)"
// "Dead code detected: 847 nodes eliminated"
```

### Bidirectional Pipeline

```fsharp
// JellyJar analysis can influence Core decisions
let reachabilityHints = JellyJar.Analytics.suggestOptimizations jellyJar
let optimizedPsg = Core.PSG.applyHints reachabilityHints psg

// Or even:
let adaptiveElimination = JellyJar.Adaptive.createEliminationStrategy()
let psg = buildProgramSemanticGraphWithStrategy adaptiveElimination
```

### Unified Memory Model

```fsharp
// PSG becomes a view over DuckDB tables
type DuckDBBackedPSG = {
    Database: DuckDBConnection
    NodesView: string      // "psg_nodes" table
    EdgesView: string      // "psg_edges" table  
    SymbolsView: string    // "psg_symbols" table
}

// Query PSG using PGQ directly
let eliminationCandidates = 
    jellyJar.Query("""
        SELECT node_id FROM psg_nodes 
        WHERE reachability_distance > 5
        AND symbol_kind = 'MemberOrFunctionOrValue'
    """)
```

## Benefits of Deep Integration

1. **Zero File I/O**: PSG data never touches disk during analysis
2. **Real-time Feedback**: Developers see elimination rates as they happen
3. **Adaptive Compilation**: JellyJar insights influence Core decisions
4. **Memory Efficiency**: Single unified graph representation
5. **Live Debugging**: Query compilation state interactively
6. **Performance**: DuckDB's columnar engine on live compilation data

## **The Vision**

Eventually, **JellyJar becomes the PSG storage engine itself**. Core doesn't maintain an in-memory graph - it streams construction directly to DuckDB, and Dabbit queries DuckDB for MLIR generation.

The fireflies are captured **as they're created**, synchronized in real-time, and released as optimized MLIR!

## Future State: Firefly as Language Server/TSR

### Current: Single-Shot CLI

```bash
firefly compile ./project.fsproj  # Build PSG → Analyze → Generate MLIR → Exit
```

### Future: Persistent Language Server

```text
VSCode Extensions:
├── Ionide (F# language server)          # F# syntax, type checking
├── MLIR Language Server                 # MLIR dialect support  
├── LLVM Language Server                 # LLVM IR analysis
└── Firefly Extension                    # Native compilation pipeline
    └── Firefly.exe (TSR Process)       # Persistent PSG + JellyJar
```

## **Persistent JellyJar Implications**

### Incremental PSG Updates

```fsharp
// File changes trigger incremental updates
type FireflyLanguageServer = {
    PSG: DuckDBBackedPSG                    // Persistent graph
    JellyJar: JellyJar.Database              // Live analytics engine
    Watchers: FileSystemWatcher[]           // F# file monitors
}

// On file change:
member this.OnFileChanged(fileName: string) =
    let updatedNodes = this.reprocessFile fileName
    this.JellyJar.updateNodes updatedNodes  // Incremental DuckDB update
    this.broadcastPSGChanges()              // Notify VSCode
```

### Live Developer Queries

```json
// VSCode Command Palette
"Firefly: Show Reachability for Current Function"
"Firefly: Why Was This Function Eliminated?"
"Firefly: Show Call Graph from Here"
"Firefly: Analyze Memory Layout"
"Firefly: Preview MLIR Generation"
```

### Real-Time Compilation Feedback

```typescript
// VSCode Extension API
firefly.onReachabilityChanged((results) => {
    // Update editor decorations
    editor.showEliminatedCode(results.eliminatedNodes);
    editor.showCallPaths(results.reachablePaths);
});
```

## **Architecture Benefits**

1. **Warm PSG**: No cold startup, instant analysis
2. **Incremental Everything**: Only reprocess changed files
3. **Live Metrics**: Real-time elimination rates in status bar
4. **Interactive Debugging**: Query PSG from VSCode directly  
5. **Multi-Project**: Manage multiple PSGs simultaneously
6. **Cross-Language**: F# → MLIR → LLVM seamless workflow

## **JellyJar as Persistent Analytics Engine**

Instead of ephemeral analysis, JellyJar becomes the **live brain** of Firefly:

- **Persistent DuckDB**: Maintains compilation history, performance metrics
- **Change Detection**: Tracks which modifications affect reachability
- **Predictive Analysis**: "This change will eliminate 23 functions"
- **Performance Profiling**: "Compilation bottleneck in Math.fs correlation"

This positions Firefly as a **true native development environment** - not just a compiler, but an intelligent compilation assistant that understands your codebase evolution in real-time.

The TSR architecture makes the in-memory DuckDB integration not just elegant, but **essential** for responsive developer experience!
