# DuckDB PSG Audit Tool Plan

## Overview

This plan outlines a pragmatic approach to using DuckDB's Property Graph Query (PGQ) extension as an external audit tool for PSG analysis, with a path toward eventual integration into Firefly for real-time compilation statistics.

## Phase 1: External DuckDB Audit Tool

### 1.1 Loading Current JSON Outputs

Create a DuckDB script to ingest the existing PSG JSON files:

```sql
-- Create schema for PSG audit database
CREATE SCHEMA IF NOT EXISTS psg_audit;
USE psg_audit;

-- Import JSON intermediate files
CREATE TABLE raw_nodes AS 
SELECT * FROM read_json('intermediates/*.nodes.json', 
    columns = {
        id: 'VARCHAR',
        kind: 'VARCHAR', 
        symbol: 'VARCHAR',
        range: 'STRUCT(startLine INT, startColumn INT, endLine INT, endColumn INT)',
        sourceFile: 'VARCHAR',
        parentId: 'VARCHAR',
        children: 'VARCHAR[]'
    });

CREATE TABLE raw_edges AS
SELECT * FROM read_json('intermediates/*.edges.json',
    columns = {
        source: 'VARCHAR',
        target: 'VARCHAR',
        kind: 'VARCHAR'
    });

CREATE TABLE raw_symbols AS
SELECT * FROM read_json('intermediates/*.symbols.json',
    columns = {
        name: 'VARCHAR',
        kind: 'VARCHAR',
        hash: 'BIGINT',
        file: 'VARCHAR'
    });

-- Create property graph from raw data
CREATE PROPERTY GRAPH psg_graph
VERTEX TABLES (
    raw_nodes LABEL Node 
    PROPERTIES (id, kind, symbol, sourceFile) 
    PRIMARY KEY (id)
)
EDGE TABLES (
    raw_edges SOURCE KEY (source) REFERENCES raw_nodes (id)
              DESTINATION KEY (target) REFERENCES raw_nodes (id)
              LABEL Edge
              PROPERTIES (kind)
);
```

### 1.2 Graph Structure Analysis Queries

Key queries to identify current implementation issues:

```sql
-- Check for orphaned nodes (no parent, not root)
WITH root_modules AS (
    SELECT id FROM raw_nodes 
    WHERE kind = 'Module' AND parentId IS NULL
)
SELECT COUNT(*) as orphaned_count,
       kind,
       sourceFile
FROM raw_nodes n
WHERE n.parentId IS NULL 
  AND n.id NOT IN (SELECT id FROM root_modules)
GROUP BY kind, sourceFile;

-- Analyze self-referencing edges
SELECT COUNT(*) as self_ref_count,
       e.kind as edge_kind
FROM raw_edges e
WHERE e.source = e.target
GROUP BY e.kind;

-- Check parent-child consistency
WITH parent_child_issues AS (
    SELECT n.id as node_id,
           n.parentId as claimed_parent,
           CASE 
               WHEN p.id IS NULL THEN 'missing_parent'
               WHEN NOT ARRAY_CONTAINS(p.children, n.id) THEN 'not_in_parent_children'
               ELSE 'ok'
           END as issue
    FROM raw_nodes n
    LEFT JOIN raw_nodes p ON n.parentId = p.id
    WHERE n.parentId IS NOT NULL
)
SELECT issue, COUNT(*) as count
FROM parent_child_issues
WHERE issue != 'ok'
GROUP BY issue;

-- Symbol correlation success rate
SELECT 
    COUNT(CASE WHEN symbol IS NOT NULL THEN 1 END) as correlated,
    COUNT(*) as total,
    ROUND(100.0 * COUNT(CASE WHEN symbol IS NOT NULL THEN 1 END) / COUNT(*), 2) as correlation_rate
FROM raw_nodes
WHERE kind IN ('Binding', 'Application', 'Identifier');
```

### 1.3 Graph Traversal Analysis

Using DuckDB's PGQ capabilities to analyze reachability:

```sql
-- Find unreachable nodes from entry points
FROM GRAPH_TABLE (psg_graph
    MATCH 
        (entry:Node WHERE entry.symbol = 'main' OR entry.symbol = '[<EntryPoint>]')
        -[path:Edge*]->
        (reachable:Node)
    COLUMNS (reachable.id as reachable_id)
) 
SELECT COUNT(DISTINCT n.id) as unreachable_count
FROM raw_nodes n
WHERE n.id NOT IN (SELECT reachable_id FROM reachable_nodes);

-- Analyze connected components
WITH RECURSIVE components AS (
    -- Start with each node as its own component
    SELECT id, id as component_id, 1 as depth
    FROM raw_nodes
    
    UNION ALL
    
    -- Propagate component IDs through edges
    SELECT e.target, c.component_id, c.depth + 1
    FROM components c
    JOIN raw_edges e ON c.id = e.source
    WHERE c.depth < 100  -- Prevent infinite loops
)
SELECT 
    component_id,
    COUNT(DISTINCT id) as node_count,
    MAX(depth) as max_depth
FROM components
GROUP BY component_id
HAVING COUNT(DISTINCT id) > 1
ORDER BY node_count DESC;
```

### 1.4 Coupling and Cohesion Metrics

```sql
-- Module coupling analysis
WITH module_dependencies AS (
    SELECT DISTINCT
        n1.sourceFile as from_module,
        n2.sourceFile as to_module,
        COUNT(*) as dependency_count
    FROM raw_edges e
    JOIN raw_nodes n1 ON e.source = n1.id
    JOIN raw_nodes n2 ON e.target = n2.id
    WHERE n1.sourceFile != n2.sourceFile
    GROUP BY n1.sourceFile, n2.sourceFile
)
SELECT 
    from_module,
    COUNT(DISTINCT to_module) as dependencies,
    SUM(dependency_count) as total_refs,
    ROUND(AVG(dependency_count), 2) as avg_refs_per_dependency
FROM module_dependencies
GROUP BY from_module
ORDER BY dependencies DESC;

-- Function cohesion (calls within same module)
WITH function_calls AS (
    SELECT 
        n1.symbol as caller,
        n2.symbol as callee,
        n1.sourceFile as caller_file,
        n2.sourceFile as callee_file
    FROM raw_edges e
    JOIN raw_nodes n1 ON e.source = n1.id
    JOIN raw_nodes n2 ON e.target = n2.id
    WHERE e.kind = 'FunctionCall'
      AND n1.symbol IS NOT NULL
      AND n2.symbol IS NOT NULL
)
SELECT 
    caller_file,
    COUNT(CASE WHEN caller_file = callee_file THEN 1 END) as internal_calls,
    COUNT(CASE WHEN caller_file != callee_file THEN 1 END) as external_calls,
    ROUND(100.0 * COUNT(CASE WHEN caller_file = callee_file THEN 1 END) / COUNT(*), 2) as cohesion_percentage
FROM function_calls
GROUP BY caller_file;
```

### 1.5 Shared PSG Audit Module

```fsharp
// src/Core/Analysis/PSGAudit.fs
module Core.Analysis.PSGAudit

open System
open System.IO
open DuckDB.NET
open Core.PSG.Types

type AuditReport = {
    GraphStructure: StructureAnalysis
    CorrelationAnalysis: CorrelationStats
    ReachabilityAnalysis: ReachabilityStats
    CouplingCohesion: CouplingStats
}

and StructureAnalysis = {
    TotalNodes: int
    OrphanedNodes: int
    SelfReferencingEdges: int
    DisconnectedComponents: int
}

and CorrelationStats = {
    TotalSymbols: int
    CorrelatedSymbols: int
    Rate: float
}

and ReachabilityStats = {
    ReachableNodes: int
    UnreachableNodes: int
    DeadCodePercentage: float
}

and CouplingStats = {
    ModuleCoupling: ModuleCouplingInfo list
    AverageCoupling: float
}

/// Core analysis functions used by both standalone and integrated modes
module Analysis =
    let analyzeStructure (conn: DuckDBConnection) =
        let orphans = conn.ExecuteScalar<int>("""
            SELECT COUNT(*) FROM raw_nodes 
            WHERE parentId IS NULL AND kind != 'Module'
        """)
        
        let selfRefs = conn.ExecuteScalar<int>("""
            SELECT COUNT(*) FROM raw_edges WHERE source = target
        """)
        
        let components = conn.ExecuteScalar<int>("""
            WITH RECURSIVE component_assignment AS (
                SELECT id, id as component_id FROM raw_nodes
                UNION
                SELECT e.target, c.component_id
                FROM raw_edges e
                JOIN component_assignment c ON e.source = c.id
            )
            SELECT COUNT(DISTINCT component_id) FROM component_assignment
        """)
        
        {
            TotalNodes = conn.ExecuteScalar<int>("SELECT COUNT(*) FROM raw_nodes")
            OrphanedNodes = orphans
            SelfReferencingEdges = selfRefs
            DisconnectedComponents = components
        }

    let analyzeCorrelation (conn: DuckDBConnection) =
        let stats = conn.QuerySingle<{| total: int; correlated: int |}>("""
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN symbol IS NOT NULL THEN 1 END) as correlated
            FROM raw_nodes
            WHERE kind IN ('Binding', 'Application', 'Identifier')
        """)
        
        {
            TotalSymbols = stats.total
            CorrelatedSymbols = stats.correlated
            Rate = if stats.total > 0 then 100.0 * float stats.correlated / float stats.total else 0.0
        }

/// Load JSON files from intermediates directory
let loadFromJson (conn: DuckDBConnection) (intermediatesDir: string) =
    // Create tables
    conn.Execute("""
        CREATE TABLE IF NOT EXISTS raw_nodes AS 
        SELECT * FROM read_json('{}/*.nodes.json', 
            columns = {
                id: 'VARCHAR',
                kind: 'VARCHAR', 
                symbol: 'VARCHAR',
                range: 'STRUCT(startLine INT, startColumn INT, endLine INT, endColumn INT)',
                sourceFile: 'VARCHAR',
                parentId: 'VARCHAR',
                children: 'VARCHAR[]'
            });
    """, Path.Combine(intermediatesDir))
    
    conn.Execute("""
        CREATE TABLE IF NOT EXISTS raw_edges AS
        SELECT * FROM read_json('{}/*.edges.json',
            columns = {
                source: 'VARCHAR',
                target: 'VARCHAR',
                kind: 'VARCHAR'
            });
    """, Path.Combine(intermediatesDir))

/// Load from in-memory PSG (used during compilation)
let loadFromPSG (conn: DuckDBConnection) (psg: ProgramSemanticGraph) =
    // Clear existing data
    conn.Execute("DROP TABLE IF EXISTS raw_nodes; DROP TABLE IF EXISTS raw_edges;")
    
    // Create and populate nodes table
    conn.Execute("""
        CREATE TABLE raw_nodes (
            id VARCHAR,
            kind VARCHAR,
            symbol VARCHAR,
            sourceFile VARCHAR,
            parentId VARCHAR
        )
    """)
    
    use insertNode = conn.PrepareStatement(
        "INSERT INTO raw_nodes VALUES (?, ?, ?, ?, ?)"
    )
    
    psg.Nodes |> Map.iter (fun nodeId node ->
        insertNode.Execute(
            nodeId,
            node.SyntaxKind,
            node.Symbol |> Option.map (fun s -> s.DisplayName) |> Option.defaultValue null,
            node.SourceFile,
            node.ParentId |> Option.map (fun p -> p.Value) |> Option.defaultValue null
        )
    )
    
    // Create and populate edges table
    conn.Execute("""
        CREATE TABLE raw_edges (
            source VARCHAR,
            target VARCHAR,
            kind VARCHAR
        )
    """)
    
    use insertEdge = conn.PrepareStatement(
        "INSERT INTO raw_edges VALUES (?, ?, ?)"
    )
    
    psg.Edges |> List.iter (fun edge ->
        insertEdge.Execute(
            edge.Source.Value,
            edge.Target.Value,
            edge.Kind.ToString()
        )
    )

/// Generate full audit report
let generateReport (conn: DuckDBConnection) : AuditReport =
    {
        GraphStructure = Analysis.analyzeStructure conn
        CorrelationAnalysis = Analysis.analyzeCorrelation conn
        ReachabilityAnalysis = { 
            ReachableNodes = 0; UnreachableNodes = 0; DeadCodePercentage = 0.0 
        } // TODO
        CouplingCohesion = { 
            ModuleCoupling = []; AverageCoupling = 0.0 
        } // TODO
    }

/// Format report for console output
let formatReportForConsole (report: AuditReport) (verbose: bool) =
    if verbose then
        printfn "=== PSG STRUCTURE ANALYSIS ==="
        printfn "  Total nodes: %d" report.GraphStructure.TotalNodes
        printfn "  Orphaned nodes: %d" report.GraphStructure.OrphanedNodes
        printfn "  Self-referencing edges: %d" report.GraphStructure.SelfReferencingEdges
        printfn "  Disconnected components: %d" report.GraphStructure.DisconnectedComponents
        printfn ""
        printfn "=== SYMBOL CORRELATION ==="
        printfn "  Total symbols: %d" report.CorrelationAnalysis.TotalSymbols
        printfn "  Correlated: %d (%.1f%%)" 
            report.CorrelationAnalysis.CorrelatedSymbols 
            report.CorrelationAnalysis.Rate
    else
        // Compact output for normal compilation
        if report.GraphStructure.OrphanedNodes > 0 then
            printfn "⚠️  PSG: %d orphaned nodes detected" report.GraphStructure.OrphanedNodes
        if report.CorrelationAnalysis.Rate < 90.0 then
            printfn "⚠️  PSG: Low symbol correlation (%.1f%%)" report.CorrelationAnalysis.Rate
```

### 1.6 Standalone Audit Tool

```fsharp
// src/Tools/PSGAudit/Program.fs
module PSGAudit.Program

open System
open Core.Analysis.PSGAudit
open DuckDB.NET

[<EntryPoint>]
let main argv =
    match argv with
    | [| intermediatesDir |] ->
        use conn = new DuckDBConnection(":memory:")
        conn.Open()
        
        // Load JSON files
        printfn "Loading PSG JSON files from %s..." intermediatesDir
        loadFromJson conn intermediatesDir
        
        // Generate and display report
        let report = generateReport conn
        formatReportForConsole report true  // Always verbose for standalone
        
        0
    | _ ->
        eprintfn "Usage: PSGAudit <intermediates-directory>"
        1
```

### 1.7 Integration in Firefly Compilation Pipeline

```fsharp
// Modified src/Core/IngestionPipeline.fs
module Core.IngestionPipeline

open Core.Analysis.PSGAudit

let runIngestionPipeline (config: PipelineConfig) (projectFile: string) =
    // ... existing PSG construction ...
    
    // Integrated PSG audit during compilation
    if config.VerboseOutput || config.ShowPSGStats then
        use auditConn = new DuckDBConnection(":memory:")
        auditConn.Open()
        
        // Load current PSG state
        PSGAudit.loadFromPSG auditConn currentPSG
        
        // Generate report
        let report = PSGAudit.generateReport auditConn
        
        // Display inline during compilation
        PSGAudit.formatReportForConsole report config.VerboseOutput
        
        // After reachability analysis
        if prunedPSG.Nodes.Count < currentPSG.Nodes.Count then
            printfn "✓ Dead code elimination: %d nodes removed (%.1f%%)" 
                (currentPSG.Nodes.Count - prunedPSG.Nodes.Count)
                (100.0 * float (currentPSG.Nodes.Count - prunedPSG.Nodes.Count) / float currentPSG.Nodes.Count)
```

## Phase 2: Partial DuckDB Integration

### 2.1 Integration Architecture

```fsharp
// src/Core/Analysis/DuckDBAudit.fs
module Core.Analysis.DuckDBAudit

open DuckDB.NET
open Core.PSG.Types

type AuditDatabase = {
    Connection: DuckDBConnection
    IsInMemory: bool
}

type PSGAuditStats = {
    InitialNodeCount: int
    InitialEdgeCount: int
    OrphanedNodes: int
    SelfReferencingEdges: int
    CorrelationRate: float
    UnreachableNodes: int
    CouplingMetrics: ModuleCoupling list
    CohesionMetrics: ModuleCohesion list
}

and ModuleCoupling = {
    Module: string
    IncomingDependencies: int
    OutgoingDependencies: int
    CouplingScore: float
}

and ModuleCohesion = {
    Module: string
    InternalConnections: int
    ExternalConnections: int
    CohesionScore: float
}

/// Initialize in-memory DuckDB for PSG analysis
let initializeAuditDB () : AuditDatabase =
    let conn = new DuckDBConnection("Data Source=:memory:")
    conn.Open()
    
    // Install and load PGQ extension
    conn.Execute("INSTALL pgq;")
    conn.Execute("LOAD pgq;")
    
    // Create schema
    conn.Execute("""
        CREATE TABLE psg_nodes (
            id VARCHAR PRIMARY KEY,
            kind VARCHAR,
            symbol VARCHAR,
            source_file VARCHAR,
            parent_id VARCHAR,
            type_info VARCHAR,
            memory_size INTEGER
        );
        
        CREATE TABLE psg_edges (
            source VARCHAR,
            target VARCHAR,
            kind VARCHAR,
            metadata JSON
        );
    """)
    
    { Connection = conn; IsInMemory = true }

/// Load PSG into DuckDB for analysis
let loadPSG (psg: ProgramSemanticGraph) (db: AuditDatabase) =
    use trans = db.Connection.BeginTransaction()
    
    // Insert nodes
    let insertNode = db.Connection.PrepareStatement("""
        INSERT INTO psg_nodes (id, kind, symbol, source_file, parent_id, type_info, memory_size)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """)
    
    psg.Nodes |> Map.iter (fun nodeId node ->
        insertNode.Execute(
            nodeId,
            node.SyntaxKind,
            node.Symbol |> Option.map (fun s -> s.DisplayName) |> Option.defaultValue null,
            node.SourceFile,
            node.ParentId |> Option.map (fun p -> p.Value) |> Option.defaultValue null,
            node.Type |> Option.map (fun t -> t.Format()) |> Option.defaultValue null,
            node.MemoryLayout |> Option.map (fun m -> m.Size) |> Option.defaultValue 0
        )
    )
    
    // Insert edges
    let insertEdge = db.Connection.PrepareStatement("""
        INSERT INTO psg_edges (source, target, kind, metadata)
        VALUES (?, ?, ?, ?)
    """)
    
    psg.Edges |> List.iter (fun edge ->
        let metadata = 
            match edge.Kind with
            | TypeInstantiation args -> 
                JsonSerializer.Serialize({| typeArgs = args |> List.map (fun t -> t.Format()) |})
            | _ -> "{}"
        
        insertEdge.Execute(
            edge.Source.Value,
            edge.Target.Value,
            edge.Kind.ToString(),
            metadata
        )
    )
    
    trans.Commit()

/// Compute comprehensive statistics
let computeStats (db: AuditDatabase) : PSGAuditStats =
    // Basic counts
    let nodeCounts = db.Connection.QuerySingle<{| total: int; orphaned: int |}>("""
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN parent_id IS NULL AND kind != 'Module' THEN 1 END) as orphaned
        FROM psg_nodes
    """)
    
    let edgeCounts = db.Connection.QuerySingle<{| total: int; self_ref: int |}>("""
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN source = target THEN 1 END) as self_ref
        FROM psg_edges
    """)
    
    // Correlation rate
    let correlationRate = db.Connection.QueryScalar<float>("""
        SELECT 
            100.0 * COUNT(CASE WHEN symbol IS NOT NULL THEN 1 END) / NULLIF(COUNT(*), 0)
        FROM psg_nodes
        WHERE kind IN ('Binding', 'Application', 'Identifier')
    """)
    
    // Coupling analysis
    let coupling = db.Connection.Query<ModuleCoupling>("""
        WITH module_deps AS (
            SELECT 
                n1.source_file as from_module,
                n2.source_file as to_module,
                COUNT(*) as dep_count
            FROM psg_edges e
            JOIN psg_nodes n1 ON e.source = n1.id
            JOIN psg_nodes n2 ON e.target = n2.id
            WHERE n1.source_file != n2.source_file
            GROUP BY n1.source_file, n2.source_file
        )
        SELECT 
            from_module as Module,
            COUNT(DISTINCT CASE WHEN from_module = m.from_module THEN to_module END) as OutgoingDependencies,
            COUNT(DISTINCT CASE WHEN to_module = m.from_module THEN from_module END) as IncomingDependencies,
            AVG(dep_count) as CouplingScore
        FROM module_deps m
        GROUP BY from_module
    """) |> Seq.toList
    
    // Create property graph and analyze reachability
    db.Connection.Execute("""
        CREATE PROPERTY GRAPH psg_graph
        VERTEX TABLES (psg_nodes PROPERTIES (id, kind, symbol) PRIMARY KEY (id))
        EDGE TABLES (psg_edges SOURCE KEY (source) REFERENCES psg_nodes (id)
                              DESTINATION KEY (target) REFERENCES psg_nodes (id))
    """)
    
    let unreachableCount = db.Connection.QueryScalar<int>("""
        WITH reachable AS (
            FROM GRAPH_TABLE (psg_graph
                MATCH (entry:psg_nodes WHERE entry.symbol = 'main')-[*]->(n:psg_nodes)
                COLUMNS (n.id as node_id)
            )
            SELECT DISTINCT node_id
        )
        SELECT COUNT(*) FROM psg_nodes WHERE id NOT IN (SELECT node_id FROM reachable)
    """)
    
    {
        InitialNodeCount = nodeCounts.total
        InitialEdgeCount = edgeCounts.total
        OrphanedNodes = nodeCounts.orphaned
        SelfReferencingEdges = edgeCounts.self_ref
        CorrelationRate = correlationRate
        UnreachableNodes = unreachableCount
        CouplingMetrics = coupling
        CohesionMetrics = []  // TODO: Implement cohesion
    }
```

### 2.2 Integration Points in Pipeline

```fsharp
// Modified src/Core/IngestionPipeline.fs
let runIngestionPipeline (config: PipelineConfig) (projectFile: string) =
    // ... existing PSG construction ...
    
    // Optional DuckDB audit
    if config.EnableDuckDBAudit then
        use auditDB = DuckDBAudit.initializeAuditDB()
        
        // Audit initial PSG
        DuckDBAudit.loadPSG initialPSG auditDB
        let initialStats = DuckDBAudit.computeStats auditDB
        
        printfn "=== Initial PSG Statistics ==="
        printfn "Nodes: %d (Orphaned: %d)" initialStats.InitialNodeCount initialStats.OrphanedNodes
        printfn "Edges: %d (Self-referencing: %d)" initialStats.InitialEdgeCount initialStats.SelfReferencingEdges
        printfn "Symbol Correlation: %.1f%%" initialStats.CorrelationRate
        
        // Clear and reload after reachability
        auditDB.Connection.Execute("DELETE FROM psg_nodes; DELETE FROM psg_edges;")
        DuckDBAudit.loadPSG prunedPSG auditDB
        let prunedStats = DuckDBAudit.computeStats auditDB
        
        printfn "\n=== After Reachability Analysis ==="
        printfn "Nodes: %d (Eliminated: %d)" 
            prunedStats.InitialNodeCount 
            (initialStats.InitialNodeCount - prunedStats.InitialNodeCount)
        printfn "Dead Code Elimination: %.1f%%" 
            (100.0 * float (initialStats.InitialNodeCount - prunedStats.InitialNodeCount) / float initialStats.InitialNodeCount)
        
        // Coupling/Cohesion report
        printfn "\n=== Module Coupling Analysis ==="
        prunedStats.CouplingMetrics |> List.iter (fun m ->
            printfn "%s: In=%d, Out=%d, Score=%.2f" 
                m.Module m.IncomingDependencies m.OutgoingDependencies m.CouplingScore
        )
```

### 2.3 Visual Reports Generation

```fsharp
// src/Core/Analysis/PSGReports.fs
module Core.Analysis.PSGReports

open DuckDB.NET

/// Generate HTML report with D3.js visualizations
let generateHTMLReport (stats: PSGAuditStats) (outputPath: string) =
    let html = sprintf """
<!DOCTYPE html>
<html>
<head>
    <title>PSG Analysis Report</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        .metric { 
            display: inline-block; 
            padding: 20px; 
            margin: 10px;
            background: #f0f0f0;
            border-radius: 8px;
        }
        .metric h3 { margin: 0 0 10px 0; }
        .metric .value { font-size: 2em; font-weight: bold; }
    </style>
</head>
<body>
    <h1>PSG Analysis Report</h1>
    
    <div class="metrics">
        <div class="metric">
            <h3>Total Nodes</h3>
            <div class="value">%d</div>
        </div>
        <div class="metric">
            <h3>Correlation Rate</h3>
            <div class="value">%.1f%%</div>
        </div>
        <div class="metric">
            <h3>Unreachable Nodes</h3>
            <div class="value">%d</div>
        </div>
    </div>
    
    <div id="coupling-chart"></div>
    
    <script>
        // D3.js visualization of coupling metrics
        const data = %s;
        // ... D3 code to render coupling graph ...
    </script>
</body>
</html>
    """ stats.InitialNodeCount stats.CorrelationRate stats.UnreachableNodes 
        (JsonSerializer.Serialize(stats.CouplingMetrics))
    
    File.WriteAllText(outputPath, html)
```

## Phase 3: Future Full Integration

### 3.1 Embedded DuckDB as PSG Backend

```fsharp
// Future: src/Core/PSG/DuckDBBackend.fs
module Core.PSG.DuckDBBackend

/// PSG implementation backed by DuckDB for efficient graph operations
type DuckDBProgramSemanticGraph(connection: DuckDBConnection) =
    interface IProgramSemanticGraph with
        member this.AddNode(node: PSGNode) =
            // Direct insertion into DuckDB
            
        member this.AddEdge(edge: PSGEdge) =
            // Direct edge insertion
            
        member this.FindReachableNodes(entryPoints: NodeId list) =
            // Use PGQ for efficient reachability
            connection.Query<NodeId>("""
                FROM GRAPH_TABLE (psg_graph
                    MATCH (entry)-[*]->(reachable)
                    WHERE entry.id = ANY(?)
                    COLUMNS (DISTINCT reachable.id)
                )
            """, entryPoints)
            
        member this.ComputeCoupling() =
            // Real-time coupling analysis using SQL
```

### 3.2 Compilation Segmentation and Caching Strategy

**Future Vision**: Leverage DuckDB PGQ and coupling/cohesion analysis for intelligent compilation caching.

```fsharp
// Future: src/Core/Compilation/IncrementalCache.fs
module Core.Compilation.IncrementalCache

/// Compilation unit determined by coupling analysis
type CompilationSegment = {
    SegmentId: string
    RootModules: Set<string>
    CohesionScore: float
    LastCompiled: DateTime
    MLIRCache: byte[]
}

/// Use DuckDB to identify compilation boundaries
let computeCompilationSegments (db: DuckDBConnection) : CompilationSegment list =
    // Use coupling/cohesion metrics to find natural boundaries
    db.Query<CompilationSegment>("""
        WITH module_clusters AS (
            -- Community detection algorithm to find highly cohesive groups
            FROM GRAPH_TABLE (psg_graph
                MATCH (m1:Module)-[e:Edge*..3]-(m2:Module)
                WHERE coupling_weight(e) > 0.7
                COLUMNS (
                    cluster_id(m1, m2) as segment_id,
                    array_agg(DISTINCT m1.id) as modules,
                    avg(cohesion_score(m1, m2)) as cohesion
                )
            )
        )
        SELECT 
            segment_id as SegmentId,
            modules as RootModules,
            cohesion as CohesionScore
        FROM module_clusters
        WHERE array_length(modules) > 1
    """) |> Seq.toList

/// Incremental compilation using cached segments
let compileIncremental (changedFiles: Set<string>) (db: DuckDBConnection) =
    // Find affected segments using graph reachability
    let affectedSegments = db.Query<string>("""
        FROM GRAPH_TABLE (psg_graph
            MATCH 
                (changed:Node WHERE changed.source_file = ANY(?))
                -[*]->
                (segment:CompilationSegment)
            COLUMNS (DISTINCT segment.id)
        )
    """, changedFiles |> Set.toArray)
    
    // Only recompile affected segments
    affectedSegments |> Seq.iter recompileSegment
```

### 3.3 Benefits of DuckDB Integration

1. **Efficient Graph Operations**: PGQ provides optimized graph traversal
2. **SQL Analytics**: Complex metrics become simple queries
3. **Incremental Updates**: Can efficiently update graph during compilation
4. **Memory Efficiency**: Columnar storage for large codebases
5. **Export Flexibility**: Easy export to Parquet, JSON, or other formats
6. **Intelligent Caching**: Use coupling/cohesion analysis to determine optimal compilation boundaries
7. **Enterprise Scale**: Segment large codebases into independently cacheable compilation units
8. **Change Impact Analysis**: PGQ can quickly identify which segments are affected by source changes

## Implementation Timeline

1. **Immediate**: External audit tool using existing JSON files
2. **Short-term**: Partial integration for compilation statistics
3. **Medium-term**: Replace in-memory PSG with DuckDB-backed implementation
4. **Long-term**: Full integration with incremental compilation support

This approach provides immediate value for debugging the current PSG implementation while building toward a more sophisticated graph analysis infrastructure.