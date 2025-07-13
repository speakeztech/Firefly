# PSG Enhancement Requirements Memo

## Executive Summary

DuckDB analysis of Firefly's Program Semantic Graph reveals that the PSG foundation is **significantly more robust** than initial assessments indicated. However, critical gaps in symbol correlation and developer visibility require targeted enhancements for production readiness.

## Key Findings

### PSG Structural Integrity: **SOLID**
- 63% of nodes (565/892) properly maintain parent-child relationships
- 100% symbol correlation success rate across all files
- Proper entry point detection and graph traversal
- 16 FunctionCall edges successfully created

### Critical Gap: Symbol Name Resolution
The PSG correctly identifies function dependencies through symbol hashes (`sym_writeNewLine_6ae5fa57`, `sym_minLength_583a1fa2`) but fails to preserve human-readable symbol names. Developers cannot trace `use buffer = stackBuffer<byte> 256` to its Alloy library dependencies despite the edges existing in the graph structure.

### Disconnected Analysis Pipelines
Reachability analysis successfully identifies the 5-function call chain (`Error`, `Some`, `minLength`, `Alloy.Console.writeNewLine`, `Examples.HelloWorldDirect.hello`) through FCS typed AST processing, while PSG construction operates purely on syntax trees. This creates two parallel but uncoordinated dependency detection systems.

## Required Enhancements

### 1. Symbol Table Export
The PSG symbol table is not being exported to JSON artifacts, preventing post-compilation symbol resolution. Enable full symbol table serialization with hash-to-name mappings.

### 2. Symbol Name Preservation
Enhance correlation pipeline to maintain human-readable symbol names alongside hashes in correlation artifacts. Currently, function names disappear between PSG construction and JSON export.

### 3. Typed AST Integration
Bridge the gap between syntax tree PSG construction and typed AST reachability analysis. The PSG should incorporate semantic information from `checkResults.AssemblyContents.ImplementationFiles` during construction rather than post-processing.

### 4. Developer Traceability
Implement query capabilities that allow developers to trace specific code lines (like `stackBuffer<byte> 256`) through the complete dependency chain to library implementations.

## Impact Assessment

**Current State**: PSG provides accurate structural foundation but insufficient developer visibility for dependency validation.

**Post-Enhancement**: Developers gain complete transparency into compilation decisions with query-driven dependency analysis, enabling confident native code generation.

**Priority**: High - Dabbit MLIR generation requires complete symbol resolution for accurate code generation.