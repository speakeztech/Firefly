# Firefly Migration Plan: From AST to Program Semantic Graph

## Overview

This migration plan outlines the steps to transform Firefly's compilation pipeline from using separate AST and symbol information to a unified Program Semantic Graph (PSG) representation. This focuses on Phase 1 (FCS Ingestion) of the compiler pipeline, keeping changes as localized as possible while providing a solid foundation for downstream components.

## Phase 1: Planning and Infrastructure

### 1.1 Define the Program Semantic Graph Structure

**Files affected:**
- Create new file: `Core/PSG/Types.fs`

We've implemented the core PSG structure in Types.fs, defining the EnrichedNode type that combines syntax, semantics, and metadata. The implementation includes source location tracking, node ID generation, and parent-child relationships between nodes. The current PSG structure has been adapted to work with FCS 43.9.300 and includes maps for different node types and cross-reference capabilities.

### 1.2 XML Documentation Parser

**Files affected:**
- Create new file: `Core/PSG/XmlDocParser.fs`

Basic XML documentation parser is now operational using XParsec. The current implementation can extract MLIR dialect, operation names, and parameter maps. We've encountered challenges with more complex XML structures and are working on enhancing the parser's robustness.

### 1.3 Create Serialization for Intermediates

**Files affected:**
- Create new file: `Core/PSG/Serialization.fs`

Initial serialization functionality has been implemented in IntermediateWriter.fs rather than a dedicated Serialization module. The current implementation supports JSON serialization of PSG structures but requires enhancements for more efficient serialization and deserialization.

## Phase 2: Integration with FCS

### 2.1 Extend Project Processor

**Files affected:**
- Modify: `Core/FCSIngestion/ProjectProcessor.fs`

We've updated Pipeline.fs to properly integrate with FCS 43.9.300. The key challenge was adapting to the changed ParseFile method signature which no longer accepts SourceTextOpt or direct FSharpProjectOptions. We now explicitly get parsing options and create source text objects for each file, allowing for proper AST construction with the new FCS API.

### 2.2 Symbol-to-AST Mapper

**Files affected:**
- Create new file: `Core/PSG/SymbolMapper.fs`

Symbol-to-AST mapping functionality has been integrated into the Builder.fs file rather than a separate SymbolMapper module. The current implementation maps FSharpSymbolUse arrays to create relationships between ranges, symbols, and nodes. Further work is needed to improve the robustness of this mapping, particularly for complex cases.

### 2.3 XML Documentation Extractor

**Files affected:**
- Create new file: `Core/PSG/XmlDocExtractor.fs`

XML documentation extraction is currently handled directly in Builder.fs rather than a separate extractor module. The implementation extracts documentation from FSharpSymbol.XmlDoc property and passes it to the XmlDocParser for metadata extraction.

## Phase 3: PSG Construction

### 3.1 AST Traversal with Symbol Attachment

**Files affected:**
- Create new file: `Core/PSG/GraphBuilder.fs`

Initial implementation of PSG construction with AST traversal is working in Builder.fs. The approach traverses the parsed AST, creates enriched nodes with symbol information, and builds a connected graph. Current challenges include adapting to FCS 43.9.300's changed AST structure, particularly for pattern matching on union cases with additional trivia parameters.

### 3.2 Type Layout Extraction

**Files affected:**
- Create new file: `Core/PSG/TypeExtractor.fs`

Type layout extraction is planned but not yet implemented. This will be crucial for zero-allocation verification and memory layout analysis in later stages.

### 3.3 Dependency Analysis

**Files affected:**
- Create new file: `Core/PSG/DependencyAnalyzer.fs`

Dependency analysis functionality is planned but not yet implemented. This will build the dependency graph between symbols for reachability analysis.

## Phase 4: Integrating with Compilation Pipeline

### 4.1 Update Intermediate Writer

**Files affected:**
- Modify: `Core/Utilities/IntermediateWriter.fs`

Initial updates to IntermediateWriter.fs have been made to support PSG serialization. This needs to be extended for more comprehensive intermediate asset generation.

### 4.2 Modify Compilation Orchestrator

**Files affected:**
- Modify: `Dabbit/Pipeline/CompilationOrchestrator.fs`

Updates to the compilation orchestrator are planned but not yet implemented. This will involve modifying the pipeline to use the PSG instead of separate AST and symbol information.

### 4.3 Update Reachability Analysis

**Files affected:**
- Modify: `Core/AST/Reachability.fs`

Reachability analysis updates are planned but not yet implemented. Currently, we're focused on ensuring entry point detection works correctly as a prerequisite for reachability analysis.

## Phase 5: Testing and Validation

### 5.1 Create Test Suite for PSG

**Files affected:**
- Create new directory: `tests/PSG`

Create comprehensive tests for:
- XML documentation parsing
- Symbol-to-AST mapping
- PSG construction
- Serialization/deserialization

### 5.2 Validate with Test Projects

Validate the changes with existing test projects, ensuring:
- All compiler features still work
- Intermediates are correctly generated
- MLIR generation produces equivalent output

## Transition Strategy

To ensure a smooth transition, we'll:

1. Keep both pipelines functional during development
2. Add flags to toggle between AST-only and PSG modes
3. Generate both sets of intermediates for comparison
4. Implement PSG features incrementally, starting with simple cases
5. Migrate downstream components once PSG stabilizes

## Phase 6: VSCode Integration & Developer Tooling

### 6.1 VSCode-Friendly PSG Serialization

**Files affected:**
- Create new file: `Core/PSG/VSCodeSerializer.fs`

VSCode-friendly serialization is planned but not yet implemented. This will enable effective visualization and debugging of PSG structures in the IDE.

### 6.2 Source-to-PSG Mapping Artifacts

**Files affected:**
- Create new file: `Core/PSG/SourceMapper.fs`

Source-to-PSG mapping is planned but not yet implemented. This will create bidirectional links between source code positions and PSG nodes.

### 6.3 TextMate Grammar Enhancements

**Files affected:**
- Create new file: `tools/vscode/syntaxes/firefly.tmLanguage.json`

Create a TextMate grammar that can highlight:
- Regular F# syntax
- Alloy special types and functions
- Functions with MLIR metadata annotations
- Zero-allocation guarantees
- Stack vs. static allocation sites

### 6.4 Semantic Highlighting Provider

**Files affected:**
- Create new file: `tools/vscode/src/semanticHighlighting.ts`

Implement a semantic highlighting provider that:
- Uses the PSG to provide accurate syntax highlighting
- Highlights MLIR-mapped functions differently
- Shows allocation sites and their types
- Differentiates between stack and static allocations

### 6.5 PSG Explorer UI

**Files affected:**
- Create new directory: `tools/vscode/src/psgExplorer`

Implement a VSCode view that:
- Shows the PSG structure as a navigable tree
- Allows drilling down into modules, types, and functions
- Displays MLIR metadata when present
- Shows symbol dependencies and call graphs
- Provides "jump to source" for any PSG node

### 6.6 Hover Provider for MLIR Metadata

**Files affected:**
- Create new file: `tools/vscode/src/hoverProvider.ts`

Implement a hover provider that:
- Shows MLIR metadata for functions when hovering
- Displays memory allocation information
- Shows symbol dependencies
- Provides links to related PSG nodes

## Long-term Benefits

This migration:
1. Creates a single source of truth for program representation
2. Enables richer analyses that combine syntactic and semantic information
3. Provides a foundation for more sophisticated MLIR mapping
4. Simplifies the compilation pipeline by reducing data duplication
5. Makes Alloy metadata integration more straightforward
6. Enables powerful developer tooling through VSCode integration
7. Creates a transparent compilation process visible through IDE tools

## Code Structure Evolution

```
Core/
├── AST/                    # Current AST processing
│   ├── Extraction.fs       
│   ├── Dependencies.fs     
│   ├── Reachability.fs     
│   └── Validation.fs       
├── PSG/                    # New Program Semantic Graph
│   ├── Types.fs            # PSG structure definitions
│   ├── XmlDocParser.fs     # Parse XML docs with XParsec
│   ├── SymbolMapper.fs     # Map symbols to AST nodes
│   ├── GraphBuilder.fs     # Build the complete PSG
│   ├── TypeExtractor.fs    # Extract type layouts
│   ├── DependencyAnalyzer.fs # Build dependency graph
│   └── Serialization.fs    # PSG serialization
├── FCSIngestion/           
│   ├── ProjectOptionsLoader.fs
│   └── ProjectProcessor.fs # Enhanced to build PSG
└── Utilities/
    ├── IntermediateWriter.fs # Updated to write PSG
```

This progressive approach ensures we maintain compiler functionality while evolving toward a more powerful semantic representation.

## Outstanding Issues and Next Steps

Our most pressing issue is with entry point detection. The compiler correctly identifies main functions but fails to properly detect the EntryPoint attribute. This is likely due to how attributes are represented in FCS 43.9.300. The current diagnostic output shows "Found potential entry point: main (attribute: false, main function: true)" which confirms function detection works but attribute recognition needs improvement.

The next major steps are:
1. Fix entry point attribute detection by improving how we access attribute information in FCS 43.9.300
2. Enhance symbol-to-node mapping to reliably connect symbols to PSG nodes
3. Complete PSG construction with full AST traversal and symbol attachment
4. Begin integration with downstream components, starting with reachability analysis

We've made significant progress on the FCS 43.9.300 integration, particularly with Pipeline.fs, Builder.fs, and IntermediateWriter.fs, but more work is needed to ensure robust PSG construction and entry point detection.