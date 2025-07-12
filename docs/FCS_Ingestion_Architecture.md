# Firefly Core Architecture - FCS Ingestion

## Overview

This new architecture represents an advancement in how Firefly leverages F# Compiler Services (FCS). We continue to use FCS as the single source of truth for all semantic information along with advanced memory layout. We're also adding some XML hints in base libraries in order to inform MLIR processing as well as "coupling and cohesion" analysis to inform downstream compilation optimization.

## Core Folder Structure

```
Core/
├── FCS/                    # FCS Processing
│   ├── ProjectContext.fs
│   ├── SymbolAnalysis.fs
│   └── TypedASTAccess.fs
├── Meta/                   # Metadata Integration
│   ├── MetadataParser.fs
│   └── AlloyHints.fs
├── Analysis/               # Firefly Intelligence
│   ├── CouplingCohesion.fs
│   ├── Reachability.fs
│   └── MemoryLayout.fs
├── Templates/              # Platform Definitions
│   ├── TemplateTypes.fs
│   └── TemplateLoader.fs
└── IngestionPipeline.fs    # Orchestration
```

## Key Architectural Changes

### 1. **Eliminated Custom Correlation**

**Old Approach:**
- Custom PSG builder that manually correlated symbols to AST nodes
- Range-based mapping maintained separately
- Duplicate traversal of syntax and typed trees

**New Approach:**
- Direct use of `FSharpCheckProjectResults.GetAllUsesOfAllSymbols()`
- FCS's built-in position tracking via `GetSymbolUseAtLocation`
- Single traversal with FCS providing correlations

### 2. **Simplified Data Structures**

**Old:**
```fsharp
type EnrichedNode<'TSyntax> = {
    Syntax: 'TSyntax
    Symbol: FSharpSymbol option
    Metadata: MLIRMetadata option
    SourceLocation: SourceLocation
    Id: NodeId
    ParentId: NodeId option
    Children: NodeId list
}
```

**New:**
```fsharp
// Direct use of FCS types
type ProjectResults = {
    CheckResults: FSharpCheckProjectResults
    ParseResults: FSharpParseFileResults[]
    SymbolUses: FSharpSymbolUse[]
    CompilationOrder: string[]
}
```

### 3. **Coupling/Cohesion Analysis**

The most significant addition is the coupling/cohesion analysis that directly informs memory layout decisions:

```fsharp
// Semantic units detected through analysis
type SemanticUnit = 
    | Module of FSharpEntity
    | Namespace of string
    | FunctionGroup of FSharpMemberOrFunctionOrValue list
    | TypeCluster of FSharpEntity list

// Components with measured cohesion
type Component = {
    Id: string
    Units: SemanticUnit list
    Cohesion: float
    AverageCoupling: float
    Boundaries: ComponentBoundary list
}
```

### 4. **Memory Layout Intelligence**

Memory layout is now derived from semantic relationships rather than manual hints:

```fsharp
// Layout hints from coupling analysis
type MemoryLayoutHint = 
    | Contiguous of units: SemanticUnit list      // High cohesion
    | Isolated of unit: SemanticUnit              // Low coupling
    | SharedRegion of units: SemanticUnit list * accessPattern: AccessPattern
    | Tiered of hot: SemanticUnit list * cold: SemanticUnit list
```

### 5. **Platform Templates**

TOML-based platform templates replace hardcoded assumptions:

```toml
[platform]
family = "STM32"
architecture = "ARMv8-M"

[region.tcm]
base = "0x20000000"
size = "64KB"
attributes = ["fast", "secure"]
```

## Module Organization

### Core/FCS/
**Deep FCS Integration** - Replaces custom AST processing
- `ProjectContext.fs` - Manages FSharpChecker with caching
- `SymbolAnalysis.fs` - Direct symbol relationship extraction
- `TypedASTAccess.fs` - Simplified typed AST traversal

### Core/Meta/
**Metadata Processing** - Alloy hint extraction from XML docs
- `MetadataParser.fs` - XParsec-based XML doc parsing
- `AlloyHints.fs` - Platform-aware hint validation and merging

### Core/Analysis/
**Firefly Intelligence** - Novel analysis capabilities
- `CouplingCohesion.fs` - Component detection through metrics
- `Reachability.fs` - FCS-based dead code elimination
- `MemoryLayout.fs` - Coupling-informed memory organization

### Core/Templates/
**Platform Abstraction** - Hardware capability definitions
- `TemplateTypes.fs` - Platform constraint modeling
- `TemplateLoader.fs` - TOML template parsing

### Core/IngestionPipeline.fs
**FCS Ingestion Orchestration** - Coordinates the analysis phases

## File Organization Notes

- All modules are under the `Core` namespace with sub-namespaces matching folder names
- Each folder represents a distinct concern in the ingestion pipeline
- Dependencies flow: FCS → Meta → Analysis → Templates → IngestionPipeline
- No circular dependencies between folders

## Benefits of New Architecture

1. **Correctness** - Uses FCS's proven semantic analysis
2. **Performance** - Leverages FCS's optimized caching
3. **Maintainability** - Less custom code to maintain
4. **Intelligence** - Coupling analysis reveals program structure
5. **Flexibility** - Platform templates allow easy adaptation

## Migration Path

1. Remove old Core/AST/* modules (except as reference)
2. Remove Core/PSG/* custom correlation code
3. Update imports to use new Core/FCS/* modules
4. Update metadata processing to use Core/Meta/* modules
5. Add platform templates for target hardware
6. Enable coupling analysis in pipeline config

## Example Usage

```fsharp
open Core.IngestionPipeline

let config = {
    defaultConfig with
        TemplateName = Some "stm32l5"
        EnableCouplingAnalysis = true
        EnableMemoryOptimization = true
        OutputIntermediates = true
        IntermediatesDir = Some "./intermediates"
}

let! result = runPipeline "MyProject.fsproj" config

match result with
| { Success = true; CouplingAnalysis = Some ca } ->
    printfn "Found %d components with average cohesion %.2f" 
        ca.Components.Length 
        ca.Report.AverageCohesion
| _ -> 
    printfn "Compilation failed"
```

## Future Enhancements

1. **Machine Learning** - Learn optimal layouts from successful compilations
2. **Interactive Mode** - Real-time coupling visualization in IDE
3. **Cross-Platform** - Automatic template selection based on code patterns
4. **Safety Proofs** - Formal verification of memory isolation