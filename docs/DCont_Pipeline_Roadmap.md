# Firefly Continuation Pipeline Prototyping Roadmap

## Overview: From PSG Morass to "DCont All The Way Down"

This roadmap transforms your current FCS-based ProgramSemanticGraph into a continuation-centric compilation pipeline targeting WAMI (WebAssembly with stack switching) through MLIR's delimited continuation dialect. The focus remains on rapid prototyping with in-memory processing to achieve demonstrable results quickly using simple CLI proof-of-concepts.

## Phase 0: PSG Foundation Cleanup
**Goal**: Establish proper tombstone behavior and reachability analysis in existing ProgramSemanticGraph

### PSG Reachability Infrastructure
- [ ] Complete tombstone behavior implementation for soft-delete reachability analysis
- [ ] Finalize `ReachabilityHelpers.markReachable` and `ReachabilityHelpers.markUnreachable` functions
- [ ] Ensure `IsReachable`, `EliminationPass`, and `EliminationReason` fields work correctly
- [ ] Validate reachability analysis produces consistent results before adding continuation layers

**Success Criteria**: PSG reachability analysis works correctly with consistent tombstone behavior

## Phase 1: FCS Pipeline Enhancement  
**Goal**: Enhance FCS integration to extract continuation boundaries and preserve semantic information

### Enhanced FCS Processing
- [ ] Extend existing FCS correlation to identify async boundaries, resource scopes, and effect boundaries
- [ ] Map F# `async { }` blocks to continuation delimiter metadata in PSG nodes
- [ ] Track `use` bindings and `try-with` constructs as resource and exception boundaries
- [ ] Preserve delimited continuation semantic information through existing PSG.Types structure

### DuckDB FCS Integration
- [ ] Store FCS symbol correlation data in DuckDB for persistent cross-compilation analysis
- [ ] Create event log for FCS processing steps (parsing, type checking, correlation)
- [ ] Implement DuckDB queries for symbol usage analysis and cross-file dependencies

**Success Criteria**: PSG reachability analysis works correctly with consistent tombstone behavior

## Phase 1: FCS Pipeline Enhancement  
**Goal**: Enhance FCS integration to extract continuation boundaries and preserve semantic information

### Enhanced FCS Processing
- [ ] Extend existing FCS correlation to identify async boundaries, resource scopes, and effect boundaries
- [ ] Map F# `async { }` blocks to continuation delimiter metadata in PSG nodes
- [ ] Track `use` bindings and `try-with` constructs as resource and exception boundaries
- [ ] Preserve delimited continuation semantic information through existing PSG.Types structure
- [ ] Validate continuation boundary detection with HelloWorldDirect example

**Success Criteria**: FCS processing identifies and marks continuation boundaries in existing ProgramSemanticGraph structure

## Phase 2: Bidirectional Zipper Implementation
**Goal**: Implement bidirectional zipper for ProgramSemanticGraph traversal and transformation

### Bidirectional Zipper Core
```fsharp
// Bidirectional zipper operations for ProgramSemanticGraph
module PSGZipper =
    let moveDown: NodeId -> PSGZipper -> PSGZipper option           // Navigate to child
    let moveUp: PSGZipper -> PSGZipper option                       // Navigate to parent  
    let moveLeft: PSGZipper -> PSGZipper option                     // Navigate to previous sibling
    let moveRight: PSGZipper -> PSGZipper option                    // Navigate to next sibling
    let moveToRoot: PSGZipper -> PSGZipper                          // Navigate to root
    let moveToNode: NodeId -> PSGZipper -> PSGZipper option         // Direct navigation to specific node
```

### Context-Aware Transformations
- [ ] Implement zipper context preservation during transformations
- [ ] Add capability to transform focused node while maintaining graph invariants
- [ ] Support bulk transformations across zipper paths
- [ ] Integrate with existing `ChildrenStateHelpers` for consistent state management
- [ ] Test zipper operations with simple async code transformations

**Success Criteria**: Bidirectional zipper enables complex PSG transformations while preserving context and graph integrity

## Phase 3: In-Memory Property List Generation
**Goal**: Generate comprehensive property lists for control and data flow analysis using in-memory structures

### Control Flow Property Generation
```fsharp
// Generate control flow properties for continuation boundaries identified in PSG
let analyzeControlFlow: ProgramSemanticGraph -> NodeId -> ControlFlowProperties
let findEntryPoints:    ProgramSemanticGraph -> NodeId list
let findExitPoints:     ProgramSemanticGraph -> NodeId list  
let findSuspendPoints:  ProgramSemanticGraph -> SuspendPoint list
```

### Data Flow Property Generation
```fsharp
// Track variable definitions, uses, and dependencies across continuation boundaries
let analyzeDataFlow:        ProgramSemanticGraph -> DataFlowProperties
let extractDefinitions:     PSGNode -> VariableDefinition list
let extractUses:           PSGNode -> VariableUse list
let computeLiveVariables:  ProgramSemanticGraph -> LivenessInfo
```

### In-Memory Property Storage
- [ ] Design property list data structures for in-memory storage and fast lookup
- [ ] Store control flow properties as queryable maps with efficient access patterns
- [ ] Store data flow dependencies as graph relationships in memory
- [ ] Implement incremental property updates when PSG changes
- [ ] Create JSON serialization for property lists to enable debugging and inspection

### Effect Flow Properties
- [ ] Track async effects, resource acquisition and cleanup, and exception boundaries
- [ ] Store effect dependencies in memory-based graph structures
- [ ] Generate effect flow property lists for MLIR effect dialect integration

**Success Criteria**: Complete flow analysis generates actionable property lists stored in memory with fast lookup and JSON debugging capability

## Phase 4: MLIR Integration Pipeline
**Goal**: Compile ProgramSemanticGraph with continuation semantics to MLIR DCont dialect

### MLIR Dialect Mapping Strategy
- [ ] Map PSG nodes with continuation boundary metadata to MLIR DCont operations
- [ ] Map async boundaries to `dcont.reset` and `dcont.shift` operations
- [ ] Map resource boundaries to automatic cleanup patterns in MLIR
- [ ] Preserve F# type information through MLIR type system
- [ ] Validate mapping with HelloWorldDirect async patterns

### Type System Preservation
```fsharp
// Map F# types to MLIR types while preserving continuation structure
let deriveFunctionType:     PSGNode -> TypeMapping -> MLIRType
let generateTypeDefinitions: ProgramSemanticGraph -> MLIRModule -> TypeMapping  
let preserveContinuationTypes: ProgramSemanticGraph -> MLIRModule -> unit
```

### Code Generation Pipeline
```fsharp
// Core compilation functions mapping PSG to MLIR operations
let compilePSGToMLIR:         ProgramSemanticGraph -> MLIRModule
let compileContinuationNode:  PSGNode -> TypeMapping -> MLIRModule -> MLIROperation
let compileEffectfulNode:     PSGNode -> TypeMapping -> MLIRBlock -> MLIROperation list
```

**Success Criteria**: F# async code in PSG compiles to valid MLIR DCont dialect operations with preserved semantics

## Phase 5: WAMI Target Implementation  
**Goal**: Lower MLIR DCont operations to WebAssembly with stack switching (WAMI)

### DCont to WAMI Lowering
```fsharp
// Lower delimited continuations to WAMI stack switching operations
let lowerDContToWAMI:     DContOperation -> WAMIOperation list
let lowerAsyncToWAMI:     AsyncOperation -> WAMIOperation list
let lowerResourceToWAMI:  ResourceOperation -> WAMIOperation list
```

### Stack Switching Implementation
- [ ] Map `dcont.reset` to `stack.new` and `stack.switch` operations
- [ ] Map `dcont.shift` to `suspend` with effect tags
- [ ] Map `dcont.resume` to `resume` with continuation passing
- [ ] Implement continuation capture and restore runtime functions

### WAMI Runtime Infrastructure
- [ ] Generate WAMI module with stack switching support
- [ ] Implement effect handling with WebAssembly exception integration
- [ ] Add debugging support for continuation flow in WebAssembly
- [ ] Test complete pipeline with HelloWorldDirect compilation to executable WASM

### CLI Integration and Testing
- [ ] Implement command-line interface for end-to-end compilation
- [ ] Create test harness for validating WASM execution against expected output
- [ ] Establish baseline performance measurements for simple CLI programs
- [ ] Document compilation process and debugging procedures

**Success Criteria**: HelloWorldDirect F# source compiles to executable WebAssembly that runs correctly at command line with preserved continuation semantics

## Phase 3: Property List Generation and Flow Analysis
**Goal**: Generate comprehensive property lists for control and data flow analysis using DuckDB

### Control Flow Property Generation
```fsharp
// Generate control flow properties for continuation boundaries identified in PSG
let analyzeControlFlow: ProgramSemanticGraph -> NodeId -> ControlFlowProperties
let findEntryPoints:    ProgramSemanticGraph -> NodeId list
let findExitPoints:     ProgramSemanticGraph -> NodeId list  
let findSuspendPoints:  ProgramSemanticGraph -> SuspendPoint list
```

### Data Flow Property Generation
```fsharp
// Track variable definitions, uses, and dependencies across continuation boundaries
let analyzeDataFlow:        ProgramSemanticGraph -> DataFlowProperties
let extractDefinitions:     PSGNode -> VariableDefinition list
let extractUses:           PSGNode -> VariableUse list
let computeLiveVariables:  ProgramSemanticGraph -> LivenessInfo
```

### DuckDB Property Store
- [ ] Design property list schema in DuckDB with PGQ graph queries
- [ ] Store control flow properties as graph edges with metadata
- [ ] Store data flow dependencies as queryable graph relationships
- [ ] Create event store for tracking property list generation and updates
- [ ] Implement incremental property updates when PSG changes

### Effect Flow Properties
- [ ] Track async effects, resource acquisition/cleanup, and exception boundaries
- [ ] Store effect dependencies in DuckDB graph structure
- [ ] Generate effect flow property lists for MLIR effect dialect integration

**Success Criteria**: Complete flow analysis generates actionable property lists stored in DuckDB with fast graph queries

## Phase 4: MLIR Integration Pipeline
**Goal**: Compile ProgramSemanticGraph with continuation semantics to MLIR DCont dialect

### MLIR Dialect Mapping Strategy
- [ ] Map PSG nodes with continuation boundary metadata to MLIR DCont operations
- [ ] Map async boundaries to `dcont.reset`/`dcont.shift` operations
- [ ] Map resource boundaries to automatic cleanup patterns in MLIR
- [ ] Preserve F# type information through MLIR type system

### DuckDB MLIR Integration
- [ ] Store MLIR operation generation events in DuckDB event store
- [ ] Track mapping from PSG NodeId to MLIR operation for debugging
- [ ] Create queries for analyzing MLIR generation patterns and optimization opportunities

### Type System Preservation
```fsharp
// Map F# types to MLIR types while preserving continuation structure
let deriveFunctionType:     PSGNode -> TypeMapping -> MLIRType
let generateTypeDefinitions: ProgramSemanticGraph -> MLIRModule -> TypeMapping  
let preserveContinuationTypes: ProgramSemanticGraph -> MLIRModule -> unit
```

### Code Generation Pipeline
```fsharp
// Core compilation functions mapping PSG to MLIR operations
let compilePSGToMLIR:         ProgramSemanticGraph -> MLIRModule
let compileContinuationNode:  PSGNode -> TypeMapping -> MLIRModule -> MLIROperation
let compileEffectfulNode:     PSGNode -> TypeMapping -> MLIRBlock -> MLIROperation list
```

**Success Criteria**: F# async code in PSG compiles to valid MLIR DCont dialect operations with preserved semantics

## Phase 5: WAMI Target Implementation  
**Goal**: Lower MLIR DCont operations to WebAssembly with stack switching (WAMI)

### DCont to WAMI Lowering
```fsharp
// Lower delimited continuations to WAMI stack switching operations
let lowerDContToWAMI:     DContOperation -> WAMIOperation list
let lowerAsyncToWAMI:     AsyncOperation -> WAMIOperation list
let lowerResourceToWAMI:  ResourceOperation -> WAMIOperation list
```

### Stack Switching Implementation
- [ ] Map `dcont.reset` to `stack.new` + `stack.switch` operations
- [ ] Map `dcont.shift` to `suspend` with effect tags
- [ ] Map `dcont.resume` to `resume` with continuation passing
- [ ] Implement continuation capture/restore runtime functions

### DuckDB WAMI Integration
- [ ] Store WAMI generation events and stack switching patterns in event store
- [ ] Track performance characteristics of different continuation lowering strategies
- [ ] Create queries for analyzing WAMI output optimization opportunities

### WAMI Runtime Infrastructure
- [ ] Generate WAMI module with stack switching support
- [ ] Implement effect handling with WebAssembly exception integration
- [ ] Add debugging support for continuation flow in WebAssembly

**Success Criteria**: F# async code runs in WebAssembly with preserved continuation semantics and "dcont all the way down"

## Phase 6: Optimization and Compiler Intelligence
**Goal**: Production-ready compilation pipeline with DuckDB-powered optimization

### PSG Optimization Passes
```fsharp
// Optimization passes operating on ProgramSemanticGraph with DuckDB intelligence
let inlineTrivialContinuations:   ProgramSemanticGraph -> ProgramSemanticGraph  
let mergeContinuationChains:      ProgramSemanticGraph -> ProgramSemanticGraph
let optimizeResourceScopes:       ProgramSemanticGraph -> ProgramSemanticGraph
let eliminateUnusedContinuations: ProgramSemanticGraph -> ProgramSemanticGraph
```

### DuckDB-Powered Compiler Intelligence
- [ ] Use DuckDB PGQ queries to identify optimization opportunities across compilation history
- [ ] Implement learning from previous compilation patterns stored in event store
- [ ] Create adaptive optimization based on property list analysis patterns
- [ ] Generate compilation reports and insights from DuckDB analytics

### MLIR Pass Pipeline Integration
- [ ] Implement `dcont-optimize` pass with DuckDB guidance
- [ ] Implement `async-to-dcont` lowering pass with historical optimization data
- [ ] Implement `effect-optimize` pass using flow property insights
- [ ] Implement `convert-dcont-to-wami` final lowering with performance tracking

### Debug and Tooling Infrastructure
- [ ] Source maps from F# through PSG to MLIR to WASM using DuckDB correlation
- [ ] Debug continuation flow visualization powered by DuckDB graph queries
- [ ] Performance profiling for continuation overhead with historical analysis
- [ ] Integration with existing Fidelity toolchain and BAREWire protocols

**Success Criteria**: Production-ready F# to WAMI compilation with DuckDB-powered optimization and comprehensive debugging support

### 2.1 Continuation Boundary Detection
```fsharp
// Key patterns to detect in F# AST
let identifyAsyncBoundaries:    SynExpr -> ContinuationBoundary list
let identifyResourceBoundaries: SynExpr -> ContinuationBoundary list  
let identifyEffectBoundaries:   SynExpr -> ContinuationBoundary list
```

### 2.2 PSG Region Analysis
- [ ] Implement `collectRegionNodes` - group PSG nodes by continuation boundaries
- [ ] Implement `findRegionDependencies` - track data flow between regions
- [ ] Create control flow analysis: `buildControlFlowFromNodes`

### 2.3 AST to Continuation Transformation
- [ ] Transform F# `async { }` blocks to `DelimitedContinuation` with `AsyncBoundary`
- [ ] Transform `use` bindings to `ResourceBoundary` continuations
- [ ] Transform `try-with` to `ExceptionBoundary` continuations

**Success Criteria**: Can migrate simple F# async code to continuation PSG

## Phase 3: Flow Analysis (Weeks 9-12)
**Goal**: Generate property lists for control and data flow analysis

### 3.1 Control Flow Properties
```fsharp
// Implement comprehensive control flow analysis
let analyzeControlFlow: DelimitedContinuation -> ControlFlowProperties
let findEntryPoints:    DelimitedContinuation -> ContinuationId list
let findExitPoints:     DelimitedContinuation -> ContinuationId list
let findSuspendPoints:  DelimitedContinuation -> SuspendPoint list
```

### 3.2 Data Flow Properties
```fsharp
// Track variable definitions, uses, and dependencies
let analyzeDataFlow:        DelimitedContinuation -> DataFlowProperties
let extractDefinitions:     ContinuationBody -> VariableDefinition list
let extractUses:           ContinuationBody -> VariableUse list
let computeLiveVariables:  DelimitedContinuation -> LivenessInfo
```

### 3.3 Effect Flow Properties
```fsharp
// Track effects and their dependencies
let analyzeEffectFlow:      DelimitedContinuation -> EffectFlowProperties
let extractPerformedEffects: ContinuationBody -> PerformedEffect list
let extractHandledEffects:   ContinuationBody -> HandledEffect list
```

**Success Criteria**: Complete flow analysis for continuation PSG, property lists generated

## Phase 4: MLIR Integration (Weeks 13-20)
**Goal**: Compile continuation PSG to MLIR with DCont dialect

### 4.1 MLIR Dialect Mapping
- [ ] Set up MLIR C++ FFI bindings for F#
- [ ] Map `DelimitedContinuation` to `dcont.reset`/`dcont.shift` operations  
- [ ] Map `AsyncBoundary` to `async.func` operations
- [ ] Map `ResourceBoundary` to automatic cleanup patterns

### 4.2 Type System Integration
```fsharp
// Map F# types to MLIR types preserving continuation structure
let deriveFunctionType:     DelimitedContinuation -> TypeMapping -> MLIRType
let generateTypeDefinitions: ContinuationPSG -> MLIRModule -> TypeMapping
```

### 4.3 Code Generation
```fsharp
// Core compilation functions  
let compileContinuationToFunction: DelimitedContinuation -> TypeMapping -> MLIRModule -> MLIROperation
let compileContinuationBody:       ContinuationBody -> TypeMapping -> MLIRBlock -> MLIROperation list
let compileEffectfulExpression:    EffectfulExpression -> TypeMapping -> MLIRBlock -> MLIROperation list
```

**Success Criteria**: F# async code compiles to MLIR DCont dialect operations

## Phase 5: WAMI Target (Weeks 21-28)
**Goal**: Lower MLIR DCont to WebAssembly with stack switching

### 5.1 DCont to WAMI Lowering
```fsharp
// Lower delimited continuations to WAMI stack switching
let lowerDContToWAMI:     DContOperation -> WAMIOperation list
let lowerAsyncToWAMI:     AsyncOperation -> WAMIOperation list  
let lowerResourceToWAMI:  ResourceOperation -> WAMIOperation list
```

### 5.2 Stack Management
- [ ] Map `dcont.reset` to `stack.new` + `stack.switch`
- [ ] Map `dcont.shift` to `suspend` with effect tags
- [ ] Map `dcont.resume` to `resume` with continuation passing

### 5.3 WAMI Runtime
- [ ] Generate WAMI module with stack switching support
- [ ] Add continuation capture/restore runtime functions
- [ ] Implement effect handling with WebAssembly exceptions

**Success Criteria**: F# async code runs in WebAssembly with preserved continuation semantics

## Phase 6: Optimization and Production (Weeks 29-36)
**Goal**: Production-ready compilation pipeline with optimizations

### 6.1 PSG Optimizations
```fsharp
let inlineTrivialContinuations:   ContinuationPSG -> ContinuationPSG
let mergeContinuationChains:      ContinuationPSG -> ContinuationPSG
let optimizeResourceScopes:       ContinuationPSG -> ContinuationPSG
let eliminateUnusedContinuations: ContinuationPSG -> ContinuationPSG
```

### 6.2 MLIR Pass Pipeline
- [ ] Implement `dcont-optimize` pass
- [ ] Implement `async-to-dcont` lowering pass  
- [ ] Implement `effect-optimize` pass
- [ ] Implement `convert-dcont-to-wami` final lowering

### 6.3 Debug and Tooling
- [ ] Source maps from F# through MLIR to WASM
- [ ] Debug continuation flow visualization
- [ ] Performance profiling for continuation overhead
- [ ] Integration with existing Fidelity toolchain

**Success Criteria**: Production-ready F# to WAMI compilation with debugging support

**Success Criteria**: HelloWorldDirect F# source compiles to executable WebAssembly that runs correctly at command line with preserved continuation semantics

## Implementation Priorities

### Foundation Phase
The PSG tombstone behavior forms the critical foundation for all subsequent analysis and optimization passes. Establishing proper reachability analysis with soft-delete semantics ensures that continuation boundary detection and flow analysis operate on reliable data structures.

The FCS enhancement phase builds directly on existing correlation systems to identify continuation boundaries, ensuring the pipeline can recognize and preserve F# async semantics from the earliest compilation stage. This phase validates the approach using the constrained HelloWorldDirect example.

### Core Compilation Pipeline
The bidirectional zipper implementation enables sophisticated PSG transformations while maintaining context, forming the basis for all code transformations required by the continuation compilation process. The zipper operations provide the navigation and transformation capabilities needed for the MLIR lowering process.

Property list generation converts PSG analysis into actionable intelligence that drives the compilation process. The in-memory approach eliminates database complexity while preserving the analytical depth required for effective continuation compilation.

### Target Implementation
MLIR DCont integration provides the bridge to modern compiler infrastructure while preserving high-level continuation semantics. The mapping from PSG continuation boundaries to MLIR delimited continuation operations represents the critical architectural transformation point.

WAMI stack switching implementation delivers the "dcont all the way down" vision, enabling functional programming patterns to run efficiently in WebAssembly environments while preserving continuation semantics throughout the execution pipeline.

## Risk Mitigation Strategy

### Technical Risk Management
The approach leverages existing MLIR infrastructure rather than building custom solutions, reducing implementation risk while providing access to the broader MLIR ecosystem. Starting with the constrained HelloWorldDirect example allows validation of core concepts before expanding to more complex scenarios.

The in-memory processing approach eliminates database integration complexity during the prototyping phase while preserving the ability to add persistence capabilities when the architecture matures. This strategy reduces technical risk while maintaining architectural flexibility.

### Validation Approach
Each phase builds meaningful value that can be validated independently, reducing cumulative risk while progressing toward the compilation pipeline goals. The HelloWorldDirect example provides a concrete validation target that exercises core language features while remaining simple enough for manual verification.

Success metrics focus on demonstrable compilation results rather than performance optimization, ensuring that architectural decisions are validated before complexity increases. The CLI execution target provides immediate feedback on compilation correctness.

## Success Metrics and Validation

### Foundation Success Indicators
The PSG tombstone behavior functions correctly with consistent reachability analysis results. FCS processing successfully identifies async boundaries, resource scopes, and effect boundaries in the HelloWorldDirect source code. The bidirectional zipper enables complex PSG transformations while maintaining graph integrity and context.

### Pipeline Success Indicators  
Property list generation provides actionable compilation information stored in efficient in-memory data structures. MLIR integration produces valid DCont dialect operations from F# async constructs with preserved semantic information. The compilation pipeline processes the HelloWorldDirect example without errors through all transformation stages.

### Target Success Indicators
WAMI WebAssembly successfully executes the HelloWorldDirect program with continuation semantics preserved throughout the compilation pipeline. The generated WebAssembly binary runs correctly at the command line and produces expected output. Debug tooling provides clear visibility into continuation flow from F# source through to WebAssembly execution.

## Integration Philosophy

This roadmap preserves the existing ProgramSemanticGraph foundation while extending it with continuation-aware analysis capabilities. The approach builds incrementally on proven FCS integration patterns while adding the sophisticated analysis capabilities needed for modern functional language compilation.

The bidirectional zipper provides the navigation and transformation capabilities needed for sophisticated compiler passes, while the property list approach separates analysis concerns from the core PSG structure. This separation enables the compiler to maintain multiple analysis perspectives on the same program structure while keeping the core representation clean and focused.

The in-memory processing approach enables rapid iteration and immediate feedback while maintaining architectural coherence. The HelloWorldDirect validation target ensures that all architectural decisions are grounded in concrete compilation requirements rather than theoretical considerations.

The ultimate goal remains unchanged: enabling F# functional programming patterns to compile efficiently to WebAssembly with continuation semantics preserved throughout the pipeline, delivering on the vision of "dcont all the way down" while maintaining the performance and reliability required for systems programming applications.
