# PSG Function Tracing

The objective is to enable interactive developer tooling where a right-click context menu on "use buffer" can generate a complete, semantically meaningful trace through the dependency chain. This represents a fundamental shift from batch validation to real-time developer assistance.

## Developer Experience Requirements

The PSG must maintain sufficient semantic depth to power interactive code exploration tools. When a developer selects "use buffer" and requests a trace, the system needs to reconstruct the complete call path from that specific syntax element through the stackBuffer function call to the Memory module implementation. This reconstruction must present results using the actual function names, file locations, and semantic relationships that correspond directly to the developer's source code.

The trace functionality requires both symbolic and semantic graph elements working in coordination. The symbolic elements preserve the syntactic structure that corresponds to what developers see in their code editor, while the semantic elements maintain the type information and symbol correlations that enable meaningful dependency analysis. Without this dual representation, the trace results would either lack sufficient detail for developer understanding or become disconnected from the recognizable code structure.

## Graph Structure Requirements for Interactive Tracing

This developer experience places specific demands on PSG architecture. The graph must preserve bidirectional relationships between syntax elements and their corresponding semantic symbols, enabling traversal from any source code position to its complete dependency chain. The correlation system must maintain stable mappings between syntax tree positions and FSharp symbol objects to support consistent trace results across compilation sessions.

The ChildrenState enhancement supports this objective by providing deterministic graph structure that enables reliable traversal algorithms. When the IDE requests a trace from a specific syntax position, the graph traversal logic can follow explicit parent-child relationships and edge connections with confidence that the results accurately reflect the dependency relationships present in the source code.

## Implementation Strategy for IDE Integration

The native DuckDB PSG architecture becomes essential for responsive interactive tracing. The current JSON artifact approach cannot support real-time trace requests within acceptable response times for developer tooling. Native graph storage enables the system to respond to trace requests with the immediacy that developers expect from IDE features.

The trace results must be formatted to support developer mental models of code organization. Rather than presenting internal graph node identifiers, the trace should display actual function names, module paths, and source code line numbers that enable developers to navigate directly to relevant implementations. This presentation layer transforms the graph traversal results into actionable developer information that supports confident code understanding and modification decisions.

This interactive tracing capability positions Firefly as a comprehensive development environment that provides transparency into compilation decisions rather than simply producing native executables. The ability to explore dependency relationships interactively enables developers to validate that their code will compile to the expected native implementations and understand the complete scope of dependencies included in the final executable.