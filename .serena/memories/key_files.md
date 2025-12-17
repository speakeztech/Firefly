# Key Files Reference

## Main Project Files

| File | Purpose |
|------|---------|
| `/src/Firefly.fsproj` | Main compiler project |
| `/src/Core/IngestionPipeline.fs` | Pipeline orchestration |
| `/src/Core/CompilerConfig.fs` | Compiler configuration |

## FCS Integration (F# Compiler Services)

| File | Purpose |
|------|---------|
| `/src/Core/FCS/Helpers.fs` | FCS utility functions |
| `/src/Core/FCS/ProjectContext.fs` | Project context management |
| `/src/Core/FCS/TypedASTAccess.fs` | Typed AST access |

## PSG (Program Semantic Graph)

| File | Purpose |
|------|---------|
| `/src/Core/PSG/Types.fs` | PSG type definitions |
| `/src/Core/PSG/Builder.fs` | PSG construction |
| `/src/Core/PSG/Builder/*.fs` | Builder sub-modules |
| `/src/Core/PSG/Reachability.fs` | Dead code elimination |
| `/src/Core/PSG/TypeIntegration.fs` | Type information integration |
| `/src/Core/PSG/Correlation.fs` | Symbol correlation |
| `/src/Core/PSG/SymbolAnalysis.fs` | Symbol analysis |
| `/src/Core/PSG/DebugOutput.fs` | Debug output generation |

## PSG Nanopasses

| File | Purpose |
|------|---------|
| `/src/Core/PSG/Nanopass/IntermediateEmission.fs` | Emit intermediate files |
| `/src/Core/PSG/Nanopass/FlattenApplications.fs` | Flatten function applications |
| `/src/Core/PSG/Nanopass/ReducePipeOperators.fs` | Reduce pipe operators |
| `/src/Core/PSG/Nanopass/DefUseEdges.fs` | Definition-use edges |
| `/src/Core/PSG/Nanopass/ParameterAnnotation.fs` | Parameter annotations |
| `/src/Core/PSG/Nanopass/ClassifyOperations.fs` | Operation classification |
| `/src/Core/PSG/Nanopass/LowerInterpolatedStrings.fs` | Lower interpolated strings |

## Alex (Code Generation)

| File | Purpose |
|------|---------|
| `/src/Alex/Generation/Transfer.fs` | **MLIR generation** - the ONLY MLIR gen file |
| `/src/Alex/Traversal/PSGZipper.fs` | Zipper traversal engine |
| `/src/Alex/Traversal/MLIRZipper.fs` | MLIR output composition |
| `/src/Alex/Traversal/PSGXParsec.fs` | XParsec pattern matching |
| `/src/Alex/Bindings/BindingTypes.fs` | Binding type definitions |
| `/src/Alex/Bindings/Console/ConsoleBindings.fs` | Console I/O bindings |
| `/src/Alex/Bindings/Time/TimeBindings.fs` | Time-related bindings |
| `/src/Alex/CodeGeneration/MLIRBuilder.fs` | MLIR construction |
| `/src/Alex/CodeGeneration/TypeMapping.fs` | F# to MLIR type mapping |
| `/src/Alex/Patterns/PSGPatterns.fs` | PSG pattern definitions |
| `/src/Alex/Pipeline/CompilationTypes.fs` | Compilation type definitions |
| `/src/Alex/Pipeline/CompilationOrchestrator.fs` | Full compilation orchestration |

## CLI (Paper-thin Wrapper)

| File | Purpose |
|------|---------|
| `/src/CLI/Program.fs` | Entry point - thin wrapper calling orchestrator |
| `/src/CLI/Commands/VerifyCommand.fs` | Verify toolchain command |
| `/src/CLI/Commands/DoctorCommand.fs` | Doctor diagnostics command |

## Project Configuration

| File | Purpose |
|------|---------|
| `/src/Core/FCS/ProjectConfig.fs` | Project configuration parsing |
| `/src/Core/FCS/FidprojLoader.fs` | .fidproj file loader |
| `/src/Core/Toolchain.fs` | External tool calls (mlir-opt, llc, clang) - isolated for self-hosting |

## Documentation (Essential Reading)

| Document | Purpose |
|----------|---------|
| `/docs/Architecture_Canonical.md` | **AUTHORITATIVE**: Two-layer model, extern primitives, anti-patterns |
| `/docs/PSG_architecture.md` | PSG design decisions, node identity, reachability |
| `/docs/PSG_Nanopass_Architecture.md` | Nanopass design, def-use edges, enrichment |
| `/docs/Alex_Architecture_Overview.md` | Alex overview (references canonical doc) |
| `/docs/XParsec_PSG_Architecture.md` | XParsec integration with Zipper |
| `/docs/HelloWorld_Lessons_Learned.md` | Common pitfalls and solutions |
| `/docs/FCS_Ingestion_Architecture.md` | FCS integration details |

## Sample Projects

| Sample | Purpose |
|--------|---------|
| `/samples/console/HelloWorld/` | Minimal validation sample |
| `/samples/console/TimeLoop/` | Mutable state, while loops, DateTime, Sleep |
| `/samples/console/FidelityHelloWorld/01_HelloWorldDirect/` | Direct module calls |
| `/samples/console/FidelityHelloWorld/02_HelloWorldSaturated/` | Saturated function calls |
| `/samples/console/FidelityHelloWorld/03_HelloWorldHalfCurried/` | Pipe operators, partial application |
| `/samples/console/FidelityHelloWorld/04_HelloWorldFullCurried/` | Full currying, Result.map, lambdas |
