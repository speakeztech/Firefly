# Codebase Structure

## Root Directory
```
/home/hhh/repos/Firefly/
├── src/                    # Main compiler source
├── docs/                   # Architecture documentation
├── samples/                # Example projects
├── helpers/                # Synced dependency repos (LOCAL COPIES)
│   ├── Alloy/              # Native F# library (has shadow .fsproj for LSP)
│   ├── Farscape/           # Distributed compute library
│   ├── XParsec/            # Parser combinator library
│   └── sync.sh             # Script to resync from source repos
├── tools/                  # Developer tooling
│   └── serena-mlir-patch/  # MLIR LSP integration for Serena
├── CLAUDE.md               # AI assistant context
├── README.md               # Project readme
└── LICENSE                 # Dual Apache 2.0 / Commercial
```

## Helper Repos (helpers/)

These are LOCAL COPIES of dependency repos, synced for Serena indexing.
Run `./helpers/sync.sh` to update from source repos.

**Alloy** (`helpers/Alloy/src/`) - Native F# standard library
- Core.fs, Math.fs, Memory.fs, Text.fs, Utf8.fs
- Primitives.fs (extern declarations for `__fidelity`)
- Console.fs, Time.fs (I/O using primitives)
- NativeTypes/ (VESTIGIAL - wrapper types to be removed; FNCS provides native semantics for standard F# types)
- Has shadow Alloy.fsproj for F# LSP indexing

**XParsec** (`helpers/XParsec/src/XParsec/`) - Parser combinators
- Types.fs, Combinators.fs, Parsers.fs
- CharParsers.fs, ByteParsers.fs
- ErrorFormatting.fs, OperatorParsing.fs

**Farscape** (`helpers/Farscape/src/`) - Distributed compute (FUTURE)
- Farscape.Core/ (Types, CodeGenerator, CppParser, TypeMapper)
- Farscape.Cli/

## Source Code (/src/)
```
src/
├── Firefly.fsproj          # Main project file
├── Core/                   # Core compiler components
│   ├── FCS/                # F# Compiler Services integration
│   │   ├── Helpers.fs
│   │   ├── ProjectContext.fs
│   │   └── TypedASTAccess.fs
│   ├── PSG/                # Program Semantic Graph
│   │   ├── Types.fs
│   │   ├── Reachability.fs
│   │   ├── TypeIntegration.fs
│   │   ├── Correlation.fs
│   │   ├── SymbolAnalysis.fs
│   │   ├── Builder.fs      # PSG construction
│   │   ├── Builder/        # Sub-modules
│   │   ├── Nanopass/       # PSG transformations
│   │   └── DebugOutput.fs
│   ├── Meta/               # Metadata processing
│   ├── Templates/          # Platform templates
│   ├── Types/              # Core types (Dialects, MLIRTypes)
│   ├── Utilities/          # Utility functions
│   ├── XParsec/            # Parser foundation
│   ├── MLIR/               # MLIR-related code
│   ├── CompilerConfig.fs   # Compiler configuration
│   └── IngestionPipeline.fs # Pipeline orchestration
├── Alex/                   # Code generation layer
│   ├── Bindings/           # Platform-specific MLIR generation
│   │   ├── BindingTypes.fs
│   │   ├── Console/
│   │   └── Time/
│   ├── CodeGeneration/     # Type mapping, MLIR builders
│   ├── Traversal/          # Zipper and XParsec traversal
│   ├── Patterns/           # PSG patterns
│   ├── Transforms/         # Code transformations
│   └── Pipeline/           # Compilation orchestration
└── CLI/                    # Command-line interface
    ├── Commands/           # CompileCommand, VerifyCommand, DoctorCommand
    ├── Configurations/     # ProjectConfig, FidprojLoader
    ├── Diagnostics/        # EnvironmentInfo, ToolChainVerification
    └── Program.fs          # Entry point
```

## Documentation (/docs/)
Key documents:
- `Architecture_Canonical.md` - AUTHORITATIVE architecture reference
- `PSG_architecture.md` - PSG design decisions
- `PSG_Nanopass_Architecture.md` - Nanopass design
- `Alex_Architecture_Overview.md` - Alex overview
- `XParsec_PSG_Architecture.md` - XParsec integration
- `HelloWorld_Lessons_Learned.md` - Common pitfalls

## Samples (/samples/)
```
samples/
├── console/                # Console applications
│   ├── HelloWorld/         # Minimal validation
│   ├── TimeLoop/           # Mutable state, loops
│   └── FidelityHelloWorld/ # Progressive complexity
│       ├── 01_HelloWorldDirect/
│       ├── 02_HelloWorldSaturated/
│       ├── 03_HelloWorldHalfCurried/
│       └── 04_HelloWorldFullCurried/
├── embedded/               # ARM Cortex-M targets
└── sbc/                    # Single-board computers
```
