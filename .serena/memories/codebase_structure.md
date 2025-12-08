# Codebase Structure

## Root Directory
```
/home/hhh/repos/Firefly/
├── src/                    # Main compiler source
├── docs/                   # Architecture documentation
├── samples/                # Example projects
├── CLAUDE.md               # AI assistant context
├── README.md               # Project readme
└── LICENSE                 # Dual Apache 2.0 / Commercial
```

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
