# Fidelity Tooling Roadmap

## Overview

This document outlines the tooling ecosystem needed to support the Fidelity framework with fsnative (FNCS) as a distinct frontend. It covers editor integration, language server architecture, and the path from .NET-hosted tooling to eventual self-hosting.

## Current F# Tooling Architecture

The existing F# tooling stack:

```
Editor (VSCode/Vim/Emacs)
    ↓
Editor Plugin (Ionide-vscode-fsharp / Ionide-vim)
    ↓ LSP
FsAutoComplete (FSAC)
    ↓
├── FSharp.Compiler.Service (FCS) ← Type checking, parsing
├── Ionide.ProjInfo ← MSBuild project loading
├── FSharpLint ← Linting
├── Fantomas ← Formatting
└── FSharp.Analyzers.SDK ← Custom analyzers
```

### Key Dependencies

| Component | Version | Purpose |
|-----------|---------|---------|
| FSharp.Compiler.Service | >= 43.10.100 | Core compiler services |
| Ionide.ProjInfo | >= 0.74.1 | MSBuild project evaluation |
| Ionide.LanguageServerProtocol | Latest | LSP implementation library |
| FSharp.Analyzers.SDK | 0.34.1 | Custom analyzer framework |

## Fidelity Tooling Architecture

### Target Architecture

```
Editor (VSCode/Vim/Emacs)
    ↓
Editor Plugin (Fidelity-vscode / Fidelity-vim)
    ↓ LSP
FsNativeAutoComplete (FSNAC)
    ↓
├── fsnative (FNCS) ← Native type resolution, SRTP
├── Fidelity.ProjInfo ← .fidproj loading (TOML-based)
├── Native-aware linting (future)
└── Fidelity.Analyzers.SDK ← Memory/ownership analyzers
```

### Bootstrap Strategy: .NET-Hosted First

**Critical insight**: Firefly itself is a .NET CLI tool. This provides a "get out of jail free" card for tooling bootstrap:

```
Phase 1: .NET-hosted tooling
├── FSNAC runs on .NET (like FSAC)
├── Uses fsnative (FNCS) which is .NET-hosted
├── Editor plugins communicate via standard LSP
└── Firefly compiles to native, but tooling runs on .NET

Phase 2: Partial self-hosting (future)
├── Core FSNAC logic compiled via Firefly
├── LSP transport layer remains .NET (or native via platform bindings)
└── Gradual migration of components

Phase 3: Full self-hosting (distant future)
├── FSNAC compiled entirely by Firefly
├── Native binary, no .NET dependency
└── Fidelity tooling dogfoods Fidelity
```

This mirrors the rust-analyzer approach: the tooling doesn't need to be self-hosted to be useful. rust-analyzer is written in Rust but that's orthogonal to its function as an IDE backend.

## Required New Projects

### 1. FsNativeAutoComplete (FSNAC)

**Repository**: `fsnative-autocomplete` (or `fsnac`)

**Purpose**: LSP server for Fidelity/F# Native development

**Architecture**:
```
fsnative-autocomplete/
├── src/
│   ├── FsNativeAutoComplete.Core/     # Core analysis logic
│   │   ├── FNCSIntegration.fs         # fsnative API wrapper
│   │   ├── NativeTypeDisplay.fs       # Type formatting for UI
│   │   ├── MemoryRegionInfo.fs        # Region/access kind display
│   │   └── Diagnostics.fs             # Native-specific errors
│   ├── FsNativeAutoComplete.Lsp/      # LSP handlers
│   │   ├── Server.fs                  # Main LSP server
│   │   ├── Handlers/                  # LSP method handlers
│   │   └── Extensions/                # Fidelity-specific LSP extensions
│   └── FsNativeAutoComplete/          # CLI entry point
├── tests/
└── paket.dependencies
```

**Key differences from FSAC**:

| Aspect | FSAC | FSNAC |
|--------|------|-------|
| Compiler backend | FSharp.Compiler.Service | fsnative (FNCS) |
| Project files | .fsproj (MSBuild) | .fidproj (TOML) |
| Type display | BCL types | Native types |
| String type | `System.String` | `NativeStr` |
| Option type | `'T option` | `'T voption` |
| Diagnostics | FS0001-FS9999 | FS0001-FS7999 + FS8xxx (native) |
| SRTP info | .NET method tables | Alloy witness resolution |

**Leverage points**:
- [Ionide.LanguageServerProtocol](https://github.com/ionide/LanguageServerProtocol) - Use directly, no changes needed
- FSAC structure - Follow same patterns for LSP handlers
- fsnative APIs - Consume `FNCSPublicAPI` stability layer

### 2. Fidelity.ProjInfo

**Repository**: Part of `fsnative-autocomplete` or separate

**Purpose**: Parse and evaluate `.fidproj` files

**This is dramatically simpler than Ionide.ProjInfo**:

```fsharp
// .fidproj is TOML - trivial to parse
[package]
name = "my_app"
version = "1.0.0"

[compilation]
memory_model = "stack_only"
target = "native"

[dependencies]
alloy = { path = "/path/to/Alloy/src" }

[build]
sources = ["Main.fs"]
output = "my_app"
output_kind = "console"
```

**No MSBuild complexity**:
- No SDK resolution
- No NuGet package restoration
- No target framework negotiation
- No props/targets evaluation
- Source files listed explicitly

**Implementation**:
```fsharp
module Fidelity.ProjInfo

open Tomlyn  // or similar TOML library

type FidProject = {
    Name: string
    Version: string option
    MemoryModel: MemoryModel
    Target: CompilationTarget
    Dependencies: Map<string, DependencySpec>
    Sources: string list
    Output: string
    OutputKind: OutputKind
}

let parse (path: string) : Result<FidProject, ParseError> =
    // Simple TOML parsing - no MSBuild evaluation needed
    ...
```

### 3. Fidelity-vscode

**Repository**: `fidelity-vscode`

**Purpose**: VS Code extension for Fidelity development

**Structure**:
```
fidelity-vscode/
├── src/
│   ├── extension.ts           # Entry point
│   ├── client.ts              # LSP client setup
│   ├── commands/              # Fidelity-specific commands
│   │   ├── compile.ts         # Invoke Firefly
│   │   └── run.ts             # Run native binary
│   └── views/                 # Custom UI elements
│       ├── memoryRegions.ts   # Memory region visualization
│       └── platformBindings.ts
├── syntaxes/                  # Can reuse ionide-fsgrammar
├── package.json
└── tsconfig.json
```

**Key features**:
- LSP client connecting to FSNAC
- Native type display in hover/completion
- Memory region annotations
- Firefly build integration
- Native binary execution

**Leverage points**:
- [ionide-fsgrammar](https://github.com/ionide/ionide-fsgrammar) - Reuse directly (syntax unchanged)
- ionide-vscode-fsharp patterns - Similar extension structure

### 4. Fidelity-vim

**Repository**: `fidelity-vim`

**Purpose**: Vim/Neovim plugin for Fidelity development

**Structure**:
```
fidelity-vim/
├── autoload/
│   └── fidelity.vim
├── ftplugin/
│   └── fsharp.vim           # F# filetype settings
├── lua/                     # Neovim-specific (optional)
│   └── fidelity/
│       └── init.lua
├── plugin/
│   └── fidelity.vim
└── README.md
```

**Integration approach**:
- Configure nvim-lspconfig to use FSNAC
- Minimal plugin - mostly LSP configuration
- Reuse existing F# syntax highlighting

**Simpler than Ionide-vim** because:
- No need for custom FSI integration (native binaries)
- LSP handles most functionality
- Neovim's built-in LSP client does heavy lifting

### 5. Fidelity.Analyzers.SDK (Future)

**Repository**: `fidelity-analyzers-sdk`

**Purpose**: Framework for custom Fidelity-specific analyzers

**Native-specific analyzers**:
- Memory region misuse detection
- Access kind violations
- Ownership/borrowing errors (when implemented)
- Platform binding validation
- Unsafe pointer usage warnings

**Deferred until**:
- FNCS API is stable
- Core tooling (FSNAC) is functional
- Real-world usage patterns emerge

## Reusable Components

### Can Use Directly

| Component | Repository | Reason |
|-----------|------------|--------|
| **LanguageServerProtocol** | ionide/LanguageServerProtocol | Generic LSP infrastructure |
| **ionide-fsgrammar** | ionide/ionide-fsgrammar | F# syntax unchanged |
| **Fantomas** | fsprojects/fantomas | Formatting is syntax-level |

### Need Adaptation

| Component | Original | Changes Needed |
|-----------|----------|----------------|
| **FSAC patterns** | ionide/FsAutoComplete | Replace FCS with FNCS |
| **FSharpLint rules** | fsprojects/FSharpLint | Native-specific rules |
| **Analyzers SDK** | ionide/FSharp.Analyzers.SDK | FNCS typed trees |

## Implementation Phases

### Phase 1: Minimal Viable FSNAC

**Goal**: Basic LSP server that provides hover and go-to-definition

**Tasks**:
1. Create fsnative-autocomplete repository
2. Add Ionide.LanguageServerProtocol dependency
3. Implement minimal FNCS integration
4. Basic .fidproj parsing (hardcoded paths acceptable)
5. Hover provider showing native types
6. Go-to-definition for local symbols

**Validation**:
- Test with VS Code using generic LSP extension
- Hover over `let x = "hello"` shows `NativeStr`

### Phase 2: Editor Integration

**Goal**: Dedicated VS Code extension with native type awareness

**Tasks**:
1. Create fidelity-vscode repository
2. LSP client configuration
3. Syntax highlighting (reuse fsgrammar)
4. Status bar integration
5. Basic Firefly build command

**Validation**:
- Full editing experience in VS Code
- Compile via extension command

### Phase 3: Rich Features

**Goal**: Full-featured development environment

**Tasks**:
1. Complete Fidelity.ProjInfo implementation
2. Memory region display in hover
3. SRTP resolution information
4. Go-to-definition across Alloy sources
5. Find all references
6. Symbol rename

**Validation**:
- Navigate through FidelityHelloWorld samples
- Rename symbol updates all references

### Phase 4: Vim/Neovim Support

**Goal**: First-class Vim/Neovim experience

**Tasks**:
1. Create fidelity-vim repository
2. nvim-lspconfig integration
3. Vim 8+ ALE/CoC integration
4. Documentation and examples

**Validation**:
- Full LSP features in Neovim
- Completion, hover, diagnostics working

### Phase 5: Advanced Tooling

**Goal**: Production-ready tooling ecosystem

**Tasks**:
1. Fidelity.Analyzers.SDK
2. Custom analyzers for memory safety
3. Integration with Firefly diagnostics
4. Performance optimization
5. Caching and incremental analysis

**Validation**:
- Analyzers catch memory region violations
- Responsive on large codebases

## Self-Hosting Considerations

### What Blocks Self-Hosting

1. **FNCS itself** - Currently FCS-based, needs .NET
2. **LSP transport** - JSON-RPC over stdio, needs runtime
3. **File I/O** - Project loading, source reading
4. **Process management** - Editor spawns LSP server

### Self-Hosting Roadmap

```
Current: Everything .NET-hosted
    ↓
Step 1: FNCS core logic native-compiled
    - Type resolution
    - SRTP solving
    - Diagnostic generation
    ↓
Step 2: Analysis engine native-compiled
    - Symbol lookup
    - Reference finding
    - Completion generation
    ↓
Step 3: LSP layer native-compiled
    - JSON-RPC via Platform.Bindings
    - Stdio via Platform.Bindings
    ↓
Step 4: Full self-hosting
    - FSNAC is a Fidelity-compiled native binary
    - Dogfooding complete
```

### Why Self-Hosting Matters

1. **Dogfooding**: Using Fidelity to build Fidelity tools validates the platform
2. **Performance**: Native tooling can be faster than .NET-hosted
3. **Distribution**: Single binary, no .NET SDK required
4. **Credibility**: Demonstrates Fidelity's capability for real-world tools

### Why Self-Hosting Can Wait

1. **Tooling needs to exist first** - Can't self-host what doesn't exist
2. **.NET hosting is fine** - rust-analyzer runs on Rust, but that's not why it's good
3. **Iteration speed** - .NET development is faster for tooling iteration
4. **Ecosystem integration** - .NET tooling integrates with existing F# ecosystem

## Related Resources

| Resource | Purpose |
|----------|---------|
| [rust-analyzer](https://github.com/rust-lang/rust-analyzer) | Reference for IDE-focused compiler frontend |
| [Ionide.LanguageServerProtocol](https://github.com/ionide/LanguageServerProtocol) | LSP library to use |
| [FsAutoComplete](https://github.com/ionide/FsAutoComplete) | Patterns to follow |
| [MLIR LSP](https://mlir.llvm.org/docs/Tools/MLIRLSP/) | Example of compiler-integrated LSP |

## Summary

The Fidelity tooling ecosystem requires:

| Project | Priority | Complexity | Self-Hosting Phase |
|---------|----------|------------|-------------------|
| **fsnative-autocomplete** | Critical | High | Phase 3 |
| **Fidelity.ProjInfo** | Critical | Low | Phase 2 |
| **fidelity-vscode** | High | Medium | N/A (TypeScript) |
| **fidelity-vim** | Medium | Low | N/A (VimScript) |
| **Fidelity.Analyzers.SDK** | Future | Medium | Phase 4 |

The key insight is that .NET-hosted tooling is the correct starting point. Self-hosting is a long-term goal that validates the platform, but functional tooling that helps developers today is the priority.

---

*Tooling that works today. Self-hosting that proves tomorrow.*
