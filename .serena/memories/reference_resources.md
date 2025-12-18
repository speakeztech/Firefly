# Primary Reference Resources

These resources are ESSENTIAL for understanding the project architecture and making correct decisions.

## Language & Compiler References

| Resource | Path | Purpose |
|----------|------|---------|
| **F# Native Compiler** | `~/repos/fsnative` | **PRIMARY**: Active fork of F# compiler for native compilation, FNCS implementation |
| **F# Native Spec** | `~/repos/fsnative-spec` | F# Native language specification and extensions for native targets |
| **F# Compiler Source** | `~/repos/fsharp` | REFERENCE ONLY: Upstream F# compiler for comparison, not active development |
| **F# Language Specification** | `~/repos/fslang-spec` | Standard F# language semantics (reference for deviations in fsnative-spec) |

## MLIR & Code Generation References

| Resource | Path | Purpose |
|----------|------|---------|
| **Triton CPU** | `~/triton-cpu` | Production MLIR implementation, dialect patterns, optimization passes |
| **MLIR Haskell Bindings** | `~/repos/mlir-hs` | Alternative MLIR binding approach, type-safe IR construction |

## Fidelity Ecosystem

| Resource | Path | Purpose |
|----------|------|---------|
| **Alloy** | `~/repos/Alloy` | Native F# library - ACTIVE TARGET. BCL-sympathetic API, native types, extern primitives |
| **XParsec** | `~/repos/XParsec` | Parser combinator library fork - basis for PSG traversal and pattern matching in Alex |
| **BAREWire** | `~/repos/BAREWire` | Binary serialization - FUTURE. Memory-efficient wire protocol |
| **Farscape** | `~/repos/Farscape` | Distributed compute - FUTURE. Native F# distributed processing |

## Documentation

| Resource | Path | Purpose |
|----------|------|---------|
| **Firefly Docs** | `/docs/` | PRIMARY: Architecture, design decisions, PSG, Alex, lessons learned |
| **SpeakEZ Blog** | `~/repos/SpeakEZ/hugo/content/blog` | SECONDARY: Articles explaining design philosophy |

## When to Use Each Resource

- **Encountering F# AST/syntax issues**: Explore `~/repos/fsnative` for FCS implementation (active fork)
- **Comparing against upstream F#**: Use `~/repos/fsharp` only for reference/comparison
- **Type system questions**: Reference `~/repos/fslang-spec` for standard semantics, `~/repos/fsnative-spec` for native extensions
- **Defining new native semantics**: Document in `~/repos/fsnative-spec`
- **FNCS modifications**: Implement in `~/repos/fsnative`
- **MLIR dialect patterns**: Look at `~/triton-cpu` for production examples
- **Native type implementation**: Study `~/repos/Alloy` - this is the pattern library
- **Architectural decisions**: Read `/docs/Architecture_Canonical.md` first
- **Understanding "why"**: Check `~/repos/SpeakEZ/hugo/content/blog` for philosophy

## Key Documentation Files

Essential reading in `/docs/`:
- `Architecture_Canonical.md` - **AUTHORITATIVE**: Two-layer model, extern primitives, anti-patterns
- `PSG_Nanopass_Architecture.md` - Nanopass design, typed tree overlay, SRTP
- `Baker_Architecture.md` - Two-tree zipper, type correlation
- `DCont_Pipeline_Roadmap.md` - Delimited continuations pipeline
- `PSG_architecture.md` - PSG design decisions, node identity, reachability
- `Alex_Architecture_Overview.md` - Alex overview
- `XParsec_PSG_Architecture.md` - XParsec integration with Zipper
- `HelloWorld_Lessons_Learned.md` - Common pitfalls and solutions

## Key SpeakEZ Blog Posts

Architectural philosophy at `~/repos/SpeakEZ/hugo/content/blog`:
- `Baker A Key Ingredient to Firefly.md` - Two-tree zipper, nanopass integration
- `Delimited Continuations Fidelitys Turning Point.md` - DCont as unifying abstraction
- `DCont Inet Duality.md` - Sequential vs parallel CE patterns
- `The Full Frosty Experience.md` - Async with RAII through continuations
- `FSharp Autocomplete Integration.md` - FSAC extension for `.fidproj` support

## Early Integration: FSAC for Design-Time Tooling

See `fsac_integration_plan` memory for details on:
- Extending FSAC to support `.fidproj` files
- VS Code/Ionide and nvim configuration
- Multi-pane compiler development (F# → MLIR → LLVM)
- Farscape + clangd for C/C++ binding development

This enables familiar IntelliSense and error highlighting while targeting native compilation.
