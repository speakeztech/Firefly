# Primary Reference Resources

These resources are ESSENTIAL for understanding the project architecture and making correct decisions.

## Language & Compiler References

| Resource | Path | Purpose |
|----------|------|---------|
| **F# Compiler Source** | `~/repos/fsharp` | F# compiler implementation, FSharp.Compiler.Service internals, AST structures |
| **F# Language Specification** | `~/repos/fslang-spec` | Authoritative F# language semantics, type system, evaluation rules |

## MLIR & Code Generation References

| Resource | Path | Purpose |
|----------|------|---------|
| **Triton CPU** | `~/triton-cpu` | Production MLIR implementation, dialect patterns, optimization passes |
| **MLIR Haskell Bindings** | `~/repos/mlir-hs` | Alternative MLIR binding approach, type-safe IR construction |

## Fidelity Ecosystem

| Resource | Path | Purpose |
|----------|------|---------|
| **Alloy** | `~/repos/Alloy` | Native F# library - ACTIVE TARGET. BCL-sympathetic API, native types, extern primitives |
| **BAREWire** | `~/repos/BAREWire` | Binary serialization - FUTURE. Memory-efficient wire protocol |
| **Farscape** | `~/repos/Farscape` | Distributed compute - FUTURE. Native F# distributed processing |

## Documentation

| Resource | Path | Purpose |
|----------|------|---------|
| **Firefly Docs** | `/docs/` | PRIMARY: Architecture, design decisions, PSG, Alex, lessons learned |
| **SpeakEZ Blog** | `~/repos/SpeakEZ/hugo/content/blog` | SECONDARY: Articles explaining design philosophy |

## When to Use Each Resource

- **Encountering F# AST/syntax issues**: Explore `~/repos/fsharp` for FCS implementation details
- **Type system questions**: Reference `~/repos/fslang-spec` for language semantics
- **MLIR dialect patterns**: Look at `~/triton-cpu` for production examples
- **Native type implementation**: Study `~/repos/Alloy` - this is the pattern library
- **Architectural decisions**: Read `/docs/Architecture_Canonical.md` first
- **Understanding "why"**: Check `~/repos/SpeakEZ/hugo/content/blog` for philosophy

## Key Documentation Files

Essential reading in `/docs/`:
- `Architecture_Canonical.md` - **AUTHORITATIVE**: Two-layer model, extern primitives, anti-patterns
- `PSG_architecture.md` - PSG design decisions, node identity, reachability
- `PSG_Nanopass_Architecture.md` - Nanopass design, def-use edges, enrichment
- `Alex_Architecture_Overview.md` - Alex overview
- `XParsec_PSG_Architecture.md` - XParsec integration with Zipper
- `HelloWorld_Lessons_Learned.md` - Common pitfalls and solutions
