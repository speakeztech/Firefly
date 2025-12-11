# External Reference Resources

## CRITICAL: Spawn Task Agents Prolifically

**Before making any non-trivial change to Firefly, you MUST spawn Task agents (subagent_type=Explore) to investigate the external reference resources.** This is not optional - it is the primary mechanism for maintaining referential integrity with the guiding resources.

The external repos are NOT part of Serena's active project, so symbolic tools won't reach them. Task agents are the ONLY way to explore them.

---

These resources are ESSENTIAL for understanding the project architecture and making correct decisions.
**Use agents prolifically to explore these resources before making changes.**

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
| **Alloy** | `helpers/Alloy/` | **LOCAL COPY** - Native F# library. Use Serena's symbolic tools directly! |
| **XParsec** | `helpers/XParsec/` | **LOCAL COPY** - Parser combinators. Use Serena's symbolic tools directly! |
| **Farscape** | `helpers/Farscape/` | **LOCAL COPY** - Distributed compute (FUTURE). Use Serena's symbolic tools! |
| **BAREWire** | `~/repos/BAREWire` | Binary serialization - FUTURE. (Not yet synced to helpers/) |

**NOTE:** Alloy, XParsec, and Farscape are now LOCAL COPIES in `helpers/`.
Serena can use symbolic tools (find_symbol, get_symbols_overview) on these directly!
Run `./helpers/sync.sh` to update from source repos.

## Secondary Documentation

| Resource | Path | Purpose |
|----------|------|---------|
| **Firefly Docs** | `/docs/` | PRIMARY: Architecture, design decisions, PSG, Alex, lessons learned |
| **SpeakEZ Blog** | `~/repos/SpeakEZ/hugo/content/blog` | SECONDARY: Articles explaining design philosophy, architectural thinking |

## When to Use Each Resource

### F# AST/Syntax Issues
- Explore `~/repos/fsharp` for FCS implementation details
- Look at how FCS represents constructs internally
- Example agent task: "Search ~/repos/fsharp for how FCS represents extern declarations"

### Type System Questions
- Reference `~/repos/fslang-spec` for language semantics
- Example agent task: "Look up the F# spec section on statically resolved type parameters"

### MLIR Dialect Patterns
- Look at `~/triton-cpu` for production examples
- Study how real MLIR dialects are structured
- Example agent task: "Find examples of syscall bindings in ~/triton-cpu"

### Native Type Implementation
- Study `~/repos/Alloy` - this is the pattern library
- Understand how native types map to F# constructs
- Example agent task: "Explore how Alloy handles native string encoding in Text.UTF8"

### Architectural Decisions
- Read `/docs/Architecture_Canonical.md` first
- This is the authoritative reference for the two-layer model

### Understanding "Why"
- Check `~/repos/SpeakEZ/hugo/content/blog` for philosophy
- Contains articles explaining design decisions and thinking

## Agent Usage Protocol - MANDATORY

**This protocol is MANDATORY, not advisory. Do not skip these steps.**

Before making any non-trivial change:

1. **Spawn Task agents (subagent_type=Explore) in PARALLEL** to investigate:
   - The Firefly codebase area being modified
   - `~/repos/fsharp` - How does FCS handle this construct?
   - `~/repos/Alloy` - How is similar functionality implemented natively?
   - `~/triton-cpu` - What MLIR patterns apply?
   - `~/repos/fslang-spec` - What does the F# specification say?

2. **Read documentation** in `/docs/` using Serena's symbolic tools

3. **Synthesize understanding** from all agent results before proposing changes

4. **If agents return relevant findings**, incorporate them into the solution design

**Spawn multiple agents in a single message for parallel exploration.** Do not serialize agent calls when they are independent.

## Example Agent Tasks

- "Explore how Alloy handles native string encoding in Text.UTF8"
- "Search ~/repos/fsharp for how FCS represents extern declarations"
- "Find examples of syscall bindings in ~/triton-cpu"
- "Look up the F# spec section on statically resolved type parameters"
- "Search ~/repos/fsharp for FSharpMemberOrFunctionOrValue handling"
- "Explore MLIR dialect registration patterns in ~/triton-cpu"
