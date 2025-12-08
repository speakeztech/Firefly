# Task Completion Checklist

## Before Making Any Change - MANDATORY

**Do NOT skip these steps. Spawn Task agents prolifically.**

1. [ ] **Spawn Task agents (subagent_type=Explore) in PARALLEL** to investigate:
   - `~/repos/fsharp` - How does FCS handle this?
   - `~/repos/fslang-spec` - What does the spec say?
   - `~/repos/Alloy` - How is this implemented natively?
   - `~/triton-cpu` - What MLIR patterns apply?
   
2. [ ] Read relevant documentation in `/docs/` using Serena's symbolic tools

3. [ ] Synthesize findings from all agents before proceeding

4. [ ] Trace through the full pipeline if it's a bug fix

5. [ ] Understand the full context of the change from agent results

## The Acid Test
Before committing any change, ask:
> "If someone deleted all the comments and looked only at what this code DOES, would they see library-specific logic in MLIR generation?"

If yes, you have violated the layer separation principle. Revert and fix upstream.

## Building and Verification
```bash
# Build the compiler
cd /home/hhh/repos/Firefly/src
dotnet build

# Compile a sample to verify changes
Firefly compile HelloWorld.fidproj

# With verbose output for debugging
Firefly compile HelloWorld.fidproj --verbose

# Keep intermediate files to inspect
Firefly compile HelloWorld.fidproj -k
```

## Pipeline Review Checklist (for non-syntax issues)
1. **Alloy Library Level** - Is the function actually implemented (not a stub)?
2. **FCS Ingestion Level** - Is the symbol being captured correctly?
3. **PSG Construction Level** - Is the function reachable? Are call edges correct?
4. **Nanopass Level** - Are def-use edges created? Operations classified?
5. **Alex/Zipper Level** - Is traversal following PSG structure (not symbol names)?
6. **MLIR/LLVM Level** - Is generated IR valid?

## What NOT To Do
- Don't patch where you see the symptom - find the root cause
- Don't add library-specific logic to MLIR generation
- Don't create stub implementations in Alloy
- Don't compute nanopass data during MLIR generation
- Don't import modules from different pipeline stages

## After Completing Work
1. [ ] Verify the compiler builds: `dotnet build`
2. [ ] Test with sample projects if relevant
3. [ ] Ensure native binary executes correctly if applicable
4. [ ] Consider if documentation updates are needed
