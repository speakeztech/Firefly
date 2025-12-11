# Serena MLIR/LLVM LSP Patch

This patch adds MLIR, TableGen, PDLL, and LLVM IR language support to Serena's `solidlsp` package.

## Prerequisites

### MLIR Family (Official LLVM tools)

```bash
# Arch Linux
sudo pacman -S mlir

# This provides:
# - mlir-lsp-server      (.mlir files)
# - tblgen-lsp-server    (.td files)
# - mlir-pdll-lsp-server (.pdll files)
```

### LLVM IR (Third-party, experimental)

```bash
# Download llvm-ir-lsp binary
curl -L -o ~/.local/bin/llvm-ir-lsp \
  'https://github.com/indoorvivants/llvm-ir-lsp/releases/download/v0.0.3/LLVM_LanguageServer-x86_64-pc-linux'
chmod +x ~/.local/bin/llvm-ir-lsp
```

### Verify installation

```bash
which mlir-lsp-server tblgen-lsp-server mlir-pdll-lsp-server
ls -la ~/.local/bin/llvm-ir-lsp
```

## Applying the Patch

### Dry Run (see what would change)
```bash
python apply_patch.py --dry-run
```

### Apply the Patch
```bash
python apply_patch.py
```

### Verify Installation
```bash
python apply_patch.py --verify-only
```

## What Gets Patched

1. **Language server files** are copied to `solidlsp/language_servers/`:
   - `mlir_language_server.py`
   - `tablegen_language_server.py`
   - `pdll_language_server.py`
   - `llvmir_language_server.py`

2. **`ls_config.py`** is patched to add:
   - `Language.MLIR`, `Language.TABLEGEN`, `Language.PDLL`, `Language.LLVM_IR` enum entries
   - File extension matchers (`*.mlir`, `*.td`, `*.pdll`, `*.ll`)
   - Language server class mappings

## Using with Serena

After applying the patch, update your project's `.serena/project.yml`:

```yaml
name: MyProject
languages:
  - fsharp
  - mlir
  - llvm_ir
```

Or for single-language projects:
```yaml
name: MyProject
language: mlir  # or: tablegen, pdll, llvm_ir
```

## Supported Features

### MLIR Family (Official)
- Code completion
- Go to definition
- Find references
- Hover information
- Diagnostics (syntax errors, etc.)
- Document symbols

### LLVM IR (Experimental)
- Document symbols
- Go to definition
- Hover information

Note: LLVM IR support is marked as experimental in Serena.

## Reverting the Patch

A backup of `ls_config.py` is created at `ls_config.py.bak`. To revert:

```bash
cd /path/to/solidlsp
mv ls_config.py.bak ls_config.py
rm language_servers/mlir_language_server.py
rm language_servers/tablegen_language_server.py
rm language_servers/pdll_language_server.py
rm language_servers/llvmir_language_server.py
```

## References

- [MLIR LSP Documentation](https://mlir.llvm.org/docs/Tools/MLIRLSP/)
- [VSCode MLIR Extension](https://github.com/llvm/vscode-mlir)
- [llvm-ir-lsp (Third-party)](https://github.com/indoorvivants/llvm-ir-lsp)
- [Serena Documentation](https://github.com/oraios/serena)
