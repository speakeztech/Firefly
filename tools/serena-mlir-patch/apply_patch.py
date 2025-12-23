#!/usr/bin/env python3
"""
Patch script to add MLIR, TableGen, PDLL, and LLVM IR language support to Serena's solidlsp package.

This script:
1. Copies the language server implementations to solidlsp/language_servers/
2. Patches ls_config.py to add the new Language enum entries and mappings

Usage:
    python apply_patch.py [--dry-run] [--serena-path PATH]

Options:
    --dry-run       Show what would be done without making changes
    --serena-path   Path to serena-agent installation (auto-detected if not specified)
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def find_serena_path() -> Path | None:
    """Find the solidlsp package path within the serena installation."""
    # Try to find via uv tools
    uv_tools_path = Path.home() / ".local/share/uv/tools/serena-agent"
    if uv_tools_path.exists():
        # Find the Python version directory
        lib_path = uv_tools_path / "lib"
        if lib_path.exists():
            for item in lib_path.iterdir():
                if item.name.startswith("python"):
                    solidlsp_path = item / "site-packages" / "solidlsp"
                    if solidlsp_path.exists():
                        return solidlsp_path

    # Try to find via pip/serena command
    try:
        result = subprocess.run(
            ["python3", "-c", "import solidlsp; print(solidlsp.__file__)"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            solidlsp_init = Path(result.stdout.strip())
            return solidlsp_init.parent
    except Exception:
        pass

    return None


def get_patch_dir() -> Path:
    """Get the directory containing this script and the patch files."""
    return Path(__file__).parent


def copy_language_servers(solidlsp_path: Path, dry_run: bool = False) -> list[str]:
    """Copy the language server files to solidlsp/language_servers/."""
    patch_dir = get_patch_dir()
    ls_dir = solidlsp_path / "language_servers"

    files_to_copy = [
        "mlir_language_server.py",
        "tablegen_language_server.py",
        "pdll_language_server.py",
        "llvmir_language_server.py",
    ]

    copied = []
    for filename in files_to_copy:
        src = patch_dir / filename
        dst = ls_dir / filename

        if not src.exists():
            print(f"  WARNING: Source file not found: {src}")
            continue

        if dry_run:
            print(f"  Would copy: {src} -> {dst}")
        else:
            shutil.copy2(src, dst)
            print(f"  Copied: {filename}")
        copied.append(filename)

    return copied


def patch_ls_config(solidlsp_path: Path, dry_run: bool = False) -> bool:
    """Patch ls_config.py to add new language entries."""
    ls_config_path = solidlsp_path / "ls_config.py"

    if not ls_config_path.exists():
        print(f"  ERROR: ls_config.py not found at {ls_config_path}")
        return False

    content = ls_config_path.read_text()
    original_content = content

    # Check if already patched
    if "MLIR = " in content:
        print("  ls_config.py already contains MLIR entry, checking for LLVM IR...")
        enum_patched = True
    else:
        enum_patched = False

    # Check if LLVM IR already added
    if "LLVM_IR = " in content:
        print("  ls_config.py already contains LLVM_IR entry")
        llvmir_patched = True
    else:
        llvmir_patched = False

    # 1. Add Language enum entries (after YAML = "yaml")
    if not enum_patched:
        # Find the YAML entry and add new entries after it
        yaml_pattern = r'(YAML = "yaml"\s*\n\s*"""[^"]*""")'
        yaml_match = re.search(yaml_pattern, content)

        if yaml_match:
            new_entries = '''
    MLIR = "mlir"
    """MLIR language server for .mlir files.
    Provides code completion, go-to-definition, diagnostics for MLIR IR.
    """
    TABLEGEN = "tablegen"
    """TableGen language server for .td files.
    Provides code completion, go-to-definition, diagnostics for TableGen.
    """
    PDLL = "pdll"
    """PDLL (Pattern Description Language) server for .pdll files.
    Provides code completion, go-to-definition for MLIR pattern matching.
    """
    LLVM_IR = "llvm_ir"
    """LLVM IR language server for .ll files (experimental).
    Provides document symbols, go-to-definition, hover for LLVM IR.
    Uses third-party llvm-ir-lsp from github.com/indoorvivants/llvm-ir-lsp
    """'''
            insert_pos = yaml_match.end()
            content = content[:insert_pos] + new_entries + content[insert_pos:]
            print("  Added MLIR, TABLEGEN, PDLL, LLVM_IR to Language enum")
        else:
            print("  WARNING: Could not find YAML entry in Language enum")
    elif not llvmir_patched:
        # Add just LLVM_IR after PDLL
        pdll_pattern = r'(PDLL = "pdll"\s*\n\s*"""[^"]*""")'
        pdll_match = re.search(pdll_pattern, content)

        if pdll_match:
            new_entry = '''
    LLVM_IR = "llvm_ir"
    """LLVM IR language server for .ll files (experimental).
    Provides document symbols, go-to-definition, hover for LLVM IR.
    Uses third-party llvm-ir-lsp from github.com/indoorvivants/llvm-ir-lsp
    """'''
            insert_pos = pdll_match.end()
            content = content[:insert_pos] + new_entry + content[insert_pos:]
            print("  Added LLVM_IR to Language enum")
        else:
            print("  WARNING: Could not find PDLL entry to add LLVM_IR after")

    # 2. Add to is_experimental() check - LLVM_IR is experimental
    if "LLVM_IR" in content and "self.LLVM_IR" not in content.split("is_experimental")[1].split("def ")[0]:
        # Find is_experimental method and add LLVM_IR to the set
        # Match the closing brace of the set to insert before it
        experimental_pattern = r'(return self in \{[^}]+)(,?\s*\})'
        experimental_match = re.search(experimental_pattern, content)
        if experimental_match:
            # Check if LLVM_IR is already there
            if "self.LLVM_IR" not in experimental_match.group(1):
                # Insert LLVM_IR before the closing brace, preserving formatting
                set_content = experimental_match.group(1)
                closing = experimental_match.group(2)
                # Add comma after last element if needed, then new element
                new_set = set_content.rstrip(',') + ",\n            self.LLVM_IR,\n        }"
                content = content[:experimental_match.start()] + new_set + content[experimental_match.end():]
                print("  Added LLVM_IR to experimental languages")

    # 3. Add file matchers in get_source_fn_matcher()
    if "case self.MLIR:" not in content:
        # Find the last case in get_source_fn_matcher and add before the default case
        haskell_pattern = r'(case self\.HASKELL:\s*\n\s*return FilenameMatcher\("[^"]*"[^)]*\))'
        haskell_match = re.search(haskell_pattern, content)

        if haskell_match:
            new_matchers = '''
            case self.MLIR:
                return FilenameMatcher("*.mlir")
            case self.TABLEGEN:
                return FilenameMatcher("*.td")
            case self.PDLL:
                return FilenameMatcher("*.pdll")
            case self.LLVM_IR:
                return FilenameMatcher("*.ll")'''
            insert_pos = haskell_match.end()
            content = content[:insert_pos] + new_matchers + content[insert_pos:]
            print("  Added file matchers for MLIR, TABLEGEN, PDLL, LLVM_IR")
        else:
            print("  WARNING: Could not find HASKELL matcher entry")
    elif "case self.LLVM_IR:" not in content:
        # Add just LLVM_IR matcher after PDLL
        pdll_matcher_pattern = r'(case self\.PDLL:\s*\n\s*return FilenameMatcher\("[^"]*"\))'
        pdll_matcher_match = re.search(pdll_matcher_pattern, content)
        if pdll_matcher_match:
            new_matcher = '''
            case self.LLVM_IR:
                return FilenameMatcher("*.ll")'''
            insert_pos = pdll_matcher_match.end()
            content = content[:insert_pos] + new_matcher + content[insert_pos:]
            print("  Added file matcher for LLVM_IR")
    else:
        print("  File matchers already present, skipping")

    # 4. Add to get_ls_class() method
    if "case self.MLIR:" not in content or "MLIRLanguageServer" not in content:
        # Find the last case in get_ls_class and add before the default case
        fsharp_class_pattern = r'(case self\.FSHARP:\s*\n\s*from solidlsp\.language_servers\.fsharp_language_server import FSharpLanguageServer\s*\n\s*return FSharpLanguageServer)'
        fsharp_match = re.search(fsharp_class_pattern, content)

        if fsharp_match:
            new_classes = '''
            case self.MLIR:
                from solidlsp.language_servers.mlir_language_server import MLIRLanguageServer

                return MLIRLanguageServer
            case self.TABLEGEN:
                from solidlsp.language_servers.tablegen_language_server import TableGenLanguageServer

                return TableGenLanguageServer
            case self.PDLL:
                from solidlsp.language_servers.pdll_language_server import PDLLLanguageServer

                return PDLLLanguageServer
            case self.LLVM_IR:
                from solidlsp.language_servers.llvmir_language_server import LLVMIRLanguageServer

                return LLVMIRLanguageServer'''
            insert_pos = fsharp_match.end()
            content = content[:insert_pos] + new_classes + content[insert_pos:]
            print("  Added language server class mappings")
        else:
            print("  WARNING: Could not find FSHARP class entry in get_ls_class()")
    elif "case self.LLVM_IR:" not in content or "LLVMIRLanguageServer" not in content:
        # Add just LLVM_IR class after PDLL
        pdll_class_pattern = r'(case self\.PDLL:\s*\n\s*from solidlsp\.language_servers\.pdll_language_server import PDLLLanguageServer\s*\n\s*return PDLLLanguageServer)'
        pdll_class_match = re.search(pdll_class_pattern, content)
        if pdll_class_match:
            new_class = '''
            case self.LLVM_IR:
                from solidlsp.language_servers.llvmir_language_server import LLVMIRLanguageServer

                return LLVMIRLanguageServer'''
            insert_pos = pdll_class_match.end()
            content = content[:insert_pos] + new_class + content[insert_pos:]
            print("  Added LLVM_IR language server class mapping")
    else:
        print("  Language server class mappings already present, skipping")

    # Write the patched content
    if content != original_content:
        if dry_run:
            print(f"  Would patch: {ls_config_path}")
            # Optionally show diff
            print("\n  --- Changes preview (first 100 lines of diff) ---")
            import difflib
            diff = difflib.unified_diff(
                original_content.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile="ls_config.py.orig",
                tofile="ls_config.py",
            )
            for i, line in enumerate(diff):
                if i >= 100:
                    print("  ... (truncated)")
                    break
                print(f"  {line}", end="")
        else:
            # Backup original
            backup_path = ls_config_path.with_suffix(".py.bak")
            shutil.copy2(ls_config_path, backup_path)
            print(f"  Backed up original to: {backup_path}")

            # Write patched version
            ls_config_path.write_text(content)
            print(f"  Patched: {ls_config_path}")
        return True
    else:
        print("  No changes needed to ls_config.py")
        return True


def verify_installation(solidlsp_path: Path) -> bool:
    """Verify that the patch was applied correctly."""
    print("\nVerifying installation...")

    # Check language server files exist
    ls_dir = solidlsp_path / "language_servers"
    files = [
        "mlir_language_server.py",
        "tablegen_language_server.py",
        "pdll_language_server.py",
        "llvmir_language_server.py",
    ]

    all_exist = True
    for filename in files:
        filepath = ls_dir / filename
        if not filepath.exists():
            print(f"  ✗ {filename} not found")
            all_exist = False
        else:
            print(f"  ✓ {filename} exists")

    if not all_exist:
        print("  WARNING: Some language server files are missing")

    # Try to import the modules
    try:
        sys.path.insert(0, str(solidlsp_path.parent))

        # Force reload in case of cached imports
        import importlib
        if "solidlsp.ls_config" in sys.modules:
            importlib.reload(sys.modules["solidlsp.ls_config"])

        from solidlsp.ls_config import Language

        # Check enum values exist
        languages_to_check = ["MLIR", "TABLEGEN", "PDLL", "LLVM_IR"]
        for lang in languages_to_check:
            if hasattr(Language, lang):
                print(f"  ✓ Language.{lang} present")
            else:
                print(f"  ✗ Language.{lang} missing")

        # Check file matchers work
        if hasattr(Language, "MLIR"):
            mlir_matcher = Language.MLIR.get_source_fn_matcher()
            assert mlir_matcher.is_relevant_filename("test.mlir"), "MLIR matcher failed"
        if hasattr(Language, "LLVM_IR"):
            llvmir_matcher = Language.LLVM_IR.get_source_fn_matcher()
            assert llvmir_matcher.is_relevant_filename("test.ll"), "LLVM_IR matcher failed"
        print("  ✓ File matchers work")

        # Check LS class imports work
        if hasattr(Language, "MLIR"):
            mlir_class = Language.MLIR.get_ls_class()
            assert mlir_class.__name__ == "MLIRLanguageServer", "MLIR LS class import failed"
        if hasattr(Language, "LLVM_IR"):
            llvmir_class = Language.LLVM_IR.get_ls_class()
            assert llvmir_class.__name__ == "LLVMIRLanguageServer", "LLVM_IR LS class import failed"
        print("  ✓ Language server classes importable")

        print("\n✓ All verifications passed!")
        return True

    except Exception as e:
        print(f"  ERROR during verification: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Patch Serena's solidlsp to add MLIR, TableGen, PDLL, and LLVM IR language support"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--serena-path",
        type=Path,
        help="Path to solidlsp package (auto-detected if not specified)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify the installation, don't apply patches",
    )

    args = parser.parse_args()

    # Find solidlsp path
    if args.serena_path:
        solidlsp_path = args.serena_path
    else:
        solidlsp_path = find_serena_path()

    if not solidlsp_path or not solidlsp_path.exists():
        print("ERROR: Could not find solidlsp package.")
        print("Please specify the path with --serena-path")
        sys.exit(1)

    print(f"Found solidlsp at: {solidlsp_path}")

    if args.verify_only:
        success = verify_installation(solidlsp_path)
        sys.exit(0 if success else 1)

    if args.dry_run:
        print("\n=== DRY RUN MODE ===\n")

    # Step 1: Copy language server files
    print("\nStep 1: Copying language server files...")
    copied = copy_language_servers(solidlsp_path, dry_run=args.dry_run)

    # Step 2: Patch ls_config.py
    print("\nStep 2: Patching ls_config.py...")
    patched = patch_ls_config(solidlsp_path, dry_run=args.dry_run)

    if not args.dry_run:
        # Step 3: Verify
        verify_installation(solidlsp_path)

    print("\nDone!")
    if args.dry_run:
        print("\nTo apply changes, run without --dry-run")


if __name__ == "__main__":
    main()
