# Firefly: Setup Guide

This document explains how to set up the Firefly compiler for local development and testing, assuming you already have local LLVM and MLIR dependencies installed.

## Prerequisites

- .NET SDK 8.0 or newer
- LLVM and MLIR (installed locally)
- F# development environment

## Local LLVM/MLIR Setup

The Firefly CLI tool expects LLVM and MLIR to be available in your system. The tool will use your local LLVM/MLIR installation to handle the low-level compilation steps.

### Verifying LLVM/MLIR Installation

Check that your LLVM installation is working:

```bash
llvm-config --version
```

Make sure the following commands are available in your PATH:
- `llvm-config`
- `mlir-opt`
- `mlir-translate`
- `llc`

## Building Firefly

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/firefly.git
   cd firefly
   ```

2. Build the project:
   ```bash
   dotnet build
   ```

3. Install as a local tool:
   ```bash
   dotnet pack src/Firefly.fsproj -c Release
   dotnet tool install --global --add-source src/nupkg Firefly
   ```

## Testing the Installation

Try compiling a simple F# program:

```bash
firefly compile examples/hello.fs --output hello
```

If successful, you should be able to run the resulting binary:

```bash
./hello
```

## Architecture

When you run Firefly, it performs the following steps:

1. Parses F# code using F# Compiler Services and Fantomas
2. Converts to Oak AST in the Dabbit namespace
3. Transforms the AST to eliminate closures and ensure static allocation
4. Uses XParsec to generate MLIR representations
5. Lowers through MLIR dialects to LLVM IR
6. Leverages your local LLVM installation to compile to native code

## Troubleshooting

If you encounter issues with LLVM/MLIR integration:

1. Check that all LLVM commands are in your PATH
2. Verify LLVM was built with MLIR support
3. Run with `--keep-intermediates` flag to inspect generated MLIR and LLVM IR
4. Check for missing dependencies in the build output

## Next Steps

Explore more complex examples and features:

```bash
# Verify zero-allocation guarantee
firefly verify hello --no-heap

# Apply aggressive optimizations
firefly compile examples/hello.fs --output hello --optimize aggressive
```
