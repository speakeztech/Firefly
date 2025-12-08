# Suggested Commands

## Building the Compiler
```bash
# Build the compiler
cd /home/hhh/repos/Firefly/src
dotnet build

# Build release version
dotnet build -c Release
```

## Compiling F# Projects
```bash
# Compile a sample project
/home/hhh/repos/Firefly/src/bin/Debug/net9.0/Firefly compile HelloWorld.fidproj

# With verbose output
Firefly compile HelloWorld.fidproj --verbose

# Keep intermediate files for debugging
Firefly compile HelloWorld.fidproj -k
```

## Sample Projects (in /samples/console/)
```bash
# HelloWorld - minimal validation sample
cd /home/hhh/repos/Firefly/samples/console/HelloWorld
../../src/bin/Debug/net9.0/Firefly compile HelloWorld.fidproj

# TimeLoop - mutable state, while loops, DateTime, Sleep
cd /home/hhh/repos/Firefly/samples/console/TimeLoop
../../src/bin/Debug/net9.0/Firefly compile TimeLoop.fidproj
```

## Testing (No dedicated test suite currently)
The compiler is validated by compiling sample projects and running the resulting binaries.

## Standard Unix Commands
- `git` - version control
- `ls`, `cd`, `grep`, `find` - file navigation and search
- `dotnet` - .NET CLI for building F# projects

## Project Configuration
Projects use `.fidproj` files (TOML format):
```toml
[package]
name = "ProjectName"

[compilation]
memory_model = "stack_only"
target = "native"

[dependencies]
alloy = { path = "/home/hhh/repos/Alloy/src" }

[build]
sources = ["Main.fs"]
output = "binary_name"
output_kind = "freestanding"  # or "console"
```
