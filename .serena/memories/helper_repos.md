# Helper Repos - Local Copies for Serena Indexing

The `helpers/` directory contains LOCAL COPIES of dependency repos.
These are synced from source repos and can be indexed by Serena's symbolic tools.

**Sync command:** `./helpers/sync.sh`

---

## Alloy (`helpers/Alloy/src/`)

Native F# standard library for Firefly - BCL-sympathetic API without .NET runtime.

### Key Files

| File | Purpose |
|------|---------|
| `Core.fs` | Core types and functions |
| `Primitives.fs` | `[<DllImport("__fidelity")>] extern` declarations |
| `Console.fs` | Console I/O using primitives |
| `Time.fs` | Time operations |
| `Memory.fs` | Memory operations |
| `Text.fs` | Text formatting |
| `Utf8.fs` | UTF-8 encoding |
| `Math.fs` | Math functions |

### Native Types (`NativeTypes/`)

| File | Purpose |
|------|---------|
| `NativeInt.fs` | Native integer types |
| `NativePtr.fs` | Native pointer operations |
| `NativeSpan.fs` | Stack-allocated spans |
| `NativeArray.fs` | Native arrays |
| `NativeString.fs` | Native string type (NativeStr) |

### Important Symbols

- `Alloy.Primitives.writeBytes` - extern syscall for writing bytes
- `Alloy.Primitives.readBytes` - extern syscall for reading bytes
- `Alloy.Console.Write`, `WriteLine` - Console output
- `Alloy.Console.ReadLine` - Console input
- `Alloy.Text.WritableString` - SRTP-dispatched string conversion

---

## XParsec (`helpers/XParsec/src/XParsec/`)

Parser combinator library - used by Firefly for pattern matching on PSG.

### Key Files

| File | Purpose |
|------|---------|
| `Types.fs` | Core types: Parser, Reader, ParseResult, ParseError |
| `Combinators.fs` | Parser combinators (map, bind, choice, many, etc.) |
| `Parsers.fs` | Basic parsers |
| `CharParsers.fs` | Character-based parsers |
| `ByteParsers.fs` | Byte-based parsers |
| `OperatorParsing.fs` | Operator precedence parsing |
| `ErrorFormatting.fs` | Error message formatting |

### Important Types

- `Parser<'T, 'Input, 'State>` - Core parser type
- `Reader<'T, 'State>` - Input reader with position tracking
- `ParseResult<'T>` - Success or failure result
- `ParseError` - Error with position and message

---

## Farscape (`helpers/Farscape/src/`)

Distributed compute library (FUTURE - not yet integrated with Firefly).

### Structure

- `Farscape.Core/` - Core library
  - `Types.fs` - Core types
  - `CodeGenerator.fs` - Code generation
  - `CppParser.fs` - C++ parsing
  - `TypeMapper.fs` - Type mapping
  - `BindingGenerator.fs` - Binding generation
  - `MemoryManager.fs` - Memory management

- `Farscape.Cli/` - Command-line interface
  - `Program.fs` - Entry point

---

## Usage with Serena

Since these are local copies, use Serena's symbolic tools directly:

```
# Get symbols in a file
get_symbols_overview("helpers/Alloy/src/Primitives.fs")

# Find a symbol
find_symbol("writeBytes", relative_path="helpers/Alloy")

# Search for patterns
search_for_pattern("DllImport", relative_path="helpers/Alloy")
```

**Note:** Alloy uses `.fidproj` format. A shadow `Alloy.fsproj` is created for F# LSP indexing.
The sync script regenerates this automatically.
