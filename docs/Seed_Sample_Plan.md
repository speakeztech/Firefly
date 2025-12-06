# Seed Sample Plan: From Hello World to Hardware POCs

This document is a tactical roadmap for achieving demonstrable Firefly compiler capabilities in service of a seed fundraising milestone. The goal is not academic completeness but **working samples that prove the vision**.

## The Fundraising Thesis

Firefly compiles F# to native code without garbage collection. To prove this matters, we need:

1. **Desktop console apps** - Prove the pipeline works end-to-end
2. **ARM microcontroller blinky** - Prove we can target bare metal
3. **ARM SBC blinky** - Prove we can target application processors
4. **QuantumCredential POC features** - Prove security hardware viability
5. **KeyStation POC features** - Prove air-gapped console viability

Everything else is a side quest until these work.

---

## What Exists Today

The PSG (Program Semantic Graph) foundation is largely complete:

- FSharpSymbol-based identity correlation works
- Reachability analysis with tombstone soft-delete works
- Basic MLIR generation exists

See [PSG_architecture.md](PSG_architecture.md) for the design decisions and [PSG_Initial_Lessons_Learned.md](PSG_Initial_Lessons_Learned.md) for implementation notes.

**What's missing**: The last mile to executable binaries for each target platform.

---

## Phase 1: Desktop Pipeline (DO THIS FIRST)

### Goal
Compile `samples/console/HelloWorld` to a native Linux x86-64 executable.

### The Pipeline

```
HelloWorld.fs
    ↓ FCS parses and type-checks
HelloWorld.typed.ast (in memory)
    ↓ PSG construction (existing code)
HelloWorld.psg (in memory)
    ↓ MLIR emission (existing code, needs completion)
HelloWorld.mlir
    ↓ mlir-opt (external tool)
HelloWorld.llvm.mlir
    ↓ mlir-translate (external tool)
HelloWorld.ll
    ↓ llc + clang (external tools)
hello (executable)
```

### Specific Tasks

1. **Complete MLIR emission for console output**
   - `Alloy.Console.WriteLine` must emit to MLIR's `llvm.call` to libc `puts`
   - This is string literal → syscall, nothing fancy

2. **Wire up the external tool pipeline**
   - Create a shell script or F# orchestrator that chains mlir-opt → mlir-translate → llc → clang
   - The `firefly compile` command should do this

3. **Validate with HelloWorld**
   - The program prints "Hello, Firefly!" and exits with code 0
   - That's the whole test

### What NOT To Do

- Don't implement full Alloy.Console yet - just `WriteLine` for string literals
- Don't worry about memory management - string literals are static
- Don't add JellyJar analytics yet (see [Jellyjar_Plan.md](Jellyjar_Plan.md) for future state)
- Don't implement the zipper yet (see [DCont_Pipeline_Roadmap.md](DCont_Pipeline_Roadmap.md) Phase 2)

### Success Criteria

```bash
./firefly compile samples/console/HelloWorld/HelloWorld.fidproj
./samples/console/HelloWorld/hello
# Output: Hello, Firefly!
```

---

## Phase 2: HelloWorldInteractive

### Goal
Compile `samples/console/HelloWorldInteractive` to demonstrate stack-based memory.

### What This Adds

- `stackBuffer<byte> 256` - Stack allocation in MLIR
- `readInto buffer` - Read from stdin into the buffer
- `Utf8.toString` - Stack-to-stack string conversion
- Pattern matching on `Result<int, Error>`

### The Key Insight

This sample proves that Firefly can do **zero-heap programs**. The entire execution uses only stack memory. This is the differentiator from standard F#.

### MLIR Requirements

```mlir
// Stack allocation
%buffer = memref.alloca() : memref<256xi8>

// Syscall for read
%n = llvm.call @read(%stdin, %buffer, %256) : ...

// Stack-based string building (no heap)
// This is where Alloy's zero-cost design matters
```

### What NOT To Do

- Don't implement arena memory yet - this is strictly stack
- Don't implement full error handling - happy path only

### Success Criteria

```bash
./firefly compile samples/console/HelloWorldInteractive/HelloWorldInteractive.fidproj
echo "World" | ./samples/console/HelloWorldInteractive/hello
# Output: Hello, World!
```

---

## Phase 3: ARM Cross-Compilation Infrastructure

### Goal
Configure the MLIR/LLVM pipeline to emit ARM Cortex-M33 and Cortex-A53 code.

### This Is Configuration, Not Code

The same PSG → MLIR pipeline works. What changes:

1. **Target triple** in LLVM: `thumbv8m.main-none-eabihf` (STM32L5) or `aarch64-unknown-none` (Sweet Potato)
2. **Linker script** for memory layout (already created in `samples/embedded/common/linker/`)
3. **Startup code** for vector table and initialization (already created in `samples/embedded/common/startup/`)

### Specific Tasks

1. **Add `--target` flag to firefly CLI**
   ```bash
   firefly compile --target thumbv8m.main-none-eabihf Blinky.fidproj
   ```

2. **Wire target-specific LLVM flags**
   - CPU features (e.g., `+fp-armv8`)
   - ABI (soft-float vs hard-float)
   - Code model (static for embedded)

3. **Test with assembly inspection**
   - Compile HelloWorld for ARM
   - Verify the output is valid ARM assembly (even if it won't run)

### What NOT To Do

- Don't implement GPIO yet - that's Phase 4
- Don't flash hardware yet - just verify cross-compilation works

### Success Criteria

```bash
./firefly compile --target thumbv8m.main-none-eabihf samples/console/HelloWorld/HelloWorld.fidproj
file samples/console/HelloWorld/hello
# Output: ELF 32-bit LSB executable, ARM, ...
```

---

## Phase 4: STM32L5 Blinky

### Goal
Blink the LD1 LED on NUCLEO-L552ZE-Q.

### Why This Matters

This proves Firefly can:
- Generate bare-metal code with no runtime
- Handle memory-mapped I/O (GPIO registers)
- Produce binaries that run on real hardware

### The Code Path

The sample code in `samples/embedded/stm32l5-blinky/` already defines:
- `STM32L5.fs` - GPIO driver with RCC clock enable, pin configuration
- `Main.fs` - Blink loop with delay

### MLIR Requirements

Memory-mapped I/O becomes `llvm.store` to absolute addresses:

```mlir
// GPIO BSRR register write (set pin)
%addr = llvm.mlir.constant(0x40020818 : i32) : i32
%ptr = llvm.inttoptr %addr : i32 to !llvm.ptr<i32>
%val = llvm.mlir.constant(0x80 : i32) : i32
llvm.store %val, %ptr : !llvm.ptr<i32>
```

### Specific Tasks

1. **Emit `Ptr.write` as absolute address store**
   - `Ptr.write<uint32> (nativeint 0x40020818u) value` → `llvm.store`
   - No memory management, just raw stores

2. **Emit `Ptr.read` as absolute address load**
   - Same pattern, `llvm.load`

3. **Handle inline functions**
   - `configureAsOutput`, `setPin`, `clearPin` should inline completely
   - No function call overhead

4. **Generate ELF with correct memory layout**
   - Use `stm32l552.ld` linker script
   - Vector table at 0x08000000 (Flash)
   - Stack at top of SRAM1

5. **Flash and test**
   - `openocd -f interface/stlink.cfg -f target/stm32l5x.cfg -c "program blinky.elf verify reset exit"`
   - LED blinks = success

### What NOT To Do

- Don't implement interrupts yet - polling delay loop is fine
- Don't implement UART yet - that's a separate sample
- Don't implement TrustZone yet - QuantumCredential needs it later, not now

### Success Criteria

LED LD1 (green, PC7) blinks on the NUCLEO board.

---

## Phase 5: Sweet Potato Blinky

### Goal
Blink an LED on the Libre Sweet Potato (Allwinner H6).

### Why This Matters

This proves Firefly can target application-class ARM64 processors in bare-metal mode. The Sweet Potato is the target for KeyStation.

### Key Differences from STM32L5

| Aspect | STM32L5 | Sweet Potato |
|--------|---------|--------------|
| Architecture | Cortex-M33 (ARMv8-M) | Cortex-A53 (ARMv8-A) |
| Boot | Vector table in Flash | U-Boot or direct boot |
| GPIO | AHB2 peripheral | Sunxi PIO controller |
| Complexity | Simple startup | MMU, caches to manage |

### Specific Tasks

1. **Get the board booting bare-metal code**
   - This may require U-Boot configuration
   - Or direct UART boot for development

2. **Implement basic PIO driver**
   - The code in `samples/sbc/sweet-potato-blinky/AllwinnerH6.fs` is a starting point
   - PIO base at 0x0300B000

3. **Disable MMU/caches for simplicity**
   - Early bare-metal, keep it simple
   - The startup code in `CortexA53.fs` handles this

4. **Flash/boot and test**
   - LED blinks = success

### What NOT To Do

- Don't implement framebuffer yet - that's KeyStation
- Don't enable MMU yet - bare minimum first

### Success Criteria

An LED connected to the Sweet Potato GPIO blinks.

---

## Phase 6: QuantumCredential POC Features

### Goal
Demonstrate the security-relevant capabilities for the USB hardware key.

### Features to Demonstrate

1. **USB CDC (serial over USB)**
   - Prove we can do USB stack on STM32L5
   - This is complex; consider using existing C USB stack via FFI

2. **Hardware RNG access**
   - STM32L5 has TRNG peripheral
   - Memory-mapped read, straightforward

3. **Basic crypto operation**
   - AES-256 using hardware acceleration
   - Or software implementation as fallback

### This Phase Is Exploratory

The goal is demonstrating *feasibility*, not a complete implementation. If USB proves too complex for the timeline, demonstrate:
- UART-based command interface
- Hardware RNG working
- One crypto operation working

### What NOT To Do

- Don't implement full FIDO2/WebAuthn - that's post-seed
- Don't implement TrustZone isolation yet - demonstrate first, secure later

---

## Phase 7: KeyStation POC Features

### Goal
Demonstrate the display and input capabilities for the air-gapped console.

### Features to Demonstrate

1. **Framebuffer output**
   - Write pixels to display
   - Prove we can do graphics

2. **Touch input**
   - Read touch events
   - Prove we can do input

3. **Basic UI widget**
   - Button or text field
   - Prove we can build UI

### This Phase Is Exploratory

Same as QuantumCredential - feasibility demonstration.

---

## Architecture Decisions: What NOT To Build Yet

### PHG (Program Hypergraph)

The PSG is sufficient for these samples. PHG adds:
- Multi-way relationships (hyperedges)
- Temporal projections
- Learning across compilations

None of that is needed to blink an LED. See [PSG_architecture.md](PSG_architecture.md) for why PSG is the right level for now.

### Full Coeffect System

The blog posts describe rich coeffect tracking. For seed samples:
- We need to know "this function does GPIO" (for memory-mapped I/O)
- We don't need formal coeffect algebra

Add lightweight annotations as you encounter patterns that need them. Don't build the framework first.

### Bidirectional Zipper

The [DCont_Pipeline_Roadmap.md](DCont_Pipeline_Roadmap.md) describes zippers for context-aware traversal. This is Phase 2 of that roadmap. The seed samples are "Phase 0" - they don't need sophisticated traversal.

### JellyJar Analytics

[Jellyjar_Plan.md](Jellyjar_Plan.md) describes DuckDB-based PSG analytics. This is valuable for debugging and development, but not for the fundraising demo. Build it when you need to understand why something isn't working.

### WAMI (WebAssembly)

The DCont roadmap targets WAMI. Desktop samples can use LLVM directly. WAMI is a future target, not a seed requirement.

---

## The Incremental Path That Preserves Future Work

Everything built for seed samples becomes foundation for production:

| Seed Work | Future Evolution |
|-----------|------------------|
| PSG with FSharpSymbol identity | Becomes hyperedge endpoints in PHG |
| Tombstone soft-delete | Enables temporal projections |
| Basic MLIR emission | Adds DCont dialect operations |
| `--target` flag | Expands to all LLVM targets |
| GPIO memory-mapped I/O | Becomes peripheral HAL pattern |
| Stack allocation | Adds arena allocation alongside |

**Nothing is throwaway.** The seed work is the minimal kernel that grows into the full system.

---

## Timeline Guidance (Not Estimates)

The phases are ordered by dependency and demonstration value:

1. **Desktop HelloWorld** - Proves the pipeline works at all
2. **HelloWorldInteractive** - Proves zero-allocation memory model
3. **ARM cross-compilation** - Proves we can target embedded
4. **STM32L5 Blinky** - Proves bare-metal works
5. **Sweet Potato Blinky** - Proves ARM64 bare-metal works
6. **QuantumCredential features** - Proves security hardware viability
7. **KeyStation features** - Proves air-gapped console viability

Each phase builds on the previous. Don't skip ahead.

---

## What Success Looks Like

For a seed fundraise demo:

**Must Have:**
- HelloWorld compiles and runs
- STM32L5 LED blinks
- One QuantumCredential feature works (RNG or crypto)

**Nice to Have:**
- HelloWorldInteractive works
- Sweet Potato LED blinks
- USB communication works

**Impressive:**
- Basic KeyStation UI renders
- Touch input works

The "must have" items prove the core thesis. The rest demonstrates breadth.

---

## References

- [PSG_architecture.md](PSG_architecture.md) - Why the PSG design is sound
- [DCont_Pipeline_Roadmap.md](DCont_Pipeline_Roadmap.md) - Future continuation work (Phase 2+)
- [Jellyjar_Plan.md](Jellyjar_Plan.md) - Future analytics work
- [FCS_Ingestion_Architecture.md](FCS_Ingestion_Architecture.md) - How F# sources become PSG
- [MLIR_TableGen_Guidance.md](MLIR_TableGen_Guidance.md) - MLIR dialect patterns
- `samples/README.md` - Sample project documentation
- `samples/samples.json` - Sample catalog with metadata
