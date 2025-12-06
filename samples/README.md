# Firefly Samples

This directory contains sample projects demonstrating Firefly's capabilities for compiling F# to native code across different platforms.

## Directory Structure

```
samples/
├── console/                    # Desktop console applications
│   ├── HelloWorld/             # Simplest possible native F# app
│   ├── HelloWorldInteractive/  # Stack-based memory and user input
│   └── TimeLoop/               # Platform-specific time operations
├── embedded/                   # ARM microcontroller targets
│   ├── common/                 # Shared embedded infrastructure
│   │   ├── startup/            # Vector tables, reset handlers
│   │   └── linker/             # Linker scripts for each target
│   ├── stm32l5-blinky/         # LED blink on NUCLEO-L552ZE-Q
│   └── stm32l5-uart/           # Serial communication (planned)
├── sbc/                        # Single-board computer targets
│   ├── sweet-potato-blinky/    # LED blink on Libre Sweet Potato
│   └── sweet-potato-fb/        # Framebuffer display (planned)
└── templates/                  # Platform configuration templates
    ├── desktop-x64.toml        # x86-64 desktop platforms
    ├── stm32l5.toml            # STM32L5 series MCUs
    ├── allwinner-h6.toml       # Allwinner H6 SoC
    └── nucleo-l552ze-q.toml    # NUCLEO-L552ZE-Q board specifics
```

## Console Samples

These samples demonstrate the basic Firefly compilation pipeline for desktop targets.

### HelloWorld

The simplest possible Firefly application - prints "Hello, Firefly!" and exits.

```bash
cd console/HelloWorld
firefly compile HelloWorld.fidproj
./hello
```

### HelloWorldInteractive

Demonstrates stack-based memory allocation and Result-based error handling.

```bash
cd console/HelloWorldInteractive
firefly compile HelloWorldInteractive.fidproj
./hello-interactive
```

### TimeLoop

Shows platform-specific time operations using Alloy's Time module.

```bash
cd console/TimeLoop
firefly compile TimeLoop.fidproj
./timeloop 10  # Run 10 iterations
```

## Embedded Samples

These samples target ARM microcontrollers without an operating system.

### STM32L5 Blinky

LED blink demo for the NUCLEO-L552ZE-Q development board.

**Requirements:**
- NUCLEO-L552ZE-Q board
- OpenOCD or STM32CubeProgrammer
- ARM toolchain (for linking)

```bash
cd embedded/stm32l5-blinky
firefly compile Blinky.fidproj --target thumbv8m.main-none-eabihf

# Flash using OpenOCD
openocd -f interface/stlink.cfg -f target/stm32l5x.cfg \
  -c "program blinky.elf verify reset exit"
```

**Hardware Details:**
- LD1 (green): PC7
- LD2 (blue): PB7
- LD3 (red): PA9

## SBC Samples

These samples target ARM64 single-board computers running bare-metal.

### Sweet Potato Blinky

LED blink demo for the Libre Sweet Potato (Allwinner H6).

**Requirements:**
- Libre Sweet Potato or compatible H6 board
- U-Boot bootloader
- Serial console access

```bash
cd sbc/sweet-potato-blinky
firefly compile Blinky.fidproj --target aarch64-unknown-none

# Copy to SD card and boot via U-Boot
# U-Boot> fatload mmc 0:1 0x40000000 blinky.bin
# U-Boot> go 0x40000000
```

## Platform Templates

Template files define platform-specific configurations including:

- Memory regions and layouts
- Compilation targets and features
- Peripheral base addresses
- Clock configurations
- Debug/flash settings

Templates are referenced in `.fidproj` files:

```toml
[platform]
template = "stm32l5"
variant = "STM32L552ZET6Q"
```

## Building Samples

### Prerequisites

1. **Firefly CLI** installed and in PATH
2. **LLVM/Clang** for native code generation
3. **Platform-specific toolchains:**
   - Desktop: Standard C library and linker
   - ARM Cortex-M: `arm-none-eabi-gcc` toolchain
   - ARM64: `aarch64-none-elf-gcc` toolchain

### Build Commands

```bash
# Desktop (host platform)
firefly compile <project>.fidproj

# Cross-compile for ARM Cortex-M
firefly compile <project>.fidproj --target thumbv8m.main-none-eabihf

# Cross-compile for ARM64
firefly compile <project>.fidproj --target aarch64-unknown-none

# With verbose output
firefly compile <project>.fidproj -v

# Generate intermediate MLIR (stop before LLVM lowering)
firefly compile <project>.fidproj --emit-mlir

# Generate LLVM IR (stop before native compilation)
firefly compile <project>.fidproj --emit-llvm

# Keep all intermediate files for debugging
firefly compile <project>.fidproj -k
```

> **Note:** Full native code generation is in development. Currently, `--emit-mlir`
> and `--emit-llvm` generate stub output. Run `firefly doctor` to verify your
> toolchain is ready for when full compilation is enabled.

## Memory Models

Samples demonstrate different memory management strategies:

| Model | Description | Use Case |
|-------|-------------|----------|
| `stack_only` | All allocations on stack | Simple apps, tight loops |
| `static_pools` | Pre-allocated static buffers | Embedded systems |
| `arena` | Arena-based bulk allocation | Data processing |
| `mixed` | Combination of strategies | Complex applications |

## Next Steps

After getting familiar with these samples:

1. **QuantumCredential POC** - Build on `stm32l5-blinky` to add:
   - USB-CDC communication
   - Hardware crypto acceleration
   - Status LED patterns

2. **KeyStation POC** - Build on `sweet-potato-blinky` to add:
   - Framebuffer display initialization
   - Touch input handling
   - UI widget rendering

## Troubleshooting

### Common Issues

**"Target not supported"**
- Ensure LLVM is built with the required target (ARM, AArch64)
- Check `firefly doctor` for toolchain verification

**"Linker script not found"**
- Verify the `linker_script` path in `.fidproj` is correct
- Scripts are in `embedded/common/linker/`

**"Cannot find Alloy"**
- Ensure Alloy is built and the path in `[dependencies]` is correct
- Run `firefly doctor` to verify dependencies

**"Flash failed"**
- Check USB connection to development board
- Verify OpenOCD/STM32CubeProgrammer is installed
- Ensure correct debug interface is selected

## Contributing

When adding new samples:

1. Follow the existing directory structure
2. Include a `.fidproj` configuration file
3. Add appropriate documentation
4. Test on actual hardware when possible
5. Update this README with the new sample

## License

MIT License - See [LICENSE](../LICENSE) for details.
